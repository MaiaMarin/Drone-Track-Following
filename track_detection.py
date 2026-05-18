from pathlib import Path
import heapq
import json
import cv2
import numpy as np

BACKGROUND = 0
START_PAD = 1
LANDING_PAD = 2
TRACK = 3
DRONE = 4

MASK_COLORS_BGR = np.array([
    [0, 0, 0],
    [0, 255, 0],
    [255, 255, 0],
    [0, 180, 255],
    [255, 0, 255],
], dtype=np.uint8)

def resize_depth(depth, shape):
    if depth is None:
        return None

    h, w = shape

    if depth.shape[:2] == (h, w):
        return depth

    return cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

def clean_mask(mask, open_k=3, close_k=7):
    mask = mask.astype(np.uint8)

    if open_k and open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    if close_k and close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    return mask.astype(bool)

def connected_components(mask):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    comps = []

    for i in range(1, count):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx, cy = centroids[i]
        bbox_area = max(w * h, 1)

        comps.append({
            "id": int(i),
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "cx": float(cx),
            "cy": float(cy),
            "fill": float(area / bbox_area),
        })

    return labels, comps

def make_floor_roi(bgr):
    h, w = bgr.shape[:2]
    yy, xx = np.indices((h, w))

    roi = yy > int(0.16 * h)
    roi &= yy < int(0.97 * h)
    roi &= xx > int(0.01 * w)
    roi &= xx < int(0.99 * w)

    return roi

def detect_landing_pad(bgr, roi):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]

    hue, sat, val = cv2.split(hsv)
    blue, green, red = cv2.split(bgr)
    yy, xx = np.indices((h, w))

    mask = (
        roi &
        (xx < int(0.48 * w)) &
        (hue >= 85) & (hue <= 105) &
        (sat >= 55) &
        (val >= 70) &
        (blue.astype(np.int16) - red.astype(np.int16) > 35) &
        (green.astype(np.int16) - red.astype(np.int16) > 25)
    )

    mask = clean_mask(mask, open_k=5, close_k=13)

    labels, comps = connected_components(mask)
    candidates = []

    for comp in comps:
        if comp["area"] < 400:
            continue
        if comp["area"] > 15000:
            continue
        if comp["w"] < 25 or comp["h"] < 20:
            continue
        if comp["w"] > 180 or comp["h"] > 120:
            continue
        if comp["fill"] < 0.20:
            continue

        score = comp["area"] * comp["fill"]
        candidates.append((score, comp))

    if not candidates:
        empty = np.zeros((h, w), dtype=bool)
        return empty, None, {"landing_raw": mask, "landing_pad": empty}

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]

    pad = labels == best["id"]
    center = [best["cx"], best["cy"]]

    return pad, center, {"landing_raw": mask, "landing_pad": pad}

def detect_start_pad(bgr, roi):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]

    hue, sat, val = cv2.split(hsv)
    blue, green, red = cv2.split(bgr)
    yy, xx = np.indices((h, w))

    raw = (
        roi &
        (xx > int(0.70 * w)) &
        (xx < int(0.95 * w)) &
        (yy > int(0.15 * h)) &
        (yy < int(0.50 * h)) &
        (hue >= 65) & (hue <= 105) &
        (sat >= 12) & (sat <= 120) &
        (val >= 70) &
        (green.astype(np.int16) - red.astype(np.int16) > 2) &
        (blue.astype(np.int16) - red.astype(np.int16) > -10)
    )

    raw = cv2.morphologyEx(
        raw.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    )

    raw = cv2.morphologyEx(
        raw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ).astype(bool)

    labels, comps = connected_components(raw)
    candidates = []

    for comp in comps:
        if comp["area"] < 500:
            continue

        score = comp["area"] + comp["cx"] * 2.0
        candidates.append((score, comp))

    pad = np.zeros((h, w), dtype=np.uint8)
    center = None

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]
        center = [best["cx"], best["cy"]]
        cv2.circle(pad, (int(best["cx"]), int(best["cy"])), 45, 1, -1)

    pad = pad.astype(bool)

    return pad, center, {"start_raw": raw, "start_pad": pad}

def build_track_score(bgr, roi):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    hue, sat, val = cv2.split(hsv)
    blue, green, red = cv2.split(bgr)
    lightness = lab[:, :, 0]

    blur = cv2.GaussianBlur(lightness, (0, 0), 25)
    local_bright = lightness.astype(np.int16) - blur.astype(np.int16)

    beige = (
        roi &
        (val >= 80) &
        (sat <= 120) &
        (red >= 65) &
        (green >= 60) &
        (blue >= 50) &
        (np.abs(red.astype(np.int16) - green.astype(np.int16)) < 75) &
        (np.abs(green.astype(np.int16) - blue.astype(np.int16)) < 95)
    )

    edges = cv2.Canny(gray, 55, 135)
    edges = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    ).astype(bool)

    score = np.zeros(gray.shape, dtype=np.float32)
    score += np.clip((local_bright - 1) / 24, 0, 1) * 0.50
    score += beige.astype(np.float32) * 0.35
    score += edges.astype(np.float32) * 0.15
    score *= roi.astype(np.float32)

    debug = {
        "track_score": np.clip(score * 255, 0, 255).astype(np.uint8),
        "track_beige": beige,
        "track_edges": edges,
        "track_local_bright": np.clip(local_bright + 128, 0, 255).astype(np.uint8),
    }

    return score, debug

def dijkstra_path(score, start_point, end_point):
    h, w = score.shape[:2]

    if start_point is None or end_point is None:
        return []

    scale = max(1, int(round(max(h, w) / 360)))

    small_w = max(1, w // scale)
    small_h = max(1, h // scale)

    small_score = cv2.resize(score, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_score = cv2.GaussianBlur(small_score, (5, 5), 0)

    cost = 1.0 + 80.0 * (1.0 - np.clip(small_score, 0, 1))

    sx = int(np.clip(start_point[0] / scale, 0, small_w - 1))
    sy = int(np.clip(start_point[1] / scale, 0, small_h - 1))
    ex = int(np.clip(end_point[0] / scale, 0, small_w - 1))
    ey = int(np.clip(end_point[1] / scale, 0, small_h - 1))

    dist = np.full((small_h, small_w), np.inf, dtype=np.float32)
    prev = np.full((small_h, small_w, 2), -1, dtype=np.int16)

    dist[sy, sx] = 0
    queue = [(0.0, sy, sx)]

    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1.414),
        (-1, 1, 1.414),
        (1, -1, 1.414),
        (1, 1, 1.414),
    ]

    while queue:
        current_dist, y, x = heapq.heappop(queue)

        if current_dist != dist[y, x]:
            continue

        if x == ex and y == ey:
            break

        for dx, dy, step_cost in neighbors:
            nx = x + dx
            ny = y + dy

            if nx < 0 or nx >= small_w or ny < 0 or ny >= small_h:
                continue

            new_dist = current_dist + ((cost[y, x] + cost[ny, nx]) * 0.5 * step_cost)

            if new_dist < dist[ny, nx]:
                dist[ny, nx] = new_dist
                prev[ny, nx] = [x, y]
                heapq.heappush(queue, (float(new_dist), ny, nx))

    if not np.isfinite(dist[ey, ex]):
        return []

    path_small = []
    x, y = ex, ey

    for _ in range(small_h * small_w):
        path_small.append((x, y))

        if x == sx and y == sy:
            break

        px, py = prev[y, x]

        if px < 0 or py < 0:
            break

        x = int(px)
        y = int(py)

    path_small.reverse()

    path = []

    for x, y in path_small:
        path.append([
            int(np.clip(x * scale, 0, w - 1)),
            int(np.clip(y * scale, 0, h - 1)),
        ])

    return path

def path_to_mask(path, shape, thickness=40):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if len(path) < 2:
        return mask.astype(bool)

    points = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [points], False, 255, thickness=thickness, lineType=cv2.LINE_AA)

    return mask > 0

def detect_drone_led(bgr, roi, path_mask, landing_pad):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    h, w = bgr.shape[:2]
    hue, sat, val = cv2.split(hsv)
    blue, green, red = cv2.split(bgr)

    exclude = cv2.dilate(
        landing_pad.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (85, 85)),
        iterations=1
    ).astype(bool)

    bright = (
        roi &
        ~exclude &
        (
            ((val >= 160) & (sat <= 180)) |
            ((blue >= 140) & (green >= 120) & (red >= 80) & (val >= 135))
        )
    )

    labels, comps = connected_components(bright)

    distmap = None

    if path_mask is not None and np.count_nonzero(path_mask) > 0:
        inv = (~path_mask).astype(np.uint8) * 255
        distmap = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

    candidates = []

    for comp in comps:
        if comp["area"] < 4:
            continue
        if comp["area"] > 1500:
            continue
        if comp["w"] > 80 or comp["h"] > 80:
            continue

        cx = int(comp["cx"])
        cy = int(comp["cy"])

        r = 65
        x1 = max(0, cx - r)
        x2 = min(w, cx + r)
        y1 = max(0, cy - r)
        y2 = min(h, cy + r)

        dark_fraction = float(np.mean(gray[y1:y2, x1:x2] < 95))
        brightness = float(np.mean(val[labels == comp["id"]]))

        score = brightness * 1.5 + dark_fraction * 200 + comp["area"] * 0.03

        if distmap is not None:
            score -= float(distmap[cy, cx]) * 1.2

        comp["score"] = score
        comp["dark_fraction"] = dark_fraction
        comp["brightness"] = brightness

        candidates.append(comp)

    candidates.sort(key=lambda c: c["score"], reverse=True)

    drone_mask = np.zeros((h, w), dtype=np.uint8)
    drone_center = None

    if candidates and candidates[0]["score"] > 230:
        best = candidates[0]
        drone_center = [best["cx"], best["cy"]]
        cv2.circle(drone_mask, (int(best["cx"]), int(best["cy"])), 42, 1, -1)

    drone_mask = drone_mask.astype(bool)

    debug = {
        "drone_bright_candidates": bright,
        "drone": drone_mask,
    }

    info = {
        "drone_center": drone_center,
        "drone_candidates": candidates[:5],
    }

    return drone_mask, info, debug

def sample_depth(depth, point, radius=5):
    if depth is None or point is None:
        return None

    h, w = depth.shape[:2]
    x = int(round(point[0]))
    y = int(round(point[1]))

    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)

    values = depth[y1:y2, x1:x2]
    values = values[values > 0]

    if values.size == 0:
        return None

    return float(np.median(values))

def segment_scene(bgr, depth=None, return_debug=False):
    h, w = bgr.shape[:2]
    depth = resize_depth(depth, (h, w))

    roi = make_floor_roi(bgr)

    landing_pad, landing_center, landing_debug = detect_landing_pad(bgr, roi)
    start_pad, start_center, start_debug = detect_start_pad(bgr, roi)

    track_score, track_debug = build_track_score(bgr, roi)

    path = dijkstra_path(track_score, landing_center, start_center)
    track = path_to_mask(path, (h, w), thickness=max(24, int(round(min(h, w) * 0.055))))

    drone, drone_info, drone_debug = detect_drone_led(bgr, roi, track, landing_pad)

    seg = np.zeros((h, w), dtype=np.uint8)
    seg[track] = TRACK
    seg[start_pad] = START_PAD
    seg[landing_pad] = LANDING_PAD
    seg[drone] = DRONE

    info = {
        "start_pad_center": start_center,
        "landing_pad_center": landing_center,
        "drone_center": drone_info["drone_center"],
        "path": path,
        "start_depth": sample_depth(depth, start_center),
        "landing_depth": sample_depth(depth, landing_center),
        "drone_depth": sample_depth(depth, drone_info["drone_center"]),
    }

    if not return_debug:
        return seg, info

    debug = {
        "roi": roi,
        **landing_debug,
        **start_debug,
        **track_debug,
        "track": track,
        **drone_debug,
    }

    return seg, info, debug

def colorize_mask(seg):
    return MASK_COLORS_BGR[seg]

def overlay_mask(bgr, seg, alpha=0.45):
    color = colorize_mask(seg)
    visible = seg > 0
    overlay = bgr.copy()
    blended = cv2.addWeighted(bgr, 1 - alpha, color, alpha, 0)
    overlay[visible] = blended[visible]
    return overlay

def draw_points_and_path(bgr, info):
    out = bgr.copy()

    path = info.get("path") or []

    if len(path) >= 2:
        points = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [points], False, (0, 180, 255), 4, lineType=cv2.LINE_AA)

    start = info.get("start_pad_center")
    landing = info.get("landing_pad_center")
    drone = info.get("drone_center")

    if start is not None:
        cv2.circle(out, (int(start[0]), int(start[1])), 10, (0, 255, 0), -1)
        cv2.putText(out, "START", (int(start[0]) + 12, int(start[1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if landing is not None:
        cv2.circle(out, (int(landing[0]), int(landing[1])), 10, (255, 255, 0), -1)
        cv2.putText(out, "LAND", (int(landing[0]) + 12, int(landing[1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if drone is not None:
        cv2.circle(out, (int(drone[0]), int(drone[1])), 10, (255, 0, 255), -1)
        cv2.putText(out, "DRONE", (int(drone[0]) + 12, int(drone[1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    return out

def normalize_depth_for_view(depth):
    if depth is None:
        return None

    d = depth.astype(np.float32)
    valid = d > 0

    if np.count_nonzero(valid) == 0:
        return np.zeros(depth.shape, dtype=np.uint8)

    lo = np.percentile(d[valid], 2)
    hi = np.percentile(d[valid], 98)

    if hi <= lo:
        return np.zeros(depth.shape, dtype=np.uint8)

    out = np.clip((d - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)

def make_json_safe(value):
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [make_json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value

def save_results(rgb_path, depth_path=None, output_dir="track_detection_output"):
    rgb_path = Path(rgb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {rgb_path}")

    depth = None

    if depth_path is not None:
        depth_path = Path(depth_path)

        if depth_path.exists():
            depth = np.load(str(depth_path))

    seg, info, debug = segment_scene(bgr, depth, return_debug=True)

    stem = rgb_path.stem

    np.save(output_dir / f"{stem}_segmentation.npy", seg)
    cv2.imwrite(str(output_dir / f"{stem}_segmentation_raw.png"), seg)
    cv2.imwrite(str(output_dir / f"{stem}_segmentation_vis.png"), colorize_mask(seg))
    cv2.imwrite(str(output_dir / f"{stem}_overlay.png"), overlay_mask(bgr, seg))
    cv2.imwrite(str(output_dir / f"{stem}_path_overlay.png"), draw_points_and_path(bgr, info))

    if depth is not None:
        depth_view = normalize_depth_for_view(resize_depth(depth, bgr.shape[:2]))
        cv2.imwrite(str(output_dir / f"{stem}_depth_view.png"), depth_view)

    for name, mask in debug.items():
        if mask.dtype == bool:
            cv2.imwrite(str(output_dir / f"{stem}_{name}.png"), mask.astype(np.uint8) * 255)
        else:
            cv2.imwrite(str(output_dir / f"{stem}_{name}.png"), mask)

    with open(output_dir / f"{stem}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe(info), f, indent=2)

    print(f"Saved results to: {output_dir}")
    print(json.dumps(make_json_safe(info), indent=2))

if __name__ == "__main__":
    test_pairs = [
    ("frame_1779104513_rgb.png", "frame_1779104513_depth.npy"),
    ]

    for rgb_path, depth_path in test_pairs:
        if Path(rgb_path).exists():
            save_results(rgb_path, depth_path)