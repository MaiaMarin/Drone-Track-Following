from pathlib import Path
import heapq
import json
import cv2
import numpy as np

BACKGROUND = 0
START_PAD = 1
LANDING_PAD = 2
TRACK = 3

MASK_COLORS_BGR = np.array([
    [0, 0, 0],
    [0, 255, 0],
    [255, 255, 0],
    [0, 180, 255],
], dtype=np.uint8)

CENTERLINE_RESAMPLE_STEP = 6.0
CENTERLINE_SMOOTH_WINDOW = 7
CENTERLINE_SIMPLIFY_EPSILON = 10.0


def resize_depth(depth, shape):
    if depth is None:
        return None

    h, w = shape

    if depth.shape[:2] == (h, w):
        return depth

    return cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)


def resize_mask(mask, shape):
    h, w = shape

    if mask is None:
        return np.zeros((h, w), dtype=bool)

    if mask.shape[:2] == (h, w):
        return mask.astype(bool)

    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


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
    roi &= yy < int(0.98 * h)
    roi &= xx > int(0.01 * w)
    roi &= xx < int(0.99 * w)

    return roi


def point_mask(shape, center, radius=55):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if center is None:
        return mask.astype(bool)

    x = int(round(center[0]))
    y = int(round(center[1]))

    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    cv2.circle(mask, (x, y), radius, 1, -1)

    return mask.astype(bool)


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


def normalize_point(point, shape):
    if point is None:
        return None

    h, w = shape
    x = float(point[0])
    y = float(point[1])

    x = float(np.clip(x, 0, w - 1))
    y = float(np.clip(y, 0, h - 1))

    return [x, y]


def detect_colored_pad_components(bgr, roi):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]

    hue, sat, val = cv2.split(hsv)
    blue, green, red = cv2.split(bgr)
    yy, xx = np.indices((h, w))

    raw = (
        roi &
        (yy > int(0.20 * h)) &
        (hue >= 55) & (hue <= 112) &
        (sat >= 45) &
        (val >= 55) &
        (np.maximum(blue, green).astype(np.int16) - red.astype(np.int16) > 25)
    )

    raw = cv2.morphologyEx(
        raw.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )

    raw = cv2.morphologyEx(
        raw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ).astype(bool)

    labels, comps = connected_components(raw)

    filtered = []

    for comp in comps:
        if comp["area"] < 150:
            continue
        if comp["area"] > 22000:
            continue
        if comp["w"] < 15 or comp["h"] < 10:
            continue
        if comp["w"] > 260 or comp["h"] > 180:
            continue
        if comp["fill"] < 0.10:
            continue

        filtered.append(comp)

    return raw, labels, filtered


def detect_landing_pad(bgr, roi):
    h, w = bgr.shape[:2]
    raw, labels, comps = detect_colored_pad_components(bgr, roi)

    candidates = []

    for comp in comps:
        if comp["cx"] > int(0.58 * w):
            continue
        if comp["cy"] < int(0.24 * h):
            continue

        left_bonus = (1.0 - comp["cx"] / max(w, 1)) * 500
        floor_bonus = (comp["cy"] / max(h, 1)) * 200
        area_score = comp["area"] * comp["fill"]
        score = area_score + left_bonus + floor_bonus

        candidates.append((score, comp))

    if not candidates:
        empty = np.zeros((h, w), dtype=bool)
        return empty, None, {"landing_raw": raw, "landing_pad": empty}

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]

    pad = labels == best["id"]
    center = [best["cx"], best["cy"]]

    return pad, center, {"landing_raw": raw, "landing_pad": pad}


def detect_start_pad(bgr, roi):
    h, w = bgr.shape[:2]
    raw, labels, comps = detect_colored_pad_components(bgr, roi)

    candidates = []

    for comp in comps:
        if comp["cx"] < int(0.48 * w):
            continue
        if comp["cy"] < int(0.22 * h):
            continue

        right_bonus = (comp["cx"] / max(w, 1)) * 750
        floor_bonus = (comp["cy"] / max(h, 1)) * 250
        area_score = comp["area"] * comp["fill"]
        score = area_score + right_bonus + floor_bonus

        candidates.append((score, comp))

    pad = np.zeros((h, w), dtype=np.uint8)
    center = None

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]
        center = [best["cx"], best["cy"]]
        cv2.circle(pad, (int(best["cx"]), int(best["cy"])), 55, 1, -1)

    pad = pad.astype(bool)

    return pad, center, {"start_raw": raw, "start_pad": pad}


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


def build_track_score(bgr, roi, depth=None):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    blue, green, red = cv2.split(bgr)
    lightness = lab[:, :, 0]
    _, sat, val = cv2.split(hsv)

    blur = cv2.GaussianBlur(lightness, (0, 0), 25)
    local_bright = lightness.astype(np.int16) - blur.astype(np.int16)

    track_color = (
        roi &
        (val >= 70) &
        (sat <= 130) &
        (red >= 55) &
        (green >= 50) &
        (blue >= 40) &
        (np.abs(red.astype(np.int16) - green.astype(np.int16)) < 85) &
        (np.abs(green.astype(np.int16) - blue.astype(np.int16)) < 105)
    )

    edges = cv2.Canny(gray, 55, 135)
    edges = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    ).astype(bool)

    depth_edges = np.zeros(gray.shape, dtype=bool)

    if depth is not None:
        depth_view = normalize_depth_for_view(depth)

        if depth_view is not None:
            depth_edges = cv2.Canny(depth_view, 35, 110)
            depth_edges = cv2.dilate(
                depth_edges,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1
            ).astype(bool)

    score = np.zeros(gray.shape, dtype=np.float32)
    score += np.clip((local_bright - 1) / 24, 0, 1) * 0.45
    score += track_color.astype(np.float32) * 0.35
    score += edges.astype(np.float32) * 0.15
    score += depth_edges.astype(np.float32) * 0.05
    score *= roi.astype(np.float32)
    score = np.clip(score, 0, 1)

    candidate = (score > 0.28) & track_color

    candidate = cv2.morphologyEx(
        candidate.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    ).astype(bool)

    candidate = cv2.dilate(
        candidate.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        iterations=1
    ).astype(bool)

    debug = {
        "track_score": np.clip(score * 255, 0, 255).astype(np.uint8),
        "track_color": track_color,
        "track_edges": edges,
        "track_depth_edges": depth_edges,
        "track_candidate": candidate,
        "track_local_bright": np.clip(local_bright + 128, 0, 255).astype(np.uint8),
    }

    return score, candidate, debug


def snap_endpoint_to_track(score, candidate, point, other_point, radius=155):
    if point is None:
        return None

    h, w = score.shape[:2]
    yy, xx = np.indices((h, w))

    point = np.array(point, dtype=np.float32)

    if other_point is None:
        other_point = point

    other_point = np.array(other_point, dtype=np.float32)

    d_point = np.hypot(xx - point[0], yy - point[1])
    d_other = np.hypot(xx - other_point[0], yy - other_point[1])

    mask = d_point <= radius
    mask &= candidate | (score > 0.35)

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return [float(point[0]), float(point[1])]

    values = (
        score[ys, xs] * 1000.0 +
        candidate[ys, xs].astype(np.float32) * 150.0 -
        d_point[ys, xs] * 0.8 -
        d_other[ys, xs] * 0.03
    )

    best = int(np.argmax(values))

    return [float(xs[best]), float(ys[best])]


def dijkstra_path(score, start_point, end_point, candidate_mask=None):
    h, w = score.shape[:2]

    if start_point is None or end_point is None:
        return []

    scale = max(1, int(round(max(h, w) / 360)))

    small_w = max(1, w // scale)
    small_h = max(1, h // scale)

    small_score = cv2.resize(score, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_score = cv2.GaussianBlur(small_score, (5, 5), 0)
    small_score = np.clip(small_score, 0, 1)

    if candidate_mask is None:
        small_candidate = np.ones((small_h, small_w), dtype=bool)
    else:
        small_candidate = cv2.resize(
            candidate_mask.astype(np.uint8),
            (small_w, small_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    cost = 1.0 + 100.0 * (1.0 - small_score)
    cost += (~small_candidate).astype(np.float32) * 350.0

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


def path_as_array(path):
    if path is None:
        return np.empty((0, 2), dtype=np.float32)

    arr = np.array(path, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float32)

    cleaned = []

    for p in arr:
        if not cleaned:
            cleaned.append(p)
        elif np.linalg.norm(p - cleaned[-1]) >= 1.0:
            cleaned.append(p)

    return np.array(cleaned, dtype=np.float32)


def point_distance(a, b):
    if a is None or b is None:
        return float("inf")

    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


def orient_path_start_to_landing(path, start_center, landing_center):
    arr = path_as_array(path)

    if len(arr) < 2:
        return arr

    if start_center is None or landing_center is None:
        return arr

    normal_score = point_distance(arr[0], start_center) + point_distance(arr[-1], landing_center)
    reversed_score = point_distance(arr[-1], start_center) + point_distance(arr[0], landing_center)

    if reversed_score < normal_score:
        arr = arr[::-1].copy()

    return arr


def smooth_centerline(points, window=CENTERLINE_SMOOTH_WINDOW):
    arr = path_as_array(points)

    if len(arr) < 3 or window <= 1:
        return arr

    if window % 2 == 0:
        window += 1

    if len(arr) < window:
        return arr

    half = window // 2
    out = []

    for i in range(len(arr)):
        if i == 0 or i == len(arr) - 1:
            out.append(arr[i])
            continue

        a = max(0, i - half)
        b = min(len(arr), i + half + 1)
        out.append(np.mean(arr[a:b], axis=0))

    return np.array(out, dtype=np.float32)


def resample_centerline(points, step=CENTERLINE_RESAMPLE_STEP):
    arr = path_as_array(points)

    if len(arr) < 2:
        return arr

    result = [arr[0].copy()]
    carry = 0.0
    previous = arr[0].copy()

    for i in range(1, len(arr)):
        current = arr[i].copy()
        segment = current - previous
        length = float(np.linalg.norm(segment))

        if length < 1e-6:
            continue

        direction = segment / length
        remaining = length

        while carry + remaining >= step:
            need = step - carry
            new_point = previous + direction * need
            result.append(new_point.copy())
            previous = new_point
            remaining = float(np.linalg.norm(current - previous))
            carry = 0.0

        carry += remaining
        previous = current

    if len(result) == 0:
        return arr

    if np.linalg.norm(result[-1] - arr[-1]) > 1.0:
        result.append(arr[-1].copy())

    return np.array(result, dtype=np.float32)


def simplify_centerline(points, epsilon=CENTERLINE_SIMPLIFY_EPSILON):
    arr = path_as_array(points)

    if len(arr) < 3:
        return arr

    contour = arr.reshape(-1, 1, 2).astype(np.float32)
    approx = cv2.approxPolyDP(contour, epsilon, False).reshape(-1, 2).astype(np.float32)

    if len(approx) < 2:
        return arr[[0, -1]]

    if np.linalg.norm(approx[0] - arr[0]) > 1.0:
        approx = np.vstack([arr[0], approx])

    if np.linalg.norm(approx[-1] - arr[-1]) > 1.0:
        approx = np.vstack([approx, arr[-1]])

    cleaned = []

    for p in approx:
        if not cleaned:
            cleaned.append(p)
        elif np.linalg.norm(p - cleaned[-1]) >= 8.0:
            cleaned.append(p)

    if len(cleaned) < 2:
        cleaned = [arr[0], arr[-1]]

    return np.array(cleaned, dtype=np.float32)


def sample_depths_along_centerline(depth, centerline, radius=5):
    arr = path_as_array(centerline)

    if depth is None or len(arr) == 0:
        return [], 0.0

    values = []

    for p in arr:
        values.append(sample_depth(depth, p, radius=radius))

    valid = sum(v is not None for v in values)
    ratio = float(valid / max(len(values), 1))

    return values, ratio


def centerline_lengths(centerline):
    arr = path_as_array(centerline)

    if len(arr) < 2:
        return {
            "centerline_length_px": 0.0,
            "centerline_segment_lengths_px": [],
        }

    diffs = np.diff(arr, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)

    return {
        "centerline_length_px": float(np.sum(lengths)),
        "centerline_segment_lengths_px": [float(x) for x in lengths],
    }


def centerline_turn_angles(centerline):
    arr = path_as_array(centerline)

    if len(arr) < 3:
        return []

    angles = []

    for i in range(1, len(arr) - 1):
        p0 = np.array([arr[i - 1][0], -arr[i - 1][1]], dtype=np.float32)
        p1 = np.array([arr[i][0], -arr[i][1]], dtype=np.float32)
        p2 = np.array([arr[i + 1][0], -arr[i + 1][1]], dtype=np.float32)

        v1 = p1 - p0
        v2 = p2 - p1

        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))

        if n1 < 1e-6 or n2 < 1e-6:
            angles.append(0.0)
            continue

        v1 = v1 / n1
        v2 = v2 / n2

        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
        angles.append(float(np.degrees(np.arctan2(cross, dot))))

    return angles


def build_centerline_info(path, depth=None, start_center=None, landing_center=None):
    raw_oriented = orient_path_start_to_landing(path, start_center, landing_center)
    smoothed = smooth_centerline(raw_oriented)
    resampled = resample_centerline(smoothed)
    simplified = simplify_centerline(resampled)

    depth_values, depth_valid_ratio = sample_depths_along_centerline(depth, resampled)

    info = {
        "centerline_raw": raw_oriented.tolist(),
        "centerline_smoothed": smoothed.tolist(),
        "centerline": resampled.tolist(),
        "centerline_simplified": simplified.tolist(),
        "centerline_depths": depth_values,
        "centerline_depth_valid_ratio": float(depth_valid_ratio),
        "centerline_turn_angles": centerline_turn_angles(resampled),
        "centerline_simplified_turn_angles": centerline_turn_angles(simplified),
        "centerline_point_count": int(len(resampled)),
        "centerline_simplified_point_count": int(len(simplified)),
    }

    info.update(centerline_lengths(resampled))

    return info


def _segment_with_centers(
    bgr,
    depth,
    start_center,
    landing_center,
    start_pad,
    landing_pad,
    return_debug=False,
    manual_points=False,
):
    h, w = bgr.shape[:2]
    depth = resize_depth(depth, (h, w))

    start_center = normalize_point(start_center, (h, w))
    landing_center = normalize_point(landing_center, (h, w))

    roi = make_floor_roi(bgr)
    track_score, track_candidate, track_debug = build_track_score(bgr, roi, depth=depth)

    start_track_point = snap_endpoint_to_track(
        track_score,
        track_candidate,
        start_center,
        landing_center,
        radius=155
    )

    landing_track_point = snap_endpoint_to_track(
        track_score,
        track_candidate,
        landing_center,
        start_center,
        radius=155
    )

    raw_path = dijkstra_path(
        track_score,
        start_track_point,
        landing_track_point,
        candidate_mask=track_candidate
    )

    track = path_to_mask(
        raw_path,
        (h, w),
        thickness=max(24, int(round(min(h, w) * 0.055)))
    )

    centerline_info = build_centerline_info(
        raw_path,
        depth=depth,
        start_center=start_center,
        landing_center=landing_center
    )

    seg = np.zeros((h, w), dtype=np.uint8)
    seg[track] = TRACK
    seg[start_pad.astype(bool)] = START_PAD
    seg[landing_pad.astype(bool)] = LANDING_PAD

    info = {
        "manual_points": bool(manual_points),
        "start_pad_center": start_center,
        "landing_pad_center": landing_center,
        "end_pad_center": landing_center,
        "path": raw_path,
        "track_mask": track,
        "track_candidate_mask": track_candidate,
        "start_pad_mask": start_pad.astype(bool),
        "landing_pad_mask": landing_pad.astype(bool),
        "end_pad_mask": landing_pad.astype(bool),
        "start_track_point": start_track_point,
        "landing_track_point": landing_track_point,
        "end_track_point": landing_track_point,
        "start_depth": sample_depth(depth, start_center),
        "landing_depth": sample_depth(depth, landing_center),
        "end_depth": sample_depth(depth, landing_center),
        **centerline_info,
    }

    if not return_debug:
        return seg, info

    debug = {
        "roi": roi,
        **track_debug,
        "track": track,
    }

    return seg, info, debug


def segment_scene(
    bgr,
    depth=None,
    return_debug=False,
    start_center=None,
    landing_center=None,
    start_radius=55,
    landing_radius=55,
):
    h, w = bgr.shape[:2]
    depth = resize_depth(depth, (h, w))

    if start_center is not None and landing_center is not None:
        start_center = normalize_point(start_center, (h, w))
        landing_center = normalize_point(landing_center, (h, w))

        start_pad = point_mask((h, w), start_center, radius=start_radius)
        landing_pad = point_mask((h, w), landing_center, radius=landing_radius)

        return _segment_with_centers(
            bgr,
            depth,
            start_center,
            landing_center,
            start_pad,
            landing_pad,
            return_debug=return_debug,
            manual_points=True,
        )

    roi = make_floor_roi(bgr)

    landing_pad, landing_center, landing_debug = detect_landing_pad(bgr, roi)
    start_pad, start_center, start_debug = detect_start_pad(bgr, roi)

    result = _segment_with_centers(
        bgr,
        depth,
        start_center,
        landing_center,
        start_pad,
        landing_pad,
        return_debug=return_debug,
        manual_points=False,
    )

    if not return_debug:
        return result

    seg, info, debug = result
    debug.update(landing_debug)
    debug.update(start_debug)

    return seg, info, debug


def segment_scene_from_points(
    bgr,
    depth=None,
    start_center=None,
    landing_center=None,
    return_debug=False,
    start_radius=55,
    landing_radius=55,
):
    return segment_scene(
        bgr,
        depth=depth,
        return_debug=return_debug,
        start_center=start_center,
        landing_center=landing_center,
        start_radius=start_radius,
        landing_radius=landing_radius,
    )


def build_fixed_scene(bgr, depth=None):
    seg, info = segment_scene(bgr, depth, return_debug=False)

    fixed_scene = {
        "manual_points": info.get("manual_points", False),
        "path": info.get("path") or [],
        "centerline": info.get("centerline") or [],
        "centerline_raw": info.get("centerline_raw") or [],
        "centerline_smoothed": info.get("centerline_smoothed") or [],
        "centerline_simplified": info.get("centerline_simplified") or [],
        "centerline_depths": info.get("centerline_depths") or [],
        "centerline_depth_valid_ratio": info.get("centerline_depth_valid_ratio"),
        "track_mask": info.get("track_mask"),
        "track_candidate_mask": info.get("track_candidate_mask"),
        "start_pad_center": info.get("start_pad_center"),
        "landing_pad_center": info.get("landing_pad_center"),
        "end_pad_center": info.get("end_pad_center"),
        "start_pad_mask": info.get("start_pad_mask"),
        "landing_pad_mask": info.get("landing_pad_mask"),
        "end_pad_mask": info.get("end_pad_mask"),
        "start_track_point": info.get("start_track_point"),
        "landing_track_point": info.get("landing_track_point"),
        "end_track_point": info.get("end_track_point"),
        "shape": bgr.shape[:2],
    }

    return fixed_scene


def build_fixed_scene_from_points(
    bgr,
    depth=None,
    start_center=None,
    landing_center=None,
    start_radius=55,
    landing_radius=55,
):
    seg, info = segment_scene_from_points(
        bgr,
        depth=depth,
        start_center=start_center,
        landing_center=landing_center,
        return_debug=False,
        start_radius=start_radius,
        landing_radius=landing_radius,
    )

    fixed_scene = {
        "manual_points": True,
        "path": info.get("path") or [],
        "centerline": info.get("centerline") or [],
        "centerline_raw": info.get("centerline_raw") or [],
        "centerline_smoothed": info.get("centerline_smoothed") or [],
        "centerline_simplified": info.get("centerline_simplified") or [],
        "centerline_depths": info.get("centerline_depths") or [],
        "centerline_depth_valid_ratio": info.get("centerline_depth_valid_ratio"),
        "track_mask": info.get("track_mask"),
        "track_candidate_mask": info.get("track_candidate_mask"),
        "start_pad_center": info.get("start_pad_center"),
        "landing_pad_center": info.get("landing_pad_center"),
        "end_pad_center": info.get("end_pad_center"),
        "start_pad_mask": info.get("start_pad_mask"),
        "landing_pad_mask": info.get("landing_pad_mask"),
        "end_pad_mask": info.get("end_pad_mask"),
        "start_track_point": info.get("start_track_point"),
        "landing_track_point": info.get("landing_track_point"),
        "end_track_point": info.get("end_track_point"),
        "shape": bgr.shape[:2],
    }

    return fixed_scene


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

    path = info.get("centerline") or info.get("path") or []

    if len(path) >= 2:
        points = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [points], False, (0, 0, 255), 3, lineType=cv2.LINE_AA)

    start = info.get("start_pad_center")
    landing = info.get("landing_pad_center")

    if start is not None:
        cv2.circle(out, (int(start[0]), int(start[1])), 10, (0, 255, 0), -1)
        cv2.putText(
            out,
            "START",
            (int(start[0]) + 12, int(start[1]) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    if landing is not None:
        cv2.circle(out, (int(landing[0]), int(landing[1])), 10, (255, 255, 0), -1)
        cv2.putText(
            out,
            "END",
            (int(landing[0]) + 12, int(landing[1]) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    return out


def draw_final_detection(bgr, info):
    out = bgr.copy()

    seg_overlay = np.zeros_like(out)

    track_mask = info.get("track_mask")
    start_pad = info.get("start_pad_mask")
    landing_pad = info.get("landing_pad_mask")

    if track_mask is not None:
        seg_overlay[track_mask.astype(bool)] = (0, 180, 255)

    if start_pad is not None:
        seg_overlay[start_pad.astype(bool)] = (0, 255, 0)

    if landing_pad is not None:
        seg_overlay[landing_pad.astype(bool)] = (255, 255, 0)

    visible = np.any(seg_overlay > 0, axis=2)
    blended = cv2.addWeighted(out, 0.70, seg_overlay, 0.30, 0)
    out[visible] = blended[visible]

    centerline = info.get("centerline") or info.get("path") or []

    if len(centerline) >= 2:
        pts = np.array(centerline, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        for p in pts[::18]:
            x, y = p[0]
            cv2.circle(out, (int(x), int(y)), 1, (0, 0, 255), -1)

    start = info.get("start_pad_center")
    landing = info.get("landing_pad_center")

    if start is not None:
        cv2.circle(out, (int(start[0]), int(start[1])), 11, (0, 255, 0), -1)
        cv2.putText(
            out,
            "START",
            (int(start[0]) + 12, int(start[1]) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    if landing is not None:
        cv2.circle(out, (int(landing[0]), int(landing[1])), 11, (255, 255, 0), -1)
        cv2.putText(
            out,
            "END",
            (int(landing[0]) + 12, int(landing[1]) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    length_px = info.get("centerline_length_px")
    depth_ratio = info.get("centerline_depth_valid_ratio")

    if length_px is not None:
        cv2.putText(
            out,
            f"centerline px: {length_px:.1f}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    if depth_ratio is not None:
        cv2.putText(
            out,
            f"depth valid: {depth_ratio:.2f}",
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    if info.get("manual_points"):
        cv2.putText(
            out,
            "manual START/END",
            (12, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    return out


def make_json_safe(value):
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items() if "mask" not in k}

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


def select_start_end_points(bgr, window_name="Select START then END"):
    points = []

    def render():
        display = bgr.copy()

        cv2.putText(
            display,
            "Click START, then END. r = reset, q/Esc = cancel",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA
        )

        for i, point in enumerate(points):
            x = int(point[0])
            y = int(point[1])

            if i == 0:
                color = (0, 255, 0)
                label = "START"
            else:
                color = (255, 255, 0)
                label = "END"

            cv2.circle(display, (x, y), 8, color, -1)
            cv2.putText(
                display,
                label,
                (x + 12, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                lineType=cv2.LINE_AA
            )

        cv2.imshow(window_name, display)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append([float(x), float(y)])
            render()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    render()

    while len(points) < 2:
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            cv2.destroyWindow(window_name)
            raise RuntimeError("manual point selection cancelled.")

        if key == ord("r"):
            points.clear()
            render()

    cv2.waitKey(200)
    cv2.destroyWindow(window_name)

    return points[0], points[1]


def save_results(rgb_path, depth_path=None, output_dir="track_detection_output", show=False):
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
    final_overlay = draw_final_detection(bgr, info)

    final_path = output_dir / f"{stem}_final_detection.png"
    metadata_path = output_dir / f"{stem}_metadata.json"

    cv2.imwrite(str(final_path), final_overlay)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(info), f, indent=2)

    print("Saved final detection:", final_path)
    print("Saved metadata:", metadata_path)
    print("start:", info.get("start_pad_center"))
    print("end:", info.get("landing_pad_center"))
    print("start track point:", info.get("start_track_point"))
    print("end track point:", info.get("landing_track_point"))
    print("centerline points:", info.get("centerline_point_count"))
    print("centerline length px:", round(info.get("centerline_length_px", 0), 2))
    print("depth valid ratio:", round(info.get("centerline_depth_valid_ratio", 0), 2))

    if show:
        cv2.namedWindow("Track detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Track detection", final_overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return seg, info


def save_results_from_points(
    rgb_path,
    depth_path=None,
    output_dir="track_detection_output",
    show=False,
    start_center=None,
    landing_center=None,
):
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

    if start_center is None or landing_center is None:
        start_center, landing_center = select_start_end_points(bgr)

    seg, info, debug = segment_scene_from_points(
        bgr,
        depth=depth,
        start_center=start_center,
        landing_center=landing_center,
        return_debug=True
    )

    stem = rgb_path.stem
    final_overlay = draw_final_detection(bgr, info)

    final_path = output_dir / f"{stem}_manual_final_detection.png"
    metadata_path = output_dir / f"{stem}_manual_metadata.json"

    cv2.imwrite(str(final_path), final_overlay)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(info), f, indent=2)

    print("Saved manual final detection:", final_path)
    print("Saved manual metadata:", metadata_path)
    print("manual start:", info.get("start_pad_center"))
    print("manual end:", info.get("landing_pad_center"))
    print("start track point:", info.get("start_track_point"))
    print("end track point:", info.get("landing_track_point"))
    print("centerline points:", info.get("centerline_point_count"))
    print("centerline length px:", round(info.get("centerline_length_px", 0), 2))
    print("depth valid ratio:", round(info.get("centerline_depth_valid_ratio", 0), 2))

    if show:
        cv2.namedWindow("Manual track detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Manual track detection", final_overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return seg, info


# if __name__ == "__main__":
#     test_rgb = Path("frame_1779104513_rgb.png")
#     test_depth = Path("frame_1779104513_depth.npy")

#     if test_rgb.exists():
#         save_results_from_points(
#             test_rgb,
#             test_depth if test_depth.exists() else None,
#             show=True
#         )
#     else:
#         print(f"Test frame not found: {test_rgb}")