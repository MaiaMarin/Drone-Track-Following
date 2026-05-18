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

def detect_landing_pad(bgr, roi):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]

    hue, sat, val = cv2.split(hsv)
    blue, green, red = cv2.split(bgr)
    yy, xx = np.indices((h, w))

    mask = (
        roi &
        (xx < int(0.50 * w)) &
        (hue >= 82) & (hue <= 110) &
        (sat >= 45) &
        (val >= 60) &
        (blue.astype(np.int16) - red.astype(np.int16) > 25) &
        (green.astype(np.int16) - red.astype(np.int16) > 15)
    )

    mask = clean_mask(mask, open_k=5, close_k=13)

    labels, comps = connected_components(mask)
    candidates = []

    for comp in comps:
        if comp["area"] < 250:
            continue
        if comp["area"] > 18000:
            continue
        if comp["w"] < 20 or comp["h"] < 15:
            continue
        if comp["w"] > 220 or comp["h"] > 160:
            continue
        if comp["fill"] < 0.15:
            continue

        score = comp["area"] * comp["fill"] + (1.0 - comp["cx"] / max(w, 1)) * 400
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
        (xx > int(0.48 * w)) &
        (yy > int(0.25 * h)) &
        (hue >= 55) & (hue <= 110) &
        (sat >= 12) & (sat <= 160) &
        (val >= 45) &
        (green.astype(np.int16) - red.astype(np.int16) > 0) &
        (blue.astype(np.int16) - red.astype(np.int16) > -25)
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
    candidates = []

    for comp in comps:
        if comp["area"] < 120:
            continue
        if comp["area"] > 18000:
            continue
        if comp["w"] > 240 or comp["h"] > 180:
            continue
        if comp["fill"] < 0.08:
            continue

        cx_norm = comp["cx"] / max(w, 1)
        cy_norm = comp["cy"] / max(h, 1)

        component_mask = labels == comp["id"]
        mean_b = float(np.mean(blue[component_mask]))
        mean_g = float(np.mean(green[component_mask]))
        mean_r = float(np.mean(red[component_mask]))

        greenish = mean_g - mean_r
        right_bonus = cx_norm * 700
        floor_bonus = cy_norm * 250
        area_bonus = min(comp["area"], 3000) * 0.6
        fill_bonus = comp["fill"] * 400

        score = area_bonus + right_bonus + floor_bonus + fill_bonus + greenish * 8
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

def build_track_score(bgr, roi):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    blue, green, red = cv2.split(bgr)
    lightness = lab[:, :, 0]
    _, sat, val = cv2.split(hsv)

    blur = cv2.GaussianBlur(lightness, (0, 0), 25)
    local_bright = lightness.astype(np.int16) - blur.astype(np.int16)

    beige = (
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

def make_search_mask(shape, center, radius):
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

def detect_drone_led(bgr, roi, path_mask, landing_pad, start_pad, depth=None, previous_center=None):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    h, w = bgr.shape[:2]
    depth = resize_depth(depth, (h, w))

    _, sat, val = cv2.split(hsv)
    blue, green, red = cv2.split(bgr)

    exclude_landing = cv2.dilate(
        landing_pad.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (95, 95)),
        iterations=1
    ).astype(bool)

    exclude_start = cv2.dilate(
        start_pad.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (95, 95)),
        iterations=1
    ).astype(bool)

    exclude = exclude_landing | exclude_start
    allowed = roi & ~exclude

    if previous_center is not None:
        allowed &= make_search_mask((h, w), previous_center, 260)

    white_led = (val >= 165) & (sat <= 180)

    bluish_led = (
        (blue >= 115) &
        (green >= 90) &
        (red >= 55) &
        (val >= 115)
    )

    reddish_led = (
        (red >= 120) &
        (val >= 120) &
        (sat >= 45) &
        (blue < 140)
    )

    bright = allowed & (white_led | bluish_led | reddish_led)

    labels, comps = connected_components(bright)

    distmap = None

    if path_mask is not None and np.count_nonzero(path_mask) > 0:
        inv = (~path_mask).astype(np.uint8) * 255
        distmap = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

    previous_distmap = None

    if previous_center is not None:
        prev_mask = make_search_mask((h, w), previous_center, 1).astype(np.uint8)
        previous_distmap = cv2.distanceTransform((1 - prev_mask).astype(np.uint8), cv2.DIST_L2, 5)

    scene_depth = None

    if depth is not None:
        valid_depth = depth[(depth > 0) & roi]

        if valid_depth.size > 0:
            scene_depth = float(np.median(valid_depth))

    candidates = []

    for comp in comps:
        if comp["area"] < 3:
            continue
        if comp["area"] > 1600:
            continue
        if comp["w"] > 90 or comp["h"] > 90:
            continue

        cx = int(comp["cx"])
        cy = int(comp["cy"])

        r = 75
        x1 = max(0, cx - r)
        x2 = min(w, cx + r)
        y1 = max(0, cy - r)
        y2 = min(h, cy + r)

        local_gray = gray[y1:y2, x1:x2]
        local_val = val[y1:y2, x1:x2]

        dark_fraction = float(np.mean(local_gray < 90))
        very_dark_fraction = float(np.mean(local_gray < 55))
        bright_fraction = float(np.mean(local_val > 150))
        brightness = float(np.mean(val[labels == comp["id"]]))

        depth_score = 0.0
        candidate_depth = sample_depth(depth, [cx, cy], radius=8)

        if scene_depth is not None and candidate_depth is not None:
            closer_amount = scene_depth - candidate_depth
            depth_score = float(np.clip(closer_amount / 700.0, 0.0, 1.0)) * 220.0

        score = 0.0
        score += brightness * 1.15
        score += dark_fraction * 260.0
        score += very_dark_fraction * 240.0
        score += bright_fraction * 80.0
        score += depth_score
        score += comp["area"] * 0.03

        if distmap is not None:
            score -= min(float(distmap[cy, cx]), 260.0) * 0.55

        if previous_distmap is not None:
            score -= min(float(previous_distmap[cy, cx]), 260.0) * 0.85

        if dark_fraction < 0.08:
            score -= 180.0

        if candidate_depth is None and depth is not None:
            score -= 40.0

        comp["score"] = score
        comp["dark_fraction"] = dark_fraction
        comp["very_dark_fraction"] = very_dark_fraction
        comp["bright_fraction"] = bright_fraction
        comp["brightness"] = brightness
        comp["depth"] = candidate_depth
        comp["depth_score"] = depth_score

        candidates.append(comp)

    candidates.sort(key=lambda c: c["score"], reverse=True)

    drone_mask = np.zeros((h, w), dtype=np.uint8)
    drone_center = None

    if candidates and candidates[0]["score"] > 210:
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
        "drone_method": "led",
    }

    return drone_mask, info, debug

def detect_drone_motion(
    bgr,
    previous_bgr,
    roi,
    path_mask,
    landing_pad,
    start_pad,
    depth=None,
    previous_center=None,
):
    h, w = bgr.shape[:2]
    depth = resize_depth(depth, (h, w))

    if previous_bgr is None:
        empty = np.zeros((h, w), dtype=bool)
        info = {
            "drone_center": None,
            "drone_candidates": [],
            "drone_method": "motion_unavailable",
        }
        debug = {
            "drone_motion_raw": empty,
            "drone_motion_clean": empty,
            "drone": empty,
        }
        return empty, info, debug

    if previous_bgr.shape[:2] != (h, w):
        previous_bgr = cv2.resize(previous_bgr, (w, h), interpolation=cv2.INTER_AREA)

    current_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    previous_gray = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)

    current_gray = cv2.GaussianBlur(current_gray, (7, 7), 0)
    previous_gray = cv2.GaussianBlur(previous_gray, (7, 7), 0)

    gray_diff = cv2.absdiff(current_gray, previous_gray)

    color_diff = cv2.absdiff(bgr, previous_bgr)
    color_diff = np.max(color_diff, axis=2)

    motion_raw = ((gray_diff > 18) | (color_diff > 32))

    allowed = roi.copy()

    if previous_center is not None:
        allowed &= make_search_mask((h, w), previous_center, 260)

    near_path = np.zeros((h, w), dtype=bool)

    if path_mask is not None and np.count_nonzero(path_mask) > 0:
        near_path = cv2.dilate(
            path_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (231, 231)),
            iterations=1
        ).astype(bool)

    if previous_center is None and np.count_nonzero(near_path) > 0:
        allowed &= near_path

    if previous_center is not None and np.count_nonzero(near_path) > 0:
        allowed &= (near_path | make_search_mask((h, w), previous_center, 260))

    motion = motion_raw & allowed

    motion = cv2.morphologyEx(
        motion.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    motion = cv2.morphologyEx(
        motion,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )

    motion = cv2.dilate(
        motion,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1
    ).astype(bool)

    labels, comps = connected_components(motion)

    distmap = None

    if path_mask is not None and np.count_nonzero(path_mask) > 0:
        inv = (~path_mask).astype(np.uint8) * 255
        distmap = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

    previous_distmap = None

    if previous_center is not None:
        prev_mask = make_search_mask((h, w), previous_center, 1).astype(np.uint8)
        previous_distmap = cv2.distanceTransform((1 - prev_mask).astype(np.uint8), cv2.DIST_L2, 5)

    scene_depth = None

    if depth is not None:
        valid_depth = depth[(depth > 0) & roi]

        if valid_depth.size > 0:
            scene_depth = float(np.median(valid_depth))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, _, val = cv2.split(hsv)

    candidates = []

    for comp in comps:
        if comp["area"] < 20:
            continue
        if comp["area"] > 14000:
            continue
        if comp["w"] > 220 or comp["h"] > 220:
            continue

        cx = int(comp["cx"])
        cy = int(comp["cy"])

        component_mask = labels == comp["id"]

        x1 = max(0, comp["x"])
        x2 = min(w, comp["x"] + comp["w"])
        y1 = max(0, comp["y"])
        y2 = min(h, comp["y"] + comp["h"])

        local_gray = gray[y1:y2, x1:x2]
        local_val = val[y1:y2, x1:x2]
        local_motion = gray_diff[y1:y2, x1:x2]

        dark_fraction = float(np.mean(local_gray < 95))
        very_dark_fraction = float(np.mean(local_gray < 55))
        bright_fraction = float(np.mean(local_val > 150))
        motion_energy = float(np.mean(local_motion[motion[y1:y2, x1:x2]])) if np.any(motion[y1:y2, x1:x2]) else 0.0

        depth_score = 0.0
        candidate_depth = sample_depth(depth, [cx, cy], radius=8)

        if scene_depth is not None and candidate_depth is not None:
            closer_amount = scene_depth - candidate_depth
            depth_score = float(np.clip(closer_amount / 700.0, 0.0, 1.0)) * 180.0

        score = 0.0
        score += min(comp["area"], 2500) * 0.06
        score += motion_energy * 2.0
        score += dark_fraction * 180.0
        score += very_dark_fraction * 220.0
        score += bright_fraction * 50.0
        score += depth_score

        if distmap is not None:
            score -= min(float(distmap[cy, cx]), 260.0) * 0.30

        if previous_distmap is not None:
            score -= min(float(previous_distmap[cy, cx]), 300.0) * 0.75

        if previous_center is not None:
            d_prev = float(np.hypot(cx - previous_center[0], cy - previous_center[1]))
            if d_prev > 260:
                score -= 300.0

        if comp["fill"] < 0.04:
            score -= 70.0

        if candidate_depth is None and depth is not None:
            score -= 30.0

        comp["score"] = score
        comp["dark_fraction"] = dark_fraction
        comp["very_dark_fraction"] = very_dark_fraction
        comp["bright_fraction"] = bright_fraction
        comp["motion_energy"] = motion_energy
        comp["depth"] = candidate_depth
        comp["depth_score"] = depth_score

        candidates.append(comp)

    candidates.sort(key=lambda c: c["score"], reverse=True)

    drone_mask = np.zeros((h, w), dtype=np.uint8)
    drone_center = None

    if candidates and candidates[0]["score"] > 85:
        best = candidates[0]
        drone_center = [best["cx"], best["cy"]]
        radius = int(np.clip(max(best["w"], best["h"]) * 0.75, 28, 55))
        cv2.circle(drone_mask, (int(best["cx"]), int(best["cy"])), radius, 1, -1)

    drone_mask = drone_mask.astype(bool)

    debug = {
        "drone_motion_raw": motion_raw,
        "drone_motion_clean": motion,
        "drone": drone_mask,
    }

    info = {
        "drone_center": drone_center,
        "drone_candidates": candidates[:5],
        "drone_method": "motion",
    }

    return drone_mask, info, debug

def segment_scene(bgr, depth=None, return_debug=False):
    h, w = bgr.shape[:2]
    depth = resize_depth(depth, (h, w))

    roi = make_floor_roi(bgr)

    landing_pad, landing_center, landing_debug = detect_landing_pad(bgr, roi)
    start_pad, start_center, start_debug = detect_start_pad(bgr, roi)

    track_score, track_debug = build_track_score(bgr, roi)

    path = dijkstra_path(track_score, landing_center, start_center)
    track = path_to_mask(path, (h, w), thickness=max(24, int(round(min(h, w) * 0.055))))

    drone, drone_info, drone_debug = detect_drone_led(
        bgr,
        roi,
        track,
        landing_pad,
        start_pad,
        depth=depth
    )

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
        "track_mask": track,
        "start_pad_mask": start_pad,
        "landing_pad_mask": landing_pad,
        "start_depth": sample_depth(depth, start_center),
        "landing_depth": sample_depth(depth, landing_center),
        "drone_depth": sample_depth(depth, drone_info["drone_center"]),
        "drone_candidates": drone_info["drone_candidates"],
        "drone_method": drone_info.get("drone_method"),
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

def build_fixed_scene(bgr, depth=None):
    seg, info = segment_scene(bgr, depth, return_debug=False)

    path = info.get("path") or []
    track_mask = info.get("track_mask")

    if track_mask is None:
        track_mask = path_to_mask(
            path,
            bgr.shape[:2],
            thickness=max(24, int(round(min(bgr.shape[:2]) * 0.055)))
        )

    fixed_scene = {
        "path": path,
        "track_mask": track_mask,
        "start_pad_center": info.get("start_pad_center"),
        "landing_pad_center": info.get("landing_pad_center"),
        "start_pad_mask": info.get("start_pad_mask"),
        "landing_pad_mask": info.get("landing_pad_mask"),
        "initial_drone_center": info.get("drone_center"),
        "shape": bgr.shape[:2],
    }

    return fixed_scene

def build_fixed_scene_from_points(
    bgr,
    depth=None,
    start_center=None,
    landing_center=None,
    initial_drone_center=None,
    start_radius=55,
    landing_radius=55,
):
    h, w = bgr.shape[:2]
    depth = resize_depth(depth, (h, w))

    if start_center is None or landing_center is None:
        raise ValueError("Both start_center and landing_center are required.")

    start_center = [float(start_center[0]), float(start_center[1])]
    landing_center = [float(landing_center[0]), float(landing_center[1])]

    if initial_drone_center is not None:
        initial_drone_center = [float(initial_drone_center[0]), float(initial_drone_center[1])]

    roi = make_floor_roi(bgr)
    track_score, _ = build_track_score(bgr, roi)

    path = dijkstra_path(track_score, landing_center, start_center)

    track_mask = path_to_mask(
        path,
        (h, w),
        thickness=max(24, int(round(min(h, w) * 0.055)))
    )

    start_pad_mask = point_mask((h, w), start_center, radius=start_radius)
    landing_pad_mask = point_mask((h, w), landing_center, radius=landing_radius)

    if initial_drone_center is None:
        _, drone_info, _ = detect_drone_led(
            bgr,
            roi,
            track_mask,
            landing_pad_mask,
            start_pad_mask,
            depth=depth
        )
        initial_drone_center = drone_info["drone_center"]

    fixed_scene = {
        "path": path,
        "track_mask": track_mask,
        "start_pad_center": start_center,
        "landing_pad_center": landing_center,
        "start_pad_mask": start_pad_mask,
        "landing_pad_mask": landing_pad_mask,
        "initial_drone_center": initial_drone_center,
        "shape": bgr.shape[:2],
    }

    return fixed_scene

def detect_drone_only(
    bgr,
    depth=None,
    fixed_scene=None,
    return_debug=False,
    previous_center=None,
    previous_bgr=None,
):
    if fixed_scene is None:
        raise ValueError("fixed_scene is required. Build it once with build_fixed_scene(...).")

    h, w = bgr.shape[:2]
    depth = resize_depth(depth, (h, w))
    roi = make_floor_roi(bgr)

    track_mask = resize_mask(fixed_scene.get("track_mask"), (h, w))
    start_pad = resize_mask(fixed_scene.get("start_pad_mask"), (h, w))
    landing_pad = resize_mask(fixed_scene.get("landing_pad_mask"), (h, w))

    if previous_bgr is not None:
        drone, drone_info, drone_debug = detect_drone_motion(
            bgr,
            previous_bgr,
            roi,
            track_mask,
            landing_pad,
            start_pad,
            depth=depth,
            previous_center=previous_center
        )

        if drone_info["drone_center"] is None:
            led_drone, led_info, led_debug = detect_drone_led(
                bgr,
                roi,
                track_mask,
                landing_pad,
                start_pad,
                depth=depth,
                previous_center=previous_center
            )

            drone = led_drone
            drone_info = led_info
            drone_debug.update(led_debug)
    elif previous_center is None and fixed_scene.get("initial_drone_center") is not None:
        initial_center = fixed_scene.get("initial_drone_center")
        drone = point_mask((h, w), initial_center, radius=42)
        drone_info = {
            "drone_center": initial_center,
            "drone_candidates": [],
            "drone_method": "manual_initial",
        }
        drone_debug = {
            "drone": drone,
        }
    else:
        drone, drone_info, drone_debug = detect_drone_led(
            bgr,
            roi,
            track_mask,
            landing_pad,
            start_pad,
            depth=depth,
            previous_center=previous_center
        )

    seg = np.zeros((h, w), dtype=np.uint8)
    seg[track_mask] = TRACK
    seg[start_pad] = START_PAD
    seg[landing_pad] = LANDING_PAD
    seg[drone] = DRONE

    info = {
        "start_pad_center": fixed_scene.get("start_pad_center"),
        "landing_pad_center": fixed_scene.get("landing_pad_center"),
        "drone_center": drone_info["drone_center"],
        "path": fixed_scene.get("path") or [],
        "track_mask": track_mask,
        "start_pad_mask": start_pad,
        "landing_pad_mask": landing_pad,
        "drone_depth": sample_depth(depth, drone_info["drone_center"]),
        "drone_candidates": drone_info["drone_candidates"],
        "drone_method": drone_info.get("drone_method"),
    }

    if not return_debug:
        return seg, info

    debug = {
        "track": track_mask,
        "start_pad": start_pad,
        "landing_pad": landing_pad,
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