from pathlib import Path
import json
import math
import sys
import time
import cv2
import numpy as np

from track_detection import segment_scene, draw_points_and_path, normalize_depth_for_view

VALID_SEGMENTS = {
    "short_straight",
    "medium_straight",
    "long_straight",
    "left_curve",
    "right_curve",
}

RDP_EPSILON_PX = 18.0
RESAMPLE_STEP_PX = 8.0
SMOOTH_WINDOW = 7

CURVE_MIN_ANGLE_DEG = 28.0
MIN_STRAIGHT_PX = 35.0
MIN_STRAIGHT_MM = 120.0

SHORT_STRAIGHT_MAX_PX = 85.0
MEDIUM_STRAIGHT_MAX_PX = 170.0
LONG_STRAIGHT_MAX_PX = 285.0

SHORT_STRAIGHT_MAX_MM = 260.0
MEDIUM_STRAIGHT_MAX_MM = 520.0
LONG_STRAIGHT_MAX_MM = 860.0

DEFAULT_HORIZONTAL_FOV_DEG = 69.0
DEFAULT_VERTICAL_FOV_DEG = 42.0

OUTPUT_DIR = Path("track_sequence_estimator_output")

def as_points(path):
    if path is None:
        return np.empty((0, 2), dtype=np.float32)

    pts = np.array(path, dtype=np.float32)

    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float32)

    clean = []

    for p in pts:
        if not clean:
            clean.append(p)
            continue

        if np.linalg.norm(p - clean[-1]) >= 1.0:
            clean.append(p)

    return np.array(clean, dtype=np.float32)

def point_distance(a, b):
    if a is None or b is None:
        return float("inf")

    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))

def orient_path_start_to_landing(path, start_center, landing_center):
    pts = as_points(path)

    if len(pts) < 2:
        return pts

    if start_center is None or landing_center is None:
        return pts

    start_d_first = point_distance(pts[0], start_center)
    start_d_last = point_distance(pts[-1], start_center)
    landing_d_first = point_distance(pts[0], landing_center)
    landing_d_last = point_distance(pts[-1], landing_center)

    normal_score = start_d_first + landing_d_last
    reversed_score = start_d_last + landing_d_first

    if reversed_score < normal_score:
        pts = pts[::-1].copy()

    return pts

def moving_average_path(points, window=7):
    pts = as_points(points)

    if len(pts) < 3 or window <= 1:
        return pts

    if window % 2 == 0:
        window += 1

    if len(pts) < window:
        return pts

    half = window // 2
    out = []

    for i in range(len(pts)):
        if i == 0 or i == len(pts) - 1:
            out.append(pts[i])
            continue

        a = max(0, i - half)
        b = min(len(pts), i + half + 1)
        out.append(np.mean(pts[a:b], axis=0))

    return np.array(out, dtype=np.float32)

def resample_path(points, step_px=8.0):
    pts = as_points(points)

    if len(pts) < 2:
        return pts

    result = [pts[0].copy()]
    carry = 0.0
    previous = pts[0].copy()

    for i in range(1, len(pts)):
        current = pts[i].copy()
        segment = current - previous
        length = float(np.linalg.norm(segment))

        if length < 1e-6:
            continue

        direction = segment / length
        remaining = length

        while carry + remaining >= step_px:
            need = step_px - carry
            new_point = previous + direction * need
            result.append(new_point.copy())
            previous = new_point
            remaining = float(np.linalg.norm(current - previous))
            carry = 0.0

        carry += remaining
        previous = current

    if np.linalg.norm(result[-1] - pts[-1]) > 1.0:
        result.append(pts[-1].copy())

    return np.array(result, dtype=np.float32)

def simplify_path(points, epsilon_px=18.0):
    pts = as_points(points)

    if len(pts) < 3:
        return pts

    contour = pts.reshape(-1, 1, 2).astype(np.float32)
    approx = cv2.approxPolyDP(contour, epsilon_px, False).reshape(-1, 2).astype(np.float32)

    if len(approx) < 2:
        return pts[[0, -1]]

    if np.linalg.norm(approx[0] - pts[0]) > 1.0:
        approx = np.vstack([pts[0], approx])

    if np.linalg.norm(approx[-1] - pts[-1]) > 1.0:
        approx = np.vstack([approx, pts[-1]])

    clean = []

    for p in approx:
        if not clean or np.linalg.norm(p - clean[-1]) >= 10.0:
            clean.append(p)

    if len(clean) < 2:
        clean = [pts[0], pts[-1]]

    return np.array(clean, dtype=np.float32)

def map_simplified_indices(path, simplified):
    pts = as_points(path)
    simp = as_points(simplified)

    if len(pts) == 0 or len(simp) == 0:
        return []

    indices = []
    start = 0

    for p in simp:
        sub = pts[start:]
        if len(sub) == 0:
            indices.append(len(pts) - 1)
            continue

        d = np.linalg.norm(sub - p, axis=1)
        idx = start + int(np.argmin(d))
        indices.append(idx)
        start = idx

    if indices:
        indices[0] = 0
        indices[-1] = len(pts) - 1

    return indices

def estimate_intrinsics(shape):
    h, w = shape
    fx = w / (2.0 * math.tan(math.radians(DEFAULT_HORIZONTAL_FOV_DEG) / 2.0))
    fy = h / (2.0 * math.tan(math.radians(DEFAULT_VERTICAL_FOV_DEG) / 2.0))
    cx = w / 2.0
    cy = h / 2.0

    return fx, fy, cx, cy

def sample_depth(depth, point, radius=5):
    if depth is None or point is None:
        return None

    h, w = depth.shape[:2]
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))

    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)

    values = depth[y1:y2, x1:x2]
    values = values[values > 0]

    if values.size == 0:
        return None

    return float(np.median(values))

def fill_depth_values(points, depth):
    pts = as_points(points)

    if depth is None or len(pts) == 0:
        return None, 0.0

    values = []

    for p in pts:
        values.append(sample_depth(depth, p, radius=5))

    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float32)
    valid = np.isfinite(arr)

    valid_ratio = float(np.count_nonzero(valid) / max(len(arr), 1))

    if np.count_nonzero(valid) == 0:
        return None, 0.0

    if np.count_nonzero(valid) == 1:
        arr[:] = arr[valid][0]
        return arr, valid_ratio

    idx = np.arange(len(arr))
    arr[~valid] = np.interp(idx[~valid], idx[valid], arr[valid])

    return arr, valid_ratio

def deproject_path(points, depth):
    pts = as_points(points)

    if depth is None or len(pts) == 0:
        return None, 0.0

    depths, valid_ratio = fill_depth_values(pts, depth)

    if depths is None:
        return None, 0.0

    h, w = depth.shape[:2]
    fx, fy, cx, cy = estimate_intrinsics((h, w))

    out = []

    for p, z in zip(pts, depths):
        x = (float(p[0]) - cx) * float(z) / fx
        y = (float(p[1]) - cy) * float(z) / fy
        out.append([x, y, float(z)])

    return np.array(out, dtype=np.float32), valid_ratio

def path_length_px(points):
    pts = as_points(points)

    if len(pts) < 2:
        return 0.0

    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

def path_length_depth_mm(points, depth):
    metric, valid_ratio = deproject_path(points, depth)

    if metric is None or len(metric) < 2 or valid_ratio < 0.45:
        return None, valid_ratio

    length = float(np.sum(np.linalg.norm(np.diff(metric, axis=0), axis=1)))

    return length, valid_ratio

def signed_turn_angle(a, b, c):
    p0 = np.array([a[0], -a[1]], dtype=np.float32)
    p1 = np.array([b[0], -b[1]], dtype=np.float32)
    p2 = np.array([c[0], -c[1]], dtype=np.float32)

    v1 = p1 - p0
    v2 = p2 - p1

    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))

    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0

    v1 = v1 / n1
    v2 = v2 / n2

    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))

    return math.degrees(math.atan2(cross, dot))

def classify_straight_from_length(length_px, length_mm):
    if length_mm is not None:
        if length_mm < MIN_STRAIGHT_MM:
            return []

        if length_mm <= SHORT_STRAIGHT_MAX_MM:
            return ["short_straight"]

        if length_mm <= MEDIUM_STRAIGHT_MAX_MM:
            return ["medium_straight"]

        if length_mm <= LONG_STRAIGHT_MAX_MM:
            return ["long_straight"]

        labels = []
        remaining = length_mm

        while remaining > LONG_STRAIGHT_MAX_MM * 1.15:
            labels.append("long_straight")
            remaining -= LONG_STRAIGHT_MAX_MM

        labels.extend(classify_straight_from_length(length_px, remaining))

        return labels

    if length_px < MIN_STRAIGHT_PX:
        return []

    if length_px <= SHORT_STRAIGHT_MAX_PX:
        return ["short_straight"]

    if length_px <= MEDIUM_STRAIGHT_MAX_PX:
        return ["medium_straight"]

    if length_px <= LONG_STRAIGHT_MAX_PX:
        return ["long_straight"]

    labels = []
    remaining = length_px

    while remaining > LONG_STRAIGHT_MAX_PX * 1.15:
        labels.append("long_straight")
        remaining -= LONG_STRAIGHT_MAX_PX

    labels.extend(classify_straight_from_length(remaining, None))

    return labels

def classify_curve(angle_deg):
    if abs(angle_deg) < CURVE_MIN_ANGLE_DEG:
        return None

    if angle_deg > 0:
        return "left_curve"

    return "right_curve"

def postprocess_sequence(sequence):
    out = []

    for label in sequence:
        if label not in VALID_SEGMENTS:
            continue

        if out and label in ("left_curve", "right_curve") and out[-1] == label:
            continue

        out.append(label)

    while out and out[0] in ("left_curve", "right_curve"):
        out.pop(0)

    while out and out[-1] in ("left_curve", "right_curve"):
        out.pop()

    return out

def crop_path_from_drone(path, drone_center):
    pts = as_points(path)

    if drone_center is None or len(pts) < 3:
        return pts, 0

    d = np.linalg.norm(pts - np.array(drone_center, dtype=np.float32), axis=1)
    idx = int(np.argmin(d))

    if idx >= len(pts) - 2:
        return pts, 0

    return pts[idx:], idx

def primitives_from_path(path, depth=None):
    pts = as_points(path)

    if len(pts) < 2:
        return [], {
            "simplified_path": pts.tolist(),
            "mapped_indices": [],
            "primitives": [],
        }

    smoothed = moving_average_path(pts, SMOOTH_WINDOW)
    resampled = resample_path(smoothed, RESAMPLE_STEP_PX)
    simplified = simplify_path(resampled, RDP_EPSILON_PX)

    if len(simplified) < 2:
        simplified = resampled[[0, -1]]

    mapped = map_simplified_indices(resampled, simplified)

    sequence = []
    primitives = []

    for i in range(len(simplified) - 1):
        a_idx = mapped[i]
        b_idx = mapped[i + 1]

        if b_idx <= a_idx:
            continue

        segment_points = resampled[a_idx:b_idx + 1]
        length_px = path_length_px(segment_points)
        length_mm, depth_valid_ratio = path_length_depth_mm(segment_points, depth)

        straight_labels = classify_straight_from_length(length_px, length_mm)

        for label in straight_labels:
            sequence.append(label)
            primitives.append({
                "type": label,
                "kind": "straight",
                "from_index": int(a_idx),
                "to_index": int(b_idx),
                "length_px": float(length_px),
                "length_mm": None if length_mm is None else float(length_mm),
                "depth_valid_ratio": float(depth_valid_ratio),
            })

        if i < len(simplified) - 2:
            angle = signed_turn_angle(simplified[i], simplified[i + 1], simplified[i + 2])
            curve = classify_curve(angle)

            if curve is not None:
                sequence.append(curve)
                primitives.append({
                    "type": curve,
                    "kind": "curve",
                    "at_index": int(b_idx),
                    "angle_deg": float(angle),
                })

    sequence = postprocess_sequence(sequence)

    debug = {
        "smoothed_path": smoothed.tolist(),
        "resampled_path": resampled.tolist(),
        "simplified_path": simplified.tolist(),
        "mapped_indices": [int(x) for x in mapped],
        "primitives": primitives,
    }

    return sequence, debug

def estimate_track_sequence(bgr, depth=None, crop_from_drone=False, return_debug=False):
    seg, info = segment_scene(bgr, depth, return_debug=False)

    raw_path = info.get("path") or []
    start_center = info.get("start_pad_center")
    landing_center = info.get("landing_pad_center")
    drone_center = info.get("drone_center")

    path = orient_path_start_to_landing(raw_path, start_center, landing_center)

    crop_index = 0

    if crop_from_drone:
        path, crop_index = crop_path_from_drone(path, drone_center)

    sequence, path_debug = primitives_from_path(path, depth)

    debug = {
        "sequence": sequence,
        "start_pad_center": start_center,
        "landing_pad_center": landing_center,
        "drone_center": drone_center,
        "crop_from_drone": bool(crop_from_drone),
        "crop_index": int(crop_index),
        "raw_path": as_points(raw_path).tolist(),
        "oriented_path": as_points(path).tolist(),
        "segmentation_info": {
            "start_depth": info.get("start_depth"),
            "landing_depth": info.get("landing_depth"),
            "drone_depth": info.get("drone_depth"),
            "drone_method": info.get("drone_method"),
        },
        **path_debug,
    }

    if return_debug:
        return sequence, debug

    return sequence

def draw_sequence_debug(bgr, debug):
    out = bgr.copy()

    path = np.array(debug.get("oriented_path") or [], dtype=np.float32)
    simplified = np.array(debug.get("simplified_path") or [], dtype=np.float32)

    if len(path) >= 2:
        cv2.polylines(
            out,
            [path.astype(np.int32).reshape(-1, 1, 2)],
            False,
            (0, 180, 255),
            3,
            lineType=cv2.LINE_AA
        )

    if len(simplified) >= 2:
        cv2.polylines(
            out,
            [simplified.astype(np.int32).reshape(-1, 1, 2)],
            False,
            (0, 0, 255),
            2,
            lineType=cv2.LINE_AA
        )

    for i, p in enumerate(simplified):
        x = int(round(p[0]))
        y = int(round(p[1]))
        cv2.circle(out, (x, y), 7, (0, 0, 255), -1)
        cv2.putText(
            out,
            str(i),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    info = {
        "path": path.tolist(),
        "start_pad_center": debug.get("start_pad_center"),
        "landing_pad_center": debug.get("landing_pad_center"),
        "drone_center": debug.get("drone_center"),
    }

    out = draw_points_and_path(out, info)

    y = 24

    for i, label in enumerate(debug.get("sequence", []), start=1):
        cv2.putText(
            out,
            f"{i}. {label}",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )
        y += 22

    return out

def save_estimation_debug(bgr, depth, debug, output_dir=OUTPUT_DIR, stem="track_sequence"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay = draw_sequence_debug(bgr, debug)

    cv2.imwrite(str(output_dir / f"{stem}_sequence_overlay.png"), overlay)
    cv2.imwrite(str(output_dir / f"{stem}_rgb.png"), bgr)

    if depth is not None:
        depth_view = normalize_depth_for_view(depth)

        if depth_view is not None:
            cv2.imwrite(str(output_dir / f"{stem}_depth_view.png"), depth_view)

    with open(output_dir / f"{stem}_sequence.json", "w", encoding="utf-8") as f:
        json.dump(debug, f, indent=2)

    return output_dir / f"{stem}_sequence_overlay.png"

def estimate_track_sequence_from_camera(crop_from_drone=False, return_debug=False, save_debug=True):
    from camera_module import RGBDCamera

    camera = RGBDCamera()
    camera.start()
    time.sleep(1.0)

    try:
        success, bgr, depth = camera.get_frames()

        if not success or bgr is None:
            raise RuntimeError("Could not capture RGB-D frame from camera.")

        sequence, debug = estimate_track_sequence(
            bgr,
            depth,
            crop_from_drone=crop_from_drone,
            return_debug=True
        )

        if save_debug:
            save_estimation_debug(bgr, depth, debug)

        if return_debug:
            return sequence, debug

        return sequence

    finally:
        camera.stop()

def estimate_track_sequence_from_files(rgb_path, depth_path=None, crop_from_drone=False, return_debug=False, save_debug=True):
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {rgb_path}")

    depth = None

    if depth_path is not None and Path(depth_path).exists():
        depth = np.load(str(depth_path))

    sequence, debug = estimate_track_sequence(
        bgr,
        depth,
        crop_from_drone=crop_from_drone,
        return_debug=True
    )

    if save_debug:
        stem = Path(rgb_path).stem
        save_estimation_debug(bgr, depth, debug, stem=stem)

    if return_debug:
        return sequence, debug

    return sequence

def print_sequence(sequence):
    print("TRACK_SEQUENCE = [")
    for item in sequence:
        print(f'    "{item}",')
    print("]")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        rgb_path = sys.argv[1]
        depth_path = sys.argv[2] if len(sys.argv) >= 3 else None
        sequence, debug = estimate_track_sequence_from_files(
            rgb_path,
            depth_path,
            return_debug=True,
            save_debug=True
        )
    else:
        sequence, debug = estimate_track_sequence_from_camera(
            return_debug=True,
            save_debug=True
        )

    print_sequence(sequence)
    print()
    print("Saved debug output to:", OUTPUT_DIR)