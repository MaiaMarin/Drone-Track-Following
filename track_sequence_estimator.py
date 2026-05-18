from pathlib import Path
import json
import math
import cv2
import numpy as np

from track_detection import segment_scene

OUTPUT_DIR = Path("track_sequence_estimator_output")

CURVE_SIGN_FLIP = False

RESAMPLE_STEP_PX = 4.0
SMOOTH_WINDOW = 7

USE_DEPTH_CORRECTED_LENGTH = True
MIN_DEPTH_VALID_RATIO = 0.60
DEPTH_SCALE_CLAMP_LOW = 0.65
DEPTH_SCALE_CLAMP_HIGH = 1.55

SHORT_STRAIGHT_LEN = 50.0
MEDIUM_STRAIGHT_LEN = 82.0
LONG_STRAIGHT_LEN = 140.0
CURVE_LEN = 70.0

STRAIGHT_LENGTH_TOLERANCE_LOW = 0.55
STRAIGHT_LENGTH_TOLERANCE_HIGH = 1.45
CURVE_LENGTH_TOLERANCE_LOW = 0.45
CURVE_LENGTH_TOLERANCE_HIGH = 1.65

CURVE_EXPECTED_ANGLE = 42.0
CURVE_MIN_ANGLE = 16.0
CURVE_MAX_ABS_ANGLE = 95.0

MIN_REMAINING_LEN = 18.0

LABEL_COLORS = {
    "short_straight": (255, 255, 255),
    "medium_straight": (0, 255, 255),
    "long_straight": (0, 165, 255),
    "left_curve": (255, 0, 0),
    "right_curve": (0, 0, 255),
}

SHORT_LABELS = {
    "short_straight": "S",
    "medium_straight": "M",
    "long_straight": "L",
    "left_curve": "LC",
    "right_curve": "RC",
}

TEMPLATE_LENGTHS = {
    "short_straight": SHORT_STRAIGHT_LEN,
    "medium_straight": MEDIUM_STRAIGHT_LEN,
    "long_straight": LONG_STRAIGHT_LEN,
    "left_curve": CURVE_LEN,
    "right_curve": CURVE_LEN,
}


def as_points(path):
    if path is None:
        return np.empty((0, 2), dtype=np.float32)

    pts = np.array(path, dtype=np.float32)

    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float32)

    cleaned = []

    for p in pts:
        if not cleaned:
            cleaned.append(p)
        elif np.linalg.norm(p - cleaned[-1]) >= 1.0:
            cleaned.append(p)

    return np.array(cleaned, dtype=np.float32)


def moving_average(points, window=7):
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


def cumulative_lengths(points):
    pts = as_points(points)

    if len(pts) == 0:
        return np.array([], dtype=np.float32)

    if len(pts) == 1:
        return np.array([0.0], dtype=np.float32)

    diffs = np.diff(pts, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(lengths)]).astype(np.float32)


def resample_path(points, step=4.0):
    pts = as_points(points)

    if len(pts) < 2:
        return pts

    cum = cumulative_lengths(pts)
    total = float(cum[-1])

    if total <= 0:
        return pts

    targets = np.arange(0.0, total, step, dtype=np.float32)

    if len(targets) == 0 or targets[-1] < total:
        targets = np.append(targets, total)

    out = []

    for t in targets:
        idx = int(np.searchsorted(cum, t, side="right") - 1)
        idx = max(0, min(idx, len(pts) - 2))

        span = float(cum[idx + 1] - cum[idx])

        if span <= 1e-6:
            out.append(pts[idx].copy())
            continue

        alpha = float((t - cum[idx]) / span)
        p = pts[idx] * (1.0 - alpha) + pts[idx + 1] * alpha
        out.append(p)

    return np.array(out, dtype=np.float32)


def angle_deltas(points):
    pts = as_points(points)

    if len(pts) < 3:
        return np.array([], dtype=np.float32)

    world = pts.copy()
    world[:, 1] *= -1.0

    seg = np.diff(world, axis=0)
    headings = np.arctan2(seg[:, 1], seg[:, 0])
    deltas = np.diff(headings)
    deltas = (deltas + np.pi) % (2.0 * np.pi) - np.pi

    return np.degrees(deltas).astype(np.float32)


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


def depth_values_for_path(points, depth):
    pts = as_points(points)

    if depth is None or len(pts) == 0:
        return None, 0.0

    vals = []

    for p in pts:
        vals.append(sample_depth(depth, p, radius=5))

    arr = np.array([np.nan if v is None else float(v) for v in vals], dtype=np.float32)
    valid = np.isfinite(arr)
    ratio = float(np.count_nonzero(valid) / max(len(arr), 1))

    if np.count_nonzero(valid) == 0:
        return None, 0.0

    if np.count_nonzero(valid) == 1:
        arr[:] = arr[valid][0]
        return arr, ratio

    idx = np.arange(len(arr))
    arr[~valid] = np.interp(idx[~valid], idx[valid], arr[valid])

    return arr, ratio


def depth_corrected_cumulative_lengths(points, depth):
    pts = as_points(points)
    cum_px = cumulative_lengths(pts)

    if depth is None or len(pts) < 2:
        return cum_px, 0.0, False

    depths, valid_ratio = depth_values_for_path(pts, depth)

    if depths is None or valid_ratio < MIN_DEPTH_VALID_RATIO:
        return cum_px, valid_ratio, False

    valid_depths = depths[np.isfinite(depths)]

    if len(valid_depths) == 0:
        return cum_px, valid_ratio, False

    reference_depth = float(np.median(valid_depths))

    if reference_depth <= 0:
        return cum_px, valid_ratio, False

    diffs = np.diff(pts, axis=0)
    px_lengths = np.linalg.norm(diffs, axis=1)
    pair_depths = (depths[:-1] + depths[1:]) * 0.5
    scale = pair_depths / reference_depth
    scale = np.clip(scale, DEPTH_SCALE_CLAMP_LOW, DEPTH_SCALE_CLAMP_HIGH)

    corrected = px_lengths * scale
    cum = np.concatenate([[0.0], np.cumsum(corrected)]).astype(np.float32)

    return cum, valid_ratio, USE_DEPTH_CORRECTED_LENGTH


def interval_length(cum, a, b):
    if cum is None or len(cum) == 0:
        return 0.0

    a = max(0, min(int(a), len(cum) - 1))
    b = max(0, min(int(b), len(cum) - 1))

    if b < a:
        a, b = b, a

    return float(cum[b] - cum[a])


def interval_angles(deltas, start_idx, end_idx):
    if deltas is None or len(deltas) == 0:
        return 0.0, 0.0, 0.0

    start_idx = int(start_idx)
    end_idx = int(end_idx)

    a = max(0, start_idx)
    b = min(len(deltas), end_idx - 1)

    if b <= a:
        return 0.0, 0.0, 0.0

    segment = deltas[a:b]

    net = float(np.sum(segment))
    total_abs = float(np.sum(np.abs(segment)))
    consistency = abs(net) / max(total_abs, 1e-6)

    return net, total_abs, consistency


def sample_gray(gray, x, y):
    h, w = gray.shape[:2]
    ix = int(round(float(x)))
    iy = int(round(float(y)))

    if ix < 0 or ix >= w or iy < 0 or iy >= h:
        return None

    return float(gray[iy, ix])


def compute_seam_scores(bgr, points):
    pts = as_points(points)

    if bgr is None or len(pts) < 5:
        return np.zeros(len(pts), dtype=np.float32)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scores = np.zeros(len(pts), dtype=np.float32)

    for i in range(2, len(pts) - 2):
        tangent = pts[i + 2] - pts[i - 2]
        norm = float(np.linalg.norm(tangent))

        if norm < 1e-6:
            continue

        tangent = tangent / norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)

        center_vals = []
        before_vals = []
        after_vals = []

        for offset in range(-14, 15, 2):
            p_center = pts[i] + normal * offset
            p_before = pts[i] - tangent * 8 + normal * offset
            p_after = pts[i] + tangent * 8 + normal * offset

            v = sample_gray(gray, p_center[0], p_center[1])
            vb = sample_gray(gray, p_before[0], p_before[1])
            va = sample_gray(gray, p_after[0], p_after[1])

            if v is not None:
                center_vals.append(v)
            if vb is not None:
                before_vals.append(vb)
            if va is not None:
                after_vals.append(va)

        if len(center_vals) < 4 or len(before_vals) < 4 or len(after_vals) < 4:
            continue

        center_mean = float(np.median(center_vals))
        neighbor_mean = float(np.median(before_vals + after_vals))
        darkness = max(0.0, neighbor_mean - center_mean)
        dark_fraction = float(np.mean(np.array(center_vals) < neighbor_mean - 12.0))

        score = (darkness / 35.0) * 0.65 + dark_fraction * 0.35
        scores[i] = float(np.clip(score, 0.0, 1.0))

    if len(scores) >= 7:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        scores = np.convolve(scores, kernel, mode="same").astype(np.float32)

    if np.max(scores) > 0:
        scores = scores / np.max(scores)

    scores[:3] = 0
    scores[-3:] = 0

    return scores.astype(np.float32)


def boundary_bonus(seam_scores, idx, n):
    if seam_scores is None or len(seam_scores) == 0:
        return 0.0

    idx = int(idx)

    if idx <= 2 or idx >= n - 3:
        return 0.0

    a = max(0, idx - 2)
    b = min(len(seam_scores), idx + 3)

    return float(np.max(seam_scores[a:b]))


def score_piece(label, start_idx, end_idx, cum_len, cum_px, deltas, seam_scores, n):
    length = interval_length(cum_len, start_idx, end_idx)
    length_px = interval_length(cum_px, start_idx, end_idx)
    target = TEMPLATE_LENGTHS[label]

    if length <= 1e-6:
        return None

    net_angle, total_abs_angle, consistency = interval_angles(deltas, start_idx, end_idx)
    signed_angle = -net_angle if CURVE_SIGN_FLIP else net_angle
    length_error = abs(length - target) / max(target, 1.0)
    seam = boundary_bonus(seam_scores, end_idx, n)

    if label.endswith("straight"):
        angle_penalty = abs(net_angle) / 16.0
        curvature_penalty = max(0.0, total_abs_angle - 10.0) / 22.0

        if total_abs_angle > 55.0:
            curvature_penalty += 6.0

        score = 0.85 * length_error + angle_penalty + curvature_penalty + 0.08
        score -= 0.18 * seam

        return {
            "type": label,
            "kind": "straight",
            "start_index": int(start_idx),
            "end_index": int(end_idx),
            "score": float(score),
            "length": float(length),
            "length_px": float(length_px),
            "net_angle": float(net_angle),
            "total_abs_angle": float(total_abs_angle),
            "consistency": float(consistency),
            "boundary_seam_score": float(seam),
        }

    desired_sign = 1.0 if label == "left_curve" else -1.0
    signed_for_label = signed_angle * desired_sign
    abs_angle = abs(signed_angle)

    wrong_sign_penalty = 0.0 if signed_for_label > 0 else 8.0
    too_small_penalty = max(0.0, CURVE_MIN_ANGLE - abs_angle) / 4.0
    too_large_penalty = max(0.0, abs_angle - CURVE_MAX_ABS_ANGLE) / 12.0
    angle_error = abs(abs_angle - CURVE_EXPECTED_ANGLE) / CURVE_EXPECTED_ANGLE
    consistency_penalty = max(0.0, 0.45 - consistency) * 2.0

    score = 0.65 * length_error
    score += 0.85 * angle_error
    score += wrong_sign_penalty
    score += too_small_penalty
    score += too_large_penalty
    score += consistency_penalty
    score += 0.10
    score -= 0.15 * seam

    return {
        "type": label,
        "kind": "curve",
        "start_index": int(start_idx),
        "end_index": int(end_idx),
        "score": float(score),
        "length": float(length),
        "length_px": float(length_px),
        "net_angle": float(net_angle),
        "signed_angle": float(signed_angle),
        "total_abs_angle": float(total_abs_angle),
        "consistency": float(consistency),
        "boundary_seam_score": float(seam),
    }


def possible_end_indices(cum_len, start_idx, label, n):
    target = TEMPLATE_LENGTHS[label]
    remaining = interval_length(cum_len, start_idx, n - 1)

    if label.endswith("straight"):
        low = target * STRAIGHT_LENGTH_TOLERANCE_LOW
        high = target * STRAIGHT_LENGTH_TOLERANCE_HIGH
    else:
        low = target * CURVE_LENGTH_TOLERANCE_LOW
        high = target * CURVE_LENGTH_TOLERANCE_HIGH

    if remaining < high:
        high = remaining

    if remaining < low and remaining >= MIN_REMAINING_LEN:
        low = remaining * 0.55
        high = remaining

    start_len = float(cum_len[start_idx])
    min_target = start_len + low
    max_target = start_len + high

    j_min = int(np.searchsorted(cum_len, min_target, side="left"))
    j_max = int(np.searchsorted(cum_len, max_target, side="right"))

    j_min = max(start_idx + 2, min(j_min, n - 1))
    j_max = max(j_min, min(j_max, n - 1))

    indices = list(range(j_min, j_max + 1, 2))

    if n - 1 not in indices and remaining <= high * 1.1:
        indices.append(n - 1)

    return sorted(set(indices))


def generate_candidates(start_idx, cum_len, cum_px, deltas, seam_scores, n):
    labels = [
        "short_straight",
        "medium_straight",
        "long_straight",
        "left_curve",
        "right_curve",
    ]

    candidates = []

    for label in labels:
        scored = []

        for end_idx in possible_end_indices(cum_len, start_idx, label, n):
            candidate = score_piece(label, start_idx, end_idx, cum_len, cum_px, deltas, seam_scores, n)

            if candidate is None:
                continue

            if candidate["kind"] == "curve":
                if abs(candidate["signed_angle"]) < CURVE_MIN_ANGLE * 0.55:
                    continue

            scored.append(candidate)

        scored.sort(key=lambda x: x["score"])
        candidates.extend(scored[:4])

    return candidates


def dynamic_piece_fit(points, depth=None, bgr=None):
    pts = as_points(points)

    if len(pts) < 2:
        return [], {
            "working_centerline": pts.tolist(),
            "pieces": [],
            "sequence": [],
            "depth_valid_ratio": 0.0,
            "used_depth_corrected_length": False,
            "length_px": 0.0,
            "length_used": 0.0,
        }

    cum_px = cumulative_lengths(pts)
    corrected_cum, depth_ratio, depth_ok = depth_corrected_cumulative_lengths(pts, depth)

    if USE_DEPTH_CORRECTED_LENGTH and depth_ok:
        cum_len = corrected_cum
        used_depth = True
    else:
        cum_len = cum_px
        used_depth = False

    deltas = angle_deltas(pts)
    seam_scores = compute_seam_scores(bgr, pts)
    n = len(pts)

    dp = np.full(n, np.inf, dtype=np.float32)
    choice = [None for _ in range(n)]

    dp[n - 1] = 0.0

    for i in range(n - 2, -1, -1):
        remaining = interval_length(cum_len, i, n - 1)

        if remaining < MIN_REMAINING_LEN:
            dp[i] = 0.0
            choice[i] = {
                "type": "skip",
                "kind": "skip",
                "start_index": int(i),
                "end_index": int(n - 1),
                "score": 0.0,
            }
            continue

        candidates = generate_candidates(i, cum_len, cum_px, deltas, seam_scores, n)

        for candidate in candidates:
            j = int(candidate["end_index"])

            if j <= i or j >= n:
                continue

            total = candidate["score"] + float(dp[j])

            if total < dp[i]:
                dp[i] = total
                choice[i] = candidate

    pieces = []
    idx = 0
    safety = 0

    while idx < n - 1 and safety < n * 2:
        safety += 1
        selected = choice[idx]

        if selected is None:
            next_idx = min(n - 1, idx + 8)
            fallback = score_piece(
                "short_straight",
                idx,
                next_idx,
                cum_len,
                cum_px,
                deltas,
                seam_scores,
                n
            )

            if fallback is None:
                break

            selected = fallback

        if selected["type"] != "skip":
            pieces.append(selected)

        next_idx = int(selected["end_index"])

        if next_idx <= idx:
            next_idx = idx + 1

        idx = next_idx

    sequence = [piece["type"] for piece in pieces]

    debug = {
        "working_centerline": pts.tolist(),
        "pieces": pieces,
        "sequence": sequence,
        "depth_valid_ratio": float(depth_ratio),
        "used_depth_corrected_length": bool(used_depth),
        "length_px": float(cum_px[-1]) if len(cum_px) else 0.0,
        "length_used": float(cum_len[-1]) if len(cum_len) else 0.0,
        "seam_scores": seam_scores.tolist(),
        "total_score": None if not np.isfinite(dp[0]) else float(dp[0]),
    }

    return pieces, debug


def estimate_pieces_from_centerline(centerline, depth=None, bgr=None):
    raw = as_points(centerline)
    smoothed = moving_average(raw, SMOOTH_WINDOW)
    pts = resample_path(smoothed, RESAMPLE_STEP_PX)

    pieces, debug = dynamic_piece_fit(pts, depth=depth, bgr=bgr)

    return pieces, debug


def piece_midpoint(points, piece):
    pts = as_points(points)

    if len(pts) == 0:
        return None

    a = int(piece["start_index"])
    b = int(piece["end_index"])

    a = max(0, min(a, len(pts) - 1))
    b = max(0, min(b, len(pts) - 1))

    m = (a + b) // 2

    return pts[m]


def draw_text_with_outline(image, text, origin, scale, color, thickness=1):
    x, y = origin

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        lineType=cv2.LINE_AA
    )

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        lineType=cv2.LINE_AA
    )


def draw_piece_overlay(bgr, detection_info, sequence_debug):
    out = bgr.copy()

    track_mask = detection_info.get("track_mask")
    start_pad = detection_info.get("start_pad_mask")
    landing_pad = detection_info.get("landing_pad_mask")

    seg_overlay = np.zeros_like(out)

    if track_mask is not None:
        seg_overlay[track_mask.astype(bool)] = (0, 180, 255)

    if start_pad is not None:
        seg_overlay[start_pad.astype(bool)] = (0, 255, 0)

    if landing_pad is not None:
        seg_overlay[landing_pad.astype(bool)] = (255, 255, 0)

    visible = np.any(seg_overlay > 0, axis=2)
    blended = cv2.addWeighted(out, 0.82, seg_overlay, 0.18, 0)
    out[visible] = blended[visible]

    pts = as_points(sequence_debug.get("working_centerline") or [])
    pieces = sequence_debug.get("pieces") or []

    if len(pts) >= 2:
        cv2.polylines(
            out,
            [pts.astype(np.int32).reshape(-1, 1, 2)],
            False,
            (0, 0, 255),
            1,
            lineType=cv2.LINE_AA
        )

    for i, piece in enumerate(pieces, start=1):
        a = int(piece["start_index"])
        b = int(piece["end_index"])

        a = max(0, min(a, len(pts) - 1))
        b = max(0, min(b, len(pts) - 1))

        segment = pts[a:b + 1]

        if len(segment) >= 2:
            color = LABEL_COLORS.get(piece["type"], (255, 255, 255))

            cv2.polylines(
                out,
                [segment.astype(np.int32).reshape(-1, 1, 2)],
                False,
                color,
                2,
                lineType=cv2.LINE_AA
            )

        p0 = pts[a]
        p1 = pts[b]
        pm = piece_midpoint(pts, piece)

        cv2.circle(out, (int(p0[0]), int(p0[1])), 3, (0, 0, 255), -1)
        cv2.circle(out, (int(p1[0]), int(p1[1])), 3, (0, 0, 255), -1)

        if pm is not None:
            x = int(pm[0])
            y = int(pm[1])
            short_label = SHORT_LABELS.get(piece["type"], piece["type"])
            color = LABEL_COLORS.get(piece["type"], (255, 255, 255))
            draw_text_with_outline(out, f"{i}:{short_label}", (x + 5, y - 5), 0.42, color, 1)

    start = detection_info.get("start_pad_center")
    landing = detection_info.get("landing_pad_center")

    if start is not None:
        cv2.circle(out, (int(start[0]), int(start[1])), 9, (0, 255, 0), -1)
        draw_text_with_outline(out, "START", (int(start[0]) + 12, int(start[1]) - 12), 0.62, (0, 255, 0), 2)

    if landing is not None:
        cv2.circle(out, (int(landing[0]), int(landing[1])), 9, (255, 255, 0), -1)
        draw_text_with_outline(out, "END", (int(landing[0]) + 12, int(landing[1]) - 12), 0.62, (255, 255, 0), 2)

    y = 24

    draw_text_with_outline(
        out,
        f"pieces: {len(pieces)}",
        (12, y),
        0.48,
        (255, 255, 255),
        1
    )

    y += 20

    draw_text_with_outline(
        out,
        f"depth corrected: {sequence_debug.get('used_depth_corrected_length')}",
        (12, y),
        0.48,
        (255, 255, 255),
        1
    )

    y += 24

    for i, label in enumerate(sequence_debug.get("sequence") or [], start=1):
        short_label = SHORT_LABELS.get(label, label)
        text = f"{i}. {short_label}"

        draw_text_with_outline(
            out,
            text,
            (12, y),
            0.42,
            LABEL_COLORS.get(label, (255, 255, 255)),
            1
        )

        y += 16

        if y > out.shape[0] - 18:
            break

    return out


def make_json_safe(value):
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items() if "mask" not in k}

    if isinstance(value, list):
        return [make_json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


def estimate_track_sequence(bgr, depth=None, return_debug=False):
    seg, info, detection_debug = segment_scene(bgr, depth, return_debug=True)

    centerline = info.get("centerline") or info.get("path") or []
    pieces, sequence_debug = estimate_pieces_from_centerline(centerline, depth=depth, bgr=bgr)

    sequence = [piece["type"] for piece in pieces]

    debug = {
        "sequence": sequence,
        "detection_info": info,
        "sequence_debug": sequence_debug,
    }

    if return_debug:
        return sequence, debug

    return sequence


def save_results(rgb_path, depth_path=None, output_dir=OUTPUT_DIR, show=True):
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

    sequence, debug = estimate_track_sequence(bgr, depth, return_debug=True)

    overlay = draw_piece_overlay(
        bgr,
        debug["detection_info"],
        debug["sequence_debug"]
    )

    stem = rgb_path.stem
    overlay_path = output_dir / f"{stem}_piece_sequence_overlay.png"
    json_path = output_dir / f"{stem}_piece_sequence.json"

    cv2.imwrite(str(overlay_path), overlay)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(debug), f, indent=2)

    print("Saved piece overlay:", overlay_path)
    print("Saved sequence JSON:", json_path)
    print()
    print("TRACK_SEQUENCE = [")
    for item in sequence:
        print(f'    "{item}",')
    print("]")

    if show:
        cv2.namedWindow("Track sequence estimator", cv2.WINDOW_NORMAL)
        cv2.imshow("Track sequence estimator", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return sequence, debug

#test of frame I had in my pc
# if __name__ == "__main__":
#     test_rgb = Path("frame_1779104513_rgb.png")
#     test_depth = Path("frame_1779104513_depth.npy")

#     if test_rgb.exists():
#         save_results(test_rgb, test_depth if test_depth.exists() else None, show=True)
#     else:
#         print(f"Test frame not found: {test_rgb}")