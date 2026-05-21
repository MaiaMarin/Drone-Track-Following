from pathlib import Path
import json
import time
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

SHORT_STRAIGHT_LEN = 42.0
MEDIUM_STRAIGHT_LEN = 68.0
LONG_STRAIGHT_LEN = 108.0
CURVE_LEN = 56.0

FINE_LENGTH_SCALES = [1.00, 0.88, 0.76, 0.66, 0.58]
TARGET_PIECE_LENGTH = 52.0
MIN_TARGET_PIECES = 15
MAX_TARGET_PIECES = 24
MAX_REASONABLE_PIECES = 32

STRAIGHT_LENGTH_TOLERANCE_LOW = 0.45
STRAIGHT_LENGTH_TOLERANCE_HIGH = 1.42
CURVE_LENGTH_TOLERANCE_LOW = 0.38
CURVE_LENGTH_TOLERANCE_HIGH = 1.60

CURVE_EXPECTED_ANGLE = 42.0
CURVE_MIN_ANGLE = 12.0
CURVE_MAX_ABS_ANGLE = 115.0

MIN_REMAINING_LEN = 10.0
PIECE_REWARD = 0.055
SEAM_REWARD = 0.28

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

BASE_TEMPLATE_LENGTHS = {
    "short_straight": SHORT_STRAIGHT_LEN,
    "medium_straight": MEDIUM_STRAIGHT_LEN,
    "long_straight": LONG_STRAIGHT_LEN,
    "left_curve": CURVE_LEN,
    "right_curve": CURVE_LEN,
}


def template_length(label, length_scale=1.0):
    return BASE_TEMPLATE_LENGTHS[label] * float(length_scale)


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

        for offset in range(-16, 17, 2):
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
        dark_fraction = float(np.mean(np.array(center_vals) < neighbor_mean - 10.0))

        score = (darkness / 30.0) * 0.62 + dark_fraction * 0.38
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

    a = max(0, idx - 3)
    b = min(len(seam_scores), idx + 4)

    return float(np.max(seam_scores[a:b]))


def score_piece(label, start_idx, end_idx, cum_len, cum_px, deltas, seam_scores, n, length_scale=1.0):
    length = interval_length(cum_len, start_idx, end_idx)
    length_px = interval_length(cum_px, start_idx, end_idx)
    target = template_length(label, length_scale)

    if length <= 1e-6:
        return None

    net_angle, total_abs_angle, consistency = interval_angles(deltas, start_idx, end_idx)
    signed_angle = -net_angle if CURVE_SIGN_FLIP else net_angle
    length_error = abs(length - target) / max(target, 1.0)
    seam = boundary_bonus(seam_scores, end_idx, n)

    if label.endswith("straight"):
        angle_penalty = abs(net_angle) / 18.0
        curvature_penalty = max(0.0, total_abs_angle - 12.0) / 28.0

        if total_abs_angle > 70.0:
            curvature_penalty += 7.0

        label_penalty = 0.0

        if label == "medium_straight":
            label_penalty = 0.012

        if label == "long_straight":
            label_penalty = 0.075

        score = 0.70 * length_error + angle_penalty + curvature_penalty + label_penalty
        score -= PIECE_REWARD
        score -= SEAM_REWARD * seam

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
            "length_scale": float(length_scale),
        }

    desired_sign = 1.0 if label == "left_curve" else -1.0
    signed_for_label = signed_angle * desired_sign
    abs_angle = abs(signed_angle)

    wrong_sign_penalty = 0.0 if signed_for_label > 0 else 8.0
    too_small_penalty = max(0.0, CURVE_MIN_ANGLE - abs_angle) / 4.0
    too_large_penalty = max(0.0, abs_angle - CURVE_MAX_ABS_ANGLE) / 12.0
    angle_error = abs(abs_angle - CURVE_EXPECTED_ANGLE) / CURVE_EXPECTED_ANGLE
    consistency_penalty = max(0.0, 0.38 - consistency) * 2.0

    score = 0.52 * length_error
    score += 0.70 * angle_error
    score += wrong_sign_penalty
    score += too_small_penalty
    score += too_large_penalty
    score += consistency_penalty
    score -= PIECE_REWARD
    score -= SEAM_REWARD * seam

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
        "length_scale": float(length_scale),
    }


def possible_end_indices(cum_len, start_idx, label, n, length_scale=1.0):
    target = template_length(label, length_scale)
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
        low = remaining * 0.40
        high = remaining

    start_len = float(cum_len[start_idx])
    min_target = start_len + low
    max_target = start_len + high

    j_min = int(np.searchsorted(cum_len, min_target, side="left"))
    j_max = int(np.searchsorted(cum_len, max_target, side="right"))

    j_min = max(start_idx + 2, min(j_min, n - 1))
    j_max = max(j_min, min(j_max, n - 1))

    indices = list(range(j_min, j_max + 1, 1))

    if n - 1 not in indices and remaining <= high * 1.1:
        indices.append(n - 1)

    return sorted(set(indices))


def generate_candidates(start_idx, cum_len, cum_px, deltas, seam_scores, n, length_scale=1.0):
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

        for end_idx in possible_end_indices(cum_len, start_idx, label, n, length_scale=length_scale):
            candidate = score_piece(
                label,
                start_idx,
                end_idx,
                cum_len,
                cum_px,
                deltas,
                seam_scores,
                n,
                length_scale=length_scale
            )

            if candidate is None:
                continue

            if candidate["kind"] == "curve":
                if abs(candidate["signed_angle"]) < CURVE_MIN_ANGLE * 0.45:
                    continue

            scored.append(candidate)

        scored.sort(key=lambda x: x["score"])
        candidates.extend(scored[:12])

    return candidates


def dynamic_piece_fit(points, depth=None, bgr=None, length_scale=1.0):
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
            "length_scale": float(length_scale),
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

        candidates = generate_candidates(
            i,
            cum_len,
            cum_px,
            deltas,
            seam_scores,
            n,
            length_scale=length_scale
        )

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
            next_idx = min(n - 1, idx + 6)
            fallback = score_piece(
                "short_straight",
                idx,
                next_idx,
                cum_len,
                cum_px,
                deltas,
                seam_scores,
                n,
                length_scale=length_scale
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
        "length_scale": float(length_scale),
    }

    return pieces, debug


def target_piece_count(length_used):
    if length_used <= 0:
        return MIN_TARGET_PIECES

    target = int(round(length_used / TARGET_PIECE_LENGTH))
    target = max(MIN_TARGET_PIECES, target)
    target = min(MAX_TARGET_PIECES, target)

    return target


def choose_best_fit(fits):
    valid = []

    for pieces, debug in fits:
        count = len(pieces)

        if count == 0:
            continue

        if count <= MAX_REASONABLE_PIECES:
            valid.append((pieces, debug))

    if not valid:
        return fits[0]

    length_used = max(float(debug.get("length_used") or 0.0) for _, debug in valid)
    target = target_piece_count(length_used)

    def ranking(item):
        pieces, debug = item
        count = len(pieces)
        total_score = debug.get("total_score")

        if total_score is None:
            total_score = 9999.0

        under_penalty = max(0, target - count) * 2.0
        over_penalty = max(0, count - target) * 0.55
        count_distance = abs(count - target)

        return (
            under_penalty + over_penalty + count_distance * 0.35,
            total_score * 0.04,
            -count,
        )

    valid.sort(key=ranking)

    chosen_pieces, chosen_debug = valid[0]
    chosen_debug["target_piece_count"] = int(target)

    return chosen_pieces, chosen_debug


def estimate_pieces_from_centerline(centerline, depth=None, bgr=None):
    raw = as_points(centerline)
    smoothed = moving_average(raw, SMOOTH_WINDOW)
    pts = resample_path(smoothed, RESAMPLE_STEP_PX)

    fits = []

    for scale in FINE_LENGTH_SCALES:
        pieces, debug = dynamic_piece_fit(pts, depth=depth, bgr=bgr, length_scale=scale)
        fits.append((pieces, debug))

    pieces, debug = choose_best_fit(fits)

    debug["candidate_piece_counts"] = [
        {
            "length_scale": float(candidate_debug.get("length_scale")),
            "pieces": int(len(candidate_pieces)),
            "total_score": candidate_debug.get("total_score"),
        }
        for candidate_pieces, candidate_debug in fits
    ]

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
        f"target: {sequence_debug.get('target_piece_count')}",
        (12, y),
        0.48,
        (255, 255, 255),
        1
    )

    y += 20

    draw_text_with_outline(
        out,
        f"scale: {sequence_debug.get('length_scale')}",
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
        "manual_points": False,
    }

    if return_debug:
        return sequence, debug

    return sequence


def estimate_track_sequence_from_points(
    bgr,
    depth=None,
    start_center=None,
    landing_center=None,
    return_debug=False,
    start_radius=55,
    landing_radius=55,
):
    if start_center is None or landing_center is None:
        raise ValueError("start_center and landing_center are required.")

    seg, info, detection_debug = segment_scene(
        bgr,
        depth=depth,
        return_debug=True,
        start_center=start_center,
        landing_center=landing_center,
        start_radius=start_radius,
        landing_radius=landing_radius,
    )

    centerline = info.get("centerline") or info.get("path") or []
    pieces, sequence_debug = estimate_pieces_from_centerline(centerline, depth=depth, bgr=bgr)

    sequence = [piece["type"] for piece in pieces]

    debug = {
        "sequence": sequence,
        "detection_info": info,
        "sequence_debug": sequence_debug,
        "manual_points": True,
        "manual_start_center": [float(start_center[0]), float(start_center[1])],
        "manual_landing_center": [float(landing_center[0]), float(landing_center[1])],
    }

    if return_debug:
        return sequence, debug

    return sequence


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
            x, y = int(point[0]), int(point[1])

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


def save_sequence_outputs(bgr, depth, sequence, debug, output_dir=OUTPUT_DIR, prefix="track", show=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay = draw_piece_overlay(
        bgr,
        debug["detection_info"],
        debug["sequence_debug"]
    )

    rgb_path = output_dir / f"{prefix}_rgb.png"
    depth_path = output_dir / f"{prefix}_depth.npy"
    overlay_path = output_dir / f"{prefix}_piece_sequence_overlay.png"
    json_path = output_dir / f"{prefix}_piece_sequence.json"

    cv2.imwrite(str(rgb_path), bgr)

    if depth is not None:
        np.save(str(depth_path), depth)

    cv2.imwrite(str(overlay_path), overlay)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(debug), f, indent=2)

    print("Saved RGB frame:", rgb_path)

    if depth is not None:
        print("Saved depth map:", depth_path)

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

    return overlay_path, json_path


def save_results(rgb_path, depth_path=None, output_dir=OUTPUT_DIR, show=True):
    rgb_path = Path(rgb_path)

    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {rgb_path}")

    depth = None

    if depth_path is not None:
        depth_path = Path(depth_path)

        if depth_path.exists():
            depth = np.load(str(depth_path))

    sequence, debug = estimate_track_sequence(bgr, depth, return_debug=True)

    return save_sequence_outputs(
        bgr,
        depth,
        sequence,
        debug,
        output_dir=output_dir,
        prefix=rgb_path.stem,
        show=show
    )


def save_results_from_manual_points(rgb_path, depth_path=None, output_dir=OUTPUT_DIR, show=True):
    rgb_path = Path(rgb_path)

    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {rgb_path}")

    depth = None

    if depth_path is not None:
        depth_path = Path(depth_path)

        if depth_path.exists():
            depth = np.load(str(depth_path))

    start_center, landing_center = select_start_end_points(bgr)

    sequence, debug = estimate_track_sequence_from_points(
        bgr,
        depth=depth,
        start_center=start_center,
        landing_center=landing_center,
        return_debug=True
    )

    return save_sequence_outputs(
        bgr,
        depth,
        sequence,
        debug,
        output_dir=output_dir,
        prefix=f"{rgb_path.stem}_manual",
        show=show
    )


def capture_frame_from_camera():
    try:
        from camera_module import RGBDCamera
    except ImportError:
        from rgbd_camera import RGBDCamera

    camera = RGBDCamera()
    camera.start()

    try:
        print("Camera started.")
        print("A preview window will open.")
        print("Press Enter/Space/c to capture the frame. Press q/Esc to cancel.")

        latest_bgr = None
        latest_depth = None

        cv2.namedWindow("Live camera - capture frame", cv2.WINDOW_NORMAL)

        for _ in range(15):
            success, bgr, depth = camera.get_frames()

            if success and bgr is not None and depth is not None:
                latest_bgr = bgr
                latest_depth = depth

            time.sleep(0.03)

        while True:
            success, bgr, depth = camera.get_frames()

            if success and bgr is not None and depth is not None:
                latest_bgr = bgr
                latest_depth = depth

            if latest_bgr is not None:
                display = latest_bgr.copy()
                cv2.putText(
                    display,
                    "Enter/Space/c = capture   q/Esc = cancel",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    lineType=cv2.LINE_AA
                )
                cv2.imshow("Live camera - capture frame", display)

            key = cv2.waitKey(1) & 0xFF

            if key in (10, 13, 32, ord("c")):
                if latest_bgr is None or latest_depth is None:
                    raise RuntimeError("No valid frame captured from camera.")

                captured_bgr = latest_bgr.copy()
                captured_depth = latest_depth.copy()
                cv2.destroyWindow("Live camera - capture frame")
                return captured_bgr, captured_depth

            if key in (27, ord("q")):
                cv2.destroyWindow("Live camera - capture frame")
                raise RuntimeError("camera capture cancelled.")

    finally:
        camera.stop()


def run_live_manual_sequence_capture(output_dir=OUTPUT_DIR, show=True):
    bgr, depth = capture_frame_from_camera()

    start_center, landing_center = select_start_end_points(bgr)

    sequence, debug = estimate_track_sequence_from_points(
        bgr,
        depth=depth,
        start_center=start_center,
        landing_center=landing_center,
        return_debug=True
    )

    timestamp = int(time.time())
    prefix = f"live_manual_{timestamp}"

    save_sequence_outputs(
        bgr,
        depth,
        sequence,
        debug,
        output_dir=output_dir,
        prefix=prefix,
        show=show
    )

    return sequence, debug


if __name__ == "__main__":
    run_live_manual_sequence_capture(show=True)