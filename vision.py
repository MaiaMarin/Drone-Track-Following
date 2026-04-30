"""
vision.py
---------
all computer vision for the static top-down camera.
no depth stream needed here — altitude is handled by the drone's own
bottom range sensor (see movement_controller.py).

strategy:
  1. threshold the frame to isolate the wooden track (sandy/tan colour)
  2. keep only the largest connected blob to discard floor noise
  3. skeletonize the track mask to get a single-pixel centerline path
  4. detect the drone as the dominant squarish blob brighter than the floor
  5. project the drone onto the skeleton, pick a lookahead point ahead of it
  6. return a normalised (dx, dy) direction vector toward that lookahead
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize


# ── tuning ────────────────────────────────────────────────────────────────────

LOOKAHEAD_PX   = 60    # pixels ahead on the skeleton to aim for
MIN_TRACK_AREA = 3000  # minimum contour area to be considered the track
MIN_DRONE_AREA = 400   # minimum blob area counted as drone
MAX_DRONE_AREA = 8000  # maximum blob area (filters large floor patches)


# ── track detection ───────────────────────────────────────────────────────────

def detect_track_mask(color_frame):
    """
    isolate the wooden track from a dark wood floor.
    the track is sandy/tan and significantly lighter than the dark floor.
    the yellow landing pad is explicitly subtracted so the skeleton
    never bleeds onto it.

    returns a binary mask (white = track).
    """
    hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)

    # sandy/tan track: hue 8-28, moderate sat, HIGH value (floor is much darker)
    lower = np.array([8,  30,  130])
    upper = np.array([28, 160, 255])
    mask  = cv2.inRange(hsv, lower, upper)

    # remove yellow landing pad — dilated so fringe is also removed
    lower_yellow = np.array([18, 60, 100])
    upper_yellow = np.array([38, 255, 255])
    yellow_mask  = cv2.inRange(hsv, lower_yellow, upper_yellow)
    kernel_y     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    yellow_mask  = cv2.dilate(yellow_mask, kernel_y, iterations=1)
    mask         = cv2.bitwise_and(mask, cv2.bitwise_not(yellow_mask))

    # close small gaps, remove specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # keep only the single largest blob — the real track is one connected region
    mask = _keep_largest_blob(mask, min_area=MIN_TRACK_AREA)

    return mask


def _keep_largest_blob(mask, min_area=0):
    """return a mask with only the largest connected component kept."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return np.zeros_like(mask)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [largest], -1, 255, -1)
    return out


def get_track_skeleton(track_mask):
    """
    reduce the track mask to a single-pixel centerline and return it as
    an ordered list of (x, y) points from the left entry toward the yellow pad.
    returns an empty list if the track is not found.
    """
    bool_mask = (track_mask > 0).astype(bool)
    skeleton  = skeletonize(bool_mask)

    ys, xs = np.where(skeleton)
    if len(xs) < 10:
        return []

    points    = list(zip(xs.tolist(), ys.tolist()))
    start_idx = int(np.argmin(xs))
    return _sort_skeleton_points(points, start_idx)


def _sort_skeleton_points(points, start_idx):
    """greedy nearest-neighbour walk to order skeleton pixels into a path."""
    pts    = list(points)
    result = [pts.pop(start_idx)]
    while pts:
        last        = result[-1]
        dists       = [abs(p[0] - last[0]) + abs(p[1] - last[1]) for p in pts]
        nearest_idx = int(np.argmin(dists))
        result.append(pts.pop(nearest_idx))
    return result


# ── drone detection ───────────────────────────────────────────────────────────

def detect_drone_position(color_frame):
    """
    find the drone's pixel centroid.
    the drone top plate is lighter than the dark wood floor, so we
    threshold for light pixels using otsu then subtract track and pad regions.

    returns (cx, cy) or None.
    """
    gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

    # otsu finds the natural split between dark floor and lighter objects
    _, light = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # subtract track so the track doesn't get picked as the drone
    track_mask = detect_track_mask(color_frame)
    light      = cv2.bitwise_and(light, cv2.bitwise_not(track_mask))

    # subtract yellow pad area
    hsv          = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([18, 60, 100])
    upper_yellow = np.array([38, 255, 255])
    yellow_mask  = cv2.inRange(hsv, lower_yellow, upper_yellow)
    kernel_y     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    yellow_mask  = cv2.dilate(yellow_mask, kernel_y, iterations=1)
    light        = cv2.bitwise_and(light, cv2.bitwise_not(yellow_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    light  = cv2.morphologyEx(light, cv2.MORPH_OPEN,  kernel, iterations=1)
    light  = cv2.morphologyEx(light, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if MIN_DRONE_AREA < cv2.contourArea(c) < MAX_DRONE_AREA]
    if not valid:
        return None

    best, best_score = None, float('inf')
    for c in valid:
        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 3.0:
            continue
        score = aspect + abs(cv2.contourArea(c) - 3000) / 1000
        if score < best_score:
            best_score, best = score, c

    if best is None:
        return None

    M = cv2.moments(best)
    if M["m00"] == 0:
        return None

    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


# ── lookahead direction ───────────────────────────────────────────────────────

def get_lookahead_direction(drone_pos, skeleton_path):
    """
    find the lookahead point on the skeleton and return the normalised
    direction vector from the drone toward it.

    returns (dx, dy, lookahead_pt, closest_pt) with dx/dy in [-1, 1],
    or None if either input is missing.
    """
    if not skeleton_path or drone_pos is None:
        return None

    pts = np.array(skeleton_path, dtype=np.float32)
    dp  = np.array(drone_pos,     dtype=np.float32)

    dists       = np.linalg.norm(pts - dp, axis=1)
    closest_idx = int(np.argmin(dists))

    lookahead_idx = closest_idx
    accumulated   = 0.0
    for i in range(closest_idx, len(pts) - 1):
        accumulated += np.linalg.norm(pts[i + 1] - pts[i])
        if accumulated >= LOOKAHEAD_PX:
            lookahead_idx = i + 1
            break
    else:
        lookahead_idx = len(pts) - 1

    lookahead_pt = pts[lookahead_idx]
    closest_pt   = pts[closest_idx]
    direction    = lookahead_pt - dp
    norm         = np.linalg.norm(direction)

    if norm < 1e-6:
        return None

    dx, dy = (direction / norm).tolist()
    return (
        float(dx), float(dy),
        (int(lookahead_pt[0]), int(lookahead_pt[1])),
        (int(closest_pt[0]),  int(closest_pt[1])),
    )


# ── yellow landing pad ────────────────────────────────────────────────────────

def detect_yellow_landing(color_frame):
    """
    detect the yellow landing square via hsv thresholding.
    lowered saturation minimum to 60 so the washed-out/printed pad still detects.
    returns (mask, landing_detected_bool).
    """
    hsv   = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
    # saturation min lowered from 100 to 60 — pad has white circle printed on it
    # which desaturates the centre; 60 catches the whole pad including faded areas
    lower = np.array([18,  60, 100])
    upper = np.array([38, 255, 255])
    mask  = cv2.inRange(hsv, lower, upper)

    # close small holes so the white-circle area inside the pad doesn't break detection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask, cv2.countNonZero(mask) > 1500


# ── debug overlay ─────────────────────────────────────────────────────────────

def draw_debug(frame, skeleton_path, drone_pos, lookahead_result, landing_detected):
    """draw all vision elements onto a copy of the frame for monitoring."""
    out = frame.copy()

    if skeleton_path:
        for pt in skeleton_path:
            cv2.circle(out, pt, 1, (0, 200, 100), -1)

    if drone_pos:
        cv2.circle(out, drone_pos, 10, (255, 100, 0), 2)
        cv2.circle(out, drone_pos, 3,  (255, 100, 0), -1)

    if lookahead_result:
        dx, dy, lookahead_pt, closest_pt = lookahead_result
        if drone_pos:
            cv2.line(out, drone_pos, closest_pt, (150, 150, 150), 1)
            cv2.arrowedLine(out, drone_pos, lookahead_pt, (0, 120, 255), 2, tipLength=0.3)
        cv2.circle(out, lookahead_pt, 6, (0, 120, 255), -1)
        cv2.putText(out, f"dx:{dx:+.2f}  dy:{dy:+.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)

    if landing_detected:
        cv2.putText(out, "landing pad visible", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 230), 2)

    return out