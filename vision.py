import cv2
import numpy as np

def detect_color_mask(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array(lower)
    upper = np.array(upper)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask

def detect_track_mask(img, exclude_mask=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([5, 25, 80])
    upper = np.array([28, 180, 230])

    mask = cv2.inRange(hsv, lower, upper)

    if exclude_mask is not None:
        exclude_mask = cv2.dilate(exclude_mask, np.ones((15, 15), np.uint8), iterations=1)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask

def get_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    return max(contours, key=cv2.contourArea)

def keep_largest_component(mask):
    contour = get_largest_contour(mask)

    if contour is None:
        return mask

    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [contour], -1, 255, thickness=cv2.FILLED)

    return clean

def skeletonize_mask(mask):
    mask = keep_largest_component(mask)
    skeleton = np.zeros_like(mask)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    current = mask.copy()

    while True:
        eroded = cv2.erode(current, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(current, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        current = eroded.copy()

        if cv2.countNonZero(current) == 0:
            break

    return skeleton

def get_skeleton_points(skeleton, max_jump=30):
    ys, xs = np.where(skeleton > 0)

    if len(xs) == 0:
        return []

    points = list(zip(xs.tolist(), ys.tolist()))

    start_index = min(range(len(points)), key=lambda i: points[i][0])
    ordered = [points.pop(start_index)]

    while points:
        last = ordered[-1]

        nearest_index = min(
            range(len(points)),
            key=lambda i: (points[i][0] - last[0]) ** 2 + (points[i][1] - last[1]) ** 2
        )

        nearest = points[nearest_index]
        distance = np.linalg.norm(np.array(nearest) - np.array(last))

        if distance > max_jump:
            break

        ordered.append(points.pop(nearest_index))

    return ordered

def draw_skeleton(img, skeleton):
    debug = img.copy()

    ys, xs = np.where(skeleton > 0)

    for x, y in zip(xs, ys):
        cv2.circle(debug, (x, y), 1, (0, 0, 255), -1)

    return debug

def get_lookahead_direction_from_index(points, start_index, lookahead_distance=100, max_segment_gap=35):
    if len(points) < 2 or start_index >= len(points) - 1:
        return None

    start = points[start_index]
    accumulated = 0

    for i in range(start_index + 1, len(points)):
        prev = np.array(points[i - 1])
        curr = np.array(points[i])
        segment_length = np.linalg.norm(curr - prev)

        if segment_length > max_segment_gap:
            return None

        accumulated += segment_length

        if accumulated >= lookahead_distance:
            target = points[i]
            break
    else:
        target = points[-1]

    direction = np.array(target, dtype=np.float32) - np.array(start, dtype=np.float32)
    norm = np.linalg.norm(direction)

    if norm == 0:
        return None

    direction = direction / norm

    return {
        "start": start,
        "target": target,
        "dx": float(direction[0]),
        "dy": float(direction[1])
    }

def get_path_directions(points, step=80, lookahead_distance=100):
    directions = []

    if len(points) < 2:
        return directions

    for start_index in range(0, len(points), step):
        direction = get_lookahead_direction_from_index(
            points,
            start_index,
            lookahead_distance=lookahead_distance
        )

        if direction is not None:
            directions.append(direction)

    return directions

def draw_path_directions(img, directions):
    debug = img.copy()

    for direction in directions:
        start = direction["start"]
        target = direction["target"]

        cv2.circle(debug, start, 5, (255, 0, 0), -1)
        cv2.circle(debug, target, 5, (0, 255, 255), -1)
        cv2.arrowedLine(debug, start, target, (0, 255, 0), 2, tipLength=0.3)

    return debug

def get_follow_direction_from_position(position, points, lookahead_distance=120, max_segment_gap=35):
    if len(points) < 2:
        return None

    position_array = np.array(position, dtype=np.float32)

    nearest_index = min(
        range(len(points)),
        key=lambda i: np.linalg.norm(np.array(points[i], dtype=np.float32) - position_array)
    )

    nearest = points[nearest_index]
    accumulated = 0

    for i in range(nearest_index + 1, len(points)):
        prev = np.array(points[i - 1])
        curr = np.array(points[i])
        segment_length = np.linalg.norm(curr - prev)

        if segment_length > max_segment_gap:
            return None

        accumulated += segment_length

        if accumulated >= lookahead_distance:
            target = points[i]
            break
    else:
        target = points[-1]

    direction = np.array(target, dtype=np.float32) - position_array
    norm = np.linalg.norm(direction)

    if norm == 0:
        return None

    direction = direction / norm

    return {
        "position": position,
        "nearest": nearest,
        "target": target,
        "dx": float(direction[0]),
        "dy": float(direction[1]),
        "nearest_index": nearest_index
    }

def draw_follow_direction(img, position, direction):
    debug = img.copy()

    cv2.circle(debug, position, 8, (255, 0, 255), -1)

    if direction is None:
        return debug

    nearest = direction["nearest"]
    target = direction["target"]

    cv2.circle(debug, nearest, 7, (255, 255, 0), -1)
    cv2.circle(debug, target, 9, (0, 255, 255), -1)

    cv2.line(debug, position, nearest, (255, 255, 0), 2)
    cv2.arrowedLine(debug, position, target, (0, 255, 255), 3, tipLength=0.3)

    return debug