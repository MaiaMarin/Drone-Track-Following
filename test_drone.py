from pathlib import Path
import math
import time
import cv2
import numpy as np

from drone_controller import DroneController
from track_detection import segment_scene, draw_points_and_path

USE_LIVE_CAMERA = False
EXECUTE_DRONE = True

RGB_PATH = "frame_1779104513_rgb.png"
DEPTH_PATH = "frame_1779104513_depth.npy"

FORWARD_PITCH = 35
YAW_VALUE = 70
ROLL_VALUE = 25
THROTTLE_VALUE = 0

TAKEOFF_WAIT = 1.8
DESCEND_AFTER_TAKEOFF = True
DESCEND_THROTTLE = -30
DESCEND_TIME = 0.7

PATH_STEP_PX = 95
PATH_SMOOTH_WINDOW = 5

FORWARD_PX_PER_SECOND = 170.0
MIN_FORWARD_TIME = 0.18
MAX_FORWARD_TIME = 0.75

YAW_SECONDS_PER_DEGREE = 0.85 / 90.0
MIN_YAW_DEG = 8
MAX_YAW_TIME = 0.75
YAW_SIGN = 1

OFFSET_DEADZONE_PX = 35
OFFSET_SECONDS_PER_PX = 0.004
MAX_OFFSET_CORRECTION_TIME = 0.35
ROLL_SIGN = 1

PAUSE_BETWEEN_COMMANDS = 0.08

def load_scene_from_files(rgb_path, depth_path):
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

    if bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {rgb_path}")

    depth = None

    if depth_path is not None and Path(depth_path).exists():
        depth = np.load(str(depth_path))

    return bgr, depth

def angle_deg(p1, p2):
    dx = p2[0] - p1[0]
    dy = -(p2[1] - p1[1])
    return math.degrees(math.atan2(dy, dx))

def angle_diff_deg(a, b):
    return (b - a + 180) % 360 - 180

def distance(p1, p2):
    return float(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))

def smooth_path(points, window=5):
    if len(points) < window:
        return points

    arr = np.array(points, dtype=np.float32)
    out = []

    half = window // 2

    for i in range(len(arr)):
        start = max(0, i - half)
        end = min(len(arr), i + half + 1)
        out.append(np.mean(arr[start:end], axis=0))

    return [[float(p[0]), float(p[1])] for p in out]

def resample_path(points, step_px=90):
    if len(points) < 2:
        return points

    result = [points[0]]
    accumulated = 0.0
    previous = np.array(points[0], dtype=np.float32)

    for i in range(1, len(points)):
        current = np.array(points[i], dtype=np.float32)
        segment = current - previous
        segment_length = float(np.linalg.norm(segment))

        if segment_length < 1e-6:
            continue

        direction = segment / segment_length
        remaining = segment_length

        while accumulated + remaining >= step_px:
            needed = step_px - accumulated
            new_point = previous + direction * needed
            result.append([float(new_point[0]), float(new_point[1])])
            previous = new_point
            remaining = float(np.linalg.norm(current - previous))
            accumulated = 0.0

        accumulated += remaining
        previous = current

    if distance(result[-1], points[-1]) > 20:
        result.append(points[-1])

    return result

def get_path_from_scene(bgr, depth):
    seg, info = segment_scene(bgr, depth, return_debug=False)

    path = info.get("path") or []

    if len(path) < 2:
        raise RuntimeError("No usable path was detected.")

    path = list(reversed(path))
    path = smooth_path(path, PATH_SMOOTH_WINDOW)
    path = resample_path(path, PATH_STEP_PX)

    debug = draw_points_and_path(bgr, {
        "path": path,
        "start_pad_center": info.get("start_pad_center"),
        "landing_pad_center": info.get("landing_pad_center"),
        "drone_center": info.get("drone_center"),
    })

    cv2.imwrite("test_drone_planned_path.png", debug)

    return path, info

def build_commands_from_path(path):
    if len(path) < 2:
        return []

    commands = []
    current_angle = angle_deg(path[0], path[1])

    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]

        desired_angle = angle_deg(p1, p2)
        yaw_delta = angle_diff_deg(current_angle, desired_angle)

        if abs(yaw_delta) >= MIN_YAW_DEG:
            yaw_time = min(abs(yaw_delta) * YAW_SECONDS_PER_DEGREE, MAX_YAW_TIME)
            yaw_value = int(np.sign(yaw_delta) * YAW_SIGN * YAW_VALUE)

            commands.append({
                "type": "yaw",
                "value": yaw_value,
                "duration": yaw_time,
                "angle_delta": yaw_delta,
            })

            current_angle = desired_angle

        segment_distance = distance(p1, p2)
        forward_time = segment_distance / FORWARD_PX_PER_SECOND
        forward_time = float(np.clip(forward_time, MIN_FORWARD_TIME, MAX_FORWARD_TIME))

        commands.append({
            "type": "forward",
            "value": FORWARD_PITCH,
            "duration": forward_time,
            "distance_px": segment_distance,
        })

    commands.append({
        "type": "land",
        "duration": 0.0,
    })

    return commands

def stop_command():
    return {"roll": 0, "pitch": 0, "yaw": 0, "throttle": 0}

def send_for(controller, command, duration):
    controller.send_command(command)
    time.sleep(duration)
    controller.send_command(stop_command())
    time.sleep(PAUSE_BETWEEN_COMMANDS)

def command_to_drone_command(command):
    if command["type"] == "forward":
        return {"roll": 0, "pitch": int(command["value"]), "yaw": 0, "throttle": THROTTLE_VALUE}

    if command["type"] == "yaw":
        return {"roll": 0, "pitch": 0, "yaw": int(command["value"]), "throttle": 0}

    if command["type"] == "roll":
        return {"roll": int(command["value"]), "pitch": 0, "yaw": 0, "throttle": 0}

    return stop_command()

def closest_path_error(point, path):
    if point is None or len(path) < 2:
        return None

    px, py = point
    best = None

    for i in range(len(path) - 1):
        ax, ay = path[i]
        bx, by = path[i + 1]

        vx = bx - ax
        vy = by - ay
        wx = px - ax
        wy = py - ay

        denom = vx * vx + vy * vy

        if denom < 1e-6:
            continue

        t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
        cx = ax + t * vx
        cy = ay + t * vy

        dx = px - cx
        dy = py - cy

        dist = math.hypot(dx, dy)
        signed = (vx * (py - ay) - vy * (px - ax)) / math.sqrt(denom)

        if best is None or dist < best["distance"]:
            best = {
                "distance": dist,
                "signed_error": signed,
                "segment_index": i,
                "closest": [cx, cy],
            }

    return best

def try_offset_correction(controller, camera, path):
    if camera is None:
        return

    success, bgr, depth = camera.get_frames()

    if not success:
        return

    try:
        _, info = segment_scene(bgr, depth, return_debug=False)
    except Exception:
        return

    drone_center = info.get("drone_center")

    if drone_center is None:
        return

    error = closest_path_error(drone_center, path)

    if error is None:
        return

    signed_error = error["signed_error"]

    if abs(signed_error) < OFFSET_DEADZONE_PX:
        return

    duration = min(abs(signed_error) * OFFSET_SECONDS_PER_PX, MAX_OFFSET_CORRECTION_TIME)
    roll_value = int(np.sign(signed_error) * ROLL_SIGN * ROLL_VALUE)

    send_for(controller, {"roll": roll_value, "pitch": 0, "yaw": 0, "throttle": 0}, duration)

def execute_commands(commands, path, camera=None):
    controller = DroneController()

    try:
        print("connecting...")
        controller.connect()

        print("taking off...")
        controller.takeoff()
        time.sleep(TAKEOFF_WAIT)

        if DESCEND_AFTER_TAKEOFF:
            print("small descent...")
            send_for(controller, {"roll": 0, "pitch": 0, "yaw": 0, "throttle": DESCEND_THROTTLE}, DESCEND_TIME)

        for index, command in enumerate(commands):
            print(index, command)

            if command["type"] == "land":
                print("landing...")
                controller.land()
                return

            if command["type"] == "forward":
                try_offset_correction(controller, camera, path)

            drone_command = command_to_drone_command(command)
            send_for(controller, drone_command, command["duration"])

        print("landing...")
        controller.land()

    except BaseException as e:
        print("something went wrong:", e)
        try:
            controller.land()
        except BaseException:
            pass

    finally:
        try:
            controller.close()
        except BaseException:
            pass

def print_commands(commands):
    total_forward = sum(c["duration"] for c in commands if c["type"] == "forward")
    total_yaw = sum(c["duration"] for c in commands if c["type"] == "yaw")

    print("Generated commands:")
    for i, command in enumerate(commands):
        print(i, command)

    print("Total forward time:", round(total_forward, 2))
    print("Total yaw time:", round(total_yaw, 2))

def main():
    camera = None

    if USE_LIVE_CAMERA:
        from rgbd_camera import RGBDCamera
        camera = RGBDCamera()
        camera.start()
        time.sleep(1.0)
        success, bgr, depth = camera.get_frames()

        if not success:
            raise RuntimeError("Could not capture live RGB-D frame.")
    else:
        bgr, depth = load_scene_from_files(RGB_PATH, DEPTH_PATH)

    path, info = get_path_from_scene(bgr, depth)
    commands = build_commands_from_path(path)

    print("Vision info:")
    print("start:", info.get("start_pad_center"))
    print("landing:", info.get("landing_pad_center"))
    print("drone:", info.get("drone_center"))
    print("path points:", len(path))

    print_commands(commands)

    if EXECUTE_DRONE:
        execute_commands(commands, path, camera=camera)
    else:
        print("EXECUTE_DRONE is False, so no drone commands were sent.")
        print("Check test_drone_planned_path.png first. Then set EXECUTE_DRONE = True.")

    if camera is not None:
        camera.stop()

if __name__ == "__main__":
    main()