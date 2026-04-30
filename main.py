import cv2
from vision import (
    detect_track_mask,
    skeletonize_mask,
    get_skeleton_points,
    get_path_directions,
    get_follow_direction_from_position,
)
from drone_controller import direction_to_command, DroneController

IMAGE_PATH = "track.jpeg"
DRONE_POSITION = (430, 560)

img = cv2.imread(IMAGE_PATH)

if img is None:
    exit(1)

mask = detect_track_mask(img)
skeleton = skeletonize_mask(mask)
points = get_skeleton_points(skeleton, max_jump=30)

follow_direction = get_follow_direction_from_position(
    DRONE_POSITION,
    points,
    lookahead_distance=120
)

if follow_direction is None:
    exit(1)

nearest_index = follow_direction["nearest_index"]
path_from_drone = points[nearest_index:]

directions = get_path_directions(
    path_from_drone,
    step=80,
    lookahead_distance=100
)

commands = [direction_to_command(direction) for direction in directions]

if len(commands) == 0:
    exit(1)

controller = DroneController()

try:
    controller.connect()
    controller.takeoff()
    controller.follow_commands(commands)
    controller.land()
finally:
    controller.close()