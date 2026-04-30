import cv2
from vision import (
    detect_track_mask,
    skeletonize_mask,
    get_skeleton_points,
    get_path_directions,
    get_follow_direction_from_position,
    draw_skeleton,
    draw_path_directions,
    draw_follow_direction,
)
from drone_controller import direction_to_command, DroneController

IMAGE_PATH = "track.jpeg"
DRONE_POSITION = (430, 560)

img = cv2.imread(IMAGE_PATH)

if img is None:
    print("Could not load image.")
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
    print("No follow direction found.")
    directions = []
    commands = []
    nearest_index = None
    path_from_drone = []
else:
    nearest_index = follow_direction["nearest_index"]
    path_from_drone = points[nearest_index:]

    directions = get_path_directions(
        path_from_drone,
        step=80,
        lookahead_distance=100
    )

    commands = [direction_to_command(direction) for direction in directions]

if len(commands) > 0:
    controller = DroneController()

    try:
        controller.connect()
        controller.takeoff()
        controller.follow_commands(commands)
        controller.land()
    except BaseException as e:
        print("Drone flight failed:", e)
    finally:
        try:
            controller.close()
        except BaseException:
            pass
else:
    print("No commands generated, skipping flight.")

print("skeleton points:", len(points))
print("nearest index:", nearest_index)
print("path points from drone:", len(path_from_drone))
print("directions:", len(directions))
print("commands:")
for i, command in enumerate(commands, start=1):
    print(i, command)

debug = draw_skeleton(img, skeleton)
debug = draw_path_directions(debug, directions)

if follow_direction is not None:
    debug = draw_follow_direction(debug, DRONE_POSITION, follow_direction)

cv2.imshow("original", img)
cv2.imshow("track mask", mask)
cv2.imshow("skeleton", skeleton)
cv2.imshow("debug follow", debug)

cv2.waitKey(0)
cv2.destroyAllWindows()