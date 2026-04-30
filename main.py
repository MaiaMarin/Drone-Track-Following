import cv2
from scanner import Scanner
from tracker import Tracker
from vision import (
    detect_track_mask,
    skeletonize_mask,
    get_skeleton_points,
    get_follow_direction_from_position,
    draw_debug,
)
from drone_controller import DroneController

FLY = True
DISPLAY = True

MAX_JUMP = 30
LOOKAHEAD_DISTANCE = 120
MAX_SEGMENT_GAP = 35

END_MARGIN_POINTS = 80
END_CONFIRM_FRAMES = 10

LOST_CONFIRM_FRAMES = 60
WARMUP_FRAMES = 30

def main():
    scanner = Scanner()
    tracker = Tracker(
        floor_depth_mm=1800,
        min_height_above_floor_mm=150,
        min_depth_mm=200,
        min_area=250,
        max_area=30000
    )
    controller = DroneController()

    end_counter = 0
    lost_counter = 0

    try:
        scanner.start()

        for _ in range(WARMUP_FRAMES):
            scanner.get_frames()

        controller.connect()
        controller.takeoff()

        while True:
            color_image, depth_image = scanner.get_frames()

            if color_image is None or depth_image is None:
                continue

            drone_position, drone_mask = tracker.get_drone_position(depth_image)

            track_mask = detect_track_mask(color_image)
            skeleton = skeletonize_mask(track_mask)
            points = get_skeleton_points(skeleton, max_jump=MAX_JUMP)

            follow_direction = get_follow_direction_from_position(
                drone_position,
                points,
                lookahead_distance=LOOKAHEAD_DISTANCE,
                max_segment_gap=MAX_SEGMENT_GAP
            )

            if follow_direction is not None:
                lost_counter = 0

                nearest_index = follow_direction["nearest_index"]

                if len(points) - nearest_index <= END_MARGIN_POINTS:
                    end_counter += 1
                else:
                    end_counter = 0

                controller.send_direction(follow_direction)
            else:
                lost_counter += 1
                end_counter = 0
                controller.hover()

            if DISPLAY:
                debug = draw_debug(color_image, track_mask, skeleton, drone_position, follow_direction)
                cv2.imshow("debug", debug)
                cv2.imshow("track mask", track_mask)
                cv2.imshow("drone mask", drone_mask)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if end_counter >= END_CONFIRM_FRAMES:
                break

            if lost_counter >= LOST_CONFIRM_FRAMES:
                break

    except KeyboardInterrupt:
        pass

    finally:
        controller.close()
        scanner.stop()

        if DISPLAY:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()