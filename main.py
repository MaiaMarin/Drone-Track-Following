"""
main.py
-------
main flight loop. uses only the realsense colour stream —
depth is not needed because the drone reads its own altitude via the
bottom range sensor inside movement_controller.update().
"""

import pyrealsense2 as rs
import numpy as np
import cv2

from vision import (
    detect_track_mask,
    get_track_skeleton,
    detect_drone_position,
    get_lookahead_direction,
    draw_debug,
    detect_yellow_landing,
)
from movement_controller import TrackController


# ── realsense — colour only ───────────────────────────────────────────────────

pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# depth stream is not enabled here; altitude comes from drone.get_bottom_range()


# ── main ──────────────────────────────────────────────────────────────────────

controller = TrackController()

try:
    pipeline.start(config)
    controller.connect()

    while True:
        frames      = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("emergency land.")
            controller._land()
            break
        # ── vision ────────────────────────────────────────────────────────────
        track_mask       = detect_track_mask(img)
        skeleton         = get_track_skeleton(track_mask)
        drone_pos        = detect_drone_position(img)
        direction_result = get_lookahead_direction(drone_pos, skeleton)
        _, landing       = detect_yellow_landing(img)

        # ── movement (sensors read inside update()) ───────────────────────────
        controller.update(direction_result, landing, drone_pos)

        # ── debug display ─────────────────────────────────────────────────────
        debug = draw_debug(img, skeleton, drone_pos, direction_result, landing)
        cv2.imshow("track follower", debug)
        cv2.imshow("track mask",     track_mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("manual quit.")
            break

finally:
    controller.disconnect()
    pipeline.stop()
    cv2.destroyAllWindows()