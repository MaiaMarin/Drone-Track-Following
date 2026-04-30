"""
movement_controller.py
----------------------
translates direction vectors (from vision.py) into codrone_edu commands.

sensor usage from the manual:
  bottom range sensor  — measures altitude (cm) directly on the drone.
                         we use this instead of the realsense depth stream.
                         api: drone.get_bottom_range()

  front range sensor   — measures distance to obstacle ahead (cm).
                         we use this as a safety brake near the end wall
                         and to slow down approaching the yellow pad.
                         api: drone.get_front_range()

  optical flow sensor  — helps the drone hold its x/y position between
                         command cycles. it's always active; we don't read
                         it directly but it stabilises the hover.

axis map:
  because the camera is mounted to the side of the track, pixel directions
  don't map 1:1 to drone axes. run this file directly to calibrate:

      python movement_controller.py

  move the drone by hand and watch which dx/dy value changes in each direction.
  then update AXIS_MAP below.
"""

from codrone_edu.drone import Drone
import time


# ── axis mapping — fill this in after running calibration mode ────────────────
#
# camera +x  (right in image)  usually = drone roll  (right in world)
# camera +y  (down  in image)  usually = drone pitch (backward in world)
#
# sign +1 = same direction, -1 = invert.

AXIS_MAP = {
    "roll":  ("dx",  -1),
    "pitch": ("dy", 1),
}

# if the camera is nearly side-on and camera_dy doesn't encode forward progress,
# set this to a fixed crawl speed (e.g. 12) and pitch will ignore the vector.
CONSTANT_FORWARD_PITCH = None


# ── altitude (bottom range sensor) ───────────────────────────────────────────

TARGET_ALTITUDE_CM   = 10     # 30 cm as specified
ALTITUDE_DEADBAND_CM = 3      # tolerated error before throttle kicks in
ALTITUDE_KP          = 0.8    # proportional gain (cm error -> throttle units)
MAX_THROTTLE         = 30


# ── front range safety brake ──────────────────────────────────────────────────

FRONT_BRAKE_CM       = 40     # if obstacle closer than this, reduce forward pitch
FRONT_STOP_CM        = 15     # if closer than this, stop pitching entirely


# ── movement ──────────────────────────────────────────────────────────────────

ROLL_SCALE           = 15     # scale normalised dx to roll units
PITCH_SCALE          = 10
MAX_ROLL             = 20
MAX_PITCH            = 15

MOVE_DURATION        = 0.05   # seconds per cycle


# ── landing ───────────────────────────────────────────────────────────────────

LANDING_PAD_FRAMES   = 10     # consecutive frames before landing is triggered


# ── controller class ──────────────────────────────────────────────────────────

class TrackController:

    def __init__(self):
        self.drone = Drone()
        self._landing_counter = 0
        self._is_flying       = False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def connect(self):
        self.drone.pair()
        time.sleep(2)
        print("paired. taking off...")
        self.drone.takeoff()
        time.sleep(3)
        self._is_flying = True
        print("airborne. starting track-following loop.")

    def disconnect(self):
        if self._is_flying:
            self._land()
        self.drone.close()

    # ── main update ───────────────────────────────────────────────────────────

    def update(self, direction_result, landing_detected):
        """
        one frame of movement.

        args:
            direction_result : (dx, dy, lookahead_pt, closest_pt) from
                               vision.get_lookahead_direction(), or None.
            landing_detected : bool from vision.detect_yellow_landing().

        altitude and front obstacle distance are read directly from the
        drone's own sensors inside this method.
        """
            # emergency land if q is pressed
        import cv2
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("emergency land triggered.")
            self._land()
            return

        if not self._is_flying:
            return
    
        if not self._is_flying:
            return

        # ── read onboard sensors ──────────────────────────────────────────────
        altitude_cm  = self.drone.get_bottom_range()   # cm to floor
        front_cm     = self.drone.get_front_range()    # cm to nearest obstacle ahead

        # ── landing check ─────────────────────────────────────────────────────
        if landing_detected and drone_pos is not None and drone_pos[0] > 500:
            self._landing_counter += 1
            print(f"landing pad visible {self._landing_counter}/{LANDING_PAD_FRAMES}")
            if self._landing_counter >= LANDING_PAD_FRAMES:
                print("landing pad confirmed — landing.")
                self._land()
                return
        else:
            self._landing_counter = 0

        # ── direction vector ──────────────────────────────────────────────────
        dx = direction_result[0] if direction_result is not None else 0.0
        dy = direction_result[1] if direction_result is not None else 0.0

        # ── roll ──────────────────────────────────────────────────────────────
        roll_cam, roll_sign = AXIS_MAP["roll"]
        roll = int((dx if roll_cam == "dx" else dy) * roll_sign * ROLL_SCALE)
        roll = max(-MAX_ROLL, min(MAX_ROLL, roll))

        # ── pitch (with front range safety brake) ─────────────────────────────
        if CONSTANT_FORWARD_PITCH is not None:
            base_pitch = CONSTANT_FORWARD_PITCH
        else:
            pitch_cam, pitch_sign = AXIS_MAP["pitch"]
            base_pitch = int((dx if pitch_cam == "dx" else dy) * pitch_sign * PITCH_SCALE)
            base_pitch = max(-MAX_PITCH, min(MAX_PITCH, base_pitch))

        # scale pitch down as the drone approaches an obstacle
        if front_cm is not None and front_cm > 0:
            if front_cm <= FRONT_STOP_CM:
                pitch = 0
                print(f"front obstacle at {front_cm}cm — stopping forward motion.")
            elif front_cm <= FRONT_BRAKE_CM:
                # linear taper from full speed at FRONT_BRAKE_CM to 0 at FRONT_STOP_CM
                scale = (front_cm - FRONT_STOP_CM) / (FRONT_BRAKE_CM - FRONT_STOP_CM)
                pitch = int(base_pitch * scale)
                print(f"braking — front: {front_cm}cm, pitch scaled to {pitch}")
            else:
                pitch = base_pitch
        else:
            pitch = base_pitch

        # ── throttle from bottom range sensor ────────────────────────────────
        if altitude_cm is not None and altitude_cm > 0:
            err = TARGET_ALTITUDE_CM - altitude_cm
            if abs(err) > ALTITUDE_DEADBAND_CM:
                throttle = int(err * ALTITUDE_KP)
                throttle = max(-MAX_THROTTLE, min(MAX_THROTTLE, throttle))
            else:
                throttle = 0
        else:
            throttle = 0

        # ── send ──────────────────────────────────────────────────────────────
        self.drone.set_roll(roll)
        self.drone.set_pitch(pitch)
        self.drone.set_throttle(throttle)
        self.drone.set_yaw(0)
        self.drone.move(MOVE_DURATION)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _land(self):
        """halt forward motion, then land."""
        self.drone.set_roll(0)
        self.drone.set_pitch(0)
        self.drone.set_throttle(0)
        self.drone.set_yaw(0)
        self.drone.move(0.2)
        self.drone.land()
        time.sleep(3)
        self._is_flying = False
        print("landed.")


# ── run this file directly for axis-map calibration ──────────────────────────
#
# place the drone in front of the camera.
# move it slowly to the right by hand → watch which of dx/dy changes.
# move it forward along the track    → watch which of dx/dy changes.
# update AXIS_MAP at the top of this file accordingly.

if __name__ == "__main__":
    import numpy as np
    import cv2
    from vision import (
        detect_track_mask, get_track_skeleton,
        detect_drone_position, get_lookahead_direction,
        draw_debug, detect_yellow_landing,
    )

    IMAGE_PATH = "track.jpeg"

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"could not read {IMAGE_PATH}")
        exit(1)

    # resize to match what the camera would send
    max_width = 900
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)))

    print("static image mode — press q to quit, drone commands will print but not fly.")
    print("pair your drone first? (y/n): ", end="")
    do_pair = input().strip().lower() == "y"

    controller = TrackController()
    if do_pair:
        controller.connect()  # takeoff happens here — careful!
    else:
        controller.drone.pair()
        import time
        time.sleep(2)
        controller._is_flying = True
        print("paired without takeoff — commands will be sent but drone stays grounded.")

    try:
        while True:
            mask      = detect_track_mask(img)
            skeleton  = get_track_skeleton(mask)
            drone_pos = detect_drone_position(img)
            direction = get_lookahead_direction(drone_pos, skeleton)
            _, land   = detect_yellow_landing(img)

            if direction:
                print(f"dx: {direction[0]:+.3f}  dy: {direction[1]:+.3f}", end="  ")

            # print what would be sent instead of actually flying
            controller.update(direction, land)

            cv2.imshow("static calibration", draw_debug(img, skeleton, drone_pos, direction, land))
            cv2.imshow("track mask", mask)

            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        controller._is_flying = False  # skip auto-land on exit
        controller.drone.close()
        cv2.destroyAllWindows()