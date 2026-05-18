# track_sequence_runner.py
# -------------------------
# drives the drone through a user-defined sequence of track pieces.
#
# usage:
#   edit TRACK_SEQUENCE at the bottom of this file, then run:
#       python track_sequence_runner.py
#
# track types available:
#   SHORT_STRAIGHT   - short wooden straight segment
#   MEDIUM_STRAIGHT  - medium wooden straight segment  (confirmed working)
#   LONG_STRAIGHT    - long wooden straight segment
#   LEFT_CURVE       - quarter-circle left turn        (confirmed working)
#   RIGHT_CURVE      - quarter-circle right turn
#
# all timings are derived from the confirmed-working manual_test.py values.
# if a segment type drifts, only adjust its duration constant below.

from drone_controller import DroneController
import time
import threading
import sys

# ── platform-specific non-blocking keyread ────────────────────────────────────
# used by the emergency-stop watcher thread.
# on windows we use msvcrt; on linux/mac we use termios + tty.

if sys.platform == "win32":
    import msvcrt

    def _read_key():
        """return the next keypress as a lowercase str, or None if nothing pressed."""
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            try:
                return ch.decode("utf-8").lower()
            except UnicodeDecodeError:
                return None
        return None

else:
    import tty
    import termios

    def _read_key():
        """return the next keypress without blocking (unix)."""
        import select
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch = sys.stdin.read(1)
                return ch.lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return None


# ── track type labels ─────────────────────────────────────────────────────────

SHORT_STRAIGHT  = "short_straight"
MEDIUM_STRAIGHT = "medium_straight"
LONG_STRAIGHT   = "long_straight"
LEFT_CURVE      = "left_curve"
RIGHT_CURVE     = "right_curve"


# ── movement parameters (tuned from manual_test.py) ──────────────────────────

# ── movement parameters ───────────────────────────────────────────────────────
#
# all values below are derived from the movement_recorder.py session.
# do not edit the guessed constants here — edit from a fresh recording instead.

DESCEND_THROTTLE = -60
FORWARD_PITCH    = 20
TURN_YAW         = 90     # kept for any legacy reference; curves use CURVE_YAW_RATE

# ── straight durations (seconds of continuous forward pitch) ──────────────────
#
# anchor: post-curve MEDIUM_STRAIGHT measured directly from recording.
#   presses: 0.108 + 0.144 + 0.146 + 0.144 + 0.106 = 0.648 s
#
# SHORT and LONG are scaled from that anchor using the original physical ratios
# (0.25 : 0.35 : 0.60).  sanity check: 0.463 + 0.648 + 1.111 = 2.222 s, which
# matches the 2.194 s total of the 13 recorded straight presses to within 1 %.

STRAIGHT_DURATION = {
    SHORT_STRAIGHT:  0.46,   # 0.648 * (0.25 / 0.35)  =  0.463 s
    MEDIUM_STRAIGHT: 0.65,   # measured directly from recording
    LONG_STRAIGHT:   1.11,   # 0.648 * (0.60 / 0.35)  =  1.111 s
}

# ── curve shape ───────────────────────────────────────────────────────────────
#
# each RIGHT_CURVE / LEFT_CURVE command does:
#   yaw half 1  →  forward bridge  →  yaw half 2  →  forward bridge  →  settle
#
# recorded yaw totals:
#   curve 1: 0.191 + 0.173 + 0.137 + 0.132 = 0.633 s  (4 taps)
#   curve 2: 0.106 + 0.137 + 0.148 + 0.147 + 0.131 = 0.669 s  (5 taps)
#   average: 0.651 s  →  each half = 0.326 s, rounded to 0.33 s
#
# recorded bridge forwards after curve 1:  0.142 s, 0.141 s  →  avg 0.142 s

CURVE_YAW_RATE        = 60     # yaw power (lower = gentler arc)
CURVE_HALF_DURATION   = 0.09   # seconds per yaw half
CURVE_BRIDGE_PITCH    = 15     # forward pitch during bridge moves
CURVE_BRIDGE_DURATION = 0.14   # seconds of forward motion between and after yaws

# the second yaw half of each curve tends to overshoot because rotational
# momentum carries over from the first half.  this scale factor shortens
# only the final yaw segment without touching anything else.
# 1.0 = same as half 1.  try 0.6-0.8 first, then nudge up/down by 0.05.
CURVE_FINAL_HALF_SCALE = 0.0009

# short pause between any two segments so commands don't blend
INTER_SEGMENT_PAUSE  = 0.08

# how long to hold a zero command after each movement so the drone
# stops and lets the optical flow sensor recenter it before the next move.
# raise this if the drone is still carrying momentum into the next segment.
SETTLE_PAUSE = 0.5

# extra settle after a curve yaw — the optical flow sensor needs longer
# to re-lock onto the floor after the heading changes.
# if the drone still drifts backward after turning, raise this first.
CURVE_SETTLE_PAUSE = 0.8


# ── sequence runner ───────────────────────────────────────────────────────────

class TrackSequenceRunner:
    """
    executes a list of track-piece instructions on a codrone edu.
    each entry in the sequence is one of the five track-type constants.
    """

    def __init__(self, sequence):
        """
        args:
            sequence : list of track-type strings, e.g.
                       [MEDIUM_STRAIGHT, LEFT_CURVE, MEDIUM_STRAIGHT]
        """
        self.sequence        = sequence
        self.controller      = DroneController()
        self._stop_flag      = threading.Event()   # set when emergency land fires
        self._watcher_thread = None

    # ── landing ───────────────────────────────────────────────────────────────

    def _land(self):
        """
        zero all axes then land using the raw drone object directly.
        falls back to controller.land() if the raw object is not accessible.
        always waits 3 s after the land command so the drone has time to touch down.
        """
        raw = self._get_raw_drone()
        try:
            self.controller.send_command(
                {"roll": 0, "pitch": 0, "yaw": 0, "throttle": 0}
            )
        except Exception:
            pass
        time.sleep(0.15)

        if raw is not None:
            try:
                raw.land()
                print("  land command sent via raw drone.")
                time.sleep(3)
                return
            except Exception as e:
                print(f"  raw drone land() failed: {e} — trying controller fallback.")

        try:
            self.controller.land()
            print("  land command sent via controller.")
            time.sleep(3)
        except Exception as e:
            print(f"  controller.land() also failed: {e}")
            print("  drone may still be airborne — use the physical controller to land.")

    # ── emergency stop ────────────────────────────────────────────────────────

    def _emergency_land(self):
        """
        cut all movement immediately and land.
        called from the watcher thread — must be thread-safe.
        sets _stop_flag so the main loop exits cleanly.
        """
        print("\n*** EMERGENCY LAND — space pressed ***")
        self._stop_flag.set()
        self._land()

    def _start_watcher(self):
        """
        start a background thread that polls for the space bar.
        polling interval is 50 ms — fast enough to react within one frame.
        the thread exits only after _stop_flag is set AND the land sequence
        has finished, so space remains active even during the final landing.
        """
        def _watch():
            print()
            print("  ╔══════════════════════════════════════╗")
            print("  ║  SPACE = emergency land at any time  ║")
            print("  ╚══════════════════════════════════════╝")
            print()
            while not self._stop_flag.is_set():
                key = _read_key()
                if key == " ":
                    self._emergency_land()
                    return
                time.sleep(0.05)

        self._watcher_thread = threading.Thread(target=_watch, daemon=True)
        self._watcher_thread.start()

    def _stop_watcher(self):
        """signal the watcher thread to exit and wait for it."""
        self._stop_flag.set()
        if self._watcher_thread is not None:
            self._watcher_thread.join(timeout=1.0)

    # ── trim helpers ──────────────────────────────────────────────────────────

    def _get_raw_drone(self):
        """
        returns the underlying codrone_edu Drone object if accessible,
        or None if the controller wraps it under a different attribute name.
        adjust the attribute name here if needed.
        """
        for attr in ("drone", "_drone", "codrone", "_codrone"):
            if hasattr(self.controller, attr):
                return getattr(self.controller, attr)
        return None

    def _print_trim(self):
        """read and print the trim currently stored on the drone."""
        raw = self._get_raw_drone()
        if raw is None:
            print("  [trim] could not access raw drone object — check _get_raw_drone()")
            return None
        try:
            t = raw.get_trim()
            print(f"  [trim]  roll: {t[0]:+d}   pitch: {t[1]:+d}")
            return t
        except Exception as e:
            print(f"  [trim] get_trim() failed: {e}")
            return None

    def _apply_trim(self, roll, pitch):
        """clamp and apply a new trim value to the drone."""
        raw = self._get_raw_drone()
        if raw is None:
            print("  [trim] could not apply trim — raw drone not accessible")
            return
        roll  = max(-100, min(100, roll))
        pitch = max(-100, min(100, pitch))
        raw.set_trim(roll, pitch)
        time.sleep(0.3)
        print(f"  [trim] applied  ->  roll: {roll:+d}   pitch: {pitch:+d}")

    # ── pre-flight interactive trim loop ──────────────────────────────────────

    def _trim_loop(self):
        """
        shown before the sequence starts (while the drone is hovering at altitude).
        lets you nudge roll/pitch trim and re-hover until it looks stable,
        then press enter to proceed.

        controls
          w  pitch +5 (corrects backward drift)
          s  pitch -5 (corrects forward drift)
          d  roll  +5 (corrects left drift)
          a  roll  -5 (corrects right drift)
          t  run a 3-second hover test (drone already airborne)
          p  print current trim
          enter / blank  proceed with sequence
        """
        raw = self._get_raw_drone()
        if raw is None:
            print("[pre-flight trim] skipping — raw drone not accessible.")
            return

        step = 5
        t    = self._print_trim()
        if t is None:
            return
        roll, pitch = t[0], t[1]

        print()
        print("─" * 52)
        print("  pre-flight trim check  (drone is hovering)")
        print("  w/s = pitch +/-   a/d = roll +/-")
        print("  t   = hover test (3 s)   p = show trim")
        print("  press enter with no input to start the sequence")
        print("─" * 52)

        while True:
            try:
                key = input("  trim> ").strip().lower()
            except EOFError:
                break

            if key == "":
                # blank enter — proceed
                break
            elif key == "w":
                pitch += step
                self._apply_trim(roll, pitch)
                pitch = max(-100, min(100, pitch))
            elif key == "s":
                pitch -= step
                self._apply_trim(roll, pitch)
                pitch = max(-100, min(100, pitch))
            elif key == "d":
                roll += step
                self._apply_trim(roll, pitch)
                roll = max(-100, min(100, roll))
            elif key == "a":
                roll -= step
                self._apply_trim(roll, pitch)
                roll = max(-100, min(100, roll))
            elif key == "t":
                print("  hovering 3 s — watch for drift...")
                self._send(roll=0, pitch=0, yaw=0, throttle=0)
                time.sleep(3)
                print("  hover test done.")
            elif key == "p":
                t = self._print_trim()
                if t:
                    roll, pitch = t[0], t[1]
            else:
                print("  unknown key. use w/s/a/d/t/p or enter to continue.")

        print("  trim confirmed — starting sequence.")
        print()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def run(self):
        """connect, take off, fly the full sequence, then land."""
        try:
            print("connecting...")
            self.controller.connect()

            # show what trim the drone has stored right now
            print("trim at startup:")
            self._print_trim()
            print()

            print("taking off...")
            self.controller.takeoff()
            time.sleep(1.5)

            # descend to track-following altitude before starting
            print("descending to track altitude...")
            self._send(roll=0, pitch=0, yaw=0, throttle=DESCEND_THROTTLE)
            time.sleep(1.9)

            # optional pre-flight trim check while the drone is hovering
            self._trim_loop()

            # start the emergency-stop watcher — space bar lands immediately
            self._start_watcher()

            # fly each segment in order
            for i, segment in enumerate(self.sequence):
                # check whether space was pressed between segments
                if self._stop_flag.is_set():
                    print("stop flag set — aborting sequence.")
                    break
                print(f"segment {i + 1}/{len(self.sequence)}: {segment}")
                self._fly_segment(segment)
                time.sleep(INTER_SEGMENT_PAUSE)

            if not self._stop_flag.is_set():
                print("sequence complete — landing.")
                self._land()          # land while watcher is still active
                self._stop_watcher()  # only kill the watcher once the drone is down
            else:
                # _emergency_land() already landed; just clean up the thread
                self._stop_watcher()

        except BaseException as e:
            print("error during sequence:", e)
            self._stop_flag.set()
            self._land()

        finally:
            self._stop_flag.set()
            try:
                self.controller.close()
            except BaseException:
                pass

    # ── segment dispatcher ────────────────────────────────────────────────────

    def _fly_segment(self, segment):
        """dispatch a single track segment to the correct handler."""
        if segment in (SHORT_STRAIGHT, MEDIUM_STRAIGHT, LONG_STRAIGHT):
            self._fly_straight(segment)
        elif segment == LEFT_CURVE:
            self._fly_curve(direction="left")
        elif segment == RIGHT_CURVE:
            self._fly_curve(direction="right")
        else:
            print(f"  unknown segment type '{segment}' — skipping.")

    # ── straight handler ──────────────────────────────────────────────────────

    def _fly_straight(self, segment):
        """
        pitch forward for the duration that matches the physical track length,
        then hold a zero command for SETTLE_PAUSE seconds so the optical flow
        sensor can recenter the drone before the next segment fires.
        bails immediately if the emergency stop flag is set.
        """
        if self._stop_flag.is_set():
            return
        duration = STRAIGHT_DURATION[segment]
        print(f"  pitching forward for {duration:.2f}s")
        self._send(roll=0, pitch=FORWARD_PITCH, yaw=0, throttle=0)
        time.sleep(duration)

        if self._stop_flag.is_set():
            return

        # stop and let the drone recenter
        self._send(roll=0, pitch=0, yaw=0, throttle=0)
        print(f"  settling for {SETTLE_PAUSE:.2f}s...")
        time.sleep(SETTLE_PAUSE)

    # ── curve handler ─────────────────────────────────────────────────────────

    def _fly_curve(self, direction):
        """
        executes a quarter-circle curve as two gentle half-turns with a short
        forward bridge between them, then a final bridge to enter the next segment.

        shape:  [yaw half] -> [forward bridge] -> [yaw half] -> [forward bridge]

        tune CURVE_YAW_RATE, CURVE_HALF_DURATION, CURVE_BRIDGE_DURATION at
        the top of this file, or record your own values with movement_recorder.py.

        args:
            direction : "left" or "right"
        """
        if self._stop_flag.is_set():
            return

        yaw_value = CURVE_YAW_RATE if direction == "left" else -CURVE_YAW_RATE

        # ── first half-turn ───────────────────────────────────────────────────
        print(f"  curve {direction} — half 1  (yaw={yaw_value}, {CURVE_HALF_DURATION:.2f}s)")
        self._send(roll=0, pitch=0, yaw=yaw_value, throttle=0)
        time.sleep(CURVE_HALF_DURATION)
        if self._stop_flag.is_set():
            return

        # ── forward bridge between the two yaws ───────────────────────────────
        print(f"  curve bridge  (pitch={CURVE_BRIDGE_PITCH}, {CURVE_BRIDGE_DURATION:.3f}s)")
        self._send(roll=0, pitch=CURVE_BRIDGE_PITCH, yaw=0, throttle=0)
        time.sleep(CURVE_BRIDGE_DURATION)
        if self._stop_flag.is_set():
            return

        # ── second half-turn (scaled down to avoid overshoot) ────────────────
        final_duration = CURVE_HALF_DURATION * CURVE_FINAL_HALF_SCALE
        print(f"  curve {direction} — half 2  (yaw={yaw_value}, {final_duration:.3f}s, scale={CURVE_FINAL_HALF_SCALE})")
        self._send(roll=0, pitch=0, yaw=yaw_value, throttle=0)
        time.sleep(final_duration)
        if self._stop_flag.is_set():
            return

        # ── final forward bridge to enter the next segment ────────────────────
        print(f"  curve exit bridge  ({CURVE_BRIDGE_DURATION:.3f}s)")
        self._send(roll=0, pitch=CURVE_BRIDGE_PITCH, yaw=0, throttle=0)
        time.sleep(CURVE_BRIDGE_DURATION)
        if self._stop_flag.is_set():
            return

        # stop and settle — longer than normal so optical flow re-locks
        # onto the floor in the new heading before any pitch command fires.
        self._send(roll=0, pitch=0, yaw=0, throttle=0)
        print(f"  post-curve settle for {CURVE_SETTLE_PAUSE:.2f}s...")
        time.sleep(CURVE_SETTLE_PAUSE)

    # ── command helper ────────────────────────────────────────────────────────

    def _send(self, roll, pitch, yaw, throttle):
        """thin wrapper around controller.send_command for readability."""
        self.controller.send_command({
            "roll":     roll,
            "pitch":    pitch,
            "yaw":      yaw,
            "throttle": throttle,
        })


TRACK_SEQUENCE = [
    SHORT_STRAIGHT,
    MEDIUM_STRAIGHT,
    LONG_STRAIGHT,
    RIGHT_CURVE,
    RIGHT_CURVE,
    MEDIUM_STRAIGHT,
]


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("track sequence to fly:")
    for i, seg in enumerate(TRACK_SEQUENCE):
        print(f"  {i + 1}. {seg}")
    print()

    runner = TrackSequenceRunner(TRACK_SEQUENCE)
    runner.run()