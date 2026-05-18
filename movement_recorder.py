# movement_recorder.py
# ---------------------
# fly the drone manually and record every key press with its exact duration.
# at the end it prints a timestamped log and a ready-to-paste constants block
# you can drop straight into manual_autom.py.
#
# install dependency once:
#   pip install pynput
#
# usage:
#   python movement_recorder.py
#
# controls while recording:
#   e        yaw right
#   q        yaw left
#   w        pitch forward
#   s        pitch backward
#   a        roll left
#   d        roll right
#   space    emergency land  (stops recording too)
#   esc      finish recording and print summary  (lands cleanly)
#
# the drone receives live commands while you fly, so what you feel is
# what the log captures — timings are real flight timings.

from codrone_edu.drone import Drone
from pynput import keyboard
import time
import threading


# ── command values sent while a key is held ───────────────────────────────────

YAW_POWER   = 60    # matches CURVE_YAW_RATE default in manual_autom.py
PITCH_POWER = 20    # matches FORWARD_PITCH
ROLL_POWER  = 20

# how often the held-key command is re-sent to the drone (seconds)
COMMAND_INTERVAL = 0.05

# ── initial descent — matches manual_autom.py so altitude is consistent ───────

DESCEND_THROTTLE = -60
DESCEND_DURATION = 0.7


# ── key → axis mapping ────────────────────────────────────────────────────────

KEY_MAP = {
    "e": {"label": "yaw_right",    "roll": 0,           "pitch": 0,           "yaw": -YAW_POWER,  "throttle": 0},
    "q": {"label": "yaw_left",     "roll": 0,           "pitch": 0,           "yaw":  YAW_POWER,  "throttle": 0},
    "w": {"label": "forward",      "roll": 0,           "pitch": PITCH_POWER, "yaw":  0,          "throttle": 0},
    "s": {"label": "backward",     "roll": 0,           "pitch":-PITCH_POWER, "yaw":  0,          "throttle": 0},
    "a": {"label": "roll_left",    "roll":-ROLL_POWER,  "pitch": 0,           "yaw":  0,          "throttle": 0},
    "d": {"label": "roll_right",   "roll": ROLL_POWER,  "pitch": 0,           "yaw":  0,          "throttle": 0},
}


# ── recorder ──────────────────────────────────────────────────────────────────

class MovementRecorder:

    def __init__(self):
        self.drone           = Drone()
        self._log            = []          # list of {label, duration, start_time}
        self._active_keys    = {}          # key_char -> press_start timestamp
        self._lock           = threading.Lock()
        self._stop_flag      = threading.Event()
        self._command_thread = None
        self._landed_via_space = False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def run(self):
        print("pairing...")
        self.drone.pair()
        time.sleep(2)
        print("paired.\n")

        try:
            t = self.drone.get_trim()
            print(f"trim on drone: roll={t[0]:+d}  pitch={t[1]:+d}")
        except Exception:
            pass

        print("taking off...")
        self.drone.takeoff()
        time.sleep(3)
        print("airborne.")

        # descend to track-following altitude so the optical flow sensor
        # is reading the floor at the same height as during real runs.
        # the drone applies throttle=-60 for 1.9 s — same as manual_autom.py.
        print("descending to track altitude...")
        self.drone.set_roll(0)
        self.drone.set_pitch(0)
        self.drone.set_yaw(0)
        self.drone.set_throttle(DESCEND_THROTTLE)
        self.drone.move(DESCEND_DURATION)
        time.sleep(DESCEND_DURATION)

        # zero throttle and let it settle before handing control to the user
        self.drone.set_throttle(0)
        self.drone.move(0.1)
        time.sleep(0.5)
        print("at track altitude — ready.\n")

        print("─" * 52)
        print("  q/e = yaw left/right    w/s = forward/back")
        print("  a/d = roll left/right")
        print("  space = emergency land")
        print("  esc   = finish and print summary")
        print()
        print("  tip: after a yaw, pause 0.5-1 s before pressing w.")
        print("  the optical flow sensor needs a moment to re-lock")
        print("  onto the floor in the new heading — that pause is")
        print("  what stops the drone drifting back after a turn.")
        print("─" * 52)
        print("recording... fly the manoeuvre you want to measure.\n")

        # start the command re-sender thread
        self._command_thread = threading.Thread(
            target=self._command_loop, daemon=True
        )
        self._command_thread.start()

        # start keyboard listener (blocks until esc or space)
        with keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        ) as listener:
            listener.join()

        # stop the command loop
        self._stop_flag.set()
        if self._command_thread:
            self._command_thread.join(timeout=1.0)

        # only land here if esc was used — space already called _land() directly
        if not getattr(self, "_landed_via_space", False):
            self._land()

        self.drone.close()
        self._print_summary()

    # ── landing ───────────────────────────────────────────────────────────────

    def _land(self):
        """
        zero all axes then land immediately.
        called directly from key handlers so landing never depends on
        the run() cleanup code being reached.
        """
        try:
            self.drone.set_roll(0)
            self.drone.set_pitch(0)
            self.drone.set_yaw(0)
            self.drone.set_throttle(0)
            self.drone.move(0.15)
        except Exception as e:
            print(f"  zero command error: {e}")

        time.sleep(0.1)

        try:
            self.drone.land()
            print("  land command sent.")
        except Exception as e:
            print(f"  land() error: {e}")
            print("  use the physical controller to land manually.")

        time.sleep(3)

    # ── keyboard callbacks ────────────────────────────────────────────────────

    def _on_press(self, key):
        # resolve the character
        try:
            ch = key.char.lower() if key.char else None
        except AttributeError:
            ch = None

        # space = emergency land — land immediately, don't wait for run() cleanup
        if key == keyboard.Key.space:
            print("\n*** SPACE — emergency land ***")
            self._stop_flag.set()
            self._landed_via_space = True
            self._land()
            return False   # stops the listener

        # esc = finish normally — lands via run() cleanup after listener exits
        if key == keyboard.Key.esc:
            print("\nesc pressed — finishing recording.")
            self._stop_flag.set()
            return False

        if ch not in KEY_MAP:
            return

        with self._lock:
            if ch not in self._active_keys:
                self._active_keys[ch] = time.perf_counter()

    def _on_release(self, key):
        try:
            ch = key.char.lower() if key.char else None
        except AttributeError:
            ch = None

        if ch not in KEY_MAP:
            return

        with self._lock:
            if ch in self._active_keys:
                duration = time.perf_counter() - self._active_keys.pop(ch)
                entry = {
                    "label":    KEY_MAP[ch]["label"],
                    "key":      ch,
                    "duration": round(duration, 3),
                    "t":        round(time.perf_counter(), 3),
                }
                self._log.append(entry)
                print(f"  logged: {entry['label']:15s}  {entry['duration']:.3f}s")

    # ── command loop ──────────────────────────────────────────────────────────

    def _command_loop(self):
        """
        re-sends the currently held command every COMMAND_INTERVAL seconds.
        if multiple keys are held, the last pressed key wins.
        if nothing is held, sends a zero command to keep the drone stable.
        """
        while not self._stop_flag.is_set():
            with self._lock:
                active = list(self._active_keys.keys())

            if active:
                # use the most recently pressed key
                ch  = active[-1]
                cmd = KEY_MAP[ch]
                self._send_command(
                    cmd["roll"], cmd["pitch"], cmd["yaw"], cmd["throttle"]
                )
            else:
                self._send_command(0, 0, 0, 0)

            time.sleep(COMMAND_INTERVAL)

    def _send_command(self, roll, pitch, yaw, throttle):
        try:
            self.drone.set_roll(roll)
            self.drone.set_pitch(pitch)
            self.drone.set_yaw(yaw)
            self.drone.set_throttle(throttle)
            self.drone.move(COMMAND_INTERVAL)
        except Exception as e:
            print(f"  send error: {e}")

    # ── summary ───────────────────────────────────────────────────────────────

    def _print_summary(self):
        if not self._log:
            print("\nno movements recorded.")
            return

        print("\n" + "═" * 52)
        print("  movement log")
        print("═" * 52)
        print(f"  {'#':<4} {'action':<16} {'duration':>10}s   {'key'}")
        print("  " + "─" * 44)
        for i, entry in enumerate(self._log, 1):
            print(f"  {i:<4} {entry['label']:<16} {entry['duration']:>10.3f}    {entry['key']}")

        print("\n" + "═" * 52)
        print("  ready-to-paste constants for manual_autom.py")
        print("═" * 52)

        # find the dominant yaw entries to suggest curve constants
        yaw_entries = [e for e in self._log if "yaw" in e["label"]]
        fwd_entries = [e for e in self._log if e["label"] == "forward"]

        if yaw_entries:
            avg_yaw = sum(e["duration"] for e in yaw_entries) / len(yaw_entries)
            print(f"\n  # curve values (averaged from {len(yaw_entries)} yaw press(es))")
            print(f"  CURVE_YAW_RATE        = {YAW_POWER}   # power used during recording")
            print(f"  CURVE_HALF_DURATION   = {avg_yaw:.3f}  # average yaw press duration")

        if fwd_entries:
            avg_fwd = sum(e["duration"] for e in fwd_entries) / len(fwd_entries)
            print(f"\n  # forward bridge values (averaged from {len(fwd_entries)} forward press(es))")
            print(f"  CURVE_BRIDGE_PITCH    = {PITCH_POWER}   # power used during recording")
            print(f"  CURVE_BRIDGE_DURATION = {avg_fwd:.3f}  # average forward press duration")

        # also print every individual duration so you can pick manually
        print("\n  # all individual durations if you want to pick manually:")
        for entry in self._log:
            print(f"  #   {entry['label']:<16} {entry['duration']:.3f}s")

        print("═" * 52)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    recorder = MovementRecorder()
    recorder.run()