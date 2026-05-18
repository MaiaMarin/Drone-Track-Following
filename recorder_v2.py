# recorder_v2.py
# --------------
# enhanced movement recorder. fly the drone manually while the script records
# every key press with its exact duration, then saves a dated json file that
# trainer.py uses to build a flight model.
#
# usage:
#   python recorder_v2.py
#
# controls while flying:
#   w        pitch forward
#   s        pitch backward
#   a        roll left
#   d        roll right
#   q        yaw left
#   e        yaw right
#   tab      mark end of current segment and move to next
#   space    emergency land  (saves log before exiting)
#   esc      finish recording cleanly  (saves log)
#
# --- why pynput was dropped for flight input ---
# on windows, pynput's keyboard.Listener uses a low-level win32 keyboard hook
# that requires the hook thread to be pumping win32 messages. the codrone_edu
# serial driver also needs win32 message processing during drone.move() calls.
# these two requirements conflict — the listener thread starves and never
# receives any keystrokes, so nothing works and the drone can't be stopped.
#
# fix: use msvcrt (windows built-in) for non-blocking key polling directly
# inside the main loop. no separate listener thread, no win32 hook conflicts.
# on linux/mac we fall back to termios+select, same approach as learned_runner.
#
# key-release detection without a key-up event:
# msvcrt has no key-up event. we detect release by timeout: if a held key
# stops sending repeat characters for more than HOLD_RELEASE_TIMEOUT seconds
# (set to 120 ms, longer than the os key-repeat interval of ~30 ms), we treat
# that as a release. a truly held key will always re-trigger before the timeout.

from codrone_edu.drone import Drone
import time
import json
import os
import sys
from datetime import datetime


# ── movement power values ─────────────────────────────────────────────────────

YAW_POWER        = 60
PITCH_POWER      = 20
ROLL_POWER       = 20
COMMAND_INTERVAL = 0.05

DESCEND_THROTTLE = -60
DESCEND_DURATION = 0.7

# how long (seconds) without a repeat event before a key is treated as released.
# must be longer than the os key-repeat interval (~30 ms on windows).
HOLD_RELEASE_TIMEOUT = 0.12

# ── available track segment types ─────────────────────────────────────────────

VALID_SEGMENTS = [
    "short_straight",
    "medium_straight",
    "long_straight",
    "left_curve",
    "right_curve",
]

# ── key -> drone command ──────────────────────────────────────────────────────

KEY_MAP = {
    "e": {"label": "yaw_right",  "roll": 0,          "pitch": 0,            "yaw": -YAW_POWER,  "throttle": 0},
    "q": {"label": "yaw_left",   "roll": 0,          "pitch": 0,            "yaw":  YAW_POWER,  "throttle": 0},
    "w": {"label": "forward",    "roll": 0,          "pitch": PITCH_POWER,  "yaw":  0,          "throttle": 0},
    "s": {"label": "backward",   "roll": 0,          "pitch":-PITCH_POWER,  "yaw":  0,          "throttle": 0},
    "a": {"label": "roll_left",  "roll":-ROLL_POWER, "pitch": 0,            "yaw":  0,          "throttle": 0},
    "d": {"label": "roll_right", "roll": ROLL_POWER, "pitch": 0,            "yaw":  0,          "throttle": 0},
}

_TAB_CHAR   = "\t"
_SPACE_CHAR = " "
_ESC_CHAR   = "\x1b"


# ── platform key reader ───────────────────────────────────────────────────────
# returns a single character if a key is waiting, or None. never blocks.
# identical pattern to learned_runner.py which is confirmed working on windows.

if sys.platform == "win32":
    import msvcrt

    def _read_key():
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            # special keys (arrows, fn keys) send two bytes: 0xe0 then a code.
            # consume and discard the second byte so it does not corrupt the log.
            if ch in (b"\xe0", b"\x00"):
                msvcrt.getch()
                return None
            try:
                return ch.decode("utf-8")
            except UnicodeDecodeError:
                return None
        return None

else:
    import tty, termios, select

    def _read_key():
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return None


# ── recorder ──────────────────────────────────────────────────────────────────

class RecorderV2:

    def __init__(self, track_sequence):
        self.track_sequence  = track_sequence
        self.drone           = Drone()
        self._events         = []
        self._t0             = None
        self._segment_index  = 0
        self._held           = {}
        # when True, taps are tagged "correction_<action>" instead of "<action>".
        # press c to toggle. use it whenever you nudge the drone back into
        # position rather than making genuine progress along the track.
        self._correction_mode = False

    # ── helpers ───────────────────────────────────────────────────────────────

    def _t(self):
        return round(time.perf_counter() - self._t0, 4)

    def _action_label(self, ch):
        """return the action label, prefixed with 'correction_' when in correction mode."""
        base = KEY_MAP[ch]["label"]
        return f"correction_{base}" if self._correction_mode else base

    # ── landing ───────────────────────────────────────────────────────────────

    def _land(self):
        # always called from the main thread — no concurrent drone access.
        try:
            self.drone.set_roll(0)
            self.drone.set_pitch(0)
            self.drone.set_yaw(0)
            self.drone.set_throttle(0)
            self.drone.move(0.15)
        except Exception as e:
            print(f"  zero error: {e}")
        time.sleep(0.1)
        try:
            self.drone.land()
            print("  land command sent.")
        except Exception as e:
            print(f"  land() error: {e}")
            print("  use the physical controller to land manually.")
        time.sleep(3)

    # ── key-release flush ─────────────────────────────────────────────────────

    def _release_all(self):
        """log a release for every key still tracked as held."""
        now = time.perf_counter()
        for ch, press_t in list(self._held.items()):
            duration = round(now - press_t, 4)
            self._events.append({
                "type":          "release",
                "key":           ch,
                "action":        self._action_label(ch),
                "duration":      duration,
                "segment_index": self._segment_index,
                "t":             self._t(),
            })
        self._held.clear()

    # ── key processing ────────────────────────────────────────────────────────

    def _process_key(self, ch):
        """
        handle one character from _read_key().
        returns True if the flight loop should exit (land and stop).
        """
        now = time.perf_counter()

        if ch == _SPACE_CHAR:
            print("\n*** space — emergency land ***")
            self._release_all()
            self._land()
            return True

        if ch == _ESC_CHAR:
            print("\nesc — finishing recording.")
            self._release_all()
            self._land()
            return True

        if ch == _TAB_CHAR:
            seg_type = (
                self.track_sequence[self._segment_index]
                if self._segment_index < len(self.track_sequence)
                else "unknown"
            )
            print(f"  [tab] segment {self._segment_index + 1} done: {seg_type}")
            self._events.append({
                "type":          "segment_end",
                "segment_index": self._segment_index,
                "segment_type":  seg_type,
                "t":             self._t(),
            })
            self._segment_index += 1
            remaining = len(self.track_sequence) - self._segment_index
            if remaining > 0:
                print(f"  next: {self.track_sequence[self._segment_index]}  ({remaining} remaining)")
            else:
                print("  all segments marked — press esc to finish.")
            return False

        # c = toggle correction mode
        if ch.lower() == "c":
            self._correction_mode = not self._correction_mode
            state = "CORRECTION (nudges to stay in place)" if self._correction_mode else "PROGRESS (moving along track)"
            print(f"  [c] mode → {state}")
            return False

        ch_lower = ch.lower()
        if ch_lower not in KEY_MAP:
            return False

        if ch_lower not in self._held:
            # leading edge — new press
            self._held[ch_lower] = now
            self._events.append({
                "type":          "press",
                "key":           ch_lower,
                "action":        self._action_label(ch_lower),
                "segment_index": self._segment_index,
                "t":             self._t(),
            })
        else:
            # key-repeat — update the last-seen timestamp so the timeout resets
            self._held[ch_lower] = now

        return False

    # ── main flight loop ──────────────────────────────────────────────────────
    # single-threaded: key polling, state tracking, and drone commands all run
    # here so there is zero concurrent access to the drone object.

    def _flight_loop(self):
        while True:
            now = time.perf_counter()

            # poll for a key character
            ch = _read_key()
            if ch is not None:
                should_exit = self._process_key(ch)
                if should_exit:
                    return

            # detect releases by timeout: if a held key has not sent a
            # repeat character within HOLD_RELEASE_TIMEOUT, treat as released.
            for held_ch in list(self._held.keys()):
                if now - self._held[held_ch] > HOLD_RELEASE_TIMEOUT:
                    press_t   = self._held.pop(held_ch)
                    duration  = round(now - press_t, 4)
                    action    = self._action_label(held_ch)
                    seg_label = (
                        self.track_sequence[self._segment_index]
                        if self._segment_index < len(self.track_sequence)
                        else "?"
                    )
                    self._events.append({
                        "type":          "release",
                        "key":           held_ch,
                        "action":        action,
                        "duration":      duration,
                        "segment_index": self._segment_index,
                        "t":             self._t(),
                    })
                    mode_tag = " [CORR]" if self._correction_mode else ""
                    print(f"  {action:25s}  {duration:.3f}s   [seg {self._segment_index + 1}: {seg_label}]{mode_tag}")

            # send the drone command for whichever key is currently held.
            # if multiple keys are held, use the most recently pressed one.
            if self._held:
                active_ch = max(self._held, key=lambda k: self._held[k])
                cmd       = KEY_MAP[active_ch]
                try:
                    self.drone.set_roll(cmd["roll"])
                    self.drone.set_pitch(cmd["pitch"])
                    self.drone.set_yaw(cmd["yaw"])
                    self.drone.set_throttle(cmd["throttle"])
                    self.drone.move(COMMAND_INTERVAL)
                except Exception as e:
                    print(f"  send error: {e}")
            else:
                try:
                    self.drone.set_roll(0)
                    self.drone.set_pitch(0)
                    self.drone.set_yaw(0)
                    self.drone.set_throttle(0)
                    self.drone.move(COMMAND_INTERVAL)
                except Exception:
                    pass

    # ── save ──────────────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs("recordings", exist_ok=True)
        filename = os.path.join(
            "recordings",
            datetime.now().strftime("rec_%Y%m%d_%H%M%S.json"),
        )
        data = {
            "recorded_at":    datetime.now().isoformat(),
            "track_sequence": self.track_sequence,
            "yaw_power":      YAW_POWER,
            "pitch_power":    PITCH_POWER,
            "events":         self._events,
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  saved to {filename}")
        return filename

    def _ask_keep(self):
        print()
        print("─" * 56)
        print("  keep this recording for training? (y/n)")
        print("  y = save to recordings/ and use in trainer.py")
        print("  n = discard (nothing written to disk)")
        print("─" * 56)
        while True:
            raw = input("  keep? [y/n]: ").strip().lower()
            if raw in ("y", "yes"):
                return self._save()
            if raw in ("n", "no"):
                print("  recording discarded — nothing saved.")
                return None
            print("  please enter y or n.")

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self):
        print("pairing...")
        self.drone.pair()
        time.sleep(2)
        print("paired.\n")

        try:
            t = self.drone.get_trim()
            print(f"trim: roll={t[0]:+d}  pitch={t[1]:+d}\n")
        except Exception:
            pass

        print("taking off...")
        self.drone.takeoff()
        time.sleep(3)

        print("descending to track altitude...")
        self.drone.set_roll(0)
        self.drone.set_pitch(0)
        self.drone.set_yaw(0)
        self.drone.set_throttle(DESCEND_THROTTLE)
        self.drone.move(DESCEND_DURATION)
        time.sleep(DESCEND_DURATION)
        self.drone.set_throttle(0)
        self.drone.move(0.1)
        time.sleep(0.5)
        print("at track altitude.\n")

        print("─" * 56)
        print(f"  track sequence ({len(self.track_sequence)} segments):")
        for i, seg in enumerate(self.track_sequence):
            print(f"    {i + 1}. {seg}")
        print()
        print("  w/s = forward/back    a/d = roll    q/e = yaw")
        print("  c   = toggle correction mode  (use when nudging drone back into place)")
        print("  tab = mark segment done and move to next")
        print("  space = emergency land    esc = finish")
        print("─" * 56)
        print(f"  starting segment 1: {self.track_sequence[0]}")
        print("  [keys are live — fly now]\n")

        self._t0 = time.perf_counter()
        self._events.append({"type": "recording_start", "t": 0.0})

        try:
            self._flight_loop()
        except Exception as e:
            print(f"  flight loop error: {e}")
            self._land()

        filename = self._ask_keep()

        try:
            self.drone.close()
        except Exception:
            pass

        self._print_summary(filename)

    # ── summary ───────────────────────────────────────────────────────────────

    def _print_summary(self, filename):
        releases = [e for e in self._events if e["type"] == "release"]
        markers  = [e for e in self._events if e["type"] == "segment_end"]

        print("\n" + "=" * 56)
        print("  recording summary")
        print("=" * 56)
        print(f"  total events   : {len(self._events)}")
        print(f"  key presses    : {len(releases)}")
        print(f"  segments marked: {len(markers)} / {len(self.track_sequence)}")
        if len(markers) < len(self.track_sequence):
            print(f"  note: only {len(markers)} of {len(self.track_sequence)} segments were marked.")
        if filename:
            print(f"\n  saved to: {filename}")
            print("  run trainer.py to update the model.")
        else:
            print("\n  recording was discarded — model unchanged.")
        print("=" * 56)


# ── track sequence — edit this when you want to record a different layout ─────

TRACK_SEQUENCE = [
    "short_straight",
    "right_curve",
    "medium_straight",
    "long_straight",
    "left_curve",
    "medium_straight",
    "medium_straight",
    "short_straight",
    "medium_straight",
    "medium_straight",
    "right_curve",
    "medium_straight",
    "long_straight",
    "right_curve",
    "medium_straight",
    "medium_straight",
    "left_curve",
]


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("sequence to record:")
    for i, s in enumerate(TRACK_SEQUENCE):
        print(f"  {i + 1}. {s}")
    print()
    recorder = RecorderV2(TRACK_SEQUENCE)
    recorder.run()