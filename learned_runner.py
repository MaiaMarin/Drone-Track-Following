# learned_runner.py
# -----------------
# loads model.json and flies the hardcoded track sequence below by replaying
# the learned tap-pause-tap pattern for each segment.
#
# changes vs previous version:
#   - space bar emergency lands from anywhere (watcher + main loop both check)
#   - taps are consolidated: multiple short taps are merged into fewer longer
#     ones so the drone does not mistake them for a recalibration trigger.
#     total flight time per segment stays the same.
#   - press 'h' during flight to enter manual hover mode: the drone holds
#     position while you correct it with w/s/a/d/q/e. when you press enter,
#     those corrections are recorded and folded back into model.json so the
#     runner improves over time.
#   - the track sequence is hardcoded below. edit TRACK_SEQUENCE to change it.
#
# usage:
#   python learned_runner.py

from codrone_edu.drone import Drone
import json
import time
import threading
import sys
import os
import statistics
from datetime import datetime
from collections import defaultdict

MODEL_FILE     = "model.json"
VALID_SEGMENTS = [
    "short_straight", "medium_straight", "long_straight",
    "left_curve", "right_curve",
]

DESCEND_THROTTLE = -60
DESCEND_DURATION = 0.7

YAW_POWER   = 60    # must match what was used during recording
PITCH_POWER = 20
ROLL_POWER  = 20

# ── tap consolidation ─────────────────────────────────────────────────────────
#
# when the model has many short taps for one action, the drone can interpret
# rapid identical commands as a self-calibration trigger. consolidation merges
# consecutive same-action taps into fewer, longer ones while keeping the total
# active-command time identical. the saved gaps are spread proportionally so
# the total segment duration is also preserved.
#
# set to 1 to disable consolidation entirely.
TAP_CONSOLIDATION_RATIO = 0.5   # merge taps until each is at least this many
                                 # times longer than avg_inter_gap. 0.5 means
                                 # each tap lasts at least half the gap duration.
                                 # raise toward 1.0 for even longer taps.

# minimum tap duration after consolidation (seconds).
# the drone ignores commands shorter than ~80 ms on most firmware builds.
MIN_TAP_DURATION = 0.18

# ── curve trim ────────────────────────────────────────────────────────────────
#
# after consolidation, curve segments (left_curve / right_curve) are trimmed
# further so the drone does not over-rotate or drift too far forward.
#
# each ratio is a fraction of the consolidated tap count for that action class.
# 0.5 = keep half the taps (rounded up to at least 1).
# 1.0 = no change.
#
# CURVE_GAP_MULTIPLIER stretches the inter-tap gap for curve segments so the
# drone has more time to recenter between each yaw. 1.0 = unchanged.
#
# only left_curve and right_curve are affected; straight segments are ignored.

CURVE_YAW_RATIO     = 0.35   # fraction of yaw taps to keep
CURVE_FORWARD_RATIO = 0.5   # fraction of forward taps to keep
CURVE_GAP_MULTIPLIER = 1.3  # multiply inter-tap gap by this for curves

# ── manual hover correction key bindings ─────────────────────────────────────

HOVER_KEY_MAP = {
    "w": ("forward",    0,           PITCH_POWER,  0,          0),
    "s": ("backward",   0,          -PITCH_POWER,  0,          0),
    "a": ("roll_left",  -ROLL_POWER, 0,            0,          0),
    "d": ("roll_right",  ROLL_POWER, 0,            0,          0),
    "q": ("yaw_left",   0,           0,            YAW_POWER,  0),
    "e": ("yaw_right",  0,           0,           -YAW_POWER,  0),
}

# how long each manual correction tap lasts (seconds)
CORRECTION_TAP_DURATION = 0.20

# how long to hold zero between manual correction taps so the drone recenters
CORRECTION_GAP = 0.30

# ── action -> drone command ───────────────────────────────────────────────────

ACTION_COMMANDS = {
    "forward":    (0,           PITCH_POWER,  0,          0),
    "backward":   (0,          -PITCH_POWER,  0,          0),
    "yaw_right":  (0,           0,           -YAW_POWER,  0),
    "yaw_left":   (0,           0,            YAW_POWER,  0),
    "roll_right": (ROLL_POWER,  0,            0,          0),
    "roll_left":  (-ROLL_POWER, 0,            0,          0),
}

# ── hardcoded track sequence ──────────────────────────────────────────────────
# edit this list to change the route. shortcuts in comments for reference:
#   ss = short_straight   ms = medium_straight   ls = long_straight
#   lc = left_curve       rc = right_curve

TRACK_SEQUENCE = [
    "short_straight",    # segment 1
    "right_curve",       # segment 2
    "medium_straight",   # segment 3
    "long_straight",     # segment 4
    "left_curve",        # segment 5
    "medium_straight",   # segment 6
    "medium_straight",   # segment 7
    "short_straight",    # segment 8
    "medium_straight",   # segment 9
    "medium_straight",   # segment 10
    "right_curve",       # segment 11
    "medium_straight",   # segment 12
    "long_straight",     # segment 13
    "right_curve",       # segment 14
    "short_straight",    # segment 15
    "medium_straight",   # segment 16
    "left_curve",        # segment 17
]


# ── platform key reader ───────────────────────────────────────────────────────
# returns one character if a key is waiting, or None. never blocks.

if sys.platform == "win32":
    import msvcrt

    def _read_key():
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch in (b"\xe0", b"\x00"):
                msvcrt.getch()
                return None
            try:
                return ch.decode("utf-8")
            except Exception:
                return None
        return None

else:
    import tty, termios, select

    def _read_key():
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            if select.select([sys.stdin], [], [], 0.05)[0]:
                return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return None


# ── tap consolidation logic ───────────────────────────────────────────────────

def _consolidate_taps(action_sequence, tap_duration, inter_gap):
    """
    merge consecutive same-action taps into fewer, longer taps so the drone
    does not interpret rapid identical commands as a recalibration signal.

    the total active-command time is preserved:
      original: n taps * tap_duration
      merged:   m taps * new_tap_duration   where n * tap_duration == m * new_tap_duration

    the gaps are scaled proportionally so total segment duration is unchanged.

    returns (new_sequence, new_tap_duration, new_inter_gap).
    """
    if not action_sequence or tap_duration <= 0:
        return action_sequence, tap_duration, inter_gap

    # group consecutive identical actions into runs
    runs = []
    for action in action_sequence:
        if runs and runs[-1][0] == action:
            runs[-1][1] += 1
        else:
            runs.append([action, 1])

    # for each run, decide how many merged taps to use.
    # target: each merged tap >= MIN_TAP_DURATION and >= gap * TAP_CONSOLIDATION_RATIO.
    target_min = max(MIN_TAP_DURATION, inter_gap * TAP_CONSOLIDATION_RATIO)

    new_sequence = []
    for action, count in runs:
        total_active = count * tap_duration
        # how many merged taps do we need so each is >= target_min?
        merged_count = max(1, int(total_active / target_min))
        merged_count = min(merged_count, count)   # never more than original
        new_sequence.extend([action] * merged_count)

    if not new_sequence:
        return action_sequence, tap_duration, inter_gap

    # recalculate tap duration: keep total active time the same.
    # total active time = sum over runs of (run_original_count * tap_duration)
    original_total_active = len(action_sequence) * tap_duration
    new_tap_duration = round(original_total_active / len(new_sequence), 4)
    new_tap_duration = max(new_tap_duration, MIN_TAP_DURATION)

    # recalculate inter_gap: keep total segment duration the same.
    # total duration = taps * tap_d + (taps - 1) * gap + end_settle
    # end_settle is not touched here so just scale the inter_gap.
    if len(new_sequence) > 1 and len(action_sequence) > 1:
        original_gaps_total = (len(action_sequence) - 1) * inter_gap
        new_inter_gap = round(original_gaps_total / (len(new_sequence) - 1), 4)
    else:
        new_inter_gap = inter_gap

    return new_sequence, new_tap_duration, new_inter_gap


# ── curve trim logic ─────────────────────────────────────────────────────────

def _trim_curve_sequence(sequence, tap_duration, inter_gap):
    """
    reduce the number of yaw and forward taps in a curve segment so the drone
    does not over-rotate or travel too far forward.

    for each action class (yaw_*, forward) the count is multiplied by the
    corresponding ratio and rounded up to at least 1 so the action still fires.
    the inter-tap gap is also stretched by CURVE_GAP_MULTIPLIER to give the
    drone more time to recenter between taps.

    tap duration is recalculated so total active command time is preserved.

    returns (new_sequence, new_tap_duration, new_inter_gap).
    """
    if not sequence:
        return sequence, tap_duration, inter_gap

    # count how many taps of each action class exist before trimming
    original_count = len(sequence)

    # walk the sequence and trim per action class, preserving order
    # group into runs so we can apply the ratio per run
    runs = []
    for action in sequence:
        if runs and runs[-1][0] == action:
            runs[-1][1] += 1
        else:
            runs.append([action, 1])

    new_sequence = []
    for action, count in runs:
        base = action.replace("correction_", "")
        if base in ("yaw_right", "yaw_left"):
            ratio = CURVE_YAW_RATIO
        elif base == "forward":
            ratio = CURVE_FORWARD_RATIO
        else:
            ratio = 1.0
        # round up, keep at least 1
        trimmed = max(1, round(count * ratio))
        # never exceed the original count
        trimmed = min(trimmed, count)
        new_sequence.extend([action] * trimmed)

    trimmed_total = len(new_sequence)

    if trimmed_total == 0:
        return sequence, tap_duration, inter_gap

    # recalculate tap duration to preserve total active command time
    original_active = original_count * tap_duration
    new_tap_duration = round(original_active / trimmed_total, 4)
    new_tap_duration = max(new_tap_duration, MIN_TAP_DURATION)

    # stretch the gap so the drone has longer to recenter between taps
    new_inter_gap = round(inter_gap * CURVE_GAP_MULTIPLIER, 4)

    return new_sequence, new_tap_duration, new_inter_gap


# ── model updater — folds manual corrections back into model.json ─────────────

def _update_model_with_corrections(model_file, seg_type, correction_taps):
    """
    read model.json, append the correction taps for seg_type as a new
    pseudo-demonstration, recompute averages, and write the file back.

    correction_taps: list of {"action": str, "duration": float}
    """
    if not correction_taps:
        return

    try:
        with open(model_file) as f:
            model = json.load(f)
    except Exception as e:
        print(f"  [model update] could not read {model_file}: {e}")
        return

    if seg_type not in model:
        print(f"  [model update] {seg_type} not in model — skipping update")
        return

    entry = model[seg_type]

    # current averages
    current_seq      = entry.get("action_sequence", [])
    current_tap_dur  = entry.get("avg_tap_duration", 0.0)
    current_gap      = entry.get("avg_inter_gap", 0.0)
    current_settle   = entry.get("avg_end_settle", 0.0)
    n                = entry.get("demonstrations", 1)

    # treat the correction block as an extra demonstration:
    # its tap durations are averaged in with the existing avg.
    new_durations = [t["duration"] for t in correction_taps]
    all_durations = [current_tap_dur] * n + new_durations
    new_avg_dur   = round(sum(all_durations) / len(all_durations), 4)

    # append correction action labels to the canonical sequence
    # (only add actions not already present to avoid bloating it)
    existing_set = set(current_seq)
    for t in correction_taps:
        action = f"correction_{t['action']}"
        if action not in existing_set:
            current_seq.append(action)
            existing_set.add(action)

    entry["demonstrations"]   = n + 1
    entry["action_sequence"]  = current_seq
    entry["avg_tap_duration"] = new_avg_dur
    # gap and settle are unchanged — corrections are zero-gap by design

    model[seg_type] = entry

    try:
        with open(model_file, "w") as f:
            json.dump(model, f, indent=2)
        print(f"  [model update] {seg_type} updated — {n + 1} demo(s) now in model")
    except Exception as e:
        print(f"  [model update] write failed: {e}")


# ── runner ────────────────────────────────────────────────────────────────────

class LearnedRunner:

    def __init__(self, sequence, model):
        self.sequence         = sequence
        self.model            = model
        self.drone            = Drone()
        self._stop_flag       = threading.Event()
        # set by 'h' keypress to request a manual hover pause
        self._hover_flag      = threading.Event()
        self._watcher         = None
        self._current_seg     = None   # name of the segment being flown

    # ── landing ───────────────────────────────────────────────────────────────

    def _land(self):
        try:
            self.drone.set_roll(0)
            self.drone.set_pitch(0)
            self.drone.set_yaw(0)
            self.drone.set_throttle(0)
            self.drone.move(0.15)
        except Exception:
            pass
        time.sleep(0.1)
        try:
            self.drone.land()
            print("  land command sent.")
        except Exception as e:
            print(f"  land() error: {e}")
        time.sleep(3)

    # ── watcher thread ────────────────────────────────────────────────────────
    # monitors space (emergency land) and h (manual hover) between segments.
    # during _send_tap / _zero the main thread owns the drone object, so the
    # watcher only sets flags — it never touches the drone directly.

    def _start_watcher(self):
        def _watch():
            print()
            print("  ╔═══════════════════════════════════════════════════╗")
            print("  ║  space = emergency land   h = manual hover/correct ║")
            print("  ╚═══════════════════════════════════════════════════╝")
            print()
            while not self._stop_flag.is_set():
                ch = _read_key()
                if ch == " ":
                    print("\n*** emergency land — space pressed ***")
                    self._stop_flag.set()
                    self._land()
                    return
                if ch in ("h", "H"):
                    if not self._hover_flag.is_set():
                        print("\n  [h] manual hover requested — will pause after current tap")
                        self._hover_flag.set()
                time.sleep(0.05)

        self._watcher = threading.Thread(target=_watch, daemon=True)
        self._watcher.start()

    def _stop_watcher(self):
        self._stop_flag.set()
        if self._watcher:
            self._watcher.join(timeout=1.0)

    # ── raw drone sender (used in hover mode without the watcher interfering) ─

    def _send_raw(self, roll, pitch, yaw, throttle, duration):
        """send a command for duration seconds. does not check stop flag."""
        self.drone.set_roll(roll)
        self.drone.set_pitch(pitch)
        self.drone.set_yaw(yaw)
        self.drone.set_throttle(throttle)
        self.drone.move(duration)
        time.sleep(duration)

    # ── tap sender ────────────────────────────────────────────────────────────

    def _send_tap(self, action, duration):
        """send one tap command for duration seconds."""
        if self._stop_flag.is_set():
            return
        base_action = action.replace("correction_", "")
        cmd = ACTION_COMMANDS.get(base_action)
        if cmd is None:
            print(f"  unknown action '{action}' — skipping tap")
            return
        roll, pitch, yaw, throttle = cmd
        self.drone.set_roll(roll)
        self.drone.set_pitch(pitch)
        self.drone.set_yaw(yaw)
        self.drone.set_throttle(throttle)
        self.drone.move(duration)
        time.sleep(duration)

    def _zero(self, duration):
        """send zero command for duration seconds so the drone self-centers."""
        if self._stop_flag.is_set() or duration <= 0:
            return
        self.drone.set_roll(0)
        self.drone.set_pitch(0)
        self.drone.set_yaw(0)
        self.drone.set_throttle(0)
        self.drone.move(duration)
        time.sleep(duration)

    # ── manual hover / correction mode ────────────────────────────────────────

    def _manual_hover(self, seg_type):
        """
        pause automatic flight and hand control to the operator.
        the drone hovers on its own (zero command loop) while you press keys
        to nudge it back into position. press enter to resume the sequence.
        all corrections are recorded and folded into model.json automatically.
        """
        print()
        print("  ┌─────────────────────────────────────────────────┐")
        print("  │  manual hover mode  (drone holding position)    │")
        print("  │  w/s = pitch   a/d = roll   q/e = yaw           │")
        print("  │  space = emergency land                         │")
        print("  │  press enter when done — sequence will resume   │")
        print("  └─────────────────────────────────────────────────┘")
        print()

        correction_taps = []

        # hold release timeout for manual corrections (same approach as recorder_v2)
        HOLD_TIMEOUT = 0.12
        held         = {}

        # run a tight loop: poll keys, send commands, detect releases by timeout
        # enter key (b'\r' on windows, '\n' or '\r' on unix) ends the loop.
        resume    = threading.Event()
        loop_lock = threading.Lock()

        def _hover_loop():
            while not resume.is_set() and not self._stop_flag.is_set():
                now = time.perf_counter()
                ch  = _read_key()

                if ch is not None:
                    if ch == " ":
                        print("\n*** emergency land — space in hover mode ***")
                        self._stop_flag.set()
                        resume.set()
                        return
                    if ch in ("\r", "\n"):
                        resume.set()
                        return
                    if ch.lower() in HOVER_KEY_MAP:
                        k = ch.lower()
                        if k not in held:
                            held[k] = now
                        else:
                            held[k] = now   # key-repeat: reset timeout

                # release detection by timeout
                for k in list(held.keys()):
                    if now - held[k] > HOLD_TIMEOUT:
                        press_t  = held.pop(k)
                        duration = round(now - press_t, 4)
                        action   = HOVER_KEY_MAP[k][0]
                        correction_taps.append({"action": action, "duration": duration})
                        print(f"  correction: {action}  {duration:.3f}s")

                # send current command
                if held:
                    active = max(held, key=lambda k: held[k])
                    _, roll, pitch, yaw, throttle = HOVER_KEY_MAP[active]
                    try:
                        self.drone.set_roll(roll)
                        self.drone.set_pitch(pitch)
                        self.drone.set_yaw(yaw)
                        self.drone.set_throttle(throttle)
                        self.drone.move(0.05)
                    except Exception:
                        pass
                    time.sleep(0.01)
                else:
                    try:
                        self.drone.set_roll(0)
                        self.drone.set_pitch(0)
                        self.drone.set_yaw(0)
                        self.drone.set_throttle(0)
                        self.drone.move(0.05)
                    except Exception:
                        pass
                    time.sleep(0.01)

        _hover_loop()

        self._hover_flag.clear()

        if self._stop_flag.is_set():
            return

        if correction_taps:
            print(f"  {len(correction_taps)} correction tap(s) recorded for '{seg_type}'")
            _update_model_with_corrections(MODEL_FILE, seg_type, correction_taps)
        else:
            print("  no corrections recorded.")

        print("  resuming sequence...")
        print()

    # ── segment executor ──────────────────────────────────────────────────────

    def _fly_segment(self, seg_type, params):
        """
        replay the consolidated tap sequence for this segment.
        checks hover_flag before each tap so h can pause the sequence cleanly.
        """
        raw_sequence = params["action_sequence"]
        tap_duration = params["avg_tap_duration"]
        inter_gap    = params["avg_inter_gap"]
        end_settle   = params["avg_end_settle"]

        if not raw_sequence:
            print(f"  no action sequence for {seg_type} — skipping")
            return

        # consolidate short taps into fewer, longer ones
        sequence, tap_duration, inter_gap = _consolidate_taps(
            raw_sequence, tap_duration, inter_gap
        )

        original_count = len(raw_sequence)
        merged_count   = len(sequence)
        if merged_count < original_count:
            print(f"  consolidated {original_count} taps -> {merged_count} taps"
                  f"  (tap={tap_duration:.4f}s  gap={inter_gap:.4f}s)")

        # for curve segments, trim yaw and forward counts and stretch the gap
        if seg_type in ("left_curve", "right_curve"):
            pre_trim = len(sequence)
            sequence, tap_duration, inter_gap = _trim_curve_sequence(
                sequence, tap_duration, inter_gap
            )
            if len(sequence) < pre_trim:
                print(f"  curve trim: {pre_trim} taps -> {len(sequence)} taps"
                      f"  (tap={tap_duration:.4f}s  gap={inter_gap:.4f}s)")

        for i, action in enumerate(sequence):
            # check for manual hover request before each tap
            if self._hover_flag.is_set():
                self._manual_hover(seg_type)
                if self._stop_flag.is_set():
                    return

            if self._stop_flag.is_set():
                return

            label = f"[corr] {action}" if action.startswith("correction_") else action
            print(f"  tap {i + 1}/{len(sequence)}: {label}  {tap_duration:.4f}s")
            self._send_tap(action, tap_duration)

            # recenter pause between taps (skip after the last tap)
            if i < len(sequence) - 1:
                if self._stop_flag.is_set():
                    return
                print(f"  recenter pause  {inter_gap:.4f}s")
                self._zero(inter_gap)

        if self._stop_flag.is_set():
            return
        print(f"  end settle  {end_settle:.4f}s")
        self._zero(end_settle)

    # ── flight plan display ───────────────────────────────────────────────────

    def print_plan(self):
        print("\n" + "═" * 66)
        print("  flight plan  (with tap consolidation)")
        print("═" * 66)
        missing = []
        for i, seg in enumerate(self.sequence):
            params = self.model.get(seg)
            if params is None:
                print(f"  {i + 1:2d}. {seg:20s}  NO DATA")
                missing.append(seg)
            else:
                raw_seq  = params["action_sequence"]
                tap_d    = params["avg_tap_duration"]
                gap      = params["avg_inter_gap"]
                settle   = params["avg_end_settle"]
                merged, new_tap_d, new_gap = _consolidate_taps(raw_seq, tap_d, gap)
                if seg in ("left_curve", "right_curve"):
                    merged, new_tap_d, new_gap = _trim_curve_sequence(merged, new_tap_d, new_gap)
                n        = params["demonstrations"]
                print(f"  {i + 1:2d}. {seg:20s}  {len(raw_seq)} taps -> {len(merged)} merged"
                      + (" (curve trimmed)" if seg in ("left_curve", "right_curve") else ""))
                print(f"       tap={new_tap_d:.4f}s  gap={new_gap:.4f}s  settle={settle:.4f}s  ({n} demo(s))")
        print("═" * 66)
        if missing:
            print(f"\n  missing from model: {missing}")
            print("  record demonstrations containing those segment types first.")
            return False
        return True

    # ── main run ──────────────────────────────────────────────────────────────

    def run(self):
        ok = self.print_plan()
        if not ok:
            return

        print("\npress enter to take off, or ctrl+c to abort.")
        try:
            input()
        except KeyboardInterrupt:
            print("aborted.")
            return

        print("pairing...")
        self.drone.pair()
        time.sleep(2)

        try:
            t = self.drone.get_trim()
            print(f"trim: roll={t[0]:+d}  pitch={t[1]:+d}")
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

        self._start_watcher()

        try:
            for i, seg in enumerate(self.sequence):
                if self._stop_flag.is_set():
                    break

                # check hover flag between segments too
                if self._hover_flag.is_set():
                    self._manual_hover(seg)
                    if self._stop_flag.is_set():
                        break

                self._current_seg = seg
                print(f"\nsegment {i + 1}/{len(self.sequence)}: {seg}")
                self._fly_segment(seg, self.model[seg])

            if not self._stop_flag.is_set():
                print("\nsequence complete — landing.")
                self._land()
                self._stop_watcher()
            else:
                self._stop_watcher()

        except Exception as e:
            print(f"error: {e}")
            self._stop_flag.set()
            self._land()
        finally:
            try:
                self.drone.close()
            except Exception:
                pass


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        print(f"model.json not found.")
        print("run recorder_v2.py then trainer.py first.")
        sys.exit(1)

    with open(MODEL_FILE) as f:
        model = json.load(f)

    print(f"loaded model — {len(model)} segment type(s): {list(model.keys())}")
    print(f"track sequence: {len(TRACK_SEQUENCE)} segments (hardcoded)")

    runner = LearnedRunner(TRACK_SEQUENCE, model)
    runner.run()