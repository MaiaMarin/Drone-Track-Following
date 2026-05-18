# trainer.py
# ----------
# loads every json file in recordings/, extracts the real tap-pause-tap
# pattern from each segment, and saves model.json for learned_runner.py.
#
# what changed vs the old trainer:
#   - instead of summing forward_time into one number and the runner sending
#     one continuous command, we now extract the actual ordered sequence of
#     actions (forward / yaw_right / ...) and the gap between consecutive
#     taps. the runner replays that sequence tap by tap, with the learned
#     pause between each one — which is what makes the drone recenter.
#
# usage:
#   python trainer.py

import json
import os
import glob
import statistics
from collections import defaultdict

RECORDINGS_DIR = "recordings"
MODEL_FILE     = "model.json"


def load_recordings():
    files = sorted(glob.glob(os.path.join(RECORDINGS_DIR, "rec_*.json")))
    if not files:
        print(f"no recordings found in {RECORDINGS_DIR}/")
        print("run recorder_v2.py first.")
        return []
    print(f"found {len(files)} recording(s):")
    recs = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        print(f"  {f}  seq: {data['track_sequence']}")
        recs.append(data)
    return recs


def split_into_segments(recording):
    events   = recording["events"]
    sequence = recording["track_sequence"]
    segments = []
    bucket   = []

    for ev in events:
        if ev["type"] == "segment_end":
            segments.append({
                "segment_type": ev["segment_type"],
                "events":       bucket,
                "marker_t":     ev["t"],
            })
            bucket = []
        elif ev["type"] != "recording_start":
            bucket.append(ev)

    if bucket and len(segments) < len(sequence):
        seg_type = sequence[len(segments)] if len(segments) < len(sequence) else "unknown"
        last_t   = bucket[-1]["t"] if bucket else 0
        segments.append({"segment_type": seg_type, "events": bucket, "marker_t": last_t})

    return segments


def extract_taps(seg_events, marker_t):
    """
    returns:
      taps      : list of {"action": str, "duration": float, "release_t": float}
      gaps      : inter-tap gaps — the pause AFTER tap[i], before tap[i+1].
                  this is when the drone self-centers. len == len(taps) - 1.
      end_settle: gap from last release to segment_end marker (tab press).
    """
    releases = sorted(
        [e for e in seg_events if e["type"] == "release"],
        key=lambda e: e["t"],
    )
    presses = sorted(
        [e for e in seg_events if e["type"] == "press"],
        key=lambda e: e["t"],
    )

    if not releases:
        return [], [], 0.0

    taps = [
        {"action": r["action"], "duration": r["duration"], "release_t": r["t"]}
        for r in releases
    ]

    gaps = []
    for tap in taps[:-1]:
        next_press_times = [p["t"] for p in presses if p["t"] > tap["release_t"]]
        gap = round(min(next_press_times) - tap["release_t"], 4) if next_press_times else 0.0
        gaps.append(max(0.0, gap))

    end_settle = round(max(0.0, marker_t - taps[-1]["release_t"]), 4)
    return taps, gaps, end_settle


def representative_sequence(all_tap_lists):
    """pick the demo closest to median tap count as the canonical action order."""
    if not all_tap_lists:
        return []
    lengths    = [len(t) for t in all_tap_lists]
    median_len = statistics.median(lengths)
    best       = min(all_tap_lists, key=lambda t: abs(len(t) - median_len))
    return [tap["action"] for tap in best]


def compute_model(collected):
    model = {}
    for seg_type, data in collected.items():
        tap_lists    = data["tap_lists"]
        all_gaps     = data["all_gaps"]
        all_settles  = data["all_end_settles"]
        all_durations = data["all_durations"]
        n            = len(tap_lists)

        def avg(lst):
            lst = [v for v in lst if v is not None]
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        def std(lst):
            lst = [v for v in lst if v is not None]
            return round(statistics.stdev(lst), 4) if len(lst) >= 2 else 0.0

        model[seg_type] = {
            "demonstrations":   n,
            "action_sequence":  representative_sequence(tap_lists),
            "avg_tap_duration": avg(all_durations),
            "tap_duration_std": std(all_durations),
            # inter-tap gap = the drone's recenter pause between taps
            "avg_inter_gap":    avg(all_gaps),
            "inter_gap_std":    std(all_gaps),
            "avg_end_settle":   avg(all_settles),
            "end_settle_std":   std(all_settles),
            "_tap_counts":      [len(t) for t in tap_lists],
        }
    return model


def print_model(model):
    print("\n" + "═" * 68)
    print("  learned model  (tap-based)")
    print("═" * 68)
    for seg_type, d in model.items():
        n   = d["demonstrations"]
        seq = d["action_sequence"]
        corrections = [a for a in seq if a.startswith("correction_")]
        progress    = [a for a in seq if not a.startswith("correction_")]
        print(f"\n  {seg_type}  ({n} demo(s),  tap counts per demo: {d['_tap_counts']})")
        print(f"    action sequence : {seq}")
        print(f"    progress taps   : {len(progress)}  {progress}")
        if corrections:
            print(f"    correction taps : {len(corrections)}  {corrections}  ← stabilising nudges")
        else:
            print(f"    correction taps : 0  (none recorded)")
        print(f"    tap duration    : {d['avg_tap_duration']:.4f}s  ±{d['tap_duration_std']:.4f}")
        print(f"    inter-tap gap   : {d['avg_inter_gap']:.4f}s  ±{d['inter_gap_std']:.4f}  ← recenter pause")
        print(f"    end settle      : {d['avg_end_settle']:.4f}s  ±{d['end_settle_std']:.4f}")
        if n < 3:
            print(f"    note: only {n} demo(s) — 3+ gives more reliable averages")
    print("\n" + "═" * 68)


def main():
    recordings = load_recordings()
    if not recordings:
        return

    collected = defaultdict(lambda: {
        "tap_lists":       [],
        "all_gaps":        [],
        "all_end_settles": [],
        "all_durations":   [],
    })

    for rec in recordings:
        segments = split_into_segments(rec)
        print(f"\n  {rec['recorded_at']}:")
        for seg in segments:
            taps, gaps, end_settle = extract_taps(seg["events"], seg["marker_t"])
            if not taps:
                print(f"    {seg['segment_type']:20s}  (no taps recorded)")
                continue

            collected[seg["segment_type"]]["tap_lists"].append(taps)
            collected[seg["segment_type"]]["all_gaps"].extend(gaps)
            collected[seg["segment_type"]]["all_end_settles"].append(end_settle)
            collected[seg["segment_type"]]["all_durations"].extend(
                [t["duration"] for t in taps]
            )

            actions = [t["action"] for t in taps]
            avg_gap = round(sum(gaps) / len(gaps), 3) if gaps else 0
            print(f"    {seg['segment_type']:20s}  taps={len(taps)}"
                  f"  actions={actions}"
                  f"  avg_gap={avg_gap:.3f}s  settle={end_settle:.3f}s")

    model = compute_model(dict(collected))
    print_model(model)

    with open(MODEL_FILE, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\n  model saved to {MODEL_FILE}")
    print("  run learned_runner.py to fly using these timings.")


if __name__ == "__main__":
    main()