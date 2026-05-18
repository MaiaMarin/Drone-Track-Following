"""
trim_tuner.py
-------------
interactive trim tuning for codrone edu.
lets you adjust roll and pitch trim in small steps, run hover tests,
and save the best values — without restarting between attempts.

trim value meaning
  [roll, pitch]
  roll  : negative = left,   positive = right
  pitch : negative = backward, positive = forward

  range is roughly -100 to +100 per axis.
  the factory default is [0, 0].
  your drone had [0, -50] stored — that is a heavy backward bias.

usage
  python trim_tuner.py
  follow the menu. start with small steps (5–10), hover, adjust, repeat.
"""

from codrone_edu.drone import Drone
import time


# ── how long each hover test lasts ───────────────────────────────────────────
HOVER_SECONDS = 4

# ── step sizes available in the menu ─────────────────────────────────────────
STEPS = [2, 5, 10, 20]


def print_trim(drone):
    try:
        t = drone.get_trim()
        print(f"  current trim  ->  roll: {t[0]:+d}   pitch: {t[1]:+d}")
        return t
    except Exception as e:
        print(f"  could not read trim: {e}")
        return [0, 0]


def apply_trim(drone, roll, pitch):
    """clamp to [-100, 100] and apply."""
    roll  = max(-100, min(100, roll))
    pitch = max(-100, min(100, pitch))
    drone.set_trim(roll, pitch)
    time.sleep(0.3)
    print(f"  applied trim  ->  roll: {roll:+d}   pitch: {pitch:+d}")
    return [roll, pitch]


def hover_test(drone):
    print(f"\n  taking off — will hover for {HOVER_SECONDS} s then land.")
    print("  watch: does it drift forward/back/left/right?")
    drone.takeoff()
    time.sleep(2)           # settle after takeoff burst
    drone.hover(HOVER_SECONDS)
    drone.land()
    time.sleep(3)
    print("  landed.")


def menu(current_roll, current_pitch, step):
    print()
    print("─" * 48)
    print(f"  roll: {current_roll:+d}    pitch: {current_pitch:+d}    step: {step}")
    print("─" * 48)
    print("  pitch adjust")
    print("    w  — pitch +step  (corrects backward drift)")
    print("    s  — pitch -step  (corrects forward drift)")
    print("  roll adjust")
    print("    d  — roll  +step  (corrects left drift)")
    print("    a  — roll  -step  (corrects right drift)")
    print("  other")
    print("    t  — run hover test")
    print("    r  — reset trim to [0, 0]")
    print("    c  — change step size")
    print("    p  — print current trim from drone")
    print("    q  — quit and save nothing  (trim already applied to drone)")
    print("─" * 48)
    return input("  choice: ").strip().lower()


def change_step():
    print(f"  available steps: {STEPS}")
    raw = input("  enter step size: ").strip()
    try:
        v = int(raw)
        if v < 1 or v > 100:
            print("  must be 1-100.")
            return None
        return v
    except ValueError:
        print("  not a number.")
        return None


def main():
    drone = Drone()

    print("pairing...")
    drone.pair()
    time.sleep(2)
    print("paired.\n")

    t    = print_trim(drone)
    roll  = t[0]
    pitch = t[1]
    step  = 5

    print()
    print("tip: your pre-reset trim was [0, -50].")
    print("     pitch -50 means the drone was told to lean backward constantly.")
    print("     if it still drifts backward a little, raise pitch (press w).")
    print("     if it drifts forward, lower pitch (press s).")
    print("     start with small steps (5) and run a hover test after each change.\n")

    while True:
        choice = menu(roll, pitch, step)

        if choice == "w":
            pitch += step
            [roll, pitch] = apply_trim(drone, roll, pitch)

        elif choice == "s":
            pitch -= step
            [roll, pitch] = apply_trim(drone, roll, pitch)

        elif choice == "d":
            roll += step
            [roll, pitch] = apply_trim(drone, roll, pitch)

        elif choice == "a":
            roll -= step
            [roll, pitch] = apply_trim(drone, roll, pitch)

        elif choice == "t":
            hover_test(drone)

        elif choice == "r":
            drone.reset_trim()
            time.sleep(0.3)
            t     = print_trim(drone)
            roll  = t[0]
            pitch = t[1]

        elif choice == "c":
            new_step = change_step()
            if new_step is not None:
                step = new_step

        elif choice == "p":
            t     = print_trim(drone)
            roll  = t[0]
            pitch = t[1]

        elif choice == "q":
            break

        else:
            print("  unknown key — use w/s/a/d/t/r/c/p/q")

    # ── final state ───────────────────────────────────────────────────────────
    print()
    print_trim(drone)
    print("  trim is saved on the drone — it will persist until you reset it.")
    print("  note these values down so you can restore them if needed:")
    print(f"    drone.set_trim({roll}, {pitch})")
    drone.close()
    print("done.")


if __name__ == "__main__":
    main()