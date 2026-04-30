"""
test_vision.py
--------------
tests the vision pipeline on a static image — no realsense camera needed.
drop your track photo in the same folder and update IMAGE_PATH below.

run with:
    python test_vision.py

press any key to cycle through the debug windows, q to quit.
"""

import cv2
import numpy as np
import sys
import os

# ── point this at your track photo ───────────────────────────────────────────
IMAGE_PATH = "track.jpeg"   # rename/replace with your actual filename


def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"image not found: {IMAGE_PATH}")
        print("put your track photo in the same folder and update IMAGE_PATH.")
        sys.exit(1)

    # import vision functions (make sure vision.py is in the same folder)
    try:
        from vision import (
            detect_track_mask,
            get_track_skeleton,
            detect_drone_position,
            get_lookahead_direction,
            draw_debug,
            detect_yellow_landing,
        )
    except ImportError as e:
        print(f"could not import vision.py: {e}")
        print("make sure vision.py is in the same folder as this script.")
        sys.exit(1)

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"opencv could not read: {IMAGE_PATH}")
        sys.exit(1)

    # resize if the image is huge (phone photos are often 4000px wide)
    max_width = 900
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img   = cv2.resize(img, (max_width, int(h * scale)))
        print(f"resized image to {img.shape[1]}x{img.shape[0]}")

    print("running vision pipeline on static image...")

    # ── run each vision step ──────────────────────────────────────────────────

    track_mask = detect_track_mask(img)
    print(f"  track mask: {cv2.countNonZero(track_mask)} non-zero pixels")

    skeleton = get_track_skeleton(track_mask)
    print(f"  skeleton:   {len(skeleton)} ordered path points")

    drone_pos = detect_drone_position(img)
    print(f"  drone pos:  {drone_pos}")

    direction = get_lookahead_direction(drone_pos, skeleton)
    if direction:
        dx, dy, lp, cp = direction
        print(f"  direction:  dx={dx:+.3f}  dy={dy:+.3f}  lookahead={lp}  closest={cp}")
    else:
        print("  direction:  None (drone or skeleton not detected)")

    yellow_mask, landing = detect_yellow_landing(img)
    print(f"  yellow pad: {'detected' if landing else 'not detected'}")

    # ── build debug frame ─────────────────────────────────────────────────────

    debug = draw_debug(img, skeleton, drone_pos, direction, landing)

    # overlay skeleton point count and drone position on the image
    cv2.putText(debug, f"skeleton pts: {len(skeleton)}", (10, img.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    if drone_pos:
        cv2.putText(debug, f"drone: {drone_pos}", (10, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 1)

    # ── show results ──────────────────────────────────────────────────────────

    # skeleton drawn on a black canvas so it's easy to inspect
    skel_canvas = np.zeros_like(img)
    if skeleton:
        for pt in skeleton:
            cv2.circle(skel_canvas, pt, 2, (0, 200, 100), -1)

    print("\nshowing windows — press q to quit.")

    windows = [
        ("original",        img),
        ("track mask",      cv2.cvtColor(track_mask, cv2.COLOR_GRAY2BGR)),
        ("skeleton",        skel_canvas),
        ("yellow mask",     cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)),
        ("debug overlay",   debug),
    ]

    for title, frame in windows:
        cv2.imshow(title, frame)

    # wait for q
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()

    # ── quick diagnostic summary ──────────────────────────────────────────────

    print("\n--- diagnostic summary ---")
    print(f"track pixels found : {'yes' if cv2.countNonZero(track_mask) > 500 else 'NO — adjust hsv range in detect_track_mask()'}")
    print(f"skeleton points    : {'ok' if len(skeleton) > 10 else 'too few — track mask may be empty or noisy'}")
    print(f"drone detected     : {'yes' if drone_pos else 'NO — adjust threshold in detect_drone_position()'}")
    print(f"direction vector   : {'ok' if direction else 'None — needs drone + skeleton to work'}")
    print(f"yellow pad         : {'detected' if landing else 'not detected — check hsv range or pad visibility'}")


if __name__ == "__main__":
    main()