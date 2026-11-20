"""Generate a synthetic 'robot arm camera feed' of small PNG frames.

Simulates the real problem: a camera recording at a fixed rate produces long
runs of near-duplicate frames (the arm barely moves), punctuated by a few
genuinely distinct scenes. Labeling every frame wastes budget.

Scenes: a few distinct 'workcell' backgrounds, each with a colored shape
(the 'object') drifting a few pixels per frame. Consecutive frames within a
scene are near-duplicates on purpose.
"""

import argparse
import math
import os
import random

from PIL import Image, ImageDraw

SIZE = 128

# (bg_color, accent_color, shape, n_frames) - long dwell on some scenes,
# short on others, like a real teleop recording.
SCENES = [
    ((40, 44, 52), (231, 76, 60), "circle", 22),
    ((40, 44, 52), (231, 76, 60), "circle", 0),  # placeholder, unused
    ((34, 49, 63), (46, 204, 113), "square", 8),
    ((60, 42, 33), (241, 196, 15), "triangle", 24),
    ((45, 52, 54), (155, 89, 182), "circle", 6),
    ((25, 42, 86), (26, 188, 156), "square", 12),
]


def draw_frame(bg, accent, shape, t, jitter_rng):
    img = Image.new("RGB", (SIZE, SIZE), bg)
    d = ImageDraw.Draw(img)

    # static scene furniture: a 'table edge' and a 'gripper' bar
    d.rectangle([0, int(SIZE * 0.78), SIZE, SIZE],
                fill=tuple(min(255, c + 25) for c in bg))
    d.rectangle([int(SIZE * 0.44), 0, int(SIZE * 0.56), int(SIZE * 0.18)],
                fill=(120, 124, 130))

    # object drifts slowly: ~1.5 px/frame plus tiny sensor jitter
    cx = SIZE * 0.3 + t * 1.5 + jitter_rng.uniform(-0.8, 0.8)
    cy = SIZE * 0.55 + 6 * math.sin(t * 0.25) + jitter_rng.uniform(-0.8, 0.8)
    r = 16

    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=accent)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=accent)
    else:  # triangle
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)],
                  fill=accent)

    # mild per-frame brightness flicker, like auto-exposure
    flicker = jitter_rng.randint(-4, 4)
    if flicker:
        img = Image.eval(img, lambda v: max(0, min(255, v + flicker)))
    return img


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic camera feed")
    ap.add_argument("--out", default="data/frames")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    idx = 0
    for bg, accent, shape, n in SCENES:
        for t in range(n):
            img = draw_frame(bg, accent, shape, t, rng)
            img.save(os.path.join(args.out, f"frame_{idx:03d}.png"))
            idx += 1
    print(f"Wrote {idx} frames to {args.out}")


if __name__ == "__main__":
    main()
