"""frame-picker: embedding-based data curation for a labeling budget.

Given a folder of frames, embed each one with a pretrained ResNet-18
(ImageNet weights, CPU), then pick the K most diverse frames with greedy
farthest-point sampling (coreset selection - the core idea behind Lightly's
Selection product) and compare against naive every-Nth-frame sampling.

Outputs both selections, a diversity score for each, and a grid image
showing selected vs skipped frames.
"""

import argparse
import glob
import os

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageOps


def load_images(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.png"))
                   + glob.glob(os.path.join(folder, "*.jpg")))
    if not paths:
        raise SystemExit(f"No images found in {folder}")
    return paths, [Image.open(p).convert("RGB") for p in paths]


def embed(images, batch_size=16):
    """Embed images with ResNet-18 (penultimate layer, 512-d), CPU only."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()

    tf = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embs = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack([tf(im) for im in images[i:i + batch_size]])
            embs.append(model(batch).numpy())
    embs = np.concatenate(embs, axis=0)
    # L2-normalize so distances are cosine-like
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    return embs


def farthest_point_sampling(embs, k, start=0):
    """Greedy coreset selection: repeatedly pick the point farthest from
    the already-selected set (max-min distance)."""
    n = len(embs)
    selected = [start]
    min_dist = np.linalg.norm(embs - embs[start], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(embs - embs[nxt], axis=1))
    return sorted(selected)


def every_nth(n_total, k):
    idx = np.linspace(0, n_total - 1, k).round().astype(int)
    return sorted(set(int(i) for i in idx))


def diversity_score(embs, selected):
    """Mean pairwise embedding distance within the selection (higher =
    more diverse = fewer redundant labels)."""
    sub = embs[selected]
    dists = []
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            dists.append(np.linalg.norm(sub[i] - sub[j]))
    return float(np.mean(dists))


def min_gap(embs, selected):
    """Smallest pairwise distance in the selection - exposes near-dupes."""
    sub = embs[selected]
    best = np.inf
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            best = min(best, float(np.linalg.norm(sub[i] - sub[j])))
    return best


def save_grid(images, fps_sel, nth_sel, out_path, cols=12, thumb=72):
    """Grid of all frames; green border = picked by FPS (coreset),
    blue corner = picked by every-Nth, dimmed = skipped by FPS."""
    fps_set, nth_set = set(fps_sel), set(nth_sel)
    rows = (len(images) + cols - 1) // cols
    pad = 4
    W = cols * (thumb + pad) + pad
    H = rows * (thumb + pad) + pad + 26
    grid = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(grid)
    d.text((pad, 4), "green border = coreset (FPS) pick   "
                     "blue dot = every-Nth pick   dim = skipped",
           fill=(220, 220, 220))

    for i, im in enumerate(images):
        th = im.resize((thumb, thumb))
        if i not in fps_set:
            th = ImageOps.autocontrast(th.point(lambda v: v // 3))
        x = pad + (i % cols) * (thumb + pad)
        y = 26 + pad + (i // cols) * (thumb + pad)
        grid.paste(th, (x, y))
        if i in fps_set:
            dd = ImageDraw.Draw(grid)
            dd.rectangle([x - 2, y - 2, x + thumb + 1, y + thumb + 1],
                         outline=(46, 204, 113), width=3)
        if i in nth_set:
            dd = ImageDraw.Draw(grid)
            dd.ellipse([x + thumb - 12, y + 4, x + thumb - 4, y + 12],
                       fill=(52, 152, 219))
    grid.save(out_path)


def main():
    ap = argparse.ArgumentParser(description="Pick the K most diverse frames")
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--k", type=int, default=6,
                    help="labeling budget (frames to select)")
    ap.add_argument("--grid", default="selection_grid.png")
    args = ap.parse_args()

    paths, images = load_images(args.frames)
    names = [os.path.basename(p) for p in paths]
    print(f"Embedding {len(images)} frames with ResNet-18 (CPU)...")
    embs = embed(images)

    fps_sel = farthest_point_sampling(embs, args.k)
    nth_sel = every_nth(len(images), args.k)

    print(f"\nLabeling budget K = {args.k} of {len(images)} frames\n")
    print("Every-Nth sampling (naive):")
    print("  " + ", ".join(names[i] for i in nth_sel))
    print(f"  diversity (mean pairwise dist): {diversity_score(embs, nth_sel):.4f}")
    print(f"  closest pair distance:          {min_gap(embs, nth_sel):.4f}")
    print("\nCoreset selection (greedy farthest-point sampling):")
    print("  " + ", ".join(names[i] for i in fps_sel))
    print(f"  diversity (mean pairwise dist): {diversity_score(embs, fps_sel):.4f}")
    print(f"  closest pair distance:          {min_gap(embs, fps_sel):.4f}")

    save_grid(images, fps_sel, nth_sel, args.grid)
    print(f"\nSaved selection grid to {args.grid}")


if __name__ == "__main__":
    main()
