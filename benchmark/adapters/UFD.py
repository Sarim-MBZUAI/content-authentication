#!/usr/bin/env python
"""Benchmark adapter: UniversalFakeDetect (Ojha et al., CVPR 2023).

CLIP ViT-L/14 backbone + linear head (fc_weights.pth). Score = sigmoid(fc(CLIP feat))
= P(fake); higher = more likely fake (repo convention already matches).
"""
import argparse
import csv
import os
import sys

REPO = "detectors/2023/UFD"
CODE = os.path.join(REPO, "code")
CKPT = os.path.join(REPO, "weights", "fc_weights.pth")
# Official OpenAI CLIP ViT-L/14 checkpoint. If $CLIP_VIT_L14 points to a local copy it is
# symlinked into the cache dir that the repo's bundled clip.load() checks; otherwise
# clip.load() downloads it into ~/.cache/clip.
CLIP_BACKBONE = os.environ.get("CLIP_VIT_L14")

sys.path.insert(0, CODE)


def ensure_clip_cache():
    cache_dir = os.path.expanduser("~/.cache/clip")
    target = os.path.join(cache_dir, "ViT-L-14.pt")
    if CLIP_BACKBONE and not os.path.lexists(target):
        os.makedirs(cache_dir, exist_ok=True)
        os.symlink(os.path.abspath(CLIP_BACKBONE), target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    import torchvision.transforms as transforms
    from PIL import Image

    ensure_clip_cache()
    from models import get_model  # repo module

    device = torch.device(args.device)
    model = get_model("CLIP:ViT-L/14")
    state_dict = torch.load(CKPT, map_location="cpu")
    model.fc.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Transform from code/validate.py (arch = CLIP -> clip stats, CenterCrop only)
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[: args.limit]

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    BATCH = 16
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generator", "gen_year", "label", "path", "score"])
        batch_imgs, batch_rows = [], []

        def flush():
            if not batch_rows:
                return
            try:
                with torch.no_grad():
                    x = torch.stack(batch_imgs).to(device)
                    scores = model(x).sigmoid().flatten().tolist()
            except Exception as e:
                print(f"batch failed: {e}", file=sys.stderr)
                scores = [""] * len(batch_rows)
            for r, s in zip(batch_rows, scores):
                writer.writerow([r["generator"], r["gen_year"], r["label"], r["path"], s])
            batch_imgs.clear()
            batch_rows.clear()

        for row in rows:
            try:
                img = Image.open(row["path"]).convert("RGB")
                batch_imgs.append(transform(img))
                batch_rows.append(row)
            except Exception as e:
                print(f"error on {row['path']}: {e}", file=sys.stderr)
                flush()
                writer.writerow([row["generator"], row["gen_year"], row["label"], row["path"], ""])
                continue
            if len(batch_rows) >= BATCH:
                flush()
        flush()

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
