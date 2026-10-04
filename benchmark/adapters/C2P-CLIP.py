#!/usr/bin/env python
"""Benchmark adapter for C2P-CLIP (AAAI 2025).

Official code:    detectors/2025/C2P-CLIP/code (scripts/inference.py)
Official weights: detectors/2025/C2P-CLIP/weights/C2P_CLIP_release_20240901.pth
Model: CLIP ViT-L/14 (transformers CLIPModel, vision tower only) + linear head.
Preprocessing (repo test transform, loadSize=cropSize=224, no_resize/no_crop=False):
  translate_duplicate(img, 224) -> CenterCrop(224) -> ToTensor -> CLIP normalize.
Score: sigmoid(logit) = P(fake); higher = fake (repo convention, fake label = 1).
"""
import argparse
import csv
import os
import sys

CODE = "detectors/2025/C2P-CLIP/code"
CKPT = "detectors/2025/C2P-CLIP/weights/C2P_CLIP_release_20240901.pth"
sys.path.insert(0, os.path.join(CODE, "scripts"))
sys.path.insert(0, CODE)

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from inference import C2P_CLIP          # official model class (code/scripts/inference.py)
from data.datasets import translate_duplicate  # official test-time resize helper

BATCH_SIZE = 16


def parse_args():
    p = argparse.ArgumentParser(description="C2P-CLIP benchmark adapter")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def read_manifest(path, limit):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def main():
    args = parse_args()
    rows = read_manifest(args.manifest, args.limit)
    device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.Lambda(lambda img: translate_duplicate(img, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    model = C2P_CLIP(name="openai/clip-vit-large-patch14", num_classes=1)
    state_dict = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    scores = [None] * len(rows)

    def flush(idx_buf, ten_buf):
        if not ten_buf:
            return
        batch = torch.stack(ten_buf).to(device)
        try:
            with torch.no_grad():
                out = model(batch).sigmoid().flatten().tolist()
            for i, s in zip(idx_buf, out):
                scores[i] = float(s)
        except Exception:
            for i, t in zip(idx_buf, ten_buf):
                try:
                    with torch.no_grad():
                        s = model(t.unsqueeze(0).to(device)).sigmoid().flatten().item()
                    scores[i] = float(s)
                except Exception as e:
                    print(f"[C2P-CLIP] error scoring {rows[i]['path']}: {e}", file=sys.stderr)

    idx_buf, ten_buf = [], []
    for i, row in enumerate(rows):
        try:
            with open(row["path"], "rb") as f:
                img = Image.open(f).convert("RGB")
            ten_buf.append(transform(img))
            idx_buf.append(i)
        except Exception as e:
            print(f"[C2P-CLIP] error loading {row['path']}: {e}", file=sys.stderr)
        if len(ten_buf) >= BATCH_SIZE:
            flush(idx_buf, ten_buf)
            idx_buf, ten_buf = [], []
    flush(idx_buf, ten_buf)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generator", "gen_year", "label", "path", "score"])
        for row, s in zip(rows, scores):
            w.writerow([row["generator"], row["gen_year"], row["label"], row["path"],
                        "" if s is None else s])
    print(f"[C2P-CLIP] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
