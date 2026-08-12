#!/usr/bin/env python
"""Benchmark adapter for DDA - Dual Data Alignment (2025).

Official code:    detectors/2025/DualDataAlignment/code/Inference/inference.py
Official weights: detectors/2025/DualDataAlignment/weights/DDA_ckpt.pth
Model: DINOv2 ViT-L/14 + LoRA (rank 8, alpha 1) + linear head
       (DINOv2ModelWithLoRA from the repo; backbone fetched via torch.hub with
        TORCH_HOME pointed at the benchmark cache, then fully overwritten by the
        official checkpoint which contains all backbone + LoRA + fc weights).
Preprocessing (repo test transform): CenterCrop(336) -> ToTensor -> CLIP normalize.
Score: sigmoid(logit) = P(fake); higher = fake (repo convention, fake label = 1).
"""
import argparse
import csv
import os
import sys

os.environ.setdefault("TORCH_HOME", "/shared/home/sarim.hashmi/usenix/benchmark/torch_home")

CODE = "/shared/home/sarim.hashmi/usenix/detectors/2025/DualDataAlignment/code/Inference"
CKPT = "/shared/home/sarim.hashmi/usenix/detectors/2025/DualDataAlignment/weights/DDA_ckpt.pth"
sys.path.insert(0, CODE)

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from models.dinov2_models_lora import DINOv2ModelWithLoRA  # official code

BATCH_SIZE = 8


def parse_args():
    p = argparse.ArgumentParser(description="DDA benchmark adapter")
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

    # repo's Inference/inference.py test_transform
    transform = transforms.Compose([
        transforms.CenterCrop(336),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    model = DINOv2ModelWithLoRA(name="dinov2_vitl14", lora_rank=8, lora_alpha=1,
                                lora_targets=None)
    checkpoint = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
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
                    print(f"[DDA] error scoring {rows[i]['path']}: {e}", file=sys.stderr)

    idx_buf, ten_buf = [], []
    for i, row in enumerate(rows):
        try:
            img = Image.open(row["path"]).convert("RGB")
            ten_buf.append(transform(img))
            idx_buf.append(i)
        except Exception as e:
            print(f"[DDA] error loading {row['path']}: {e}", file=sys.stderr)
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
    print(f"[DDA] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
