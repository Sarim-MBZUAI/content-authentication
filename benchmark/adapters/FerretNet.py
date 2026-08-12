#!/usr/bin/env python
"""Benchmark adapter for FerretNet (2025) - local pixel dependency detector.

Official code:    detectors/2025/FerretNet/code (demo_image.py + configs/Test.yaml)
Official weights: detectors/2025/FerretNet/code/checkpoints/4cls_ckpt/ferretnet-b-median-3.pth
Model: Ferret (dim=96, depths=[2,2], lpd_func=median, window_size=3) built via the
       repo's src.model.get_model with configs/Test.yaml.
Preprocessing (Test.yaml test_transforms): CenterCrop((256,256)) -> ToTensor ->
       Normalize(CLIP stats).
Score: sigmoid(logit) = P(fake); higher = fake (repo convention: "This is a
       synthetic image" when sigmoid(output) > 0.5).
"""
import argparse
import csv
import os
import sys

CODE = "/shared/home/sarim.hashmi/usenix/detectors/2025/FerretNet/code"
CFG_PATH = os.path.join(CODE, "configs/Test.yaml")
CKPT = os.path.join(CODE, "checkpoints/4cls_ckpt/ferretnet-b-median-3.pth")
sys.path.insert(0, CODE)

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from util import get_cfg, load_state_dict  # official code
from src.model import get_model            # official code

BATCH_SIZE = 32


def parse_args():
    p = argparse.ArgumentParser(description="FerretNet benchmark adapter")
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

    # Test.yaml: CenterCrop(256,256) + ToTensor + Normalize(CLIP stats)
    transform = transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711],
                             inplace=True),
    ])

    cfg = get_cfg(cfg_path=CFG_PATH, mode="test")
    model = get_model(cfg)
    load_state_dict(model, CKPT)  # repo loader (reads ckpt['model'], strips 'module.')
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
                    print(f"[FerretNet] error scoring {rows[i]['path']}: {e}", file=sys.stderr)

    idx_buf, ten_buf = [], []
    for i, row in enumerate(rows):
        try:
            img = Image.open(row["path"]).convert("RGB")
            ten_buf.append(transform(img))
            idx_buf.append(i)
        except Exception as e:
            print(f"[FerretNet] error loading {row['path']}: {e}", file=sys.stderr)
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
    print(f"[FerretNet] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
