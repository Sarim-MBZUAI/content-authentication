#!/usr/bin/env python
"""Benchmark adapter for D3 (CVPR 2025) - Discrepancy Deepfake Detector.

Official code:    detectors/2025/D3/code (validate_for_robustness.py is the working entry)
Official weights: detectors/2025/D3/weights/classifier.pth (attention head only;
                  CLIP ViT-L/14 backbone is frozen/off-the-shelf, downloaded by the
                  repo's bundled clip module to ~/.cache/clip on first use).
Model: CLIPModelShuffleAttentionPenultimateLayer("ViT-L/14", shuffle_times=1,
       original_times=1, patch_size=[14])  (exact eval configuration of the repo).
Preprocessing (repo eval transform): Resize((224,224)) -> ToTensor -> CLIP normalize.
Score: sigmoid(logit) = P(fake); higher = fake (repo convention, fake label = 1).
Note: the model shuffles patches with torch.randperm; seed is fixed to 418 as in
      the repo's validate_for_robustness.py for reproducibility.
"""
import argparse
import csv
import os
import random
import sys

CODE = "/shared/home/sarim.hashmi/usenix/detectors/2025/D3/code"
CKPT = "/shared/home/sarim.hashmi/usenix/detectors/2025/D3/weights/classifier.pth"
sys.path.insert(0, CODE)

import types

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# The repo's models/__init__.py imports its MOCO baselines, which need a
# timm API removed in current timm (timm.models.layers.helpers).  Register the
# "models" package without executing that __init__.py so we can import the
# CLIP-based detector modules (unmodified official files) directly.
_models_pkg = types.ModuleType("models")
_models_pkg.__path__ = [os.path.join(CODE, "models")]
sys.modules.setdefault("models", _models_pkg)

from models.clip_models import CLIPModelShuffleAttentionPenultimateLayer  # official

BATCH_SIZE = 16


def set_seed(seed=418):  # same seed as repo's validate_for_robustness.py
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    p = argparse.ArgumentParser(description="D3 benchmark adapter")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def read_manifest(path, limit):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def predict(model, batch):
    with torch.no_grad():
        out = model(batch)
    if out.shape[-1] == 2:  # repo's validate() convention
        out = out[:, 0]
    return out.sigmoid().flatten().tolist()


def main():
    args = parse_args()
    set_seed(418)
    rows = read_manifest(args.manifest, args.limit)
    device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    granularity = 14
    model = CLIPModelShuffleAttentionPenultimateLayer(
        "ViT-L/14", shuffle_times=1, original_times=1, patch_size=[granularity])
    state_dict = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.attention_head.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    scores = [None] * len(rows)

    def flush(idx_buf, ten_buf):
        if not ten_buf:
            return
        batch = torch.stack(ten_buf).to(device)
        try:
            out = predict(model, batch)
            for i, s in zip(idx_buf, out):
                scores[i] = float(s)
        except Exception:
            for i, t in zip(idx_buf, ten_buf):
                try:
                    scores[i] = float(predict(model, t.unsqueeze(0).to(device))[0])
                except Exception as e:
                    print(f"[D3] error scoring {rows[i]['path']}: {e}", file=sys.stderr)

    idx_buf, ten_buf = [], []
    for i, row in enumerate(rows):
        try:
            img = Image.open(row["path"]).convert("RGB")
            ten_buf.append(transform(img))
            idx_buf.append(i)
        except Exception as e:
            print(f"[D3] error loading {row['path']}: {e}", file=sys.stderr)
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
    print(f"[D3] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
