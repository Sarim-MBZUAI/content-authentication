#!/usr/bin/env python
"""Benchmark adapter for DGS-Net (2026).

Official code:    /shared/home/sarim.hashmi/usenix/detectors/2026/DGS-Net/code
Official weights: .../weights/DGS-Net/checkpoints/model_epoch_step2.pth (final, step2)
                  (step1 checkpoint is loaded internally by CLIPModel.__init__ via
                   the relative path ./checkpoints/model_epoch_step1.pth, so we chdir
                   into the weights dir before constructing the model.)

Score = sigmoid(img_logit), higher = more likely FAKE (repo convention: label 1 = fake,
see validate.py: model(in_tens)[0].sigmoid()).
Preprocessing = repo's CustomDataset test transform:
PatchSelectionTransform(32, 49) -> ToTensor -> Normalize(ImageNet).
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = "/shared/home/sarim.hashmi/usenix/detectors/2026/DGS-Net"
CODE = os.path.join(ROOT, "code")
WEIGHTS_DIR = os.path.join(ROOT, "weights", "DGS-Net")
CKPT = os.path.join(WEIGHTS_DIR, "checkpoints", "model_epoch_step2.pth")

sys.path.insert(0, CODE)


def seed_all(seed=100):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def build_model(device):
    # CLIPModel.__init__ loads "./checkpoints/model_epoch_step1.pth" (relative).
    os.chdir(WEIGHTS_DIR)
    from models.clip_models import CLIPModel
    from options.test_options import TestOptions

    argv_bak = sys.argv
    try:
        sys.argv = ["DGS-Net", "--gpu_ids", "-1"]  # keep opt on CPU-safe defaults
        opt = TestOptions().parse(print_options=False)
    finally:
        sys.argv = argv_bak

    # CLIPModel.__init__ internally calls torch.load without map_location on a
    # CUDA-saved checkpoint; force CPU mapping while constructing (repo untouched).
    _orig_load = torch.load

    def _cpu_load(*a, **k):
        k.setdefault("map_location", "cpu")
        return _orig_load(*a, **k)

    torch.load = _cpu_load
    try:
        model = CLIPModel(opt)
        state = torch_load(CKPT)
    finally:
        torch.load = _orig_load
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def build_transform():
    from data.PatchSelectionTransform import PatchSelectionTransform

    # CustomDataset test transform (no flip at test time).
    return transforms.Compose([
        PatchSelectionTransform(patch_size=32, num_patches=49),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def score_batch(model, device, tensors):
    x = torch.stack(tensors).to(device)
    logits = model(x)[0]
    return torch.sigmoid(logits).flatten().cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest = os.path.abspath(args.manifest)
    out_path = os.path.abspath(args.out)
    device = torch.device(args.device)

    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    seed_all(100)
    model = build_model(device)
    transform = build_transform()

    scores = [None] * len(rows)
    batch_size = 32 if device.type == "cuda" else 4

    pending_idx, pending_tensors = [], []

    def flush():
        if not pending_tensors:
            return
        try:
            out = score_batch(model, device, pending_tensors)
            for i, s in zip(pending_idx, out):
                scores[i] = s
        except Exception as exc:  # fall back to per-image scoring
            print(f"[warn] batch failed ({exc}); retrying per image", file=sys.stderr)
            for i, t in zip(pending_idx, pending_tensors):
                try:
                    scores[i] = score_batch(model, device, [t])[0]
                except Exception as exc2:
                    print(f"[warn] scoring failed for row {i}: {exc2}", file=sys.stderr)
        pending_idx.clear()
        pending_tensors.clear()

    for i, row in enumerate(rows):
        try:
            img = Image.open(row["path"]).convert("RGB")
            pending_tensors.append(transform(img))
            pending_idx.append(i)
        except Exception as exc:
            print(f"[warn] failed to load {row['path']}: {exc}", file=sys.stderr)
        if len(pending_tensors) >= batch_size:
            flush()
    flush()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generator", "gen_year", "label", "path", "score"])
        for row, score in zip(rows, scores):
            writer.writerow([
                row["generator"], row["gen_year"], row["label"], row["path"],
                "" if score is None else float(score),
            ])
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
