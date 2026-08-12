#!/usr/bin/env python
"""Benchmark adapter for SICA (2026).

Official code:    /shared/home/sarim.hashmi/usenix/detectors/2026/SICA/code (inference.py)
Official weights: .../weights/SICA_for_MCC_weights/sica_weight_full.pth

Model: CLIP_LORA_PURE (ViT-L/14 backbone + LoRA parametrization + linear head).
Score = model(image)["pred_label"] = sigmoid(logit), higher = more likely FAKE.
Preprocessing = model.preprocess (the CLIP test transform kept by the model class),
exactly as done in the repo's inference.py.
"""
import argparse
import csv
import os
import sys

import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = "/shared/home/sarim.hashmi/usenix/detectors/2026/SICA"
CODE = os.path.join(ROOT, "code")
CKPT = os.path.join(ROOT, "weights", "SICA_for_MCC_weights", "sica_weight_full.pth")

sys.path.insert(0, CODE)


def build_model(device):
    from model import CLIP_LORA_PURE

    model = CLIP_LORA_PURE(name="ViT-L/14", num_classes=1, tune_mode="lora")
    checkpoint = torch.load(CKPT, map_location="cpu", weights_only=False)
    # Same checkpoint-format handling as the repo's inference.py.
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def score_batch(model, device, tensors):
    x = torch.stack(tensors).to(device)
    out = model(image=x)
    return out["pred_label"].flatten().cpu().tolist()


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

    model = build_model(device)
    preprocess = model.preprocess

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
        except Exception as exc:
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
            pending_tensors.append(preprocess(img))
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
