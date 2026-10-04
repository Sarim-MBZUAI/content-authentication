#!/usr/bin/env python
"""Benchmark adapter for IAPL (2026).

Official code:    detectors/2026/IAPL/code (main.py / engine.py)
Official weights: .../weights/checkpoint_best_acc_progan.pth

The model is a prompt-learned CLIP ViT-L/14 (MaPLe-style adapters + DCT-conditioned
context). IAPL's headline mode does test-time adaptation (test_time.py); here we run
the NON-ADAPTIVE forward pass of the base prompt-learned model, i.e. the same path
used by main.py --eval / engine.evaluate() (model(images) in eval mode, then sigmoid),
because TTA updates model state per test batch and depends on batch composition.

Model hyperparameters are taken from the argparse Namespace stored inside the
official checkpoint (checkpoint['args']). The repo hardcodes the CLIP backbone path
'/Path/to/ViT-L-14.pt' inside models/clip_models.load_clip_to_cpu; we monkeypatch
that function (adapter-side only, repo untouched) to point at the official OpenAI
ViT-L/14 weights.

Score = sigmoid(logit), higher = more likely FAKE (engine.evaluate convention).
Preprocessing = repo test transform (utils/dataset.py):
Resize((256,256)) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet).
"""
import argparse
import csv
import os
import sys

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = "detectors/2026/IAPL"
CODE = os.path.join(ROOT, "code")
CKPT = os.path.join(ROOT, "weights", "checkpoint_best_acc_progan.pth")
CLIP_VIT_L14 = os.path.expanduser("~/.cache/clip/ViT-L-14.pt")

sys.path.insert(0, CODE)


def build_model(device):
    checkpoint = torch.load(CKPT, map_location="cpu", weights_only=False)
    margs = checkpoint["args"]
    margs.tta = False  # non-adaptive forward pass (base prompt-learned model)
    margs.device = str(device)
    # Fill fields added to main.py's parser after this checkpoint was trained.
    for name, default in {
        "smooth": False,
        "ema": False,
        "ois": False,
        "use_contrast": False,
        "loss_adapter": 1.0,
        "loss_contrast": 1.0,
        "loss_condition": 1.0,
    }.items():
        if not hasattr(margs, name):
            setattr(margs, name, default)

    # Redirect the hardcoded '/Path/to/ViT-L-14.pt' to the real CLIP weights.
    import models.clip_models as clip_models_mod

    orig_load = clip_models_mod.load_clip_to_cpu
    clip_models_mod.load_clip_to_cpu = (
        lambda model_path, *a, **k: orig_load(CLIP_VIT_L14, *a, **k)
    )

    from models import build_model as repo_build_model

    model = repo_build_model(margs)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    img_resolution = int(getattr(margs, "img_resolution", 256))
    crop_resolution = int(getattr(margs, "crop_resolution", 224))
    transform = transforms.Compose([
        transforms.Resize((img_resolution, img_resolution)),
        transforms.CenterCrop(crop_resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


@torch.no_grad()
def score_batch(model, device, tensors):
    x = torch.stack(tensors).to(device)
    logits = model(x)
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

    model, transform = build_model(device)

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
