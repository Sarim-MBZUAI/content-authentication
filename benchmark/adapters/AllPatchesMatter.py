#!/usr/bin/env python
"""Benchmark adapter for AllPatchesMatter (2026) — GenImage-trained CLIP-LoRA variant.

Reference: detectors/2026/AllPatchesMatter/code/scripts/test_GenImage_clip_lora.sh
  -> scripts/train.py --is_test --model_name clip+lora --input_size 224 --is_crop
Model: CLIP ViT-L/14 vision tower + LoRA (r=4, q_proj/v_proj) + 2-class fc head
       (network/combined.py::Clip_Lora). Checkpoint contains the FULL vision
       tower + LoRA + fc, so the backbone is built from config (random init)
       and fully overwritten by the official checkpoint.
Preprocessing: repo's own create_val_transforms(size=224, is_crop=True)
       (albumentations: PadIfNeeded -> CenterCrop 224 -> PadIfNeeded ->
        ImageNet-normalize), images read with cv2 BGR->RGB like the repo's
        dataset (data/dataset.py::read_image).
Score: softmax(logits)[:, 1] = P(fake)  (GenImage convention: nature=0, ai=1).
"""

import argparse
import csv
import os
import sys

REPO = "detectors/2026/AllPatchesMatter"
CODE = os.path.join(REPO, "code")
CKPT = os.path.join(REPO, "weights", "GenImage_clip_lora_best.pth")

sys.path.insert(0, CODE)


def read_manifest(path, limit=None):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v)
                         for k, v in row.items()})
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_out(path, rows, scores):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generator", "gen_year", "label", "path", "score"])
        for row, s in zip(rows, scores):
            w.writerow([row["generator"], row["gen_year"], row["label"],
                        row["path"], "" if s is None else f"{s:.6f}"])


def build_model(device):
    import torch
    from transformers import CLIPVisionConfig, CLIPVisionModel
    from network.combined import Clip_Lora

    # CLIP ViT-L/14 vision tower config (openai/clip-vit-large-patch14).
    # Weights come entirely from the official checkpoint below.
    cfg = CLIPVisionConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
        projection_dim=768,
    )
    visual_model = CLIPVisionModel(cfg)
    net = Clip_Lora(visual_model, num_classes=2)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    # Everything in the official checkpoint must be consumed.
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]} ...")
    bad = [k for k in missing if ("lora" in k or k.startswith("fc."))]
    if bad:
        raise RuntimeError(f"Checkpoint is missing trained weights: {bad[:10]} ...")
    if missing:
        # Should be empty: checkpoint holds the full vision tower as well.
        raise RuntimeError(f"Missing keys after load: {missing[:10]} ...")

    net.to(device).eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import cv2
    import torch
    from data.transform import create_val_transforms

    device = torch.device(args.device)
    model = build_model(device)
    transform = create_val_transforms(size=224, is_crop=True)

    rows = read_manifest(args.manifest, args.limit)
    scores = [None] * len(rows)

    batch_size = 16 if device.type == "cuda" else 4
    buf_idx, buf_img = [], []

    def flush():
        if not buf_idx:
            return
        x = torch.stack(buf_img, dim=0).to(device).float()
        with torch.no_grad():
            logits = model(x)
            prob_fake = torch.softmax(logits, dim=1)[:, 1]
        for i, p in zip(buf_idx, prob_fake.detach().cpu().tolist()):
            scores[i] = float(p)
        buf_idx.clear()
        buf_img.clear()

    for i, row in enumerate(rows):
        try:
            img = cv2.imread(row["path"])
            if img is None:
                raise IOError(f"cv2 failed to read {row['path']}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform(image=img)["image"]
            buf_idx.append(i)
            buf_img.append(tensor)
            if len(buf_idx) >= batch_size:
                flush()
        except Exception as e:  # per-image failure -> empty score
            print(f"[AllPatchesMatter] error on {row['path']}: {e}", file=sys.stderr)
    flush()

    write_out(args.out, rows, scores)
    done = sum(s is not None for s in scores)
    print(f"[AllPatchesMatter] scored {done}/{len(rows)} images -> {args.out}")


if __name__ == "__main__":
    main()
