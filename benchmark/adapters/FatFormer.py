#!/usr/bin/env python
"""Benchmark adapter: FatFormer (Liu et al., CVPR 2024).

CLIP ViT-L/14 with forgery-aware adapters + language-guided alignment
(weights/fatformer_4class_ckpt.pth). Score = softmax(logits)[:, 1] = P(fake);
higher = more likely fake (repo convention already matches, see code/main.py).
Test transform from code/utils/dataset.py: Resize((256,256)) + CenterCrop(224)
+ ImageNet norm.

The repo's bundled clip.load() reads the OpenAI backbone from the relative
path 'pretrained/ViT-L-14.pt'; we point that at the official checkpoint
already on disk (sha256 matches OpenAI's URL hash) via a symlink under
benchmark/assets/ and chdir there while building the model.
"""
import argparse
import csv
import os
import sys

REPO = "/shared/home/sarim.hashmi/usenix/detectors/2024/FatFormer"
CODE = os.path.join(REPO, "code")
CKPT = os.path.join(REPO, "weights", "fatformer_4class_ckpt.pth")
CLIP_BACKBONE = "/shared/home/sarim.hashmi/usenix/detectors/2025/Chimera/weights/models/deepfake/ViT-L-14.pt"
ASSETS = "/shared/home/sarim.hashmi/usenix/benchmark/assets/fatformer"

sys.path.insert(0, CODE)


def build_fatformer(device_str):
    import torch
    from models import build_model  # repo module

    # Defaults from code/main.py get_args_parser(), except num_vit_adapter:
    # the released fatformer_4class_ckpt.pth has forgery-aware adapters at
    # resblocks 7/15/23 only, i.e. it was trained with num_vit_adapter=3.
    args = argparse.Namespace(
        backbone="CLIP:ViT-L/14",
        num_classes=2,
        num_vit_adapter=3,
        num_context_embedding=8,
        init_context_embedding="",
        hidden_dim=768,
        clip_vision_width=1024,
        frequency_encoder_layer=2,
        decoder_layer=4,
        num_heads=12,
        device=device_str,
    )

    # clip.load() opens 'pretrained/ViT-L-14.pt' relative to cwd.
    link = os.path.join(ASSETS, "pretrained", "ViT-L-14.pt")
    if not os.path.exists(link):
        os.makedirs(os.path.dirname(link), exist_ok=True)
        os.symlink(CLIP_BACKBONE, link)
    cwd = os.getcwd()
    os.chdir(ASSETS)
    try:
        model = build_model(args)
    finally:
        os.chdir(cwd)

    checkpoint = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    return model


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

    device = torch.device(args.device)
    model = build_fatformer(args.device)
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[: args.limit]

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    BATCH = 8
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
                    scores = model(x).softmax(dim=1)[:, 1].flatten().tolist()
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
