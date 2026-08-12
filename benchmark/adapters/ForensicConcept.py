#!/usr/bin/env python
"""Benchmark adapter for ForensicConcept (2026).

Official code:    /shared/home/sarim.hashmi/usenix/detectors/2026/ForensicConcept/code
Official weights: .../weights/weights/detectors/clip_vitl14_codebook_stage2.pth

NOTE ON BACKBONE CHOICE: the paper's primary DINOv3 variant
(dinov3_vitl16_concept_stage2.pth) requires the facebook DINOv3 ViT-L/16 backbone
weights, which are GATED on Hugging Face (403 with the available token) and the
official dinov3 GitHub code. Per fallback instructions this adapter therefore uses
the released CLIP variant (CGCI: concept-guided codebook injection into CLIP
ViT-L/14), configured exactly as code/configs/clip_codebook.yaml.

Checkpoint loading mirrors networks/base_model.py load_networks() (trainmode='lora'
branch): load 'fc' + 'codebook_head' state dicts, non-strict LoRA state load, reset
LoRA merge flags, then merge once for eval.

Score = sigmoid(main head logits) — the repo's primary evaluation head
(train_with_config.py _pick_primary_head returns "main"); higher = FAKE.
Preprocessing = repo test transform with legacy_preprocess=True (data/datasets.py
binary_dataset, testing: data_aug=false, no_resize=false, no_crop=false):
Resize((224,224)) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet).
"""
import argparse
import csv
import os
import sys
from types import SimpleNamespace

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = "/shared/home/sarim.hashmi/usenix/detectors/2026/ForensicConcept"
CODE = os.path.join(ROOT, "code")
CKPT = os.path.join(ROOT, "weights", "weights", "detectors", "clip_vitl14_codebook_stage2.pth")
CODEBOOK = os.path.join(CODE, "assets", "codebooks", "cleandift_codebook.npy")
CLIP_VIT_L14 = os.path.expanduser("~/.cache/clip/ViT-L-14.pt")

sys.path.insert(0, CODE)


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def build_model(device):
    from models import get_model

    # Values from code/configs/clip_codebook.yaml (stage-2 CGCI inference config).
    opt = SimpleNamespace(
        trainmode="lora",
        modelname="CLIP:ViT-L/14",
        clip_model_path=CLIP_VIT_L14,
        clip_load_init_weights=True,
        clip_lora_rank=4,
        clip_lora_alpha=8,
        clip_use_main_head=True,
        clip_use_codebook_inject=True,
        clip_codebook_path=CODEBOOK,
        clip_codebook_dim=1280,
        clip_num_concepts=200,
        clip_tau=0.1,
        clip_codebook_l2=True,
        clip_freeze_codebook=True,
        clip_num_patches=256,
        clip_cb_topk=20,
        clip_cb_topr=8,
        clip_cb_tau_w=0.05,
        clip_cb_weight_mode="score",
        clip_cb_mlp_hidden=768,
        clip_cb_detach_tokens=True,
        cropSize=224,
        num_classes=1,
    )
    model = get_model("CLIP:ViT-L/14", opt)

    # Mirror networks/base_model.py load_networks() lora branch (eval path).
    state_dict = torch_load(CKPT)
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    model.fc.load_state_dict(state_dict["fc"])
    model.codebook_head.load_state_dict(state_dict["codebook_head"])
    lora_state = state_dict.get("lora", {})
    if any(k.startswith("model.") for k in lora_state):
        model.load_state_dict(lora_state, strict=False)
    else:
        model.model.load_state_dict(lora_state, strict=False)

    def _reset_lora_merge_flag(module):
        if getattr(module, "r", 0) > 0 and hasattr(module, "merged"):
            module.merged = False

    model.apply(_reset_lora_merge_flag)

    def _merge_once(module):
        if getattr(module, "r", 0) > 0 and hasattr(module, "lora_train"):
            try:
                module.merged = False  # force a fresh merge
                module.lora_train(False)
            except Exception:
                pass

    model.apply(_merge_once)

    model.to(device)
    model.eval()

    # Test transform, legacy_preprocess=True / data_aug=False / no_resize=False /
    # no_crop=False / loadSize=cropSize=224 (data/datasets.py binary_dataset).
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


@torch.no_grad()
def score_batch(model, device, tensors):
    x = torch.stack(tensors).to(device)
    out = model(x)
    logits = out["logits"] if isinstance(out, dict) else out
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
