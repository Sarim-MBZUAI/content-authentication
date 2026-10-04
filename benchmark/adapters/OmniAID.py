#!/usr/bin/env python
"""Benchmark adapter for OmniAID (2026) — DINOv3-based hybrid MoE, v2 checkpoint.

Reference: detectors/2026/OmniAID/code/scripts/eval.sh
  -> main_finetune.py --model OmniAID_DINO --img_size 448 --eval True --is_hybrid True
     --resume weights/checkpoint_omniaid_dino_v2.pth
     --moe_config_path weights/config/config_omniaid_dino_v2.json
Model: models/OmniAID_DINO.py::OmniAID_DINO (DINOv3 ViT-L/16 backbone turned
       into an SVD-MoE with 6 experts + gating network + 2-class head).
       The official checkpoint contains ALL weights (incl. the DINOv3
       feature_extractor), so the gated facebook/dinov3-vitl16 hub repo is not
       needed: we shim DINOv3ViTModel.from_pretrained to return a
       config-initialized model and then strictly load the checkpoint.
Preprocessing: repo's own eval transform (data/datasets.py, DINO branch):
       Resize(448,448) -> ToTensor -> ImageNet normalize.
Score: model forward returns dict; 'prob' = softmax(cls_logits)[:,1] = P(fake)
       (folders: 0_real/nature=0, 1_fake/ai=1).
"""

import argparse
import csv
import json
import os
import sys
from types import SimpleNamespace

REPO = "detectors/2026/OmniAID"
CODE = os.path.join(REPO, "code")
CKPT = os.path.join(REPO, "weights", "checkpoint_omniaid_dino_v2.pth")
MOE_CONFIG = os.path.join(REPO, "weights", "config", "config_omniaid_dino_v2.json")

sys.path.insert(0, CODE)

IMG_SIZE = 448  # eval.sh OmniAID_DINO branch

# facebook/dinov3-vitl16-pretrain-lvd1689m architecture (verified against the
# checkpoint's feature_extractor.* tensor shapes: 24 layers, hidden 1024,
# mlp 4096, patch 16, 4 register tokens, no k_proj bias, plain (non-gated) MLP).
DINOV3_VITL16_CFG = dict(
    patch_size=16,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=24,
    num_attention_heads=16,
    hidden_act="gelu",
    layer_norm_eps=1e-5,
    rope_theta=100.0,
    image_size=224,
    num_channels=3,
    query_bias=True,
    key_bias=False,
    value_bias=True,
    proj_bias=True,
    mlp_bias=True,
    layerscale_value=1.0,
    use_gated_mlp=False,
    num_register_tokens=4,
    pos_embed_rescale=2.0,
)


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
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    # transformers >= 5 renamed DINOv3ViTDropPath -> Dinov3ViTDropPath; alias it
    # so the (unmodified) repo module imports cleanly.
    import transformers.models.dinov3_vit.modular_dinov3_vit as _tmod
    if not hasattr(_tmod, "DINOv3ViTDropPath") and hasattr(_tmod, "Dinov3ViTDropPath"):
        _tmod.DINOv3ViTDropPath = _tmod.Dinov3ViTDropPath

    import models.OmniAID_DINO as omni_mod

    # Shim: the repo calls DINOv3ViTModel.from_pretrained(config.DINOV3_path),
    # a gated hub repo. All backbone weights live in the official checkpoint,
    # so return a config-initialized DINOv3 instead (loaded strictly below).
    # transformers >= 5 nests the transformer blocks under `.model.layer`; the
    # repo (written against the 4.x layout) accesses `.layer` directly, so
    # expose it via a property.
    class _DINOv3Compat(DINOv3ViTModel):
        @property
        def layer(self):
            return self.model.layer

    class _ShimDINOv3:
        @staticmethod
        def from_pretrained(path, *a, **k):
            return _DINOv3Compat(DINOv3ViTConfig(**DINOV3_VITL16_CFG))

    omni_mod.DINOv3ViTModel = _ShimDINOv3

    with open(MOE_CONFIG) as f:
        moe_cfg = json.load(f)
    cfg = SimpleNamespace(**moe_cfg)
    cfg.is_hybrid = True  # eval.sh: --is_hybrid True

    model = omni_mod.OmniAID_DINO(config=cfg)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    # Checkpoint was saved with the 4.x DINOv3 layout (feature_extractor.layer.*);
    # transformers >= 5 nests those blocks as feature_extractor.model.layer.*.
    state = {
        (k.replace("feature_extractor.layer.", "feature_extractor.model.layer.")
         if k.startswith("feature_extractor.layer.") else k): v
        for k, v in ckpt["model"].items()
    }
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
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
    model = build_model(device)

    # data/datasets.py eval transform, DINO branch (ImageNet mean/std).
    transform = transforms.Compose([
        transforms.Resize([IMG_SIZE, IMG_SIZE]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    rows = read_manifest(args.manifest, args.limit)
    scores = [None] * len(rows)

    batch_size = 8 if device.type == "cuda" else 2
    buf_idx, buf_img = [], []

    def flush():
        if not buf_idx:
            return
        x = torch.stack(buf_img, dim=0).to(device)
        with torch.no_grad():
            out = model(x)
            prob_fake = out["prob"]
        for i, p in zip(buf_idx, prob_fake.detach().cpu().tolist()):
            scores[i] = float(p)
        buf_idx.clear()
        buf_img.clear()

    for i, row in enumerate(rows):
        try:
            img = Image.open(row["path"]).convert("RGB")
            buf_idx.append(i)
            buf_img.append(transform(img))
            if len(buf_idx) >= batch_size:
                flush()
        except Exception as e:
            print(f"[OmniAID] error on {row['path']}: {e}", file=sys.stderr)
    flush()

    write_out(args.out, rows, scores)
    done = sum(s is not None for s in scores)
    print(f"[OmniAID] scored {done}/{len(rows)} images -> {args.out}")


if __name__ == "__main__":
    main()
