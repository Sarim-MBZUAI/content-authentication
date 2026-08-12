#!/usr/bin/env python
"""Benchmark adapter for PGC (2026) — combined ProGAN + SDv1.4 training checkpoint.

Reference: detectors/2026/PGC/code/test.py + engine/evaluator.py.
Model: models/pgc.py::PGCNetwork — DINOv2-large RGB stream (HF Dinov2Model,
       LoRA r=8 injected into mlp.fc1/fc2 via the repo's own matcher) +
       residual stream + PGCM calibration + 1-logit head. Hyperparameters
       match the checkpoint's saved opt (dino_variant=dinov2-large,
       cropSize=224, lora_rank=8, lora_alpha=1.0, lora_dropout=0.1,
       tau_rgb=tau_res=0.5).
Backbone init: facebook/dinov2-large from the cluster-shared HF hub snapshot
       (the repo requires a local path via pretrained_root); the official
       checkpoint then overwrites all trained weights (strict load, as test.py).
Preprocessing: repo's create_eval_transforms(224, MEAN['dino'], STD['dino']):
       PadCenterCrop(224) -> ToTensor -> AppendResidual -> Normalize (6ch).
Score: sigmoid(logit) = P(fake) (evaluator.py; folders 0_real/1_fake).
"""

import argparse
import csv
import glob
import os
import sys

REPO = "/shared/home/sarim.hashmi/usenix/detectors/2026/PGC"
CODE = os.path.join(REPO, "code")
CKPT = os.path.join(REPO, "weights", "PGC_train_progan_sdv1_4_ckpt.pth")

sys.path.insert(0, CODE)


def find_dinov2_large_snapshot():
    pats = [
        "/datasets/huggingface/hub/models--facebook--dinov2-large/snapshots/*/",
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--facebook--dinov2-large/snapshots/*/"),
    ]
    for pat in pats:
        for d in sorted(glob.glob(pat)):
            if os.path.exists(os.path.join(d, "config.json")):
                return d.rstrip("/")
    raise FileNotFoundError("facebook/dinov2-large snapshot not found in HF caches")


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
    from models.pgc import PGCNetwork

    model = PGCNetwork(
        dino_variant="dinov2-large",
        lora_rank=8,
        lora_alpha=1.0,
        lora_dropout=0.1,
        lora_targets=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        pretrained_root=find_dinov2_large_snapshot(),
        tau_rgb=0.5,
        tau_res=0.5,
    )
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])  # strict, as in test.py
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
    from PIL import Image
    from data.transforms import MEAN, STD, create_eval_transforms

    device = torch.device(args.device)
    model = build_model(device)
    transform = create_eval_transforms(image_size=224, mean=MEAN["dino"], std=STD["dino"])

    rows = read_manifest(args.manifest, args.limit)
    scores = [None] * len(rows)

    batch_size = 16 if device.type == "cuda" else 4
    buf_idx, buf_img = [], []

    def flush():
        if not buf_idx:
            return
        x = torch.stack(buf_img, dim=0).to(device)
        with torch.no_grad():
            _features, logits = model(x, return_feature=True)
            probs = torch.sigmoid(logits.view(-1))
        for i, p in zip(buf_idx, probs.detach().cpu().tolist()):
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
            print(f"[PGC] error on {row['path']}: {e}", file=sys.stderr)
    flush()

    write_out(args.out, rows, scores)
    done = sum(s is not None for s in scores)
    print(f"[PGC] scored {done}/{len(rows)} images -> {args.out}")


if __name__ == "__main__":
    main()
