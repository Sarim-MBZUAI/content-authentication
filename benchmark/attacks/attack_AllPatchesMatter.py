#!/usr/bin/env python
"""White-box PGD (Linf) attack harness for AllPatchesMatter (2026).

Follows PGD_ATTACK_CONTRACT.md: eps = 8/255, 10 steps, alpha = 2/255, random
start; ascend the TRUE-label loss. Perturbation lives in the model's input
pixel space in [0,1] (BEFORE normalization); [0,1] and the eps-ball enforced
every step.

MODEL: imported verbatim from ../adapters/AllPatchesMatter.py (build_model()).
CLIP ViT-L/14 vision tower + LoRA (r=4, q/v_proj) + 2-class fc head. Forward
returns logits[B,2]; score = softmax(logits)[:,1] = P(fake). Fully
differentiable core -> end-to-end white-box attack.

NON-DIFFERENTIABLE / REIMPLEMENTED PREPROCESSING (documented approximation):
  The adapter reads images with cv2 (BGR->RGB) and preprocesses with an
  albumentations pipeline create_val_transforms(size=224, is_crop=True):
      PadIfNeeded(224) -> CenterCrop(224) -> PadIfNeeded(224) -> Normalize(IN).
  albumentations runs on numpy (not autograd-traceable). For our 512x512
  inputs the two PadIfNeeded steps are no-ops (image already >=224 before and
  ==224 after the crop), so the pipeline reduces EXACTLY to a deterministic
  centre-crop(224) followed by ImageNet normalization ((img/255 - mean)/std,
  albumentations' default max_pixel_value=255). We reimplement precisely that
  in torch: read the image with cv2 the same way, scale to [0,1], take the
  centre 224x224 crop once (a fixed pixel selection), and attack that [0,1]
  tensor; the normalize is a differentiable op inside the graph. No BPDA of a
  non-invertible op is needed because nothing between the attacked tensor and
  the logits is non-differentiable.
"""

import argparse
import csv
import importlib
import os
import sys

BENCH = "benchmark"
sys.path.insert(0, os.path.join(BENCH, "adapters"))

EPS = 8.0 / 255.0
ALPHA = 2.0 / 255.0
STEPS = 10
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def read_manifest(path, limit=None):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v)
                         for k, v in row.items()})
            if limit is not None and len(rows) >= limit:
                break
    return rows


def center_crop(x, size):
    import torch.nn.functional as F
    _, _, h, w = x.shape
    ph, pw = max(0, size - h), max(0, size - w)
    if ph or pw:  # matches albumentations PadIfNeeded (BORDER_CONSTANT, value=0)
        x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=0.0)
        _, _, h, w = x.shape
    top = (h - size) // 2
    left = (w - size) // 2
    return x[:, :, top:top + size, left:left + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    device = torch.device(args.device)

    adapter = importlib.import_module("AllPatchesMatter")
    model = adapter.build_model(device)
    for p in model.parameters():
        p.requires_grad_(False)

    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def prob_fake(x01):
        logits = model((x01 - mean) / std)
        return torch.softmax(logits, dim=1)[:, 1].view(-1)

    def pgd(x0, y):
        x0 = x0.detach()
        delta = torch.empty_like(x0).uniform_(-EPS, EPS)
        x = torch.clamp(x0 + delta, 0.0, 1.0)
        for _ in range(STEPS):
            x = x.detach().requires_grad_(True)
            p = prob_fake(x).clamp(1e-6, 1.0 - 1e-6)
            loss = F.binary_cross_entropy(p, y)
            grad = torch.autograd.grad(loss, x)[0]
            x = x.detach() + ALPHA * grad.sign()
            x = torch.max(torch.min(x, x0 + EPS), x0 - EPS)
            x = torch.clamp(x, 0.0, 1.0)
        return x.detach()

    rows = read_manifest(args.manifest, args.limit)
    results = []
    linf_max = 0.0
    for row in rows:
        rec = {"path": row["path"], "label": row["label"],
               "score_clean": None, "pred_clean": None,
               "score_adv": None, "pred_adv": None}
        try:
            label = int(float(row["label"]))
            img = cv2.imread(row["path"])
            if img is None:
                raise IOError(f"cv2 failed to read {row['path']}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # match repo dataset
            x_full = torch.from_numpy(np.ascontiguousarray(img)).float().div(255.0)
            x_full = x_full.permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
            x0 = center_crop(x_full, IMG_SIZE)
            y = torch.tensor([float(label)], device=device)

            with torch.no_grad():
                sc = float(prob_fake(x0).item())
            x_adv = pgd(x0, y)
            with torch.no_grad():
                sa = float(prob_fake(x_adv).item())

            linf = float((x_adv - x0).abs().max().item())
            linf_max = max(linf_max, linf)
            assert linf <= EPS + 1e-5, f"Linf {linf} exceeds eps {EPS}"

            rec.update(score_clean=sc, pred_clean=int(sc >= 0.5),
                       score_adv=sa, pred_adv=int(sa >= 0.5))
        except Exception as e:
            print(f"[AllPatchesMatter] error on {row['path']}: {e}", file=sys.stderr)
        results.append(rec)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "score_clean", "pred_clean",
                    "score_adv", "pred_adv"])
        for r in results:
            w.writerow([r["path"], r["label"],
                        "" if r["score_clean"] is None else f"{r['score_clean']:.6f}",
                        "" if r["pred_clean"] is None else r["pred_clean"],
                        "" if r["score_adv"] is None else f"{r['score_adv']:.6f}",
                        "" if r["pred_adv"] is None else r["pred_adv"]])
    print(f"[AllPatchesMatter] wrote {len(results)} rows -> {args.out} (max Linf={linf_max:.5f}, eps={EPS:.5f})")


if __name__ == "__main__":
    main()
