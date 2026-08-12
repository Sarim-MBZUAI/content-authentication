#!/usr/bin/env python
"""White-box PGD (Linf) attack harness for PGC (2026).

Follows PGD_ATTACK_CONTRACT.md: eps = 8/255, 10 steps, alpha = 2/255, random
start; ascend the TRUE-label loss. Perturbation lives in the RGB pixel space in
[0,1] (BEFORE normalization); [0,1] and the eps-ball are enforced every step.

MODEL / PREPROCESSING: model imported verbatim from ../adapters/PGC.py
(build_model()). PGC = DINOv2-large RGB stream (+LoRA) + a residual stream +
PGCM calibration + 1-logit head. Score = sigmoid(logit) = P(fake).

The adapter's eval transform is PadCenterCrop(224) -> ToTensor -> AppendResidual
-> Normalize(6ch). For our 512x512 inputs PadCenterCrop needs no padding, so it
is exactly a deterministic CenterCrop(224); we take that crop once (a fixed
pixel selection) and attack the resulting 224x224 [0,1] RGB tensor.

NON-DIFFERENTIABLE PREPROCESSING / BPDA (documented approximation):
  AppendResidual concatenates a 6th-channel "quantization residual" computed
  from the image by models/encoder/residual_extractor.py. That op contains
  `torch.round(x_yuv)` (rounding to integer YCbCr), which has ZERO gradient
  almost everywhere -> the residual channels would otherwise contribute no
  gradient to the pixels. We therefore:
    (1) RECOMPUTE the residual from the current adversarial image at EVERY PGD
        step (so the residual stream always sees the true residual of x), and
    (2) use a straight-through / BPDA estimator for the rounding: the forward
        value is the exact `torch.round(.)` but the backward pass treats round
        as identity (`q = v + (round(v) - v).detach()`).
  This makes the full 6-channel graph differentiable end-to-end while the
  forward pass remains numerically identical to the adapter. The RGB stream
  (the dominant path) is exactly differentiable regardless. Everything else
  (RGB + residual ImageNet/[0,mean 0.5-std] normalization) is reimplemented as
  differentiable torch ops matching data/transforms.py.
"""

import argparse
import csv
import importlib
import os
import sys

BENCH = "/shared/home/sarim.hashmi/usenix/benchmark"
sys.path.insert(0, os.path.join(BENCH, "adapters"))

EPS = 8.0 / 255.0
ALPHA = 2.0 / 255.0
STEPS = 10
IMG_SIZE = 224
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]
RES_MEAN = [0.0, 0.0, 0.0]
RES_STD = [0.5, 0.5, 0.5]


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
    _, _, h, w = x.shape
    # Pad if smaller than target (matches PadCenterCrop), then centre crop.
    ph, pw = max(0, size - h), max(0, size - w)
    if ph or pw:
        import torch.nn.functional as F
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

    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    from PIL import Image

    torch.manual_seed(0)
    device = torch.device(args.device)

    adapter = importlib.import_module("PGC")
    model = adapter.build_model(device)
    for p in model.parameters():
        p.requires_grad_(False)

    # YCbCr matrices used by the repo's residual extractor.
    from models.encoder.residual_extractor import M_YUV, M_YUV_INV
    M = M_YUV.to(device)
    Minv = M_YUV_INV.to(device)

    rgb_mean = torch.tensor(RGB_MEAN, device=device).view(1, 3, 1, 1)
    rgb_std = torch.tensor(RGB_STD, device=device).view(1, 3, 1, 1)
    res_mean = torch.tensor(RES_MEAN, device=device).view(1, 3, 1, 1)
    res_std = torch.tensor(RES_STD, device=device).view(1, 3, 1, 1)

    def residual_ste(x01):
        """Differentiable (straight-through round) quantization residual.

        Forward value == repo's QuantizationResidualExtractor; backward treats
        torch.round as identity (BPDA)."""
        xp = (x01 * 255.0).permute(0, 2, 3, 1)          # [B,H,W,3]
        yuv = torch.matmul(xp, M.T)
        q = yuv + (torch.round(yuv) - yuv).detach()     # straight-through
        recon = torch.matmul(q, Minv.T)
        res = (xp - recon).permute(0, 3, 1, 2)
        return torch.clamp(res, -1.0, 1.0)

    def prob_fake(x01):
        """Differentiable P(fake) from a [B,3,224,224] RGB tensor in [0,1]."""
        res = residual_ste(x01)
        rgb_n = (x01 - rgb_mean) / rgb_std
        res_n = (res - res_mean) / res_std
        six = torch.cat([rgb_n, res_n], dim=1)
        logit = model(six)
        if isinstance(logit, (tuple, list)):
            logit = logit[-1]
        return torch.sigmoid(logit.view(-1))

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
            img = Image.open(row["path"]).convert("RGB")
            x_full = TF.to_tensor(img).unsqueeze(0).to(device)  # [1,3,H,W] in [0,1]
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
            print(f"[PGC] error on {row['path']}: {e}", file=sys.stderr)
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
    print(f"[PGC] wrote {len(results)} rows -> {args.out} (max Linf={linf_max:.5f}, eps={EPS:.5f})")


if __name__ == "__main__":
    main()
