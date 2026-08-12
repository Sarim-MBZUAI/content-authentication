#!/usr/bin/env python
"""White-box PGD (l-inf) attack harness for the SICA detector.

Detector: SICA (adapters/SICA.py) -- CLIP ViT-L/14 + LoRA + linear head.
    score = model(image)["pred_label"] = sigmoid(logit),  higher = more likely FAKE.
    Native threshold: 0.5 (probability score).

Attack (PGD_ATTACK_CONTRACT.md, FIXED for every detector):
    l-inf PGD, eps = 8/255, 10 steps, alpha = 2/255, random start in the ball.
    Flip the TRUE label: FAKE (label 1) -> push score DOWN toward "real";
    REAL (label 0) -> push score UP toward "fake". Equivalently ascend the objective
        obj = direction * score,   direction = +1 if label==0 else -1
    which moves the prediction away from the true label (== ascending the true-label loss).

Differentiability / preprocessing:
    The perturbation lives in the detector's input PIXEL space in [0,1] -- the 224x224
    tensor produced by the CLIP preprocess up to and including ToTensor, BEFORE mean/std
    normalization. The CLIP preprocess (PIL Resize(BICUBIC)+CenterCrop+ToTensor) is applied
    ONCE to build the clean pixel tensor; the mean/std Normalize is re-implemented as a
    differentiable op INSIDE the attacked graph. The whole CLIP+LoRA path is differentiable,
    so this is a fully end-to-end white-box attack (NO approximation).

CLI (contract):  --manifest <csv> --out <csv> [--device cuda|cpu] [--limit N]
Output columns (contract): path,label,score_clean,pred_clean,score_adv,pred_adv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torchvision.transforms import Compose, Normalize

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Import the detector (model + preprocessing) from the adapter -- do not rewrite it.
ADAPTERS = str(Path(__file__).resolve().parent.parent / "adapters")
sys.path.insert(0, ADAPTERS)
import SICA as adapter  # noqa: E402

EPS = 8.0 / 255.0
ALPHA = 2.0 / 255.0
STEPS = 10
SEED = 1
THRESHOLD = 0.5  # native probability threshold


def split_preprocess(preprocess):
    """Split the CLIP preprocess into (geometric-> [0,1] tensor, differentiable Normalize)."""
    norm = [t for t in preprocess.transforms if isinstance(t, Normalize)]
    if not norm:
        raise RuntimeError("Could not find a Normalize in the SICA preprocess.")
    norm = norm[0]
    geom = Compose([t for t in preprocess.transforms if not isinstance(t, Normalize)])
    return geom, norm


def pgd(score_fn, x_clean, label):
    """l-inf PGD ascending direction*score. x_clean in [0,1]; returns x_adv in [0,1]."""
    x_clean = x_clean.detach()
    direction = 1.0 if int(label) == 0 else -1.0  # real: push up; fake: push down
    delta = torch.empty_like(x_clean).uniform_(-EPS, EPS)
    x = torch.clamp(x_clean + delta, 0.0, 1.0).detach()
    for _ in range(STEPS):
        x.requires_grad_(True)
        obj = direction * score_fn(x).sum()
        grad = torch.autograd.grad(obj, x)[0]
        with torch.no_grad():
            x = x + ALPHA * grad.sign()
            x = torch.min(torch.max(x, x_clean - EPS), x_clean + EPS)
            x = torch.clamp(x, 0.0, 1.0)
        x = x.detach()
    return x


def main():
    ap = argparse.ArgumentParser(description="SICA PGD attack harness")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[SICA-atk] cuda unavailable; using cpu.")
        device = "cpu"
    device = torch.device(device)
    print(f"[SICA-atk] device={device} eps={EPS:.5f} steps={STEPS} alpha={ALPHA:.5f}")

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    print(f"[SICA-atk] {len(rows)} rows.")

    model = adapter.build_model(device)
    for p in model.parameters():
        p.requires_grad_(False)  # only the input needs gradients
    geom, norm = split_preprocess(model.preprocess)
    mean = torch.tensor(norm.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(norm.std, device=device).view(1, 3, 1, 1)

    def score_fn(x):  # x: pixel tensor in [0,1]; returns sigmoid prob (higher=fake)
        xn = (x - mean) / std
        return model(image=xn)["pred_label"]

    results = []  # (score_clean, score_adv) or (None, None)
    for i, row in enumerate(rows):
        try:
            img = Image.open(row["path"]).convert("RGB")
            x_clean = geom(img).unsqueeze(0).to(device)
            with torch.no_grad():
                s_clean = float(score_fn(x_clean).flatten()[0].item())
            x_adv = pgd(score_fn, x_clean, row["label"])
            linf = (x_adv - x_clean).abs().max().item()
            assert linf <= EPS + 1e-5, f"linf {linf} > eps {EPS}"
            with torch.no_grad():
                s_adv = float(score_fn(x_adv).flatten()[0].item())
            results.append((s_clean, s_adv))
            print(f"[SICA-atk] {i+1}/{len(rows)} label={row['label']} "
                  f"clean={s_clean:.4f} adv={s_adv:.4f} linf={linf:.5f}")
        except Exception as e:
            print(f"[SICA-atk] ERROR row {i} ({row['path']}): {type(e).__name__}: {str(e)[:160]}")
            results.append((None, None))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "score_clean", "pred_clean", "score_adv", "pred_adv"])
        for row, (sc, sa) in zip(rows, results):
            if sc is None:
                w.writerow([row["path"], row["label"], "", "", "", ""])
            else:
                pc = int(sc > THRESHOLD)
                pa = int(sa > THRESHOLD)
                w.writerow([row["path"], row["label"], sc, pc, sa, pa])
    print(f"[SICA-atk] wrote {len(rows)} rows to {args.out} (threshold={THRESHOLD})")


if __name__ == "__main__":
    main()
