#!/usr/bin/env python
"""White-box PGD (l-inf) attack harness for the FerretNet detector.

Detector: FerretNet (adapters/FerretNet.py) -- local-pixel-dependency CNN
    (Ferret dim=96, depths=[2,2], median LPD). score = sigmoid(logit) = P(fake),
    higher = more likely FAKE. Native threshold: 0.5.

Attack (PGD_ATTACK_CONTRACT.md, FIXED): l-inf PGD, eps=8/255, 10 steps, alpha=2/255,
    random start. Flip the true label: REAL (0) push score UP, FAKE (1) push score DOWN
    (== ascending the direction*score objective, direction=+1 if label==0 else -1).

Differentiability: perturbation lives in the input PIXEL space in [0,1] -- the 256x256
    tensor after CenterCrop+ToTensor, BEFORE CLIP mean/std normalization. The Normalize is
    re-implemented as a differentiable op inside the attacked graph. FerretNet's LPD/median
    conv path is differentiable -> fully end-to-end white-box attack (no approximation).

CLI: --manifest <csv> --out <csv> [--device cuda|cpu] [--limit N]
Output: path,label,score_clean,pred_clean,score_adv,pred_adv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageFile
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Reuse the detector's official model/weights via its adapter's build path.
ADAPT = str(Path(__file__).resolve().parent.parent / "adapters")
sys.path.insert(0, ADAPT)
CODE = "/shared/home/sarim.hashmi/usenix/detectors/2025/FerretNet/code"
CFG_PATH = os.path.join(CODE, "configs/Test.yaml")
CKPT = os.path.join(CODE, "checkpoints/4cls_ckpt/ferretnet-b-median-3.pth")
sys.path.insert(0, CODE)
from util import get_cfg, load_state_dict          # noqa: E402
from src.model import get_model                    # noqa: E402

EPS, ALPHA, STEPS, THRESHOLD = 8.0 / 255.0, 2.0 / 255.0, 10, 0.5
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]

# geometric part only: CenterCrop(256) + ToTensor -> pixel tensor in [0,1]
_geom = T.Compose([T.CenterCrop((256, 256)), T.ToTensor()])


def build_model(device):
    cfg = get_cfg(cfg_path=CFG_PATH, mode="test")
    model = get_model(cfg)
    load_state_dict(model, CKPT)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser(description="FerretNet PGD attack harness")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    dev = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    model = build_model(dev)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    torch.manual_seed(1)

    def score_fn(x):  # x in [0,1] -> P(fake), differentiable
        return model((x - mean) / std).sigmoid().flatten()

    def pgd(x0, label):
        x0 = x0.detach()
        d = 1.0 if int(label) == 0 else -1.0
        x = torch.clamp(x0 + torch.empty_like(x0).uniform_(-EPS, EPS), 0, 1).detach()
        for _ in range(STEPS):
            x.requires_grad_(True)
            g = torch.autograd.grad((d * score_fn(x)).sum(), x)[0]
            with torch.no_grad():
                x = torch.clamp(torch.min(torch.max(x + ALPHA * g.sign(), x0 - EPS), x0 + EPS), 0, 1)
            x = x.detach()
        return x

    out = []
    for r in rows:
        try:
            img = Image.open(r["path"]).convert("RGB")
            x0 = _geom(img).unsqueeze(0).to(dev)
            with torch.no_grad():
                sc = float(score_fn(x0).item())
            xadv = pgd(x0, r["label"])
            with torch.no_grad():
                sa = float(score_fn(xadv).item())
            out.append([r["path"], r["label"], sc, int(sc > THRESHOLD), sa, int(sa > THRESHOLD)])
        except Exception as e:
            print(f"[FerretNet-atk] {r['path']}: {type(e).__name__}: {e}", file=sys.stderr)
            out.append([r["path"], r["label"], "", "", "", ""])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "score_clean", "pred_clean", "score_adv", "pred_adv"])
        w.writerows(out)
    print(f"[FerretNet-atk] wrote {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
