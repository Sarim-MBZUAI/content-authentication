#!/usr/bin/env python
"""Benchmark adapter for DEAR (2026) — DEAR-c variant (Corvi base, mask-gated).

Reference: detectors/2026/DEAR/code/scripts/inference.py
  -> --model dear_c: dear/detector/corvi_mask_gated_detector.py::CorviMaskGatedDetector
     (ResNet-50, stride0=1, gated layer4 channels, single-logit head).
Checkpoint: weights/dear_c/model_best.pth (contains backbone + gate buffer),
     loaded via the detector's own .load() as in inference.py.
Preprocessing: repo's own (inference.py): native resolution, no resize/crop;
     ToTensor + ImageNet normalize, batch size 1 (variable image sizes).
Score: sigmoid(logit) = P(fake); positive logit -> FAKE (per inference.py).
"""

import argparse
import csv
import os
import sys

REPO = "/shared/home/sarim.hashmi/usenix/detectors/2026/DEAR"
CODE = os.path.join(REPO, "code")
CKPT = os.path.join(REPO, "weights", "dear_c", "model_best.pth")

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
    from dear.detector.corvi_mask_gated_detector import CorviMaskGatedDetector

    device = args.device
    detector = CorviMaskGatedDetector(device=device, pretrained=False)
    detector.load(CKPT)
    detector.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    rows = read_manifest(args.manifest, args.limit)
    scores = [None] * len(rows)

    for i, row in enumerate(rows):
        try:
            img = Image.open(row["path"]).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = detector.predict(x).squeeze().item()
            scores[i] = float(torch.sigmoid(torch.tensor(logit)).item())
        except Exception as e:
            print(f"[DEAR] error on {row['path']}: {e}", file=sys.stderr)

    write_out(args.out, rows, scores)
    done = sum(s is not None for s in scores)
    print(f"[DEAR] scored {done}/{len(rows)} images -> {args.out}")


if __name__ == "__main__":
    main()
