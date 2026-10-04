#!/usr/bin/env python
"""Benchmark adapter: FreqNet (Tan et al., AAAI 2024).

freqnet(num_classes=1) + 4-classes-freqnet-v2.pth. Score = sigmoid(logit)
= P(fake); higher = more likely fake (repo convention already matches).
Test transform follows code/test.py ForenSynths setting
(no_resize=False, no_crop=True): Resize((256,256)) + ImageNet norm.
"""
import argparse
import csv
import os
import sys

REPO = "detectors/2024/FreqNet"
CODE = os.path.join(REPO, "code")
CKPT = os.path.join(REPO, "weights", "4-classes-freqnet-v2.pth")

sys.path.insert(0, CODE)


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

    # networks/freqnet.py hardcodes .cuda() when constructing its spectrum
    # conv parameters; redirect to the requested device so CPU works.
    torch.Tensor.cuda = lambda self, *a, **kw: self.to(device)

    from networks.freqnet import freqnet  # repo module

    model = freqnet(num_classes=1)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=True)
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[: args.limit]

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    BATCH = 16
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
                    scores = model(x).sigmoid().flatten().tolist()
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
