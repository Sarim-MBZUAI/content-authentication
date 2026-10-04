#!/usr/bin/env python
"""White-box PGD (Linf) attack harness for OmniAID (2026).

Follows PGD_ATTACK_CONTRACT.md: eps = 8/255, 10 steps, alpha = 2/255, random
start; ascend the TRUE-label loss (flip direction: fake -> push score down,
real -> push score up). Perturbation lives in the model's input pixel space in
[0,1] (the tensor BEFORE ImageNet normalization); [0,1] and the eps-ball are
enforced every step.

MODEL / PREPROCESSING: imported verbatim from ../adapters/OmniAID.py
(build_model()). OmniAID = DINOv3 ViT-L/16 turned into an SVD-MoE; forward
returns dict with 'prob' = softmax(cls_logits)[:,1] = P(fake). Fully
differentiable core (MoE gating is soft / top-k with a differentiable expert
path), so the attack is end-to-end white-box.

DIFFERENTIABILITY NOTE (documented approximation):
  The adapter's eval transform is PIL Resize(448) -> ToTensor -> Normalize.
  The PIL resize is a non-differentiable resample; per the contract we treat it
  as a FIXED preprocessing step (BPDA identity) and attack the tensor that
  feeds the first differentiable module, i.e. the 448x448 [0,1] tensor that
  results from the resize. ImageNet mean/std normalization is reimplemented as
  a differentiable op INSIDE the attacked graph. No other approximation.
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
IMG_SIZE = 448
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image

    torch.manual_seed(0)
    device = torch.device(args.device)

    adapter = importlib.import_module("OmniAID")
    model = adapter.build_model(device)
    for p in model.parameters():
        p.requires_grad_(False)

    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    to_tensor = T.Compose([T.Resize([IMG_SIZE, IMG_SIZE]), T.ToTensor()])

    def prob_fake(x01):
        """Differentiable P(fake) from a [B,3,H,W] tensor in [0,1]."""
        out = model((x01 - mean) / std)
        p = out["prob"] if isinstance(out, dict) else out
        return p.view(-1)

    def pgd(x0, y):
        x0 = x0.detach()
        delta = torch.empty_like(x0).uniform_(-EPS, EPS)
        x = torch.clamp(x0 + delta, 0.0, 1.0)
        for _ in range(STEPS):
            x = x.detach().requires_grad_(True)
            p = prob_fake(x).clamp(1e-6, 1.0 - 1e-6)
            loss = F.binary_cross_entropy(p, y)  # ascend true-label loss
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
            x0 = to_tensor(img).unsqueeze(0).to(device)
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
            print(f"[OmniAID] error on {row['path']}: {e}", file=sys.stderr)
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
    print(f"[OmniAID] wrote {len(results)} rows -> {args.out} (max Linf={linf_max:.5f}, eps={EPS:.5f})")


if __name__ == "__main__":
    main()
