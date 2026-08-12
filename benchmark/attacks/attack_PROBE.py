#!/usr/bin/env python
"""PGD ell-inf attack harness for PROBE (2026) - DINOv2 patch classifier.

Spec: ell-inf PGD, eps = 8/255, 10 steps, alpha = 2/255, random start; ascend the
TRUE-label loss (flip). Perturbation in [0,1] pixel space BEFORE ImageNet
normalization (normalization done differentiably inside the graph).

================= DIFFERENTIABILITY =================
PROBE's pipeline is END-TO-END DIFFERENTIABLE -> NO BPDA / NO approximation.
Verified against adapters/PROBE.py + code/Detector/{util.py,evaluate_dino.py}:
  1. util.data_augment with all corruption probs = 0 reduces to a deterministic
     geometric pad/tile (pad_image, mode='repeat'). For the 512x512 benchmark
     images this pad is a no-op (>= crop_size 336). We run it ONCE on the clean
     image purely to obtain the padded canvas, then attack the tensor of that
     canvas in [0,1] -- i.e. the exact tensor the model consumes. This is not an
     approximation: it is the true model input space; the layout is fixed, not a
     gradient shortcut.
  2. ToTensor + ImageNet Normalize -> done as differentiable tensor ops.
  3. generate_sliding_patches = Tensor.unfold (pure slicing) -> differentiable.
  4. dino_classifier forward + mean over patch logits -> differentiable.
Attack target: mean patch logit; score = sigmoid(mean logit); pred threshold 0.5.
label: 0=real, 1=fake.
"""
import argparse
import csv
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(HERE, "..", "adapters", "PROBE.py")

EPS = 8.0 / 255.0
ALPHA = 2.0 / 255.0
STEPS = 10
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_adapter():
    spec = importlib.util.spec_from_file_location("probe_adapter", os.path.abspath(ADAPTER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # sets HF_HOME + sys.path to repo CODE dir
    return mod


def pgd_attack(forward_logit, x0, label, device):
    x0 = x0.to(device)
    y = torch.tensor([[float(label)]], device=device)
    bce = torch.nn.BCEWithLogitsLoss()
    delta = torch.empty_like(x0).uniform_(-EPS, EPS)
    x_adv = torch.clamp(x0 + delta, 0.0, 1.0).detach()
    for _ in range(STEPS):
        x_adv.requires_grad_(True)
        logit = forward_logit(x_adv).view(1, 1)
        loss = bce(logit, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + ALPHA * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x0 + EPS), x0 - EPS)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = x_adv.detach()
    return x_adv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    adapter = load_adapter()

    from PIL import Image
    import numpy as np
    from model.dino_classifier import dino_classifier
    from util import data_augment

    model = dino_classifier(model_name=adapter.BACKBONE, classifier_type="linear")
    ckpt = torch.load(adapter.CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    ps = adapter.CROP_SIZE
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def forward_logit(x01):
        # x01: [1,3,H,W] in [0,1] -> normalize -> sliding patches -> mean logit
        x = (x01 - mean) / std
        patches = adapter.sliding_patches(x[0], ps)  # [P,3,ps,ps]
        if patches.shape[0] == 0:
            raise RuntimeError("no patches")
        out = model(patches.to(device)).view(-1)
        return out.mean().view(1, 1)

    with open(os.path.abspath(args.manifest), newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "score_clean", "pred_clean", "score_adv", "pred_adv"])
        for row in rows:
            path, label = row["path"], int(row["label"])
            try:
                img = Image.open(path).convert("RGB")
                padded = data_augment(img, data_augment_params=adapter.DATA_AUGMENT_PARAMS,
                                      mode="fix", crop_size=ps)  # deterministic layout
                x0 = torch.from_numpy(
                    np.asarray(padded, dtype=np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W] in [0,1]

                with torch.no_grad():
                    logit_c = forward_logit(x0.to(device)).view(-1)[0]
                score_c = torch.sigmoid(logit_c).item()

                x_adv = pgd_attack(forward_logit, x0, label, device)
                linf = (x_adv - x0.to(device)).abs().max().item()
                with torch.no_grad():
                    logit_a = forward_logit(x_adv).view(-1)[0]
                score_a = torch.sigmoid(logit_a).item()

                w.writerow([path, label, f"{score_c:.6f}", int(score_c >= 0.5),
                            f"{score_a:.6f}", int(score_a >= 0.5)])
                print(f"[PROBE] {os.path.basename(path)} label={label} "
                      f"clean={score_c:.4f} adv={score_a:.4f} linf={linf:.5f}")
            except Exception as e:
                print(f"[PROBE] error on {path}: {e}", file=sys.stderr)
                w.writerow([path, label, "", "", "", ""])
    print(f"[PROBE] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
