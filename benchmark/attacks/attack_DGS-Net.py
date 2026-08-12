#!/usr/bin/env python
"""PGD ell-inf attack harness for DGS-Net (2026).

Spec (PGD_ATTACK_CONTRACT.md): ell-inf PGD, eps = 8/255, 10 steps, alpha = 2/255,
random start; ascend the TRUE-label loss (flip). Perturbation lives in [0,1] pixel
space BEFORE ImageNet mean/std normalization; normalization is done as a
differentiable op inside the attacked graph.

================= NON-DIFFERENTIABLE PREPROCESSING / BPDA =================
DGS-Net's test transform is
    PatchSelectionTransform(32, 49) -> ToTensor -> Normalize(ImageNet).
PatchSelectionTransform (data/PatchSelectionTransform.py) is NON-differentiable:
it splits the image into 32x32 patches, computes each patch's SPECTRAL ENTROPY
via numpy FFT, ARGSORTS the entropies to pick the 49 lowest+highest patches, then
RANDOM-SHUFFLES them and stitches a 224x224 uint8 mosaic. FFT-on-numpy + argsort +
random.shuffle + uint8 re-quantization all break the gradient.

BPDA approximation used here:
  * The patch-selection + shuffle + stitch is computed ONCE on the CLEAN image and
    then held CONSTANT (identity w.r.t. the perturbation) for all PGD steps. The
    attack variable is the STITCHED 224x224 image in [0,1] (x0 = ToTensor(stitched)),
    which is exactly the tensor that feeds the first differentiable module.
  * The differentiable core we actually attack is:
        x -> ImageNet-normalize -> CLIP(LoRA) image encoder -> img_fc -> logit.
    (CLIPModel.forward(x, text=[]) returns (img_logit, img_logit); we take [0].)
This is the standard BPDA treatment: we perturb the pixels of the selected mosaic
and let CLIP's gradient guide them, keeping which/where patches were chosen fixed.
Consequence: the reported adversarial mosaic is a valid <=8/255 ell-inf perturbation
of the CLEAN mosaic (the detector's true input), NOT of the original photo. This is
the honest attackable surface given the non-diff selection; documented per contract.

Score = sigmoid(img_logit); pred threshold 0.5 (prob). label: 0=real,1=fake.
"""
import argparse
import csv
import importlib.util
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(HERE, "..", "adapters", "DGS-Net.py")

EPS = 8.0 / 255.0
ALPHA = 2.0 / 255.0
STEPS = 10
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_adapter():
    spec = importlib.util.spec_from_file_location("dgs_adapter", os.path.abspath(ADAPTER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # top of file inserts repo CODE dir on sys.path
    return mod


def pgd_attack(forward_logit, x0, label, device):
    """ell-inf PGD ascending BCE(true label). forward_logit: x[0,1]->logit tensor."""
    x0 = x0.to(device)
    y = torch.tensor([[float(label)]], device=device)
    bce = torch.nn.BCEWithLogitsLoss()

    delta = torch.empty_like(x0).uniform_(-EPS, EPS)
    x_adv = torch.clamp(x0 + delta, 0.0, 1.0).detach()
    for _ in range(STEPS):
        x_adv.requires_grad_(True)
        logit = forward_logit(x_adv).view(1, 1)
        loss = bce(logit, y)  # ascend -> push away from true label
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

    # Resolve paths BEFORE build_model: the adapter chdir's into the weights dir.
    manifest_path = os.path.abspath(args.manifest)
    out_path = os.path.abspath(args.out)

    device = torch.device(args.device)
    adapter = load_adapter()
    adapter.seed_all(100)
    model = adapter.build_model(device)  # chdir into weights dir; eval mode

    from PIL import Image
    from data.PatchSelectionTransform import PatchSelectionTransform

    patch_select = PatchSelectionTransform(patch_size=32, num_patches=49)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def forward_logit(x01):
        # NOTE: CLIPModel.forward(x, text=[]) wraps encode_image in torch.no_grad()
        # (an inference-only guard), which blocks gradients. We call the SAME two
        # modules directly to reproduce that path's math with autograd enabled:
        #     img_logit = img_fc(clip_model.encode_image(x))
        # (identical to model(x)[0]).
        x = (x01 - mean) / std
        features = model.clip_model.encode_image(x)
        return model.img_fc(features)  # img_logit

    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "score_clean", "pred_clean", "score_adv", "pred_adv"])
        for row in rows:
            path, label = row["path"], int(row["label"])
            try:
                img = Image.open(path).convert("RGB")
                # Fixed BPDA selection: compute mosaic once on the clean image.
                random.seed(100)
                stitched = patch_select(img)  # PIL 224x224 (selection held constant)
                x0 = torch.from_numpy(
                    np.asarray(stitched, dtype=np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0)  # [1,3,224,224] in [0,1]

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
                print(f"[DGS-Net] {os.path.basename(path)} label={label} "
                      f"clean={score_c:.4f} adv={score_a:.4f} linf={linf:.5f}")
            except Exception as e:
                print(f"[DGS-Net] error on {path}: {e}", file=sys.stderr)
                w.writerow([path, label, "", "", "", ""])
    print(f"[DGS-Net] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
