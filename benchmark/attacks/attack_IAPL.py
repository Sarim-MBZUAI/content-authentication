#!/usr/bin/env python
"""PGD ell-inf attack harness for IAPL (2026) - prompt-learned CLIP ViT-L/14.

Spec: ell-inf PGD, eps = 8/255, 10 steps, alpha = 2/255, random start; ascend the
TRUE-label loss (flip). Perturbation in [0,1] pixel space BEFORE ImageNet
normalization (normalization done differentiably inside the graph).

================= DIFFERENTIABILITY =================
IAPL's non-adaptive forward (adapters/IAPL.py -> models/clip_models.py::CLIPModel
.forward, eval mode) is END-TO-END DIFFERENTIABLE -> NO BPDA / NO approximation.
Verified in code/models:
  * The frequency branch is DCT_Condition_Module (models/dct.py). Its "DCT" is a
    fixed-matrix matmul (_DCT_patch @ x @ _DCT_patch_T) -> differentiable. It does
    call torch.sort to pick top grade patches, but selection is via torch.gather on
    the sorted indices: gradient still flows through the GATHERED values (piecewise
    differentiable, exactly like max-pooling). So it is attackable end-to-end; the
    selection is recomputed each PGD step on the current image (this is a true
    gradient path, not a held-constant BPDA identity).
  * pytorch_wavelets DWTForward / cv2 are imported at module top but are NOT used in
    the CLIPModel.forward path (only in the unused freq_stem), so no non-diff op is
    hit. The wavelet transform is a linear conv anyway; it never executes here.
  * Rest of graph: MaPLe prompt tokens + CLIP ViT-L/14 visual encoder + fc_binary ->
    logit. All differentiable.
Attack target: logit; score = sigmoid(logit); pred threshold 0.5. label: 0=real,1=fake.
"""
import argparse
import csv
import importlib.util
import os
import sys

import torch
import torchvision.transforms as transforms

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(HERE, "..", "adapters", "IAPL.py")

EPS = 8.0 / 255.0
ALPHA = 2.0 / 255.0
STEPS = 10


def load_adapter():
    spec = importlib.util.spec_from_file_location("iapl_adapter", os.path.abspath(ADAPTER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # inserts repo CODE dir on sys.path
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
    model, transform = adapter.build_model(device)  # eval mode; returns test transform

    from PIL import Image

    # Split the repo test transform: reuse Resize/CenterCrop/ToTensor to get [0,1]
    # pixels, and apply the trailing Normalize as a differentiable op in the graph.
    tlist = list(transform.transforms)
    assert isinstance(tlist[-1], transforms.Normalize), "expected Normalize last"
    norm = tlist[-1]
    pre = transforms.Compose(tlist[:-1])  # -> [0,1] tensor
    mean = torch.tensor(norm.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(norm.std, device=device).view(1, 3, 1, 1)

    def forward_logit(x01):
        x = (x01 - mean) / std
        return model(x).view(1, 1)  # eval -> logit

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
                x0 = pre(img).unsqueeze(0)  # [1,3,224,224] in [0,1]

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
                print(f"[IAPL] {os.path.basename(path)} label={label} "
                      f"clean={score_c:.4f} adv={score_a:.4f} linf={linf:.5f}")
            except Exception as e:
                print(f"[IAPL] error on {path}: {e}", file=sys.stderr)
                w.writerow([path, label, "", "", "", ""])
    print(f"[IAPL] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
