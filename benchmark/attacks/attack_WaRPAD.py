#!/usr/bin/env python
"""White-box PGD (l-inf) attack harness for the WaRPAD detector.

Detector: WaRPAD (adapters/WaRPAD.py) -- TRAINING-FREE. A vanilla DINOv2 ViT-L/14 is
    used with no fine-tuned head. The image is resized to 896x896, ImageNet-normalized,
    unfolded into 16 non-overlapping 224x224 patches. For each patch a high-frequency
    residual is obtained by Haar DWT (J=2), zeroing the low-pass band and inverse-DWT;
    a perturbed patch = patch - 0.1*highfreq. The DINOv2 cls-token cosine similarity
    between patch and perturbed patch is averaged over the 16 patches.
        raw_sim  : HIGHER for REAL, LOWER for FAKE.
        score    : -mean_sim   (contract: higher = more likely FAKE).
    No natural 0.5 threshold -> decision boundary = MEDIAN of the clean scores over the
    batch (NOTE this; the full-run threshold should be calibrated on a proper set).

Attack (PGD_ATTACK_CONTRACT.md, FIXED): l-inf PGD, eps=8/255, 10 steps, alpha=2/255,
    random start. Ascend obj = direction*score, direction=+1 if label==0 else -1
    (fake -> push score down / increase similarity toward "real").

Differentiability / preprocessing:
    The perturbation lives in the [0,1] pixel space -- the 896x896 tensor after
    Resize(BICUBIC)+ToTensor, BEFORE ImageNet normalization. The PIL Resize is applied
    ONCE to build the clean pixel tensor (fixed preprocessing, outside the attacked ball);
    the mean/std Normalize, the unfold/patchify, the Haar DWT/IDWT (pytorch_wavelets, conv
    based) and the two DINOv2 forward passes are ALL differentiable -- verified that DWT
    passes gradients -- so this is a FULLY END-TO-END white-box attack (NO BPDA / NO
    approximation).

CLI (contract):  --manifest <csv> --out <csv> [--device cuda|cpu] [--limit N]
Output columns (contract): path,label,score_clean,pred_clean,score_adv,pred_adv
"""
import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

# Import the adapter first: it sets TORCH_HOME + sys.path for the DINOv2 hub cache and
# exposes the exact WaRPAD constants. We reuse them and load the same model/preproc.
ADAPTERS = str(Path(__file__).resolve().parent.parent / "adapters")
sys.path.insert(0, ADAPTERS)
import WaRPAD as adapter  # noqa: E402

import torch  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402
from PIL import Image, ImageFile  # noqa: E402
from pytorch_wavelets import DWTForward, DWTInverse  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

PREP_SIZE = adapter.PREP_SIZE
PATCH_SIZE = adapter.PATCH_SIZE
NOISE_LEVEL = adapter.NOISE_LEVEL
SEED = adapter.SEED

EPS = 8.0 / 255.0
ALPHA = 2.0 / 255.0
STEPS = 10


def pgd(score_fn, x_clean, label):
    x_clean = x_clean.detach()
    direction = 1.0 if int(label) == 0 else -1.0
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
    ap = argparse.ArgumentParser(description="WaRPAD PGD attack harness")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WaRPAD-atk] cuda unavailable; using cpu.")
        device = "cpu"
    device = torch.device(device)
    print(f"[WaRPAD-atk] device={device} eps={EPS:.5f} steps={STEPS} alpha={ALPHA:.5f}")

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    print(f"[WaRPAD-atk] {len(rows)} rows.")

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    dwt = DWTForward(J=2, wave="haar").to(device)
    idwt = DWTInverse(wave="haar").to(device)
    for m in (dwt, idwt):
        for p in m.parameters():
            p.requires_grad_(False)

    # Geometric preprocessing (PIL) up to [0,1] tensor; Normalize done differentiably.
    geom = transforms.Compose([
        transforms.Resize((PREP_SIZE, PREP_SIZE), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def score_fn(x):
        """x: [0,1] pixel tensor [B,3,896,896]; returns -mean_sim (higher=fake), [B]."""
        xn = (x - mean) / std
        b, c, h, w = xn.shape
        patches = xn.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        patches = patches.reshape(b, c, -1, PATCH_SIZE, PATCH_SIZE).transpose(1, 2)
        patches = patches.reshape(-1, c, PATCH_SIZE, PATCH_SIZE)
        yl, yh = dwt(patches)
        pert_hf = idwt((torch.zeros_like(yl), yh))
        perturbed = patches - NOISE_LEVEL * pert_hf
        out = model.forward_features(patches, None)["x_norm_clstoken"]
        pout = model.forward_features(perturbed, None)["x_norm_clstoken"]
        sim = torch.nn.functional.cosine_similarity(out, pout, dim=-1).reshape(b, -1)
        return -sim.mean(dim=1)  # higher = fake

    results = []  # (score_clean, score_adv)
    for i, row in enumerate(rows):
        try:
            img = Image.open(row["path"]).convert("RGB")
            x_clean = geom(img).unsqueeze(0).to(device)
            with torch.no_grad():
                s_clean = float(score_fn(x_clean)[0].item())
            x_adv = pgd(score_fn, x_clean, row["label"])
            linf = (x_adv - x_clean).abs().max().item()
            assert linf <= EPS + 1e-5, f"linf {linf} > eps {EPS}"
            with torch.no_grad():
                s_adv = float(score_fn(x_adv)[0].item())
            results.append((s_clean, s_adv))
            print(f"[WaRPAD-atk] {i+1}/{len(rows)} label={row['label']} "
                  f"clean={s_clean:.5f} adv={s_adv:.5f} linf={linf:.5f}")
        except Exception as e:
            print(f"[WaRPAD-atk] ERROR row {i} ({row['path']}): {type(e).__name__}: {str(e)[:160]}")
            results.append((None, None))

    # NOTE: WaRPAD has no natural threshold -> use MEDIAN of clean scores as boundary.
    clean_scores = [sc for sc, _ in results if sc is not None]
    threshold = statistics.median(clean_scores) if clean_scores else 0.0
    print(f"[WaRPAD-atk] median-of-clean threshold = {threshold:.6f} "
          f"(NOTE: batch-median boundary, not a calibrated 0.5)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "score_clean", "pred_clean", "score_adv", "pred_adv"])
        for row, (sc, sa) in zip(rows, results):
            if sc is None:
                w.writerow([row["path"], row["label"], "", "", "", ""])
            else:
                pc = int(sc > threshold)
                pa = int(sa > threshold)
                w.writerow([row["path"], row["label"], sc, pc, sa, pa])
    print(f"[WaRPAD-atk] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
