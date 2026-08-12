#!/usr/bin/env python
"""White-box PGD (l-inf) attack harness for the AEROBLADE detector.

Detector: AEROBLADE (adapters/AEROBLADE.py) -- TRAINING-FREE. It passes an image through
    one or more latent-diffusion autoencoders (encode->decode) and measures the LPIPS
    reconstruction error. Generated images lie near the AE manifold (LOW error); real
    images reconstruct with HIGHER error.
        recon_error(x) = min over AEs of  LPIPS(x, AE_decode(AE_encode(x)))
        score          = -recon_error         (contract: higher = more likely FAKE).
    No natural 0.5 threshold -> decision boundary = MEDIAN of the clean scores over the
    batch (NOTE this; the full-run threshold should be calibrated separately).

Attack (PGD_ATTACK_CONTRACT.md, FIXED): l-inf PGD, eps=8/255, 10 steps, alpha=2/255,
    random start. Ascend obj = direction*score, direction=+1 if label==0 else -1.
    For a FAKE (label 1) direction=-1 => ascend (-score) = ascend recon_error, i.e. the
    attack INCREASES the reconstruction error to push the (negated) score DOWN toward
    "real". For a REAL (label 0) it DECREASES recon error to push the score up.

Differentiability / preprocessing:
    The perturbation lives in the [0,1] pixel space (the tensor fed to the AE, before the
    internal [-1,1] scaling). The VAE encode/decode and the repo's LPIPS (VGG) are all
    differentiable, so this is a FULLY END-TO-END white-box attack. One deliberate change
    vs the adapter's SCORING path: latents are taken as the distribution MODE
    (retrieve_latents sample_mode="argmax") instead of a seeded random .sample(), so the
    graph is deterministic and gives stable gradients. score_clean/score_adv here are both
    computed with this same deterministic graph, so their comparison is exact; absolute
    values differ negligibly from the adapter's sampled score. This is the only
    approximation. This is HEAVY on CPU (2 VAE round-trips + LPIPS + backward per step).

CLI (contract):  --manifest <csv> --out <csv> [--device cuda|cpu] [--limit N]
Output columns (contract): path,label,score_clean,pred_clean,score_adv,pred_adv
"""
import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

# Import the adapter (loads repo path, LPIPS + AE loaders, transform). main() is guarded.
ADAPTERS = str(Path(__file__).resolve().parent.parent / "adapters")
sys.path.insert(0, ADAPTERS)
import AEROBLADE as adapter  # noqa: E402

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from diffusers.models import VQModel  # noqa: E402

EPS = 8.0 / 255.0
ALPHA = 2.0 / 255.0
STEPS = 10
SEED = 1
LPIPS_LAYER = adapter.LPIPS_LAYER


def reconstruct_diff(ae, img):
    """Differentiable encode->decode. img in [0,1]; returns reconstruction in [0,1]."""
    x = (img * 2.0 - 1.0).to(ae.dtype)
    latents = adapter.retrieve_latents(ae.encode(x), sample_mode="argmax")
    if isinstance(ae, VQModel):
        rec = ae.decode(latents.to(ae.dtype), force_not_quantize=True, return_dict=False)[0]
    else:
        rec = ae.decode(latents.to(ae.dtype), return_dict=False)[0]
    return (rec / 2 + 0.5).clamp(0, 1)


def lpips_diff(lpips_model, img, rec):
    """Differentiable positive LPIPS_vgg_<LPIPS_LAYER> reconstruction distance, [B]."""
    img = img.to(torch.float32)
    rec = rec.to(torch.float32)
    h = min(img.shape[2], rec.shape[2])
    w = min(img.shape[3], rec.shape[3])
    img = img[..., :h, :w]
    rec = rec[..., :h, :w]
    _, per_layer = lpips_model(img, rec, retPerLayer=True, normalize=True)
    return per_layer[LPIPS_LAYER - 1].mean(dim=(1, 2, 3))  # [B] positive LPIPS


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
    ap = argparse.ArgumentParser(description="AEROBLADE PGD attack harness")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--autoencoders", nargs="+", default=adapter.DEFAULT_AUTOENCODERS)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[AEROBLADE-atk] cuda unavailable; using cpu.")
        device = "cpu"
    dtype = torch.float32  # attack runs on cpu in fp32 for stable gradients
    print(f"[AEROBLADE-atk] device={device} dtype={dtype} eps={EPS:.5f} steps={STEPS}")

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    print(f"[AEROBLADE-atk] {len(rows)} rows.")

    lpips_model = adapter.load_lpips(device)
    for p in lpips_model.parameters():
        p.requires_grad_(False)
    autoencoders = {}
    for repo_id in args.autoencoders:
        try:
            ae, kind = adapter.load_autoencoder(repo_id, device, dtype)
            for p in ae.parameters():
                p.requires_grad_(False)
            autoencoders[repo_id] = ae
            print(f"[AEROBLADE-atk] loaded AE {repo_id} [{kind}]")
        except Exception as e:
            print(f"[AEROBLADE-atk] SKIP AE {repo_id}: {type(e).__name__}: {str(e)[:160]}")
    if not autoencoders:
        raise RuntimeError("No autoencoders could be loaded; cannot run AEROBLADE.")
    print(f"[AEROBLADE-atk] using {len(autoencoders)} AE(s): {list(autoencoders)}")

    def score_fn(x):
        dists = [lpips_diff(lpips_model, x, reconstruct_diff(ae, x)) for ae in autoencoders.values()]
        recon_error = torch.stack(dists, 0).min(0).values  # min over AEs, [B]
        return -recon_error  # higher = fake

    results = []
    for i, row in enumerate(rows):
        try:
            img = adapter.TRANSFORM(Image.open(row["path"]).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                s_clean = float(score_fn(img)[0].item())
            x_adv = pgd(score_fn, img, row["label"])
            linf = (x_adv - img).abs().max().item()
            assert linf <= EPS + 1e-5, f"linf {linf} > eps {EPS}"
            with torch.no_grad():
                s_adv = float(score_fn(x_adv)[0].item())
            results.append((s_clean, s_adv))
            print(f"[AEROBLADE-atk] {i+1}/{len(rows)} label={row['label']} "
                  f"clean={s_clean:.5f} adv={s_adv:.5f} linf={linf:.5f}")
        except Exception as e:
            print(f"[AEROBLADE-atk] ERROR row {i} ({row['path']}): {type(e).__name__}: {str(e)[:160]}")
            results.append((None, None))

    # NOTE: AEROBLADE has no natural threshold -> use MEDIAN of clean scores as boundary.
    clean_scores = [sc for sc, _ in results if sc is not None]
    threshold = statistics.median(clean_scores) if clean_scores else 0.0
    print(f"[AEROBLADE-atk] median-of-clean threshold = {threshold:.6f} "
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
    print(f"[AEROBLADE-atk] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
