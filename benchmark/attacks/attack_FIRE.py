#!/usr/bin/env python
"""PGD ell-inf attack harness for FIRE (CVPR 2025) - freq-guided VAE recon error.

Spec: ell-inf PGD, eps = 8/255, 10 steps, alpha = 2/255, random start; ascend the
TRUE-label loss (flip). Perturbation in [0,1] pixel space (FIRE feeds a bare
ToTensor with NO mean/std normalization, so [0,1] is exactly the model input).

================= DIFFERENTIABILITY / COST =================
FIRE's forward (adapters/FIRE.py::FIREWrapper, mirroring utils.network_utils
.FIRE_model.forward) is END-TO-END DIFFERENTIABLE and we attack it END-TO-END
through the full reconstruction graph -- NO BPDA:
  x -> fft_filter (torch.fft middle-band filter, differentiable)
    -> SD-1.5 VAE encode+decode of x AND of the filtered image (2x encode+decode)
    -> |recon_x - x| and |recon_mf - x| stacked -> 6-ch InstanceNorm ResNet-50 -> logit.
All ops are torch (torch.fft.*, AutoencoderKL enc/dec, conv). The gradient flows
through the whole VAE reconstruction path.

Approximations / notes (documented per contract):
  * retrieve_latents() SAMPLES the VAE posterior (mean + std*eps) stochastically.
    The reparameterization is differentiable, but the random eps would make PGD
    steps inconsistent. We therefore FIX torch's RNG (manual_seed(0)) at the start
    of every forward so the sampled eps is identical across clean scoring and all
    PGD steps -> a deterministic, consistent gradient surface. (Same seed the
    adapter uses for scoring.)
  * COST: on CPU this is heavy -- each forward is 2 full SD-1.5 VAE encode+decode
    passes at 256x256; a 10-step PGD needs 10 forward + 10 backward of that graph
    per image. It is feasible at --limit 4 (a few minutes/image on CPU-limit-4) but
    the full 2000-image run should be done on a GPU.

Score = raw logit (higher = fake); pred threshold 0.0 (logit). label: 0=real,1=fake.
"""
import argparse
import csv
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(HERE, "..", "adapters", "FIRE.py")

EPS = 8.0 / 255.0
ALPHA = 2.0 / 255.0
STEPS = 10


def load_adapter():
    spec = importlib.util.spec_from_file_location("fire_adapter", os.path.abspath(ADAPTER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # sets sys.path to repo CODE dir, imports diffusers
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

    import torchvision.transforms as transforms
    from PIL import Image

    model = adapter.FIREWrapper()
    state = torch.load(adapter.CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(256),
        transforms.ToTensor(),  # [0,1], no normalization (FIRE convention)
    ])

    def forward_logit(x01):
        torch.manual_seed(0)  # fix VAE posterior sampling -> consistent gradient
        return model(x01).view(1, 1)

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
                x0 = transform(img).unsqueeze(0)  # [1,3,256,256] in [0,1]

                with torch.no_grad():
                    logit_c = forward_logit(x0.to(device)).view(-1)[0].item()

                x_adv = pgd_attack(forward_logit, x0, label, device)
                linf = (x_adv - x0.to(device)).abs().max().item()
                with torch.no_grad():
                    logit_a = forward_logit(x_adv).view(-1)[0].item()

                w.writerow([path, label, f"{logit_c:.6f}", int(logit_c >= 0.0),
                            f"{logit_a:.6f}", int(logit_a >= 0.0)])
                print(f"[FIRE] {os.path.basename(path)} label={label} "
                      f"clean={logit_c:.4f} adv={logit_a:.4f} linf={linf:.5f}")
            except Exception as e:
                print(f"[FIRE] error on {path}: {e}", file=sys.stderr)
                w.writerow([path, label, "", "", "", ""])
    print(f"[FIRE] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
