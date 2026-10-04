#!/usr/bin/env python
"""Benchmark adapter for FIRE (CVPR 2025) - Frequency-guided Reconstruction Error.

Official code:    detectors/2025/FIRE/code (eval.py, utils/network_utils.py)
Official weights: detectors/2025/FIRE/weights/imagenet_w_adm.pt
                  (full FIRE_model state dict: SD-v1.5 VAE + FMRE fft filter +
                   6-channel InstanceNorm ResNet-50 classifier - 318 keys).

The repo's FIRE_model hardcodes .to("cuda") and downloads the full SD v1.5
pipeline (runwayml repo, now delisted from HF) just to get the VAE.  Since the
official checkpoint already contains ALL VAE weights, this adapter rebuilds the
identical module graph (attribute names vae / resnet / fft_filter_module match
FIRE_model) on an arbitrary device, loads the checkpoint strictly, and
replicates FIRE_model.forward() line-for-line (fp32 everywhere; the repo casts
the VAE to fp16 on cuda only as a speed optimization).

Preprocessing: eval.py feeds 256x256 images (DiffusionForensics) with a bare
ToTensor (no normalization); the fft filter is built for 256x256 inputs.  For
arbitrary-sized benchmark images we use Resize(256) + CenterCrop(256) (the
CenterCrop(256) appears - commented out - in the repo's own transform stack)
followed by ToTensor.
Score: raw logit (monotone with the repo's sigmoid P(fake)); higher = fake.
       The sigmoid saturates (>0.9999) on out-of-domain images, producing tied
       scores at float precision, so the adapter emits the pre-sigmoid logit.
Note: VAE latents are sampled stochastically (retrieve_latents default), as in
the official forward; torch seed is fixed for reproducibility.
"""
import argparse
import csv
import os
import sys

CODE = "detectors/2025/FIRE/code"
CKPT = "detectors/2025/FIRE/weights/imagenet_w_adm.pt"
sys.path.insert(0, CODE)

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import diffusers
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from utils.network_utils import get_frq_resnet_model, fft_filter  # official code

BATCH_SIZE = 4

# Stable Diffusion v1.5 VAE architecture (vae/config.json of the official repo);
# every weight is loaded from the official FIRE checkpoint below (strict=True).
SD15_VAE_CONFIG = dict(
    in_channels=3,
    out_channels=3,
    down_block_types=("DownEncoderBlock2D",) * 4,
    up_block_types=("UpDecoderBlock2D",) * 4,
    block_out_channels=(128, 256, 512, 512),
    layers_per_block=2,
    act_fn="silu",
    latent_channels=4,
    norm_num_groups=32,
    sample_size=512,
    scaling_factor=0.18215,
)


class FIREWrapper(nn.Module):
    """Same module graph as utils.network_utils.FIRE_model, device-agnostic."""

    def __init__(self):
        super().__init__()
        self.vae = diffusers.AutoencoderKL(**SD15_VAE_CONFIG)
        for p in self.vae.parameters():
            p.requires_grad = False
        self.resnet = get_frq_resnet_model(mode="frq", norm_layer="instance",
                                           pretrained=False)
        self.fft_filter_module = fft_filter(radiuslow=40, radiushigh=120,
                                            rows=256, cols=256)

    def forward(self, x):
        # replicates FIRE_model.forward (utils/network_utils.py)
        decode_dtype = next(iter(self.vae.post_quant_conv.parameters())).dtype
        (middle_freq_image, middle_filtered_image,
         mask_mid_frq, mask_mid_filterd) = self.fft_filter_module(x)

        latents_x = retrieve_latents(self.vae.encode(x.to(decode_dtype)))
        reconstructions_x = self.vae.decode(latents_x.to(decode_dtype),
                                            return_dict=False)[0]

        latents_mf = retrieve_latents(
            self.vae.encode(middle_filtered_image.to(decode_dtype)))
        reconstructions_mf = self.vae.decode(latents_mf.to(decode_dtype),
                                             return_dict=False)[0]

        raw_delta = torch.abs(reconstructions_x - x)
        filtered_delta = torch.abs(reconstructions_mf - x)
        return self.resnet(torch.cat([raw_delta, filtered_delta], dim=1))


def parse_args():
    p = argparse.ArgumentParser(description="FIRE benchmark adapter")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def read_manifest(path, limit):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def main():
    args = parse_args()
    torch.manual_seed(0)
    rows = read_manifest(args.manifest, args.limit)
    device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    model = FIREWrapper()
    state_dict = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)  # fp16 vae weights cast to fp32
    model.to(device)
    model.eval()

    scores = [None] * len(rows)

    def flush(idx_buf, ten_buf):
        if not ten_buf:
            return
        batch = torch.stack(ten_buf).to(device)
        try:
            with torch.no_grad():
                out = model(batch).flatten().tolist()  # raw logit, monotone with sigmoid
            for i, s in zip(idx_buf, out):
                scores[i] = float(s)
        except Exception:
            for i, t in zip(idx_buf, ten_buf):
                try:
                    with torch.no_grad():
                        s = model(t.unsqueeze(0).to(device)).flatten().item()
                    scores[i] = float(s)
                except Exception as e:
                    print(f"[FIRE] error scoring {rows[i]['path']}: {e}", file=sys.stderr)

    idx_buf, ten_buf = [], []
    for i, row in enumerate(rows):
        try:
            img = Image.open(row["path"]).convert("RGB")
            ten_buf.append(transform(img))
            idx_buf.append(i)
        except Exception as e:
            print(f"[FIRE] error loading {row['path']}: {e}", file=sys.stderr)
        if len(ten_buf) >= BATCH_SIZE:
            flush(idx_buf, ten_buf)
            idx_buf, ten_buf = [], []
    flush(idx_buf, ten_buf)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generator", "gen_year", "label", "path", "score"])
        for row, s in zip(rows, scores):
            w.writerow([row["generator"], row["gen_year"], row["label"], row["path"],
                        "" if s is None else s])
    print(f"[FIRE] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
