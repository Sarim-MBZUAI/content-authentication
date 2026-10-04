#!/usr/bin/env python
"""Benchmark adapter for AEROBLADE (Ricker et al., CVPR 2024).

AEROBLADE is TRAINING-FREE. It detects latent-diffusion images by measuring the
autoencoder RECONSTRUCTION ERROR: an image is passed through a set of LDM
autoencoders (encode -> decode) and the LPIPS distance between the image and its
reconstruction is measured. Generated images already lie close to the AE
manifold, so they reconstruct with a LOW error; real images reconstruct with a
HIGHER error. The AEROBLADE distance is:

    d(x) = min over AEs of  LPIPS(x, AE_decode(AE_encode(x)))

That raw distance is LOW for fakes and HIGH for reals. This benchmark's contract
requires `score` with HIGHER = more likely FAKE, so this adapter outputs the
NEGATED reconstruction error:

    score = -min_over_AEs LPIPS(x, recon)

so fakes (low error) get a HIGH score. (This matches the repo's own convention:
in aeroblade.distances.LPIPS._postprocess the per-AE LPIPS is stored negated, and
`compute_max` takes the max over AEs == -min over AEs of the raw LPIPS.)

This adapter reuses the official repo code:
  - LPIPS is computed with the repo's own `_PatchedLPIPS` (aeroblade.distances),
    which returns layer-wise, non-upsampled LPIPS exactly as in the paper.
  - The reconstruction math mirrors `aeroblade.image.compute_reconstructions`
    (normalize to [-1,1], encode via `retrieve_latents`, VQModel vs KL decode,
    de-normalize). We load the AE once and run it in memory rather than using the
    repo's disk/hash/cpu-offload machinery, which reloads the pipeline per call.

Distance metric: `lpips_vgg_2` (VGG, 2nd LPIPS layer) -- the default of the repo's
entry script scripts/run_aeroblade.py, which the authors use for detection.

CLI (contract): --manifest <csv> --out <csv> [--device cuda|cpu] [--limit N]
Output columns (contract): generator,gen_year,label,path,score  (higher = FAKE)
"""

import argparse
import os
import sys
import types
import warnings
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Repo wiring
# --------------------------------------------------------------------------- #
REPO_ROOT = Path("detectors/2024/AEROBLADE/code")
REPO_SRC = REPO_ROOT / "src"

# HuggingFace cache / token for gated AE downloads (see report notes).
os.environ.setdefault("HF_HOME", "~/.cache/huggingface")

# Default AEs, in the repo's order. SD2-base is included but currently 404 on the
# Hub (repo removed by Stability); the adapter skips any AE that fails to load and
# takes the min over whatever AEs are available -- AEROBLADE works with any subset.
DEFAULT_AUTOENCODERS = [
    "CompVis/stable-diffusion-v1-1",        # SD1  (vae / AutoencoderKL)
    "stabilityai/stable-diffusion-2-base",  # SD2  (vae / AutoencoderKL) -- may 404
    "kandinsky-community/kandinsky-2-1",    # KD2.1 (movq / VQModel)
]

# LPIPS layer used as the reconstruction distance.
# distance_metric "lpips_vgg_2" -> repo layer index 2 -> res_no_up[1] (see below).
LPIPS_NET = "vgg"
LPIPS_LAYER = 2  # 1..5 select a single VGG layer; res_no_up index == LPIPS_LAYER-1
RECON_SEED = 1

sys.path.insert(0, str(REPO_SRC))
# aeroblade.distances imports `pyiqa` at module top-level but only uses it inside
# the unused PyIQADistance path; stub it so we can import `_PatchedLPIPS`.
if "pyiqa" not in sys.modules:
    sys.modules["pyiqa"] = types.ModuleType("pyiqa")

import torch  # noqa: E402
import torchvision.transforms.v2 as tf  # noqa: E402
from PIL import Image  # noqa: E402

from aeroblade.distances import _PatchedLPIPS  # noqa: E402  (official repo LPIPS)
from diffusers import AutoencoderKL  # noqa: E402
from diffusers.models import VQModel  # noqa: E402
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (  # noqa: E402
    retrieve_latents,
)

# Repo's default image transform (ImageFolder): PIL -> float32 tensor in [0,1].
# No resize -> "use the repo's own preprocessing" (contract).
TRANSFORM = tf.Compose([tf.ToImage(), tf.ToDtype(torch.float32, scale=True)])


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_lpips(device):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _PatchedLPIPS(spatial=True, net=LPIPS_NET)
    return model.to(device).eval()


def load_autoencoder(repo_id, device, dtype):
    """Load just the AE component (mirrors compute_reconstructions' AE extraction).

    Kandinsky uses a MoVQ (VQModel); Stable Diffusion uses a KL VAE (AutoencoderKL).
    """
    if "kandinsky-2" in repo_id:
        ae = VQModel.from_pretrained(repo_id, subfolder="movq", torch_dtype=dtype)
        kind = "movq/VQModel"
    else:
        ae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae", torch_dtype=dtype)
        kind = "vae/AutoencoderKL"
    return ae.to(device).eval(), kind


# --------------------------------------------------------------------------- #
# Core: reconstruction + LPIPS distance
# --------------------------------------------------------------------------- #
@torch.no_grad()
def reconstruct(ae, img, device, generator):
    """Encode/decode an image through the AE. Mirrors compute_reconstructions()."""
    x = img.to(device, dtype=ae.dtype) * 2.0 - 1.0
    latents = retrieve_latents(ae.encode(x), generator=generator)
    if isinstance(ae, VQModel):
        rec = ae.decode(latents.to(ae.dtype), force_not_quantize=True, return_dict=False)[0]
    else:
        rec = ae.decode(latents.to(ae.dtype), return_dict=False)[0]
    rec = (rec / 2 + 0.5).clamp(0, 1)
    return rec


@torch.no_grad()
def lpips_distance(lpips_model, img, rec, device):
    """Raw (positive) LPIPS_vgg_2 reconstruction distance between img and rec.

    The AE downsamples by its scale factor, so a reconstruction of an image whose
    H/W are not multiples of that factor comes back slightly smaller. Crop both to
    the common size before the perceptual comparison (minimal, content-preserving).
    """
    img = img.to(device, dtype=torch.float32)
    rec = rec.to(device, dtype=torch.float32)
    h = min(img.shape[2], rec.shape[2])
    w = min(img.shape[3], rec.shape[3])
    img = img[..., :h, :w]
    rec = rec[..., :h, :w]
    # retPerLayer -> (val=sum, res_no_up=[layer0..layer4]); repo's lpips_vgg_2
    # corresponds to res_no_up[LPIPS_LAYER-1], spatially averaged (spatial=False).
    _, per_layer = lpips_model(img, rec, retPerLayer=True, normalize=True)
    dist = per_layer[LPIPS_LAYER - 1].mean(dim=(1, 2, 3))  # [B] positive LPIPS
    return dist


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="AEROBLADE benchmark adapter.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--autoencoders", nargs="+", default=DEFAULT_AUTOENCODERS,
        help="HF repo ids of LDM autoencoders (min over the ones that load).",
    )
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[AEROBLADE] cuda requested but unavailable; falling back to cpu.")
        device = "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[AEROBLADE] device={device} dtype={dtype}")

    # Load manifest.
    df = pd.read_csv(args.manifest)
    if args.limit is not None:
        df = df.head(args.limit).copy()
    print(f"[AEROBLADE] {len(df)} rows to score.")

    # Load LPIPS + autoencoders once.
    lpips_model = load_lpips(device)
    autoencoders = {}
    for repo_id in args.autoencoders:
        try:
            ae, kind = load_autoencoder(repo_id, device, dtype)
            autoencoders[repo_id] = ae
            print(f"[AEROBLADE] loaded AE {repo_id} [{kind}]")
        except Exception as e:  # unreachable / gated / missing -> skip
            print(f"[AEROBLADE] SKIP AE {repo_id}: {type(e).__name__}: {str(e)[:160]}")
    if not autoencoders:
        raise RuntimeError("No autoencoders could be loaded; cannot run AEROBLADE.")
    print(f"[AEROBLADE] using {len(autoencoders)} AE(s): {list(autoencoders)}")

    generators = {
        repo_id: torch.Generator(device=device).manual_seed(RECON_SEED)
        for repo_id in autoencoders
    }

    scores = []
    for i, path in enumerate(df["path"].tolist()):
        try:
            img = TRANSFORM(Image.open(path).convert("RGB")).unsqueeze(0)
            per_ae = []
            for repo_id, ae in autoencoders.items():
                rec = reconstruct(ae, img, device, generators[repo_id])
                d = lpips_distance(lpips_model, img, rec, device)
                per_ae.append(float(d.item()))
            recon_error = min(per_ae)          # min LPIPS over AEs (low for fakes)
            score = -recon_error               # higher => more likely FAKE
            scores.append(score)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"[AEROBLADE] {i + 1}/{len(df)} score={score:.5f} "
                      f"(per-AE={['%.4f' % v for v in per_ae]})")
        except Exception as e:
            print(f"[AEROBLADE] ERROR on row {i} ({path}): "
                  f"{type(e).__name__}: {str(e)[:160]}")
            scores.append("")  # per-image error -> empty score, keep going

    out = df[["generator", "gen_year", "label", "path"]].copy()
    out["score"] = scores
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[AEROBLADE] wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
