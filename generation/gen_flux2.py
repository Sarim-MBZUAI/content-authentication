#!/usr/bin/env python3
"""Generate benchmark images with FLUX.2-dev from prompts.csv.

Resumable: already-existing outputs are skipped. Filenames in prompts.csv are
reused for outputs so images stay prompt-matched with the real set.

Usage:
  python gen_flux2.py --prompts prompts.csv --out out/FLUX.2/fake \
      [--model black-forest-labs/FLUX.2-dev | /local/path/FLUX.2-dev] \
      [--seed 42] [--size 1024] [--steps 50] [--guidance 4.0]

Requires: diffusers>=0.39, HF_TOKEN with the black-forest-labs/FLUX.2-dev
license gate accepted. VRAM: the full bf16 pipeline is ~110 GB; on smaller
GPUs the script enables model CPU offload automatically (slower but works
on ~48 GB+; use --no-offload to keep everything on-GPU on large cards).
"""
import argparse, csv, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="black-forest-labs/FLUX.2-dev")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--guidance", type=float, default=4.0)
ap.add_argument("--no-offload", action="store_true")
args = ap.parse_args()

import torch
from diffusers import Flux2Pipeline

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(args.prompts)))
print(f"{len(rows)} prompts", flush=True)

print("loading FLUX.2 pipeline...", flush=True)
pipe = Flux2Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
if args.no_offload or total_vram > 130:
    pipe = pipe.to("cuda")
else:
    pipe.enable_model_cpu_offload()
print(f"pipeline ready (vram={total_vram:.0f}GB, offload={not (args.no_offload or total_vram > 130)})", flush=True)

done = skip = fail = 0
for i, r in enumerate(rows):
    dst = out / r["filename"]
    if dst.exists():
        skip += 1; continue
    t0 = time.time()
    try:
        img = pipe(prompt=r["prompt"], height=args.size, width=args.size,
                   num_inference_steps=args.steps, guidance_scale=args.guidance,
                   generator=torch.Generator("cuda").manual_seed(args.seed)).images[0]
        img.save(dst)
        done += 1
        print(f"[{i+1}/{len(rows)}] {dst.name} ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        fail += 1
        print(f"[{i+1}/{len(rows)}] FAILED {dst.name}: {type(e).__name__}: {str(e)[:200]}", flush=True)

print(f"DONE generated={done} skipped={skip} failed={fail}", flush=True)
if fail:
    sys.exit(1)
