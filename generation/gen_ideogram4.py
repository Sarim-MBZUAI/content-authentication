#!/usr/bin/env python3
"""Generate benchmark images with Ideogram 4 (fp8) from prompts.csv.

Resumable: already-existing outputs are skipped. Filenames in prompts.csv are
reused for outputs so images stay prompt-matched with the real set.

Usage:
  python gen_ideogram4.py --prompts prompts.csv --out out/Ideogram4/fake \
      [--local-weights /path/to/ideogram-4-fp8] [--seed 42] [--size 1024]

Requires: the `ideogram4` package (pip install git+https://github.com/ideogram-oss/ideogram4),
HF_TOKEN with the ideogram-ai/ideogram-4-fp8 gate accepted (license click on the model page).
"""
import argparse, csv, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--local-weights", default=None, help="optional local snapshot dir of ideogram-ai/ideogram-4-fp8")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--preset", default="V4_QUALITY_48")
ap.add_argument("--limit", type=int, default=None, help="generate only the first N prompts")
args = ap.parse_args()

def structured_caption(prompt):
    """Wrap a benchmark caption in Ideogram 4's required JSON schema."""
    return json.dumps(
        {
            "high_level_description": prompt,
            "compositional_deconstruction": {
                "background": "A natural scene matching the description, without captions, labels, logos, signatures, watermarks, or other added text.",
                "elements": [],
            },
        },
        ensure_ascii=False, separators=(",", ":"),
    )

if args.local_weights:
    LOCAL = Path(args.local_weights)
    def local_hf_hub_download(repo_id=None, filename=None, **kw):
        p = LOCAL / filename
        if not p.exists():
            from huggingface_hub.errors import EntryNotFoundError
            raise EntryNotFoundError(f"not in local snapshot: {filename}")
        return str(p)
    import ideogram4.pipeline_ideogram4 as pi
    import ideogram4.quantized_loading as ql
    pi.hf_hub_download = local_hf_hub_download
    if hasattr(ql, "hf_hub_download"):
        ql.hf_hub_download = local_hf_hub_download

import torch
from ideogram4.pipeline_ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig
from ideogram4.sampler_configs import PRESETS

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(args.prompts)))
if args.limit is not None:
    rows = rows[:args.limit]
print(f"{len(rows)} prompts", flush=True)

print("loading pipeline (fp8)...", flush=True)
pipe = Ideogram4Pipeline.from_pretrained(
    config=Ideogram4PipelineConfig(weights_repo="ideogram-ai/ideogram-4-fp8"),
    device="cuda", dtype=torch.bfloat16,
)
preset = PRESETS[args.preset]
print("pipeline ready", flush=True)

done = skip = fail = 0
for i, r in enumerate(rows):
    dst = out / r["filename"]
    if dst.exists():
        skip += 1; continue
    t0 = time.time()
    try:
        caption = structured_caption(r["prompt"])
        images = pipe(prompts=caption, height=args.size, width=args.size,
                      num_steps=preset.num_steps,
                      guidance_schedule=preset.guidance_schedule,
                      mu=preset.mu, std=preset.std, seed=args.seed,
                      raise_on_caption_issues=True)
        images[0].save(dst)
        done += 1
        print(f"[{i+1}/{len(rows)}] {dst.name} ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        fail += 1
        print(f"[{i+1}/{len(rows)}] FAILED {dst.name}: {type(e).__name__}: {str(e)[:200]}", flush=True)

print(f"DONE generated={done} skipped={skip} failed={fail}", flush=True)
if fail:
    sys.exit(1)
