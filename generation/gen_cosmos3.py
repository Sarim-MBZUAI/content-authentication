#!/usr/bin/env python3
"""Generate the benchmark set with NVIDIA Cosmos3-Super-Text2Image."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler


ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="nvidia/Cosmos3-Super-Text2Image")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--guidance", type=float, default=4.0)
ap.add_argument("--flow-shift", type=float, default=3.0)
ap.add_argument("--device-map", default="balanced")
ap.add_argument("--limit", type=int)
args = ap.parse_args()


def structured_prompt(prompt: str, size: int) -> str:
    """Create NVIDIA's deterministic text-to-image JSON prompt locally."""
    data = {
        "subjects": [{"description": prompt}],
        "subject_details": {"source_caption": prompt},
        "background_setting": (
            "A coherent real-world environment consistent with the source caption."
        ),
        "lighting": {
            "conditions": "Natural, scene-appropriate lighting",
            "direction": "Consistent with the depicted environment",
            "shadows": "Realistic scene-appropriate shadows",
            "illumination_effect": "Balanced exposure and clear visible detail",
        },
        "aesthetics": {
            "composition": "Natural documentary composition",
            "color_scheme": "Colors consistent with the source caption",
            "mood_atmosphere": "Natural everyday realism",
            "patterns": "None unless described by the source caption",
        },
        "cinematography": {
            "framing": "Scene-appropriate framing",
            "camera_angle": "Eye-level",
            "depth_of_field": "Natural depth of field",
            "focus": "Primary subjects and actions",
            "lens_focal_length": "Natural perspective",
        },
        "style_medium": "Photography",
        "artistic_style": "Naturalistic documentary photography",
        "context": "A realistic depiction of the source caption",
        "text_and_signage_elements": [],
        "quadrant_scan": {
            "top_left": "Scene-consistent content",
            "top_right": "Scene-consistent content",
            "bottom_left": "Scene-consistent content",
            "bottom_right": "Scene-consistent content",
            "absolute_center": "Primary scene content",
        },
        "comprehensive_t2i_caption": prompt,
        "resolution": {"H": size, "W": size},
        "aspect_ratio": "1:1",
    }
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
with open(args.prompts, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if args.limit is not None:
    rows = rows[: args.limit]

print(f"loading {args.model} across device_map={args.device_map}", flush=True)
pipe = Cosmos3OmniPipeline.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map=args.device_map,
    enable_safety_checker=True,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config,
    flow_shift=args.flow_shift,
)
print("pipeline ready", flush=True)

done = skip = fail = 0
for index, row in enumerate(rows, start=1):
    filename = Path(row["filename"])
    if filename.name != row["filename"]:
        raise ValueError(f"unsafe output filename: {row['filename']!r}")
    destination = out / filename
    if destination.exists():
        skip += 1
        continue

    started = time.time()
    try:
        result = pipe(
            prompt=structured_prompt(row["prompt"], args.size),
            negative_prompt="",
            num_frames=1,
            height=args.size,
            width=args.size,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=torch.Generator(device="cuda:0").manual_seed(args.seed),
        )
        image = result.video[0]
        temporary = destination.with_suffix(destination.suffix + ".part")
        image.save(temporary, format="PNG")
        temporary.replace(destination)
        done += 1
        print(
            f"[{index}/{len(rows)}] {destination.name} "
            f"({time.time() - started:.1f}s)",
            flush=True,
        )
    except Exception as error:
        fail += 1
        print(
            f"[{index}/{len(rows)}] FAILED {destination.name}: "
            f"{type(error).__name__}: {str(error)[:300]}",
            flush=True,
        )

print(f"DONE generated={done} skipped={skip} failed={fail}", flush=True)
if fail:
    sys.exit(1)
