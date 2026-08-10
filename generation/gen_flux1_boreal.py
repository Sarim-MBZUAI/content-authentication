#!/usr/bin/env python3
"""Generate the fixed benchmark set with FLUX.1-dev + Boreal v2.

The Boreal model card recommends the trigger word ``photo`` and warns that v2
is overtrained, so the default LoRA scale is deliberately below 1.0. Outputs
are written atomically and existing valid files are skipped, making the run
safe to resume after interruption.
"""

import argparse
import csv
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument(
        "--lora",
        default=".hf-boreal-flux1-adapter/boreal-v2.safetensors",
    )
    parser.add_argument("--trigger", default="photo")
    parser.add_argument("--lora-scale", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    return parser.parse_args()


def read_prompts(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"filename", "prompt"}.issubset(rows[0]):
        raise ValueError("prompts CSV must contain filename and prompt columns")
    filenames = [row["filename"] for row in rows]
    if any(not name for name in filenames) or len(filenames) != len(set(filenames)):
        raise ValueError("prompt filenames must be non-empty and unique")
    return rows


def valid_existing(path, expected_size):
    if not path.is_file():
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            return image.size == (expected_size, expected_size)
    except Exception:
        return False


def main():
    args = parse_args()

    import torch
    from diffusers import FluxPipeline

    rows = read_prompts(args.prompts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lora = Path(args.lora).resolve()
    if not lora.is_file():
        raise FileNotFoundError(f"Boreal LoRA not found: {lora}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print(f"{len(rows)} prompts", flush=True)
    print(f"loading FLUX.1-dev from {args.model}", flush=True)
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(
        str(lora.parent), weight_name=lora.name, adapter_name="boreal"
    )
    pipe.set_adapters(["boreal"], adapter_weights=[args.lora_scale])
    pipe.to("cuda")
    print(
        "pipeline ready "
        f"(trigger={args.trigger!r}, scale={args.lora_scale}, "
        f"steps={args.steps}, guidance={args.guidance})",
        flush=True,
    )

    generated = skipped = failed = 0
    for index, row in enumerate(rows, start=1):
        destination = out / row["filename"]
        if valid_existing(destination, args.size):
            skipped += 1
            print(f"[{index}/{len(rows)}] SKIP {destination.name}", flush=True)
            continue

        prompt = row["prompt"].strip()
        if args.trigger and args.trigger.lower() not in prompt.lower().split():
            prompt = f"{args.trigger}, {prompt}"
        started = time.time()
        temporary = destination.with_name(destination.name + ".partial.png")
        try:
            image = pipe(
                prompt=prompt,
                height=args.size,
                width=args.size,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
            ).images[0]
            image.save(temporary, format="PNG")
            temporary.replace(destination)
            generated += 1
            print(
                f"[{index}/{len(rows)}] {destination.name} "
                f"({time.time() - started:.1f}s)",
                flush=True,
            )
        except Exception as error:
            failed += 1
            temporary.unlink(missing_ok=True)
            print(
                f"[{index}/{len(rows)}] FAILED {destination.name}: "
                f"{type(error).__name__}: {str(error)[:300]}",
                flush=True,
            )

    print(
        f"DONE generated={generated} skipped={skipped} failed={failed}", flush=True
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
