#!/usr/bin/env python3
"""Generate the 100-prompt benchmark set with FLUX.2-dev + Boreal dev2 LoRA (fused).
Identical to gen_flux2_plain_100.py except the Boreal LoRA is loaded and fused into
the transformer right after the pipeline is created (before device placement).
Prompts verbatim from prompts.csv. seed 42, 1024x1024, 50 steps, guidance 4.0.
Resumable (skips already-generated outputs).

LoRA API note: diffusers 0.39.0 Flux2Pipeline subclasses Flux2LoraLoaderMixin
(diffusers/loaders/lora_pipeline.py), which provides both load_lora_weights() and
fuse_lora(), so the same fuse workflow as the FLUX.1 script is supported.
Caveat: the Boreal LoRA (boreal-flux-dev2-diffusers.safetensors) was trained for
FLUX.1-dev. Its key layout may not fully match the FLUX.2 transformer; if
load_lora_weights raises on unexpected/missing keys at run time, drop fuse_lora()
and/or pass low_cpu_mem_usage, or load without fusing.
"""
import csv, os, shutil, time
import torch
from diffusers import Flux2Pipeline

MODEL = "/shared/home/sarim.hashmi/usenix/generators/FLUX.2-dev"
LORA = "/shared/home/sarim.hashmi/usenix/generators/boreal-dev2/boreal-flux-dev2-diffusers.safetensors"
OUT = "/shared/home/sarim.hashmi/usenix/inverted_stuff/FLUX2-boreal-photoreal"
PROMPTS = "/shared/home/sarim.hashmi/usenix/liars-dividend-generation/generation/prompts_photoreal.csv"
MANIFEST = "/shared/home/sarim.hashmi/usenix/benchmark/manifest.csv"

VRAM_THRESHOLD_GB = 130

os.makedirs(f"{OUT}/fake", exist_ok=True)
os.makedirs(f"{OUT}/real", exist_ok=True)

# Copy the 100 matched reals (FLUX-dev label=0 rows from the manifest).
reals = {os.path.basename(r["path"]): r["path"] for r in csv.DictReader(open(MANIFEST))
         if r["generator"] == "FLUX-dev" and r["label"] == "0"}
for name, src in reals.items():
    dst = f"{OUT}/real/{name}"
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
print(f"copied {len(reals)} real images", flush=True)

rows = list(csv.DictReader(open(PROMPTS)))
print(f"{len(rows)} prompts", flush=True)

print("loading FLUX.2-dev + Boreal dev2 LoRA...", flush=True)
pipe = Flux2Pipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

# Flux2Pipeline supports load_lora_weights + fuse_lora via Flux2LoraLoaderMixin.
# Fuse before device placement / offload so the merged weights travel with the model.
pipe.load_lora_weights(LORA)
pipe.fuse_lora()

total_vram_gb = sum(
    torch.cuda.get_device_properties(i).total_memory
    for i in range(torch.cuda.device_count())
) / (1024 ** 3)
print(f"total GPU VRAM: {total_vram_gb:.1f} GB", flush=True)
if total_vram_gb < VRAM_THRESHOLD_GB:
    print("VRAM < 130 GB -> enable_model_cpu_offload()", flush=True)
    pipe.enable_model_cpu_offload()
    gen_device = "cpu"  # generator on CPU when offloading
else:
    print("VRAM >= 130 GB -> pipe.to('cuda')", flush=True)
    pipe.to("cuda")
    gen_device = "cuda"
print("pipeline ready (LoRA fused)", flush=True)

done = skip = fail = 0
for i, r in enumerate(rows):
    dst = f"{OUT}/fake/{r['filename']}"
    if os.path.exists(dst):
        skip += 1; continue
    t0 = time.time()
    try:
        img = pipe(prompt=r["prompt"], height=1024, width=1024,
                   num_inference_steps=50, guidance_scale=4.0,
                   generator=torch.Generator(gen_device).manual_seed(42)).images[0]
        img.save(dst)
        done += 1
        print(f"[{i+1}/{len(rows)}] {r['filename']} ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        fail += 1
        print(f"[{i+1}/{len(rows)}] FAILED {r['filename']}: {type(e).__name__}: {str(e)[:200]}", flush=True)

print(f"DONE generated={done} skipped={skip} failed={fail}", flush=True)
