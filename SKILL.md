# SKILL: Generate the 2025/2026 benchmark image sets (FLUX.2-dev + Ideogram 4)

Goal: produce 100 synthetic images per generator from `generation/prompts.csv`
(the same 100 COCO-style prompts as the existing SD2.1/SD3/SD3.5/FLUX.1 sets),
prompt-matched by filename, seed 42, 1024x1024. Outputs feed an era-matched
deepfake-detection benchmark, so **determinism and naming matter**.

## 0. Requirements

- 1 CUDA GPU. Ideogram 4 fp8 needs ~30 GB VRAM. FLUX.2-dev bf16 needs ~110 GB
  on-GPU or ~48 GB+ with the automatic CPU offload the script enables.
- ~150 GB disk for model weights (FLUX.2 ~113 GB, Ideogram 4 ~28 GB).
- A Hugging Face token in `HF_TOKEN` whose account has accepted BOTH license
  gates (one click each, auto-approved, non-commercial licenses):
  - https://huggingface.co/black-forest-labs/FLUX.2-dev
  - https://huggingface.co/ideogram-ai/ideogram-4-fp8

## 1. Environment

```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
# pick the torch CUDA index matching the machine (cu121/cu124/cu128):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "diffusers>=0.39" "transformers>=5" accelerate safetensors sentencepiece protobuf pillow
pip install git+https://github.com/ideogram-oss/ideogram4   # Ideogram 4 inference code
export HF_TOKEN=hf_...   # token with both gates accepted
```

## 2. Run Ideogram 4 (2026 generator)

```bash
python generation/gen_ideogram4.py \
  --prompts generation/prompts.csv \
  --out out/Ideogram4/fake
```

- Downloads `ideogram-ai/ideogram-4-fp8` on first run (~28 GB).
- ~15-40 s/image (48-step quality preset) -> roughly 30-70 min for 100.
- Notes: each original prompt is preserved verbatim inside a deterministic structured JSON
  caption, because Ideogram 4 is trained on JSON captions. The wrapper adds no text
  elements and explicitly disallows captions, logos, signatures, and watermarks.
  Caption validation is strict; no magic-prompt API or additional API key is needed.

## 3. Run FLUX.2-dev (2025 generator)

```bash
python generation/gen_flux2.py \
  --prompts generation/prompts.csv \
  --out out/FLUX.2/fake
```

- Downloads `black-forest-labs/FLUX.2-dev` on first run (~113 GB, gated).
- On <130 GB GPUs the script auto-enables CPU offload (slower; expect ~1-3 min/image).
- If diffusers errors on `Flux2Pipeline`, upgrade diffusers (needs >=0.39).

## 4. Invariants — do not change

- **Seed 42** for every image (matches how the existing SD/FLUX.1 sets were made).
- **1024x1024** resolution.
- **Output filename = `filename` column** of prompts.csv (prompt-matched pairing
  with the real images depends on this).
- Both scripts are **resumable**: rerun after any crash; existing files are skipped.
- Do not edit prompts.csv.

## 5. Verify and hand results back

```bash
ls out/Ideogram4/fake | wc -l   # must be 100
ls out/FLUX.2/fake | wc -l      # must be 100
python - <<'EOF'
from PIL import Image; from pathlib import Path
for p in list(Path("out").rglob("*.png")) + list(Path("out").rglob("*.jpg")):
    Image.open(p).verify()
print("all images valid")
EOF
```

Then either:
- `tar czf generated_sets.tar.gz out/` and transfer back to the main machine
  (`/shared/home/sarim.hashmi/usenix/inverted_stuff/` — folders `Ideogram4/fake`
  and `FLUX.2/fake`), or
- upload to a private HF dataset repo and share the repo id.

## 6. Inversion (optional follow-up)

`inversion/` holds the RF-Inversion pipelines used for SD2.1/SD3/SD3.5
(`sd*_all_in.py`: invert + resynthesize; hyperparameters: 28 steps, guidance 3.5,
gamma 0.5, eta base 0.95 steps 0-9, seed 42, float16). `rf-inversion/` is the
upstream RF-Inversion reference implementation. Extending inversion to FLUX.2 /
Ideogram 4 requires porting the same recipe to their pipelines — not needed for
plain generation.
