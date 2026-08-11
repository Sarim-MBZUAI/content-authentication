# SKILL: Generate a photorealistic-prefixed 100-image set (for Codex)

Goal: generate 100 images from a text-to-image model using the **photorealism-prefixed**
prompts, so we can test whether a "make it look like a real photo" instruction helps the
generator evade deepfake detectors — same benchmark protocol as every other generator.

## Inputs (already in this repo)

- `generation/prompts_photoreal.csv` — the 100 prompts, each prefixed with:
  `A photorealistic amateur photo, natural lighting, taken on a phone camera. `
  Columns: `filename,prompt`. **Use the `prompt` column verbatim** (prefix already applied).
- `generation/prompts.csv` — the same 100 without the prefix (for reference / a plain baseline).

## HARD INVARIANTS — do not change (the benchmark breaks otherwise)

1. **Output filename = the `filename` column of the CSV**, unchanged. This is how each fake
   is paired with its real image and with every other generator. Never rename, never re-order.
2. **seed = 42** for every image.
3. **1024 x 1024** resolution.
4. Generate all 100 rows. **Resumable**: if an output file already exists, skip it.
5. Do NOT edit either CSV.

## Output folder layout (exact)

Pick a short generator tag `<GEN>` (e.g. `FLUX2-photoreal`, `HiDream-photoreal`). Write:

```
out/<GEN>/fake/<filename>.png     # the 100 generated images
out/<GEN>/real/<filename>.png     # the 100 matched REAL images (copy from the shared real set)
```

- The `fake/` filenames MUST match the CSV `filename` values exactly.
- For `real/`, copy the same 100 real images every generator uses. If you don't have them
  locally, they are the `FLUX-dev` real set (COCO-caption-named PNGs); ask for the
  `real/` folder or pull it from the benchmark reals. `real/` and `fake/` must contain the
  **same 100 filenames**.

## Generation settings

Reuse the model's existing script in `generation/` and just point `--prompts` at the
photoreal CSV. Examples:

```bash
# FLUX.2-dev + photoreal prefix
python generation/gen_flux2.py \
    --prompts generation/prompts_photoreal.csv \
    --out out/FLUX2-photoreal/fake \
    --seed 42 --size 1024

# FLUX.2-dev + Boreal LoRA + photoreal prefix (stack both)
python generation/gen_flux2_boreal_100.py   # edit its PROMPTS path to prompts_photoreal.csv, OUT to FLUX2-boreal-photoreal
```

- diffusers pipelines: pass the prompt as a KEYWORD arg (`prompt=...`), 50 steps, guidance 4.0.
- On <130 GB GPUs enable `pipe.enable_model_cpu_offload()` (the scripts already auto-detect).
- `HF_TOKEN` must be set and the model's license gate accepted on the account.

## Verify before handing back

```bash
ls out/<GEN>/fake | wc -l    # must be 100
ls out/<GEN>/real | wc -l    # must be 100
# names must match:
diff <(ls out/<GEN>/fake | sort) <(ls out/<GEN>/real | sort)   # no output = good
python - <<'PY'
from PIL import Image; from pathlib import Path
for p in Path("out/<GEN>").rglob("*.png"): Image.open(p).verify()
print("all valid")
PY
```

## Hand back

Upload to an HF dataset so the scoring machine can pull it (this is how FLUX.2 was handed back):

```bash
huggingface-cli upload-large-folder Sarim-Hash/liars-dividend-<GEN> \
    out/<GEN> --repo-type=dataset --include "**/*.png"
```

Then tell the scoring side the dataset id `Sarim-Hash/liars-dividend-<GEN>` and the `<GEN>` tag.
The scoring side normalizes to 512x512 and runs all 19 detectors — no further action needed from you.
