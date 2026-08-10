# SKILL: The arms-race graph — exact definition (do not reinterpret)

This file pins down THE experiment. If you are an agent asked to "make the arms
race graph", follow this literally. Deviations that have already been tried and
REJECTED by the author:
- ❌ era-cohort matching (year-Y detectors vs only year-Y generators)
- ❌ multi-line FPR operating-point curves as the headline
- ❌ per-detector scatter as the headline
The accepted design is the cumulative **best-vs-best (minimax)** curve below,
with **accuracy** on the y-axis.

## Step 1 — Full inference matrix (prerequisite)

EVERY detector scores EVERY image in the manifest: all generated images AND
their corresponding real images, for ALL generators (100 real + 100 fake per
generator, fixed seed-42 sample, `benchmark/manifest.csv`). No detector or
generator is skipped because "years don't match" — the matrix is complete.

Scores follow `ADAPTER_CONTRACT.md`: one CSV per detector with columns
`generator,gen_year,label,path,score`, score higher = more likely fake.

## Step 2 — The curve

For each year Y on the x-axis (2023, 2024, 2025, 2026):

```
D(Y) = all detectors released in year <= Y     # defender's arsenal
G(Y) = all generators released in year <= Y    # attacker's arsenal

value(Y) = min over g in G(Y) of [ max over d in D(Y) of ACC(d, g) ]
```

In words: the defender picks the **best available detector** (any year up to Y,
not just Y), the attacker picks the **hardest available generator** (any year up
to Y). ACC(d, g) = accuracy of detector d on generator g's 200 images (100 real
+ 100 fake), threshold 0.5 for probability scores / 0.0 for logit scores.

## Step 3 — The plot (`plots/arms_race.png`)

- x-axis: year (2023-2026). y-axis: **accuracy** (0.28-1.05 range, chance line
  dotted at 0.5, labeled "chance").
- ONE blue line (#2a78d6), round markers with white edge, value labels above
  each point.
- Under each x-tick, annotate the winning matchup in muted gray:
  `<best detector>\nvs <hardest generator>`.
- Title: "Arms race — best available detector vs. hardest available generator".
- Recessive gray grid, no top/right spines, white background, PNG + PDF.

Supporting figures (keep, not headline): full detector x generator accuracy/AUC
heatmap (`plots/heatmap.png`), minimax-AUC companion (`plots/arms_race_auc.png`).
`results.csv` keeps acc, AUC and recall@{1,5,10,20}%FPR per cell for the paper.

## Detector release years (fixed)

2023: UFD · 2024: FreqNet, NPR, FatFormer · 2025: C2P-CLIP, D3, FIRE, DDA,
FerretNet, WaRPAD · 2026: AllPatchesMatter, OmniAID, PGC, PROBE, DEAR, DGS-Net,
SICA, IAPL, ForensicConcept

## Generator release years (fixed)

2022: SD2.1 · 2024: SD3, SD3.5, FLUX-dev, FLUX-LoRA · 2025: FLUX.2-dev ·
2026: Ideogram 4

## Run it

```bash
cd benchmark
venv/bin/python extend_manifest.py     # adds FLUX.2 / Ideogram4 rows when their images exist
venv/bin/python run_all.py --device cuda   # full matrix -> scores/<detector>.csv
venv/bin/python analyze.py                 # results.csv/md + plots/
# on the MBZ cluster all GPU work MUST go through SLURM:
sbatch run_benchmark.sbatch
```

## Invariants

- Manifest sampling is seed-42 and never resampled; extend, don't regenerate.
- A detector's score CSV must cover the full manifest (missing rows allowed only
  on per-image errors, written as empty score).
- New generators/detectors: add to the year maps in `analyze.py` AND here.
- Accuracy is the headline metric (author's decision, 2026-08-10).
