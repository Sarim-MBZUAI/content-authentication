# Benchmark — deepfake detectors vs. generators

20 detectors × 10 generators. Each cell: 100 real + 100 fake, prompt-matched, seed 42,
images normalized to 512×512 (removes the real/fake resolution confound).

## Layout

```
adapters/            one <Detector>.py per detector — the inference wrappers (the "baseline code")
scores/              raw per-image scores, original 5 generators   (<Detector>.csv)
scores_new_gens/     raw scores, Boreal-FLUX + HiDream-O1
scores_flux2/        raw scores, FLUX2-plain + FLUX2-boreal
scores_gpt2_full/    raw scores, GPT-image-2 (photoreal)
scores_jsonl/        per-image audit: one JSON object per image, with score_type
results.json         consolidated: per detector×generator accuracy / fake_recall / auroc
auroc_matrix.csv     long-form AUROC table
manifest*.csv        image lists (generator,gen_year,label,path)
RESULTS.md           the three summary tables (accuracy, fake-recall, AUROC) + findings
plots/               all figures (.png + .pdf)
analyze.py           builds RESULTS tables + plots from scores/
run_all.py           runs every adapter over a manifest -> scores/<Detector>.csv
to_jsonl.py          scores/*.csv -> scores_jsonl/*.jsonl
normalize_images.py  512px normalization
*.sbatch             SLURM launchers (cluster: GPU work must go via SLURM)
ADAPTER_CONTRACT.md  the interface every adapter obeys
SKILL_ARMS_RACE.md   pinned definition of the arms-race experiment
```

## Adapters (inference code)

Each `adapters/<Detector>.py` is a self-contained wrapper following `ADAPTER_CONTRACT.md`:
CLI `--manifest --out --device [--limit]`, output columns `generator,gen_year,label,path,score`
(score higher = more likely FAKE). Adapters load each detector's **official checkpoint and
code from `../detectors/<year>/<Name>/`** (weights are NOT in this repo — HF-gated / large;
see the detector zoo). Run all:

```bash
python run_all.py --device cuda --manifest manifest_normalized.csv --scores-dir scores
python analyze.py
```

## Detectors (20)

UFD (2023) · FreqNet, NPR, FatFormer, AEROBLADE (2024) · C2P-CLIP, D3, FIRE, DDA, FerretNet,
WaRPAD (2025) · AllPatchesMatter, OmniAID, PGC, PROBE, DEAR, DGS-Net, SICA, IAPL, ForensicConcept (2026).

Cited-but-not-evaluated (no public weights): **DIRE** (ICCV'23), **LaRE²** (CVPR'24).
Threshold-degenerate (read via AUROC): FIRE, WaRPAD, AEROBLADE.
