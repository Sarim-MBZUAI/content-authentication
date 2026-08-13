# Benchmark — deepfake detectors vs. generators

20 detectors × 10 generators. Each cell: 100 real + 100 fake, prompt-matched, seed 42,
images normalized to 512×512 (removes the real/fake resolution confound).

## Layout

```
adapters/            one <Detector>.py per detector — the inference wrappers (the "baseline code")
attacks/             one attack_<Detector>.py per detector — white-box PGD harnesses + shared engines
attacks_out/         raw per-image clean-vs-adversarial scores (<Detector>.csv)

scores/              raw per-image scores, original 5 generators   (<Detector>.csv)
scores_new_gens/     raw scores, Boreal-FLUX + HiDream-O1
scores_flux2/        raw scores, FLUX2-plain + FLUX2-boreal
scores_gpt2_full/    raw scores, GPT-image-2 (photoreal)
scores_newbench/     raw scores on New_benchmark
scores_jsonl/        per-image audit: one JSON object per image, with score_type
plots/               all figures (.png + .pdf)

manifests/           image lists (generator,gen_year,label,path) — manifest*.csv
results/             consolidated outputs: results*.{json,md}, RESULTS.md, acc/auroc matrices
tables/              LaTeX tables for the paper (tab_attack_results.tex, …)
slurm/               SLURM launchers — *.sbatch (cluster: GPU work must go via SLURM)
docs/                contracts + skills: ADAPTER_CONTRACT, PGD_ATTACK_CONTRACT, SKILL_ARMS_RACE

analyze.py           builds RESULTS tables + plots from scores/
run_all.py           runs every adapter over a manifest -> scores/<Detector>.csv
aggregate_attacks.py attacks_out/*.csv -> results/results_attack.{json,md} + tables/*.tex
to_jsonl.py          scores/*.csv -> scores_jsonl/*.jsonl
normalize_images.py  512px normalization
extend_manifest.py   append new-generator rows to a manifest
```

Note: the analysis/SLURM scripts here are a curated snapshot; they resolve paths against the
absolute live working tree (`B = .../usenix/benchmark`), not against their location in this repo.

## Adapters (inference code)

Each `adapters/<Detector>.py` is a self-contained wrapper following `docs/ADAPTER_CONTRACT.md`:
CLI `--manifest --out --device [--limit]`, output columns `generator,gen_year,label,path,score`
(score higher = more likely FAKE). Adapters load each detector's **official checkpoint and
code from `../detectors/<year>/<Name>/`** (weights are NOT in this repo — HF-gated / large;
see the detector zoo). Run all:

```bash
python run_all.py --device cuda --manifest manifests/manifest_normalized.csv --scores-dir scores
python analyze.py
```

## Detectors (20)

UFD (2023) · FreqNet, NPR, FatFormer, AEROBLADE (2024) · C2P-CLIP, D3, FIRE, DDA, FerretNet,
WaRPAD (2025) · AllPatchesMatter, OmniAID, PGC, PROBE, DEAR, DGS-Net, SICA, IAPL, ForensicConcept (2026).

Cited-but-not-evaluated (no public weights): **DIRE** (ICCV'23), **LaRE²** (CVPR'24).
Threshold-degenerate (read via AUROC): FIRE, WaRPAD, AEROBLADE.
