# Certification of Real Images through Calibrated Content Authentication

Sarim Hashmi, Abdelrahman Elsayed, Mohammed Talha Alam, Samuele Poppi, Nils Lukas

*Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)*

This repository contains the code, evaluation benchmark, and analysis for the paper. It
covers (i) a benchmark of twenty deepfake detectors against ten generators released between
December 2022 and May 2026, (ii) their robustness under bounded adversarial perturbations, and
(iii) our calibrated content-authentication method that *certifies* an image as real with a
bounded error rate instead of classifying every input.

![Detection accuracy declines as new generators are released](benchmark/plots/frontier_time_unrestricted.png)

## Summary

Post-hoc deepfake detection is unreliable in the long run: a capable generator can reproduce
authentic content exactly (for example through memorization), so content alone does not
determine its provenance and any detector that labels every input makes errors at a rate it
cannot control. We instead ask a measurable question. Given a suspect image, can a known
generator reconstruct it faithfully under inversion? If it can, the image is *plausibly
deniable* and we abstain. If no known generator can, we certify the image as authentic
relative to the tested generators. The decision threshold is calibrated so that at most a
chosen fraction of generated content is wrongly certified, and a stricter threshold preserves
the bound against an adaptive adversary.

![Traditional detection versus calibrated resynthesis](assets/fig_concept.png)

## Key findings

- Across ten generators from 2022 to 2026, the best of twenty detectors drops from 99.5% to
  76% accuracy, and the best detector changes twice (PROBE, then SICA, then D3).
- Under an ℓ∞ attack with ε = 8/255, all twenty detectors fall below 2% accuracy; the
  strongest, D3, retains 1.75%.
- Calibrated so that at most 1% of generated content is wrongly certified, our method still
  certifies authentic content, an operating point at which most baseline detectors reach
  near-zero recall, including the strongest with 93% accuracy.
- Calibrating a stricter security threshold on attacked samples preserves the bound against
  adaptive adversaries: for SD3 Medium it moves from 0.0365 to 0.038, a best-of-100 attacker
  with PGD only raises its best candidate from 0.0148 to 0.0154, and twenty-three of
  twenty-four semantic transformations stay below the safety threshold.
- Post-hoc verifiability is eroding: of 3,000 Reddit images, 1,116 resist reproduction by a
  2022 generator but only 55 to 79 resist 2024 generators.

## Method

For every generator `G` in the tested set, we invert the query image `x` with RF-Inversion and
compare it with its reconstruction `x̃` at four levels: pixels (PSNR), structure (SSIM),
perception (1 − LPIPS), and semantics (CLIP similarity). A weighted combination is mapped to the
**Authenticity Index (A-index)** in [0, 1]:

```
s(x, x̃) = α1·PSNR + α2·SSIM + α3·(1 − LPIPS) + α4·CLIP
A(x, x̃) = exp(−σ·s) / (1 + exp(−σ·s))
```

with fitted weights α1 = −0.0181, α2 = 1.380, α3 = −4.058, α4 = 8.066 and σ = 0.9 (implemented
in [`inversion/metric.py`](inversion/metric.py)). Faithful reconstructions, typical of generated
content, receive low scores.

For each generator, the **safety threshold** is the (1 − α)-quantile of A-index scores on
held-out generated samples (α = 1%), and the **security threshold** is the same quantile after
attacking those samples with ℓ∞-bounded PGD. A query is certified as authentic only if its
A-index is at or above the threshold for every generator; otherwise the detector abstains and
returns the reconstruction as evidence.

![Computing the authenticity score](assets/fig_method.png)

The calibrated thresholds are generator-specific: 0.015 for SD2.1, 0.0365 for SD3 Medium, 0.0365
for SD3.5 Medium, 0.035 for FLUX.1 Dev, and 0.038 for FLUX.1 Dev with the Realism LoRA.
Certification takes 11.67 seconds per image and generator on one NVIDIA RTX 5000 Ada.

Reconstruction fidelity separates authentic from generated images (SD3 Medium):

![A-index separates real and generated images](assets/fig_separation.png)

## Results

**Adversarial robustness.** PGD with ε = 8/255 on 2,000 images (1,000 fake, 1,000 real,
512×512). Only samples classified correctly before the attack are perturbed. † marks
threshold-degenerate detectors; "–" means no sample of that class was correct before the attack.

| Model | Correct fake (before) | Correct real (before) | Acc. before (%) | Correct fake (after) | Correct real (after) | Acc. after (%) | Attack success on fake (%) | Attack success on real (%) |
|---|---|---|---|---|---|---|---|---|
| UFD | 1 | 974 | 48.75 | 0 | 0 | 0.00 | 100.0 | 100.0 |
| FreqNet | 53 | 995 | 52.40 | 0 | 0 | 0.00 | 100.0 | 100.0 |
| NPR | 86 | 653 | 36.95 | 0 | 0 | 0.00 | 100.0 | 100.0 |
| FatFormer | 1 | 987 | 49.40 | 0 | 0 | 0.00 | 100.0 | 100.0 |
| AEROBLADE† | 778 | 778 | 77.80 | 0 | 30 | 1.50 | 100.0 | 96.1 |
| C2PClip | 51 | 949 | 50.00 | 0 | 2 | 0.10 | 100.0 | 99.8 |
| D3 | 736 | 942 | 83.90 | 24 | 11 | **1.75** | 96.7 | 98.8 |
| FIRE† | 998 | 0 | 49.90 | 1 | 0 | 0.05 | 99.9 | – |
| DDA | 576 | 975 | 77.55 | 0 | 4 | 0.20 | 100.0 | 99.6 |
| FerretNet | 40 | 962 | 50.10 | 0 | 0 | 0.00 | 100.0 | 100.0 |
| WaRPAD† | 689 | 689 | 68.90 | 15 | 0 | 0.75 | 97.8 | 100.0 |
| APM | 34 | 987 | 51.05 | 0 | 0 | 0.00 | 100.0 | 100.0 |
| OmniAID | 917 | 948 | **93.25** | 0 | 0 | 0.00 | 100.0 | 100.0 |
| PGC | 304 | 951 | 62.75 | 0 | 0 | 0.00 | 100.0 | 100.0 |
| PROBE | 539 | 990 | 76.45 | 0 | 1 | 0.05 | 100.0 | 99.9 |
| DEAR | 417 | 993 | 70.50 | 0 | 0 | 0.00 | 100.0 | 100.0 |
| DGS-Net | 10 | 847 | 42.85 | 0 | 0 | 0.00 | 100.0 | 100.0 |
| SICA | 716 | 924 | 82.00 | 0 | 0 | 0.00 | 100.0 | 100.0 |
| IAPL | 14 | 980 | 49.70 | 0 | 0 | 0.00 | 100.0 | 100.0 |
| ForensicCpt | 55 | 982 | 51.85 | 0 | 2 | 0.10 | 100.0 | 99.8 |

PGD collapses binary detectors but leaves the A-index distributions separated:

![PGD perturbations collapse binary detectors but preserve separated A-index distributions](assets/fig_pgd.png)

**Social-media study.** Five generator-specific thresholds applied to the same SD3 inversion
scores of 3,000 unverified Reddit images: 1,116 images exceed the SD2.1 threshold, against 55 to
79 for the four newer configurations.

<p align="center"><img src="assets/fig_reddit.png" width="420" alt="Images exceeding each generator's threshold"></p>

**Video (preliminary).** On 100 videos from Deepfake-Eval-2024 (50 authentic, 50 generated),
video detectors transfer poorly, while frame-aggregated A-index scores preserve the image-level
ordering.

| Model | AUC | Prec. | Rec. | F1 |
|---|---|---|---|---|
| GenConViT | 0.615 | 0.59 | 0.49 | 0.53 |
| FTCN | 0.483 | 0.50 | 0.64 | 0.40 |
| StyleFlow | 0.509 | 0.53 | 0.42 | 0.47 |

## Repository structure

```
benchmark/     Detector benchmark and adversarial evaluation
  adapters/    One wrapper per detector (uniform inference interface)
  attacks/     White-box PGD harnesses (one per detector) + shared engine
  docs/        Adapter and PGD-attack contracts
  plots/       Figure shown at the top
  *.py         normalize_images.py, run_all.py, analyze.py, aggregate_attacks.py

inversion/     Calibrated resynthesis: RF-Inversion through SD 2, SD 3 and SD 3.5
               (unconditioned and conditioned) and the A-index metric (metric.py)

assets/        Figures from the paper
```

The detector checkpoints and generator weights are large or gated and are **not** stored in
this repository; adapters load them from a local detector zoo, `detectors/<year>/<Name>/weights/`
(see [`benchmark/docs/ADAPTER_CONTRACT.md`](benchmark/docs/ADAPTER_CONTRACT.md)).

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

GPU is required for detector inference, inversion, and the PGD attacks.

## Reproducing the benchmark

Run from the repository root. The evaluation manifest `benchmark/manifest.csv` lists one image per
row with columns `generator,gen_year,label,path`.

```bash
# 1. normalize images to 512x512 to remove the real/fake resolution confound
#    (writes benchmark/manifest_normalized.csv)
python benchmark/normalize_images.py
# 2. run every detector over the normalized manifest -> benchmark/scores/<Detector>.csv
python benchmark/run_all.py --device cuda --manifest benchmark/manifest_normalized.csv --scores-dir benchmark/scores
# 3. build the accuracy / AUROC tables and figures (benchmark/results.csv, benchmark/plots/)
python benchmark/analyze.py
```

Adversarial robustness (white-box PGD, ε = 8/255, 10 steps):

```bash
cd benchmark
# per-detector harness -> attacks_out/<Detector>.csv
python attacks/attack_<Detector>.py --manifest <manifest.csv> \
    --out attacks_out/<Detector>.csv --device cuda
# aggregate into the before/after table (results_attack.json, results_attack.md)
python aggregate_attacks.py
```

Each detector adapter follows a fixed interface (`--manifest --out --device [--limit]`) and
writes `generator,gen_year,label,path,score` with higher scores meaning more likely fake; see
[`benchmark/docs/ADAPTER_CONTRACT.md`](benchmark/docs/ADAPTER_CONTRACT.md) and
[`benchmark/docs/PGD_ATTACK_CONTRACT.md`](benchmark/docs/PGD_ATTACK_CONTRACT.md).

## Reproducing the certification method

The method resynthesizes a query image through a known generator with RF-Inversion and scores
the similarity between the image and its reconstruction into the A-index.

```bash
# resynthesize query images through a generator (example: SD3.5)
python inversion/sd3.5_all_in.py --input_dir <query_images> --output_dir <reconstructions>
# similarity metrics (PSNR, SSIM, LPIPS, CLIP) and the A-index for each image
python inversion/metric.py --source_dir <query_images> --recon_dir <reconstructions> --out metrics.jsonl
```

Calibrate the decision threshold on generated samples so that at most a chosen fraction are
certified, then evaluate certification on held-out content. The `metrics.jsonl` output holds the
per-image scores used for calibration.

## Detectors evaluated

UFD, FreqNet, NPR, FatFormer, AEROBLADE, C2P-CLIP, D3, FIRE, DDA, FerretNet, WaRPAD,
AllPatchesMatter, OmniAID, PGC, PROBE, DEAR, DGS-Net, SICA, IAPL, ForensicConcept. Each has an
adapter in [`benchmark/adapters/`](benchmark/adapters) and a PGD harness in
[`benchmark/attacks/`](benchmark/attacks).

## Generators evaluated

SD 2.1, SD 3, SD 3.5, FLUX.1 (with LoRA and Boreal adapters), FLUX.2, GPT Image-2, and
HiDream-O1, spanning December 2022 to May 2026. Certification is calibrated for five
configurations: SD2.1, SD3 Medium, SD3.5 Medium, FLUX.1 Dev, and FLUX.1 Dev with the Realism LoRA.

## Citation

```bibtex
@article{hashmi2026certification,
  title   = {Certification of Real Images through Calibrated Content Authentication},
  author  = {Hashmi, Sarim and Elsayed, Abdelrahman and Alam, Mohammed Talha and Poppi, Samuele and Lukas, Nils},
  year    = {2026}
}
```

## Contact

Questions and issues are welcome via the repository issue tracker or by email to the authors
at MBZUAI.
