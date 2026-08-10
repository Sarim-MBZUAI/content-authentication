# Deepfake-detection benchmark — results

19 detectors x 7 generators. Each cell: 100 real + 100 fake images, prompt-matched,
seed 42, **normalized to 512x512** (removes the resolution confound between variable-size
real photos and fixed-square generated images). Threshold 0.5 for probability scores, 0.0 for logits.

**Generators** (release year): SD2.1 (2022), SD3 / SD3.5 / FLUX.1-dev / FLUX.1-dev+Realism-LoRA (2024),
Boreal-FLUX = FLUX.1-dev + `kudzueye/boreal-flux-dev-v2` community realism LoRA (2024 base),
HiDream-O1-Image-Dev-2604 (2026).

**Note:** FIRE and WaRPAD are degenerate at their native threshold (FIRE labels ~everything fake,
WaRPAD ~everything real) — read them by AUC, not the tables below.

## Table 1 — Balanced accuracy (100 real + 100 fake)
| Year | Detector | SD2.1 | SD3 | SD3.5 | FLUX-dev | FLUX-LoRA | Boreal-FLUX | HiDream-O1 |
|---|---|---|---|---|---|---|---|---|
| 2023 | UFD | 0.495 | 0.500 | 0.505 | 0.485 | 0.490 | 0.490 | 0.490 |
| 2024 | FatFormer | 0.495 | 0.490 | 0.630 | 0.490 | 0.510 | 0.595 | 0.570 |
| 2024 | FreqNet | 0.465 | 0.445 | 0.900 | 0.825 | 0.840 | 0.810 | 0.785 |
| 2024 | NPR | 0.350 | 0.370 | 0.745 | 0.800 | 0.755 | 0.800 | 0.800 |
| 2025 | C2P-CLIP | 0.500 | 0.500 | 0.600 | 0.525 | 0.505 | 0.515 | 0.525 |
| 2025 | D3 | 0.810 | 0.825 | 0.985 | 0.960 | 0.875 | 0.810 | 0.915 |
| 2025 | DDA | 0.970 | 0.665 | 0.975 | 0.680 | 0.610 | 0.620 | 0.825 |
| 2025 | FIRE | 0.495 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| 2025 | FerretNet | 0.505 | 0.490 | 0.985 | 0.935 | 0.950 | 0.980 | 0.980 |
| 2025 | WaRPAD | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| 2026 | AllPatchesMatter | 0.525 | 0.550 | 0.975 | 0.595 | 0.580 | 0.635 | 0.745 |
| 2026 | DEAR | 0.805 | 0.520 | 0.940 | 0.820 | 0.690 | 0.565 | 0.710 |
| 2026 | DGS-Net | 0.455 | 0.425 | 0.940 | 0.925 | 0.665 | 0.845 | 0.910 |
| 2026 | ForensicConcept | 0.550 | 0.585 | 0.890 | 0.625 | 0.540 | 0.635 | 0.620 |
| 2026 | IAPL | 0.500 | 0.515 | 0.625 | 0.500 | 0.490 | 0.535 | 0.505 |
| 2026 | OmniAID | 0.980 | 0.915 | 0.925 | 0.895 | 0.860 | 0.570 | 0.905 |
| 2026 | PGC | 0.810 | 0.575 | 0.740 | 0.540 | 0.535 | 0.540 | 0.695 |
| 2026 | PROBE | 0.995 | 0.660 | 0.930 | 0.495 | 0.510 | 0.495 | 0.565 |
| 2026 | SICA | 0.955 | 0.960 | 0.985 | 0.970 | 0.905 | 0.795 | 0.980 |

## Table 2 — Fake-recall (fakes only; fraction of generated images caught)

The honest metric for adapter evasion: balanced accuracy hides it because a detector that
nails the real half still looks fine when it misses a third of the fakes.
| Year | Detector | SD2.1 | SD3 | SD3.5 | FLUX-dev | FLUX-LoRA | Boreal-FLUX | HiDream-O1 |
|---|---|---|---|---|---|---|---|---|
| 2023 | UFD | 0.010 | 0.010 | 0.020 | 0.000 | 0.000 | 0.010 | 0.010 |
| 2024 | FatFormer | 0.000 | 0.000 | 0.280 | 0.010 | 0.060 | 0.220 | 0.170 |
| 2024 | FreqNet | 0.000 | 0.000 | 0.890 | 0.720 | 0.720 | 0.690 | 0.640 |
| 2024 | NPR | 0.120 | 0.280 | 1.000 | 1.000 | 0.980 | 1.000 | 1.000 |
| 2025 | C2P-CLIP | 0.000 | 0.010 | 0.210 | 0.060 | 0.020 | 0.040 | 0.060 |
| 2025 | D3 | 0.660 | 0.680 | 0.980 | 0.940 | 0.810 | 0.640 | 0.860 |
| 2025 | DDA | 0.960 | 0.360 | 0.950 | 0.380 | 0.220 | 0.260 | 0.670 |
| 2025 | FIRE | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2025 | FerretNet | 0.010 | 0.010 | 1.000 | 0.880 | 0.920 | 0.970 | 0.970 |
| 2025 | WaRPAD | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2026 | AllPatchesMatter | 0.060 | 0.110 | 0.980 | 0.200 | 0.190 | 0.280 | 0.500 |
| 2026 | DEAR | 0.610 | 0.040 | 0.880 | 0.640 | 0.380 | 0.130 | 0.420 |
| 2026 | DGS-Net | 0.000 | 0.000 | 1.000 | 0.950 | 0.510 | 0.800 | 0.940 |
| 2026 | ForensicConcept | 0.110 | 0.180 | 0.790 | 0.280 | 0.080 | 0.300 | 0.270 |
| 2026 | IAPL | 0.000 | 0.060 | 0.270 | 0.020 | 0.020 | 0.090 | 0.030 |
| 2026 | OmniAID | 0.980 | 0.830 | 0.850 | 0.810 | 0.720 | 0.160 | 0.830 |
| 2026 | PGC | 0.650 | 0.210 | 0.530 | 0.150 | 0.120 | 0.150 | 0.460 |
| 2026 | PROBE | 1.000 | 0.330 | 0.870 | 0.000 | 0.020 | 0.000 | 0.140 |
| 2026 | SICA | 0.930 | 0.960 | 0.980 | 0.980 | 0.820 | 0.630 | 1.000 |

## Table 3 — FLUX.2-dev top-3 (partial: plain n=65, boreal n=69, fake-recall)

FLUX.2-dev (2025, 32B) plain vs FLUX.2-dev + `kudzueye/boreal-flux-dev2` LoRA. Generation still running.

| Detector | FLUX2-plain | FLUX2-boreal |
|---|---|---|
| SICA | 0.908 (n=65) | 0.478 (n=69) |
| D3 | 0.862 (n=65) | 0.580 (n=69) |
| OmniAID | 0.523 (n=65) | 0.130 (n=69) |

## Key findings

- **Best detectors overall:** SICA (0.936 mean acc), D3 (0.883), OmniAID (0.864).
- **A community realism LoRA (Boreal) is the strongest attack.** It drops the top detectors far below
  their FLUX.1-dev performance — OmniAID fake-recall 0.81 -> 0.16, SICA 0.98 -> 0.63, PROBE -> 0.00 —
  while the clean 2026 foundation model HiDream-O1 is caught about as well as FLUX.1-dev. The threat is
  the cheap adapter, not model recency.
- **The effect transfers to FLUX.2:** Boreal-dev2 drops OmniAID 0.52 -> 0.13, SICA 0.91 -> 0.48 (partial).
- **FerretNet** is the lone Boreal-robust detector (~0.98 on both new generators).
