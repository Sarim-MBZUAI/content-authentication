# Deepfake-detection benchmark — results

19 detectors x 10 generators. Each cell: 100 real + 100 fake, prompt-matched, seed 42,
**normalized to 512x512** (removes the real/fake resolution confound). Threshold 0.5 (prob) / 0.0 (logit).
Column headers show each generator's release year; the first column shows each detector's release year.

**Generators:** SD2.1 (2022); SD3, SD3.5, FLUX.1-dev, FLUX.1-dev+Realism-LoRA (2024);
Boreal-FLUX = FLUX.1-dev + `kudzueye/boreal-flux-dev-v2` LoRA (2024 base);
FLUX2-plain = FLUX.2-dev, FLUX2-boreal = FLUX.2-dev + `kudzueye/boreal-flux-dev2` LoRA (2025);
HiDream-O1-Image-Dev-2604 (2026); GPTimage2-photoreal = OpenAI GPT-image-2 with a photorealism prompt prefix (2026).

**Caveat:** FIRE / WaRPAD are degenerate at their threshold (all-fake / all-real) — judge by AUC, not these tables.

## Table 1 — Balanced accuracy (100 real + 100 fake)

| Det. year | Detector | SD2.1 (2022) | SD3 (2024) | SD3.5 (2024) | FLUX-dev (2024) | FLUX-LoRA (2024) | Boreal-FLUX (2024) | FLUX2-plain (2025) | FLUX2-boreal (2025) | HiDream-O1 (2026) | GPTimage2-photoreal (2026) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2023 | UFD | 0.495 | 0.500 | 0.505 | 0.485 | 0.490 | 0.490 | 0.490 | 0.485 | 0.490 | 0.500 |
| 2024 | FatFormer | 0.495 | 0.490 | 0.630 | 0.490 | 0.510 | 0.595 | 0.525 | 0.520 | 0.570 | 0.605 |
| 2024 | FreqNet | 0.465 | 0.445 | 0.900 | 0.825 | 0.840 | 0.810 | 0.830 | 0.745 | 0.785 | 0.810 |
| 2024 | NPR | 0.350 | 0.370 | 0.745 | 0.800 | 0.755 | 0.800 | 0.800 | 0.800 | 0.800 | 0.795 |
| 2025 | C2P-CLIP | 0.500 | 0.500 | 0.600 | 0.525 | 0.505 | 0.515 | 0.500 | 0.505 | 0.525 | 0.510 |
| 2025 | D3 | 0.810 | 0.825 | 0.985 | 0.960 | 0.875 | 0.810 | 0.930 | 0.760 | 0.915 | 0.840 |
| 2025 | DDA | 0.970 | 0.665 | 0.975 | 0.680 | 0.610 | 0.620 | 0.850 | 0.695 | 0.825 | 0.565 |
| 2025 | FIRE | 0.495 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| 2025 | FerretNet | 0.505 | 0.490 | 0.985 | 0.935 | 0.950 | 0.980 | 0.870 | 0.985 | 0.980 | 0.995 |
| 2025 | WaRPAD | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| 2026 | AllPatchesMatter | 0.525 | 0.550 | 0.975 | 0.595 | 0.580 | 0.635 | 0.855 | 0.645 | 0.745 | 0.920 |
| 2026 | DEAR | 0.805 | 0.520 | 0.940 | 0.820 | 0.690 | 0.565 | 0.705 | 0.570 | 0.710 | 0.510 |
| 2026 | DGS-Net | 0.455 | 0.425 | 0.940 | 0.925 | 0.665 | 0.845 | 0.950 | 0.890 | 0.910 | 0.930 |
| 2026 | ForensicConcept | 0.550 | 0.585 | 0.890 | 0.625 | 0.540 | 0.635 | 0.680 | 0.580 | 0.620 | 0.740 |
| 2026 | IAPL | 0.500 | 0.515 | 0.625 | 0.500 | 0.490 | 0.535 | 0.500 | 0.500 | 0.505 | 0.610 |
| 2026 | OmniAID | 0.980 | 0.915 | 0.925 | 0.895 | 0.860 | 0.570 | 0.720 | 0.575 | 0.905 | 0.505 |
| 2026 | PGC | 0.810 | 0.575 | 0.740 | 0.540 | 0.535 | 0.540 | 0.560 | 0.495 | 0.695 | 0.590 |
| 2026 | PROBE | 0.995 | 0.660 | 0.930 | 0.495 | 0.510 | 0.495 | 0.555 | 0.510 | 0.565 | 0.540 |
| 2026 | SICA | 0.955 | 0.960 | 0.985 | 0.970 | 0.905 | 0.795 | 0.950 | 0.705 | 0.980 | 0.700 |

## Table 2 — Fake-recall (fakes only; fraction of generated images caught)

Balanced accuracy hides adapter/prompt evasion (a detector that nails reals looks fine even missing a third of fakes); fake-recall exposes it.

| Det. year | Detector | SD2.1 (2022) | SD3 (2024) | SD3.5 (2024) | FLUX-dev (2024) | FLUX-LoRA (2024) | Boreal-FLUX (2024) | FLUX2-plain (2025) | FLUX2-boreal (2025) | HiDream-O1 (2026) | GPTimage2-photoreal (2026) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2023 | UFD | 0.010 | 0.010 | 0.020 | 0.000 | 0.000 | 0.010 | 0.010 | 0.000 | 0.010 | 0.030 |
| 2024 | FatFormer | 0.000 | 0.000 | 0.280 | 0.010 | 0.060 | 0.220 | 0.080 | 0.070 | 0.170 | 0.240 |
| 2024 | FreqNet | 0.000 | 0.000 | 0.890 | 0.720 | 0.720 | 0.690 | 0.730 | 0.560 | 0.640 | 0.690 |
| 2024 | NPR | 0.120 | 0.280 | 1.000 | 1.000 | 0.980 | 1.000 | 1.000 | 1.000 | 1.000 | 0.990 |
| 2025 | C2P-CLIP | 0.000 | 0.010 | 0.210 | 0.060 | 0.020 | 0.040 | 0.010 | 0.020 | 0.060 | 0.030 |
| 2025 | D3 | 0.660 | 0.680 | 0.980 | 0.940 | 0.810 | 0.640 | 0.890 | 0.540 | 0.860 | 0.710 |
| 2025 | DDA | 0.960 | 0.360 | 0.950 | 0.380 | 0.220 | 0.260 | 0.720 | 0.410 | 0.670 | 0.150 |
| 2025 | FIRE | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2025 | FerretNet | 0.010 | 0.010 | 1.000 | 0.880 | 0.920 | 0.970 | 0.750 | 0.980 | 0.970 | 1.000 |
| 2025 | WaRPAD | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2026 | AllPatchesMatter | 0.060 | 0.110 | 0.980 | 0.200 | 0.190 | 0.280 | 0.720 | 0.300 | 0.500 | 0.850 |
| 2026 | DEAR | 0.610 | 0.040 | 0.880 | 0.640 | 0.380 | 0.130 | 0.410 | 0.140 | 0.420 | 0.020 |
| 2026 | DGS-Net | 0.000 | 0.000 | 1.000 | 0.950 | 0.510 | 0.800 | 0.980 | 0.860 | 0.940 | 0.940 |
| 2026 | ForensicConcept | 0.110 | 0.180 | 0.790 | 0.280 | 0.080 | 0.300 | 0.390 | 0.190 | 0.270 | 0.510 |
| 2026 | IAPL | 0.000 | 0.060 | 0.270 | 0.020 | 0.020 | 0.090 | 0.020 | 0.020 | 0.030 | 0.240 |
| 2026 | OmniAID | 0.980 | 0.830 | 0.850 | 0.810 | 0.720 | 0.160 | 0.460 | 0.170 | 0.830 | 0.030 |
| 2026 | PGC | 0.650 | 0.210 | 0.530 | 0.150 | 0.120 | 0.150 | 0.190 | 0.060 | 0.460 | 0.250 |
| 2026 | PROBE | 1.000 | 0.330 | 0.870 | 0.000 | 0.020 | 0.000 | 0.120 | 0.030 | 0.140 | 0.090 |
| 2026 | SICA | 0.930 | 0.960 | 0.980 | 0.980 | 0.820 | 0.630 | 0.940 | 0.450 | 1.000 | 0.440 |

## Key findings

- **Best detectors:** SICA, D3, OmniAID.
- **Community realism LoRAs (Boreal) are a strong attack**, not model recency: on both FLUX bases the LoRA sharply cuts
  top-detector fake-recall vs the plain base, while the clean 2026 foundation model HiDream-O1 stays easy to catch.
- **GPT-image-2 (closed frontier model) + a photorealism prompt evades the top detectors with no adapter at all** —
  OmniAID fake-recall 0.03, SICA 0.44; only D3 (0.71) holds up. Consistent with its architecture being off the SD/FLUX lineage detectors were trained on.
- **OmniAID FLUX-realism progression (fake-recall):** FLUX1-dev 0.81 -> FLUX1-Boreal 0.16 -> FLUX2-plain 0.46 -> FLUX2-Boreal 0.17; GPT-image-2 0.03.
- **FerretNet** is the lone Boreal-robust detector.
