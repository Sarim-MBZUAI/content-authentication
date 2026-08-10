# Deepfake-detection benchmark — results

19 detectors x 7 generators. Each cell: 100 real + 100 fake, prompt-matched, seed 42,
**normalized to 512x512** (removes the real/fake resolution confound). Threshold 0.5 (prob) / 0.0 (logit).
Column headers show each generator's release year; the first column shows each detector's release year.

**Generators:** SD2.1 (2022); SD3, SD3.5, FLUX.1-dev, FLUX.1-dev+Realism-LoRA (2024);
Boreal-FLUX = FLUX.1-dev + `kudzueye/boreal-flux-dev-v2` realism LoRA (2024 base); HiDream-O1-Image-Dev-2604 (2026).

**Caveat:** FIRE / WaRPAD are degenerate at their threshold (all-fake / all-real) — judge by AUC, not these tables.

## Table 1 — Balanced accuracy (100 real + 100 fake)

| Det. year | Detector | SD2.1 (2022) | SD3 (2024) | SD3.5 (2024) | FLUX-dev (2024) | FLUX-LoRA (2024) | Boreal-FLUX (2024) | HiDream-O1 (2026) |
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

Balanced accuracy hides adapter evasion (a detector that nails reals looks fine even missing a third of fakes); fake-recall exposes it.

| Det. year | Detector | SD2.1 (2022) | SD3 (2024) | SD3.5 (2024) | FLUX-dev (2024) | FLUX-LoRA (2024) | Boreal-FLUX (2024) | HiDream-O1 (2026) |
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

## Table 3 — FLUX.2-dev top-3 (fake-recall; FLUX.2 = 2025 · plain n=79, boreal n=84, generation ongoing)

FLUX.2-dev vs FLUX.2-dev + `kudzueye/boreal-flux-dev2` LoRA.

| Detector (year) | FLUX2-plain (2025) | FLUX2-boreal (2025) |
|---|---|---|
| SICA (2026) | 0.91 | 0.43 |
| D3 (2026) | 0.90 | 0.52 |
| OmniAID (2026) | 0.53 | 0.13 |

## Key findings

- **Best detectors:** SICA (0.936 mean acc), D3 (0.883), OmniAID (0.864).
- **A community realism LoRA (Boreal) is the strongest attack**, not model recency: it drops top detectors far below their
  FLUX.1-dev level (OmniAID fake-recall 0.81->0.16, SICA 0.98->0.63, PROBE->0.00), while the clean 2026 foundation model
  HiDream-O1 is caught about as well as FLUX.1-dev.
- **Transfers to FLUX.2:** Boreal-dev2 drops OmniAID 0.53->0.13, SICA 0.91->0.43, D3 0.90->0.52.
- **FerretNet** is the lone Boreal-robust detector (~0.98).
