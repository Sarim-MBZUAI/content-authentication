# Human-subset accuracy (detector × generator)

Subset = benchmark images whose caption mentions a person (identified from the image filename). Same detectors, same scores, same decision thresholds as the full benchmark (calibration vs `acc_matrix.csv` on the full set: exact, max diff 0.0000). `†` = threshold-degenerate detector.

Human images per generator (real / fake): FLUX-LoRA 29/24, FLUX-dev 25/26, SD2.1 43/36, SD3 29/26, SD3.5 29/33, Boreal-FLUX 25/25, HiDream-O1 25/25, FLUX2-boreal 25/25, FLUX2-plain 25/25, GPTimage2-photoreal 25/25

| Detector | FLUX-LoRA | FLUX-dev | SD2.1 | SD3 | SD3.5 | Boreal-FLUX | HiDream-O1 | FLUX2-boreal | FLUX2-plain | GPTimage2-photoreal |
|---|---|---|---|---|---|---|---|---|---|---|
| AEROBLADE† | 0.55 | 0.49 | 0.54 | 0.53 | 0.47 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| AllPatchesMatter | 0.64 | 0.59 | 0.56 | 0.60 | 1.00 | 0.64 | 0.82 | 0.70 | 0.92 | 0.98 |
| C2P-CLIP | 0.53 | 0.55 | 0.54 | 0.53 | 0.55 | 0.52 | 0.50 | 0.50 | 0.48 | 0.48 |
| D3 | 0.91 | 0.96 | 0.70 | 0.80 | 0.98 | 0.72 | 0.90 | 0.72 | 0.96 | 0.86 |
| DDA | 0.62 | 0.76 | 0.95 | 0.65 | 1.00 | 0.60 | 0.84 | 0.74 | 0.82 | 0.54 |
| DEAR | 0.68 | 0.84 | 0.84 | 0.55 | 0.97 | 0.58 | 0.70 | 0.60 | 0.66 | 0.50 |
| DGS-Net | 0.62 | 0.98 | 0.48 | 0.45 | 0.97 | 0.88 | 0.98 | 0.94 | 0.94 | 0.96 |
| FIRE† | 0.45 | 0.51 | 0.44 | 0.47 | 0.53 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| FatFormer | 0.53 | 0.49 | 0.53 | 0.53 | 0.58 | 0.60 | 0.58 | 0.50 | 0.52 | 0.60 |
| FerretNet | 1.00 | 0.96 | 0.54 | 0.53 | 1.00 | 0.98 | 0.98 | 0.98 | 0.88 | 1.00 |
| ForensicConcept | 0.57 | 0.65 | 0.56 | 0.58 | 0.90 | 0.60 | 0.52 | 0.56 | 0.66 | 0.72 |
| FreqNet | 0.87 | 0.82 | 0.49 | 0.51 | 0.95 | 0.92 | 0.82 | 0.74 | 0.88 | 0.82 |
| IAPL | 0.53 | 0.49 | 0.54 | 0.53 | 0.60 | 0.50 | 0.48 | 0.48 | 0.48 | 0.56 |
| NPR | 0.74 | 0.88 | 0.37 | 0.33 | 0.84 | 0.88 | 0.88 | 0.88 | 0.88 | 0.86 |
| OmniAID | 0.91 | 0.92 | 0.97 | 0.96 | 0.97 | 0.58 | 0.88 | 0.58 | 0.70 | 0.54 |
| PGC | 0.55 | 0.55 | 0.82 | 0.58 | 0.66 | 0.48 | 0.62 | 0.46 | 0.46 | 0.54 |
| PROBE | 0.57 | 0.49 | 1.00 | 0.71 | 0.92 | 0.50 | 0.52 | 0.50 | 0.54 | 0.50 |
| SICA | 1.00 | 0.98 | 0.92 | 0.96 | 1.00 | 0.78 | 0.98 | 0.78 | 0.92 | 0.74 |
| UFD | 0.55 | 0.49 | 0.54 | 0.53 | 0.47 | 0.50 | 0.50 | 0.50 | 0.50 | 0.52 |
| WaRPAD† | 0.55 | 0.49 | 0.54 | 0.53 | 0.47 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
