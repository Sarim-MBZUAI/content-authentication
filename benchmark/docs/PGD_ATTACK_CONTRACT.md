# PGD attack harness contract

For each detector, write `attacks/attack_<Detector>.py` that runs a white-box PGD
ℓ∞ attack and reports clean vs adversarial predictions.

## Attack spec (FIXED — identical for every detector)
- ℓ∞ PGD, **ε = 8/255**, **10 steps**, step size **α = 2/255**, random start within the ball.
- **Untargeted / flip:** for a FAKE image push the score DOWN (toward "real"); for a
  REAL image push the score UP (toward "fake"). i.e. ascend the loss of the true label.
- Perturbation lives in the detector's **input pixel space in [0,1]** (the tensor BEFORE
  mean/std normalization), clipped to [0,1] each step. If the detector applies mean/std
  normalization, do it as a differentiable op INSIDE the attacked graph.
- Reuse the model + preprocessing from `../adapters/<Detector>.py` (import it; don't rewrite the model).

## Non-differentiable preprocessing
If the pipeline has a non-differentiable step (DCT via cv2, patch-selection by entropy/argsort,
DWT, JPEG, resize done in PIL): attack the **differentiable core** — perturb the tensor that
feeds the first differentiable module, and treat any preceding non-diff op as fixed (BPDA
identity). Document exactly what you approximated in a top-of-file comment.
If a detector genuinely cannot be attacked (no gradient path at all), still produce the CLEAN
columns and leave adv = clean, and flag it clearly.

## CLI (exact)
```
python attacks/attack_<Detector>.py --manifest <csv> --out <csv> [--device cuda|cpu] [--limit N]
```

## Output CSV (exact columns)
```
path,label,score_clean,pred_clean,score_adv,pred_adv
```
- `label`: 0=real, 1=fake.
- `score_*`: the detector's raw score (higher = more likely fake), clean and after attack.
- `pred_*`: 0/1 using the detector's native threshold (0.5 for prob scores, 0.0 for logit).
- One row per manifest row; on per-image error write empty scores and continue.

## Rules
- Smoke-test with `--device cpu --limit 4` and verify: (1) ‖x_adv − x_clean‖∞ ≤ 8/255
  (+ tiny fp tol), (2) adv scores move in the flip direction vs clean.
- Manifest: 1000 real + 1000 fake images, 512×512-normalized (output of `normalize_images.py`).
