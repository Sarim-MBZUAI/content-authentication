#!/usr/bin/env python
"""Shared PGD-attack machinery for the deepfake-detector attack harnesses.

Attack spec (from PGD_ATTACK_CONTRACT.md, FIXED for every detector):
  - ell_inf PGD, eps = 8/255, 10 steps, step size alpha = 2/255, random start.
  - Untargeted / true-label flip: ascend BCE(logit, true_label). For a FAKE
    image (label=1) this drives the logit / score DOWN (toward "real"); for a
    REAL image (label=0) it drives the score UP (toward "fake").
  - The perturbation lives in the detector's INPUT PIXEL SPACE in [0,1] (the
    tensor produced by the repo's geometric transform + ToTensor, BEFORE
    mean/std normalization). Mean/std normalization is applied as a
    differentiable op INSIDE the attacked graph (see each attack_<D>.py).
  - Projected to the eps-ball and clipped to [0,1] each step.

This module only provides the loop + manifest/CSV plumbing + a helper to split
a torchvision transform into (geometric-prefix, mean, std). Each attack_<D>.py
supplies a `logit_fn(x_pixels)->logit[B]` closure that normalizes internally
and runs the (differentiable) detector.
"""
import argparse
import csv
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F


def load_adapter(path, name):
    """Import an adapter module by file path (runs its top-level imports /
    sys.path setup; its main() is __main__-guarded so it does not execute)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

EPS = 8.0 / 255.0
ALPHA = 2.0 / 255.0
STEPS = 10
LINF_TOL = 1e-5  # tiny fp tolerance for the ||adv-clean||_inf <= eps check


def parse_args(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def read_manifest(path, limit=None):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v)
                         for k, v in row.items()})
            if limit is not None and len(rows) >= limit:
                break
    return rows


def split_normalize(transform):
    """Split a torchvision Compose into (geometric prefix Compose, mean, std).

    Everything up to (but not including) the first Normalize becomes the
    geometric prefix that produces a [0,1] pixel tensor; the Normalize's
    mean/std are returned as (C,1,1) tensors to be applied inside the graph.
    If there is no Normalize, identity mean/std are returned.
    """
    import torchvision.transforms as T

    prefix, norm = [], None
    for t in transform.transforms:
        if isinstance(t, T.Normalize):
            norm = t
        else:
            prefix.append(t)
    if norm is None:
        mean = torch.zeros(3).view(-1, 1, 1)
        std = torch.ones(3).view(-1, 1, 1)
    else:
        mean = torch.as_tensor(norm.mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.as_tensor(norm.std, dtype=torch.float32).view(-1, 1, 1)
    return T.Compose(prefix), mean, std


def pgd(logit_fn, x0, y, eps=EPS, alpha=ALPHA, steps=STEPS, seed_reset=None):
    """ell_inf PGD ascending BCE(logit, y) in [0,1] pixel space.

    logit_fn : callable(x_pixels[B,C,H,W] in [0,1]) -> logit[B]
    x0       : clean pixel tensor in [0,1]
    y        : true labels (float, 0=real 1=fake)
    seed_reset : optional callable run before every forward (used by D3 to
                 freeze its random patch-shuffle permutation so the surrogate
                 is deterministic across steps).
    Returns the adversarial pixel tensor (detached, in [0,1]).
    """
    x0 = x0.detach()
    x = torch.clamp(x0 + torch.empty_like(x0).uniform_(-eps, eps), 0.0, 1.0).detach()
    for _ in range(steps):
        x.requires_grad_(True)
        if seed_reset is not None:
            seed_reset()
        logit = logit_fn(x).flatten()
        loss = F.binary_cross_entropy_with_logits(logit, y)
        grad = torch.autograd.grad(loss, x)[0]
        with torch.no_grad():
            x = x + alpha * grad.sign()
            x = torch.min(torch.max(x, x0 - eps), x0 + eps)
            x = torch.clamp(x, 0.0, 1.0)
        x = x.detach()
    return x


@torch.no_grad()
def score_of(logit_fn, x, seed_reset=None):
    if seed_reset is not None:
        seed_reset()
    return torch.sigmoid(logit_fn(x).flatten())


def run(desc, build):
    """Generic driver. `build(args, device)` returns a dict with:
        logit_fn   : callable(x_pixels)->logit[B]  (differentiable, normalizes inside)
        to_pixels  : callable(PIL.Image)->x0[C,H,W] in [0,1]
        seed_reset : optional callable (or None)
        tag        : short string for log messages
    """
    args = parse_args(desc)
    device = torch.device(args.device)
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    ctx = build(args, device)
    logit_fn = ctx["logit_fn"]
    to_pixels = ctx["to_pixels"]
    seed_reset = ctx.get("seed_reset")
    tag = ctx.get("tag", "ATK")

    rows = read_manifest(args.manifest, args.limit)
    out_cols = ["path", "label", "score_clean", "pred_clean", "score_adv", "pred_adv"]
    results = []
    max_linf = 0.0
    flip_ok = 0
    n_done = 0

    for i, row in enumerate(rows):
        rec = {"path": row["path"], "label": row.get("label", ""),
               "score_clean": "", "pred_clean": "", "score_adv": "", "pred_adv": ""}
        try:
            label = int(float(row["label"]))
            img = Image.open(row["path"]).convert("RGB")
            x0 = to_pixels(img).unsqueeze(0).to(device)
            y = torch.tensor([float(label)], device=device)

            s_clean = float(score_of(logit_fn, x0, seed_reset).item())
            x_adv = pgd(logit_fn, x0, y, seed_reset=seed_reset)
            s_adv = float(score_of(logit_fn, x_adv, seed_reset).item())

            linf = float((x_adv - x0).abs().max().item())
            max_linf = max(max_linf, linf)
            # flip direction: fake(1) should go DOWN, real(0) should go UP
            moved = (s_adv < s_clean) if label == 1 else (s_adv > s_clean)
            flip_ok += int(moved)
            n_done += 1

            rec["score_clean"] = s_clean
            rec["pred_clean"] = int(s_clean > 0.5)
            rec["score_adv"] = s_adv
            rec["pred_adv"] = int(s_adv > 0.5)
            print(f"[{tag}] {i} label={label} clean={s_clean:.4f} adv={s_adv:.4f} "
                  f"Linf={linf:.5f} moved={moved}", file=sys.stderr)
        except Exception as e:  # per-image error: empty scores, continue
            print(f"[{tag}] error on {row.get('path')}: {e}", file=sys.stderr)
        results.append(rec)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(results)

    within = max_linf <= EPS + LINF_TOL
    print(f"[{tag}] wrote {len(results)} rows -> {args.out}")
    print(f"[{tag}] SUMMARY: scored={n_done}/{len(rows)} "
          f"max_Linf={max_linf:.6f} (eps={EPS:.6f}, within={within}) "
          f"flip_moved={flip_ok}/{n_done}")
    return within, max_linf, flip_ok, n_done
