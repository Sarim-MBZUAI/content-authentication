#!/usr/bin/env python
"""Shared white-box PGD (l-inf) harness for the deepfake-detector attacks.

Contract (PGD_ATTACK_CONTRACT.md):
  - l-inf PGD, eps = 8/255, 10 steps, step alpha = 2/255, random start in the ball.
  - Untargeted "flip": ascend the loss of the TRUE label. For a FAKE image (label 1)
    this drives the fake-probability DOWN (toward real); for a REAL image (label 0)
    it drives it UP (toward fake).
  - The perturbation lives in the detector's INPUT PIXEL SPACE in [0,1] (the tensor
    produced by ToTensor, BEFORE mean/std normalization), clipped to [0,1] each step.
    Mean/std normalization is applied as a differentiable op INSIDE the attacked graph.
  - Any geometric / non-differentiable preprocessing (PIL Resize / CenterCrop /
    translate_duplicate) is treated as fixed: we perturb the [0,1] tensor that comes
    out of it (BPDA-identity on the preceding fixed ops). This is documented per
    detector in each attack_<D>.py header.

Each attack_<D>.py implements build(device) -> (model, geo_transform, forward_fn):
  - model: the detector (already on `device`, eval()).
  - geo_transform: torchvision transform PIL.Image -> float tensor CxHxW in [0,1]
    (all resize/crop/ToTensor, but NO Normalize).
  - forward_fn(x01_batch, labels) -> (scores, loss):
      * x01_batch: [B,C,H,W] float in [0,1], requires_grad during the attack.
      * applies the detector's mean/std normalization differentiably, runs the model.
      * scores: [B] tensor, P(fake) in [0,1] (higher = more likely fake).
      * loss: scalar = summed per-sample loss of the TRUE label (to be ASCENDED).
Output CSV columns (exact): path,label,score_clean,pred_clean,score_adv,pred_adv
"""
import argparse
import csv
import os
import sys

EPS = 8.0 / 255.0
ALPHA = 2.0 / 255.0
STEPS = 10
THRESHOLD = 0.5  # native prob threshold: score > 0.5 -> pred fake(1)


ADAPTERS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "adapters")


def load_adapter(filename):
    """Import an adapters/<filename> module (handles hyphenated names) so we can
    reuse its constants / helper functions and its sys.path side effects."""
    import importlib.util
    path = os.path.join(ADAPTERS_DIR, filename)
    modname = "adapter_" + os.path.splitext(filename)[0].replace("-", "_")
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def make_normalizer(mean, std, device):
    import torch
    m = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    s = torch.tensor(std, device=device).view(1, -1, 1, 1)
    return lambda x: (x - m) / s


def make_binary_forward(model, normalize):
    """sigmoid-logit detectors: model(xn) -> logit [B] or [B,1]. score = sigmoid."""
    import torch
    import torch.nn.functional as F

    def fn(x01, labels):
        logit = model(normalize(x01)).flatten()
        score = torch.sigmoid(logit)
        loss = F.binary_cross_entropy_with_logits(
            logit, labels.float(), reduction="sum")
        return score, loss
    return fn


def make_softmax2_forward(model, normalize):
    """2-logit detectors (FatFormer): model(xn) -> [B,2]. score = softmax[:,1]."""
    import torch
    import torch.nn.functional as F

    def fn(x01, labels):
        logits = model(normalize(x01))
        score = logits.softmax(dim=1)[:, 1]
        loss = F.cross_entropy(logits, labels.long(), reduction="sum")
        return score, loss
    return fn


def pgd(forward_fn, x0, labels, eps=EPS, alpha=ALPHA, steps=STEPS):
    """l-inf PGD that ASCENDS the true-label loss. x0 in [0,1]. Returns x_adv in [0,1]."""
    import torch
    # random start inside the eps-ball, then project to the valid image range.
    delta = torch.empty_like(x0).uniform_(-eps, eps)
    x = torch.clamp(x0 + delta, 0.0, 1.0).detach()
    for _ in range(steps):
        x.requires_grad_(True)
        _, loss = forward_fn(x, labels)
        grad = torch.autograd.grad(loss, x)[0]
        with torch.no_grad():
            x = x + alpha * grad.sign()               # ascend the loss
            x = x0 + torch.clamp(x - x0, -eps, eps)    # project onto l-inf ball
            x = torch.clamp(x, 0.0, 1.0)               # keep a valid image
        x = x.detach()
    return x


def run(build_fn, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args(argv)

    import torch
    from PIL import Image

    device = torch.device(args.device)
    torch.manual_seed(0)

    model, geo_transform, forward_fn = build_fn(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[: args.limit]

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    # results[i] = (score_clean, pred_clean, score_adv, pred_adv) or None on error.
    results = [None] * len(rows)

    def process_batch(idxs, tensors):
        x0 = torch.stack(tensors).to(device)
        labels = torch.tensor([int(rows[i]["label"]) for i in idxs], device=device)
        with torch.no_grad():
            s_clean, _ = forward_fn(x0, labels)
        x_adv = pgd(forward_fn, x0, labels, args.eps, args.alpha, args.steps)
        with torch.no_grad():
            s_adv, _ = forward_fn(x_adv, labels)
        # sanity: l-inf budget respected (+ tiny fp tolerance)
        linf = (x_adv - x0).abs().amax().item()
        assert linf <= args.eps + 1e-5, f"linf {linf} exceeds eps {args.eps}"
        s_clean = s_clean.detach().cpu().tolist()
        s_adv = s_adv.detach().cpu().tolist()
        for j, i in enumerate(idxs):
            sc, sa = float(s_clean[j]), float(s_adv[j])
            results[i] = (sc, int(sc > THRESHOLD), sa, int(sa > THRESHOLD))

    idx_buf, ten_buf = [], []

    def flush():
        if not ten_buf:
            return
        try:
            process_batch(idx_buf, ten_buf)
        except Exception as e:
            print(f"batch failed ({e}); retrying per-image", file=sys.stderr)
            for i, t in zip(idx_buf, ten_buf):
                try:
                    process_batch([i], [t])
                except Exception as e2:
                    print(f"error on {rows[i]['path']}: {e2}", file=sys.stderr)
        idx_buf.clear()
        ten_buf.clear()

    for i, row in enumerate(rows):
        try:
            img = Image.open(row["path"]).convert("RGB")
            ten_buf.append(geo_transform(img))
            idx_buf.append(i)
        except Exception as e:
            print(f"error loading {row['path']}: {e}", file=sys.stderr)
            continue
        if len(ten_buf) >= args.batch:
            flush()
    flush()

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "score_clean", "pred_clean",
                    "score_adv", "pred_adv"])
        for row, r in zip(rows, results):
            if r is None:
                w.writerow([row["path"], row["label"], "", "", "", ""])
            else:
                sc, pc, sa, pa = r
                w.writerow([row["path"], row["label"], sc, pc, sa, pa])
    print(f"wrote {args.out}")
