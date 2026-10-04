#!/usr/bin/env python
"""Benchmark adapter for PROBE (2026) — DINOv2 (with registers) patch classifier.

Reference: detectors/2026/PROBE/code/Detector/evaluate_dino.py.
Model: Detector/model/dino_classifier.py::dino_classifier (HF
       facebook/dinov2-with-registers-large backbone + linear 1-logit head).
       Backbone weights are fully contained in the official checkpoint
       (loaded strict=True as in evaluate_dino.py); from_pretrained only
       provides the architecture (cached under the user's HF cache).
Preprocessing (per evaluate_dino.py, defaults): PIL RGB -> util.data_augment
       (all corruption probs 0, pad to >= 336 with mode='repeat') ->
       ToTensor + ImageNet normalize -> non-overlapping 336x336 sliding
       patches (stride 336) -> mean patch logit -> sigmoid.
Score: sigmoid(mean logit) = P(fake) (label 1 = fake in the repo).
"""

import argparse
import csv
import os
import sys

# The cluster-wide HF_HOME (~/.cache/huggingface) does not contain
# dinov2-with-registers-large; it is cached in the user's own HF cache.
os.environ["HF_HOME"] = "~/.cache/huggingface"

REPO = "detectors/2026/PROBE"
CODE = os.path.join(REPO, "code", "Detector")
CKPT = os.path.join(REPO, "weights", "DINOv2_best_model_step_34999.pth")
BACKBONE = "facebook/dinov2-with-registers-large"
CROP_SIZE = 336  # evaluate_dino.py default --crop_size

sys.path.insert(0, CODE)

# data_augment params exactly as built in evaluate_dino.py __main__ with
# default CLI values (all corruption probabilities zero).
DATA_AUGMENT_PARAMS = {
    "blur_prob": 0.0, "blur_sig_min": 0.0, "blur_sig_max": 3.0, "blur_sig": 0.0,
    "jpeg_prob": 0.0, "jpeg_quality_min": 60, "jpeg_quality_max": 100,
    "jpeg_quality": 100,
    "cutout_prob": 0.0, "cutout_ratio_min": 0.1, "cutout_ratio_max": 0.5,
    "noise_prob": 0.0, "noise_std_min": 0.0, "noise_std_max": 50.0,
    "noise_std": 0.0,
    "resize_prob": 0.0, "resize_scale_min": 0.5, "resize_scale_max": 2.0,
    "resize_scale": 1.0,
    "pad_mode": "repeat",
}


def read_manifest(path, limit=None):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v)
                         for k, v in row.items()})
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_out(path, rows, scores):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generator", "gen_year", "label", "path", "score"])
        for row, s in zip(rows, scores):
            w.writerow([row["generator"], row["gen_year"], row["label"],
                        row["path"], "" if s is None else f"{s:.6f}"])


def sliding_patches(img_tensor, ps):
    """Non-overlapping ps x ps patches (evaluate_dino.py generate_sliding_patches)."""
    import torch
    C, H, W = img_tensor.shape
    if H < ps or W < ps:
        return torch.zeros(0, 3, ps, ps)
    patches = img_tensor.unfold(1, ps, ps).unfold(2, ps, ps)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    return patches.view(-1, C, ps, ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    from model.dino_classifier import dino_classifier
    from util import data_augment

    device = torch.device(args.device)

    model = dino_classifier(model_name=BACKBONE, classifier_type="linear")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    # 'imagenet' normalization (evaluate_dino.py default --normalization).
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    rows = read_manifest(args.manifest, args.limit)
    scores = [None] * len(rows)

    patch_bs = 32 if device.type == "cuda" else 4

    for i, row in enumerate(rows):
        try:
            image = Image.open(row["path"]).convert("RGB")
            image = data_augment(image, data_augment_params=DATA_AUGMENT_PARAMS,
                                 mode="fix", crop_size=CROP_SIZE)
            patches = sliding_patches(transform(image), CROP_SIZE)
            if patches.shape[0] == 0:
                raise RuntimeError("image produced no patches")
            logits = []
            with torch.no_grad():
                for b in range(0, patches.shape[0], patch_bs):
                    out = model(patches[b:b + patch_bs].to(device))
                    logits.append(out.view(-1).cpu())
            avg_logit = torch.cat(logits).mean()
            scores[i] = float(torch.sigmoid(avg_logit).item())
        except Exception as e:
            print(f"[PROBE] error on {row['path']}: {e}", file=sys.stderr)

    write_out(args.out, rows, scores)
    done = sum(s is not None for s in scores)
    print(f"[PROBE] scored {done}/{len(rows)} images -> {args.out}")


if __name__ == "__main__":
    main()
