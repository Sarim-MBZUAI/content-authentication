#!/usr/bin/env python
"""Benchmark adapter for WaRPAD (NeurIPS 2025) - training-free AI-image detection
via cropping robustness.

Official code:    detectors/2025/WaRPAD/code/final_imgn.py (score logic replicated
                  below; the script itself is a monolithic GenImage evaluation and
                  cannot be imported without its hardcoded dataset paths).
Official weights: detectors/2025/WaRPAD/weights/dinov2_vitl14_pretrain.pth
                  (vanilla DINOv2 ViT-L/14; served to torch.hub via TORCH_HOME
                  cache - training-free method, no fine-tuned head).

Pipeline (final_imgn.py defaults: prep_size=896, patch_size=224, noise_level=0.1,
seed=1):
  Resize((896,896), bicubic) -> ToTensor -> ImageNet normalize
  -> unfold into 16 non-overlapping 224x224 patches
  -> Haar DWT (J=2), zero the low-pass band, inverse DWT = high-frequency image
  -> perturbed = patch - 0.1 * highfreq
  -> cosine similarity between DINOv2 cls tokens of patch and perturbed patch
  -> raw score = mean similarity over the 16 patches ("similarity_min" in repo).

Polarity: real images score HIGHER mean similarity (DINOv2 features of natural
images are robust to removing the reconstructed high-frequency component), fakes
score LOWER.  The contract requires higher = fake, so we output the NEGATED mean
similarity.  (Verified empirically on labeled manifest images.)
"""
import argparse
import csv
import os
import sys

os.environ.setdefault("TORCH_HOME", "/shared/home/sarim.hashmi/usenix/benchmark/torch_home")

CODE = "/shared/home/sarim.hashmi/usenix/detectors/2025/WaRPAD/code"
sys.path.insert(0, CODE)

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from pytorch_wavelets import DWTForward, DWTInverse

ImageFile.LOAD_TRUNCATED_IMAGES = True

PREP_SIZE = 896
PATCH_SIZE = 224
NOISE_LEVEL = 0.1
SEED = 1


def parse_args():
    p = argparse.ArgumentParser(description="WaRPAD benchmark adapter")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def read_manifest(path, limit):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def cal_metric(model, dwt, idwt, input_tensor):
    """Replicates cal_metric() of final_imgn.py; returns mean patch similarity."""
    b, c, h, w = input_tensor.shape
    with torch.no_grad():
        zero_tensor = input_tensor.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        input_tensor = zero_tensor.reshape([b, c, -1, PATCH_SIZE, PATCH_SIZE]).transpose(1, 2)
        input_tensor = input_tensor.reshape([-1, c, PATCH_SIZE, PATCH_SIZE])

        yl, yh = dwt(input_tensor)
        yl_zeros = torch.zeros_like(yl)
        pert_hf = idwt((yl_zeros, yh))
        perturbed_tensor = input_tensor - NOISE_LEVEL * pert_hf

        outputs = model.forward_features(input_tensor, None)["x_norm_clstoken"]
        perturbed_outputs = model.forward_features(perturbed_tensor, None)["x_norm_clstoken"]

        similarity = torch.nn.functional.cosine_similarity(outputs, perturbed_outputs, dim=-1)
        similarity = similarity.unsqueeze(1).reshape([b, -1])

    return torch.mean(similarity, 1).view([b])


def main():
    args = parse_args()
    rows = read_manifest(args.manifest, args.limit)
    device = torch.device(args.device)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model.eval()
    model = model.to(device)

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    dino_transform = transforms.Compose([
        transforms.Resize((PREP_SIZE, PREP_SIZE), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    dwt = DWTForward(J=2, wave="haar").to(device)
    idwt = DWTInverse(wave="haar").to(device)

    scores = [None] * len(rows)
    for i, row in enumerate(rows):
        try:
            img = Image.open(row["path"]).convert("RGB")
            x = dino_transform(img).unsqueeze(0).to(device)
            sim = cal_metric(model, dwt, idwt, x).item()
            # higher similarity => more likely REAL; negate so higher = fake
            scores[i] = -float(sim)
        except Exception as e:
            print(f"[WaRPAD] error scoring {row['path']}: {e}", file=sys.stderr)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generator", "gen_year", "label", "path", "score"])
        for row, s in zip(rows, scores):
            w.writerow([row["generator"], row["gen_year"], row["label"], row["path"],
                        "" if s is None else s])
    print(f"[WaRPAD] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
