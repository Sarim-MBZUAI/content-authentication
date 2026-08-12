#!/usr/bin/env python
"""White-box PGD ell_inf attack on DEAR-c (Corvi-style gated ResNet-50, single logit).

Reuses model + preprocessing from adapters/DEAR.py (imported, not rewritten):
same CorviMaskGatedDetector, same checkpoint via detector.load().

Differentiable note: the adapter scores with detector.predict(), which is
decorated @torch.no_grad(). For the attack we call detector.model(x) directly
(the GatedResNet forward -> single logit); that path is fully differentiable.
The fixed 0/1 channel gate is just a buffer multiply, so gradients flow.

Native resolution, batch 1 (per repo inference.py): the 512x512 input pixel
tensor is attacked directly. Perturbation is in [0,1] pixel space (after
ToTensor, before ImageNet mean/std normalization); normalization is in-graph.
"""
import os
import sys

import torch
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pgd_lib

ADAPTER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "adapters", "DEAR.py")


def build(args, device):
    dear = pgd_lib.load_adapter(ADAPTER, "adapter_DEAR")  # sets sys.path to DEAR code
    from dear.detector.corvi_mask_gated_detector import CorviMaskGatedDetector

    detector = CorviMaskGatedDetector(device=str(device), pretrained=False)
    detector.load(dear.CKPT)
    detector.eval()

    # repo transform: native resolution, ToTensor + ImageNet normalize (batch 1)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    prefix, mean, std = pgd_lib.split_normalize(transform)
    mean, std = mean.to(device), std.to(device)

    def logit_fn(x):
        xn = (x - mean) / std
        return detector.model(xn).reshape(xn.shape[0], -1)[:, 0]

    return {"logit_fn": logit_fn, "to_pixels": prefix, "tag": "DEAR"}


if __name__ == "__main__":
    pgd_lib.run("DEAR PGD attack", build)
