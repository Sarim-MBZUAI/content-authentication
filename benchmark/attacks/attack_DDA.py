#!/usr/bin/env python
"""White-box PGD ell_inf attack on DDA (DINOv2 ViT-L/14 + LoRA + linear head).

Reuses model + preprocessing from adapters/DDA.py (imported, not rewritten).
Fully differentiable: model(x) returns the raw logit, so the attack graph is
just  normalize(x) -> model -> logit,  end to end.

Perturbation is in [0,1] pixel space (after CenterCrop(336)+ToTensor, before
CLIP mean/std normalization); normalization is applied inside the graph.
"""
import os
import sys

import torch
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pgd_lib

ADAPTER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "adapters", "DDA.py")


def build(args, device):
    dda = pgd_lib.load_adapter(ADAPTER, "adapter_DDA")

    model = dda.DINOv2ModelWithLoRA(name="dinov2_vitl14", lora_rank=8,
                                    lora_alpha=1, lora_targets=None)
    ckpt = torch.load(dda.CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    transform = T.Compose([
        T.CenterCrop(336),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    prefix, mean, std = pgd_lib.split_normalize(transform)
    mean, std = mean.to(device), std.to(device)

    def logit_fn(x):
        xn = (x - mean) / std
        return model(xn).reshape(xn.shape[0], -1)[:, 0]

    return {"logit_fn": logit_fn, "to_pixels": prefix, "tag": "DDA"}


if __name__ == "__main__":
    pgd_lib.run("DDA PGD attack", build)
