#!/usr/bin/env python
"""White-box PGD (l-inf) attack on UniversalFakeDetect (UFD, Ojha et al. CVPR 2023).

Model + preprocessing reused from adapters/UFD.py: CLIP ViT-L/14 backbone + linear
head, score = sigmoid(fc(CLIP feat)) = P(fake). The whole image->score path is
differentiable, so PGD is exact here (no BPDA approximation).

Attacked pixel space: the [0,1] tensor after CenterCrop(224)+ToTensor. The PIL
CenterCrop is a fixed crop (treated as identity for gradients); CLIP mean/std
normalization is applied differentiably inside the graph.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _pgd_common as C


def build(device):
    import torch
    import torchvision.transforms as T

    A = C.load_adapter("UFD.py")          # sets sys.path to UFD/code, gives helpers
    A.ensure_clip_cache()
    from models import get_model          # repo module

    model = get_model("CLIP:ViT-L/14")
    model.fc.load_state_dict(torch.load(A.CKPT, map_location="cpu"))
    model.eval().to(device)

    geo = T.Compose([T.CenterCrop(224), T.ToTensor()])
    normalize = C.make_normalizer(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711], device=device)
    forward_fn = C.make_binary_forward(model, normalize)
    return model, geo, forward_fn


if __name__ == "__main__":
    C.run(build)
