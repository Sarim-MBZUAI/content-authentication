#!/usr/bin/env python
"""White-box PGD (l-inf) attack on C2P-CLIP (AAAI 2025).

Model + preprocessing reused from adapters/C2P-CLIP.py: CLIP ViT-L/14 vision tower
(HF transformers CLIPModel) + linear head, score = sigmoid(logit) = P(fake). The
vision tower / projection have requires_grad_(False), but gradients still flow to
the INPUT, so PGD is exact here (no BPDA approximation).

Attacked pixel space: the [0,1] tensor after translate_duplicate(224)+CenterCrop(224)
+ToTensor. translate_duplicate only tiles up images smaller than 224 (our 512x512
inputs pass through unchanged) and CenterCrop are fixed PIL ops (identity for
gradients); CLIP mean/std normalization is applied differentiably inside the graph.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _pgd_common as C


def build(device):
    import torch
    import torchvision.transforms as T

    A = C.load_adapter("C2P-CLIP.py")     # sets sys.path, imports C2P_CLIP + helper

    model = A.C2P_CLIP(name="openai/clip-vit-large-patch14", num_classes=1)
    state_dict = torch.load(A.CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    geo = T.Compose([
        T.Lambda(lambda img: A.translate_duplicate(img, 224)),
        T.CenterCrop(224),
        T.ToTensor(),
    ])
    normalize = C.make_normalizer(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711], device=device)
    forward_fn = C.make_binary_forward(model, normalize)
    return model, geo, forward_fn


if __name__ == "__main__":
    C.run(build)
