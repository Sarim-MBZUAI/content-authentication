#!/usr/bin/env python
"""White-box PGD (l-inf) attack on FatFormer (Liu et al. CVPR 2024).

Model + preprocessing reused from adapters/FatFormer.py: CLIP ViT-L/14 with
forgery-aware adapters + language-guided alignment. The repo model outputs 2-class
logits; score = softmax(logits)[:,1] = P(fake). The forward path (CLIP vision tower,
frequency encoder, adapters, text alignment) is differentiable, so PGD is exact here
(no BPDA approximation). PGD ascends cross-entropy of the TRUE label.

Attacked pixel space: the [0,1] tensor after Resize((256,256))+CenterCrop(224)+
ToTensor. The PIL resize/crop are fixed (identity for gradients); ImageNet mean/std
normalization is applied differentiably inside the graph.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _pgd_common as C


def build(device):
    import torchvision.transforms as T

    A = C.load_adapter("FatFormer.py")    # sets sys.path to FatFormer/code
    model = A.build_fatformer(str(device))  # reuse adapter's exact model construction
    model.eval().to(device)

    geo = T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor()])
    normalize = C.make_normalizer(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device=device)
    forward_fn = C.make_softmax2_forward(model, normalize)
    return model, geo, forward_fn


if __name__ == "__main__":
    C.run(build)
