#!/usr/bin/env python
"""White-box PGD (l-inf) attack on NPR (Tan et al. CVPR 2024).

Model + preprocessing reused from adapters/NPR.py: resnet50(num_classes=1),
score = sigmoid(logit) = P(fake). Fully differentiable ResNet, so PGD is exact
here (no BPDA approximation).

Attacked pixel space: the [0,1] tensor after Resize((256,256))+ToTensor. The PIL
Resize is fixed (identity for gradients); ImageNet mean/std normalization is applied
differentiably inside the graph.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _pgd_common as C


def build(device):
    import torch
    import torchvision.transforms as T

    A = C.load_adapter("NPR.py")          # sets sys.path to NPR/code
    from networks.resnet import resnet50  # repo module

    model = resnet50(num_classes=1)
    sd = torch.load(A.CKPT, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)

    geo = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    normalize = C.make_normalizer(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device=device)
    forward_fn = C.make_binary_forward(model, normalize)
    return model, geo, forward_fn


if __name__ == "__main__":
    C.run(build)
