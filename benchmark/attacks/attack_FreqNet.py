#!/usr/bin/env python
"""White-box PGD (l-inf) attack on FreqNet (Tan et al. AAAI 2024).

Model + preprocessing reused from adapters/FreqNet.py: freqnet(num_classes=1),
score = sigmoid(logit) = P(fake). FreqNet's spectral-conv layers use torch.fft
(fft2/ifft2/fftshift) and F.conv2d only -- all differentiable in torch -- so PGD
is exact here (no BPDA approximation).

Attacked pixel space: the [0,1] tensor after Resize((256,256))+ToTensor. The PIL
Resize is fixed (identity for gradients); ImageNet mean/std normalization is applied
differentiably inside the graph.

Note: networks/freqnet.py hardcodes .cuda() when allocating its spectrum-conv
Parameters; we redirect Tensor.cuda to the requested device (same trick the adapter
uses) so CPU smoke tests work.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _pgd_common as C


def build(device):
    import torch
    import torchvision.transforms as T

    A = C.load_adapter("FreqNet.py")      # sets sys.path to FreqNet/code
    torch.Tensor.cuda = lambda self, *a, **kw: self.to(device)
    from networks.freqnet import freqnet  # repo module

    model = freqnet(num_classes=1)
    model.load_state_dict(torch.load(A.CKPT, map_location="cpu"), strict=True)
    model.eval().to(device)

    geo = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    normalize = C.make_normalizer(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device=device)
    forward_fn = C.make_binary_forward(model, normalize)
    return model, geo, forward_fn


if __name__ == "__main__":
    C.run(build)
