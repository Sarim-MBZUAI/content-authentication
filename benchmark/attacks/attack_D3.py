#!/usr/bin/env python
"""White-box PGD ell_inf attack on D3 (CLIP ViT-L/14 + patch-shuffle attention head).

Reuses the model + preprocessing from adapters/D3.py (imported, not rewritten).

Differentiable-core note (BPDA-free):
  The official CLIPModelShuffleAttentionPenultimateLayer.forward wraps the CLIP
  backbone feature extraction in `torch.no_grad()` (gradient obfuscation), so
  the stock forward yields no gradient path to the input. We re-assemble the
  SAME modules (model.model.encode_image via its ln_post forward-hook -> the
  penultimate feature captured in model.features -> model.attention_head) with
  autograd ENABLED. No module is reimplemented; we only drop the no_grad guard,
  giving true gradients through the frozen (requires_grad=False) CLIP backbone
  to the input pixels.

  D3's defense is a RANDOM patch-shuffle (torch.randperm) inside the forward.
  We deliberately do NOT freeze the permutation per image: the seed is set once
  (418, as in the repo's validate_for_robustness.py), then the RNG advances
  naturally, so each forward -- clean, every PGD step, and the adv measurement --
  draws a different permutation, exactly as the detector behaves in the full
  benchmark run. This is an honest attack against the randomized defense (each
  PGD step ascends w.r.t. a fresh permutation; the final score is measured under
  yet another draw), so D3 tends to only partially collapse rather than flip to
  exactly 0/1 -- the most robust baseline, as the paper reports. Freezing the
  seed would let a fixed-permutation surrogate defeat the shuffle unrealistically.

Perturbation is in [0,1] pixel space (after Resize(224)+ToTensor, before CLIP
mean/std normalization); the normalization is applied inside the graph.
"""
import os
import sys

import torch
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pgd_lib

ADAPTER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "adapters", "D3.py")


def build(args, device):
    d3 = pgd_lib.load_adapter(ADAPTER, "adapter_D3")

    model = d3.CLIPModelShuffleAttentionPenultimateLayer(
        "ViT-L/14", shuffle_times=1, original_times=1, patch_size=[14])
    state_dict = torch.load(d3.CKPT, map_location="cpu", weights_only=True)
    model.attention_head.load_state_dict(state_dict)
    model.eval().to(device)

    # exact repo eval transform, split into geometric-prefix + normalize
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    prefix, mean, std = pgd_lib.split_normalize(transform)
    mean, std = mean.to(device), std.to(device)

    d3.set_seed(418)  # seed once (repo convention); RNG then advances per forward

    def logit_fn(x):
        xn = (x - mean) / std
        feats = []
        for _ in range(model.shuffle_times):
            model.model.encode_image(model.shuffle_patches(xn, model.patch_size[0]))
            feats.append(model.features)
        model.model.encode_image(xn)
        for _ in range(model.original_times):
            feats.append(model.features)
        out = model.attention_head(torch.stack(feats, dim=-2))
        return out[:, 0] if out.shape[-1] == 2 else out.reshape(out.shape[0], -1)[:, 0]

    return {"logit_fn": logit_fn, "to_pixels": prefix,
            "seed_reset": None, "tag": "D3"}


if __name__ == "__main__":
    pgd_lib.run("D3 PGD attack", build)
