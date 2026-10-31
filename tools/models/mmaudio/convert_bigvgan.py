#!/usr/bin/env python3
"""Fold the weight norm of nvidia/bigvgan_v2_44khz_128band_512x.

Reads bigvgan_generator.pt ({"generator": state_dict} with weight_g /
weight_v pairs, torch.nn.utils.weight_norm dim 0) and writes
bigvgan_v2_44khz_128band_512x.safetensors (FP32) with plain `weight`
tensors, which gufo's MMAudio vocoder (src/models/mmaudio/vocoder.cpp)
loads. Everything else (biases, SnakeBeta alpha / beta in log scale, the
anti-alias filters) is copied unchanged.

    python -I convert_bigvgan.py <bigvgan_generator.pt> <out.safetensors>

CPU only. Written with AI assistance (Claude Code).
"""

import sys

import torch
from safetensors.torch import save_file


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
    state = state.get("generator", state)
    out = {}
    for name, value in state.items():
        if name.endswith(".weight_g"):
            continue
        if name.endswith(".weight_v"):
            base = name[: -len(".weight_v")]
            g = state[base + ".weight_g"].float()
            v = value.float()
            # weight_norm(dim=0): norm over every dimension but the first.
            norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
            out[base + ".weight"] = (v * (g / norm)).contiguous()
        elif name.endswith(".parametrizations.weight.original0"):
            continue
        elif name.endswith(".parametrizations.weight.original1"):
            base = name[: -len(".parametrizations.weight.original1")]
            g = state[base + ".parametrizations.weight.original0"].float()
            v = value.float()
            norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
            out[base + ".weight"] = (v * (g / norm)).contiguous()
        else:
            out[name] = value.float().contiguous()
    save_file(out, sys.argv[2])
    print(f"wrote {len(out)} tensors to {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
