#!/usr/bin/env python3
"""Checks a converted triposg.safetensors against the HF snapshot it came from.

    python -P -E verify_gufo_file.py <HF snapshot> <triposg.safetensors>

* header: tensor count, metadata, expected top-level shapes
* passthrough tensors (norms, biases) are bit-identical F32
* BF16 matrices round-trip within BF16 precision
* head un-interleaving: for DiT block 0 / 10 / 20 and VAE decoder blocks 0 / 16 the
  converted plain q/k/v reproduce the per-head q/k/v that TripoSGAttnProcessor2_0 derives
  from the original stacked weights (random input, numpy)
* DINOv2 fused qkv == concat(query, key, value); baked 224 px table keeps the CLS row
* prints the SHA-256 of the file
numpy only; nothing from the model folder is executed. AI-assisted (Claude Code), 2026-11-12.
"""
import hashlib
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_hf_to_gufo import SafetensorsIn, deinterleave_cross, deinterleave_self  # noqa: E402

EXPECTED_COUNT = 1128
EXPECTED_SHAPES = {
    "dit.proj_in.weight": (2048, 64), "dit.proj_out.weight": (64, 2048),
    "dit.time_proj.linear_1.weight": (8192, 2048), "dit.blocks.20.skip_linear.weight": (2048, 4096),
    "dit.blocks.0.attn2.to_k.weight": (2048, 1024), "dit.blocks.0.attn1.norm_q.weight": (128,),
    "vae.post_quant.weight": (1024, 64), "vae.decoder.proj_query.weight": (1024, 51),
    "vae.decoder.proj_out.weight": (1, 1024), "vae.decoder.blocks.16.attn2.norm_cross.weight": (1024,),
    "dino.embeddings.position_embeddings_224": (257, 1024), "dino.embeddings.position_embeddings": (1, 1370, 1024),
    "dino.encoder.layer.23.attention.qkv.weight": (3072, 1024), "dino.layernorm.weight": (1024,),
}


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


class GufoFile(SafetensorsIn):
    def read_any(self, name):
        info = self.tensors[name]
        start, end = info["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.base + start)
            raw = f.read(end - start)
        if info["dtype"] == "BF16":
            return bf16_to_f32(np.frombuffer(raw, dtype=np.uint16)).reshape(info["shape"])
        return np.frombuffer(raw, dtype=np.float32).reshape(info["shape"])


def rel_err(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def main():
    snapshot, path = sys.argv[1], sys.argv[2]
    out = GufoFile(path)
    problems = []

    def check(ok, what):
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            problems.append(what)

    check(len(out.tensors) == EXPECTED_COUNT, f"tensor count {len(out.tensors)} == {EXPECTED_COUNT}")
    check(out.metadata.get("gufo.triposg.format") == "1", f"metadata {out.metadata}")
    for name, shape in EXPECTED_SHAPES.items():
        check(name in out.tensors and out.shape(name) == shape, f"{name} shape {shape}")

    dit = SafetensorsIn(os.path.join(snapshot, "transformer", "diffusion_pytorch_model.safetensors"))
    vae = SafetensorsIn(os.path.join(snapshot, "vae", "diffusion_pytorch_model.safetensors"))
    dino = SafetensorsIn(os.path.join(snapshot, "image_encoder_dinov2", "model.safetensors"))

    # passthrough / bf16 round trip
    for src, prefix, name in ((dit, "dit", "blocks.5.norm1.weight"), (dit, "dit", "blocks.20.skip_norm.bias"),
                              (vae, "vae", "decoder.proj_query.weight"), (dino, "dino", "layernorm.bias"),
                              (dino, "dino", "encoder.layer.3.layer_scale1.lambda1")):
        check(np.array_equal(src.read(name), out.read_any(f"{prefix}.{name}")), f"{prefix}.{name} bit-identical")
    for src, prefix, name in ((dit, "dit", "blocks.7.ff.net.0.proj.weight"), (dit, "dit", "time_proj.linear_2.weight"),
                              (vae, "vae", "decoder.blocks.16.attn2.to_q.weight"), (dino, "dino", "encoder.layer.0.mlp.fc1.weight")):
        e = rel_err(out.read_any(f"{prefix}.{name}"), src.read(name))
        check(e < 4e-3, f"{prefix}.{name} bf16 round trip rel err {e:.2e}")

    # head un-interleaving against the upstream view/split semantics
    rng = np.random.default_rng(1)

    def upstream_self(src, block, heads):
        wq, wk, wv = (src.read(f"{block}.attn1.to_{c}.weight") for c in "qkv")
        x = rng.standard_normal((4, wq.shape[1])).astype(np.float32)
        d = wq.shape[0] // heads
        qkv = np.concatenate([x @ wq.T, x @ wk.T, x @ wv.T], axis=-1).reshape(4, heads, 3 * d)
        return x, qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]

    def upstream_cross(src, block, heads):
        wk, wv = (src.read(f"{block}.attn2.to_{c}.weight") for c in "kv")
        c = rng.standard_normal((4, wk.shape[1])).astype(np.float32)
        d = wk.shape[0] // heads
        kv = np.concatenate([c @ wk.T, c @ wv.T], axis=-1).reshape(4, heads, 2 * d)
        return c, kv[..., :d], kv[..., d:]

    for src, prefix, block, heads, has_self, has_cross in (
            (dit, "dit", "blocks.0", 16, True, True), (dit, "dit", "blocks.10", 16, True, True),
            (dit, "dit", "blocks.20", 16, True, True),
            (vae, "vae", "decoder.blocks.0", 8, True, False), (vae, "vae", "decoder.blocks.16", 8, False, True)):
        if has_self:
            x, uq, uk, uv = upstream_self(src, block, heads)
            d = uq.shape[-1]
            for c, ref in zip("qkv", (uq, uk, uv)):
                w = out.read_any(f"{prefix}.{block}.attn1.to_{c}.weight")
                got = (x @ w.T).reshape(4, heads, d)
                e = rel_err(got, ref)
                check(e < 1e-2, f"{prefix}.{block}.attn1.to_{c} per-head layout rel err {e:.2e}")
            # also prove the plain (un-permuted) weights would be wrong
            w_plain = src.read(f"{block}.attn1.to_k.weight")
            e_plain = rel_err((x @ w_plain.T).reshape(4, heads, d), uk)
            check(e_plain > 0.5, f"{prefix}.{block}: unpermuted to_k differs from upstream heads (rel err {e_plain:.2f})")
        if has_cross:
            c, uk, uv = upstream_cross(src, block, heads)
            d = uk.shape[-1]
            for cc, ref in zip("kv", (uk, uv)):
                w = out.read_any(f"{prefix}.{block}.attn2.to_{cc}.weight")
                e = rel_err((c @ w.T).reshape(4, heads, d), ref)
                check(e < 1e-2, f"{prefix}.{block}.attn2.to_{cc} per-head layout rel err {e:.2e}")
            wq_src = src.read(f"{block}.attn2.to_q.weight")
            e = rel_err(out.read_any(f"{prefix}.{block}.attn2.to_q.weight"), wq_src)
            check(e < 4e-3, f"{prefix}.{block}.attn2.to_q kept as is (rel err {e:.2e})")

    # DINOv2 fused qkv and baked positions
    for layer in (0, 23):
        base = f"encoder.layer.{layer}.attention.attention"
        ref = np.concatenate([dino.read(f"{base}.{c}.weight") for c in ("query", "key", "value")], axis=0)
        e = rel_err(out.read_any(f"dino.encoder.layer.{layer}.attention.qkv.weight"), ref)
        check(e < 4e-3, f"dino layer {layer} qkv fused rel err {e:.2e}")
        refb = np.concatenate([dino.read(f"{base}.{c}.bias") for c in ("query", "key", "value")], axis=0)
        check(np.array_equal(out.read_any(f"dino.encoder.layer.{layer}.attention.qkv.bias"), refb),
              f"dino layer {layer} qkv bias concatenated")
    pos = dino.read("embeddings.position_embeddings")[0]
    baked = out.read_any("dino.embeddings.position_embeddings_224")
    check(np.array_equal(baked[0], pos[0]), "baked positions keep the CLS row")
    # the 16x16 table must stay close to the 37x37 table's value range
    check(abs(float(baked[1:].mean()) - float(pos[1:].mean())) < 1e-3 and
          baked[1:].std() < pos[1:].std() * 1.05, "baked positions in range")

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    print("sha256", h.hexdigest(), os.path.getsize(path), "bytes")
    print("FAILED" if problems else "all checks passed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
