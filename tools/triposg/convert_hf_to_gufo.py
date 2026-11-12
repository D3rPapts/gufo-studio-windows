#!/usr/bin/env python3
"""Converts the HF snapshot of VAST-AI/TripoSG (MIT) into one gufo safetensors file.

    python -I convert_hf_to_gufo.py D:\\llm-modelle\\VAST-AI\\TripoSG out\\triposg.safetensors
    python -I convert_hf_to_gufo.py D:\\llm-modelle\\VAST-AI\\TripoSG --dry-run

Standard library + numpy (numpy only for writing; --dry-run needs none); no torch, no code from the model folder is executed
(safetensors are parsed by hand, configs are plain JSON). Tensors are streamed one
at a time, so peak memory stays around one DiT block (~50 MB) plus the output buffer
of the tensor being written.

What the converter does
  * prefixes: transformer/ -> "dit.", vae/ -> "vae.", image_encoder_dinov2/ -> "dino."
  * un-interleaves the head-major q|k|v (self) and k|v (cross) layout that
    TripoSGAttnProcessor2_0 expects (it views cat(q,k,v) as [tokens, heads, 3*d]);
    the output has plain to_q/to_k/to_v whose rows h*d..(h+1)*d belong to head h
  * fuses DINOv2 query/key/value into one `attention.qkv` ([3072, 1024] + bias) in the
    order q|k|v, as wan22/moge.hip expects
  * bakes `dino.embeddings.position_embeddings_224` ([257, 1024]): the 37x37 patch
    table resampled to 16x16 with PyTorch's bicubic (A = -0.75, align_corners=False,
    scale_factor=(16+0.1)/37, i.e. transformers 4.45 Dinov2 semantics; --pos-mode size
    switches to the newer size=(16,16) call) and the CLS row kept in front
  * stores matrices (2-D, both dims >= 64) as BF16 and everything else (norms, biases,
    tables, the 51-column proj_query) as F32; --keep-f32 keeps everything F32
  * skips the VAE encoder/quant (point-cloud encoding only) unless --with-encoder

The dry run prints the output tensor table with shapes/dtypes/bytes and verifies the
configs against the constants in src/models/triposg/config.hpp.
"""

import argparse
import json
import math
import os
import struct
import sys

try:  # numpy is only needed for writing; the dry run works with the standard library
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

EXPECTED = {
    "transformer/config.json": {"in_channels": 64, "num_attention_heads": 16, "num_layers": 21,
                                "width": 2048, "cross_attention_dim": 1024},
    "vae/config.json": {"latent_channels": 64, "num_attention_heads": 8, "width_decoder": 1024,
                        "width_encoder": 512, "num_layers_decoder": 16, "num_layers_encoder": 8,
                        "embed_frequency": 8, "embed_include_pi": False, "embedding_type": "frequency",
                        "in_channels": 3},
    "image_encoder_dinov2/config.json": {"hidden_size": 1024, "num_hidden_layers": 24, "patch_size": 14,
                                         "num_attention_heads": 16, "image_size": 518,
                                         "layer_norm_eps": 1e-06, "use_swiglu_ffn": False,
                                         "mlp_ratio": 4, "apply_layernorm": True},
    "scheduler/scheduler_config.json": {"num_train_timesteps": 1000, "shift": 1,
                                        "use_dynamic_shifting": False},
}
COMPONENTS = [
    ("dit", "transformer", "diffusion_pytorch_model.safetensors"),
    ("vae", "vae", "diffusion_pytorch_model.safetensors"),
    ("dino", "image_encoder_dinov2", "model.safetensors"),
]
DTYPE_SIZE = {"F32": 4, "F16": 2, "BF16": 2}


# ----------------------------------------------------------------------------- IO
class SafetensorsIn:
    """Reads one safetensors file tensor by tensor (header + seek + read)."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n).decode("utf-8"))
        self.metadata = header.pop("__metadata__", {})
        self.tensors = header
        self.base = 8 + n

    def names(self):
        return list(self.tensors)

    def shape(self, name):
        return tuple(self.tensors[name]["shape"])

    def dtype(self, name):
        return self.tensors[name]["dtype"]

    def read(self, name):
        info = self.tensors[name]
        start, end = info["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.base + start)
            raw = f.read(end - start)
        dt = info["dtype"]
        dtype_np = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}
        if dt not in dtype_np:
            raise SystemExit(f"{self.path}: unsupported dtype {dt} for {name}")
        arr = np.frombuffer(raw, dtype=dtype_np[dt]).reshape(info["shape"])
        if dt == "F16":
            arr = arr.astype(np.float32)
        elif dt == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return np.ascontiguousarray(arr, dtype=np.float32)


def shards_for(component_dir, filename):
    """Returns the list of safetensors files for a component (single file or index)."""
    index = os.path.join(component_dir, filename + ".index.json")
    if os.path.isfile(index):
        with open(index, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        return sorted({os.path.join(component_dir, v) for v in weight_map.values()})
    return [os.path.join(component_dir, filename)]


def to_bf16(arr):
    bits = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    rounded = np.where(nan, (bits >> 16) | 0x40, rounded)
    return rounded.astype(np.uint16)


# ------------------------------------------------------------------- transforms
def deinterleave_self(wq, wk, wv, heads):
    """Upstream views cat(q,k,v) as [tokens, heads, 3*d]; returns plain q/k/v weights."""
    stacked = np.concatenate([wq, wk, wv], axis=0)  # [3*W, in]
    width = wq.shape[0]
    d = width // heads
    stacked = stacked.reshape(heads, 3, d, stacked.shape[1])
    return (np.ascontiguousarray(stacked[:, 0].reshape(width, -1)),
            np.ascontiguousarray(stacked[:, 1].reshape(width, -1)),
            np.ascontiguousarray(stacked[:, 2].reshape(width, -1)))


def deinterleave_cross(wk, wv, heads):
    """Upstream views cat(k,v) as [tokens, heads, 2*d]; returns plain k/v weights."""
    stacked = np.concatenate([wk, wv], axis=0)
    width = wk.shape[0]
    d = width // heads
    stacked = stacked.reshape(heads, 2, d, stacked.shape[1])
    return (np.ascontiguousarray(stacked[:, 0].reshape(width, -1)),
            np.ascontiguousarray(stacked[:, 1].reshape(width, -1)))


def cubic_weights(t, a=-0.75):
    """PyTorch's cubic convolution coefficients for fractional offset t in [0, 1)."""
    def k1(x):  # |x| <= 1
        return ((a + 2) * x - (a + 3)) * x * x + 1

    def k2(x):  # 1 < |x| < 2
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a

    return np.array([k2(t + 1), k1(t), k1(1 - t), k2(2 - t)], dtype=np.float64)


def bicubic_resample(table, n_in, n_out, mode):
    """table [n_in*n_in, C] (row-major y, x) -> [n_out*n_out, C], torch bicubic,
    align_corners=False. mode 'scale' uses scale_factor=(n_out+0.1)/n_in like
    transformers 4.45 Dinov2; mode 'size' uses size=(n_out, n_out)."""
    if mode == "scale":
        scale = 1.0 / ((n_out + 0.1) / n_in)
        assert int(math.floor(n_in * (n_out + 0.1) / n_in)) == n_out
    else:
        scale = n_in / n_out
    channels = table.shape[1]
    grid = table.reshape(n_in, n_in, channels).astype(np.float64)

    def axis_matrix():
        m = np.zeros((n_out, n_in), dtype=np.float64)
        for o in range(n_out):
            src = (o + 0.5) * scale - 0.5
            i0 = int(math.floor(src))
            t = src - i0
            w = cubic_weights(t)
            for k in range(4):
                idx = min(max(i0 - 1 + k, 0), n_in - 1)
                m[o, idx] += w[k]
        return m

    m = axis_matrix()
    out = np.einsum("oy,yxc->oxc", m, grid)
    out = np.einsum("px,oxc->opc", m, out)
    return out.reshape(n_out * n_out, channels).astype(np.float32)


# ------------------------------------------------------------------------ plan
class Plan:
    def __init__(self, keep_f32):
        self.keep_f32 = keep_f32
        self.entries = []  # (name, shape, dtype, producer)

    def add(self, name, shape, producer, force_f32=False):
        shape = tuple(int(s) for s in shape)
        bf16 = (not self.keep_f32 and not force_f32 and len(shape) == 2
                and shape[0] >= 64 and shape[1] >= 64)
        self.entries.append((name, shape, "BF16" if bf16 else "F32", producer))

    def nbytes(self):
        return sum(math.prod(s) * DTYPE_SIZE[d] for _, s, d, _ in self.entries)


def plan_component(plan, prefix, src, heads, with_encoder):
    names = set(src.names())
    done = set()

    def passthrough(name, force_f32=False):
        plan.add(prefix + "." + name, src.shape(name), lambda n=name: src.read(n), force_f32)
        done.add(name)

    if prefix in ("dit", "vae"):
        blocks = sorted({n.split(".")[-5] if False else ".".join(n.split(".")[:-3])
                         for n in names if ".attn1.to_q.weight" in n or ".attn2.to_q.weight" in n})
        for block in sorted(blocks, key=lambda b: [int(p) if p.isdigit() else p for p in b.split(".")]):
            if prefix == "vae" and block.startswith("encoder") and not with_encoder:
                continue
            q, k, v = (f"{block}.attn1.to_{c}.weight" for c in "qkv")
            if q in names:
                def self_qkv(q=q, k=k, v=v):
                    return deinterleave_self(src.read(q), src.read(k), src.read(v), heads)
                for i, n in enumerate((q, k, v)):
                    plan.add(prefix + "." + n, src.shape(n), lambda i=i, f=self_qkv: f()[i])
                    done.add(n)
            k2, v2 = (f"{block}.attn2.to_{c}.weight" for c in "kv")
            if k2 in names:
                def cross_kv(k=k2, v=v2):
                    return deinterleave_cross(src.read(k), src.read(v), heads)
                for i, n in enumerate((k2, v2)):
                    plan.add(prefix + "." + n, src.shape(n), lambda i=i, f=cross_kv: f()[i])
                    done.add(n)
        for name in sorted(names - done):
            if prefix == "vae" and not with_encoder and (name.startswith("encoder.") or name.startswith("quant.")):
                continue
            passthrough(name, force_f32=name.endswith("proj_query.weight") or name.endswith("proj_out.weight") and src.shape(name)[0] == 1)
    else:  # dino
        layers = sorted({int(n.split(".")[2]) for n in names if n.startswith("encoder.layer.")})
        for layer in layers:
            base = f"encoder.layer.{layer}.attention.attention"
            parts = [f"{base}.{c}.weight" for c in ("query", "key", "value")]
            biases = [f"{base}.{c}.bias" for c in ("query", "key", "value")]
            plan.add(f"dino.encoder.layer.{layer}.attention.qkv.weight", (3 * src.shape(parts[0])[0], src.shape(parts[0])[1]),
                     lambda p=parts: np.concatenate([src.read(n) for n in p], axis=0))
            plan.add(f"dino.encoder.layer.{layer}.attention.qkv.bias", (3 * src.shape(biases[0])[0],),
                     lambda b=biases: np.concatenate([src.read(n) for n in b], axis=0))
            done.update(parts + biases)
        pos = "embeddings.position_embeddings"
        if pos in names:
            n_in = int(round(math.sqrt(src.shape(pos)[1] - 1)))
            n_out = 16
            plan.add("dino.embeddings.position_embeddings_224", (1 + n_out * n_out, src.shape(pos)[2]),
                     lambda: bake_positions(src.read(pos)[0], n_in, n_out, plan.pos_mode), force_f32=True)
        for name in sorted(names - done):
            if name == "embeddings.mask_token":
                continue  # training only
            # Only the MLP / output projections are matrices worth BF16; the patch
            # embedding (4-D), tables, norms and LayerScale stay F32.
            passthrough(name, force_f32=not (name.endswith(".weight") and (".mlp." in name or ".output.dense" in name)))


def bake_positions(table, n_in, n_out, mode):
    cls = table[:1]
    patches = bicubic_resample(table[1:], n_in, n_out, mode)
    return np.concatenate([cls, patches], axis=0)


# ----------------------------------------------------------------------- write
def write_safetensors(path, plan, metadata):
    header = {"__metadata__": metadata}
    offset = 0
    for name, shape, dtype, _ in plan.entries:
        n = math.prod(shape) * DTYPE_SIZE[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + n]}
        offset += n
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * ((8 - len(text) % 8) % 8)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for i, (name, shape, dtype, producer) in enumerate(plan.entries):
            arr = np.ascontiguousarray(producer(), dtype=np.float32)
            if tuple(arr.shape) != shape:
                raise SystemExit(f"{name}: produced shape {arr.shape} != planned {shape}")
            data = to_bf16(arr) if dtype == "BF16" else arr
            f.write(data.tobytes())
            if i % 50 == 0 or i + 1 == len(plan.entries):
                print(f"  {i + 1}/{len(plan.entries)} {name}", file=sys.stderr)
    os.replace(tmp, path)


# -------------------------------------------------------------------- selftest
def selftest():
    """Checks the head un-interleaving against the upstream view/split semantics
    (random weights, 4 heads x 8) and the bicubic resample against a constant table."""
    rng = np.random.default_rng(0)
    heads, d, width, tokens, cross = 4, 8, 32, 5, 12
    x = rng.standard_normal((tokens, width)).astype(np.float32)
    c = rng.standard_normal((tokens, cross)).astype(np.float32)
    wq, wk, wv = (rng.standard_normal((width, width)).astype(np.float32) for _ in range(3))
    # upstream: cat(q,k,v) viewed as [tokens, heads, 3*d]
    qkv = np.concatenate([x @ wq.T, x @ wk.T, x @ wv.T], axis=-1).reshape(tokens, heads, 3 * d)
    uq, uk, uv = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]
    pq, pk, pv = deinterleave_self(wq, wk, wv, heads)
    ok = (np.allclose((x @ pq.T).reshape(tokens, heads, d), uq) and
          np.allclose((x @ pk.T).reshape(tokens, heads, d), uk) and
          np.allclose((x @ pv.T).reshape(tokens, heads, d), uv))
    wk2, wv2 = (rng.standard_normal((width, cross)).astype(np.float32) for _ in range(2))
    kv = np.concatenate([c @ wk2.T, c @ wv2.T], axis=-1).reshape(tokens, heads, 2 * d)
    pk2, pv2 = deinterleave_cross(wk2, wv2, heads)
    ok = ok and (np.allclose((c @ pk2.T).reshape(tokens, heads, d), kv[..., :d]) and
                 np.allclose((c @ pv2.T).reshape(tokens, heads, d), kv[..., d:]))
    const = np.full((37 * 37, 3), 2.5, dtype=np.float32)
    for mode in ("scale", "size"):
        out = bicubic_resample(const, 37, 16, mode)
        ok = ok and out.shape == (256, 3) and np.allclose(out, 2.5, atol=1e-5)
    ramp = np.repeat(np.arange(37, dtype=np.float32), 37)[:, None]  # linear in y
    out = bicubic_resample(ramp, 37, 16, "size").reshape(16, 16)
    ok = ok and np.allclose(out[1:-1, 0], out[1:-1, 5]) and out[8, 0] > out[2, 0]
    print("selftest", "ok" if ok else "FAILED", file=sys.stderr)
    return 0 if ok else 1


# ------------------------------------------------------------------------ main
def check_configs(root):
    problems = []
    for rel, expected in EXPECTED.items():
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            problems.append(f"missing {rel}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key, want in expected.items():
            if cfg.get(key) != want:
                problems.append(f"{rel}: {key} = {cfg.get(key)!r}, expected {want!r}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", nargs="?", default="", help="HF snapshot of VAST-AI/TripoSG")
    ap.add_argument("output", nargs="?", help="output safetensors (omit with --dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="list the output tensors, write nothing")
    ap.add_argument("--with-encoder", action="store_true", help="also convert the VAE encoder + quant")
    ap.add_argument("--keep-f32", action="store_true", help="store everything as F32")
    ap.add_argument("--pos-mode", choices=("scale", "size"), default="scale",
                    help="bicubic call used for the 224 px position table (default: scale)")
    ap.add_argument("--revision", default="", help="HF revision recorded in the metadata")
    ap.add_argument("--selftest", action="store_true", help="check the head un-interleaving and resampling")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.model_dir:
        ap.error("model_dir required")
    if not args.dry_run and not args.output:
        ap.error("output path required unless --dry-run")

    problems = check_configs(args.model_dir)
    for p in problems:
        print("config:", p, file=sys.stderr)
    if problems:
        return 2

    plan = Plan(args.keep_f32)
    plan.pos_mode = args.pos_mode
    source_params = 0
    for prefix, sub, filename in COMPONENTS:
        heads = {"dit": 16, "vae": 8, "dino": 16}[prefix]
        comp_dir = os.path.join(args.model_dir, sub)
        for shard in shards_for(comp_dir, filename):
            if not os.path.isfile(shard):
                print(f"missing {shard}", file=sys.stderr)
                return 2
            src = SafetensorsIn(shard)
            source_params += sum(math.prod(src.shape(n)) for n in src.names())
            plan_component(plan, prefix, src, heads, args.with_encoder)

    total = plan.nbytes()
    if args.dry_run:
        width = max(len(n) for n, _, _, _ in plan.entries)
        for name, shape, dtype, _ in plan.entries:
            print(f"{name:<{width}}  {dtype:<4}  {str(list(shape)):<24}  {math.prod(shape) * DTYPE_SIZE[dtype]:>12,d}")
    out_params = sum(math.prod(s) for _, s, _, _ in plan.entries)
    print(f"{len(plan.entries)} tensors, {out_params:,d} parameters "
          f"({source_params:,d} in the source), {total / 2**30:.2f} GiB output", file=sys.stderr)
    if args.dry_run:
        return 0
    if np is None:
        print("numpy is required to write the output (dry run works without it)", file=sys.stderr)
        return 2

    metadata = {
        "format": "pt",
        "gufo.triposg.format": "1",
        "gufo.triposg.source": "VAST-AI/TripoSG" + (f"@{args.revision}" if args.revision else ""),
        "gufo.triposg.license": "MIT (VAST-AI-Research), see README.md of the HF snapshot",
        "gufo.triposg.pos_mode": args.pos_mode,
        "gufo.triposg.with_encoder": "1" if args.with_encoder else "0",
    }
    write_safetensors(args.output, plan, metadata)
    print(f"wrote {args.output} ({total / 2**30:.2f} GiB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
