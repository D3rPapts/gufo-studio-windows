#!/usr/bin/env python3
"""Converts the HF snapshot of microsoft/TRELLIS.2-4B (MIT) plus the TRELLIS v1
sparse-structure decoder into gufo safetensors files, one file per part.

    python -P -E convert_hf_to_gufo.py --trellis2 D:\\llm-modelle\\microsoft\\TRELLIS.2-4B \\
        --trellis1 D:\\llm-modelle\\microsoft\\TRELLIS-image-large --out D:\\llm-modelle\\microsoft\\gufo-trellis2 \\
        [--parts ss_flow,ss_dec,shape_flow_512,shape_dec,tex_flow_512,tex_dec,...] [--dry-run]
    python -P -E convert_hf_to_gufo.py --selftest

Standard library + numpy (the dry run needs no numpy). Safetensors are parsed by hand,
tensors are streamed one at a time (peak ~ one tensor, largest 110 MB as F32), nothing
from the model folder is executed. Run with -P -E (or -I) so the download folder can
never shadow a module.

Rules (see src/models/trellis2/model_description.hpp):
  * names unchanged, except the per-head qk RMS gammas [12,128] -> [1536] (head-major)
  * 5-D conv weights -> [Co, 27*Ci], column t*Ci + ci, tap t = (dx+1)*9 + (dy+1)*3 + (dz+1).
    TRELLIS.2 sparse decoders ship [Co,3,3,3,Ci] (FlexGEMM), ss_dec ships [Co,Ci,3,3,3]
    (PyTorch); both are brought to the same layout
  * 2-D matrices with both dims >= 64 -> BF16, everything else F32 (--keep-f32: all F32)
  * the dense-structure encoder/ texture encoder (training only) are not converted
Outputs: <out>/trellis2-<part>.safetensors with metadata gufo.trellis2.*.
"""

import argparse
import json
import math
import os
import struct
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

DTYPE_SIZE = {"F32": 4, "F16": 2, "BF16": 2}

# part -> (snapshot, relative file, conv layout)
PARTS = {
    "ss_flow": ("trellis2", "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors", None),
    "shape_flow_512": ("trellis2", "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors", None),
    "shape_flow_1024": ("trellis2", "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors", None),
    "tex_flow_512": ("trellis2", "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors", None),
    "tex_flow_1024": ("trellis2", "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors", None),
    "shape_dec": ("trellis2", "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors", "co_tap_ci"),
    "tex_dec": ("trellis2", "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors", "co_tap_ci"),
    "ss_dec": ("trellis1", "ckpts/ss_dec_conv3d_16l8_fp16.safetensors", "co_ci_tap"),
}
DEFAULT_PARTS = ["ss_flow", "ss_dec", "shape_flow_512", "shape_dec", "tex_flow_512", "tex_dec"]

# Expected config values (ckpts/*.json) per part.
EXPECTED = {
    "ss_flow": {"resolution": 16, "in_channels": 8, "out_channels": 8, "model_channels": 1536, "cond_channels": 1024,
                "num_blocks": 30, "num_heads": 12, "pe_mode": "rope", "share_mod": True, "qk_rms_norm": True,
                "qk_rms_norm_cross": True},
    "shape_flow_512": {"resolution": 32, "in_channels": 32, "out_channels": 32, "model_channels": 1536, "num_blocks": 30,
                       "num_heads": 12, "pe_mode": "rope", "share_mod": True},
    "shape_flow_1024": {"resolution": 64, "in_channels": 32, "out_channels": 32, "model_channels": 1536, "num_blocks": 30},
    "tex_flow_512": {"resolution": 32, "in_channels": 64, "out_channels": 32, "model_channels": 1536, "num_blocks": 30},
    "tex_flow_1024": {"resolution": 64, "in_channels": 64, "out_channels": 32, "model_channels": 1536, "num_blocks": 30},
    "shape_dec": {"resolution": 256, "model_channels": [1024, 512, 256, 128, 64], "latent_channels": 32,
                  "num_blocks": [4, 16, 8, 4, 0], "use_fp16": True},
    "tex_dec": {"out_channels": 6, "model_channels": [1024, 512, 256, 128, 64], "latent_channels": 32,
                "num_blocks": [4, 16, 8, 4, 0], "pred_subdiv": False},
    "ss_dec": {"out_channels": 1, "latent_channels": 8, "num_res_blocks": 2, "num_res_blocks_middle": 2,
               "channels": [512, 128, 32], "use_fp16": True},
}


class SafetensorsIn:
    """Reads one safetensors file tensor by tensor (header + seek + read)."""

    def __init__(self, path):
        self.path = path
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n == 0 or n > 64 * 1024 * 1024 or 8 + n > size:
                raise SystemExit(f"{path}: implausible header length {n} (zero-filled download?)")
            header = json.loads(f.read(n).decode("utf-8"))
        self.metadata = header.pop("__metadata__", {}) or {}
        self.tensors = header
        self.base = 8 + n
        end = max((v["data_offsets"][1] for v in header.values()), default=0)
        if self.base + end != size:
            raise SystemExit(f"{path}: payload ends at {self.base + end}, file has {size} bytes")

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
        if dt == "F32":
            arr = np.frombuffer(raw, dtype=np.float32)
        elif dt == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dt == "BF16":
            arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            raise SystemExit(f"{self.path}: unsupported dtype {dt} for {name}")
        return np.ascontiguousarray(arr.reshape(info["shape"]), dtype=np.float32)


def to_bf16(arr):
    bits = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    rounded = np.where(nan, (bits >> 16) | 0x40, rounded)
    return rounded.astype(np.uint16)


# ------------------------------------------------------------------ transforms
def flatten_conv(w, layout):
    """5-D conv weight -> [Co, 27*Ci] with column t*Ci + ci (t = kd*9 + kh*3 + kw)."""
    if layout == "co_tap_ci":          # [Co, kd, kh, kw, Ci] (FlexGEMM)
        co, kd, kh, kw, ci = w.shape
        return np.ascontiguousarray(w.reshape(co, kd * kh * kw * ci))
    if layout == "co_ci_tap":          # [Co, Ci, kd, kh, kw] (PyTorch)
        co, ci, kd, kh, kw = w.shape
        return np.ascontiguousarray(np.transpose(w, (0, 2, 3, 4, 1)).reshape(co, kd * kh * kw * ci))
    raise SystemExit(f"unknown conv layout {layout}")


def converted_shape(name, shape, layout):
    if len(shape) == 5:
        if layout == "co_tap_ci":
            return (shape[0], shape[1] * shape[2] * shape[3] * shape[4])
        return (shape[0], shape[2] * shape[3] * shape[4] * shape[1])
    if name.endswith("rms_norm.gamma") and len(shape) == 2:
        return (shape[0] * shape[1],)
    return tuple(shape)


def output_dtype(name, shape, keep_f32):
    if keep_f32:
        return "F32"
    if len(shape) == 2 and shape[0] >= 64 and shape[1] >= 64 and not name.endswith("gamma"):
        return "BF16"
    return "F32"


# ------------------------------------------------------------------------ plan
def plan_part(part, src, keep_f32):
    _, _, layout = PARTS[part]
    entries = []
    for name in sorted(src.names(), key=natural_key):
        shape = src.shape(name)
        out_shape = converted_shape(name, shape, layout)
        dtype = output_dtype(name, out_shape, keep_f32)

        def producer(n=name, s=shape, lay=layout):
            arr = src.read(n)
            if len(s) == 5:
                return flatten_conv(arr, lay)
            if n.endswith("rms_norm.gamma") and len(s) == 2:
                return arr.reshape(-1)
            return arr

        entries.append((name, out_shape, dtype, producer))
    return entries


def natural_key(name):
    return [int(p) if p.isdigit() else p for p in name.replace(".", " . ").split()]


def write_safetensors(path, entries, metadata):
    header = {"__metadata__": metadata}
    offset = 0
    for name, shape, dtype, _ in entries:
        n = math.prod(shape) * DTYPE_SIZE[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + n]}
        offset += n
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * ((8 - len(text) % 8) % 8)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for i, (name, shape, dtype, producer) in enumerate(entries):
            arr = np.ascontiguousarray(producer(), dtype=np.float32)
            if tuple(arr.shape) != tuple(shape):
                raise SystemExit(f"{name}: produced shape {arr.shape} != planned {shape}")
            f.write((to_bf16(arr) if dtype == "BF16" else arr).tobytes())
            del arr
            if i % 50 == 0 or i + 1 == len(entries):
                print(f"  {i + 1}/{len(entries)} {name}", file=sys.stderr)
    os.replace(tmp, path)


# -------------------------------------------------------------------- selftest
def selftest():
    """Both conv layouts produce the same flattened matrix and the flattened matrix
    reproduces a dense cross-correlation at one voxel; gamma flattening is head-major."""
    rng = np.random.default_rng(0)
    co, ci = 3, 4
    w_torch = rng.standard_normal((co, ci, 3, 3, 3)).astype(np.float32)        # [Co, Ci, kd, kh, kw]
    w_flex = np.ascontiguousarray(np.transpose(w_torch, (0, 2, 3, 4, 1)))       # [Co, kd, kh, kw, Ci]
    a = flatten_conv(w_torch, "co_ci_tap")
    b = flatten_conv(w_flex, "co_tap_ci")
    ok = a.shape == (co, 27 * ci) and np.array_equal(a, b)
    # dense check: 3x3x3 input patch, cross-correlation out = sum_{k,c} w[o,c,kd,kh,kw] * x[c,kd,kh,kw]
    x = rng.standard_normal((ci, 3, 3, 3)).astype(np.float32)
    dense = np.einsum("ocdhw,cdhw->o", w_torch, x)
    im2col = np.transpose(x, (1, 2, 3, 0)).reshape(-1)   # t*Ci + ci with t = kd*9 + kh*3 + kw
    ok = ok and np.allclose(a @ im2col, dense, atol=1e-5)
    gamma = np.arange(12 * 128, dtype=np.float32).reshape(12, 128)
    flat = gamma.reshape(-1)
    ok = ok and flat[128] == 128 and flat[12 * 128 - 1] == 1535
    bf = to_bf16(np.array([1.0, 1.00390625, 1.0078125, 3.4e38, -2.5], dtype=np.float32))
    ok = ok and list(bf) == [0x3F80, 0x3F80, 0x3F81, 0x7F80, 0xC020]  # tie -> even, 1+2^-7, overflow -> inf
    print("selftest", "ok" if ok else "FAILED", file=sys.stderr)
    return 0 if ok else 1


def check_config(part, path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    args = cfg.get("args", {})
    problems = [f"{part}: {k} = {args.get(k)!r}, expected {v!r}" for k, v in EXPECTED[part].items() if args.get(k) != v]
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trellis2", default="", help="HF snapshot of microsoft/TRELLIS.2-4B")
    ap.add_argument("--trellis1", default="", help="HF snapshot of microsoft/TRELLIS-image-large (ss_dec)")
    ap.add_argument("--out", default="", help="output directory")
    ap.add_argument("--parts", default=",".join(DEFAULT_PARTS), help="comma-separated parts (default: 512 path)")
    ap.add_argument("--dry-run", action="store_true", help="list the output tensors, write nothing")
    ap.add_argument("--keep-f32", action="store_true")
    ap.add_argument("--revision", default="", help="HF revision recorded in the metadata")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    for p in parts:
        if p not in PARTS:
            ap.error(f"unknown part {p}; known: {', '.join(PARTS)}")
    if not args.dry_run and not args.out:
        ap.error("--out required unless --dry-run")
    if not args.dry_run and np is None:
        print("numpy is required to write (dry run works without it)", file=sys.stderr)
        return 2

    rc = 0
    for part in parts:
        snapshot, rel, _ = PARTS[part]
        root = args.trellis2 if snapshot == "trellis2" else args.trellis1
        if not root:
            print(f"{part}: --{snapshot} not given, skipped", file=sys.stderr)
            rc = 2
            continue
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            print(f"{part}: missing {path}", file=sys.stderr)
            rc = 2
            continue
        problems = check_config(part, path[:-len(".safetensors")] + ".json")
        for p in problems:
            print("config:", p, file=sys.stderr)
        if problems:
            rc = 2
            continue
        src = SafetensorsIn(path)
        entries = plan_part(part, src, args.keep_f32)
        total = sum(math.prod(s) * DTYPE_SIZE[d] for _, s, d, _ in entries)
        params = sum(math.prod(s) for _, s, _, _ in entries)
        src_params = sum(math.prod(src.shape(n)) for n in src.names())
        if args.dry_run:
            width = max(len(n) for n, _, _, _ in entries)
            for name, shape, dtype, _ in entries:
                print(f"{part}  {name:<{width}}  {dtype:<4}  {str(list(shape)):<24}  {math.prod(shape) * DTYPE_SIZE[dtype]:>12,d}")
        print(f"{part}: {len(entries)} tensors, {params:,d} parameters ({src_params:,d} in the source), "
              f"{total / 2**30:.2f} GiB output", file=sys.stderr)
        if params != src_params:
            print(f"{part}: parameter count changed!", file=sys.stderr)
            rc = 1
        if args.dry_run:
            continue
        os.makedirs(args.out, exist_ok=True)
        out = os.path.join(args.out, f"trellis2-{part}.safetensors")
        metadata = {
            "format": "pt",
            "gufo.trellis2.format": "1",
            "gufo.trellis2.part": part,
            "gufo.trellis2.source": ("microsoft/TRELLIS.2-4B" if snapshot == "trellis2" else "microsoft/TRELLIS-image-large")
                                    + (f"@{args.revision}" if args.revision else ""),
            "gufo.trellis2.license": "MIT (Microsoft), see LICENSE of the HF snapshot",
            "gufo.trellis2.conv_layout": "co_tap_ci",
            "gufo.trellis2.source_file": rel,
        }
        write_safetensors(out, entries, metadata)
        print(f"wrote {out} ({total / 2**30:.2f} GiB)", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
