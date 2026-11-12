#!/usr/bin/env python3
"""Replay the MOSS-Music audio frontend (chunking, Conv2d stem, stem_proj,
sinusoidal positions) in plain numpy and compare it with the dumps of
``gufo-moss-music --stage frontend``.

The weights come straight from the safetensors shards of the checkpoint
(read by header offset, BF16 widened to F32; nothing in the model folder is
executed). The log-mel features come from the gufo dump, so this checks the
stem and the chunking, not the mel frontend (which Qwen3-ASR validates).

Usage:
  check_frontend.py --model-dir D:/llm-modelle/OpenMOSS-Team/MOSS-Music-8B-Instruct \
                    --dump D:/claude/gufo-arbeit/moss-port/dumps/fabian_ref

Written with AI assistance (Claude Code), 12.11.2026. Runs with ``python -I``.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

CHANNELS = 480
D_MODEL = 1280
MEL_BINS = 128
CHUNK_FRAMES = 400


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def load_tensor(model_dir: Path, name: str, cache: dict) -> np.ndarray:
    """One tensor as float32, read by offset from its shard."""
    if "index" not in cache:
        with open(model_dir / "model.safetensors.index.json", encoding="utf-8") as f:
            cache["index"] = json.load(f)["weight_map"]
        cache["headers"] = {}
    shard = model_dir / cache["index"][name]
    if shard not in cache["headers"]:
        cache["headers"][shard] = read_header(shard)
    header, start = cache["headers"][shard]
    entry = header[name]
    begin, end = entry["data_offsets"]
    with open(shard, "rb") as f:
        f.seek(start + begin)
        raw = f.read(end - begin)
    if entry["dtype"] == "BF16":
        bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        values = bits.view(np.float32)
    elif entry["dtype"] == "F32":
        values = np.frombuffer(raw, dtype=np.float32)
    elif entry["dtype"] == "F16":
        values = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        raise SystemExit(f"{name}: dtype {entry['dtype']}")
    return values.reshape(entry["shape"])


def erf(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import erf as scipy_erf  # type: ignore
        return scipy_erf(x)
    except Exception:
        return np.frompyfunc(math.erf, 1, 1)(x.astype(np.float64)).astype(np.float32)


def gelu(x: np.ndarray) -> np.ndarray:
    # nn.GELU() default: exact erf form.
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def conv3x3_s2(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """x [cin, H, W], weight [cout, cin, 3, 3], stride 2, padding 1 -> [cout, oh, ow]."""
    cin, h, w = x.shape
    cout = weight.shape[0]
    oh = (h - 1) // 2 + 1
    ow = (w - 1) // 2 + 1
    padded = np.zeros((cin, h + 2, w + 2), dtype=np.float32)
    padded[:, 1:h + 1, 1:w + 1] = x
    patches = np.empty((oh * ow, cin, 3, 3), dtype=np.float32)
    for ky in range(3):
        for kx in range(3):
            # output (y, x) reads padded (2y + ky, 2x + kx)
            patches[:, :, ky, kx] = padded[:, ky:ky + 2 * oh:2, kx:kx + 2 * ow:2].reshape(cin, oh * ow).T
    out = patches.reshape(oh * ow, cin * 9) @ weight.reshape(cout, cin * 9).T + bias
    return out.T.reshape(cout, oh, ow)


def sinusoid_positions(positions: int, d_model: int) -> np.ndarray:
    half = d_model // 2
    increment = math.log(10000.0) / (half - 1)
    inv = np.exp(np.float32(-increment) * np.arange(half, dtype=np.float32))
    scaled = np.arange(positions, dtype=np.float32)[:, None] * inv[None, :]
    return np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dump", required=True, help="PREFIX of gufo-moss-music --dump")
    parser.add_argument("--max-chunks", type=int, default=0, help="compare only the first N chunks (0 = all)")
    args = parser.parse_args()

    with open(args.dump + ".meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    frames = int(meta["frames"])
    tokens = int(meta["tokens"])
    mel = np.fromfile(args.dump + ".mel.f32", dtype=np.float32).reshape(MEL_BINS, frames)
    gufo = np.fromfile(args.dump + ".frontend.f32", dtype=np.float32).reshape(tokens, D_MODEL)

    cache: dict = {}
    w1 = load_tensor(args.model_dir, "audio_encoder.conv1.weight", cache)
    b1 = load_tensor(args.model_dir, "audio_encoder.conv1.bias", cache)
    w2 = load_tensor(args.model_dir, "audio_encoder.conv2.weight", cache)
    b2 = load_tensor(args.model_dir, "audio_encoder.conv2.bias", cache)
    w3 = load_tensor(args.model_dir, "audio_encoder.conv3.weight", cache)
    b3 = load_tensor(args.model_dir, "audio_encoder.conv3.bias", cache)
    wp = load_tensor(args.model_dir, "audio_encoder.stem_proj.weight", cache)
    bp = load_tensor(args.model_dir, "audio_encoder.stem_proj.bias", cache)
    inv_ref = load_tensor(args.model_dir, "audio_encoder.embed_positions.inv_timescales", cache)
    assert w1.shape == (CHANNELS, 1, 3, 3) and w3.shape == (CHANNELS, CHANNELS, 3, 3)
    assert wp.shape == (D_MODEL, CHANNELS * 16), wp.shape

    positions = sinusoid_positions(50, D_MODEL)
    inv_ours = np.exp(np.float32(-math.log(10000.0) / (D_MODEL // 2 - 1)) * np.arange(D_MODEL // 2, dtype=np.float32))
    inv_diff = float(np.max(np.abs(inv_ours - inv_ref)))
    print(f"inv_timescales: checkpoint buffer vs recomputed max|diff| = {inv_diff:.3e}")

    chunks = []
    rest = frames
    while rest > 0:
        chunks.append(min(CHUNK_FRAMES, rest))
        rest -= chunks[-1]

    def conv_len(n: int) -> int:
        return (n - 1) // 2 + 1

    expected_tokens = sum(conv_len(conv_len(conv_len(c))) for c in chunks)
    print(f"frames {frames} -> chunks {chunks} -> tokens {expected_tokens} (dump says {tokens})")
    if expected_tokens != tokens:
        print("FAIL: token count")
        return 1

    begin = 0
    row = 0
    worst_abs = 0.0
    worst_rel = 0.0
    compared = 0
    for index, length in enumerate(chunks):
        if args.max_chunks and index >= args.max_chunks:
            break
        # The reference pads every chunk of a batch to the batch's longest chunk
        # (400 when a full chunk is present) before the convolutions.
        padded = max(chunks[:64]) if len(chunks) > 1 else length
        x = np.zeros((1, MEL_BINS, padded), dtype=np.float32)
        x[0, :, :length] = mel[:, begin:begin + length]
        x = gelu(conv3x3_s2(x, w1, b1))
        x = gelu(conv3x3_s2(x, w2, b2))
        x = gelu(conv3x3_s2(x, w3, b3))  # [480, 16, t]
        t = conv_len(conv_len(conv_len(length)))
        x = x[:, :, :t]
        features = x.transpose(2, 0, 1).reshape(t, CHANNELS * x.shape[1])  # [t, c*16+f]
        out = features @ wp.T + bp + positions[:t]
        ref = gufo[row:row + t]
        diff = np.abs(out - ref)
        rel = diff / (np.abs(ref) + 1e-3)
        cos = float(np.dot(out.ravel(), ref.ravel()) / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-12))
        print(f"chunk {index}: {length} frames -> {t} tokens, max|diff| {diff.max():.3e}, "
              f"max rel {rel.max():.3e}, cos {cos:.7f}, |ref| max {np.abs(ref).max():.3f}")
        worst_abs = max(worst_abs, float(diff.max()))
        worst_rel = max(worst_rel, float(rel.max()))
        compared += t
        begin += length
        row += t
    ok = worst_abs <= 2e-3 and worst_rel <= 2e-2 and compared > 0
    print(f"compared {compared}/{tokens} tokens: max|diff| {worst_abs:.3e}, max rel {worst_rel:.3e} -> "
          f"{'OK' if ok else 'FAIL'} (F32 vs F32, summation order only)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
