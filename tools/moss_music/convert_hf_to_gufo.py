#!/usr/bin/env python3
"""Convert the MOSS-Music-8B-Instruct checkpoint (OpenMOSS-Team, safetensors)
into the files gufo's native port reads.

Outputs (``<out>/<stem>`` defaults to ``moss-music-8b``):

* ``<stem>.audio.gguf`` - audio encoder (Conv2d stem, 32 Whisper layers, final
  LayerNorm), ``audio_adapter`` and the three ``deepstack_audio_merger_list``
  gated MLPs. Tensor names stay the Hugging Face names; matrices keep BF16,
  vectors (biases, norms) become F32. The sinusoid buffer is dropped: gufo
  recomputes it.
* ``<stem>.llm-bf16/`` - the Qwen3-8B language model as a plain
  ``Qwen3ForCausalLM`` folder (``language_model.*`` renamed to ``model.*``,
  ``lm_head.weight`` kept, config.json + tokenizer files copied). llama.cpp's
  ``convert_hf_to_gguf.py --outtype bf16`` and ``llama-quantize`` turn it into
  the ``qwen3`` GGUF (Q8_0 / Q4_K_M) the gufo LLM runtime loads.

Everything is streamed tensor by tensor (16 MiB blocks), so the 18 GB
checkpoint never sits in RAM; the host needs ~100 MB.

``--dry-run`` lists the mapping from model.safetensors.index.json alone
(weights not needed) and checks the parameter count against the index
metadata; it only imports the standard library.

Usage:
  convert_hf_to_gufo.py --dry-run SRC_OR_INDEX_JSON
  convert_hf_to_gufo.py audio SRC OUT_DIR [--stem moss-music-8b] [--force]
  convert_hf_to_gufo.py llm   SRC OUT_DIR [--stem moss-music-8b] [--force]
  convert_hf_to_gufo.py all   SRC OUT_DIR

Written with AI assistance (Claude Code), 12.11.2026.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dimensions (config.json of the checkpoint; src/models/moss_music/config.hpp
# carries the same numbers and tests the derived shapes).

AUDIO = {
    "mel_bins": 128,
    "d_model": 1280,
    "layers": 32,
    "heads": 20,
    "ffn": 5120,
    "conv_channels": 480,
    "stem_frequency": 16,  # 128 -> 64 -> 32 -> 16
    "n_window": 200,       # chunk = 400 mel frames = 4 s = 50 tokens
}
ADAPTER = {"input": 1280, "hidden": 8192, "output": 4096}
TEXT = {
    "layers": 36,
    "hidden": 4096,
    "heads": 32,
    "kv_heads": 8,
    "head_dim": 128,
    "intermediate": 12288,
    "vocab": 151936,
}

# Qwen3 vocabulary ids used by the prompt (processing_moss_music.py).
TOKENS = {"audio_pad": 151654, "audio_start": 151669, "audio_end": 151670,
          "im_start": 151644, "im_end": 151645, "endoftext": 151643}

INDEX_PARAMETERS = 9_052_463_488   # metadata.total_parameters (no buffer)
INDEX_BYTES = 18_104_928_256       # metadata.total_size (BF16, with buffer)

DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2}
GGML_F32 = 0
GGML_F16 = 1
GGML_BF16 = 30
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32
BLOCK_BYTES = 16 << 20


# ---------------------------------------------------------------------------
# Mapping table: HF name -> (target, gufo name, shape, kind)
#   target: "audio" (our GGUF), "llm" (renamed safetensors), "drop"
#   kind:   "matrix" (BF16 stays BF16), "vector" (-> F32), "buffer"

def _gated_mlp(prefix: str, target: str) -> list[tuple[str, str, str, tuple, str]]:
    a = ADAPTER
    return [
        (f"{prefix}.gate_proj.weight", target, f"{prefix}.gate_proj.weight", (a["hidden"], a["input"]), "matrix"),
        (f"{prefix}.up_proj.weight", target, f"{prefix}.up_proj.weight", (a["hidden"], a["input"]), "matrix"),
        (f"{prefix}.down_proj.weight", target, f"{prefix}.down_proj.weight", (a["output"], a["hidden"]), "matrix"),
    ]


def build_mapping() -> dict[str, tuple[str, str, tuple, str]]:
    """Returns {hf_name: (target, gufo_name, shape, kind)} for every tensor."""
    rows: list[tuple[str, str, str, tuple, str]] = []
    d, c, ffn = AUDIO["d_model"], AUDIO["conv_channels"], AUDIO["ffn"]
    flat = c * AUDIO["stem_frequency"]
    enc = "audio_encoder"
    rows += [
        (f"{enc}.conv1.weight", "audio", f"{enc}.conv1.weight", (c, 1, 3, 3), "matrix"),
        (f"{enc}.conv1.bias", "audio", f"{enc}.conv1.bias", (c,), "vector"),
        (f"{enc}.conv2.weight", "audio", f"{enc}.conv2.weight", (c, c, 3, 3), "matrix"),
        (f"{enc}.conv2.bias", "audio", f"{enc}.conv2.bias", (c,), "vector"),
        (f"{enc}.conv3.weight", "audio", f"{enc}.conv3.weight", (c, c, 3, 3), "matrix"),
        (f"{enc}.conv3.bias", "audio", f"{enc}.conv3.bias", (c,), "vector"),
        (f"{enc}.stem_proj.weight", "audio", f"{enc}.stem_proj.weight", (d, flat), "matrix"),
        (f"{enc}.stem_proj.bias", "audio", f"{enc}.stem_proj.bias", (d,), "vector"),
        (f"{enc}.embed_positions.inv_timescales", "drop", "", (d // 2,), "buffer"),
    ]
    for layer in range(AUDIO["layers"]):
        p = f"{enc}.layers.{layer}"
        rows += [
            (f"{p}.self_attn_layer_norm.weight", "audio", f"{p}.self_attn_layer_norm.weight", (d,), "vector"),
            (f"{p}.self_attn_layer_norm.bias", "audio", f"{p}.self_attn_layer_norm.bias", (d,), "vector"),
            (f"{p}.self_attn.q_proj.weight", "audio", f"{p}.self_attn.q_proj.weight", (d, d), "matrix"),
            (f"{p}.self_attn.q_proj.bias", "audio", f"{p}.self_attn.q_proj.bias", (d,), "vector"),
            (f"{p}.self_attn.k_proj.weight", "audio", f"{p}.self_attn.k_proj.weight", (d, d), "matrix"),
            (f"{p}.self_attn.v_proj.weight", "audio", f"{p}.self_attn.v_proj.weight", (d, d), "matrix"),
            (f"{p}.self_attn.v_proj.bias", "audio", f"{p}.self_attn.v_proj.bias", (d,), "vector"),
            (f"{p}.self_attn.out_proj.weight", "audio", f"{p}.self_attn.out_proj.weight", (d, d), "matrix"),
            (f"{p}.self_attn.out_proj.bias", "audio", f"{p}.self_attn.out_proj.bias", (d,), "vector"),
            (f"{p}.final_layer_norm.weight", "audio", f"{p}.final_layer_norm.weight", (d,), "vector"),
            (f"{p}.final_layer_norm.bias", "audio", f"{p}.final_layer_norm.bias", (d,), "vector"),
            (f"{p}.fc1.weight", "audio", f"{p}.fc1.weight", (ffn, d), "matrix"),
            (f"{p}.fc1.bias", "audio", f"{p}.fc1.bias", (ffn,), "vector"),
            (f"{p}.fc2.weight", "audio", f"{p}.fc2.weight", (d, ffn), "matrix"),
            (f"{p}.fc2.bias", "audio", f"{p}.fc2.bias", (d,), "vector"),
        ]
    rows += [
        (f"{enc}.layer_norm.weight", "audio", f"{enc}.layer_norm.weight", (d,), "vector"),
        (f"{enc}.layer_norm.bias", "audio", f"{enc}.layer_norm.bias", (d,), "vector"),
    ]
    rows += _gated_mlp("audio_adapter", "audio")
    for k in range(3):
        rows += _gated_mlp(f"deepstack_audio_merger_list.{k}", "audio")

    t = TEXT
    h, q, kv = t["hidden"], t["heads"] * t["head_dim"], t["kv_heads"] * t["head_dim"]
    rows.append(("language_model.embed_tokens.weight", "llm", "model.embed_tokens.weight", (t["vocab"], h), "matrix"))
    for layer in range(t["layers"]):
        p = f"language_model.layers.{layer}"
        g = f"model.layers.{layer}"
        rows += [
            (f"{p}.input_layernorm.weight", "llm", f"{g}.input_layernorm.weight", (h,), "vector"),
            (f"{p}.self_attn.q_proj.weight", "llm", f"{g}.self_attn.q_proj.weight", (q, h), "matrix"),
            (f"{p}.self_attn.k_proj.weight", "llm", f"{g}.self_attn.k_proj.weight", (kv, h), "matrix"),
            (f"{p}.self_attn.v_proj.weight", "llm", f"{g}.self_attn.v_proj.weight", (kv, h), "matrix"),
            (f"{p}.self_attn.o_proj.weight", "llm", f"{g}.self_attn.o_proj.weight", (h, q), "matrix"),
            (f"{p}.self_attn.q_norm.weight", "llm", f"{g}.self_attn.q_norm.weight", (t["head_dim"],), "vector"),
            (f"{p}.self_attn.k_norm.weight", "llm", f"{g}.self_attn.k_norm.weight", (t["head_dim"],), "vector"),
            (f"{p}.post_attention_layernorm.weight", "llm", f"{g}.post_attention_layernorm.weight", (h,), "vector"),
            (f"{p}.mlp.gate_proj.weight", "llm", f"{g}.mlp.gate_proj.weight", (t["intermediate"], h), "matrix"),
            (f"{p}.mlp.up_proj.weight", "llm", f"{g}.mlp.up_proj.weight", (t["intermediate"], h), "matrix"),
            (f"{p}.mlp.down_proj.weight", "llm", f"{g}.mlp.down_proj.weight", (h, t["intermediate"]), "matrix"),
        ]
    rows.append(("language_model.norm.weight", "llm", "model.norm.weight", (h,), "vector"))
    rows.append(("lm_head.weight", "llm", "lm_head.weight", (t["vocab"], h), "matrix"))

    mapping = {}
    for hf, target, gufo, shape, kind in rows:
        if hf in mapping:
            raise SystemExit(f"duplicate mapping row {hf}")
        mapping[hf] = (target, gufo, tuple(int(x) for x in shape), kind)
    return mapping


def elements(shape) -> int:
    n = 1
    for x in shape:
        n *= int(x)
    return n


# ---------------------------------------------------------------------------
# safetensors reading (streamed)

def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def load_index(src: Path) -> tuple[dict, Path]:
    """Accepts a model directory or the index json itself."""
    index_path = src if src.is_file() else src / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"no model.safetensors.index.json at {src}")
    with open(index_path, encoding="utf-8") as f:
        return json.load(f), index_path.parent


def collect(src: Path) -> dict[str, tuple[Path, dict, int]]:
    """{name: (shard, entry, data_start)} from the shards named by the index."""
    index, root = load_index(src)
    tensors: dict[str, tuple[Path, dict, int]] = {}
    for shard_name in sorted(set(index["weight_map"].values())):
        shard = root / shard_name
        if not shard.exists():
            raise SystemExit(f"shard missing: {shard}")
        header, start = read_header(shard)
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if entry["dtype"] not in DTYPE_BYTES:
                raise SystemExit(f"{shard}: unsupported dtype {entry['dtype']} for {name}")
            if name in tensors:
                raise SystemExit(f"duplicate tensor {name}")
            tensors[name] = (shard, entry, start)
    return tensors


def stream_blocks(shard: Path, entry: dict, start: int):
    begin, end = entry["data_offsets"]
    with open(shard, "rb") as f:
        f.seek(start + begin)
        remaining = end - begin
        while remaining > 0:
            take = min(remaining, BLOCK_BYTES)
            raw = f.read(take)
            if len(raw) != take:
                raise SystemExit(f"short read in {shard}")
            remaining -= take
            yield raw


def bf16_to_f32_bytes(raw: bytes) -> bytes:
    import numpy as np
    bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32).tobytes()


def f16_to_bf16_bytes(raw: bytes) -> bytes:
    import numpy as np
    values = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    bits = values.view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return rounded.tobytes()


def f32_to_bf16_bytes(raw: bytes) -> bytes:
    import numpy as np
    bits = np.frombuffer(raw, dtype=np.float32).view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return rounded.tobytes()


def convert_block(raw: bytes, source: str, target: str) -> bytes:
    if source == target:
        return raw
    if target == "F32":
        if source == "BF16":
            return bf16_to_f32_bytes(raw)
        if source == "F16":
            import numpy as np
            return np.frombuffer(raw, dtype=np.float16).astype(np.float32).tobytes()
    if target == "BF16":
        if source == "F32":
            return f32_to_bf16_bytes(raw)
        if source == "F16":
            return f16_to_bf16_bytes(raw)
    raise SystemExit(f"no conversion {source} -> {target}")


# ---------------------------------------------------------------------------
# GGUF writing (v3, minimal: strings, u32, u64, f32 metadata)

class GgufWriter:
    def __init__(self):
        self.kv: list[tuple[str, int, object]] = []
        self.tensors: list[tuple[str, tuple, int, int]] = []  # name, shape, ggml type, nbytes

    def add_string(self, key: str, value: str):
        self.kv.append((key, 8, value))

    def add_u32(self, key: str, value: int):
        self.kv.append((key, 4, value))

    def add_f32(self, key: str, value: float):
        self.kv.append((key, 6, value))

    def add_u64(self, key: str, value: int):
        self.kv.append((key, 10, value))

    def add_tensor(self, name: str, shape: tuple, ggml_type: int, nbytes: int):
        self.tensors.append((name, shape, ggml_type, nbytes))

    @staticmethod
    def _string(value: str) -> bytes:
        data = value.encode("utf-8")
        return struct.pack("<Q", len(data)) + data

    def header(self) -> tuple[bytes, list[int]]:
        """Returns (header bytes padded to the alignment, data offsets per tensor)."""
        out = bytearray()
        out += GGUF_MAGIC + struct.pack("<IQQ", GGUF_VERSION, len(self.tensors), len(self.kv))
        for key, kind, value in self.kv:
            out += self._string(key) + struct.pack("<I", kind)
            if kind == 8:
                out += self._string(value)
            elif kind == 4:
                out += struct.pack("<I", value)
            elif kind == 6:
                out += struct.pack("<f", value)
            elif kind == 10:
                out += struct.pack("<Q", value)
            else:
                raise SystemExit(f"unsupported metadata kind {kind}")
        offsets = []
        offset = 0
        for name, shape, ggml_type, nbytes in self.tensors:
            # GGUF stores dimensions innermost first (ne[0] = columns).
            dims = list(reversed(shape))
            out += self._string(name) + struct.pack("<I", len(dims))
            for dim in dims:
                out += struct.pack("<Q", dim)
            out += struct.pack("<IQ", ggml_type, offset)
            offsets.append(offset)
            offset += nbytes
            offset += (-offset) % GGUF_ALIGNMENT
        pad = (-len(out)) % GGUF_ALIGNMENT
        out += b"\0" * pad
        return bytes(out), offsets


# ---------------------------------------------------------------------------
# Commands

def dry_run(src: Path) -> int:
    index, _root = load_index(src)
    weight_map = index["weight_map"]
    mapping = build_mapping()
    missing = sorted(set(mapping) - set(weight_map))
    unexpected = sorted(set(weight_map) - set(mapping))
    totals = {"audio": [0, 0], "llm": [0, 0], "drop": [0, 0]}
    print(f"{'hf name':70} {'->':2} {'target':5} {'gufo name':40} {'shape':24} {'kind'}")
    for name in sorted(weight_map):
        if name not in mapping:
            continue
        target, gufo, shape, kind = mapping[name]
        totals[target][0] += 1
        totals[target][1] += elements(shape)
        print(f"{name:70} -> {target:5} {gufo or '-':40} {str(list(shape)):24} {kind}")
    params = sum(elements(m[2]) for n, m in mapping.items() if m[3] != "buffer")
    with_buffer = sum(elements(m[2]) for m in mapping.values())
    print()
    for target, (count, count_elements) in totals.items():
        print(f"{target:5}: {count:4} tensors, {count_elements:>14,} elements")
    print(f"index: {len(weight_map)} tensors in {len(set(weight_map.values()))} shards")
    print(f"parameters (no buffer): {params:,} (index metadata {index.get('metadata', {}).get('total_parameters', '?'):,})")
    print(f"BF16 bytes with buffer: {with_buffer * 2:,} (index metadata {index.get('metadata', {}).get('total_size', '?'):,})")
    ok = True
    if missing:
        ok = False
        print(f"MISSING in index ({len(missing)}):", *missing[:10], sep="\n  ")
    if unexpected:
        ok = False
        print(f"UNEXPECTED in index ({len(unexpected)}):", *unexpected[:10], sep="\n  ")
    meta = index.get("metadata", {})
    if meta.get("total_parameters") not in (None, params):
        ok = False
        print("parameter count differs from the index metadata")
    if meta.get("total_size") not in (None, with_buffer * 2):
        ok = False
        print("byte count differs from the index metadata (not all BF16?)")
    print("dry run:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


def check_tensors(tensors: dict, mapping: dict, target: str) -> list[str]:
    names = []
    for name, (t, _gufo, shape, kind) in mapping.items():
        if t != target:
            continue
        if name not in tensors:
            raise SystemExit(f"tensor missing in the checkpoint: {name}")
        entry = tensors[name][1]
        if tuple(entry["shape"]) != shape:
            raise SystemExit(f"{name}: shape {entry['shape']} differs from the expected {list(shape)}")
        names.append(name)
    return names


def convert_audio(src: Path, out: Path, force: bool) -> None:
    if out.exists() and not force:
        raise SystemExit(f"{out} exists (use --force)")
    tensors = collect(src)
    mapping = build_mapping()
    names = check_tensors(tensors, mapping, "audio")
    writer = GgufWriter()
    writer.add_string("general.architecture", "moss-music")
    writer.add_string("general.name", "MOSS-Music-8B-Instruct audio encoder + adapters")
    writer.add_string("moss-music.source", "OpenMOSS-Team/MOSS-Music-8B-Instruct")
    writer.add_u32("moss-music.audio.mel_bins", AUDIO["mel_bins"])
    writer.add_u32("moss-music.audio.d_model", AUDIO["d_model"])
    writer.add_u32("moss-music.audio.layers", AUDIO["layers"])
    writer.add_u32("moss-music.audio.heads", AUDIO["heads"])
    writer.add_u32("moss-music.audio.ffn", AUDIO["ffn"])
    writer.add_u32("moss-music.audio.conv_channels", AUDIO["conv_channels"])
    writer.add_u32("moss-music.audio.n_window", AUDIO["n_window"])
    writer.add_f32("moss-music.audio.layer_norm_eps", 1.0e-5)
    writer.add_string("moss-music.audio.deepstack_layers", "8,16,24")
    writer.add_u32("moss-music.adapter.hidden", ADAPTER["hidden"])
    writer.add_u32("moss-music.text.hidden", TEXT["hidden"])
    for key, value in TOKENS.items():
        writer.add_u32(f"moss-music.tokens.{key}", value)
    plan = []
    for name in names:
        _t, gufo, shape, kind = mapping[name]
        target_dtype = "F32" if kind == "vector" else "BF16"
        ggml = GGML_F32 if target_dtype == "F32" else GGML_BF16
        nbytes = elements(shape) * DTYPE_BYTES[target_dtype]
        writer.add_tensor(gufo, shape, ggml, nbytes)
        plan.append((name, gufo, target_dtype, nbytes))
    header, offsets = writer.header()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with open(tmp, "wb") as f:
        f.write(header)
        data_start = f.tell()
        for i, ((name, gufo, target_dtype, nbytes), offset) in enumerate(zip(plan, offsets), 1):
            f.seek(data_start + offset)
            shard, entry, start = tensors[name]
            written = 0
            for raw in stream_blocks(shard, entry, start):
                block = convert_block(raw, entry["dtype"], target_dtype)
                f.write(block)
                written += len(block)
            if written != nbytes:
                raise SystemExit(f"{name}: wrote {written} bytes, expected {nbytes}")
            if i % 50 == 0 or i == len(plan):
                print(f"  audio {i}/{len(plan)} tensors", file=sys.stderr, flush=True)
        # Pad the tail to the alignment like the other tensors.
        end = f.tell()
        f.write(b"\0" * ((-end) % GGUF_ALIGNMENT))
    os.replace(tmp, out)
    print(f"wrote {out} ({out.stat().st_size / 2**30:.2f} GiB)")


LLM_SIDE_FILES = ("tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
                  "added_tokens.json", "chat_template.jinja", "generation_config.json")


def convert_llm(src: Path, out_dir: Path, force: bool) -> None:
    """Writes a Qwen3ForCausalLM folder: one BF16 safetensors file plus config/tokenizer."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "model.safetensors"
    if out.exists() and not force:
        raise SystemExit(f"{out} exists (use --force)")
    tensors = collect(src)
    mapping = build_mapping()
    names = check_tensors(tensors, mapping, "llm")
    entries = {}
    offset = 0
    for name in names:
        _t, gufo, shape, _kind = mapping[name]
        size = elements(shape) * DTYPE_BYTES["BF16"]
        entries[gufo] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    header = {"__metadata__": {"format": "pt", "source": "OpenMOSS-Team/MOSS-Music-8B-Instruct language_model",
                               "converter": "gufo tools/moss_music/convert_hf_to_gufo.py"}, **entries}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((-len(header_bytes)) % 8)
    tmp = out.with_suffix(out.suffix + ".part")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for i, name in enumerate(names, 1):
            shard, entry, start = tensors[name]
            for raw in stream_blocks(shard, entry, start):
                f.write(convert_block(raw, entry["dtype"], "BF16"))
            if i % 50 == 0 or i == len(names):
                print(f"  llm {i}/{len(names)} tensors", file=sys.stderr, flush=True)
    os.replace(tmp, out)

    root = src if src.is_dir() else src.parent
    with open(root / "config.json", encoding="utf-8") as f:
        full = json.load(f)
    language = dict(full["language_config"])
    language["architectures"] = ["Qwen3ForCausalLM"]
    language["model_type"] = "qwen3"
    language["torch_dtype"] = "bfloat16"
    language.setdefault("tie_word_embeddings", False)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(language, f, indent=2)
    for side in LLM_SIDE_FILES:
        if (root / side).exists():
            shutil.copyfile(root / side, out_dir / side)
    print(f"wrote {out_dir} ({out.stat().st_size / 2**30:.2f} GiB); next: llama.cpp convert_hf_to_gguf.py "
          f"--outtype bf16, then llama-quantize Q8_0 / Q4_K_M")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", nargs="?", choices=["audio", "llm", "all"], help="what to convert")
    parser.add_argument("src", type=Path, help="checkpoint directory (or its index json for --dry-run)")
    parser.add_argument("out", type=Path, nargs="?", help="output directory")
    parser.add_argument("--stem", default="moss-music-8b")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="list the mapping from the index only")
    args = parser.parse_args()
    if args.dry_run:
        return dry_run(args.src)
    if args.kind is None or args.out is None:
        parser.error("kind and out are required unless --dry-run")
    if args.kind in ("audio", "all"):
        convert_audio(args.src, args.out / f"{args.stem}.audio.gguf", args.force)
    if args.kind in ("llm", "all"):
        convert_llm(args.src, args.out / f"{args.stem}.llm-bf16", args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
