# DSpine (tencent/Qwen3.8-27B-DSpine-b16, safetensors BF16) -> GGUF v3, Architektur "dspine", ohne gguf-Paket.
# Tensor-Namen nach gufos DFlash-GGUF-Schema (blk.N.attn_q, fc.weight, enc.output_norm …) + dspine.bridge.*.
# Streamend: je Tensor einzeln, Q8_0 zeilenweise in Stuecken; bricht ab, wenn freier RAM < 2 GB.
# Aufruf: python -I konvertieren.py <bf16|q8_0>   KI-unterstuetzt (Claude Code), 09.11.2026.
import ctypes, json, os, struct, sys
import numpy as np

SRC = r"D:\llm-modelle\tencent\Qwen3.8-27B-DSpine-b16"
DST = r"D:\llm-modelle\tencent\Qwen3.8-27B-DSpine-b16-GGUF"
MODE = (sys.argv[1] if len(sys.argv) > 1 else "q8_0").lower()
assert MODE in ("bf16", "q8_0")
ALIGN = 32
F32, Q8_0, BF16 = 0, 8, 30


class MEM(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("a", ctypes.c_ulong), ("b", ctypes.c_ulonglong),
                ("avail", ctypes.c_ulonglong)] + [(f"x{i}", ctypes.c_ulonglong) for i in range(5)]


def frei_gb():
    m = MEM(); m.dwLength = ctypes.sizeof(MEM); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.avail / 2**30


def pruefe_ram():
    if frei_gb() < 2.0:
        raise SystemExit(f"Abbruch: nur {frei_gb():.1f} GB frei")


cfg = json.load(open(os.path.join(SRC, "config.json"), encoding="utf-8"))
dc = cfg["dflash_config"]; br = dc["selected_prefix_code_bridge"]


# --- Rohzugriff auf safetensors (BF16 als uint16) ---
with open(os.path.join(SRC, "model.safetensors"), "rb") as fh:
    hlen = struct.unpack("<Q", fh.read(8))[0]
    HDR = json.loads(fh.read(hlen)); BASE = 8 + hlen


def roh(name):
    e = HDR[name]; a, b = e["data_offsets"]
    with open(os.path.join(SRC, "model.safetensors"), "rb") as fh:
        fh.seek(BASE + a); buf = fh.read(b - a)
    dt = {"BF16": np.uint16, "I64": np.int64, "F32": np.float32}[e["dtype"]]
    return np.frombuffer(buf, dtype=dt).reshape(e["shape"]), e["dtype"]


def bf16_zu_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


# --- Namen ---
MAP = {"fc.weight": "fc.weight", "hidden_norm.weight": "enc.output_norm.weight", "norm.weight": "output_norm.weight",
       "selected_prefix_bridge.codebook": "dspine.bridge.codebook",
       "selected_prefix_bridge.read_weight": "dspine.bridge.read",
       "selected_prefix_bridge.early_read_weights": "dspine.bridge.early_read",
       "selected_prefix_bridge.feature_write_weights": "dspine.bridge.feature_write",
       "selected_prefix_bridge.gate.weight": "dspine.bridge.gate.weight",
       "selected_prefix_bridge.gate.bias": "dspine.bridge.gate.bias",
       "selected_prefix_bridge.prev.weight": "dspine.bridge.prev.weight",
       "selected_prefix_bridge.predecessor_gate.weight": "dspine.bridge.predecessor_gate.weight",
       "selected_prefix_bridge.out.weight": "dspine.bridge.out.weight"}
LAYER = {"input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight",
         "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
         "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
         "self_attn.q_norm.weight": "attn_q_norm.weight", "self_attn.k_norm.weight": "attn_k_norm.weight",
         "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
         "mlp.down_proj.weight": "ffn_down.weight"}


def ziel(name):
    if name in MAP:
        return MAP[name]
    if name.startswith("layers."):
        _, i, rest = name.split(".", 2)
        return f"blk.{i}.{LAYER[rest]}"
    return None


versions = {}
plan = []  # (gguf_name, src_name, gguf_type, gguf_dims, nbytes)
for name, e in HDR.items():
    if name == "__metadata__":
        continue
    if e["dtype"] == "I64":
        versions[name] = int(roh(name)[0].reshape(-1)[0])
        continue
    g = ziel(name)
    if g is None:
        raise SystemExit(f"unbekannter Tensor {name}")
    shape = e["shape"]; dims = list(reversed(shape)); n = int(np.prod(shape))
    is_vec = len(shape) == 1
    if is_vec:
        ty = F32; nb = n * 4
    elif MODE == "q8_0" and dims[0] % 32 == 0 and not g.startswith("dspine.bridge."):
        # Wie dspine_mlx.quantize_draft: die Code-Bruecke bleibt BF16 (Nachschaerf-Schritt haengt an ihrer Rundung).
        ty = Q8_0; nb = n // 32 * 34
    else:
        ty = BF16; nb = n * 2
    plan.append((g, name, ty, dims, nb))
plan.sort(key=lambda p: p[0])

meta = [
    ("general.architecture", 8, "dspine"), ("general.name", 8, "Qwen3.8-27B-DSpine-b16"),
    ("general.organization", 8, "tencent"), ("general.license", 8, "apache-2.0"),
    ("general.source.url", 8, "https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16"),
    ("general.file_type", 4, 7 if MODE == "q8_0" else 32),
    ("dspine.block_count", 4, cfg["num_hidden_layers"]), ("dspine.context_length", 4, cfg["max_position_embeddings"]),
    ("dspine.embedding_length", 4, cfg["hidden_size"]), ("dspine.feed_forward_length", 4, cfg["intermediate_size"]),
    ("dspine.vocab_size", 4, cfg["vocab_size"]),
    ("dspine.attention.head_count", 4, cfg["num_attention_heads"]),
    ("dspine.attention.head_count_kv", 4, cfg["num_key_value_heads"]),
    ("dspine.attention.key_length", 4, cfg["head_dim"]), ("dspine.attention.value_length", 4, cfg["head_dim"]),
    ("dspine.attention.layer_norm_rms_epsilon", 6, float(cfg["rms_norm_eps"])),
    ("dspine.attention.causal", 7, bool(dc["draft_block_causal"])),
    ("dspine.attention.sliding_window", 4, cfg["sliding_window"]),
    ("dspine.rope.freq_base", 6, float(cfg["rope_theta"])),
    ("dspine.block_size", 4, cfg["block_size"]), ("dspine.mask_token_id", 4, dc["mask_token_id"]),
    ("dspine.target_layers", 9, (4, dc["target_layer_ids"])),
    ("dspine.target_layer_count", 4, cfg["num_target_layers"]),
    ("dspine.bridge.code_size", 4, br["code_size"]), ("dspine.bridge.rank", 4, br["rank"]),
    ("dspine.bridge.eps", 6, float(br["eps"])), ("dspine.bridge.kappa", 6, float(br["kappa"])),
    ("dspine.bridge.feature_layers", 9, (4, dc["selected_prefix_feature_layers"])),
    ("dspine.bridge.write_mode", 8, dc["selected_prefix_feature_write_mode"]),
    ("dspine.bridge.terminal", 7, bool(dc["selected_prefix_terminal"])),
    ("dspine.decode_mode", 8, dc["selected_prefix_decode_mode"]),
    ("dspine.second_pass.mode", 8, cfg["code_bridge_second_pass"]["mode"]),
    ("dspine.second_pass.first_exact", 4, cfg["code_bridge_second_pass"]["first_exact"]),
    ("dspine.second_pass.head_topk", 4, cfg["code_bridge_second_pass"]["head_topk"]),
    ("dspine.last_write_refine.kind", 8, dc["selected_prefix_last_write_refine"]["kind"]),
    ("dspine.last_write_refine.head_topk", 4, dc["selected_prefix_last_write_refine"]["head_topk"]),
    ("dspine.last_write_refine.weight", 6, float(dc["selected_prefix_last_write_refine"]["weight"])),
] + [(f"dspine.version.{k.replace('selected_prefix_bridge.', '')}", 4, v) for k, v in sorted(versions.items())]


def w_str(f, s):
    b = s.encode("utf-8"); f.write(struct.pack("<Q", len(b))); f.write(b)


def w_val(f, t, v):
    if t == 4: f.write(struct.pack("<I", v))
    elif t == 6: f.write(struct.pack("<f", v))
    elif t == 7: f.write(struct.pack("<?", v))
    elif t == 8: w_str(f, v)
    elif t == 9:
        it, arr = v; f.write(struct.pack("<IQ", it, len(arr)))
        for x in arr: w_val(f, it, x)


def q8_0(rows_f32):
    r = rows_f32.reshape(-1, 32)
    amax = np.abs(r).max(axis=1)
    d = (amax / 127.0).astype(np.float32)
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0).astype(np.float32)
    q = np.clip(np.rint(r * inv[:, None]), -127, 127).astype(np.int8)
    out = np.empty(r.shape[0], dtype=[("d", "<f2"), ("q", "i1", (32,))])
    out["d"] = d.astype(np.float16); out["q"] = q
    return out.tobytes()


os.makedirs(DST, exist_ok=True)
out = os.path.join(DST, f"Qwen3.8-27B-DSpine-b16-{'Q8_0' if MODE == 'q8_0' else 'BF16'}.gguf")
tmp = out + ".part"
with open(tmp, "wb") as f:
    f.write(b"GGUF"); f.write(struct.pack("<IQQ", 3, len(plan), len(meta)))
    for k, t, v in meta:
        w_str(f, k); f.write(struct.pack("<I", t)); w_val(f, t, v)
    off = 0
    for g, _, ty, dims, nb in plan:
        w_str(f, g); f.write(struct.pack("<I", len(dims)))
        for d in dims: f.write(struct.pack("<Q", d))
        f.write(struct.pack("<IQ", ty, off))
        off += (nb + ALIGN - 1) // ALIGN * ALIGN
    f.write(b"\0" * ((-f.tell()) % ALIGN))
    for g, name, ty, dims, nb in plan:
        pruefe_ram()
        a, _ = roh(name)
        start = f.tell()
        if ty == F32:
            f.write(bf16_zu_f32(a.reshape(-1)).astype(np.float32).tobytes())
        elif ty == BF16:
            f.write(np.ascontiguousarray(a).tobytes())
        else:
            flat = a.reshape(-1, dims[0])
            step = max(1, (64 << 20) // (dims[0] * 4))
            for i in range(0, flat.shape[0], step):
                f.write(q8_0(bf16_zu_f32(flat[i:i + step])))
        assert f.tell() - start == nb, (g, f.tell() - start, nb)
        f.write(b"\0" * ((-f.tell()) % ALIGN))
        del a
        print(f"{g:42} {str(dims):24} {'F32' if ty == F32 else 'BF16' if ty == BF16 else 'Q8_0'}", flush=True)
os.replace(tmp, out)
print("fertig:", out, round(os.path.getsize(out) / 2**30, 2), "GiB | Versionen:", versions)
