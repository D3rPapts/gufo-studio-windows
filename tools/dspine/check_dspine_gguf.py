# Rueckpruefung: liest Tensoren aus den DSpine-GGUFs (Q8_0/BF16), dequantisiert und vergleicht mit safetensors.
# KI-unterstuetzt (Claude Code), 09.11.2026.
import json, os, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from gguf_read_tensor import lesen_tensor  # noqa: E402

SRC = r"D:\llm-modelle\tencent\Qwen3.8-27B-DSpine-b16\model.safetensors"
with open(SRC, "rb") as fh:
    hl = struct.unpack("<Q", fh.read(8))[0]; H = json.loads(fh.read(hl)); B = 8 + hl


def src(name):
    a, b = H[name]["data_offsets"]
    with open(SRC, "rb") as fh:
        fh.seek(B + a); u = np.frombuffer(fh.read(b - a), np.uint16)
    return (u.astype(np.uint32) << 16).view(np.float32).reshape(H[name]["shape"])


PAARE = [("fc.weight", "fc.weight"), ("layers.3.mlp.down_proj.weight", "blk.3.ffn_down.weight"),
         ("selected_prefix_bridge.codebook", "dspine.bridge.codebook"),
         ("selected_prefix_bridge.feature_write_weights", "dspine.bridge.feature_write"),
         ("norm.weight", "output_norm.weight"), ("selected_prefix_bridge.gate.bias", "dspine.bridge.gate.bias")]
for art in ("BF16", "Q8_0"):
    p = rf"D:\llm-modelle\tencent\Qwen3.8-27B-DSpine-b16-GGUF\Qwen3.8-27B-DSpine-b16-{art}.gguf"
    for s, g in PAARE:
        a = src(s); b = lesen_tensor(p, g).reshape(a.shape)
        rel = float(np.abs(a - b).max() / (np.abs(a).max() + 1e-12))
        print(f"{art:5} {g:38} max rel. Abweichung {rel:.2e} {'OK' if (rel == 0 if art == 'BF16' or a.ndim == 1 else rel < 0.01) else 'FEHLER'}")
