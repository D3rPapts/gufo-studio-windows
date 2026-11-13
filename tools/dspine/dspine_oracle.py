# DSpine-Orakel (numpy): eine Draft-Runde wie dspine_mlx (model.py, bridge.py, propose.py RefinementTail.greedy),
# mit BF16-Rundung an denselben Stellen. Ziel-Teile (Einbettung, lm_head) aus dem 27B-GGUF; Kontext-Zustaende
# synthetisch (fester Seed), weil das 27B auf der CPU nicht laeuft – fuer den Paritaetsvergleich mit dem C++-Laeufer
# genuegt das. Schichtweise Laden (wenig RAM). Ausgabe: orakel\eingabe.npz + orakel\ergebnis.npz.
# Aufruf: python -I dspine_ref.py [Q8_0|BF16]   KI-unterstuetzt (Claude Code), 09.11.2026.
import math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_np import GGUF  # noqa: E402

ART = sys.argv[1] if len(sys.argv) > 1 else "Q8_0"
D = GGUF(rf"D:\llm-modelle\tencent\Qwen3.8-27B-DSpine-b16-GGUF\Qwen3.8-27B-DSpine-b16-{ART}.gguf")
TGT = GGUF(r"D:\llm-modelle\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-Q4_K_XL.gguf")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orakel"); os.makedirs(OUT, exist_ok=True)
M = D.meta
H = M["dspine.embedding_length"]; NL = M["dspine.block_count"]; NH = M["dspine.attention.head_count"]
NKV = M["dspine.attention.head_count_kv"]; HD = M["dspine.attention.key_length"]
EPS = M["dspine.attention.layer_norm_rms_epsilon"]; THETA = M["dspine.rope.freq_base"]
T = M["dspine.block_size"]; MASK = M["dspine.mask_token_id"]; C = M["dspine.bridge.code_size"]
R = M["dspine.bridge.rank"]; BEPS = M["dspine.bridge.eps"]; K = M["dspine.second_pass.head_topk"]
V = M["dspine.vocab_size"]; NT = len(M["dspine.target_layers"])
SQC = math.sqrt(C); TINY = 1.1754943508222875e-38


def bf(x):
    """float32 -> naechster BF16-Wert (round-to-nearest-even), wieder als float32."""
    x = np.ascontiguousarray(x, np.float32)
    u = x.view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return r.view(np.float32)


def lin(x, name):
    return bf(x.astype(np.float64) @ D.tensor(name).astype(np.float64).T)


def rms(x, w, eps=EPS):
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True).astype(np.float32)
    return bf(x * (np.float32(1.0) / np.sqrt(ms + np.float32(eps))) * w)


def rope(x, pos):  # x [L, nh, HD], nicht-traditionell (Haelften)
    half = HD // 2
    inv = THETA ** (-np.arange(half, dtype=np.float64) * 2 / HD)
    ang = pos[:, None].astype(np.float64) * inv[None, :]
    c = np.cos(ang).astype(np.float32)[:, None, :]; s = np.sin(ang).astype(np.float32)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return bf(np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1))


# ---------- Eingabe (fest) ----------
rng = np.random.default_rng(7)
S = 8
ctx_hidden = bf(rng.normal(0.0, 2.0, size=(S, NT * H)).astype(np.float32))
anchor = 9707
block = np.array([anchor] + [MASK] * (T - 1), np.int64)
np.savez(os.path.join(OUT, "eingabe.npz"), ctx_hidden=ctx_hidden, block=block, S=S, art=ART)
t0 = time.time()

# ---------- Draft-Vorwaertsrechnung ----------
emb = bf(TGT.take_rows("token_embd.weight", block))  # [T, H]
ctx = rms(bf(ctx_hidden.astype(np.float64) @ D.tensor("fc.weight").astype(np.float64).T), D.tensor("enc.output_norm.weight"))  # [S, H]
pos_ctx = np.arange(S); pos_blk = np.arange(S, S + T)
residual = emb
x_in = rms(emb, D.tensor("blk.0.attn_norm.weight"))
read_last = D.tensor("dspine.bridge.read")  # [C, H]
early = D.tensor("dspine.bridge.early_read")  # [NL-1, C, H]
fwrite = D.tensor("dspine.bridge.feature_write")  # [NL, H, R]
packed_w = np.concatenate([D.tensor("dspine.bridge.gate.weight"), D.tensor("dspine.bridge.predecessor_gate.weight"),
                           D.tensor("dspine.bridge.prev.weight")], axis=0)  # [3R, C]
gate_b = D.tensor("dspine.bridge.gate.bias")
cb_anchor = D.rows("dspine.bridge.codebook", anchor, 1)[0]
stash = None
for i in range(NL):
    p = f"blk.{i}."
    # Attention: Kontext-K/V (fc-Kontext) + Block-K/V, kausal im Block, Kontext ganz sichtbar
    q = lin(x_in, p + "attn_q.weight").reshape(T, NH, HD)
    kb = lin(x_in, p + "attn_k.weight").reshape(T, NKV, HD)
    vb = lin(x_in, p + "attn_v.weight").reshape(T, NKV, HD)
    kc = lin(ctx, p + "attn_k.weight").reshape(S, NKV, HD)
    vc = lin(ctx, p + "attn_v.weight").reshape(S, NKV, HD)
    qn = D.tensor(p + "attn_q_norm.weight"); kn = D.tensor(p + "attn_k_norm.weight")
    q = rope(rms(q, qn), pos_blk); kb = rope(rms(kb, kn), pos_blk); kc = rope(rms(kc, kn), pos_ctx)
    k = np.concatenate([kc, kb], 0); v = np.concatenate([vc, vb], 0)  # [S+T, NKV, HD]
    att = np.empty((T, NH, HD), np.float32)
    mask = np.full((T, S + T), -np.inf, np.float32)
    for t in range(T):
        mask[t, :S + t + 1] = 0.0
    for h in range(NH):
        kv = h // (NH // NKV)
        sc = (q[:, h, :].astype(np.float64) @ k[:, kv, :].astype(np.float64).T) * (1.0 / math.sqrt(HD)) + mask
        sc = sc - sc.max(axis=1, keepdims=True); w = np.exp(sc); w /= w.sum(axis=1, keepdims=True)
        att[:, h, :] = w @ v[:, kv, :].astype(np.float64)
    att = bf(att).reshape(T, NH * HD)
    hsum = bf(residual + lin(att, p + "attn_output.weight"))
    hn = rms(hsum, D.tensor(p + "ffn_norm.weight"))
    g = lin(hn, p + "ffn_gate.weight"); u = lin(hn, p + "ffn_up.weight")
    act = bf((g / (1.0 + np.exp(-g))) * u)
    out = bf(hsum + lin(act, p + "ffn_down.weight"))
    # Code-Bruecke (bridge.py __call__)
    raw = out
    f = raw.astype(np.float32)
    rmsv = np.sqrt(np.maximum(np.mean(f.astype(np.float64) ** 2, axis=-1).astype(np.float32), np.float32(TINY)))  # [T]
    normalized = bf(f / (rmsv[:, None] + BEPS))
    rw = read_last if i == NL - 1 else early[i]
    projected = bf(normalized.astype(np.float64) @ rw.astype(np.float64).T)
    pn = np.sqrt(np.sum(projected.astype(np.float64) ** 2, axis=-1, keepdims=True).astype(np.float32))
    codes = bf(SQC * projected / (pn + BEPS))
    codes[0] = bf(SQC * cb_anchor)
    packed = bf(codes.astype(np.float64) @ packed_w.astype(np.float64).T)  # [T, 3R]
    if i == NL - 1:
        stash = dict(raw=raw.copy(), rms=rmsv.copy(), packed=packed.copy())
    qg, sg, vg = packed[:, :R], packed[:, R:2 * R], packed[:, 2 * R:]
    gate = 1.0 / (1.0 + np.exp(-(qg[1:] + sg[:-1] + gate_b)))
    message = bf(np.concatenate([np.zeros((1, R), np.float32), gate * vg[:-1]], 0))
    delta = bf(message.astype(np.float64) @ fwrite[i].astype(np.float64).T)  # [T, H]
    not_anchor = (np.arange(T) != 0).astype(np.float32)[:, None]
    out_w = bf(f + (rmsv[:, None] * delta) * not_anchor)
    nw = D.tensor(f"blk.{i + 1}.attn_norm.weight") if i < NL - 1 else D.tensor("output_norm.weight")
    ms = np.mean(out_w.astype(np.float64) ** 2, axis=-1, keepdims=True).astype(np.float32)
    y = bf(out_w * (np.float32(1.0) / np.sqrt(ms + np.float32(EPS))))
    x_in = bf(y * nw)
    residual = out_w
    print(f"Schicht {i} fertig ({time.time() - t0:.0f} s)", flush=True)

# ---------- Nachschaerfen (RefinementTail.greedy) ----------
rows = x_in[1:]  # [n, H]
n = rows.shape[0]
logits = np.empty((n, V), np.float32)
CH = 8192
for s0 in range(0, V, CH):
    w = TGT.rows("output.weight", s0, min(CH, V - s0))
    logits[:, s0:s0 + w.shape[0]] = rows.astype(np.float64) @ w.astype(np.float64).T
cand = np.sort(np.argsort(-logits, axis=1, kind="stable")[:, :K], axis=1)  # aufsteigend nach id
norm_w = D.tensor("output_norm.weight")
raw_s = stash["raw"][1:]; rms_s = stash["rms"][1:]; q_s = stash["packed"][1:, :R]
head_rows = np.stack([TGT.take_rows("output.weight", cand[i]) for i in range(n)])  # [n,K,H] (exakt)
head_rows = bf(head_rows)
base = np.sum((raw_s[:, None, :] * norm_w).astype(np.float64) * head_rows, axis=-1)  # [n,K]
write_w = fwrite[-1]  # [H, R]
pv = packed_w[R:]  # [2R, C]


def tg_row(v):
    return bf(bf(bf(TGT.rows("output.weight", int(v), 1)[0]) * norm_w).astype(np.float64) @ write_w.astype(np.float64))


def tsv_row(v):
    return bf(bf(SQC * D.rows("dspine.bridge.codebook", int(v), 1)[0]).astype(np.float64) @ pv.astype(np.float64).T)


sid = np.concatenate([np.full((1, K), anchor), cand[:-1]], 0)
tsv = np.stack([[tsv_row(v) for v in r] for r in sid])  # [n,K,2R]
tg = np.stack([[tg_row(v) for v in r] for r in cand])  # [n,K,R]
m = (1.0 / (1.0 + np.exp(-(q_s[:, None, :] + tsv[..., :R] + gate_b)))) * tsv[..., R:]
s = np.einsum("nkr,njr->nkj", m.astype(np.float64), tg.astype(np.float64))
score = base[:, None, :] + rms_s[:, None, None] * s
idx = np.argmax(score, axis=-1)  # [n, K] (Sender -> bester Empfaenger)
j = idx[0, 0]; drafts = []
for i in range(n):
    drafts.append(int(cand[i, j]))
    if i + 1 < n:
        j = idx[i + 1, j]
np.savez(os.path.join(OUT, "ergebnis.npz"), rows=rows, cand=cand, drafts=np.array(drafts),
         score=score, base=base, top_logits=np.take_along_axis(logits, cand, 1),
         stash_raw=stash["raw"], stash_rms=stash["rms"], stash_packed=stash["packed"])
print("Draft-Tokens:", drafts)
print(f"fertig in {time.time() - t0:.0f} s -> {OUT}")
