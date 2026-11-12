#!/usr/bin/env python3
"""Compare gufo-moss-music dumps with the Regie reference dumps (PLAN.md D1-D9).

  compare_reference.py --gufo D:/.../dumps/akt1_00002_16k --ref D:/.../referenz/dumps/akt1_00002

gufo side (PREFIX.*): mel.f32 [128,T], frontend.f32 [N,1280], enc8/16/24.f32,
final.f32, adapter.f32, merger0..2.f32 [N,4096], input_ids.i32, logits_last.f32,
greedy_ids.i32, hs0..hs3.f32, hs_final.f32 [L,4096].
Reference side (dumps/<name>/): d1_mel.npy, chunk_lengths.npy, d2_stem.npy
[chunks,50,1280] (padded), d3a_enc8 / d3b_enc16 / d3c_enc24.npy, d4_enc_final.npy,
d5_adapter.npy, d5_merger0..2.npy, d6_input_ids.npy, d7_hs0..3.npy, d7_hs36.npy,
d8_logits_last.npy, d9_greedy_ids.npy, meta.json.

Written with AI assistance (Claude Code), 12.11.2026. Runs with ``python -I``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

D_MODEL = 1280
HIDDEN = 4096
MEL_BINS = 128


def load_f32(path: str, shape=None):
    if not os.path.exists(path):
        return None
    a = np.fromfile(path, dtype=np.float32)
    return a.reshape(shape) if shape else a


def load_i32(path: str):
    if not os.path.exists(path):
        return None
    return np.fromfile(path, dtype=np.int32)


def load_npy(path: str):
    return np.load(path) if os.path.exists(path) else None


def compact(padded: np.ndarray, chunk_lengths: np.ndarray) -> np.ndarray:
    """[chunks, 50, D] padded chunk rows -> [N, D] valid rows."""
    def conv3(n: int) -> int:
        for _ in range(3):
            n = (n - 1) // 2 + 1
        return n
    rows = [padded[i, :conv3(int(length))] for i, length in enumerate(chunk_lengths)]
    return np.concatenate(rows, axis=0)


def report(name: str, ours, ref, atol: float, rel_tol: float) -> bool:
    if ours is None or ref is None:
        print(f"{name:12} fehlt ({'gufo' if ours is None else 'referenz'})")
        return True
    ours = np.asarray(ours, dtype=np.float32).reshape(-1)
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    if ours.size != ref.size:
        print(f"{name:12} FORM {ours.size} vs {ref.size}")
        return False
    diff = np.abs(ours - ref)
    scale = float(np.abs(ref).max()) + 1e-30
    cos = float(np.dot(ours, ref) / (np.linalg.norm(ours) * np.linalg.norm(ref) + 1e-30))
    exact = float(np.mean(diff == 0))
    ok = diff.max() <= atol + rel_tol * scale and cos >= 0.9999
    print(f"{name:12} n={ours.size:9d} max|d|={diff.max():.3e} mean|d|={diff.mean():.3e} "
          f"max|ref|={scale:.3e} rel={diff.max() / scale:.2e} cos={cos:.7f} exakt={exact * 100:5.1f}% "
          f"-> {'OK' if ok else 'ABWEICHUNG'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gufo", required=True, help="PREFIX der gufo-Dumps")
    ap.add_argument("--ref", required=True, help="Ordner dumps/<name> der Regie")
    ap.add_argument("--rel", type=float, default=2e-2, help="zulaessige Abweichung relativ zu max|ref| (bf16 ~ 2 ulp)")
    args = ap.parse_args()
    g, r = args.gufo, args.ref
    meta = json.load(open(os.path.join(r, "meta.json"), encoding="utf-8")) if os.path.exists(os.path.join(r, "meta.json")) else {}
    chunk_lengths = load_npy(os.path.join(r, "chunk_lengths.npy"))
    frames = int(meta.get("mel_frames", 0))
    tokens = int(meta.get("audio_tokens", 0))
    print(f"referenz: {meta.get('file')} frames={frames} chunks={list(chunk_lengths) if chunk_lengths is not None else '?'} tokens={tokens}")
    ok = True

    mel_ref = load_npy(os.path.join(r, "d1_mel.npy"))
    mel_g = load_f32(g + ".mel.f32")
    if mel_ref is not None and mel_g is not None and mel_g.size == mel_ref.size:
        mel_g = mel_g.reshape(mel_ref.shape)
    ok &= report("D1 mel", mel_g, mel_ref, 1e-3, args.rel)

    if chunk_lengths is not None:
        stem = load_npy(os.path.join(r, "d2_stem.npy"))
        ok &= report("D2 stem", load_f32(g + ".frontend.f32"), compact(stem, chunk_lengths) if stem is not None else None, 0.0, args.rel)
        for tag, name in (("d3a_enc8", "enc8"), ("d3b_enc16", "enc16"), ("d3c_enc24", "enc24")):
            ref = load_npy(os.path.join(r, tag + ".npy"))
            ok &= report("D3 " + name, load_f32(g + f".{name}.f32"), compact(ref, chunk_lengths) if ref is not None else None, 0.0, args.rel)
    ok &= report("D4 final", load_f32(g + ".final.f32"), load_npy(os.path.join(r, "d4_enc_final.npy")), 0.0, args.rel)
    ok &= report("D5 adapter", load_f32(g + ".adapter.f32"), load_npy(os.path.join(r, "d5_adapter.npy")), 0.0, args.rel)
    for k in range(3):
        ok &= report(f"D5 merger{k}", load_f32(g + f".merger{k}.f32"), load_npy(os.path.join(r, f"d5_merger{k}.npy")), 0.0, args.rel)

    ids_g = load_i32(g + ".input_ids.i32")
    ids_r = load_npy(os.path.join(r, "d6_input_ids.npy"))
    if ids_g is not None and ids_r is not None:
        same = ids_g.size == ids_r.size and bool(np.all(ids_g == ids_r.astype(np.int64)))
        print(f"{'D6 ids':12} gufo {ids_g.size} / referenz {ids_r.size} Token -> {'IDENTISCH' if same else 'VERSCHIEDEN'}")
        if not same:
            ok = False
            n = min(ids_g.size, ids_r.size)
            first = int(np.argmax(ids_g[:n] != ids_r[:n])) if n else 0
            print(f"   erste Abweichung bei {first}: gufo {ids_g[first:first + 8].tolist()} ref {ids_r[first:first + 8].tolist()}")
    for k, tag in ((0, "d7_hs0"), (1, "d7_hs1"), (2, "d7_hs2"), (3, "d7_hs3")):
        ok &= report(f"D7 hs{k}", load_f32(g + f".hs{k}.f32"), load_npy(os.path.join(r, tag + ".npy")), 0.0, args.rel)
    ok &= report("D7 hs_final", load_f32(g + ".hs_final.f32"), load_npy(os.path.join(r, "d7_hs36.npy")), 0.0, args.rel)

    lg = load_f32(g + ".logits_last.f32")
    lr = load_npy(os.path.join(r, "d8_logits_last.npy"))
    if lg is not None and lr is not None:
        report("D8 logits", lg, lr, 0.0, args.rel)
        tg, tr = np.argsort(-lg)[:10], np.argsort(-lr)[:10]
        print(f"   argmax gufo {int(tg[0])} ref {int(tr[0])} {'GLEICH' if tg[0] == tr[0] else 'VERSCHIEDEN'}; "
              f"Top-10 gemeinsam {len(set(tg.tolist()) & set(tr.tolist()))}/10")
        ok &= bool(tg[0] == tr[0])
    gg = load_i32(g + ".greedy_ids.i32")
    gr = load_npy(os.path.join(r, "d9_greedy_ids.npy"))
    if gg is not None and gr is not None:
        n = min(gg.size, gr.size)
        agree = 0
        while agree < n and gg[agree] == gr[agree]:
            agree += 1
        print(f"{'D9 greedy':12} gufo {gg.size} / referenz {gr.size} Token, identisch bis Token {agree}")
        if meta.get("greedy_text"):
            print("   referenz:", meta["greedy_text"][:200].replace("\n", " "))
    print("GESAMT:", "OK" if ok else "ABWEICHUNGEN")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
