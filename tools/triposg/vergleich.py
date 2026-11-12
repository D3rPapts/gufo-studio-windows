#!/usr/bin/env python3
"""Vergleicht gufo-Ausgaben mit den PyTorch-Referenz-Dumps (numpy .npy).

    python -P -E vergleich.py REFERENZ.npy UNSER.npy [--rows]       # ein Paar
    python -P -E vergleich.py --dumps REFERENZORDNER --gufo UNSERORDNER [--rows]
        vergleicht alle gleichnamigen .npy-Dateien (z. B. dumps\\tasse gegen gufo\\tasse)

Ausgabe je Paar: Shape, rel. RMS (||a-b|| / ||b||, b = Referenz), max |Δ|, mittlere
Kosinus-Ähnlichkeit je Zeile, schlechteste Zeile. Toleranzen (PLAN.md Abschnitt 4):
preproc_224 max |Δ| <= 1/255/0.224 ≈ 0.0175; dino_tokens rel. RMS <= 1e-2 (BF16) bzw.
<= 1e-4 (FP32-Pfad). KI-unterstuetzt (Claude Code), 12.11.2026.
"""
import argparse
import os
import sys

import numpy as np

TOLERANCES = {
    "preproc_224": ("maxabs", 1 / 255 / 0.224 + 1e-6),
    "dino_tokens": ("rel_rms", 1e-2),
    "dit_block00": ("rel_rms", 2e-2),
    "dit_block20": ("rel_rms", 2e-2),
    "dit_v_step00_cond": ("rel_rms", 2e-2),
    "latents_final": ("rel_rms", 5e-2),
    "vae_kv": ("rel_rms", 2e-2),
}


def compare(ref, got, name, rows):
    if ref.shape != got.shape:
        print(f"{name}: SHAPE {got.shape} != Referenz {ref.shape}")
        return False
    ref = ref.astype(np.float64).reshape(ref.shape[0], -1) if ref.ndim > 1 else ref.astype(np.float64)[None]
    got = got.astype(np.float64).reshape(got.shape[0], -1) if got.ndim > 1 else got.astype(np.float64)[None]
    diff = got - ref
    rel_rms = np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-30)
    max_abs = float(np.abs(diff).max())
    num = (ref * got).sum(axis=1)
    den = np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1) + 1e-30
    cos = num / den
    row_rel = np.linalg.norm(diff, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-30)
    worst = int(np.argmax(row_rel))
    kind, tol = TOLERANCES.get(name, (None, None))
    value = rel_rms if kind == "rel_rms" else max_abs
    verdict = "" if kind is None else ("  ok" if value <= tol else f"  ZU GROSS (> {tol:g})")
    print(f"{name}: shape {tuple(ref.shape)}  rel RMS {rel_rms:.3e}  max|d| {max_abs:.3e}  "
          f"cos mean {cos.mean():.6f} min {cos.min():.6f}  schlechteste Zeile {worst} (rel {row_rel[worst]:.3e}){verdict}")
    if rows:
        order = np.argsort(-row_rel)[:5]
        for r in order:
            print(f"    Zeile {r}: rel {row_rel[r]:.3e}  cos {cos[r]:.5f}  ref[:4] {ref[r, :4]}  got[:4] {got[r, :4]}")
    return kind is None or value <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", nargs="?")
    ap.add_argument("got", nargs="?")
    ap.add_argument("--dumps", help="Referenzordner (dumps\\<bild>)")
    ap.add_argument("--gufo", help="unser Ordner mit gleichnamigen .npy")
    ap.add_argument("--rows", action="store_true", help="die 5 schlechtesten Zeilen zeigen")
    args = ap.parse_args()
    ok = True
    if args.dumps and args.gufo:
        names = sorted(f for f in os.listdir(args.gufo) if f.endswith(".npy") and os.path.isfile(os.path.join(args.dumps, f)))
        if not names:
            print("keine gemeinsamen .npy-Dateien")
            return 2
        for f in names:
            ref = np.load(os.path.join(args.dumps, f))
            got = np.load(os.path.join(args.gufo, f))
            ok &= compare(ref, got, f[:-4], args.rows)
    elif args.ref and args.got:
        name = os.path.basename(args.got)[:-4] if args.got.endswith(".npy") else args.got
        ok = compare(np.load(args.ref), np.load(args.got), name, args.rows)
    else:
        ap.error("REFERENZ UNSER oder --dumps/--gufo angeben")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
