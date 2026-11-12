#!/usr/bin/env python3
"""Regression run of gufo-moss-music over the test pieces: answers, token
hashes and timings per piece, as one JSON. With --check the run is compared
with a stored JSON (same build and GGUF must give identical greedy tokens).

  regression.py --exe build/mossmusic-hip/gufo-moss-music.exe \
      --audio-gguf .../moss-music-8b.audio.gguf --llm-gguf .../moss-music-8b.llm-Q8_0.gguf \
      --audio referenz/testaudio --out regression-q8.json [--check regression-q8.json]

Written with AI assistance (Claude Code), 12.11.2026. Runs with ``python -I``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--audio-gguf", required=True)
    ap.add_argument("--llm-gguf", required=True)
    ap.add_argument("--audio", required=True, help="Ordner mit *.wav (16 kHz mono)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", help="gespeicherte Regressionsdatei zum Vergleich")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--text", default="Please give a detailed musical description of this clip.")
    args = ap.parse_args()

    result = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "exe": args.exe,
        "audio_gguf": os.path.basename(args.audio_gguf),
        "llm_gguf": os.path.basename(args.llm_gguf),
        "llm_gguf_sha256": sha256_file(args.llm_gguf)[:16],
        "max_new": args.max_new,
        "context": args.context,
        "text": args.text,
        "pieces": {},
    }
    files = sorted(f for f in os.listdir(args.audio) if f.lower().endswith(".wav"))
    with tempfile.TemporaryDirectory() as tmp:
        for fname in files:
            name = os.path.splitext(fname)[0]
            prefix = os.path.join(tmp, name)
            cmd = [args.exe, "--audio-gguf", args.audio_gguf, "--llm-gguf", args.llm_gguf,
                   "--audio", os.path.join(args.audio, fname), "--dump", prefix,
                   "--max-new", str(args.max_new), "--context", str(args.context), "--text", args.text]
            run = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if run.returncode != 0:
                print(f"{name}: FEHLER rc={run.returncode}\n{run.stdout[-800:]}\n{run.stderr[-800:]}")
                result["pieces"][name] = {"error": run.returncode}
                continue
            meta = json.load(open(prefix + ".meta.json", encoding="utf-8"))
            ids = np.fromfile(prefix + ".greedy_ids.i32", dtype=np.int32)
            logits = np.fromfile(prefix + ".logits_last.f32", dtype=np.float32)
            piece = {
                "sha256_wav": sha256_file(os.path.join(args.audio, fname))[:16],
                "frames": meta["frames"],
                "audio_tokens": meta["audio_tokens"],
                "prompt_tokens": meta["prompt_tokens"],
                "greedy_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest()[:16],
                "greedy_count": int(ids.size),
                "argmax": int(np.argmax(logits)),
                "top5": [int(i) for i in np.argsort(-logits)[:5]],
                "answer": meta["answer"],
                "encoder_s": meta.get("gpu_encoder_seconds"),
                "prefill_s": meta.get("prefill_seconds"),
                "decode_s": meta.get("decode_seconds"),
            }
            result["pieces"][name] = piece
            print(f"{name}: {piece['audio_tokens']} Audio-Token, {piece['greedy_count']} Token, "
                  f"argmax {piece['argmax']}, {piece['answer'][:90]}…")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print("geschrieben:", args.out)

    if args.check:
        stored = json.load(open(args.check, encoding="utf-8"))
        ok = True
        for name, piece in result["pieces"].items():
            old = stored.get("pieces", {}).get(name)
            if old is None:
                print(f"{name}: nicht in der Vergleichsdatei")
                continue
            same_ids = old.get("greedy_ids_sha256") == piece.get("greedy_ids_sha256")
            same_argmax = old.get("argmax") == piece.get("argmax")
            print(f"{name}: greedy {'gleich' if same_ids else 'VERSCHIEDEN'}, argmax {'gleich' if same_argmax else 'VERSCHIEDEN'}")
            ok = ok and same_ids and same_argmax
        print("REGRESSION:", "OK" if ok else "ABWEICHUNG")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
