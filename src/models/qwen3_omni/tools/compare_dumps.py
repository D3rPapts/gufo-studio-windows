#!/usr/bin/env python3
"""Compares two gufo-omni --dump outputs (e.g. the CPU reference and HIP).

    python compare_dumps.py REFERENCE_PREFIX CANDIDATE_PREFIX

Prints, for the audio embeddings and the logits after the prompt, the cosine
similarity, relative L2 error and largest absolute difference; for the logits
also the KL divergence of the softmax distributions and the top-k agreement;
and the length of the common prefix of the generated token ids.
"""

import sys

import numpy as np


def load(prefix, name):
    return np.fromfile(f"{prefix}.{name}.f32", dtype=np.float32)


def tokens(prefix):
    try:
        with open(f"{prefix}.tokens.txt", encoding="utf-8") as handle:
            return [int(line) for line in handle if line.strip()]
    except FileNotFoundError:
        return []


def report(label, reference, candidate):
    if reference.shape != candidate.shape:
        print(f"{label}: shape {reference.shape} vs {candidate.shape}")
        return
    a = reference.astype(np.float64)
    b = candidate.astype(np.float64)
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    relative = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    worst = float(np.max(np.abs(a - b)))
    print(f"{label}: cosine {cosine:.6f}  rel-L2 {relative:.3e}  "
          f"max-abs {worst:.3e}")


def softmax(x):
    x = x.astype(np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    reference, candidate = sys.argv[1], sys.argv[2]
    report("audio embeddings", load(reference, "audio"),
           load(candidate, "audio"))
    ref_logits = load(reference, "logits")
    cand_logits = load(candidate, "logits")
    report("logits", ref_logits, cand_logits)
    p = softmax(ref_logits)
    q = softmax(cand_logits)
    kl = float(np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30))))
    print(f"logits: KL(ref||cand) {kl:.3e}")
    for k in (1, 5, 10):
        top_ref = set(np.argsort(-ref_logits)[:k])
        top_cand = set(np.argsort(-cand_logits)[:k])
        print(f"logits: top-{k} overlap {len(top_ref & top_cand)}/{k}")
    ref_tokens = tokens(reference)
    cand_tokens = tokens(candidate)
    common = 0
    for a, b in zip(ref_tokens, cand_tokens):
        if a != b:
            break
        common += 1
    print(f"generated: {len(ref_tokens)} vs {len(cand_tokens)} tokens, "
          f"identical prefix {common}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
