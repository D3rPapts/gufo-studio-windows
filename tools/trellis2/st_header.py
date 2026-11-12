#!/usr/bin/env python3
"""Prints the tensor table of a safetensors file grouped by block index.

    python -I st_header.py model.safetensors [--full]

Standard library only, reads only the header (no payload). Block indices in the names
are collapsed to N (with the index range) unless --full is given.
"""
import collections
import json
import re
import struct
import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    full = "--full" in sys.argv
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n == 0 or n > 64 * 1024 * 1024:
            print("BAD HEADER LENGTH", n)
            return 1
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None)
    total = sum(__import__("math").prod(v["shape"]) for v in header.values())
    print(f"header bytes {n}, tensors {len(header)}, params {total:,}, metadata {meta}")
    if full:
        for k, v in header.items():
            print(f"{k:70s} {v['dtype']:5s} {v['shape']}")
        return 0
    groups = collections.OrderedDict()
    for k, v in header.items():
        key = re.sub(r"\.\d+\.", ".N.", k)
        g = groups.setdefault(key, [v["dtype"], v["shape"], 0, set()])
        g[2] += 1
        idx = re.findall(r"\.(\d+)\.", k)
        if idx:
            g[3].add(tuple(int(x) for x in idx))
    for key, (dt, shape, count, idx) in groups.items():
        rng = ""
        if idx:
            s = sorted(idx)
            rng = f" idx {s[0]}..{s[-1]}"
        print(f"{key:70s} {dt:5s} {str(shape):28s} x{count}{rng}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
