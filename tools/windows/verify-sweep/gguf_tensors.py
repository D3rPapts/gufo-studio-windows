"""List the verification projections of a Qwen GGUF for gufo-verify-sweep.

    python -I gguf_tensors.py <model.gguf>

Prints one line per distinct (type, m, k) projection in the transformer
layers, with the file offset of one example tensor and how many layers use
it: "<type-id> <m> <k> <offset> <count> <example-name>". Gate/up pairs of one
format stored back to back are listed as the joined (2 * m) x k matrix the
decode multiplies. Only types gufo-verify-sweep measures are listed.
"""
import collections
import struct
import sys

BLOCK_BYTES = {12: 144, 13: 176, 14: 210, 23: 136}  # Q4_K Q5_K Q6_K IQ4_XS


def read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def read_str(f):
    return f.read(read(f, "Q")).decode("utf-8", "replace")


def read_value(f, kind):
    scalars = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
               10: "Q", 11: "q", 12: "d"}
    if kind in scalars:
        return read(f, scalars[kind])
    if kind == 8:
        return read_str(f)
    if kind == 9:
        inner = read(f, "I")
        for _ in range(read(f, "Q")):
            read_value(f, inner)
    return None


def tensors(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit("not a GGUF file")
        read(f, "I")
        n_tensors, n_kv = read(f, "Q"), read(f, "Q")
        alignment = 32
        for _ in range(n_kv):
            key = read_str(f)
            value = read_value(f, read(f, "I"))
            if key == "general.alignment":
                alignment = value
        infos = []
        for _ in range(n_tensors):
            name = read_str(f)
            dims = [read(f, "Q") for _ in range(read(f, "I"))]
            kind = read(f, "I")
            infos.append((name, kind, dims, read(f, "Q")))
        start = (f.tell() + alignment - 1) // alignment * alignment
    return [(n, t, d, start + o) for n, t, d, o in infos]


def main():
    found = collections.OrderedDict()
    layers = collections.defaultdict(dict)
    for name, kind, dims, offset in tensors(sys.argv[1]):
        parts = name.split(".")
        if parts[0] != "blk" or len(dims) != 2 or kind not in BLOCK_BYTES:
            continue
        k, m = dims
        layers[parts[1]][parts[2]] = (kind, m, k, offset, name)
    for layer in layers.values():
        gate, up = layer.get("ffn_gate"), layer.get("ffn_up")
        packed = False
        if gate and up and gate[0] == up[0] and gate[1:3] == up[1:3]:
            size = gate[1] * (gate[2] // 256) * BLOCK_BYTES[gate[0]]
            packed = gate[3] + size == up[3]
        for role, (kind, m, k, offset, name) in layer.items():
            if packed and role in ("ffn_gate", "ffn_up"):
                continue  # multiplied as the joined matrix below
            entry = found.setdefault((kind, m, k), [offset, 0, name])
            entry[1] += 1
        if packed:
            key = (gate[0], 2 * gate[1], gate[2])
            entry = found.setdefault(key, [gate[3], 0, "packed ffn_gate+ffn_up"])
            entry[1] += 1
    for (kind, m, k), (offset, count, name) in found.items():
        print(kind, m, k, offset, count, name)


if __name__ == "__main__":
    main()
