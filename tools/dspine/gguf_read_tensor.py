# Liest einen Tensor aus einer GGUF-Datei (F32, BF16, Q8_0) als float32. KI-unterstuetzt (Claude Code).
import struct
import numpy as np

T = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def _s(f):
    return f.read(_rd(f, "Q"))


def _skip(f, t):
    if t in T:
        f.read(struct.calcsize("<" + T[t]))
    elif t == 8:
        _s(f)
    elif t == 9:
        it = _rd(f, "I"); n = _rd(f, "Q")
        for _ in range(n):
            _skip(f, it)


def lesen_tensor(path, name, align=32):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        _rd(f, "I"); nt = _rd(f, "Q"); nk = _rd(f, "Q")
        for _ in range(nk):
            _s(f); _skip(f, _rd(f, "I"))
        info = {}
        for _ in range(nt):
            n = _s(f).decode(); nd = _rd(f, "I"); dims = [_rd(f, "Q") for _ in range(nd)]
            info[n] = (dims, _rd(f, "I"), _rd(f, "Q"))
        data = f.tell() + ((-f.tell()) % align)
        dims, ty, off = info[name]; n = int(np.prod(dims))
        f.seek(data + off)
        if ty == 0:
            return np.frombuffer(f.read(n * 4), np.float32).copy()
        if ty == 30:
            u = np.frombuffer(f.read(n * 2), np.uint16)
            return (u.astype(np.uint32) << 16).view(np.float32)
        if ty == 8:
            blk = np.frombuffer(f.read(n // 32 * 34), dtype=[("d", "<f2"), ("q", "i1", (32,))])
            return (blk["q"].astype(np.float32) * blk["d"].astype(np.float32)[:, None]).reshape(-1)
        raise ValueError(ty)
