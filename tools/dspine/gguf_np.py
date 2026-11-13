# GGUF-Zugriff fuer das DSpine-Orakel: Metadaten, Tensoren (F32/BF16/Q8_0/Q4_K/Q6_K) als float32, auch zeilenweise.
# KI-unterstuetzt (Claude Code), 09.11.2026.
import struct
import numpy as np

_T = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
BLK = {0: (1, 4), 30: (1, 2), 8: (32, 34), 12: (256, 144), 14: (256, 210)}  # typ -> (Werte, Bytes) je Block


def _rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def _val(f, t):
    if t in _T:
        return _rd(f, _T[t])
    if t == 8:
        return f.read(_rd(f, "Q")).decode("utf-8", "replace")
    if t == 9:
        it = _rd(f, "I"); n = _rd(f, "Q")
        return [_val(f, it) for _ in range(n)]
    raise ValueError(t)


class GGUF:
    def __init__(self, path, align=32):
        self.path = path
        with open(path, "rb") as f:
            assert f.read(4) == b"GGUF"
            _rd(f, "I"); nt = _rd(f, "Q"); nk = _rd(f, "Q")
            self.meta = {}
            for _ in range(nk):
                k = f.read(_rd(f, "Q")).decode(); self.meta[k] = _val(f, _rd(f, "I"))
            self.info = {}
            for _ in range(nt):
                n = f.read(_rd(f, "Q")).decode(); nd = _rd(f, "I")
                dims = [_rd(f, "Q") for _ in range(nd)]
                self.info[n] = (dims, _rd(f, "I"), _rd(f, "Q"))
            self.data = f.tell() + ((-f.tell()) % align)

    def shape(self, name):
        """Torch-Form (umgekehrte GGUF-dims)."""
        return list(reversed(self.info[name][0]))

    def rows(self, name, start, count):
        """Zeilen [start, start+count) eines Tensors (Zeile = ne0 Werte) als float32 [count, ne0]."""
        dims, ty, off = self.info[name]
        ne0 = dims[0]; vals, nbytes = BLK[ty]
        row_bytes = ne0 // vals * nbytes
        with open(self.path, "rb") as f:
            f.seek(self.data + off + start * row_bytes)
            buf = f.read(row_bytes * count)
        return _deq(buf, ty, count * ne0).reshape(count, ne0)

    def tensor(self, name):
        dims = self.info[name][0]
        n_rows = int(np.prod(dims[1:])) if len(dims) > 1 else 1
        return self.rows(name, 0, n_rows).reshape(self.shape(name))

    def take_rows(self, name, ids):
        return np.stack([self.rows(name, int(i), 1)[0] for i in ids])


def _deq(buf, ty, n):
    if ty == 0:
        return np.frombuffer(buf, np.float32).copy()
    if ty == 30:
        return (np.frombuffer(buf, np.uint16).astype(np.uint32) << 16).view(np.float32)
    if ty == 8:
        b = np.frombuffer(buf, dtype=[("d", "<f2"), ("q", "i1", (32,))])
        return (b["q"].astype(np.float32) * b["d"].astype(np.float32)[:, None]).reshape(-1)
    if ty == 12:  # Q4_K
        b = np.frombuffer(buf, dtype=[("d", "<f2"), ("dmin", "<f2"), ("sc", "u1", (12,)), ("qs", "u1", (128,))])
        d = b["d"].astype(np.float32); dm = b["dmin"].astype(np.float32); q = b["sc"].astype(np.int32)
        sc = np.zeros((len(b), 8), np.float32); mn = np.zeros((len(b), 8), np.float32)
        for j in range(8):
            if j < 4:
                sc[:, j] = q[:, j] & 63; mn[:, j] = q[:, j + 4] & 63
            else:
                sc[:, j] = (q[:, j + 4] & 0xF) | ((q[:, j - 4] >> 6) << 4)
                mn[:, j] = (q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4)
        qs = b["qs"].reshape(len(b), 4, 32).astype(np.float32)
        lo = qs.astype(np.int32) & 0xF; hi = qs.astype(np.int32) >> 4
        out = np.empty((len(b), 4, 2, 32), np.float32)
        for c in range(4):
            out[:, c, 0] = (d * sc[:, 2 * c])[:, None] * lo[:, c] - (dm * mn[:, 2 * c])[:, None]
            out[:, c, 1] = (d * sc[:, 2 * c + 1])[:, None] * hi[:, c] - (dm * mn[:, 2 * c + 1])[:, None]
        return out.reshape(-1)
    if ty == 14:  # Q6_K
        b = np.frombuffer(buf, dtype=[("ql", "u1", (128,)), ("qh", "u1", (64,)), ("sc", "i1", (16,)), ("d", "<f2")])
        d = b["d"].astype(np.float32); sc = b["sc"].astype(np.float32)
        ql = b["ql"].astype(np.int32); qh = b["qh"].astype(np.int32)
        out = np.empty((len(b), 2, 4, 32), np.float32)
        for h in range(2):
            l_ = ql[:, 64 * h:64 * h + 64]; q_ = qh[:, 32 * h:32 * h + 32]; s_ = sc[:, 8 * h:8 * h + 8]
            q1 = ((l_[:, :32] & 0xF) | (((q_ >> 0) & 3) << 4)) - 32
            q2 = ((l_[:, 32:] & 0xF) | (((q_ >> 2) & 3) << 4)) - 32
            q3 = ((l_[:, :32] >> 4) | (((q_ >> 4) & 3) << 4)) - 32
            q4 = ((l_[:, 32:] >> 4) | (((q_ >> 6) & 3) << 4)) - 32
            for k, qq in enumerate((q1, q2, q3, q4)):
                # Skalen: is = l/16 -> zwei Haelften je 16 Werte
                s_lo = s_[:, 2 * k]; s_hi = s_[:, 2 * k + 1]
                scale = np.concatenate([np.repeat(s_lo[:, None], 16, 1), np.repeat(s_hi[:, None], 16, 1)], 1)
                out[:, h, k] = d[:, None] * scale * qq
        return out.reshape(-1)
    raise ValueError(f"Typ {ty} nicht unterstuetzt")
