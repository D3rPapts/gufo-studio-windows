#!/usr/bin/env python3
"""Backend-Ersatz fuer TRELLIS.2 auf Maschinen ohne Triton/flash-attn/spconv (ROCm-Windows,
oder CPU): reine torch-Implementierung der Submanifold-Sparse-Conv (Nachbartabelle +
Gather-GEMM) und der sparsen Volltext-Attention (Batch 1 -> dichte SDPA).

Einsatz auf der .56 (Regie) vor dem Import der Pipeline:

    import os
    os.environ["ATTN_BACKEND"] = "sdpa"            # dichte Attention: torch SDPA
    os.environ["SPARSE_CONV_BACKEND"] = "none"     # SparseTensor ohne spconv/torchsparse
    os.environ["SPARSE_ATTN_BACKEND"] = "flash_attn"   # Platzhalter, wird unten ersetzt
    import sparse_backend_cpu                       # DIESE Datei, patcht trellis2.modules.sparse
    sparse_backend_cpu.install()
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

Das Modul ist NICHT getestet (hier gibt es kein torch); es ist das Geruest fuer die
Referenz-Dumps und dokumentiert die Tap-Reihenfolge, die der gufo-Konverter annimmt:
Gewicht [Co, kd, kh, kw, Ci] (FlexGEMM-Layout), Tap t = kd*9 + kh*3 + kw, Nachbar bei
coord + (kd-1, kh-1, kw-1)  (Kreuzkorrelation wie torch Conv3d, Achsen x,y,z = D,H,W).
Sollte flex_gemm die Taps anders ordnen, zeigt es der Dump `conv_probe` (Plan Abschnitt 7).
KI-unterstuetzt (Claude Code), 12.11.2026.
"""
import torch


def neighbour_table(coords: torch.Tensor) -> torch.Tensor:
    """coords [N,4] (batch, x, y, z) int -> [N,27] Zeilenindex des Nachbarn oder -1."""
    device = coords.device
    key = (coords[:, 0].long() << 30) | (coords[:, 1].long() << 20) | (coords[:, 2].long() << 10) | coords[:, 3].long()
    order = torch.argsort(key)
    sorted_key = key[order]
    offsets = torch.stack(torch.meshgrid(*(torch.arange(-1, 2, device=device),) * 3, indexing="ij"), -1).reshape(-1, 3)
    n = coords.shape[0]
    table = torch.full((n, 27), -1, dtype=torch.long, device=device)
    for t in range(27):
        q = coords[:, 1:].long() + offsets[t]
        qkey = (coords[:, 0].long() << 30) | (q[:, 0] << 20) | (q[:, 1] << 10) | q[:, 2]
        pos = torch.searchsorted(sorted_key, qkey)
        pos = pos.clamp(max=n - 1)
        hit = sorted_key[pos] == qkey
        table[:, t] = torch.where(hit, order[pos], torch.full_like(pos, -1))
    return table


def submanifold_conv3d(feats, coords, weight, bias, table=None, block=65536):
    """feats [N,Ci], weight [Co,3,3,3,Ci] -> [N,Co]; nur aktive Nachbarn, Ausgabe nur auf aktiven Voxeln."""
    if table is None:
        table = neighbour_table(coords)
    n, ci = feats.shape
    co = weight.shape[0]
    w = weight.reshape(co, 27 * ci).to(feats.dtype)
    out = torch.empty((n, co), dtype=feats.dtype, device=feats.device)
    padded = torch.cat([feats, torch.zeros((1, ci), dtype=feats.dtype, device=feats.device)], 0)
    for start in range(0, n, block):
        idx = table[start:start + block]
        gathered = padded[idx.clamp(min=-1) % (n + 1)]            # -1 -> Nullzeile n
        out[start:start + block] = gathered.reshape(idx.shape[0], 27 * ci) @ w.t()
    if bias is not None:
        out += bias.to(out.dtype)
    return out


def install():
    """Patcht die Sparse-Conv und die sparse Attention von trellis2 auf die torch-Versionen."""
    import trellis2.modules.sparse.conv.conv as conv_mod
    import trellis2.modules.sparse.attention.full_attn as attn_mod
    import torch.nn as nn
    import torch.nn.functional as F

    def init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        assert stride == 1 and padding is None
        self.in_channels, self.out_channels = in_channels, out_channels
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.weight = nn.Parameter(torch.empty((out_channels, k, k, k, in_channels)))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    def forward(self, x):
        key = "cpu_neighbour_table"
        table = x.get_spatial_cache(key)
        if table is None:
            table = neighbour_table(x.coords)
            x.register_spatial_cache(key, table)
        return x.replace(submanifold_conv3d(x.feats, x.coords, self.weight, self.bias, table))

    conv_mod.SparseConv3d.__init__ = init
    conv_mod.SparseConv3d.forward = forward

    def sdpa(*args, **kwargs):
        # Batch 1: die VarLen-Sequenz ist eine dichte Sequenz.
        if len(args) == 1:
            qkv = args[0]
            q, k, v = qkv.feats.unbind(1)                       # [T,H,C]
            s = qkv
        elif len(args) == 2:
            q, kv = args
            s = q if hasattr(q, "feats") else None
            qf = q.feats if s is not None else q.reshape(-1, *q.shape[2:])
            kvf = kv.feats if hasattr(kv, "feats") else kv.reshape(-1, *kv.shape[2:])
            q, (k, v) = qf, kvf.unbind(1)
        else:
            q, k, v = args
            s = q if hasattr(q, "feats") else None
            q = q.feats if s is not None else q.reshape(-1, *q.shape[2:])
            k = k.feats if hasattr(k, "feats") else k.reshape(-1, *k.shape[2:])
            v = v.feats if hasattr(v, "feats") else v.reshape(-1, *v.shape[2:])
        out = F.scaled_dot_product_attention(q.permute(1, 0, 2)[None], k.permute(1, 0, 2)[None], v.permute(1, 0, 2)[None])
        out = out[0].permute(1, 0, 2)
        return s.replace(out) if s is not None else out[None]

    attn_mod.sparse_scaled_dot_product_attention = sdpa
    import trellis2.modules.sparse.attention.modules as m
    m.sparse_scaled_dot_product_attention = sdpa
