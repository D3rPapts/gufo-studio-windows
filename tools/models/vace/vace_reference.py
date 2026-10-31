#!/usr/bin/env python3
# Written with AI assistance (Claude Code).
"""CPU FP32 references for gufo's Wan 2.1 VACE port (src/models/wan22/vace*).

--context  ComfyUI's WanVaceToVideo + WAN21_Vace.extra_conds layout on the
           tiny example of tests/models/wan22/vace_context_test.cpp (prints
           the values the C++ test asserts, computed with torch's own
           nearest-exact interpolation and view/permute unshuffle).
--block    One VaceWanAttentionBlock (block_id 0: before_proj + x, the Wan
           block with t2v cross-attention, after_proj) as ComfyUI's
           comfy/ldm/wan/model.py runs it, with random weights (or block 0 of
           a VACE module file with --weights, FP8 dequantized by its
           scale_weight). Writes inputs and outputs to --out (safetensors)
           for an oracle check of gufo's Vace hook. Width 5120 / FFN 13824
           as the 14B; a few tokens only (<= ~3 GB RAM with real weights).

Run on the CPU only, e.g.
  HIP_VISIBLE_DEVICES=-1 python -I tools/models/vace/vace_reference.py --context
"""

import argparse
import json
import math
import struct
import sys

import torch
import torch.nn.functional as F

DIM, FFN, HEADS, HEAD_DIM, TEXT_DIM = 5120, 13824, 40, 128, 4096


def context_reference():
    frames, h, w = 5, 16, 16
    lf, lh, lw = (frames - 1) // 4 + 1, h // 8, w // 8
    inactive = torch.arange(16 * lf * lh * lw, dtype=torch.float32).view(
        1, 16, lf, lh, lw)
    reactive = -inactive
    mask = torch.zeros(frames, h, w)
    for f in range(frames):
        for y in range(h):
            for x in range(w):
                mask[f, y, x] = f * 1000 + y * 16 + x
    reference = (100.0 + torch.arange(16, dtype=torch.float32)).view(
        1, 16, 1, 1, 1).expand(1, 16, 1, lh, lw)
    # WanVaceToVideo (latents already normalized, as gufo encodes them).
    control = torch.cat((inactive, reactive), dim=1)
    ref = torch.cat((reference, torch.zeros_like(reference)), dim=1)
    control = torch.cat((ref, control), dim=2)
    m = mask.view(frames, lh, 8, lw, 8).permute(2, 4, 0, 1, 3).reshape(
        64, frames, lh, lw)
    m = F.interpolate(m.unsqueeze(0), size=(lf, lh, lw),
                      mode='nearest-exact').squeeze(0)
    m = torch.cat((torch.zeros_like(m[:, :1]), m), dim=1).unsqueeze(0)
    out = torch.cat((control, m), dim=1)[0]  # [96, 1 + lf, lh, lw]
    print('context shape', tuple(out.shape))
    print('reference latent c3 (1,1):', out[3, 0, 1, 1].item())
    print('reference reactive c20:', out[20, 0, 0, 1].item())
    print('inactive c2 f0 (0,1):', out[2, 1, 0, 1].item())
    print('reactive c5 f1 (1,0):', out[16 + 5, 2, 1, 0].item())
    kh, kw = 3, 6
    print('mask kh3 kw6 f0 (1,0):', out[32 + kh * 8 + kw, 1, 1, 0].item(),
          'expected', 1000 * 1 + (8 + kh) * 16 + kw)
    print('mask kh3 kw6 f1 (0,1):', out[32 + kh * 8 + kw, 2, 0, 1].item(),
          'expected', 1000 * 3 + kh * 16 + 8 + kw)
    # Control split (one pixel, mask 0.25).
    v = torch.tensor([255, 0, 51], dtype=torch.float32) / 255.0 - 0.5
    mk = 0.25
    print('split inactive', (2 * (v * (1 - mk) + 0.5) - 1).tolist(),
          'reactive', (2 * (v * mk + 0.5) - 1).tolist())


def rope_table(frames, ph, pw):
    """(cos, sin) [tokens, 64] like transformer21.hip (22 t + 21 h + 21 w)."""
    pairs = []
    for f in range(frames):
        for y in range(ph):
            for x in range(pw):
                row = []
                for j in range(64):
                    axis = 0 if j < 22 else (1 if j < 43 else 2)
                    k = j if axis == 0 else (j - 22 if axis == 1 else j - 43)
                    dims = 44 if axis == 0 else 42
                    freq = 1.0 / (10000.0 ** (2 * k / dims))
                    pos = (f, y, x)[axis]
                    row.append(pos * freq)
                pairs.append(row)
    angle = torch.tensor(pairs, dtype=torch.float64)
    return torch.cos(angle).float(), torch.sin(angle).float()


def rotate(x, cos, sin):
    # x [tokens, heads, 128]; pairs (2i, 2i + 1).
    a, b = x[..., 0::2], x[..., 1::2]
    c, s = cos[:, None, :], sin[:, None, :]
    out = torch.empty_like(x)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return out


def rms(x, weight):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight


def attention(q, k, v):
    # [rows, heads, 128] -> [rows, heads * 128]
    q, k, v = (t.transpose(0, 1) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(0, 1).reshape(q.shape[1], -1)


def vace_block(p, c, x, e0, text, cos, sin, first):
    """ComfyUI VaceWanAttentionBlock.forward -> (c_skip, c)."""
    lin = lambda t, n: t @ p[n + '.weight'].T + p[n + '.bias']
    if first:
        c = lin(c, 'before_proj') + x
    e = (p['modulation'].view(6, DIM) + e0).unbind(0)
    norm = lambda t: F.layer_norm(t, (DIM,), eps=1e-6)
    # Self-attention.
    y = norm(c) * (1 + e[1]) + e[0]
    q = rms(lin(y, 'self_attn.q'), p['self_attn.norm_q.weight'])
    k = rms(lin(y, 'self_attn.k'), p['self_attn.norm_k.weight'])
    v = lin(y, 'self_attn.v')
    shape = (-1, HEADS, HEAD_DIM)
    q, k = rotate(q.view(shape), cos, sin), rotate(k.view(shape), cos, sin)
    y = lin(attention(q, k, v.view(shape)), 'self_attn.o')
    c = c + y * e[2]
    # Cross-attention (t2v) on the embedded text, norm3 affine.
    y = F.layer_norm(c, (DIM,), p['norm3.weight'], p['norm3.bias'], 1e-6)
    q = rms(lin(y, 'cross_attn.q'), p['cross_attn.norm_q.weight'])
    k = rms(lin(text, 'cross_attn.k'), p['cross_attn.norm_k.weight'])
    v = lin(text, 'cross_attn.v')
    c = c + lin(attention(q.view(shape), k.view(shape), v.view(shape)),
                'cross_attn.o')
    # Feed-forward.
    y = norm(c) * (1 + e[4]) + e[3]
    y = lin(F.gelu(lin(y, 'ffn.0'), approximate='tanh'), 'ffn.2')
    c = c + y * e[5]
    return lin(c, 'after_proj'), c


def shapes():
    s = {'before_proj.weight': (DIM, DIM), 'after_proj.weight': (DIM, DIM),
         'ffn.0.weight': (FFN, DIM), 'ffn.2.weight': (DIM, FFN),
         'ffn.0.bias': (FFN,), 'modulation': (1, 6, DIM),
         'norm3.weight': (DIM,), 'norm3.bias': (DIM,)}
    for name in ('before_proj', 'after_proj', 'ffn.2'):
        s[name + '.bias'] = (DIM,)
    for part in ('self_attn', 'cross_attn'):
        for name in 'qkvo':
            s[f'{part}.{name}.weight'] = (DIM, DIM)
            s[f'{part}.{name}.bias'] = (DIM,)
        s[f'{part}.norm_q.weight'] = (DIM,)
        s[f'{part}.norm_k.weight'] = (DIM,)
    return s


def fp8_e4m3(raw):
    b = torch.frombuffer(bytearray(raw), dtype=torch.uint8).int()
    e, m = (b >> 3) & 15, b & 7
    v = torch.where(e == 0, m.float() * 2.0 ** -9,
                    (1 + m.float() / 8) * torch.pow(2.0, (e - 7).float()))
    return torch.where((b & 0x80) != 0, -v, v)


def load_block0(path):
    with open(path, 'rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(size))
        base = 8 + size
        out = {}
        for name, shape in shapes().items():
            entry = header['vace_blocks.0.' + name]
            begin, end = entry['data_offsets']
            f.seek(base + begin)
            raw = f.read(end - begin)
            dtype = entry['dtype']
            if dtype == 'F8_E4M3':
                scale_entry = header['vace_blocks.0.' + name.replace(
                    '.weight', '.scale_weight')]
                f.seek(base + scale_entry['data_offsets'][0])
                sraw = f.read(2 if scale_entry['dtype'] == 'BF16' else 4)
                scale = (torch.frombuffer(bytearray(sraw), dtype=torch.bfloat16)
                         if len(sraw) == 2 else
                         torch.frombuffer(bytearray(sraw), dtype=torch.float32))
                t = fp8_e4m3(raw) * scale.float()
            else:
                t = torch.frombuffer(bytearray(raw), dtype={
                    'BF16': torch.bfloat16, 'F16': torch.float16,
                    'F32': torch.float32}[dtype]).float()
            out[name] = t.view(shape).clone()
    return out


def block_reference(args):
    torch.manual_seed(args.seed)
    if args.weights:
        p = load_block0(args.weights)
    else:
        p = {}
        for name, shape in shapes().items():
            fan = shape[-1]
            if name.endswith('norm_q.weight') or name.endswith(
                    'norm_k.weight') or name == 'norm3.weight':
                p[name] = 1 + 0.1 * torch.randn(shape)
            elif name == 'modulation':
                p[name] = 0.05 * torch.randn(shape)
            else:
                p[name] = torch.randn(shape) / math.sqrt(fan) * (
                    1.0 if name.endswith('weight') else 0.1)
    frames, ph, pw, rows = 2, 3, 4, 16
    tokens = frames * ph * pw
    c = torch.randn(tokens, DIM)
    x = torch.randn(tokens, DIM)
    e0 = 0.1 * torch.randn(6, DIM)
    text = torch.randn(rows, DIM)
    cos, sin = rope_table(frames, ph, pw)
    with torch.no_grad():
        hint, state = vace_block(p, c, x, e0, text, cos, sin, True)
    print('tokens', tokens, 'hint', tuple(hint.shape), 'rms',
          hint.pow(2).mean().sqrt().item(), 'state rms',
          state.pow(2).mean().sqrt().item())
    if args.out:
        from safetensors.torch import save_file
        save_file({'c': c, 'x': x, 'e0': e0, 'text': text, 'hint': hint,
                   'state': state}, args.out,
                  metadata={'frames': str(frames), 'ph': str(ph),
                            'pw': str(pw)})
        print('wrote', args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--context', action='store_true')
    parser.add_argument('--block', action='store_true')
    parser.add_argument('--weights', help='VACE module safetensors (block 0)')
    parser.add_argument('--out', help='safetensors with inputs and outputs')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.context:
        context_reference()
    if args.block:
        block_reference(args)
    if not (args.context or args.block):
        parser.print_help()
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
