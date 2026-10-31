#!/usr/bin/env python3
"""FP32 CPU references for gufo's native MMAudio (src/models/mmaudio).

Each MMAudio component is re-implemented here in plain functional torch,
transcribed from hkchengrex/MMAudio (networks.py, transformer_layers.py,
ext/synchformer, ext/autoencoder, ext/bigvgan_v2) and OpenCLIP, and run on
small fixed random inputs with the real weights. Weights are read tensor by
tensor from the safetensors files and widened to FP32, so peak RAM stays
around 2 GB (the largest single tower layer set plus activations). The
downloaded MMAudio code itself is NOT imported (it is only read), and the
tokenizer reference uses transformers' CLIPTokenizer as an independent
implementation.

    python -I reference.py --model D:/llm-modelle/Kijai/MMAudio_safetensors \
        --out D:/llm-modelle/Kijai/MMAudio_safetensors/reference/mmaudio_reference.safetensors

Compare with: gufo-mmaudio-check --model ROOT --reference <out>.
CPU only (set HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=).
Written with AI assistance (Claude Code).
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_grad_enabled(False)
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))


class Weights:
    """Lazy FP32 access to one safetensors file."""

    def __init__(self, path):
        self.file = safe_open(path, framework="pt", device="cpu")

    def __call__(self, name):
        return self.file.get_tensor(name).float()

    def has(self, name):
        return name in self.file.keys()


def find(root, prefix):
    names = sorted(n for n in os.listdir(root)
                   if n.startswith(prefix) and n.endswith(".safetensors"))
    fp16 = [n for n in names if "fp16" in n]
    return os.path.join(root, (fp16 or names)[0])


# ---------------------------------------------------------------- tokenizer

def clip_tokens(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    if len(ids) > 77:
        ids = ids[:77]
        ids[-1] = 49407
    return ids + [0] * (77 - len(ids))


# --------------------------------------------------------------------- CLIP

def clip_block(x, w, p, heads, mask=None):
    width = x.shape[-1]
    y = F.layer_norm(x, (width,), w(p + "ln_1.weight"), w(p + "ln_1.bias"))
    qkv = F.linear(y, w(p + "attn.in_proj_weight"), w(p + "attn.in_proj_bias"))
    q, k, v = qkv.split(width, dim=-1)
    b, n, _ = q.shape
    d = width // heads
    q, k, v = (t.view(b, n, heads, d).transpose(1, 2) for t in (q, k, v))
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    a = a.transpose(1, 2).reshape(b, n, width)
    x = x + F.linear(a, w(p + "attn.out_proj.weight"), w(p + "attn.out_proj.bias"))
    y = F.layer_norm(x, (width,), w(p + "ln_2.weight"), w(p + "ln_2.bias"))
    y = F.linear(y, w(p + "mlp.c_fc.weight"), w(p + "mlp.c_fc.bias"))
    y = y * torch.sigmoid(1.702 * y)  # QuickGELU
    return x + F.linear(y, w(p + "mlp.c_proj.weight"), w(p + "mlp.c_proj.bias"))


def clip_image(w, pixels):
    x = F.conv2d(pixels, w("visual.conv1.weight"), stride=14)  # [B, 1280, 27, 27]
    x = x.flatten(2).transpose(1, 2)
    cls = w("visual.class_embedding").view(1, 1, -1).expand(x.shape[0], -1, -1)
    x = torch.cat([cls, x], dim=1) + w("visual.positional_embedding")
    x = F.layer_norm(x, (1280,), w("visual.ln_pre.weight"), w("visual.ln_pre.bias"))
    for i in range(32):
        x = clip_block(x, w, f"visual.transformer.resblocks.{i}.", 16)
    x = F.layer_norm(x, (1280,), w("visual.ln_post.weight"), w("visual.ln_post.bias"))
    pooled = x[:, 0] @ w("visual.proj")
    return F.normalize(pooled, dim=-1)


def clip_text(w, tokens):
    x = w("token_embedding.weight")[tokens].unsqueeze(0) + w("positional_embedding")
    mask = torch.full((77, 77), float("-inf")).triu(1)
    for i in range(24):
        x = clip_block(x, w, f"transformer.resblocks.{i}.", 16, mask)
    x = F.layer_norm(x, (1024,), w("ln_final.weight"), w("ln_final.bias"))
    return F.normalize(x, dim=-1)[0]


# -------------------------------------------------------------- Synchformer

def divided_attention(x, w, p, mode, frames=8, patches=196):
    # vit_helper.DividedAttention, batch 1, 12 heads of 64.
    n, dim = x.shape
    h, d = 12, 64
    q, k, v = F.linear(x, w(p + "qkv.weight"), w(p + "qkv.bias")).chunk(3, dim=-1)
    q, k, v = (t.view(n, h, d).transpose(0, 1) for t in (q, k, v))  # [h, n, d]
    q = q * d ** -0.5
    cls_out = (q[:, :1] @ k.transpose(1, 2)).softmax(-1) @ v  # [h, 1, d]
    qp, kp, vp = (t[:, 1:].reshape(h, frames, patches, d) for t in (q, k, v))
    if mode == "time":  # '(b n) f d': group per patch over frames
        qp, kp, vp = (t.transpose(1, 2) for t in (qp, kp, vp))  # [h, P, F, d]
    groups = qp.shape[1]
    ck = k[:, :1].unsqueeze(1).expand(h, groups, 1, d)
    cv = v[:, :1].unsqueeze(1).expand(h, groups, 1, d)
    kk = torch.cat([ck, kp], dim=2)
    vv = torch.cat([cv, vp], dim=2)
    out = (qp @ kk.transpose(-1, -2)).softmax(-1) @ vv
    if mode == "time":
        out = out.transpose(1, 2)
    out = torch.cat([cls_out, out.reshape(h, frames * patches, d)], dim=1)
    out = out.transpose(0, 1).reshape(n, dim)
    return F.linear(out, w(p + "proj.weight"), w(p + "proj.bias"))


def synchformer(w, frames):
    """frames [16, 3, 224, 224] -> [8, 768] (one segment)."""
    v = "vfeat_extractor."
    x = frames.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
    x = F.conv3d(x, w(v + "patch_embed_3d.proj.weight"),
                 w(v + "patch_embed_3d.proj.bias"), stride=(2, 16, 16))
    x = x.flatten(2).transpose(1, 2)[0]  # [1568, 768]
    pos = w(v + "pos_embed")[0]
    temp = w(v + "temp_embed")[0]
    x = torch.cat([w(v + "cls_token")[0, 0][None], x], dim=0)
    total = torch.cat([pos[:1], pos[1:].repeat(8, 1) + temp.repeat_interleave(196, 0)])
    x = x + total
    ln = lambda t, p: F.layer_norm(t, (768,), w(p + ".weight"), w(p + ".bias"), eps=1e-6)
    for i in range(12):
        p = f"{v}blocks.{i}."
        x = x + divided_attention(ln(x, p + "norm3"), w, p + "timeattn.", "time")
        x = x + divided_attention(ln(x, p + "norm1"), w, p + "attn.", "space")
        y = F.gelu(F.linear(ln(x, p + "norm2"), w(p + "mlp.fc1.weight"), w(p + "mlp.fc1.bias")))
        x = x + F.linear(y, w(p + "mlp.fc2.weight"), w(p + "mlp.fc2.bias"))
    x = ln(x[1:], v + "norm").view(8, 196, 768)
    a = v + "spatial_attn_agg."
    x = torch.cat([w(a + "cls_token").view(1, 1, 768).expand(8, 1, 768), x], dim=1)
    y = ln(x, a + "norm1")
    qkv = F.linear(y, w(a + "self_attn.in_proj_weight"), w(a + "self_attn.in_proj_bias"))
    q, k, vv = (t.view(8, 197, 12, 64).transpose(1, 2) for t in qkv.chunk(3, dim=-1))
    att = F.scaled_dot_product_attention(q, k, vv).transpose(1, 2).reshape(8, 197, 768)
    x = x + F.linear(att, w(a + "self_attn.out_proj.weight"), w(a + "self_attn.out_proj.bias"))
    y = F.gelu(F.linear(ln(x, a + "norm2"), w(a + "linear1.weight"), w(a + "linear1.bias")))
    x = x + F.linear(y, w(a + "linear2.weight"), w(a + "linear2.bias"))
    return x[:, 0]


# --------------------------------------------------------- MMAudio network

def conv_cl(x, weight, bias=None):
    """ChannelLastConv1d on [N, C]."""
    return F.conv1d(x.t()[None], weight, bias, padding=weight.shape[-1] // 2)[0].t()


def gated(x, w, p):
    w1 = w(p + ".w1.weight")
    if w1.dim() == 3:
        return conv_cl(F.silu(conv_cl(x, w1)) * conv_cl(x, w(p + ".w3.weight")),
                       w(p + ".w2.weight"))
    return F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w(p + ".w3.weight")),
                    w(p + ".w2.weight"))


def rope_table(length, dim, scaling):
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = freqs * scaling
    rot = torch.arange(length, dtype=torch.float32)[:, None] * freqs[None]
    return torch.stack([rot.cos(), -rot.sin(), rot.sin(), rot.cos()], -1).view(length, dim // 2, 2, 2)


def apply_rope(x, rot):  # x [h, n, d]
    xs = x.view(*x.shape[:-1], -1, 1, 2)
    out = rot[..., 0] * xs[..., 0] + rot[..., 1] * xs[..., 1]
    return out.reshape(x.shape)


def pre_attention(x, c, w, p, rot):
    mod = F.linear(F.silu(c), w(p + ".adaLN_modulation.1.weight"), w(p + ".adaLN_modulation.1.bias"))
    chunks = mod.chunk(mod.shape[-1] // 896, dim=-1)
    shift, scale = chunks[0], chunks[1]
    y = F.layer_norm(x, (896,)) * (1 + scale) + shift
    qkv = F.linear(y, w(p + ".attn.qkv.weight"), w(p + ".attn.qkv.bias"))
    n = x.shape[0]
    qkv = qkv.view(n, 14, 64, 3)
    q, k, v = (qkv[..., j].transpose(0, 1) for j in range(3))  # [h, n, d]
    eps = torch.finfo(torch.float32).eps
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps) * w(p + ".attn.q_norm.weight")
    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + eps) * w(p + ".attn.k_norm.weight")
    if rot is not None:
        q, k = apply_rope(q, rot), apply_rope(k, rot)
    return (q, k, v), chunks


def post_attention(x, attn, chunks, w, p):
    if len(chunks) == 2:
        return x
    _, _, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks
    lw = w(p + ".linear1.weight")
    lin = conv_cl(attn, lw, w(p + ".linear1.bias")) if lw.dim() == 3 else F.linear(
        attn, lw, w(p + ".linear1.bias"))
    x = x + lin * gate_msa
    r = F.layer_norm(x, (896,)) * (1 + scale_mlp) + shift_mlp
    return x + gated(r, w, p + ".ffn") * gate_mlp


def sdpa(qkv_list):
    q = torch.cat([t[0] for t in qkv_list], dim=1)
    k = torch.cat([t[1] for t in qkv_list], dim=1)
    v = torch.cat([t[2] for t in qkv_list], dim=1)
    out = F.scaled_dot_product_attention(q[None], k[None], v[None])[0]
    return out.transpose(0, 1).reshape(q.shape[1], 896)


def network_flow(w, clip_f, sync_f, text_f, latent, t, lengths):
    n, lc, ls = lengths
    # preprocess_conditions
    sync = (sync_f.view(ls // 8, 8, 768) + w("sync_pos_emb")[0, 0]).reshape(ls, 768)
    clip = gated(F.silu(F.linear(clip_f, w("clip_input_proj.0.weight"), w("clip_input_proj.0.bias"))),
                 w, "clip_input_proj.2")
    sync = gated(F.silu(conv_cl(sync, w("sync_input_proj.0.weight"), w("sync_input_proj.0.bias"))),
                 w, "sync_input_proj.2")
    text = gated(F.silu(F.linear(text_f, w("text_input_proj.0.weight"), w("text_input_proj.0.bias"))),
                 w, "text_input_proj.2")
    sync = F.interpolate(sync.t()[None], size=n, mode="nearest-exact")[0].t()
    clip_c = F.linear(clip.mean(0), w("clip_cond_proj.weight"), w("clip_cond_proj.bias"))
    text_c = F.linear(text.mean(0), w("text_cond_proj.weight"), w("text_cond_proj.bias"))
    # predict_flow
    x = gated(F.silu(conv_cl(latent, w("audio_input_proj.0.weight"), w("audio_input_proj.0.bias"))),
              w, "audio_input_proj.2")
    global_c = gated(clip_c + text_c, w, "global_cond_mlp")
    freqs = 10000.0 * (1.0 / (10000 ** (torch.arange(0, 896, 2, dtype=torch.float32) / 896)))
    args = torch.tensor(t) * freqs
    temb = torch.cat([args.cos(), args.sin()])
    temb = F.linear(F.silu(F.linear(temb, w("t_embed.mlp.0.weight"), w("t_embed.mlp.0.bias"))),
                    w("t_embed.mlp.2.weight"), w("t_embed.mlp.2.bias"))
    global_c = (temb + global_c)[None]
    extended_c = global_c + sync
    latent_rot = rope_table(n, 64, 1.0)
    clip_rot = rope_table(lc, 64, n / lc)
    for i in range(7):
        p = f"joint_blocks.{i}"
        xq, xm = pre_attention(x, extended_c, w, p + ".latent_block", latent_rot)
        cq, cm = pre_attention(clip, global_c, w, p + ".clip_block", clip_rot)
        tq, tm = pre_attention(text, global_c, w, p + ".text_block", None)
        a = sdpa([xq, cq, tq])
        x = post_attention(x, a[:n], xm, w, p + ".latent_block")
        clip = post_attention(clip, a[n:n + lc], cm, w, p + ".clip_block")
        text = post_attention(text, a[n + lc:], tm, w, p + ".text_block")
    for i in range(14):
        p = f"fused_blocks.{i}"
        xq, xm = pre_attention(x, extended_c, w, p, latent_rot)
        x = post_attention(x, sdpa([xq]), xm, w, p)
    mod = F.linear(F.silu(global_c), w("final_layer.adaLN_modulation.1.weight"),
                   w("final_layer.adaLN_modulation.1.bias"))
    shift, scale = mod.chunk(2, dim=-1)
    y = F.layer_norm(x, (896,)) * (1 + scale) + shift
    return conv_cl(y, w("final_layer.conv.weight"), w("final_layer.conv.bias"))


# ---------------------------------------------------------------------- VAE

MEL_MEAN = [
    -3.3462, -2.6723, -2.4893, -2.3143, -2.2664, -2.3317, -2.1802, -2.4006, -2.2357, -2.4597,
    -2.3717, -2.4690, -2.5142, -2.4919, -2.6610, -2.5047, -2.7483, -2.5926, -2.7462, -2.7033,
    -2.7386, -2.8112, -2.7502, -2.9594, -2.7473, -3.0035, -2.8891, -2.9922, -2.9856, -3.0157,
    -3.1191, -2.9893, -3.1718, -3.0745, -3.1879, -3.2310, -3.1424, -3.2296, -3.2791, -3.2782,
    -3.2756, -3.3134, -3.3509, -3.3750, -3.3951, -3.3698, -3.4505, -3.4509, -3.5089, -3.4647,
    -3.5536, -3.5788, -3.5867, -3.6036, -3.6400, -3.6747, -3.7072, -3.7279, -3.7283, -3.7795,
    -3.8259, -3.8447, -3.8663, -3.9182, -3.9605, -3.9861, -4.0105, -4.0373, -4.0762, -4.1121,
    -4.1488, -4.1874, -4.2461, -4.3170, -4.3639, -4.4452, -4.5282, -4.6297, -4.7019, -4.7960,
    -4.8700, -4.9507, -5.0303, -5.0866, -5.1634, -5.2342, -5.3242, -5.4053, -5.4927, -5.5712,
    -5.6464, -5.7052, -5.7619, -5.8410, -5.9188, -6.0103, -6.0955, -6.1673, -6.2362, -6.3120,
    -6.3926, -6.4797, -6.5565, -6.6511, -6.8130, -6.9961, -7.1275, -7.2457, -7.3576, -7.4663,
    -7.6136, -7.7469, -7.8815, -8.0132, -8.1515, -8.3071, -8.4722, -8.7418, -9.3975, -9.6628,
    -9.7671, -9.8863, -9.9992, -10.0860, -10.1709, -10.5418, -11.2795, -11.3861]
MEL_STD = [
    2.3804, 2.4368, 2.3772, 2.3145, 2.2803, 2.2510, 2.2316, 2.2083, 2.1996, 2.1835, 2.1769, 2.1659,
    2.1631, 2.1618, 2.1540, 2.1606, 2.1571, 2.1567, 2.1612, 2.1579, 2.1679, 2.1683, 2.1634, 2.1557,
    2.1668, 2.1518, 2.1415, 2.1449, 2.1406, 2.1350, 2.1313, 2.1415, 2.1281, 2.1352, 2.1219, 2.1182,
    2.1327, 2.1195, 2.1137, 2.1080, 2.1179, 2.1036, 2.1087, 2.1036, 2.1015, 2.1068, 2.0975, 2.0991,
    2.0902, 2.1015, 2.0857, 2.0920, 2.0893, 2.0897, 2.0910, 2.0881, 2.0925, 2.0873, 2.0960, 2.0900,
    2.0957, 2.0958, 2.0978, 2.0936, 2.0886, 2.0905, 2.0845, 2.0855, 2.0796, 2.0840, 2.0813, 2.0817,
    2.0838, 2.0840, 2.0917, 2.1061, 2.1431, 2.1976, 2.2482, 2.3055, 2.3700, 2.4088, 2.4372, 2.4609,
    2.4731, 2.4847, 2.5072, 2.5451, 2.5772, 2.6147, 2.6529, 2.6596, 2.6645, 2.6726, 2.6803, 2.6812,
    2.6899, 2.6916, 2.6931, 2.6998, 2.7062, 2.7262, 2.7222, 2.7158, 2.7041, 2.7485, 2.7491, 2.7451,
    2.7485, 2.7233, 2.7297, 2.7233, 2.7145, 2.6958, 2.6788, 2.6439, 2.6007, 2.4786, 2.2469, 2.1877,
    2.1392, 2.0717, 2.0107, 1.9676, 1.9140, 1.7102, 0.9101, 0.7164]


def edm_normalize(x, dim, eps=1e-4):
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    norm = eps + norm * math.sqrt(norm.numel() / x.numel())
    return x / norm


def mp_weight(w_raw, gain=1.0):
    wt = edm_normalize(w_raw, dim=list(range(1, w_raw.dim())))
    return wt / math.sqrt(wt[0].numel()) * gain


def mp_conv(x, w_raw, gain=1.0):  # x [B, C, T]
    wt = mp_weight(w_raw, gain)
    return F.conv1d(x, wt, padding=wt.shape[-1] // 2)


def mp_silu(x):
    return F.silu(x) / 0.596


def mp_sum(a, b, t=0.3):
    return a.lerp(b, t) / math.sqrt((1 - t) ** 2 + t ** 2)


def vae_res(x, w, p):
    x = edm_normalize(x, dim=1)
    h = mp_conv(mp_silu(x), w(p + ".conv1.weight"))
    h = mp_conv(mp_silu(h), w(p + ".conv2.weight"))
    if (p + ".nin_shortcut.weight") in w.file.keys():
        x = mp_conv(x, w(p + ".nin_shortcut.weight"))
    return mp_sum(x, h)


def vae_attn(x, w, p):
    y = mp_conv(x, w(p + ".qkv.weight"))
    y = y.reshape(y.shape[0], 1, -1, 3, y.shape[-1])
    q, k, v = edm_normalize(y, dim=2).unbind(3)
    q, k, v = (t.transpose(-1, -2) for t in (q, k, v))  # [b, h, l, c]
    h = F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(x.shape)
    return mp_sum(x, mp_conv(h, w(p + ".proj_out.weight")))


def vae_decode(w, latent):  # latent [N, 40] -> mel [2N, 128]
    d = "decoder."
    h = mp_conv(latent.t()[None], w(d + "conv_in.weight"))
    h = vae_res(h, w, d + "mid.block_1")
    h = vae_attn(h, w, d + "mid.attn_1")
    h = vae_res(h, w, d + "mid.block_2").clamp(-256, 256)
    for level in (2, 1, 0):
        for b in range(3):
            h = vae_res(h, w, f"{d}up.{level}.block.{b}").clamp(-256, 256)
        if level == 1:
            h = F.interpolate(h, scale_factor=2.0, mode="nearest-exact")
            h = mp_conv(h, w(d + "up.1.upsample.conv.weight"))
    h = mp_silu(h)
    h = mp_conv(h, w(d + "conv_out.weight"), gain=float(w(d + "learnable_gain")) + 1.0)
    mel = h * torch.tensor(MEL_STD).view(1, -1, 1) + torch.tensor(MEL_MEAN).view(1, -1, 1)
    return mel[0].t()


# ------------------------------------------------------------------ BigVGAN

def kaiser_sinc_filter(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    a = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if a > 50.0:
        beta = 0.1102 * (a - 8.7)
    elif a >= 21.0:
        beta = 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False, dtype=torch.float64)
    time = torch.arange(-half_size, half_size, dtype=torch.float64) + 0.5 if even else \
        torch.arange(kernel_size, dtype=torch.float64) - half_size
    f = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    return (f / f.sum()).float()


def snake_aa(x, alpha, beta, filt):
    """Activation1d(SnakeBeta, logscale) on [1, C, T]."""
    c = x.shape[1]
    k = filt.view(1, 1, 12).expand(c, -1, -1)
    # UpSample1d(2, 12): pad 5, transposed conv, crop 15 / 15.
    y = F.pad(x, (5, 5), mode="replicate")
    y = 2 * F.conv_transpose1d(y, k, stride=2, groups=c)
    y = y[..., 15:-15]
    a = alpha.exp().view(1, -1, 1)
    b = beta.exp().view(1, -1, 1)
    y = y + (1.0 / (b + 1e-9)) * torch.sin(y * a).pow(2)
    # DownSample1d: LowPassFilter1d(stride 2, pad 5 / 6, replicate).
    y = F.pad(y, (5, 6), mode="replicate")
    return F.conv1d(y, k, stride=2, groups=c)


def bigvgan(w, mel):  # mel [T, 128] -> audio [T * 512]
    filt = kaiser_sinc_filter(0.25, 0.3, 12)
    x = F.conv1d(mel.t()[None], w("conv_pre.weight"), w("conv_pre.bias"), padding=3)
    rates, kernels = [8, 4, 2, 2, 2, 2], [16, 8, 4, 4, 4, 4]
    for i in range(6):
        x = F.conv_transpose1d(x, w(f"ups.{i}.0.weight"), w(f"ups.{i}.0.bias"),
                               stride=rates[i], padding=(kernels[i] - rates[i]) // 2)
        xs = 0
        for j, k in enumerate([3, 7, 11]):
            p = f"resblocks.{i * 3 + j}."
            r = x
            for n, d in enumerate([1, 3, 5]):
                a1, a2 = p + f"activations.{2 * n}.act.", p + f"activations.{2 * n + 1}.act."
                xt = snake_aa(r, w(a1 + "alpha"), w(a1 + "beta"), filt)
                xt = F.conv1d(xt, w(p + f"convs1.{n}.weight"), w(p + f"convs1.{n}.bias"),
                              dilation=d, padding=d * (k - 1) // 2)
                xt = snake_aa(xt, w(a2 + "alpha"), w(a2 + "beta"), filt)
                xt = F.conv1d(xt, w(p + f"convs2.{n}.weight"), w(p + f"convs2.{n}.bias"),
                              padding=(k - 1) // 2)
                r = xt + r
            xs = xs + r
        x = xs / 3
    x = snake_aa(x, w("activation_post.act.alpha"), w("activation_post.act.beta"), filt)
    x = F.conv1d(x, w("conv_post.weight"), padding=3)
    return x.clamp(-1, 1)[0, 0], filt


# --------------------------------------------------------------------- main

def smooth_images(gen, count, channels, size):
    """Low-frequency random images in [0, 1]."""
    small = torch.rand(count, channels, 6, 6, generator=gen)
    return F.interpolate(small, size=(size, size), mode="bilinear", align_corners=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = args.model
    gen = torch.Generator().manual_seed(1234)
    out = {}
    stamp = time.time()

    def log(what):
        print(f"[{time.time() - stamp:7.1f} s] {what}", flush=True)

    # Tokenizer (transformers' CLIPTokenizer as an independent reference).
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(args.tokenizer or os.path.join(root, "tokenizer"))
    prompts = ["", "dog barking", "Footsteps on gravel, then a door slams!",
               "rain on a tin roof at 3 am, distant thunder (2 rolls)",
               "Hundegebell im Café, Gläser klirren",
               "a " + "very " * 90 + "long prompt"]
    for i, text in enumerate(prompts):
        out[f"tokenizer.{i}.text"] = torch.tensor(list(text.encode("utf-8")), dtype=torch.float32)
        out[f"tokenizer.{i}.ids"] = torch.tensor(clip_tokens(tok, text), dtype=torch.float32)
    log("tokenizer")

    out["kaiser.filter"] = kaiser_sinc_filter(0.25, 0.3, 12)

    w = Weights(find(root, "apple_DFN5B-CLIP-ViT-H-14-384"))
    tokens = clip_tokens(tok, "footsteps on gravel, then a door slams")
    out["clip_text.tokens"] = torch.tensor(tokens, dtype=torch.float32)
    out["clip_text.out"] = clip_text(w, torch.tensor(tokens))
    log("clip text")
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    pixels = (smooth_images(gen, 2, 3, 384) - mean) / std
    out["clip_image.pixels"] = pixels.contiguous()
    out["clip_image.out"] = clip_image(w, pixels)
    log("clip image")
    del w

    w = Weights(find(root, "mmaudio_synchformer"))
    frames = (smooth_images(gen, 16, 3, 224) - 0.5) / 0.5
    out["sync.frames"] = frames.contiguous()
    out["sync.out"] = synchformer(w, frames)
    log("synchformer")
    del w

    w = Weights(find(root, "mmaudio_large_44k_v2"))
    n, lc, ls = 87, 16, 40  # duration 2 s
    clip_f = F.normalize(torch.randn(lc, 1024, generator=gen), dim=-1)
    sync_f = torch.randn(ls, 768, generator=gen)
    text_f = F.normalize(torch.randn(77, 1024, generator=gen), dim=-1)
    latent = torch.randn(n, 40, generator=gen)
    t = 0.37
    out["net.lengths"] = torch.tensor([n, lc, ls], dtype=torch.float32)
    out["net.clip_f"], out["net.sync_f"], out["net.text_f"] = clip_f, sync_f, text_f
    out["net.latent"] = latent
    out["net.t"] = torch.tensor([t])
    out["net.flow"] = network_flow(w, clip_f, sync_f, text_f, latent, t, (n, lc, ls))
    log("network")
    del w

    w = Weights(find(root, "mmaudio_vae_44k"))
    vae_latent = torch.randn(24, 40, generator=gen) * 0.8
    out["vae.latent"] = vae_latent
    out["vae.mel"] = vae_decode(w, vae_latent).contiguous()
    log("vae")
    del w

    w = Weights(os.path.join(root, "bigvgan_v2_44khz_128band_512x.safetensors"))
    mel = (torch.randn(12, 128, generator=gen) * 1.5 - 5.0)
    out["bigvgan.mel"] = mel
    audio, filt = bigvgan(w, mel)
    out["bigvgan.audio"] = audio
    if w.has("activation_post.upsample.filter"):
        stored = w("activation_post.upsample.filter").flatten()
        print("kaiser filter vs stored checkpoint filter: max diff",
              float((stored - filt).abs().max()))
    log("bigvgan")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file({k: v.float().contiguous() for k, v in out.items()}, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
