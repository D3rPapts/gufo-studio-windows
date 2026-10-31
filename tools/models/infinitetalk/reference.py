#!/usr/bin/env python3
# Written with AI assistance (Claude Code).
"""InfiniteTalk / wav2vec2 reference blocks for gufo's port (CPU, FP32).

Re-implements, without importing ComfyUI, the pieces gufo ports from
ComfyUI's comfy/ldm/wan/model_multitalk.py, comfy_extras/nodes_wan.py
(WanInfiniteTalkToVideo) and comfy/audio_encoders/wav2vec2.py, and checks the
formulations gufo's C++ uses against them:

  * audio projection: gufo gathers the 8 later windows per group of 4 frames
    with explicit indices (infinitetalk.hip SetAudio) instead of ComfyUI's
    rearranges, and writes the tokens frame-major [frame][speaker*32][768];
  * speaker routing: gufo gets x_ref_attn_map from one attention launch whose
    values are the masks (channel c = mask c), instead of ComfyUI's chunked
    softmax + masked sum;
  * audio cross-attention (single and two speakers with the 1D RoPE);
  * wav2vec2-base forward (conv stack as strided GEMMs, grouped positional
    conv with weight norm, post-norm layers) and the 50 -> 25 fps resampling;
  * the window plan (frames, audio offsets) of a long video.

Random weights at the real widths, a few tokens: well under 4 GB of RAM.
With --talk-file / --wav2vec2-file the real checkpoints are read instead
(only the tensors needed: audio_proj, one block, the audio encoder).
--out writes the inputs and outputs as .npz fixtures for a C++ oracle check.

Run with the Wan venv:  python -I tools/models/infinitetalk/reference.py
"""

import argparse
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DIM, HEADS, HEAD_DIM, AUDIO, LAYERS_A, TOKENS = 5120, 40, 128, 768, 12, 32


# --------------------------------------------------------------------------
# ComfyUI formulation (re-typed from model_multitalk.py / nodes_wan.py).
def comfy_get_audio_embeds(encoded_audio, audio_start, audio_end):
    audio_embs = []
    audio_frames = encoded_audio[0].shape[0]
    indices = (torch.arange(4 + 1) - 2) * 1
    for emb in encoded_audio:
        if audio_end > audio_frames:
            pad = emb[:1].repeat(audio_end - audio_frames,
                                 *([1] * (emb.dim() - 1)))
            emb = torch.cat([emb, pad], dim=0)
        center = torch.arange(audio_start, audio_end, 1).unsqueeze(1) + \
            indices.unsqueeze(0)
        center = torch.clamp(center, min=0, max=emb.shape[0] - 1)
        audio_embs.append(emb[center].unsqueeze(0))
    return torch.cat(audio_embs, dim=0)


def comfy_project(proj, encoded_audio, audio_start, audio_end):
    embs = comfy_get_audio_embeds(encoded_audio, audio_start, audio_end)
    first = embs[:, :1, ...]
    later = embs[:, 1:, ...]
    b, f = later.shape[:2]
    later = later.reshape(b, f // 4, 4, *later.shape[2:])
    mid = 5 // 2
    lf = later[:, :, :1, :mid + 1].flatten(2, 3)
    ll = later[:, :, -1:, mid:].flatten(2, 3)
    lm = later[:, :, 1:-1, mid:mid + 1].flatten(2, 3)
    later = torch.cat([lf, lm, ll], dim=2)
    out = audio_proj_forward(proj, first, later)
    return torch.cat(out.split(1), dim=2)  # [1, frames, 32 * speakers, 768]


def audio_proj_forward(p, first, later):
    B = first.shape[0]
    n = first.shape[1] + later.shape[1]
    a = torch.relu(first.reshape(B * first.shape[1], -1) @ p["proj1.weight"].T
                   + p["proj1.bias"]).reshape(B, first.shape[1], -1)
    v = torch.relu(later.reshape(B * later.shape[1], -1) @
                   p["proj1_vf.weight"].T + p["proj1_vf.bias"]).reshape(
                       B, later.shape[1], -1)
    c = torch.cat([a, v], dim=1).reshape(B * n, -1)
    c = torch.relu(c @ p["proj2.weight"].T + p["proj2.bias"])
    t = (c @ p["proj3.weight"].T + p["proj3.bias"]).reshape(B * n, TOKENS,
                                                            AUDIO)
    t = F.layer_norm(t, (AUDIO,), p["norm.weight"], p["norm.bias"])
    return t.reshape(B, n, TOKENS, AUDIO)


def comfy_attn_map(q, k, shape, masks):
    """get_attn_map_with_target (q, k: [1, tokens, heads, 128])."""
    n_t, n_h, n_w = shape
    ref_k = k[:, :n_h * n_w]
    maps = torch.zeros(masks.shape[0], q.shape[1])
    for half in range(2):
        hs = slice(half * HEADS // 2, (half + 1) * HEADS // 2)
        vq = q[:, :, hs].transpose(1, 2) / math.sqrt(HEAD_DIM)
        rk = ref_k[:, :, hs]
        per = []
        for m in masks:
            m = m.view(1, 1, 1, -1).float()
            attn = vq @ rk.permute(0, 2, 3, 1)
            attn = (attn - attn.max(-1, keepdim=True).values).exp()
            attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)
            per.append(((attn * m).sum(-1) / (m.sum() + 1e-8)).mean(dim=1))
        maps += torch.cat(per, dim=0)
    return maps / 2


def rope_1d(x, pos):
    """RotaryPositionalEmbedding1D: interleaved pairs, base 10000."""
    freqs = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    angles = pos.float()[:, None] * freqs[None]
    angles = angles.repeat_interleave(2, dim=-1)
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x * cos + rot * sin


def comfy_audio_cross(blk, x, audio, shape, ref_map):
    """norm_x + SingleStreamMultiAttention (one or two speakers)."""
    n_t, n_h, n_w = shape
    S = n_h * n_w
    xn = F.layer_norm(x, (DIM,), blk["norm_x.weight"], blk["norm_x.bias"])
    q = (xn @ blk["q.weight"].T + blk["q.bias"]).view(n_t, S, HEADS, HEAD_DIM)
    kv = audio @ blk["kv.weight"].T + blk["kv.bias"]  # [n_t, keys, 2 * DIM]
    keys = audio.shape[1]
    k, v = kv.view(n_t, keys, 2, HEADS, HEAD_DIM).unbind(2)
    if ref_map is not None:
        r1, r2, back = (0, 4), (20, 24), 12
        mx = ref_map.max(1).values
        mn = ref_map.min(1).values
        h1 = (ref_map[0] - mn[0]) / (mx[0] - mn[0] + 1e-8) * 4 + r1[0]
        h2 = (ref_map[1] - mn[1]) / (mx[1] - mn[1] + 1e-8) * 4 + r2[0]
        pick = ref_map.argmax(dim=0)
        pos = torch.stack([h1, h2, torch.full_like(h1, back)], 1)[
            torch.arange(ref_map.shape[1]), pick]
        q = rope_1d(q.reshape(n_t * S, HEADS, HEAD_DIM).transpose(0, 1),
                    pos).transpose(0, 1).reshape(n_t, S, HEADS, HEAD_DIM)
        kpos = torch.zeros(keys)
        kpos[:keys // 2] = 2.0
        kpos[keys // 2:] = 22.0
        k = rope_1d(k.reshape(n_t * keys, HEADS, HEAD_DIM).transpose(0, 1),
                    kpos.repeat(n_t)).transpose(0, 1).reshape(
                        n_t, keys, HEADS, HEAD_DIM)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2)).transpose(1, 2)
    out = out.reshape(n_t * S, DIM) @ blk["proj.weight"].T + blk["proj.bias"]
    return out


# --------------------------------------------------------------------------
# gufo's formulation (mirrors infinitetalk.hip / talk21.cpp).
def gufo_project(proj, features, frames, start, length):
    """SetAudio: explicit gathers; tokens [latent][speakers * 32][768]."""
    speakers = len(features)
    padded = max(frames, start + length)

    def frame_of(s, i):
        i = min(max(i, 0), padded - 1)
        return features[s][0 if i >= frames else i]

    groups = (length - 1) // 4
    first = torch.stack([torch.stack([frame_of(s, start + w - 2)
                                      for w in range(5)]).flatten()
                         for s in range(speakers)])
    later = []
    for s in range(speakers):
        for g in range(groups):
            base = 1 + 4 * g
            picks = [(base, 0), (base, 1), (base, 2), (base + 1, 2),
                     (base + 2, 2), (base + 3, 2), (base + 3, 3),
                     (base + 3, 4)]
            later.append(torch.stack([frame_of(s, start + f + w - 2)
                                      for f, w in picks]).flatten())
    later = torch.stack(later)
    latent = 1 + groups
    hidden = torch.zeros(speakers * latent, 512)
    for s in range(speakers):
        hidden[s * latent] = first[s] @ proj["proj1.weight"].T
        hidden[s * latent + 1:(s + 1) * latent] = \
            later[s * groups:(s + 1) * groups] @ proj["proj1_vf.weight"].T
    rows = torch.arange(speakers * latent)
    bias = torch.where((rows % latent == 0)[:, None], proj["proj1.bias"],
                       proj["proj1_vf.bias"])
    hidden = torch.relu(hidden + bias)
    hidden = torch.relu(hidden @ proj["proj2.weight"].T + proj["proj2.bias"])
    expanded = hidden @ proj["proj3.weight"].T + proj["proj3.bias"]
    tokens = F.layer_norm(expanded.reshape(-1, AUDIO), (AUDIO,),
                          proj["norm.weight"], proj["norm.bias"])
    tokens = tokens.reshape(speakers, latent, TOKENS, AUDIO)
    return tokens.permute(1, 0, 2, 3).reshape(latent, speakers * TOKENS,
                                              AUDIO)


def gufo_attn_map(q, k, S, masks):
    """One attention over the first frame's keys with the masks as values
    (channel c = mask c): output channel c = masked probability mass."""
    qh = q[0].transpose(0, 1)  # [heads, T, 128]
    kh = k[0, :S].transpose(0, 1)  # [heads, S, 128]
    values = torch.zeros(HEADS, S, HEAD_DIM)
    values[:, :, 0] = masks[0].float()
    values[:, :, 1] = masks[1].float()
    out = F.scaled_dot_product_attention(qh, kh, values)  # [heads, T, 128]
    counts = masks.float().sum(1)
    return torch.stack([out[:, :, 0].mean(0) / (counts[0] + 1e-8),
                        out[:, :, 1].mean(0) / (counts[1] + 1e-8)])


def window_plan(total, window=81, motion=9):
    """talk21.cpp: (audio_start, kept frames) per window."""
    length = window if total >= window else 1 + 4 * ((max(total, 1) + 2) // 4)
    plan, produced, w = [], 0, 0
    while produced < total:
        start = 0 if w == 0 else produced - motion
        skip = 0 if w == 0 else motion
        keep = min(length - skip, total - produced)
        plan.append((start, keep))
        produced += keep
        w += 1
    return length, plan


# --------------------------------------------------------------------------
# wav2vec2-base (comfy/audio_encoders/wav2vec2.py, "base" config).
KERNELS = [10, 3, 3, 3, 3, 2, 2]
STRIDES = [5, 2, 2, 2, 2, 2, 2]


def w2v_reference(w, audio):
    x = audio[None, None]
    for i in range(7):
        x = F.conv1d(x, w[f"conv{i}"], stride=STRIDES[i])
        if i == 0:
            x = F.group_norm(x, 512, w["gn.weight"], w["gn.bias"])
        x = F.gelu(x)
    x = x.transpose(1, 2)
    x = F.layer_norm(x, (512,), w["fp_norm.weight"], w["fp_norm.bias"])
    x = x @ w["fp.weight"].T + w["fp.bias"]
    g, v = w["pos.g"], w["pos.v"]
    weight = g * v / v.norm(dim=(0, 1), keepdim=True)
    p = F.conv1d(x.transpose(1, 2), weight, w["pos.bias"], padding=64,
                 groups=16)[:, :, :-1]
    x = x + F.gelu(p).transpose(1, 2)
    x = F.layer_norm(x, (768,), w["enc_norm.weight"], w["enc_norm.bias"])
    outs = []
    for l in range(12):
        L = w["layers"][l]
        q = (x @ L["q.weight"].T + L["q.bias"]).view(1, -1, 12, 64)
        k = (x @ L["k.weight"].T + L["k.bias"]).view(1, -1, 12, 64)
        vv = (x @ L["v.weight"].T + L["v.bias"]).view(1, -1, 12, 64)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           vv.transpose(1, 2))
        a = a.transpose(1, 2).reshape(1, -1, 768) @ L["o.weight"].T + \
            L["o.bias"]
        x = F.layer_norm(x + a, (768,), L["ln.weight"], L["ln.bias"])
        ff = F.gelu(x @ L["up.weight"].T + L["up.bias"]) @ L["down.weight"].T \
            + L["down.bias"]
        x = F.layer_norm(x + ff, (768,), L["fln.weight"], L["fln.bias"])
        outs.append(x[0])
    return torch.stack(outs)  # [12, T, 768]


def w2v_gufo_convs(w, audio):
    """wav2vec2.hip's feature stack: convolutions 1..6 as a sum of strided
    GEMMs over the kernel taps (time-major [frames][512])."""
    x = F.conv1d(audio[None, None], w["conv0"], stride=5)
    x = F.gelu(F.group_norm(x, 512, w["gn.weight"], w["gn.bias"]))[0].T
    for i in range(1, 7):
        k, length = KERNELS[i], (x.shape[0] - KERNELS[i]) // 2 + 1
        out = torch.zeros(length, 512)
        for j in range(k):
            out += x[j:j + 2 * (length - 1) + 1:2] @ w[f"conv{i}"][:, :, j].T
        x = F.gelu(out)
    return x


def frames_for(samples):
    for k, s in zip(KERNELS, STRIDES):
        if samples < k:
            return 0
        samples = (samples - k) // s + 1
    return samples


def resample_25(layers):
    """WanInfiniteTalkToVideo linear_interpolation 50 -> 25 fps."""
    t = layers.shape[1]
    out = int(t / 50.0 * 25)
    return F.interpolate(layers.permute(0, 2, 1), size=out,
                         align_corners=True, mode="linear").permute(2, 0, 1)


# --------------------------------------------------------------------------
def rand(*shape, scale=0.02):
    return torch.randn(*shape) * scale


def random_proj():
    return {"proj1.weight": rand(512, 5 * 12 * 768), "proj1.bias": rand(512),
            "proj1_vf.weight": rand(512, 8 * 12 * 768),
            "proj1_vf.bias": rand(512),
            "proj2.weight": rand(512, 512, scale=0.05), "proj2.bias": rand(512),
            "proj3.weight": rand(TOKENS * AUDIO, 512, scale=0.05),
            "proj3.bias": rand(TOKENS * AUDIO),
            "norm.weight": 1 + rand(AUDIO), "norm.bias": rand(AUDIO)}


def random_block():
    return {"norm_x.weight": 1 + rand(DIM), "norm_x.bias": rand(DIM),
            "q.weight": rand(DIM, DIM, scale=0.01), "q.bias": rand(DIM),
            "kv.weight": rand(2 * DIM, AUDIO, scale=0.03),
            "kv.bias": rand(2 * DIM),
            "proj.weight": rand(DIM, DIM, scale=0.01), "proj.bias": rand(DIM)}


def load_talk(path):
    from safetensors import safe_open
    proj, blk = {}, {}
    with safe_open(path, "pt") as f:
        for k in random_proj():
            proj[k] = f.get_tensor("audio_proj." + k).float()
        names = {"norm_x.weight": "blocks.0.norm_x.weight",
                 "norm_x.bias": "blocks.0.norm_x.bias"}
        for short, full in (("q", "q_linear"), ("kv", "kv_linear"),
                            ("proj", "proj")):
            for kind in ("weight", "bias"):
                names[f"{short}.{kind}"] = \
                    f"blocks.0.audio_cross_attn.{full}.{kind}"
        for k, full in names.items():
            blk[k] = f.get_tensor(full).float()
    return proj, blk


def random_w2v():
    w = {"conv0": rand(512, 1, 10, scale=0.3), "gn.weight": 1 + rand(512),
         "gn.bias": rand(512)}
    for i in range(1, 7):
        w[f"conv{i}"] = rand(512, 512, KERNELS[i], scale=0.05)
    w.update({"fp_norm.weight": 1 + rand(512), "fp_norm.bias": rand(512),
              "fp.weight": rand(768, 512, scale=0.05), "fp.bias": rand(768),
              "pos.g": 1 + rand(1, 1, 128), "pos.v": rand(768, 48, 128),
              "pos.bias": rand(768), "enc_norm.weight": 1 + rand(768),
              "enc_norm.bias": rand(768), "layers": []})
    for _ in range(12):
        L = {}
        for n in ("q", "k", "v", "o"):
            L[f"{n}.weight"] = rand(768, 768, scale=0.04)
            L[f"{n}.bias"] = rand(768)
        L.update({"ln.weight": 1 + rand(768), "ln.bias": rand(768),
                  "up.weight": rand(3072, 768, scale=0.04),
                  "up.bias": rand(3072),
                  "down.weight": rand(768, 3072, scale=0.02),
                  "down.bias": rand(768), "fln.weight": 1 + rand(768),
                  "fln.bias": rand(768)})
        w["layers"].append(L)
    return w


def load_w2v(path):
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        keys = set(f.keys())
        pre = "wav2vec2." if "wav2vec2.encoder.layer_norm.bias" in keys else ""

        def t(name):
            return f.get_tensor(pre + name).float()
        fe = "feature_extractor.conv_layers."
        w = {"conv0": t(fe + "0.conv.weight"),
             "gn.weight": t(fe + "0.layer_norm.weight"),
             "gn.bias": t(fe + "0.layer_norm.bias")}
        for i in range(1, 7):
            w[f"conv{i}"] = t(fe + f"{i}.conv.weight")
        pc = "encoder.pos_conv_embed.conv."
        w.update({"fp_norm.weight": t("feature_projection.layer_norm.weight"),
                  "fp_norm.bias": t("feature_projection.layer_norm.bias"),
                  "fp.weight": t("feature_projection.projection.weight"),
                  "fp.bias": t("feature_projection.projection.bias"),
                  "pos.g": t(pc + "weight_g"), "pos.v": t(pc + "weight_v"),
                  "pos.bias": t(pc + "bias"),
                  "enc_norm.weight": t("encoder.layer_norm.weight"),
                  "enc_norm.bias": t("encoder.layer_norm.bias"),
                  "layers": []})
        for l in range(12):
            p = f"encoder.layers.{l}."
            L = {}
            for n, full in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"),
                            ("o", "out_proj")):
                L[f"{n}.weight"] = t(p + f"attention.{full}.weight")
                L[f"{n}.bias"] = t(p + f"attention.{full}.bias")
            L.update({"ln.weight": t(p + "layer_norm.weight"),
                      "ln.bias": t(p + "layer_norm.bias"),
                      "up.weight": t(p + "feed_forward.intermediate_dense.weight"),
                      "up.bias": t(p + "feed_forward.intermediate_dense.bias"),
                      "down.weight": t(p + "feed_forward.output_dense.weight"),
                      "down.bias": t(p + "feed_forward.output_dense.bias"),
                      "fln.weight": t(p + "final_layer_norm.weight"),
                      "fln.bias": t(p + "final_layer_norm.bias")})
            w["layers"].append(L)
    return w


def report(name, a, b, tol):
    err = (a - b).abs().max().item()
    scale = b.abs().max().item() + 1e-12
    ok = err <= tol * scale
    print(f"{'ok  ' if ok else 'FAIL'} {name}: max|diff| {err:.3e} "
          f"(rel {err / scale:.2e})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--talk-file", help="InfiniteTalk model patch")
    ap.add_argument("--wav2vec2-file", help="chinese-wav2vec2-base")
    ap.add_argument("--out", help="write fixtures (.npz)")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    ok = True

    # 1. Audio projection, two speakers, a continuation window.
    proj, blk = (load_talk(args.talk_file) if args.talk_file
                 else (random_proj(), random_block()))
    feature_frames = 30
    feats = [torch.randn(feature_frames, LAYERS_A, AUDIO) for _ in range(2)]
    start, length = 20, 13  # past the audio end: frame-0 padding
    ref = comfy_project(proj, feats, start, start + length)[0]
    mine = gufo_project(proj, feats, feature_frames, start, length)
    ok &= report("audio projection (gather + token layout)", mine, ref, 1e-5)

    # 2. Speaker routing map: masked softmax mass vs mask-valued attention.
    n_t, n_h, n_w = 3, 4, 6
    S, T = n_h * n_w, n_t * n_h * n_w
    q = torch.randn(1, T, HEADS, HEAD_DIM)
    k = torch.randn(1, T, HEADS, HEAD_DIM)
    masks = torch.zeros(2, S, dtype=torch.bool)
    masks[0, : S // 2] = True
    masks[1, S // 2:] = True
    ref_map = comfy_attn_map(q, k, (n_t, n_h, n_w), masks)
    ok &= report("x_ref_attn_map", gufo_attn_map(q, k, S, masks), ref_map,
                 1e-5)

    # 3. Audio cross-attention of one block (one and two speakers).
    x = torch.randn(T, DIM)
    audio1 = gufo_project(proj, feats[:1], feature_frames, 0, 9)
    out1 = comfy_audio_cross(blk, x, audio1, (n_t, n_h, n_w), None)
    audio2 = gufo_project(proj, feats, feature_frames, 0, 9)
    out2 = comfy_audio_cross(blk, x, audio2, (n_t, n_h, n_w), ref_map)
    print(f"info audio cross-attention outputs: |single| {out1.abs().mean():.4f}"
          f", |two| {out2.abs().mean():.4f}")

    # 4. wav2vec2: conv stack as strided GEMMs, then the whole encoder.
    w = load_w2v(args.wav2vec2_file) if args.wav2vec2_file else random_w2v()
    audio = torch.randn(16000) * 0.1  # 1 s
    conv_ref = w2v_reference_convs(w, audio)
    ok &= report("wav2vec2 conv stack (strided GEMMs)",
                 w2v_gufo_convs(w, audio), conv_ref, 1e-4)
    layers = w2v_reference(w, audio)
    assert layers.shape[1] == frames_for(16000), (layers.shape, frames_for(16000))
    feats25 = resample_25(layers)
    print(f"info wav2vec2: {layers.shape[1]} frames at 50 Hz -> "
          f"{feats25.shape[0]} at 25 fps")

    # 5. Window plan of a 6 s video.
    length, plan = window_plan(150)
    assert length == 81 and plan == [(0, 81), (72, 69)], plan
    length, plan = window_plan(30)
    assert length == 33 and plan == [(0, 30)], plan
    print("ok   window plan (81 frames, 9 motion frames)")

    if args.out:
        np.savez(args.out, tokens=mine.numpy(), feats0=feats[0].numpy(),
                 feats1=feats[1].numpy(), start=start, length=length,
                 q=q.numpy(), k=k.numpy(), masks=masks.numpy(),
                 ref_map=ref_map.numpy(), x=x.numpy(), out1=out1.numpy(),
                 out2=out2.numpy(), audio=audio.numpy(),
                 w2v_layers=layers.numpy(), w2v_25=feats25.numpy())
        print("wrote", args.out)
    return 0 if ok else 1


def w2v_reference_convs(w, audio):
    x = audio[None, None]
    for i in range(7):
        x = F.conv1d(x, w[f"conv{i}"], stride=STRIDES[i])
        if i == 0:
            x = F.group_norm(x, 512, w["gn.weight"], w["gn.bias"])
        x = F.gelu(x)
    return x[0].T


if __name__ == "__main__":
    sys.exit(main())
