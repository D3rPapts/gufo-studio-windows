#!/usr/bin/env python3
"""Small CPU reference for MiniMax H3 guide clips (MiniMaxH3AddGuide with a
multi-frame image batch, as in ComfyUI comfy_extras/nodes_minimax_h3.py and
comfy/ldm/minimax/{model,vae}.py at ComfyUI 0dbfd49).

Written with AI assistance (Claude Code) for gufo.

It checks, with random weights and tiny shapes (well below 1 GB of RAM), the
three pieces gufo re-implements in C++/HIP:

1. Clip length and latent count: AddGuide keeps the first 17k + 5 frames of a
   batch (fewer than 5: the first frame only); the temporal VAE encode turns
   them into 5k + 2 latent frames (17-frame chunks, the last one padded by
   repeating its final frame, the last 3 latent frames dropped).
2. The causal 3D encoder as gufo runs it (video_vae_encoder.hip
   EncodeClipTile): frame-major activations, per-frame GroupNorm, and every
   3x3x3 causal convolution as a sum of per-temporal-tap 2D convolutions over
   input frames o * stride - 2 + kt (taps before frame 0 are the zero front
   padding). Compared against the reference's padded conv3d.
3. The packed layout: a guide clip at pixel frame f occupies 5k + 2 latent
   frames of condition rows on rotary time text + 5/3 * f + 5/3 *
   cumsum({1, 4, 4, 4, 4}), and its rows patchify frame by frame
   ([24][T][h][w] -> rows t, h/2, w/2, then channel, dh, dw).

Run: python -I tools/models/minimax_h3/guide_clip_reference.py
Needs torch (CPU is enough). Prints the values the C++ unit test
(tests/models/minimax_h3_sampling_test.cpp) hard-codes and exits non-zero on
a mismatch.
"""

import math
import sys

import torch
import torch.nn.functional as F

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
CLIP_LENGTH = 17
TOKEN_DROP = 3


# --- 1. clip length (nodes_minimax_h3.MiniMaxH3AddGuide.execute) -------------

def addguide_frames(batch):
    guide_frames = batch
    if guide_frames < 5:
        return 1
    while guide_frames % 17 != 5:
        guide_frames -= 1
    return guide_frames


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


# --- 2. causal encoder: reference vs gufo's per-tap formulation --------------

def ref_causal_conv(x, weight, bias, stride_t, stride_s, pad_t, spatial_pad):
    """comfy.ldm.minimax.vae.CausalConv3d.forward (non-kitchen path)."""
    if any(spatial_pad):
        x = F.pad(x, (*spatial_pad, 0, 0), mode="reflect")
    if x.shape[2] == 1:
        # single frame: the reference truncates the temporal taps
        # ("causal_zero"), i.e. only the last tap sees the frame
        return F.conv2d(x[:, :, 0], weight[:, :, -1], bias,
                        stride=stride_s).unsqueeze(2)
    front = pad_t * 2
    if front:
        x = F.pad(x, (0, 0, 0, 0, front, 0), mode="constant")
    return F.conv3d(x, weight, bias, stride=(stride_t, stride_s, stride_s))


def gufo_causal_conv(frames, weight, bias, stride_t, stride_s, spatial_pad):
    """ConvolveFrames: frames is a list of [C, H, W]; returns a list."""
    kernel_t = weight.shape[2]
    produced = (len(frames) - 1) // stride_t + 1
    out = []
    for o in range(produced):
        acc = None
        for kt in range(kernel_t):
            source = o * stride_t + kt - 2 if kernel_t == 3 else o
            if source < 0:
                continue  # zero front padding
            x = frames[source].unsqueeze(0)
            if any(spatial_pad):
                x = F.pad(x, spatial_pad, mode="reflect")
            y = F.conv2d(x, weight[:, :, kt], None, stride=stride_s)[0]
            acc = y if acc is None else acc + y
        out.append(acc + bias.view(-1, 1, 1))
    return out


def group_norm_silu(x, gamma, beta, groups=4):
    # TemporalIsolatedGroupNorm: statistics per frame, then SiLU
    b, c, t, h, w = x.shape
    y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    y = F.group_norm(y, groups, gamma, beta, eps=1e-6)
    y = y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return F.silu(y)


def make_encoder(gen, channels=(8, 16, 16, 32), z=6):
    """A shrunken EncoderFCN3D: conv_in, per level one resnet (+ shortcut
    when channels change) and a downsample with the reference's time/space
    strides (levels 0..3 of time_down (1, 2, 2, 1), space 2), conv_out."""
    def conv(cin, cout, k):
        return (torch.randn(cout, cin, k, k, k, generator=gen) / math.sqrt(cin * k ** 3),
                torch.randn(cout, generator=gen) * 0.1)

    def norm(c):
        return (1 + 0.1 * torch.randn(c, generator=gen),
                0.1 * torch.randn(c, generator=gen))

    net = {"conv_in": conv(3, channels[0], 3), "levels": []}
    cin = channels[0]
    for level, cout in enumerate(channels):
        net["levels"].append({
            "norm1": norm(cin), "conv1": conv(cin, cout, 3),
            "norm2": norm(cout), "conv2": conv(cout, cout, 3),
            "shortcut": conv(cin, cout, 1) if cin != cout else None,
            "down": conv(cout, cout, 3),
            "time_stride": (1, 2, 2, 1)[level],
        })
        cin = cout
    net["norm_out"] = norm(cin)
    net["conv_out"] = conv(cin, 2 * z, 3)
    return net


def ref_encode(net, x):
    """x: [1, 3, T, H, W] -> moments [1, 2z, T', H', W']."""
    w, b = net["conv_in"]
    h = ref_causal_conv(x, w, b, 1, 1, 1, (1, 1, 1, 1))
    for lv in net["levels"]:
        a = group_norm_silu(h, *lv["norm1"])
        a = ref_causal_conv(a, *lv["conv1"], 1, 1, 1, (1, 1, 1, 1))
        a = group_norm_silu(a, *lv["norm2"])
        a = ref_causal_conv(a, *lv["conv2"], 1, 1, 1, (1, 1, 1, 1))
        if lv["shortcut"] is not None:
            h = ref_causal_conv(h, *lv["shortcut"], 1, 1, 0, (0, 0, 0, 0))
        h = h + a
        h = ref_causal_conv(h, *lv["down"], lv["time_stride"], 2, 1,
                            (0, 1, 0, 1))
    a = group_norm_silu(h, *net["norm_out"])
    return ref_causal_conv(a, *net["conv_out"], 1, 1, 1, (1, 1, 1, 1))


def gufo_encode(net, frames):
    """frames: list of [3, H, W] -> list of [2z, H', W'] (frame-major)."""
    def norm_frames(fs, gamma, beta):
        return [group_norm_silu(f[None, :, None], gamma, beta)[0, :, 0]
                for f in fs]

    h = gufo_causal_conv(frames, *net["conv_in"], 1, 1, (1, 1, 1, 1))
    for lv in net["levels"]:
        a = norm_frames(h, *lv["norm1"])
        a = gufo_causal_conv(a, *lv["conv1"], 1, 1, (1, 1, 1, 1))
        a = norm_frames(a, *lv["norm2"])
        a = gufo_causal_conv(a, *lv["conv2"], 1, 1, (1, 1, 1, 1))
        if lv["shortcut"] is not None:
            h = gufo_causal_conv(h, *lv["shortcut"], 1, 1, (0, 0, 0, 0))
        h = [p + q for p, q in zip(h, a)]
        h = gufo_causal_conv(h, *lv["down"], lv["time_stride"], 2,
                             (0, 1, 0, 1))
    a = norm_frames(h, *net["norm_out"])
    return gufo_causal_conv(a, *net["conv_out"], 1, 1, (1, 1, 1, 1))


def ref_encode_temporal(net, x):
    """MiniMaxH3VideoVAE.encode_temporal (without spatial tiling)."""
    zs = []
    for i in range(math.ceil(x.shape[2] / CLIP_LENGTH)):
        clip = x[:, :, i * CLIP_LENGTH:(i + 1) * CLIP_LENGTH]
        if clip.shape[2] < CLIP_LENGTH:
            pad = clip[:, :, -1:].repeat(1, 1, CLIP_LENGTH - clip.shape[2], 1, 1)
            clip = torch.cat([clip, pad], dim=2)
        zs.append(ref_encode(net, clip))
    z = torch.cat(zs, dim=2)
    return z[:, :, :-TOKEN_DROP]


def gufo_encode_clip(net, frames):
    """EncodeGuideClipMoments: same chunking on the frame list."""
    out = []
    chunks = (len(frames) + CLIP_LENGTH - 1) // CLIP_LENGTH
    for chunk in range(chunks):
        first = chunk * CLIP_LENGTH
        clip = [frames[min(first + i, len(frames) - 1)]
                for i in range(CLIP_LENGTH)]
        out.extend(gufo_encode(net, clip))
    return out[:-TOKEN_DROP]


# --- 3. layout (comfy.ldm.minimax.model.PackedLayout keyframes branch) -------

def video_t_grid(n, origin):
    spans = [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]
    out, t = [], float(origin)
    for s in spans:
        out.append(t)
        t += s
    return out


def gufo_clip_times(text_rows, frame, latent_frames):
    """sampling.cpp BuildImageGuideLayout for one guide."""
    time = text_rows + FRAME_RESCALE * frame
    out = []
    for index in range(latent_frames):
        out.append(time)
        time += FRAME_RESCALE * FRAME_PER_TOKEN[index % 5]
    return out


def patchify_ref(latent):
    """comfy patchify_video for patch (1, 2, 2): [1,C,T,H,W] -> rows."""
    b, c, t, h, w = latent.shape
    x = latent.reshape(b, c, t, h // 2, 2, w // 2, 2)
    x = x.permute(0, 2, 3, 5, 1, 4, 6)  # b t h w c dh dw
    return x.reshape(t * (h // 2) * (w // 2), c * 4)


def patchify_gufo(latent):
    """sampling.cpp PatchifyVideo on [C][T][H][W]."""
    c, t, h, w = latent.shape
    rows = []
    for ti in range(t):
        for hi in range(0, h, 2):
            for wi in range(0, w, 2):
                row = []
                for ch in range(c):
                    for dh in range(2):
                        for dw in range(2):
                            row.append(latent[ch, ti, hi + dh, wi + dw])
                rows.append(torch.stack(row))
    return torch.stack(rows)


def main():
    failures = 0

    def check(name, ok):
        nonlocal failures
        print(("ok   " if ok else "FAIL ") + name)
        failures += 0 if ok else 1

    # 1. lengths
    for batch, want in ((1, 1), (4, 1), (5, 5), (21, 5), (22, 22), (48, 39)):
        check("AddGuide keeps %d of %d frames" % (want, batch),
              addguide_frames(batch) == want)
    for frames, want in ((5, 2), (22, 7), (39, 12)):
        check("%d frames -> %d latent frames" % (frames, want),
              video_latent_t(frames) == want)

    # 2. encoder decomposition and chunking (random weights, 32x32, 22 frames)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1234)
    net = make_encoder(gen)
    with torch.no_grad():
        x = torch.randn(1, 3, CLIP_LENGTH, 32, 32, generator=gen)
        ref = ref_encode(net, x)
        mine = torch.stack(gufo_encode(net, list(x[0].unbind(1))), dim=1)[None]
        check("17-frame chunk -> %d latent frames" % ref.shape[2],
              ref.shape[2] == 5 and mine.shape == ref.shape)
        check("causal 3D conv == per-tap 2D sum (max diff %.2e)"
              % (ref - mine).abs().max().item(),
              torch.allclose(ref, mine, atol=1e-4, rtol=1e-4))
        single = x[:, :, :1]
        ref1 = ref_encode(net, single)
        mine1 = torch.stack(gufo_encode(net, [single[0, :, 0]]), dim=1)[None]
        check("single frame == last temporal tap only",
              torch.allclose(ref1, mine1, atol=1e-4, rtol=1e-4))
        # the first latent of a clip equals the single-frame encode of its
        # first frame (causality), as the H3 VAE is built for
        check("clip latent 0 == single-frame latent of frame 0",
              torch.allclose(ref[:, :, :1], ref1, atol=1e-4, rtol=1e-4))
        clip = torch.randn(1, 3, 22, 32, 32, generator=gen)
        ref_clip = ref_encode_temporal(net, clip)
        mine_clip = torch.stack(gufo_encode_clip(net, list(clip[0].unbind(1))),
                                dim=1)[None]
        check("22-frame clip -> %d latent frames" % ref_clip.shape[2],
              ref_clip.shape[2] == 7 and mine_clip.shape == ref_clip.shape)
        check("clip chunking matches (max diff %.2e)"
              % (ref_clip - mine_clip).abs().max().item(),
              torch.allclose(ref_clip, mine_clip, atol=1e-4, rtol=1e-4))

    # 3. layout and packing
    text_rows = 6
    for frame, latents in ((0, 7), (17, 7), (60, 1)):
        ref_t = video_t_grid(latents, text_rows + FRAME_RESCALE * frame)
        mine_t = gufo_clip_times(text_rows, frame, latents)
        print("     guide at frame %d: rotary times %s" % (
            frame, ", ".join("%.6f" % t for t in ref_t)))
        check("guide times at frame %d" % frame,
              all(abs(a - b) < 1e-9 for a, b in zip(ref_t, mine_t)))
    target = video_t_grid(7, text_rows)
    check("clip at frame 0 shares the target's first latent times",
          all(abs(a - b) < 1e-9
              for a, b in zip(target, gufo_clip_times(text_rows, 0, 7))))
    latent = torch.randn(1, 24, 3, 4, 6, generator=gen)
    check("patchify order of a clip latent",
          torch.equal(patchify_ref(latent), patchify_gufo(latent[0])))

    if failures:
        print("%d check(s) failed" % failures)
        return 1
    print("MiniMax H3 guide clip reference: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
