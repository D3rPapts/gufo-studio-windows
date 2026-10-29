# RIFE frame interpolation

gufo runs RIFE 4.7 (the IFNet of Practical-RIFE, arch "4.7": checkpoints
`rife47.pth` and `rife49.pth`) natively on HIP to raise the frame rate of a
finished video, as ComfyUI-Frame-Interpolation's "RIFE VFI" node does after a
video generation.

The port follows the node with its defaults: frames zero-padded to multiples
of 64, `scale_list = [8, 4, 2, 1] / scale_factor`, no ensemble (the node hands
its `ensemble` value to IFNet's `fastmode` argument, so ensembling never runs),
timesteps `k / multiplier`, output clamped to [0, 1]. The original frames are
kept, so `n` input frames give `(n - 1) * multiplier + 1` output frames.

## Weights

The checkpoints are PyTorch pickles. Convert one to safetensors once; the
converter needs only the Python standard library and does not execute the
pickle (a restricted unpickler that knows only the state_dict globals):

```sh
python -I tools/rife/convert_rife_pth.py rife49.pth rife49.safetensors
```

ComfyUI-Frame-Interpolation downloads `rife49.pth` (its default) and
`rife47.pth` from its release mirrors, e.g.
`https://huggingface.co/marduk191/rife/resolve/main/rife49.pth`.

## CLI

```sh
gufo video-interpolate --model rife49.safetensors --input in.mp4 \
  --output out.mp4 --multiplier 4
```

FFmpeg decodes the input and encodes H.264 (CRF 18, yuv420p); `ffmpeg` and
`ffprobe` beside `gufo.exe` (or on `PATH`, or `GUFO_FFMPEG` / `GUFO_FFPROBE`)
are used. The output rate is the input rate times the multiplier (24 fps x4 =
96 fps, same speed) and the audio stream is copied. `--fps N[/D]` sets another
rate (slow or fast motion) and then drops the audio; `--no-audio` drops it
always. `--scale 0.5` is the node's `scale_factor` for 4K input.
`gufo-rife` is the same driver as a standalone executable.

Diagnostics: `--fp32` runs direct FP32 convolutions instead of FP16 WMMA,
`--bench N` times the first frame pair, `--reference DIR` compares against a
PyTorch dump (`manifest.json` with `{pair, t, file}` entries of FP32
`[h, w, 3]` outputs).

## Server

`gufo serve video --rife-model rife49.safetensors` (H3 or Wan 2.2) accepts

```json
{"model": "...", "prompt": "...", "gufo": {"interpolate": 4}}
```

on `POST /v1/videos` (JSON or the multipart `gufo` field). After the
generation finishes, the MP4 is interpolated (1 = off, 2..8) and
`/v1/videos/{id}/content` returns the interpolated video with the generated
audio. Progress reports phase `interpolate` at 98-99 %. Without
`--rife-model`, a request with `interpolate > 1` is rejected with 400.

## Implementation

`src/models/rife`: every convolution, including the transposed ones
(`ConvTranspose2d(k4, s2, p1)` = a 3x3 convolution with four times the rows
plus depth-to-space; with the following `PixelShuffle(2)` the rows scatter into
a 4x4 patch), runs through one implicit-GEMM 3x3 kernel (FP16 WMMA, FP32
accumulation, 64x64 tiles). ResConv's `beta` is folded into the weights and the
residual add and LeakyReLU into the epilogue. Warps (`grid_sample` bilinear,
border, align_corners), resizes, flow and mask stay FP32 at full resolution;
each IFBlock's input is gathered from full resolution in one kernel (warp +
bilinear downsample + flow / scale). The per-frame encoder features are
computed once per input frame.
