# Real-ESRGAN upscaling

gufo runs Real-ESRGAN's compact video models (`SRVGGNetCompact` from
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), BSD 3-Clause) natively
on HIP to bring finished videos and images to a delivery size such as
1920x1200 or 1920x1080.

Supported checkpoints (Real-ESRGAN release v0.2.5.0, published in the
BSD-3-Clause repository without a separate model license):

| Checkpoint | Body convolutions | Use |
|---|---|---|
| `realesr-animevideov3.pth` | 16 | anime / drawn video (default choice) |
| `realesr-general-x4v3.pth` | 32 | general scenes |
| `realesr-general-wdn-x4v3.pth` | 32 | general, stronger denoising |

All are x4 models with 64 features and PReLU activations. The network follows
`realesrgan/archs/srvgg_arch.py`: 3x3 convolutions, PReLU, a final convolution
to `3 * 4 * 4` channels, `PixelShuffle(4)` plus the nearest-neighbour
upsampled input, clamped to [0, 1] (RGB in [0, 1] as `RealESRGANer` feeds it).
The x4 picture is then resized to the requested size with an antialiased
Lanczos-3 filter.

## Weights

Convert a checkpoint once; the converter needs only the Python standard
library and reads the pickle with the restricted unpickler of the RIFE
converter (no code from the checkpoint runs):

```sh
python -I tools/upscale/convert_realesrgan_pth.py realesr-animevideov3.pth \
  realesr-animevideov3.safetensors
```

## CLI

```sh
gufo video-upscale --model realesr-animevideov3.safetensors \
  --input in.mp4 --output out.mp4 --size 1920x1200
```

`--fit pad` (default) keeps the aspect ratio and the whole picture with black
bars (832x480 -> picture 1920x1108 at y 46 in 1920x1200); `--fit crop` keeps
the aspect ratio and fills the frame (centre crop); `--fit stretch` scales each
axis on its own. Picture sizes and offsets are even, so 4:2:0 chroma stays
aligned. The output keeps the input frame rate and copies the audio
(`--no-audio` drops it); `--crf N` sets the H.264 quality (default 18). An
output ending in `.png` writes the first frame as an image, so
`--input key.png --output key-1920.png` upscales a still.

Frames run through the network in tiles of at most `--tile` input pixels
(default 512 Ki, so 832x480 is one tile). Each tile carries a halo of one
pixel per convolution, so tiling does not change the result.
`gufo-upscale` is the same driver as a standalone executable.

Diagnostics: `--fp32` runs direct FP32 convolutions instead of FP16 WMMA,
`--bench N` times the first frame, `--reference DIR` compares the network
output against a PyTorch dump (`manifest.json` with `{frame, file}` entries
of FP32 `[4h, 4w, 3]` outputs).

## Server

`gufo serve video --upscale-model [NAME=]realesr-animevideov3.safetensors`
(repeatable; the name defaults to the file stem, the first is the default)
accepts

```json
{"gufo": {"interpolate": {"fps": 60},
          "upscale": {"width": 1920, "height": 1200, "fit": "pad",
                      "model": "realesr-animevideov3"}}}
```

on `POST /v1/videos` (H3 and Wan 2.2). `"upscale": "1920x1200"` is short for
the default model with `pad`. Width and height must be even (at most
7680x4320). The upscale runs after any interpolation (RIFE at the generated
size, then every output frame upscaled) and reports progress phase `upscale`.
Without `--upscale-model` the request is rejected with 400.

## Implementation

`src/models/realesrgan`: every convolution runs through the implicit-GEMM
3x3 kernel of the RIFE port (FP16 WMMA, FP32 accumulation, 64x64 tiles) with
the per-channel PReLU in its epilogue; the last convolution writes FP32. The
pixel shuffle, the nearest-neighbour base, the clamp and the two separable
resize passes are FP32 kernels; the result is downloaded as RGB24.
