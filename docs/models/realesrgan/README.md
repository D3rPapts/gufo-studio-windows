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

## Images on the NPU, GPU or CPU (`gufo serve upscale`)

```
gufo serve upscale --device npu --model realesr-animevideov3=body896q_h288_h3.onnx \
                   [--model realesr-animevideov3=body512q_h280_h3.onnx] \
                   [--npu-cache DIR] --port 8095
```

- `--device npu` runs a Real-ESRGAN **body** ONNX (the convolutions before
  the pixel shuffle, int8 QDQ from AMD Quark, fixed input size) on the AMD
  NPU through ONNX Runtime and the VitisAI execution provider of Windows ML.
  gufo loads `onnxruntime.dll` at run time from the Windows ML framework
  package (`--onnxruntime` overrides) and asks the Windows ML catalog for the
  provider (`--npu-ep` overrides); it never ships either. The first start
  compiles the model into `--npu-cache` (default `<model dir>/npu-cache`,
  cache key = file stem); later starts read the cache.
- `--device cpu` runs the same file on the ONNX Runtime CPU provider
  (`--threads N`). The int8 graph is deterministic: on .198 the CPU result
  of Take One's test frames equals Take One's NPU output pixel for pixel.
- `--device gpu` uses the HIP upscaler above with the converted safetensors
  (`--fp32` for the validation path).

The NPU / CPU path reproduces Take One's `npu_upscale_worker.py`: RGB /255,
two row passes (rows 0..T-1 keep the top half, rows h-T..h-1 the bottom half),
DepthToSpace(4) plus the nearest-neighbour input, clip, `uint8(v*255+0.5)`,
then OpenCV `INTER_AREA` (bit-exact port) and a centre crop: 896x512 ->
rows 16..2031 of the x4 picture -> 1920x1080; 832x512 -> 1950x1200 ->
columns 15..1934 -> 1920x1200 (`fit: "crop"`). `pad` and `stretch` use the
same area filter. One model name may have several files, one per frame
width; the request's frame picks the file.

`POST /v1/images/upscale`

```json
{"model": "realesr-animevideov3", "image": "data:image/png;base64,...",
 "size": "1920x1080", "fit": "crop", "response_format": "b64_json"}
```

returns `{created, model, device, model_file, width, height, fit,
data: [{b64_json}], cost: {seconds, queue_seconds, body_seconds,
post_seconds, png_seconds}}` and a `Server-Timing` header;
`"response_format": "png"` returns the PNG itself. Without `size` (or
`width`/`height`) the x4 picture is returned. Requests run one at a time;
more than 8 waiting give 429. As a router section (the router forwards
`/v1/images/upscale` by the body's `model`, here the section id):

```ini
[external/realesr-npu]
command = D:\path\to\gufo.exe
args = serve upscale --device npu
model = D:\path\to\body896q_h288_h3.onnx
served-model-name = realesr-npu
```

`npu-upscale-probe` (built with `GUFO_ENABLE_ONNXRUNTIME`) runs one image
and compares it with a reference: `npu-upscale-probe --device cpu --model
body896q_h288_h3.onnx --input in.png --size 1920x1080 --compare ref.png`.

## Implementation

`src/models/realesrgan`: every convolution runs through the implicit-GEMM
3x3 kernel of the RIFE port (FP16 WMMA, FP32 accumulation, 64x64 tiles) with
the per-channel PReLU in its epilogue; the last convolution writes FP32. The
pixel shuffle, the nearest-neighbour base, the clamp and the two separable
resize passes are FP32 kernels; the result is downloaded as RGB24.
