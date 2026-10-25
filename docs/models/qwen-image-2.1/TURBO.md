# Qwen-Image-2.1-Turbo generation

This fork adds native BF16 Turbo **generation** on gfx1151. Reference-image
editing remains available with the base checkpoint; Turbo editing is deferred
and returns HTTP 400. This work tracks upstream [issue #689](https://github.com/gufo-org/gufo/issues/689).

## Pinned model contract

Download the complete pipeline (approximately 33 GB, not bundled):

```sh
hf download Qwen/Qwen-Image-2.1-Turbo \
  --revision fc13ba1986fcb905854dc8de2f52eee395f49b5b \
  --exclude 'assets/*' --local-dir models/Qwen-Image-2.1-Turbo
```

The transformer architecture, prompt encoder and RGBA VAE use the existing
native BF16 operators. The trained Turbo transformer is different from the base
transformer. Its pipeline metadata supplies these sampling sigmas:

```text
1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568
```

The scheduler has `shift=1`, `use_dynamic_shifting=false`, and
`shift_terminal=null`. Gufo converts the supplied values to FP32, appends zero,
and performs eight Euler updates without the base checkpoint's dynamic shift or
terminal stretching. Invalid or inconsistent sampling metadata is rejected.
Model identity comes from its configuration, not the directory or served name.

## API

```sh
gufo serve --host 127.0.0.1 --port 28182 image \
  --model models/Qwen-Image-2.1-Turbo --served-model-name Qwen-Image-2.1-Turbo
curl --fail-with-body http://127.0.0.1:28182/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen-Image-2.1-Turbo","prompt":"A red ceramic teapot on a wooden table", "size":"1024x1024","seed":42}'
```

Omit `steps` or pass `8`. Other step counts return HTTP 400 rather than silently
using an unqualified schedule. The base model retains its default 40 steps and
2–100 range. Seeds, PNG/base64 responses, serial independent outputs, dimensions
and background settings follow the [existing image API](README.md). There is no
custom-sigma HTTP parameter. Gufo and PyTorch use different RNGs; compare actual
initial noise tensors, not equal numeric seeds.

1920×1080 is not divisible by 32: request 1920×1088 and center-crop four pixels
from the top and bottom for an exact 1920×1080 deliverable.

The separate model weights retain the non-commercial Qwen Research License.
The fork's code retains Gufo's license. Turbo quality and speed must be measured
separately from the original model; eight steps on the original weights are not
Turbo.

## Qualification

Use the model's existing CPU tests, GPU operator tests and native probe. The
probe and independent reference accept `--width`/`--height` for rectangular
trajectories. Use a Diffusers revision with checkpoint-configured
`sample_sigmas`, pinned to `aaf57d03a3edfbd9de457d2872417a92f149a91d` for this
qualification, alongside recorded Torch/Transformers versions. Run eight steps
with shared initial noise, exact schedule comparison, matched-block numerical
checks and complete decoded trajectories. Keep tensor dumps and private
deployment logs outside Git.

Deployment instructions and measured results are added only after qualification.
