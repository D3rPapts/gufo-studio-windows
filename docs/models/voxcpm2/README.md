# VoxCPM2 (OpenBMB) in gufo

Native C++ port of [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (Apache-2.0, voxcpm 2.0.3 as reference),
served by `gufo serve tts --model <VoxCPM2 dir>`. No Python, no ONNX Runtime.

*Written with AI assistance (Claude Code).*

## Model

- Text-semantic LM: MiniCPM-4, 28 layers, hidden 2048, 16 heads / 2 KV heads, head dim 128, LongRoPE
  (short factors, scaling 1.0), muP off in the shipped config.
- Residual acoustic LM: 8 layers, same shape, no RoPE.
- Local encoder (12 layers, 1024) turns a patch of 4 latents into one LM input; local DiT (12 layers, 1024)
  samples the next patch with UnifiedCFM (Euler, 10 steps, sway sampling, CFG 2.0 with CFG-zero-star,
  first step zero).
- FSQ (512 dims, scale 9) on audio positions of the LM output; stop head on the LM hidden state.
- AudioVAE V2: causal convolutions with Snake, depthwise residual units; encoder 16 kHz -> 25 Hz latents (64),
  decoder 25 Hz -> 48 kHz with sample-rate conditioning (bucket for 48 kHz).

## Request modes (`/v1/audio/speech`)

| Request | VoxCPM2 mode |
|---|---|
| no voice audio (`voice: "default"`) | zero-shot (random voice) |
| preset voice or `reference_audio` | reference (isolated prefix `[103] ref [104]`), "controllable cloning" |
| `reference_audio` + `reference_text` | reference + continuation prompt with the same clip ("ultimate cloning") |
| `emotion` and/or `instructions` | free text, sent as `"(emotion, instructions)"` in front of the input |
| `language` | accepted, ignored (VoxCPM2 reads the language from the text) |
| `response_format: "pcm"` | 160 ms chunks (one AR patch each), streamed while generating |

Reference clips longer than 60 s are rejected with 400. The bad-case retry of voxcpm (more than 6 patches per
text token -> new attempt, up to 3) is on for WAV responses and off for streaming, like the Python package.

## Runtimes

- CPU reference: FP32 math on the memory-mapped BF16 checkpoint; reuses the Ming-3.0-tts GEMV/attention kernels.
- HIP (gfx1151): both LMs with device KV caches, local encoder, the complete flow sampler on the device (HIP graph),
  FSQ/fusion/stop heads. Text embedding and the AudioVAE stay on the CPU.
  `GUFO_VOXCPM_BACKEND=cpu|hip`, `GUFO_VOXCPM_GRAPHS=0`.

## Tests

- `voxcpm2_oracle_test` (CPU, stage by stage against `src/models/voxcpm2/reference/voxcpm2_oracle.py`):
  `GUFO_VOXCPM_MODEL`, `GUFO_VOXCPM_REF`, optional `GUFO_VOXCPM_STAGES`.
- `voxcpm2_serve_test` (in-process `/v1/audio/speech`): `GUFO_VOXCPM_MODEL`, `GUFO_VOXCPM_VOICE`.
- `voxcpm2_hip_test` (GPU against oracle and CPU): same variables as the oracle test.
