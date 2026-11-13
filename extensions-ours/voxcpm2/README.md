# VoxCPM2 extension (gufo extension API 0.2)

Out-of-tree `speech_synthesis` provider for OpenBMB VoxCPM2 (model ID
`voxcpm2`, 48 kHz). Depends on the `ming_tts` extension (shared safetensors
reader, CPU/HIP kernels, resampler). Ported from the gufo-studio-windows 1.x
fork. Written with AI assistance (Claude Code).

Build: `-DGUFO_EXTENSION_DIRS=<folder that holds ming_tts/ and voxcpm2/>`.
Serve: `gufo serve tts --model <VoxCPM2>`.

## Request mapping

- `instructions`: free-text style or emotion ("froehlich", "whispering ..."),
  sent as the `(instruction)text` prefix. The 1.x `emotion` field is ignored
  by the 2.0 core; put it into `instructions`.
- `reference_audio` + `reference_text`: "ultimate cloning" (reference plus
  continuation prompt).
- `reference_audio` with `"voice_clone_mode": "speaker_embedding_only"`:
  reference mode (the 2.0 core requires a transcript otherwise).
- No reference: zero-shot voice. Reference audio is limited to 60 s.
- `language` is ignored (detected from the text).

## Tests

- `voxcpm2.request` (CPU, no checkpoint)
- `voxcpm2.oracle` (CPU, `GUFO_VOXCPM_MODEL` + `GUFO_VOXCPM_REF`)
- `voxcpm2.provider` (`GUFO_VOXCPM_MODEL` + `GUFO_VOXCPM_VOICE`)
- `voxcpm2.hip` (GPU)
