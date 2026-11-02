# Ming-3.0-tts quality

## Reference agreement (CPU runtime)

Reference: the official Python implementation (`inclusionAI/Ming-3.0-tts`,
commit `9332fb94`, transformers 4.46.3, torch 2.13 CPU, FP32, SDPA attention —
the same flash-attention downgrade `MingTTS` applies), driven by
`src/models/ming_tts/reference/ming_oracle.py` for one German EMOTION request
(`Guten Morgen, heute scheint die Sonne.`, label 高兴, 24 kHz reference voice,
seed 1234, 33 AR steps). `tests/models/ming_tts/oracle_test.cpp` compares every
stage; the gate is cosine ≥ 0.999. Measured 2026-11-02 on the Windows CPU
build (`windows-cpu-test`):

| Stage | Cosine | Max abs. error |
| --- | --- | --- |
| Tokenizer (7 cases incl. digits, umlauts, JSON caption), prompt ids (155), 7 emotion captions, text segmentation (4 cases) | exact | – |
| torchaudio-compatible resample 24 → 16 kHz | 1.0000000 | 1.2e-7 |
| Kaldi fbank (80 mel) | 1.0000000 | 3.4e-4 (of 15.9) |
| CAM++ embedding vs. ONNX Runtime | 1.0000000 | 8.3e-6 |
| Backbone prefill, layer 0 / layer 11 / final norm | 1.0000000 | 2.9e-4 / 0.10 (of 7.4e4) / 1.1e-4 |
| DiT velocity (10 evaluations, CFG pair) | 1.0000000 | ≤ 2.2e-6 |
| Teacher-forced AR, 33 steps: backbone condition, sampled latents, aggregator, stop probability | 1.0000000 | 2.1e-5 / 3.2e-6 / 7.6e-6 / 5.6e-9 |
| VAE decoder: fc1, upsample, layer 0, final hidden, ISTFT head, waveform | 1.0000000 | ≤ 1.6e-5 |
| VAE probe, 300 frames (checks the 64-key sliding window) | 1.0000000 | 5e-11 |
| ZERO_SHOT oracle (`ming_oracle_zs.py`, transcript + 116 prompt latents): resample 24 → 44.1 kHz, VAE-encoder posterior, sampled latents, 219 prompt ids, inputs (incl. aggregated latents), prefill, 4 teacher-forced steps | 1.0000000 / exact ids | ≤ 7.4e-5 |
| Trailing-silence trim length | exact (110250) | – |
| Free-running AR with the oracle's start noise: step count | exact (33) | – |
| Free-running latents (all steps) | 0.9997 | 0.17 |
| Free-running waveform RMS envelope | 0.99992 | – |

Free running amplifies FP32 summation-order differences through the
autoregressive loop (per-step latent cosine stays ≥ 0.9997); the raw waveform
cosine (0.986) is phase-sensitive and only reported. The reference itself runs
in BF16 autocast on the GPU, which differs from this FP32 oracle by more.

## Not measured yet

- Listening comparison of the CPU runtime against the Python BF16 output.
- GPU runtime (not implemented).
- Word-error rate / speaker similarity on a test set.
