# Ming-3.0-tts benchmarks

Scope: CPU reference runtime only (FP32 math, BF16 weights, AVX2 kernels,
OpenMP), Windows `windows-cpu-test` build on the AMD Strix Halo host while a
GPU job was running in parallel. Numbers varied by up to 1.5x between runs on
2026-11-02; treat them as order-of-magnitude.

| Measurement (`OMP_NUM_THREADS=16`) | Value |
| --- | --- |
| One AR step (10 DiT evaluations + aggregator + one backbone token) | 130–180 ms (80 ms of audio) |
| DiT velocity evaluation (CFG pair, 3 tokens) | 10–15 ms |
| Backbone prefill, 155 prompt tokens | 0.41–0.58 s |
| VAE decode, 33 latents (2.6 s audio) | 0.47 s |
| Free-running request, 2.5 s audio | 4.6–6.0 s (RTF 1.8–2.4) |
| In-process `/v1/audio/speech`, 2.9–3.8 s audio | 6.0–8.0 s |

The DiT head reads about 200 MB of BF16 weights per evaluation, i.e. about
2 GB per AR step, so the CPU path stays memory-bound above real time. TODO: GPU
runtime numbers.
