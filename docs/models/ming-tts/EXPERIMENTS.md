# Ming-3.0-tts experiments

- Retained: CAM++ ported natively and read from `campplus.onnx` (minimal
  protobuf reader, weights mapped by node output names) instead of loading
  ONNX Runtime; matches ONNX Runtime at cosine 1.0000000.
- Retained: ISTFT via a double-precision mixed-radix FFT (n = 3528 = 2³·3²·7²)
  instead of a DFT matrix; exact against `torch.fft.irfft`.
- Retained: the VAE decoder uses the transformers SDPA sliding-window mask
  (64 keys, `j > i - 64`), which is what the reference runs after its
  flash-attention downgrade; flash-attention would allow 65 keys.
- Retained: a 6-row x 2-column AVX2 tile for the 6-row DiT GEMMs (about 20 %
  faster DiT evaluation on the CPU, measured noisily).
- Rejected: serial attention for short sequences (no reproducible gain).
