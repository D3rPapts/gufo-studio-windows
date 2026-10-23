# verify-sweep: tune verification launch shapes for a Qwen GGUF

During speculative decoding the target model verifies 2-8 draft rows at a
time. Each projection then runs `SmallBatchKQuantExactFp32GEMMKernel`, and how
its work is split over waves (waves per block, rows per wave, 16- or 32-lane
groups, staging tiles) decides how close the GPU gets to its memory
bandwidth. `src/models/qwen/hip/kernels/verify_gemm_27b.hip` holds the shapes
measured faster on Windows for Qwen3.8-27B UD-Q4_K_XL; any projection it does
not list falls back to the generic dispatch, which for some formats and row
counts reads at half the speed.

A different GGUF of the model, a different quantization mix or a different
draft setting changes which projections and row counts matter, so the shapes
have to be measured again. This tool does that in a few minutes.

## Use

```powershell
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1        # build\release must exist
# stop every gufo / llama.cpp server: the GPU has to be idle
powershell -ExecutionPolicy Bypass -File tools\windows\verify-sweep\run.ps1 -Model D:\models\Qwen3.8-27B-UD-Q4_K_XL.gguf
```

`run.ps1` builds `build\verify-sweep\gufo-verify-sweep.exe` against
`build\release`, lists the model's projections with `gguf_tensors.py`, sweeps
every projection used by at least `-MinLayers` layers at `-Batches` rows, and
prints a table plus the winning shapes as `Launch<...>` lines. Options:
`-Batches 3,4,5`, `-Rounds 9`, `-MinGain 5` (percent), `-Out sweep.log` for
the raw lines. For a DFlash2 draft GGUF pass `-MinLayers 1`: the draft has
only five layers, so some projections occur once. Its shapes go into
`LaunchQ4K`, the draft block of `verify_gemm_27b.hip`.

## Method

- Weights are the real tensors from the GGUF (one example layer per
  projection type and shape), read at their file offset.
- Production is what `LaunchProjection` runs: `TryLaunchVerifyGemm27`, then
  `LaunchBatchedQuantGEMMFp32`. Every candidate's output is compared bit for
  bit with production; inexact candidates are reported and never timed.
- Each launch uses another copy of the weights, enough copies for at least
  512 MiB, so no run reads from the 32 MB Infinity Cache of the Strix Halo.
  Without that, matrices under ~35 MB look faster than the memory allows.
- Production and candidates alternate within each round; the result is the
  median over the rounds, 20 launches per sample after 3 warm-up launches.

## Taking a result over

Add the shape to `verify_gemm_27b.hip` next to the existing ones (the
`LaunchQ4KTarget`, `LaunchQ6KTarget` and `LaunchFfnShape` blocks show the
pattern), keyed by type, `m`, `k` and row count. Check the same shape is not
also used by the DFlash2 draft at that row count with an entry that should
stay. Then rebuild, confirm greedy output is unchanged end to end (for
example identical `[TokenTrace] sha256` from `gufo prompt -v -t 0` before and
after) and measure a real workload: a per-launch gain only matters in
proportion to the projection's share of the cycle (`GUFO_QWEN27_PROFILE=2`).

## Limits

- Only the Qwen GPU path (`src/models/qwen/hip`: Qwen3.8-27B and the DFlash2
  drafts) and the formats Q4_K, Q5_K, Q6_K and IQ4_XS. Qwen3.8-Flash-Next,
  Qwen3.6-35B-A3B and DeepSeek have their own kernels.
- The candidate set is the neighbourhood of shapes picked so far, not an
  exhaustive search.
- Measured on Windows with TheRock 10.0.0 on gfx1151; the same shapes are
  likely, but not shown, to help on Linux.
