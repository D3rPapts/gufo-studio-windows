# Windows fork on current upstream (October 2026)

This branch carries the Windows fork (pixmaate/gufo `main`, 41 commits) onto
current upstream gufo-org/gufo (`7c81965`, "add the net leaf (#711)"). The
older fork history stays untouched on `main`; nothing here rewrites it.

## Contributor

The rebase, the fixes it needed, the measurements and this document were done
by **Claude (Anthropic, model Claude Opus 5.5) working in Claude Code**, at
the request of and supervised by the repository owner. Every commit Claude
wrote or adapted carries the trailer
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; commits authored
`local build <build@local>` were made by Claude on the build machine.

## What happened to the 41 fork commits

| Outcome | Count | Notes |
| --- | --- | --- |
| Taken over (cherry-picked, conflicts resolved) | 27 | POSIX layer, scripts, tests, Qwen3.8-27B decode kernels, Flash-Next work, A3B runtime, launchers |
| Adapted to upstream's new structure | 9 | weight upload, tuning switches, A3B text runner, `--logit-eval`, server "off" values, Flash-Next IQ4_XS |
| Made redundant by upstream | 5 | tool-call parsing (now upstream `tool_calls.cpp`) and #650 |

Fixes the rebase needed on Windows:

- **Peer-disconnect detection** (`http_server.cpp`): `WSAPoll` rejects
  `POLLERR`/`POLLHUP` in `events` with `WSAEINVAL`; only `POLLIN` is requested
  (they are always reported). This aborted `http_server_test`.
- **Qwen weight upload** (`model_loader.cpp`): upstream keeps a registered
  host copy of the weights, which on Windows lands in shared system memory and
  stalled the 27B load. Weights now go straight to device memory
  (`hipMalloc`, 16 readers with `O_CONCURRENT_RANDOM`, `pread`,
  `hipMemcpyAsync`): 16.4 GiB in 2-3 s.
- **A3B text runner**: dropped upstream's removed `require_tool_call` option;
  `tool_choice: "required"` is enforced on the parsed output.
- **Flash-Next device model**: Q6_K rows re-encoded as Q8_0 where the
  kernels expect it (`token_embd`/`output` exempt).
- `GUFO_QWEN27_UPSTREAM_ATTN=1` (diagnostic, off by default) falls back from
  the Windows verify attention to upstream's, for A/B checks. Measured equal.

Flash-Next, DeepSeek and Qwen3.6-35B-A3B build but were not measured here: the
test machine has 32 GB dedicated GPU memory; they need the 128 GB machine.

## Measurement: Qwen3.8-27B against llama.cpp

Machine: Strix Halo (gfx1151), Windows 11, TheRock 10.0.0, 32 GB dedicated
GPU memory. Model Qwen3.8-27B UD-Q4_K_XL (llama.cpp: Q4_K_M, since UD-Q4_K_XL
with a DFlash2 draft at 64K context does not fit 32 GB there), DFlash2
Q4_K_M draft, 64K context.

Workload: a mix of German, English and strict JSON (`response_format` with a
schema): 7 short cases, 4 "film character" cases with thinking on (3000-token
reasoning budget, temperature 0.6, answer length open, JSON enforced) and one
17.5K-token German/English prompt. Every answer was read for garbage and every
JSON validated.

| Setup | Film + thinking (tok/s) | Short cases (tok/s) | 17.5K prefill (tok/s) | Total, all cases | Endless lists |
| --- | --- | --- | --- | --- | --- |
| llama.cpp Vulkan + MTP (previous production) | 28.2 | 18.7-24.6 | 256 | 487 s | frequent |
| llama.cpp Vulkan + DFlash2 + `--repeat-penalty 1.05` (best llama.cpp) | 28.3 | 19.1-21.2 | 257 | 431 s | none |
| **gufo**, `GUFO_QWEN27_LOOKUP=4`, `--draft-tokens 7`, `--repeat-penalty 1.05` | **28.2-31.2** | 16.1-23.6 | **512-517** | **332-349 s** | **none** |

Takeaways:

- Prefill is about twice llama.cpp's; that and prompt lookup make gufo
  20-30% faster end to end on this workload.
- Qwen3.8-27B tends to loop into endless lists (e.g. a `negative_prompt`
  array) on both servers. `--repeat-penalty 1.05` removed it (0 of 18 risky
  runs) without measurable cost.
- No gain from: the Q8_0 draft, `--draft-policy fixed`, lookup cap 3/5,
  lookup length 10, flush 8, append mode, upstream verify attention, a
  Q4_K_M target (slower; the kernels are tuned for UD-Q4_K_XL), prefill chunk
  2048, 32K context.
- llama.cpp ROCm/HIP builds (lemonade b1351 and a local HIP+rocWMMA build)
  prefill faster than its Vulkan build but decode 12-30% slower.
- gufo's text at temperature 0 is byte-identical with and without speculative
  decoding (Q4_K_M and Q8_0 draft).
- Run-to-run noise is about ±10% per case; totals depend mostly on how long
  the model thinks.

### Repeatability

The first configuration measured (`GUFO_QWEN27_LOOKUP=3`, `--draft-tokens 7`,
no repeat penalty) was run again 8 hours later. Same 10 cases:

| Case | Decode run 1 | Decode run 2 | Prefill run 1 / 2 |
| --- | --- | --- | --- |
| de-json | 23.5 | 23.8 | 347 / 353 |
| en-json | 18.2 | 18.9 | 251 / 245 |
| phish-json | 21.2 | 21.1 | 303 / 298 |
| de-text | 19.2 | 19.5 | 248 / 246 |
| en-text | 21.0 | 21.1 | 195 / 201 |
| film-figur | 17.6 | 17.1 | 281 / 286 |
| film-figur2 | 18.2 | 18.0 | 265 / 273 |
| film-think | 28.4 | 29.8 | 243 / 239 |
| film-think2 | 29.1 | 29.5 | 233 / 243 |
| lang-17k | 16.3 | 16.5 | 525 / 530 |

Decode and prefill speeds repeat within 5% (most within 2%). Total time does
not repeat as closely (342.5 s vs 372.0 s) because the sampled thinking runs
differ in length (1874 vs 2605 tokens for film-think). All answers were valid.
