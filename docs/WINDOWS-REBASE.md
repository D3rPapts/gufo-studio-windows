# Windows fork on current upstream (October 2026)

This branch carries the Windows fork (pixmaate/gufo `main`, 41 commits) onto
current upstream gufo-org/gufo (`7c81965`, "add the net leaf (#711)"). The
older fork history stays untouched on `main`; nothing here rewrites it.

**This document lists only what was run and measured on our machine.** The
rest of the repository documentation comes from upstream gufo (Linux numbers)
or from the pixmaate fork (Windows numbers from a 128 GB machine); we did not
re-check those.

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
  host copy of the weights; on our machine that filled system RAM and stalled
  the 27B load. Weights now go straight to device memory
  (`hipMalloc`, 16 readers with `O_CONCURRENT_RANDOM`, `pread`,
  `hipMemcpyAsync`): 16.4 GiB in 2-3 s.
- **A3B text runner** (compiles; not run here): dropped upstream's removed
  `require_tool_call` option; `tool_choice: "required"` is enforced on the
  parsed output.
- **Flash-Next device model** (compiles; not run here): Q6_K rows re-encoded
  as Q8_0 where the kernels expect it (`token_embd`/`output` exempt).
- **response_format like llama.cpp**: gufo rejected three forms that
  llama.cpp accepts and our client sends: the schema directly under
  `response_format` (`{"type":"json_object","schema":{...}}`), `json_schema`
  without `name`, and `strict: true` with optional properties. All three are
  now admitted and the schema is enforced; checked against the running
  server. Rejected requests log their reason (`error_message=`).
- `GUFO_QWEN27_UPSTREAM_ATTN=1` (diagnostic, off by default) falls back from
  the Windows verify attention to upstream's, for A/B checks. Measured equal.

Flash-Next, DeepSeek and Qwen3.6-35B-A3B are compiled into `gufo.exe` but
were not run here. Only Qwen3.8-27B was tested.

## Measurement: Qwen3.8-27B against llama.cpp

Machine: GMKtec EVO-X2 (Strix Halo, gfx1151, 64 GB), in its normal
performance mode (not the high-performance mode), Windows power plan
"Balanced", Windows 11, TheRock 10.0.0, 32 GB dedicated GPU memory. Both
servers ran under the same settings. Model Qwen3.8-27B UD-Q4_K_XL (llama.cpp: Q4_K_M, since UD-Q4_K_XL
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

- gufo prefill was about twice llama.cpp's (512-517 vs 256-257 tok/s), and
  the total time for all cases was 20-30% lower.
- Qwen3.8-27B tends to loop into endless lists (e.g. a `negative_prompt`
  array) on both servers. `--repeat-penalty 1.05` removed it (0 of 18 risky
  runs) without measurable cost.
- No gain from: the Q8_0 draft, `--draft-policy fixed`, lookup cap 3/5,
  lookup length 10, flush 8, append mode, upstream verify attention, a
  Q4_K_M target (slower), prefill chunk 2048, 32K context.
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

## Everything tried, and what came of it

Columns: decode tok/s averaged over the film + thinking runs; decode tok/s
averaged over the 7 short cases; prefill tok/s on the 17.5K prompt; total
time of all cases (thinking cases measured twice count as their mean, so
every row is the same 10 cases); runs that looped into an endless list or
produced broken JSON. Totals move by ±10% with how long the model thinks;
decode and prefill are the reliable numbers.

### gufo

All runs: Qwen3.8-27B UD-Q4_K_XL, DFlash2 Q4_K_M draft, 64K context unless
noted. `GUFO_QWEN27_LOOKUP=N` enables prompt lookup and caps the DFlash2
draft at N tokens; `--draft-tokens` then sets the lookup length.

| Run | Change | Think | Short | Prefill | Total | Loops | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | DFlash2, no prompt lookup | 29.8 | 18.4 | 518 | 446.4 s | 0 | starting point |
| V1 | `LOOKUP=3`, `--draft-tokens 7` | 28.8 | 19.8 | 525 | 342.5 s | 0 | **works**: short cases faster (19.8 vs 18.4), total 342.5 vs 446.4 s |
| V1 repeat | same, 8 h later | 29.6 | 19.9 | 530 | 360.8 s | 0 | repeatable (see above) |
| V2b | V1 with the Q8_0 draft | 28.8 | 19.1 | 525 | 380.7 s | 0 | no gain |
| V3b | V1 with `--draft-policy fixed` | 30.6 | 19.3 | 525 | 400.1 s | 0 | same as adaptive within noise |
| V4 | lookup flush every 8 tokens | 27.3 | 18.9 | 528 | 390.3 s | 0 | slower |
| V5 | lookup append mode | 29.1 | 19.6 | 528 | 394.9 s | 1 | no gain |
| **V6** | `LOOKUP=4`, `--draft-tokens 7` | **30.5** | 19.0 | 519 | 389.1 s | 0 | **works**: fastest decode on the main case |
| V7 | `LOOKUP=5` | 28.2 | 18.9 | 525 | 377.9 s | 1 | slower than 4 |
| V8 | V6 with upstream verify attention (`GUFO_QWEN27_UPSTREAM_ATTN=1`) | 29.7 | 19.1 | 524 | 351.3 s | 0 | equal: the Windows attention is neither faster nor slower here |
| V9b | V6 with lookup length 10 | 28.9 | 19.6 | 525 | 352.3 s | 0 | no gain |
| V10b | V6 with a Q4_K_M target | 27.7 | 18.0 | 529 | 466.3 s | 2 | slower |
| A1 | V6 + `--prefill-chunk 2048` | 30.1 | 19.1 | 512 | 344.2 s | 0 | no effect |
| A2 | V6 + `--repeat-penalty 1.05` | 29.7 | 19.2 | 517 | 349.3 s | 0 | **works**: no loops, same speed, text quality unchanged |
| **A4 = "V6R"** | A2 repeated | 28.4 | 19.0 | 512 | 331.6 s | 0 | **confirmed, best setting** |
| A5 | V6R + `--context 32768` | 30.0 | 19.0 | 513 | 364.6 s | 0 | no effect |

V2, V3, V9 and V10 were first run with their extra switches silently lost
(`powershell -File script.ps1 -X 'a','b'` passes the list as one string
`"a,b"`), so they measured V1/V6 again. They are excluded; V2b, V3b, V9b
and V10b are the valid reruns, checked against the server's real command line.

Exactness: at temperature 0, gufo's text is byte-identical without
speculation, with the Q4_K_M draft + lookup and with the Q8_0 draft + lookup
(11.1 vs 20.2 tok/s).

### llama.cpp

Same workload, same port, same client. Model Qwen3.8-27B Q4_K_M unless noted,
q8_0 KV cache, flash attention, 64K context.

| Run | Build and change | Think | Short | Prefill | Total | Loops | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | Vulkan (winget b11560) + MTP draft (previous production) | 28.2 | 23.3 | 256 | 487.0 s | 1 | reference |
| L2 | L1 + `-ub 2048` | 27.8 | 22.5 | 244 | 472.8 s | 2 | no gain |
| L3 | ROCm build (lemonade b1351) + MTP | 19.8 | 16.5 | 342 | 629.3 s | 2 | **fails**: faster prefill, decode 30% slower |
| L4 | local HIP + rocWMMA build of upstream (970b51c) + MTP | 24.7 | 19.0 | 334 | 506.2 s | 0 | **fails**: faster prefill, decode 12% slower |
| L5 | L1 with MTP `n=2` | 25.6 | 22.1 | 257 | 600.1 s | 3 | slower |
| **L6** | Vulkan + `--spec-type draft-dflash` with the DFlash2 draft | **29.1** | 22.2 | 260 | 447.2 s | 1 | **works**: best llama.cpp decode |
| L7 | L6 with `n=5` | 28.7 | 19.2 | 260 | 496.1 s | 2 | slower |
| L8 | L6 with Q4_0 target | 29.3 | 21.6 | 311 | 415.7 s | 0 | faster prefill, but perplexity +7.3% on German/English text (4.15 vs 3.87), +1.4% on English docs: rejected for quality |
| L9 | L6 with f16 KV cache | 29.9 | 21.4 | 256 | 498.1 s | 2 | no gain |
| L10 | L6 with the UD-Q4_K_XL target | - | - | - | - | - | **fails**: system memory ran out during the run and the process was ended |
| **A3** | L6 + `--repeat-penalty 1.05` | 28.3 | 20.3 | 257 | 431.1 s | 0 | **works**: best llama.cpp overall |

The local HIP build needed the same `compat/win32/hip_include/cmath` shim as
gufo, passed in `CMAKE_CXX_FLAGS` (llama.cpp compiles HIP as CXX with
`-x hip` on Windows). The halo-box llama.cpp fork was not built or tested.

### What did not work during the rebase

- Upstream's weight loader on Windows (host-registered copy): filled system
  RAM and stalled. Replaced, see above.
- `WSAPoll` with `POLLERR|POLLHUP` in `events`: `WSAEINVAL`. Fixed.
- Starting gufo from a PowerShell pipe or redirect hangs the caller (gufo
  inherits the pipe); use `Start-Process` with redirected files.
- One thinking run went on to about 34K tokens; since then the client sets
  `max_tokens` 8000 for thinking cases.

### Ideas from other forks (2026-10-23)

| Idea | Source | Result |
| --- | --- | --- |
| Long draft-context injections through hipBLASLt BF16 | mikealanni/gufo `ef5b925d` | not applicable, reverted: it only runs when the draft's `fc.weight` is BF16, and in the DFlash2 Q4_K_M draft it is Q4_K, so the path never ran here. (A 147,913-token run with it built in took 444.5 s to first token against 431.1 s in the morning; that difference comes from other changes or noise, not from this commit.) |
| MTP instead of DFlash2 for Qwen3.8-27B | gufo's own 27B MTP draft, as in mikealanni `7344cfb4` | slower: with `gufo prompt`, greedy, no penalties, 5 cases: DFlash2 + lookup 16.9-20.9 tok/s, MTP 13.6-15.4 tok/s, identical output. The HTTP server does not offer MTP for the 27B |

### Verification projection geometries (2026-10-23)

The cycle profile (`GUFO_QWEN27_PROFILE=2`) put 85% of a short-answer cycle
in the target verification pass, and the SSM and attention projections there
read their weights at about 130 GB/s against 165 GB/s for the FFN and 204
GB/s for the head. The Windows verification geometries
(`verify_gemm_27b.hip`) covered the Q5_K and IQ4_XS target shapes but not
the target's Q4_K `attn_qkv`/`attn_gate`/`attn_q` and Q6_K
`ssm_out`/`attn_output`/`attn_k`, which fell back to the generic dispatch.

A sweep with real UD-Q4_K_XL weights (rotated through >= 512 MiB of copies so
no run reads from the 32 MB Infinity Cache, nine alternating rounds, median)
found exact geometries 13-53% faster per projection at 3-8 rows. Every
candidate was compared bit for bit with the production dispatch; greedy
end-to-end output hashes are unchanged.

| | before | after |
| --- | --- | --- |
| SSM projections per cycle | 28.0 ms | 23.5 ms |
| attention projections per cycle | 8.8 ms | 7.6 ms |
| benchmark set, film + thinking | 27.6 tok/s | 29.0 tok/s |
| benchmark set, short cases | 17.9 tok/s | 18.6 tok/s |

A second round covered the FFN. At 2-8 rows the fused gate/up SwiGLU kernel
does not apply (it starts at 14 rows), so the 39 layers whose gate and up
share a format multiply one joined 34816x5120 matrix through the generic
dispatch; the other 25 layers run two 17408x5120 halves. Exact shapes for
the joined Q4_K/Q5_K/Q6_K/IQ4_XS matrices (+6-51%), IQ4_XS/Q4_K/Q6_K
`ffn_down` (+10-50%) and the Q4_K/Q6_K halves (+8-46%) followed; the Q5_K
and IQ4_XS shapes pixmaate had tuned were left as they were.

| | start | projections | + FFN |
| --- | --- | --- | --- |
| cycle, short answers | ~133 ms | ~125 ms | ~121 ms |
| `gufo prompt`, temperature 0, 5 cases | 16.9-20.9 tok/s | 17.1-21.4 tok/s | 17.8-22.2 tok/s |
| benchmark set, film + thinking | 27.6 tok/s | 29.0 tok/s | 30.8 tok/s |
| benchmark set, short cases | 17.9 tok/s | 18.6 tok/s | 19.0 tok/s |

Greedy output hashes were identical after each round. The short-case means of
single benchmark runs move by several percent between runs; the temperature-0
row repeats exactly the same tokens and is the steadier comparison.

A third round ran the sweep on the DFlash2 Q4_K_M draft (about 10% of a
cycle). At 2-5 rows, the usual draft widths under `LOOKUP=4`, only the small
matrices gained: `attn_conv_proj` 1280x5120 +7-19%, `attn_k`/`attn_v`
1024x5120 +7-8% at 2 and 5 rows, `attn_q` +5% at 2 rows, `attn_output` +6%
at 4 rows; `ffn_gate`/`ffn_up`/`ffn_down` already ran at 180-225 GB/s and
gained nothing. At 6-8 rows the draft's `attn_q`, `attn_output`, `attn_k`/
`attn_v` and `attn_conv_proj` had no shape at all and ran through the
generic dispatch at half speed; the new shapes are +24-55% there. Greedy
output hashes are identical (5 of 5). End to end the gain is too small to
measure, so the question was whether the 6-8-row fix makes longer DFlash2
drafts worth it:

| benchmark set (one run each) | before, `LOOKUP=4` | draft shapes, `LOOKUP=4` | draft shapes, `LOOKUP=7` |
| --- | --- | --- | --- |
| film + thinking (4 runs, mean) | 29.9 tok/s | 30.2 tok/s | 31.8 tok/s |
| short cases (7, mean) | 20.5 tok/s | 18.8 tok/s | 19.5 tok/s |
| 17.5K prompt | 15.2 tok/s | 15.1 tok/s | 14.8 tok/s |
| all cases, tokens / time | 24.5 tok/s | 24.8 tok/s | 25.1 tok/s |

Within the noise of single runs (thinking lengths differed by up to 2x
between runs); `LOOKUP=7` leads on the main case but is not confirmed, and
the short cases were 5-8% slower with the new build in this run, also not
confirmed. The
planned temperature-0 comparison (old against new build, identical tokens)
was not run. Production stays on `LOOKUP=4` with the new build.

The sweep is in [tools/windows/verify-sweep](../tools/windows/verify-sweep/README.md),
so another GGUF or quantization can be measured the same way.

The tuning only reshapes launches of the existing kernel, so it should carry
to Linux on the same GPU, but it was measured on Windows with TheRock 10.0.0
only.

MTP notes: Unsloth's 27B GGUFs carry the MTP layer as `blk.64.*` in Q8_0;
gufo's loader now accepts Q8_0 (and Q5_K), so `--mtp-model` can point at the
model file itself. The verifier only drafts with MTP for greedy requests
without penalties: any repeat or presence penalty, temperature above 0, a
JSON schema or a reasoning budget makes the request "sampled", and the 27B
MTP draft has no sampled proposals. `gufo prompt` applies Qwen's preset
presence penalty unless `--presence-penalty 0` is given.

### Upstream merge of 2026-10-24 and the prompt-lookup policy

Twenty upstream commits merged (`win-merge-2026-10-24`), among them #705
(faster DFlash2 drafting and verification, a new prompt lookup) and the
platform leaves of upstream's own Windows port. Two Windows issues came with
it:

- `mapped_file` returns no descriptor on Windows, and the port's loaders read
  shards through it. The 27B load fell back to the host-registered copy and
  filled system RAM (16.9 GB RSS, 1.9 GB left, never ready). The mapping now
  also holds a CRT descriptor (2b343d34).
- #705's lookup replaces the port's opt-in `GUFO_QWEN27_LOOKUP` lookup.
  `GUFO_QWEN27_LOOKUP=N` keeps capping DFlash2's own drafts at N.

Workload as above (film character JSON with thinking, temperature 0.6,
repeat penalty 1.05). 8 film runs per row unless marked * (4 runs); single
runs vary by about ±2 tok/s with the reasoning length.

| Build, settings | film + thinking | short cases | tokens / cycle | ms / cycle |
| --- | ---: | ---: | ---: | ---: |
| before the merge, `LOOKUP=4` (production) | 30.6-31.5 | 19.1 | 4.01-4.27 | 130-132 |
| merge, upstream defaults * | 28.7 | 19.1 | 3.83 | 127 |
| merge, `LOOKUP_MIN=12` * | 29.8 | | | |
| merge, `LOOKUP_REASONING=1`, `_MIN=12` | 29.1 | | 3.79 | 131 |
| merge, `LOOKUP_GENERATED=1`, `_MIN=12` | 30.5 | | 3.90 | 128 |
| merge, `_GENERATED=1`, `_MIN=12`, `LOOKUP=5` | 31.9 | | 4.18 | 130 |
| before the merge, `LOOKUP=7` | 31.8 | 20.6 | | |
| **merge, `_GENERATED=1`, `_MIN=12`, `LOOKUP=7`** | **34.2** | **20.4** | 4.39 | 132 |

Upstream's lookup copies for sampled requests only from their starting
context and after 24 matching tokens. This answer restates its own earlier
fields, so it loses copies; indexing just the finished reasoning does not
recover them. With both knobs the merged build matches the old lookup, and
only on top of #705's lookup does the longer DFlash2 cap pay off (+8 % on
the main case, +7 % on short cases, prefill unchanged, all JSON valid). The
old build gains 1 % from the same cap.

The knobs: `GUFO_QWEN27_LOOKUP_MIN=N` (strict gate), `GUFO_QWEN27_LOOKUP_GENERATED=1`
(sampled requests copy from their own generated text),
`GUFO_QWEN27_LOOKUP_REASONING=1` (only the finished reasoning; no gain here).
Defaults keep upstream's behaviour.

### Router

`gufo router` (PGoski's branch, ported in `win-router`) serves every model
behind port 8081 with one loaded at a time: Qwen3.8-27B, Qwen3-Omni and the
llama-server models through `external` sections. Windows changes:
CreateProcessW workers in a kill-on-close job, sockets closed with
`closesocket`, millisecond socket timeouts, a signal-free websocket hand-off.
Two relay bugs found here are platform-neutral and went to PGoski/gufo#1: the
response head was awaited for 30 s only (longer non-streamed replies got 502
and the worker could be evicted mid-generation), and a forwarded
`Expect: 100-continue` made the relay hand clients the previous answer.
Switching on gfx1151: 27B cold 10.6 s, Omni 22.7 s, llama-server 4B 4.5 s;
through the router the benchmark matches direct serving.

### Long context

gufo with V6R and `--context 155648` (152K) on the same 32 GB machine: a
147,913-token prompt (gufo source code) was answered correctly; prefill
343 tok/s (7.2 min), decode 10.8 tok/s at that depth, 35.8 GB GPU memory in
use (part of it beyond the 32 GB carve-out). 256K was not tried.

### Not tried

- The EVO-X2's high-performance mode and Windows power mode "Best performance" (the machine ran its normal mode and "Balanced").
- Flash-Next, DeepSeek and Qwen3.6-35B-A3B (not run at all).
- Concurrent requests (all measurements used one request at a time).
