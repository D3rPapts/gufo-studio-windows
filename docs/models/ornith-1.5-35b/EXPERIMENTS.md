# Ornith-1.5-35B experiments

- 2026-09-29: ported the qwen35moe runtime from the Flash-Next package by
  stripping the PLE n-gram, hyper-connection and sparse-indexer stages;
  attention runs dense only. Retained.
- Reusing the vendored qfn_mmq kernel build instead of duplicating it:
  retained (generic quant kernels; ds4 precedent).
- Native Q4_K routed down projections on the WMMA tier (48-row tiles):
  retained; Q6_K downs dequantize to Q8_0 at load.
- 2026-09-29 decode pass: dense decode-width projections keep a raw Q4_K
  byte view beside the Q8_0 wide-batch view and run `qfn_mmq_moe_vec` E=1
  (shared expert via `moe_gated_vec`); +11% AR (57.3 to 63.4 tok/s).
  Retained. The MTP draft's split fc_embedding/fc_hidden projections must
  NOT carry the combined tensor's whole-row view — the first cut leaked it
  and the draft proposals collapsed to 0% acceptance. Rejected: event-poll
  sync and pipeline-replay probes (WDDM overhead is not on the critical
  path; graph replay confirmed the decode graph itself is GPU-bound at
  ~17 ms/token, i.e. bandwidth).
- 2026-09-29 benchmark pass (native vs HTTP, concurrency): prepared-cohort
  HTTP decode lands within 2% of native AR (62.1 vs 63.2 tok/s) and scales
  to 198.3 tok/s aggregate at C8 with hash-exact completions. Retained
  (sequential session preparation during cohort setup is required; see
  QUALITY.md). Two engine findings, unfixed, both reproducible:
  (1) concurrent prefill requests interleave scheduler chunks and the
  width-sensitive prefill routes (F32 router hipBLAS sgemv; F16-route
  threshold at 2048 rows) shift logits by ulps, so a session prefilled
  alongside another can decode to different greedy text than the isolated
  reference — fix direction: width-invariant router kernel or chunk-invariant
  routes; (2) co-batched speculative verification beyond 8 total rows leaves
  the q8_1 vector kernels for the tiled fallback with different quantization
  and reduction, tripping the greedy-equivalence gate at C4 (mixed) / C6
  (repetitive) — fix direction: cap co-batched verify rows at the vector
  contract or make the tiled path reproduce it.
- 2026-09-29 Q8_0 artifact pass: every projection keeps its native Q8_0
  bytes end to end. Three engine changes, retained: (1) `DequantizeRow`
  decodes Q8_0 sources — the raw-block memcpy fallback would have
  corrupted the dense stacks and the alpha/beta stacks of a Q8_0 artifact;
  (2) `CopyDequantQ8_0` streams Q8_0 parts straight to the device instead
  of a dequant+requant round trip, and the split MTP projection drains the
  staging pipeline before it frees the combined buffer — that free was
  only safe while the upload was synchronous (first Q8_0 load died in the
  staging worker with `invalid argument`); (3) routed Q8_0 gate/up joins
  the F16 WMMA tier (the routed kernel already decoded Q8_0 downs):
  pp2048 2530 → 3074 tok/s, +21% over the int8 raw-moe fallback and +7%
  over the Q4_K_M build. Findings: single-stream AR decode is
  bandwidth-bound and pays the artifact's 1.7× expert bytes (-17% vs
  Q4_K_M), converging to Q4_K_M at C4-C8 where batched GEMVs are
  MAC-bound (C6 +2%); HTTP MTP loses (-14/-23%) even though native MTP
  gains (+3%), because acceptance drops (55/58% vs 63/78%) while the
  adaptive policy drafts more per cycle — sharper draft and target logits
  agree less often on near-ties. llama.cpp b11243's Q8_0 greedy is NOT
  ground truth for this artifact: it diverges from our F32 oracle at a
  0.6-logit near-tie where the GPU matches the oracle; use the oracle
  gates in QUALITY.md for Q8_0 comparisons.
- 2026-10-19 Q6_K decode GEMV pass (retained): the vendored mmvq gained
  Q6_K (vec_dot, VDR, `mul_mat_vec_moe_dispatch`, `qfn_mmq_moe_vec`
  whitelist) and `VecSupported` admits Q6_K, so `CopyDequantOne` parts
  (LM head, routed Q6_K downs, mixed shared-expert downs) keep a raw
  native view beside their Q8_0 wide-batch copy, and `Executor::Experts`
  reads the native view below the tiled threshold. The Q4_K_M artifact
  stores its head (417 MB) and 20/40 routed down stacks as Q6_K; decode
  previously read requantized Q8_0 copies of them. A/B, same prompt,
  greedy, internal bench, baseline 479175f vs this change: pp2048
  2903 ± 49 → 2922 ± 27 (prefill keeps the Q8_0/WMMA tier, unchanged);
  tg128 AR 62.16 ± 0.52 → 64.79 ± 0.44 (+4.2%); tg128 MTP 77.56 ± 0.47 →
  88.48 ± 1.36 (+14.1%); MTP d4096 85.1 → 86.5; DFlash2 open-ended
  48.7 → 88.9 (+82%) because acceptance jumped 25% → 61% — the draft is
  trained against the BF16 target and the exact Q6_K decode tracks it
  better than the requantized copy did. Greedy AR == MTP == DFlash2
  sha-identical per build (lossless property intact). Flash-Next
  Q4_K_XL sha ffa993c8 and the Ornith Q8_0 artifact sha e552e58b are
  byte-identical across the change; Flash-Next pp/tg move within noise.
  Device memory at sessions 4 rises ~4.6 GiB (native copies of the
  routed Q6_K downs and head) to 30.7 GiB. Rejected variant: per-part
  native stacks for the mixed-format attention projections (14/30 GDN
  qkv and 6/10 GQA v are Q6_K beside Q4_K parts, ~1.6% more AR) — the
  MTP draft block consumes projections across calls at different widths
  with `projections_ready`, so a width-dependent combined/per-part
  split breaks the producer-consumer layout agreement; per-part tensors
  would also need Q8_0 prefill copies. The head dominates the win
  (microbench: head shape 2.34 → 1.86 ms/token at 225 GB/s).
- 2026-10-19 chained greedy MTP (retained): ported Flash-Next's
  device-side draft chain. The catch-up and every greedy proposal run as
  one submission that feeds each step's argmax into the next step's
  embedding, so the k per-draft stream synchronizations and host round
  trips collapse into one chain download; the chain is submitted eagerly
  (its launch submission hides behind the previous cycle's GPU work —
  per-shape graph capture instead stalls a decode cycle on every new
  (catch-up, proposals) shape). Verification gained `kVerifyGreedy`:
  the batched argmax over all verified rows including the frontier rides
  the verify graph into pinned memory, which removes the per-cycle
  frontier download and the separate predictions synchronization, and
  the next anchor comes from the frontier argmax via `DeferSample`.
  Two synchronizations remain per cycle (chain, verify). Retuned
  `mtp_costs.hpp` to the chained per-width cycle costs measured at
  2048-token depth (C1/context-0 ladder 15.5/19.5/22.5/27.5/34/41/49/56.5
  ms for widths 1-8; context slope and cohort shape keep the Flash-Next
  measurements), warmed the acceptance prior with the draft's real
  profile (~90% first proposal, decaying with depth) and stepped the
  probe one width at a time so probe failures land on the deep depths
  the short-width scores do not read. Interleaved same-binary A/B at
  2a289ca, greedy completions hash-identical to the AR reference on
  every row (lossless property intact; AR and pp within noise):
  Q4_K_M MTP tg128 98.6 → 101.9 (+3.3%), tg512 105.4 → 110.4 (+4.7%);
  Q8_0 MTP tg128 82.5 → 84.4 (+2.4%), tg512 86.4 → 86.8 (+0.4%). The
  mtp_sampling policy test now asserts probe-reaches-limit and a
  score-optimal steady width under full acceptance (the depth-decaying
  priors keep unobserved deep widths pessimistic, so the steady choice
  under perfect acceptance is the cost-optimal width rather than the
  limit; consecutive full acceptances still probe to the limit first).
  Rejected variants, both bit-exact but slower: a proposal shortlist
  (gather kMtpDraftShortlist head rows once per cycle, rank chained
  steps over them) — Flash-Next's 512-row net collapses depth>=1
  acceptance on this 248k vocabulary and the candidate reduction kernel
  cannot Keep > its 1024-wide tile (each pass would grow), so every
  chained step ranks the full head; and reading the verify tokens
  straight from the device chain ring (one synchronization per cycle) —
  the width controller's acceptance evidence shifted and settled d1-heavy
  on both artifacts (Q4_K_M tg128 66.5 tok/s). Prefill keeps the
  2026-10-01 pass's WMMA routed arithmetic; see the 2026-10-19 device
  dispatch entry below for its stage profile.
- 2026-10-19 device-side routed dispatch (retained): temporary hipEvent
  stage spans around the prefill showed every layer stopping at the
  expert-counts readback — `hipEventSynchronize(counts_ready_)` plus a
  three-pass host tile-map build and upload, up to 676 ms of host wall
  per 2048-token chunk. The readback was not the wall-clock limiter (the
  host was waiting on the GPU), but it serialized every layer and it was
  the only host round trip left in the chunk. The maps now build on the
  device: one 256-thread `RoutedTileMaps` kernel compacts each map's
  (expert, row tile) entries from the device-side expert counts and pads
  the rest of a fixed per-map capacity with 0xFFFFFFFF sentinels, and the
  routed GEMM grids run at that capacity, skipping sentinels in-kernel —
  no `n_tiles` crosses to the host and `RouteHints` reads nothing. The
  count-dependent map refinements (the 128-row pair map and the
  wide-down 48/64 choice) folded into fixed token-count-and-format
  choices; their effect is inside the measured delta. Interleaved
  same-binary A/B at b0d9209, greedy completions hash-identical to the
  isolated AR reference on every row: pp2048 Q4_K_M 2951 → 3005 (+1.9%),
  Q8_0 3102 → 3233 (+4.2% — the all-native artifact's heavier per-layer
  dispatch made the readback costlier); AR tg and MTP tg unchanged within
  noise. `gufo bench --validate-prefill 8` passes (cosine 0.9995,
  scalar_winner_rank 1); qwen35moe config/gdn/mtp_sampling pass and
  attention_ops shows only its documented pre-existing 2-ulp replay
  case. The corrected prefill stage profile (hipEvent spans, one
  2048-token chunk ≈ 700 ms): GDN linear attention ~219 ms (31%), routed
  gate/up+down F16 GEMMs ~165 ms (all 256 experts' weights read per
  chunk at near the memory roofline), shared expert ~67 ms, GQA ~67 ms,
  router ~17 ms, epilogue ~30 ms, compaction and norms ~20 ms — the
  prefill is GPU-bound, and the next prefill levers are the GDN scan and
  the shared-expert F16 GEMMs (~8-16 TFLOPS effective), not the host
  path.
