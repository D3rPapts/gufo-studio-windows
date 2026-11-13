# Server reference

The complete contract of `gufo serve`: execution and queues, every endpoint and
field, errors, lifecycle, metrics and logs. The user-facing guide is
[server](../server.md); this page holds the detail behind it. Supported fields
behave as documented here, and unsupported ones return explicit errors. Gufo
doesn't reproduce OpenAI-hosted storage, billing, organization or account
behavior.

Reference protocols:

- https://developers.openai.com/api/reference/resources/responses/methods/create/
- https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/
- https://developers.openai.com/api/reference/resources/models/methods/list/
- https://platform.claude.com/docs/en/api/handling-stop-reasons
- https://developers.openai.com/api/docs/guides/streaming-responses

## Implementation

The C++20 runtime serves bounded HTTP/1.1 requests with one connection worker
per request and a separate inference scheduler. Connections close after each
response. Chat Completions, Responses and Messages stream with Server-Sent
Events; closing the socket cancels the request in the scheduler. JSON parsing
rejects duplicate keys, invalid numbers and nesting beyond 128 containers.
Python may serve development tools and conformance tests, never requests.

One `gufo` executable holds every built model graph, tokenizer and HIP kernel;
`chat` and `prompt` drive the same scheduler without HTTP, `video` runs the H3
route the video worker uses, and `transcribe` the ASR route. Commands and
options are in the [command line](../cli.md).

## Model execution

### HIP execution

Qwen HTTP requests share one immutable `QwenGpuModel` containing the mapped
weights and tokenizer. Each request leases a preallocated `QwenGpuExecutor`
with independent KV, recurrent, graph, activation, and logit state.
`--sessions N` controls the bounded execution session pool, not the number of
remembered conversations. Checkpoints have separate entry and memory limits;
see [KV cache](cache.md#limits). The scheduler batches ready
requests when the model runner supports their execution mode.

Text serving defaults match llama.cpp for context and generation length:

| Setting | Default |
| --- | --- |
| `--context` | `0`: native context from model metadata, per session |
| `--enable-yarn` | Off; Flash-Next requires it for `--context` above 262144 |
| `--max-tokens` | `-1`: until EOS or remaining context is exhausted |
| `--sessions` | `1` |
| Thinking / reasoning effort | Enabled; Qwen `xhigh`, DeepSeek `high` |

Qwen chat prompts are bounded by the session context, not a fixed size: the
rendered template may use up to 128 bytes per context token (at least 1 MiB).

Clients can set a positive `max_tokens` / `max_completion_tokens` (Chat
Completions) or `max_output_tokens` (Responses). These include reasoning tokens.
Omitting the field uses the server default. A response cannot exceed remaining
context; exhaustion reports `length` or `incomplete`, without discarding earlier
conversation tokens. Reduce `--context` or `--sessions` if their state exceeds
available memory.

Text sampling follows each model's recommended defaults in `serve`, `prompt`
and `chat`, so these commands sample unless `--temperature 0` selects greedy
decoding. Model providers supply the defaults for the effective thinking mode.
Explicit server or command-line settings override them, and explicit request
values override both. Null request values inherit where supported; explicit
zero values, including `top_p: 0`, are preserved. Provider defaults never
replace a request's output grammar.
The startup sampling log reports server defaults; requests can override each
setting independently, including when changing thinking mode.

| Model / mode | Temperature | Top-p | Top-k | Presence penalty |
| --- | --- | --- | --- | --- |
| DeepSeek V4 Flash 0731 (agentic) | 1.0 | 0.95 | 0 | 0 |
| Qwen3.8 27B / Flash-Next, thinking | 1.0 | 0.95 | 20 | 0 |
| Qwen3.8 27B / Flash-Next, thinking off | 0.7 | 0.8 | 20 | 1.5 |

Min-p and frequency penalty default to zero; repetition penalty is 1.0.
AR and DFlash2/MTP/DSpark use the same target defaults.
Use `--temperature 0` or request `"temperature": 0` for greedy
output; `bench` remains greedy by default. Audio and image generation retain
their own settings.
Use `--think off` or request `"reasoning_effort": "none"` to disable thinking.
DeepSeek maps `minimal`/`low` to `low`, `medium`/`high`/`xhigh` to `high`,
and `max` to `max`.
The [functional suite](../../tests/functional/README.md) with
`--suite sampling-defaults --sampling-preset qwen38` (or
`deepseek4`, or `provider` for a model with its own defaults) compares omitted
and explicit settings, including C2 replay.

Sources: [Qwen27B](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices),
[Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices),
[DeepSeek 0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/README.md),
[DeepSeek thinking defaults](https://api-docs.deepseek.com/guides/thinking_mode).
DeepSeek's 0.95 top-p is its agentic recommendation; neutral penalties and
disabled unspecified filters are Gufo defaults.

```sh
nix build

./result/bin/gufo serve \
  --host 127.0.0.1 \
  --port 8080 \
  --sessions 1 \
  llm \
  --model models/qwen.gguf \
  --context 4096
```

Cancellation preserves successfully executed state for continuation; failed
model operations invalidate mutable state. Model replacement is transactional,
so active requests retain their original model until their lease ends.
Generation responses include:

```text
Server-Timing: ttft;dur=<milliseconds>, inter_token;dur=<milliseconds>
```

When `stream_options.include_usage` is enabled, the terminal Chat Completions
usage chunk also includes a namespaced `usage.gufo` object. It reports
privacy-safe scheduler and stage metrics used by
`tools/serving/gufo-serving-bench.py`: queue, prefill, decode, TTFT and ITL timing;
actual prefill work; cache use; logical concurrency; physical execution width;
and the executed plan. Prompts, generated text, local paths, request IDs, and
token IDs are excluded.

DeepSeek V4 Flash batches up to eight active requests. Each request owns its
attention caches, compressor state, position, and sampling state. Dense and
expert projections share weight reads across the batch; one request uses the
single-session path.

With `--dspark-model <support.gguf>`, DSpark is selected automatically unless
`--speculative off` is explicit. Greedy and sampled requests keep independent
samplers, cache state and draft policies. C1 uses point-mass proposals and
retains seeded AR token identity; filtered sampled C>1 can use compact
probabilistic proposals with exact p/q acceptance and residual correction.
That route preserves the target distribution with its own seeded trace.
Request decoding leaves EOS and later speculative tokens out of the reusable
checkpoint. Fixed-length benchmarks may continue through EOS.
See the [DS4 benchmark and quality contract](models/deepseek-v4-flash/BENCHMARKS.md).

Qwen3.8-Flash-Next uses the same scheduler, with independent recurrent, KV,
indexer, sampling and MTP state per session. Prefix snapshots retain complete
state; `--cache-disk` adds restart-safe reuse. MTP draft length adapts within the
configured ceiling. Sampled MTP proposals use p/q acceptance and residual
correction; greedy verification follows target argmax. Draft and verification
work can batch across ready requests. See the
[Flash-Next benchmark and quality contract](models/qwen3.8-flash-next/BENCHMARKS.md).
Flash-Next's native context is 262144. `--context` above it requires
`--enable-yarn`, which enables YaRN up to factor context / 262144; the
server refuses to start without the flag, and the flag is an error at or below
262144. A session keeps the native rotation while it holds at most 262144
tokens and takes the smallest whole factor covering its positions past that,
re-rotating its cached keys when the factor changes; see the
[measurements](https://github.com/gufo-org/gufo/pull/350#issuecomment-6323981940),
the
[opt-in proposal](https://github.com/gufo-org/gufo/pull/350#issuecomment-6323982386),
the
[accuracy check](https://github.com/gufo-org/gufo/pull/350#issuecomment-6234093998)
and [Qwen3.8 Flash-Next](../models/qwen3.8-flash-next.md) for the memory
cost.

Each model chooses its prefill chunk. Beside decoders, a runner with a
scheduling profile (every in-tree text model) keeps that width and splits each
pass by layers, with a decode round between groups; a runner without one gets
fixed 512-token passes. A lone request keeps the model's own pass width.
`--prefill-chunk` is deprecated: it is accepted, ignored with a
`event=deprecated_option` warning, and will be removed together with the
512-token legacy path (#1028).

Qwen image requests reuse decoded and resized RGB pixels using a fast lookup
hash and exact encoded-byte comparison under the preprocessing version and
whether the `low` dimension bound applies. `high`, `original`, and `auto` share
entries. Each loaded Qwen runner retains up to 128 MiB of encoded bytes and
resized pixel vectors combined, evicting least-recently-used images; entry
metadata is also bounded. `cache_prompt: false` bypasses this reuse as well as
continuation reuse.
Requests share immutable buffers, so eviction does not invalidate an in-flight
request. This CPU cache is separate from the snapshot budget below and is not
written to disk. A resized 1920×1080 screenshot takes about 6 MiB of RGB storage,
plus its encoded bytes. The compressible test screenshots fit 21 entries in
this budget; larger encoded files reduce that count. Warm history still pays
for prompt rendering, tokenization and the existing pixel/position identity
hash; a new screenshot still needs vision encoding and prefill. Debug logs
expose `event=image_cache` with whether reuse is enabled, entries, retained
pixel/encoded bytes, hits, misses and evictions.

`--cache-ram-bytes 0` (the default) selects an automatic snapshot budget capped
at 32 GiB and half the available host RAM after model/state allocation, respecting
container limits. Available RAM is `MemAvailable` minus `CmaFree`: free CMA
pages, such as the kernel's KHO scratch area on Ubuntu 26.04, only hold
movable pages, not GPU allocations. 27B also checks HIP free memory. A
positive value sets a byte cap that may exceed the automatic budget, up to the
available host RAM minus 4 GiB; the startup line reports both as
`automatic_bytes` and `max_bytes`.
The pinned disk staging buffer comes out of that ceiling, and `max_bytes`
excludes it.
The 128 checkpoint records are independent of `--sessions`; more than one can
belong to a conversation.
Qwen27B and Qwen3.8-Flash-Next pin pages for up to about 89% of its budget as captures
need them, about 1 GiB at a time, and keeps them; the rest covers cache metadata as
it grows. The first step runs at startup and later ones on a background thread
ahead of need. Each takes about 200 ms while the host has free pages and longer
when the kernel must reclaim page cache. Segments use ordinary pages that are
registered with HIP: `hipHostMalloc` asks for high-order pages, and compacting
host memory for them stalled decode for seconds when most RAM belonged to the GPU.
Growth stops early while the host would keep less than 4 GiB available. The pages hold checkpoint state and KV rows, partial
tails and idle-session spill; there is no separate spill option.
Under pressure, optional copies give way before another conversation's last useful checkpoint.

`--cache-ram-bytes off` keeps no RAM cache. The server retains and captures no
checkpoint in RAM and pins no cache memory, except up to 512 MiB of backing that
Qwen and Flash-Next use for captures when `--cache-disk` is set. Each session
still reuses its own live state, and `--cache-disk` keeps working. See
[the cache reference](cache.md#limits) for what off keeps and drops.

Qwen27B uses component-cache slots in both AR and DFlash2 modes. A checkpoint
includes target KV and recurrent state, image-prefix identity, sampled frontier
and pending output tokens. DFlash2 also retains its independently positioned
ring KV, pending target features and draft controller state. Completed prefill
steps can be retained after cancellation. MTP is unsupported on Qwen27B.

Prompt reuse is enabled by default; how the cache finds, retains and
evicts that state is described in [the KV cache](../cache.md).
`cache_prompt: false` on
`/v1/chat/completions`, `/v1/responses`, `/v1/completions` or `/completion`
bypasses memory and disk lookup for that request; the result can still
populate the cache. `null` keeps the default, `true`. DeepSeek and Qwen tool
requests retain a checkpoint before the assistant-generation suffix, including
when a client drops the interrupted assistant and appends `"."` after a tool result. DeepSeek
also accounts for tokenization changes where adjacent user/tool turns join.
Qwen requests retain this checkpoint with thinking enabled or disabled.
Warm continuations preserve the reused frontier before prefill and save the
new stable boundary too. A second full-prompt checkpoint enables exact retries
without prefill; all copies share the same snapshot-memory budget.
Exact live continuations reuse generated tokens. The server reports cached
and newly processed tokens separately; resuming from the checkpoint processes
the short suffix. System instructions, tool definitions and image identities
must match the retained prefix.

`cache_persist: false` on the same endpoints keeps the request's own
checkpoints off [`--cache-disk`](cache.md#the-disk-tier): they stay in RAM
and leave with RAM eviction or a restart. The request can still restore disk
checkpoints that other requests wrote, and a later request without the flag
that sends the same prefix can write its own. `null` keeps the default, `true`.
Each response echoes the value it applied as a top-level `cache_persist`
boolean, `true` by default, so a client can confirm that the server honoured
the flag. Streams carry it on the finishing chunk and the usage chunk (Chat and
text completions) or on the final `response` object (Responses), never on a
content delta.

Qwen image identity is checked only for images consumed before each checkpoint.
Appending an image reuses the preceding text/image state in RAM or on disk;
changing, removing or moving an earlier image invalidates checkpoints after it.
New images get a checkpoint before assistant framing so later turns do not
encode or prefill them again.
With `preserve_thinking=false`, a new user turn removes reasoning from the
preceding tool cycle. Gufo retains the state before that cycle and processes
its changed suffix again.

A loaded Qwen vision sidecar is prepared while the model loads, before the RAM
cache is sized and before readiness: Gufo hashes the complete sidecar for image
cache identities and uploads the encoder weights, so neither happens in the
first image request and admission's free device memory already excludes them.
The weights stay resident even if no image arrives, including a sidecar
auto-discovered as `mmproj-BF16.gguf` in the model's directory or its parent.
The startup line `event=vision_prepared` reports `resident_bytes`,
`identity_ms` and `upload_ms`. Startup fails if the upload does not fit. Image
encoding scratch is still allocated per image.

`SIGINT` and `SIGTERM` cancel active requests and drain accepted disk writes
before exiting. `--cache-disk DIR` defaults to 8 GiB retained on disk.

`--cache-disk-bytes` counts every file the cache manages,
including temporary files of unfinished writes and files a crash left
unreferenced. Restarting with a lower budget removes those first, then the
oldest checkpoints by creation order, until the directory fits; use history does not
survive a restart. One server owns the directory: a second one fails
at startup, after loading its model but before committing its RAM budget, with
an ownership error naming the directory. The lock does not wait: a restarted
server fails if its predecessor is still draining disk writes when the new one
finishes loading its model.
Do not share a directory with another model server; the component cache removes
their checkpoint and temporary files too, including writes in progress.
The server ignores `--cache-disk-staging-bytes` and logs
`event=option_ignored`. Writes and restores stream each checkpoint through an
internal 1 MiB pinned buffer, piece by piece; a larger buffer did not change
restore or write speed and took pinned RAM and device heap from the cache and
the sessions. Remove the option from Qwen27B and Flash-Next command lines.

For DeepSeek, `--cache-disk-staging-bytes 0` (the default) selects the
smaller of one eighth of available host RAM after model/session loading
(including cgroup limits) and the disk budget. This bounds queued
captures/writes and each disk read separately; it allocates nothing upfront.
Live model state and retained RAM snapshots have separate budgets.
Snapshots that exceed either limit are skipped with their required size and
available budget logged; live conversation reuse remains available. Existing
files that exceed the current staging limit are preserved subject to disk LRU
eviction and can be reused after restarting with sufficient staging.
Qwen27B streams large checkpoints through the same fixed 1 MiB staging buffer;
`--cache-disk-bytes` still needs to fit the retained checkpoints.

For a focused cancellation check, run
`python3 tests/functional/continuation.py --output /tmp/cache-check.json`
against a private server named `cache-test` on port 5815.
It checks interruption during reasoning and visible output, with and without
reasoning replay, and greedy/seeded sampling. Use
`--tools --discard-assistant` to exercise interrupted agent tool turns; add
`--prefix-repetitions 5500` for a roughly 50K-token prefix.
For persistence, enable `--cache-disk` before the check, restart the same server,
and add `--restore /tmp/cache-check.json`.
Use `--image /path/to/image.png` for Qwen image conversations.
Add `--append-image` to introduce the image after a cached text turn, and
`--reasoning-effort high` to check a specific thinking effort.
Each case continues for a third turn; repeat `--case NAME` to select only the
cases needed for a change.
The check requires exact snapshot and matched-history replay in memory. After a
restart, sampled output may vary when a disk restore re-prefills a gap; greedy
and zero-prefill restores still require equality. The report separates successful
validation (`status`) from observed assistant-message equality (`exact`) and its
requirement (`exact_required`). The SDK conversation suite checks cache bypass.

### Hardware compute queues

The gfx1151 command processor keeps eight compute queues mapped at once,
counted across every process on the device. Past that total the firmware
scheduler time-slices them even when all of them are empty: the GPU then
reports a busy engine at its top shader clock and draws about 26 W above idle
for as long as the processes live. Queues are claimed on a process's first HIP
dispatch and never released — neither `hipStreamDestroy` nor `hipDeviceReset`
gives one back — so the count is fixed at startup, and the first dispatch costs
two queues whatever the configuration. Four HIP processes is therefore the
ceiling on one device, gufo or otherwise.

Each server reads the queues already in use from
`/sys/class/kfd/kfd/proc/*/queues/*/type`, which is world-readable and so
includes processes gufo does not own, then exports `GPU_MAX_HW_QUEUES` before
loading a model and logs a `queue_budget` event. That event is INFO-tier and
follows `--log-level` like the rest of the startup diagnostics; the
`queue_budget_exceeded` warning below is WARN-tier:

| Server | `GPU_MAX_HW_QUEUES` | Resident compute queues |
| --- | ---: | ---: |
| `serve llm` | 2 | 3 |
| `serve tts` | 1 | 2 |
| `serve asr` | 1 | 2 |
| `serve image`, `serve video`, `serve embedding` | runtime default | up to 5 |

Text and audio caps are measured to cost no throughput: `serve llm` is flat from
the runtime default down to a single queue at one and at four concurrent
requests, and both audio servers are flat within run-to-run noise. Image,
video, and embedding are unmeasured, so they keep the runtime default
unless the device is too busy to hold it.

The cap is only ever lowered to fit the free slots, never raised to fill them,
so a server's throughput does not depend on the order the servers started. A
`GPU_MAX_HW_QUEUES` set by the operator is always left alone, but it is still
checked: a cap of N resolves to at most N + 1 resident queues, so a value that
will not fit is reported even though it is honoured. When a server cannot fit,
it logs a `queue_budget_exceeded` warning and starts anyway.

### Reasoning controls

`--think auto` uses the model's default. Qwen27B and Flash-Next match the
official Jinja: thinking enabled, `xhigh` effort, prior reasoning preserved.
Use `--think off` or `chat_template_kwargs.enable_thinking=false` for direct
answers. DeepSeek defaults to thinking with `high` effort. Quality comparisons
must use the same reasoning mode and effort.

`--reasoning-budget N` caps reasoning at N tokens on Chat, Responses and
Messages requests, as in llama.cpp: `-1` (the default) is unlimited and `0`
ends reasoning at once. Chat and Responses override it per request with
`reasoning_budget_tokens` (`-1` keeps the server default), and Messages with
`thinking.budget_tokens` when `thinking.type` is `enabled`. Tokens are counted
from the reasoning start, including the newline after Qwen's `<think>`. Once
the budget is spent and the last token completes its UTF-8 character, the
model's reasoning end is forced: `\n</think>` for Qwen, `</think>` for DeepSeek.
For an ASCII Qwen response that reaches the cap, N generated reasoning tokens
precede `</think>`. A natural end can use fewer tokens, and completing a UTF-8
character can require more. A Qwen `<tool_call>` also ends reasoning. The budget
covers the initial reasoning phase. Later `<think>` spellings remain literal
answer or tool-argument data, consistent with Gufo's response framing.

Chat and Responses also accept llama.cpp's older `thinking_budget_tokens`
spelling for existing clients; a non-null `reasoning_budget_tokens` takes
precedence. Request budgets must fit a signed 32-bit integer; Messages budgets
must also be positive. The server does not automatically reserve part of
`max_tokens` for the answer: clients must set a smaller reasoning budget when
they need that room.

Keep `reasoning_effort` (Chat) or `output_config.effort` (Messages) consistent
across turns while thinking is enabled: Qwen and DeepSeek render the effort
instruction into the prompt, so changing it changes the prompt prefix and can
force a full conversation prefill.

`POST /v1/chat/completions` accepts top-level `reasoning_effort` (`none`,
`minimal`, `low`, `medium`, `high`, `xhigh`, or `max`) and Pi/llama.cpp-style
`chat_template_kwargs`:

```json
{
  "reasoning_effort": "high",
  "chat_template_kwargs": {
    "enable_thinking": true,
    "reasoning_effort": "high",
    "preserve_thinking": false
  }
}
```

Pi's native DeepSeek request shape is also accepted:

```json
{
  "thinking": {"type": "enabled"},
  "reasoning_effort": "high"
}
```

`thinking.type` accepts `enabled` or `disabled`. DeepSeek also accepts
`thinking_mode` (`thinking`, `chat`, or `auto`) in `chat_template_kwargs`.
Conflicting controls return `invalid_reasoning`. In template kwargs,
`enable_thinking=false` suppresses the configured effort, matching Qwen's
Jinja; a non-off top-level `reasoning_effort` explicitly enables reasoning.
Generated reasoning is returned as `reasoning_content` in ordinary and
streaming Chat Completions responses. Per-model effort mappings and history
policies are documented in the model cards under `docs/models/`.

The server uses compiled model-specific formatters and validates recognized
artifact template hashes during model loading. It does not accept custom Jinja
or claim to enforce a reasoning-token budget.

Diagnostic telemetry is limited to status, token counts, timing, and
cancellation state. It must not contain prompt text, model paths, machine
identity, request IDs, or token IDs.

Pass the supported GGUF or safetensors directory through `--model`, with the
matching projector or speculative sidecar where needed. Loaders validate
tensor inventory, dimensions, quantization and tokenizer metadata. Model
artifacts supply weights and metadata; executable kernels and templates are
compiled into Gufo. See [model cards](../models/README.md) for concrete paths
and supported modes.

## Process and State Ownership

HTTP workers submit requests through `TextGenerationBackend`. The text
scheduler owns request state and serializes model execution. Model runners own
weights, tokenization, prefill/decode, speculative verification and complete
cache snapshots. HTTP handlers do not implement model kernels.

## Model router

`gufo router` exposes one OpenAI-compatible front door for several preset
models across the five `serve` modalities. It spawns `gufo serve
<modality>` workers on loopback on demand, reverse-proxies HTTP, SSE,
WebSocket and multipart traffic to them, and unloads workers that have
been idle or are displaced by `--models-max`.

```text
gufo router --models-preset <FILE>
            [--models-dir <DIR>]
            [--models-max N]            default 2; 0 = unlimited
            [--sleep-idle-seconds SEC]  default 900; 0 disables idle unload
            [--load-timeout-seconds S]  default 600; 0 = unlimited wait
            [--autoload]                preload preset models at startup
            [--remote-allow LIST]       routers [remote/<id>] may name
            [--host IP] [--port N] [--api-key KEY]
            [--max-connections N] [--max-request-bytes N]
            [--log-level LEVEL] [-v]
```

Front server options use the same names and defaults as `serve`, including
the `HOST`, `PORT`, `GUFO_HOST` and `GUFO_PORT` environment fallbacks. The
router injects `GUFO_HOST=127.0.0.1` and a reserved loopback `GUFO_PORT`
into every child, so workers bind loopback and run keyless; the front
enforces `--api-key` and never forwards a client `Authorization` header.
`gufo router --help` prints the preset grammar plus, per modality, the
accepted keys generated from the same option registrations that render
`gufo serve <modality> --help`.

### Preset file

Plain UTF-8 text. Full-line `#` comments and blank lines are ignored; a
value runs to the end of the line (no inline comments, no escapes).
Each model is one section:

```ini
# Full-line comments only (leading '#'). Blank lines ignored.

[llm/qwen3.8-27b]
model        = /models/llm/Qwen3.8-27B-UD-Q4_K_XL.gguf
speculative  = dflash2
dflash-model = /models/llm/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
sessions     = 2
cache-disk   = /var/cache/gufo/qwen27b

[asr/qwen3-asr]
model = /models/qwen3-asr
```

- The section header is `[<modality>/<model-id>]` with modality in
  `llm|image|video|tts|asr`. `model-id` is the routing key, must be unique
  across the file, and becomes the worker's `--served-model-name` default
  except for `video`, whose server has no such option.
- A body line `key = value` maps to worker argv `--key value` verbatim.
  Boolean flags use `key = true|false`: `true` passes the bare flag,
  `false` omits it. Keys are validated against that modality's registered
  long options only, so a new `gufo serve` option becomes available in the
  preset file without any router change.
- `host`, `port` and `api-key` are router-managed and rejected in
  sections.
- Relative path values resolve against `--models-dir`, or against the
  router's working directory when it is unset.
- Startup validation feeds every section through the modality's real
  argument parser; unknown keys, bad values, empty bodies and duplicate
  ids abort with the same message `gufo serve <modality>` would print,
  prefixed with the section name. `--models-max` must be `0` or at least
  `1`; with `--autoload`, presets beyond the limit are not an error: the
  first `--models-max` sections preload in section order and the rest load
  on demand.

### Routing and proxying

| Endpoint | Model source |
| --- | --- |
| `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`, `/v1/messages/count_tokens`, `/v1/images/*`, `/v1/audio/speech`, `/v1/audio/music`, `/v1/videos`, `/v1/embeddings` (also `/embeddings`, `/embedding`), `/completion`, `/apply-template` | JSON `model` field |
| `/v1/audio/transcriptions` (multipart) | `model` form field |
| `/v1/realtime`, `/v1/audio/speech/stream` (WebSocket) | `model` query parameter |
| `GET /v1/audio/voices`, `GET /props`, `GET /sdapi/v1/loras` | `model` query parameter (loads the model) |
| `/v1/videos/<id>` status/content | router job table |
| `GET /v1/audio/music/<id>` status/content (`"async": true`) | `model` query parameter, else router job table |

A missing model field or query parameter returns 400
`invalid_request_error` naming the missing field; an unknown model id
returns 404 `model_not_found` listing the preset ids. The front answers
`GET /health` and `GET /ready` (with the `/healthz`, `/readyz` and
`/v1/...` aliases) once preset validation passed and the listener is up,
independent of workers. `GET /v1/models` lists every preset model in
OpenAI list shape with the extension field `"loaded": true|false`.
`GET /v1/slots` and `GET /v1/metrics` require `?model=<id>`: they proxy to
that worker when loaded and return 409 `model_not_loaded` otherwise.
Unknown paths return 404 `not_found`, and video routes keep `serve`'s verb
and job-id precedence (405 `method_not_allowed`, 404 `video_not_found`).

Response bodies relay without re-serialization: SSE bytes pass through
verbatim, WebSocket upgrades complete against the worker and frames relay
in both directions until either side closes, and multipart bodies stream
to the worker under the front's `--max-request-bytes` bound. Closing the
front connection closes the worker connection, propagating cancellation.
Video job ids returned by `POST /v1/videos` are recorded with their
model: while a tracked job is non-terminal its worker is exempt from
eviction and idle unload, and a status or content read for a recorded job
reloads that worker if it died, answering from the persisted `--root`
job state.
Async music job ids (`POST /v1/audio/music` with `"async": true`) are
recorded the same way and keep their worker busy until a status read
reports `completed` or `failed` or the audio is downloaded. Music jobs live
only in the worker's memory, so a read for a job whose worker is no longer
loaded returns 404 `not_found` instead of starting the model. With
`--comfy-dir`, ComfyUI's `GET /embeddings` answers `[]`; a `POST
/embeddings` goes to the embedding model named in its body.
A ComfyUI graph that loads a video (`LoadVideo` -> `RIFE VFI`) takes
the RIFE rate from the file's probed frame rate (ffprobe next to
`--comfy-ffmpeg`; 24 fps assumed when it is missing): x5 of a 12 fps shot
is 60 fps, so a `CreateVideo` at 60 fps and a `SaveImage` of the same
graph name one output. Outputs of one video job must share the RIFE
multiplier and the upscale; target frame rates only have to agree with
each other within 0.5 % (59.94 ~ 60), and the error names what differs.

### Figure check (`POST /v1/figuren`)

The front answers `POST /v1/figuren` itself and asks the vision model
through its own `/v1/chat/completions` (default `qwen3.8-27b`, `"modell"`
overrides). The default method lists the people of a frame with the model,
assigns them by badge or position and rates each head against the figure's
reference (request and answer fields: header comment of
`src/cli/serve/router/figuren.hpp`).

`"verfahren": "ccip"` (1.4.5, opt-in) assigns by appearance instead and never
by position or badge: the anime face detector (YOLOv8n) finds the heads,
CCIP (deepghs CAFormer character fingerprints, CPU) compares each head with
every reference image of every figure (`kopf`, `front`, `bilder`, plus
`galerie` crops from other shots), the 27B only reads hard features
(moustache, full beard, glasses, hair) from the head and from each
`steckbrief`, and a candidate whose features contradict is dropped. A figure
is named only when its diff is below `ccip_cap` (default 0.05) and the next
remaining figure is at least `ccip_abstand` (default 0.0125) farther; otherwise the
person is `"unbekannt"`. Each person returns `name`, `box` (face),
`kopf_box`, `beste`, `zweitbeste`, `abstand`, `diff` per figure, `merkmale`,
`llm_gefragt` and a German `begruendung`. The models are converted once with
`src/models/ccip/reference/convert_weights.py`; the router finds them through
`--ccip-model-dir` and `--ccip-face-dir` (Windows defaults
`D:\llm-modelle\deepghs\ccip_onnx\gufo` and
`D:\llm-modelle\deepghs\anime_face_detection\gufo`), loads them on the first
ccip request (~165 MB host RAM) and answers 503 `ccip_unavailable` when they
are missing. `--ccip-threads` (default 8) bounds the CPU threads.

### Video concat (`POST /v1/videos/concat`)

The front joins several finished clips into one mp4 itself, with the ffmpeg
given by `--comfy-ffmpeg` (ffprobe is expected next to it). The body names
2 to 20 inputs: `{"videos": ["/v1/jobs/{id}/files/video.mp4", ...],
"output_format": "mp4", "reencode": false}`. Each input is a result file of
a completed job of this router (the path, or the same URL with scheme and
host; the job id is resolved locally and anything else is `400
invalid_input`) or a `data:video/mp4;base64,...` URL of at most 64 MiB
(the request body still has to pass `--max-request-bytes`). The front
probes every clip; when all share codec, size, frame rate, pixel format and
audio layout they are joined without re-encoding (concat demuxer, `-c
copy`), otherwise, or with `"reencode": true`, every clip is scaled and
padded to the first one, re-encoded with libx264 (`-crf 18`) and AAC, and a
clip without audio gets silence so the sound of the others is kept. The
direct call answers the mp4 (`Content-Disposition: attachment;
filename="video.mp4"`); wrapped in `POST /v1/jobs` (no `model` needed) the
result file is `video.mp4` and `response.json` lists the mode and the probed
inputs. Errors: `400 invalid_input` (bad list, unknown job, unreadable
clip), `500 concat_failed` with ffmpeg's last stderr lines. Temporary files
live under `<jobs-dir>/temp/` and are removed after every call.

### Job thumbnails (`GET /v1/jobs/{id}/thumb`)

A small preview of a job's result so a gallery page does not download the
full image or video for every card. `?w=` sets the width (64..1024, default
256), `?file=` picks another of the job's files than the first image, video
or audio file. Images and videos answer a JPEG of that width with the aspect
kept (a video gives the frame at 10 % of its length, or at 0.5 s when the
length is unknown); audio answers a waveform PNG of w x w/3 (ffmpeg
`showwavespic`, colour `#5b4fc4`, transparent background). The picture is
made with the ffmpeg from `--comfy-ffmpeg` and kept next to the job's files
as `thumb-<w>.jpg` / `.png` (`thumb-<w>-<name>` for `?file=`), so every
later call is served from disk (`X-Gufo-Thumb: cached` instead of
`generated`), with `Cache-Control: public, max-age=86400`. The thumbs
expire with the job's media (`--jobs-expire-hours`). Without ffmpeg the
route answers `501 thumb_unavailable`; an unknown job or file `404`, a
file that is no image, video or audio `400 thumb_unsupported`, an ffmpeg
failure `500 thumb_failed`. The job JSON (`GET /v1/jobs`, `GET
/v1/jobs/{id}`) carries `"thumb_url": "/v1/jobs/{id}/thumb"` when the job
has such a file and `null` otherwise.

### Pool relay (`GET /v1/jobs?scope=pool`)

With placement = pool the two routers hold separate job stores; one page
reads both through either router. `GET /v1/jobs?scope=pool` (or `pool=1`)
merges this router's list with the partner's (the remote twin of the pool
presets, asked with the same filters and a 3 s limit), newest first, each
job with `"owner": "local"` or the partner's `"http://host:port"` as
`/router/status` names it; the answer's `"partner"` field is that address,
`"unreachable"` when the partner did not answer (the local jobs still
come), or `"none"` without a partner. A job id this router does not know
is asked from the partner for `GET /v1/jobs/{id}`, `GET
/v1/jobs/{id}/files/{name}`, `GET /v1/jobs/{id}/thumb`, `PATCH`/`DELETE
/v1/jobs/{id}` and `POST /v1/jobs/{id}/keep|cancel`; the answer streams
through with its headers (10 s for JSON, 10 min for files), a partner that
cannot be reached is `502 partner_unreachable`. Relayed thumbs are kept
under `<jobs-dir>/relay-cache/<id>/` and removed a day after their last
use. Every relayed request carries `X-Gufo-Relay: 1`; a request with that
header (or `X-Gufo-Forwarded`) is answered locally and never relayed on,
so the two routers cannot loop.

### Shared queue (placement = pool, 1.4.20)

The two routers still hold separate job stores, and the router a job was
submitted to stays its owner (files, result, progress, webhook, relay);
since 1.4.20 the two queues behave as one: the execution of a job may
move to the partner, the oldest job of both routers runs first, and a
job only goes to a machine that can run it.

**Capabilities.** `GET /router/capabilities` answers what this router
can run: `{"router_id", "self", "partner", "models": {"<id>":
{"modality", "placement": "local|remote|pool", "switchable", "status",
"loaded", "loras": ["winkel", ...]}}}`. The LoRA names come from the
preset: every `lora =` line of the section, split at `;` (`lora =
a=p1;b=p2`, the one-line form gufo 2.0's image worker needs), the name
before `=` of each entry. The placement follows `POST /router/placement`. A job needs its model and the LoRA
names of its body (`loras[].name` for images, `gufo.lora` for videos);
the router reads the partner's capabilities through the same route
(cached 30 s, 3 s limit). When only one router can run a job, the
job's request carries `X-Gufo-Pool: local` or `remote` and the pool
decision keeps to that side (it waits there when that machine is busy
instead of taking the other); when neither can, `POST /v1/jobs` answers
`400 no_capable_router` with a message naming what each router has
(`this router has no LoRA 'nix' for model 'qwen-image' (registered:
anime); partner http://192.168.100.56:8081 has no LoRA 'nix' ...`). A
direct request (not a job) to a pool model is checked the same way. A
partner whose capabilities cannot be read (unreachable, older version)
counts as "may have it". An explicit `X-Gufo-Pool: local|remote` on a
request is never overridden by the LoRA check: when the side it names
lacks a LoRA of the request the answer is `400 pool_side_cannot_run`
(message names the LoRA and what that side has), never a silent relay
to the other side.

**Order.** A free runner (a job slot of `--jobs-parallel`) compares its
oldest queued job with the partner's (`GET /v1/jobs?status=queued` and
`?status=in_progress` for the waiting ones, cached 2 s). When the
partner's is older and this machine can run it,
the runner calls `POST <partner>/router/pool/pull` with `{"from":
"http://host:port", "router_id", "job": <the id it wants>,
"capabilities": <its own>}`. The partner, under its lock, hands out its
oldest queued job the requester can run (never a job a runner already
took, never a concat job, only models the partner has in the pool),
answers `{"job": id, "model", "created_at", "queued": n}` or `{"job":
null, "queued": n}`, and executes the job on a thread of its own with
`X-Gufo-Pool: remote`, so its pool decision relays the request to the
requester, whose worker does the work; the owner keeps files, result,
progress and webhook as for any job. The requester's runner waits for
that job (polling the owner every 2 s) before it takes the next local
job. A pulled job that finds no machine (`503 pool_busy`,
`remote_unavailable`) goes back to the owner's queue once, like any
job. Every pull and poll carries `X-Gufo-Relay: 1`. A partner that
answers `404` on the pull route (older than 1.4.20) pauses pulling for
60 s (logged once, `event=pool_pull_unsupported`); an unreachable
partner means "as before" (local jobs run locally). Log events:
`pool_pulled` (requester), `pool_pushed` (owner), `pool_pull_failed`,
`pool_pull_lost`.

**3D jobs (1.4.23).** `POST /v1/jobs` with `request.path`
`/v1/3d/generations` wraps a `gufo serve objekt` worker: the router posts
the request, polls `GET /v1/3d/generations/{id}` every `video_poll`
(progress copied, `DELETE` on cancel) and stores `model.glb`
(`model/gltf-binary`), `preview.png` and `response.json` as the job's
files. `GET /router/capabilities` reports `"capability": "image_to_3d"`
for presets whose command line is `serve objekt`, so the pages can tell
which router runs the worker.

1.4.23c: status, content, preview reads and the cancel on `/v1/3d/generations/<id>` resolve the worker by the `model` query or, without it, by the recorded job owner (the router used to know that only for music jobs and answered 404).
1.4.24 (2.0.1): the router records the job owner of a direct `POST /v1/3d/generations` like a video job: reads without `?model=` find the worker, the open job keeps the worker busy (not stopped for another model while it computes), the job sweep asks the 3D status path, and a 3D job stays on the machine it runs on when the placement changes. The job store's polls carry the job and pool-side headers, tolerate up to 5 transient answers in a row (5xx, connection error, no JSON) and give up after 30 minutes (`objekt_timeout`, the worker job is cancelled).

**Waiting jobs (1.4.21).** With `--jobs-parallel 2` a job is
`in_progress` at once, and its request may then wait in the router's
pool for a free machine (`event=pool ... choice=wait reason="both
busy"`), so the queue alone would rarely hold anything to pull. Every
request of a job carries `X-Gufo-Job: <id>`, and the pool records per
job where the request went: `wait` (nothing started), `local` or
`remote`. A waiting job counts as queued for the shared queue (position,
`pool_queue`), shows `"running_on": null` and `"pool_wait": true`, and
is a pull candidate: the owner marks it, the pool loop answers the
waiting request `503 pool_redirected` instead of taking a machine
(`event=pool_redirected`), and the owner's runner sends the same request
again with `X-Gufo-Pool: remote` (`event=job_redirected`). A request
that already reached a worker is never redirected, so nothing runs
twice. The pull answer says `"was_waiting": true|false`.

**Priority (2.0.3).** Every job has `"priority"`: an integer 0..9, 0
(normal) by default, a higher value runs first. Given on `POST
/v1/jobs` as the header `X-Gufo-Priority` (wins when both are there),
the member `"priority"` of the submit body, the member `"priority"` of
the wrapped request body or a multipart form field `priority`; any
other value (not an integer 0..9, as a number or a digit string) is
`400 invalid_priority`, nothing is queued. Order: the queue of each
router runs the highest priority first and, among equal priorities,
the oldest job (first in, first out, as before); a job put back after
`remote_unavailable`/`pool_busy` returns to the front of its priority.
Pull: a free runner takes the partner's first job (priority, then age)
only when it runs before its own next job, i.e. `(priority desc,
created_at asc)` replaces the plain `created_at` comparison; the owner
hands out its best candidate in the same order and names `"priority"`
in the pull answer. `"position"` follows the same order. Every request
of a job carries `X-Gufo-Priority` (only when > 0), so the pool
decision on both routers knows it. `jobs.json` keeps `"priority"`; a
file from before 2.0.3 reads as 0; a partner older than 2.0.3 lists no
priority and its jobs count as normal, and its own queue stays by age.

Direct (synchronous) requests to a pool model take the priority from
the header or the body member too (`400 invalid_priority` as above;
the body is passed on unchanged, workers ignore the member). The pool
loop forms no order among several waiting requests: each polls every
200 ms and takes the first free machine it sees. The precedence rule
is therefore small: a request that sees a free machine (local or
remote) and finds a waiting request of the same model with a higher
priority lets it go first (`event=pool_yield`, `choice=wait
reason="yields to a higher priority"`), for at most 2 s in a row
(`kPoolYieldMax`): the higher one polls within 200 ms and takes the
machine; if it does not (pinned to the other side, gone), the lower one
takes it, so nothing waits longer than before (B5 holds; the yield
counts against the request's `queue-timeout-seconds`). A relayed
request (`X-Gufo-Forwarded`) never yields; the partner decided already.
Not covered: priorities across different models (a free GPU still goes
to whichever model's request sees it first), the worker's own queue
once a request reached it, and ComfyUI graphs. `/router/status` shows
per pool model `"waiting_priority"` (the highest waiting priority, -1
when none) and `"yields"`.

**View.** Every job carries `"request": {"loras": [...]}`, `"priority"`,
`"pool_wait"` and `"running_on"`: `"local"` while it runs on the owner's
worker, the partner's `http://host:port` while the partner's worker runs
it (pulled, or the pool sent the request there), `null` while it waits
in the pool or is not running. `GET /v1/jobs?scope=pool` adds
`"position"`: the rank among the queued and waiting jobs of both routers
by priority, then `created_at` (1 = next to run), `null` for the others;
the partner's `running_on` is translated to this router's point of view. `GET
/router/status` gets `"pool_queue": {"local": queued or waiting here,
"local_waiting": of those waiting in the pool, "partner": queued or
waiting there (null while unknown), "pulled": jobs this router fetched
from the partner, "pushed": jobs handed to the partner, "pulling": [ids
being run here for the partner], "pushed_active": [ids the partner runs
for this router], "pull_hold_s" (only while pulling is paused),
"partner_name", "enabled"}`.

**Self URL (1.4.22).** A pull's `"from"` and `/router/capabilities`
`"self"` never name `0.0.0.0`: a router bound to all interfaces names
the local address of the route toward the partner (else the computer's
name), and the pull owner replaces an unreachable `"from"` with the
partner URL it knows from its `[remote/...]` section (`running_on`,
log).

Limits: the pool has exactly one partner; the image preview of a pulled job runs on the requester's
worker, so the owner shows no live step for it; a job with a multipart
body states its LoRAs only through a `loras` JSON part.

### Pool failures (after 1.4.24)

**Partner failure.** When the partner answers a relayed pool request
`502` (its worker did not start), `503` (`upstream_unavailable`,
`pool_busy`, ...) or `504` (its load timed out), nothing has reached the
client yet, so the pool decides once more without the partner (this
machine, or wait for it), like it already did for `507` and for `503
remote_unavailable`. Video, music and 3D creates are not retried;
requests pinned with `X-Gufo-Pool: remote` only when this router has
the model and every LoRA they name (2.0.2). The partner then gets no
pool request of that model for 60 s, also when it could not be reached
at all (connect refused, timeout; 2.0.2) (`event=pool_remote_failed`;
pinned requests ignore the hold); `/router/status` and
`/router/placement` show `"remote_failed_s_ago"` and `"remote_failed"`
(status and detail) in the model's `pool` block.

**Never stuck (2.0.2).** A pool request that this router can run (it
serves the model with every LoRA the request names) waits on the partner
at most `pool-wait-seconds` (remote section, default 60): after that the
partner drops out of its decision and a pin to it (`X-Gufo-Pool:
remote` after a pull or redirect) no longer holds. The request then runs
here as soon as this machine can load or run the model (evicting idle
models; a busy local worker is still waited for, oldest first), and
`queue-timeout-seconds` (900) is only the last resort. Relayed requests
(`X-Gufo-Forwarded`) are left to the partner. Every switch to this
machine because of the partner (waited too long, unreachable, failed,
out of memory) logs `event=pool_fallback reason="..."` and counts in the
`pool` block's `"fallbacks"`.

**Waiting is not busy.** The `"activity"` object of a pool model in `GET
/v1/models` (what the partner's probe reads) reports `"busy"` only for
work of the model that runs or loads here (a request, socket or job, or
a pool request that picked this machine), and apart from it `"waiting"`:
the number of pool requests of the model that wait for a machine. The
partner counts only `busy` as occupied, and `"others_busy"` leaves out
other models' waiting requests too, so two requests pinned to the other
side (`X-Gufo-Pool: remote`) no longer wait for each other until the
queue timeout (12.11.: 900 s). A router older than this reports waiting
requests as busy; the field is then missing and reads as 0.
`/router/status` copies `waiting` into each remote router's
`activity`.

**Pull only with a free GPU.** A free job runner pulls a partner job
only when this machine's GPU can take it now: the job's model is not
busy here, no other GPU work runs (ComfyUI queue, an image, video, music
or 3D model with work in flight) and no pool request is claimed or
waiting here. Before, an idle runner was enough, and a machine busy with
ComfyUI pulled jobs that then waited in its pool.

**Load failures in the capabilities.** Each model in `GET
/router/capabilities` carries `"available": true|false`; `false` with
`"unavailable": "local load failed"` while a pool load or admission
failure holds (60 s) or `"autoload failed: <error>"` while a failed
autoload waits for its retry. Pulls honour it on both ends: the
requester does not pull such a model and the owner hands it no job for
it. Submitting a job and the pool's LoRA check stay capability-only, so a
passing failure never turns into `400 no_capable_router`. A router older
than this sends no field, which counts as available.

### Lifecycle

A worker spawns on the first request for its model; requests for the same
model queue behind readiness, which is the worker's `/health` answering
200 (anything else, including a refused connection, means still loading). A
worker becomes unloadable
`--sleep-idle-seconds` after its last activity, measured from both
request start and request end, including full SSE streams and open
WebSocket sessions; unload is SIGTERM followed by SIGKILL after a 10
second grace. At `--models-max` the least-recently-active unloadable
worker is evicted for a new load; when every worker is busy, the
new-model request waits in a FIFO queue for the next freed slot instead
of being refused. Workers set `PR_SET_PDEATHSIG`, so they die with the
router even after an unclean exit.

| Condition | Behavior |
| --- | --- |
| Bad preset (unknown key, bad value, duplicate id, reserved key) | Startup abort, exit 2, section-prefixed parser message |
| Worker load fails (non-zero exit) | Held requests 502 `upstream_unavailable`, model unloaded, next request retries |
| Worker dies mid-request | Streams close as with today's `serve` crash; model unloaded |
| `--models-max` reached, nothing unloadable | New-model requests queued FIFO for the next freed slot |
| Cold load (queue + spawn + readiness) exceeds `--load-timeout-seconds` | Held and queued requests 504 `load_timeout`, worker killed, model unloaded, next request retries |
| Idle timeout / eviction | Drain-then-kill; no in-flight request is ever cut |
| GPU `device_lost` in a worker | Existing worker exit path; the router treats it as death |

On `SIGINT`/`SIGTERM` the front stops accepting, refuses new loads, keeps
proxying loaded models while in-flight work drains for up to 10 seconds,
then terminates every worker and exits 0.

### Remote models

A `[remote/<model-id>]` section names a model that another gufo router
(1.2 or newer, for the async music routes) serves under the same id, for
example a second machine on a direct cable. The local router starts
nothing for it: it relays the requests and leaves loading, unloading and
memory to the remote router.

```ini
[remote/qwen-image-2.1]
url = http://192.168.100.56:8081
modality = image
api-key-env = GUFO_REMOTE56_KEY
connect-timeout-seconds = 3
gpu-slot = studio56
busy-check-url = http://192.168.100.56:8085/queue
queue-timeout-seconds = 900
max-request-bytes = 67108864
```

| Key | Meaning |
| --- | --- |
| `url` | `http://<IPv4>:<port>` of the remote router; no host names, https or path. Required |
| `modality` | `llm`, `image`, `video`, `tts`, `asr`, `embedding` or `omni`, as `/v1/models` lists it. Required |
| `api-key-env` | Environment variable holding the remote router's `--api-key`; sent as `Authorization: Bearer`. Unset variable: exit 2 |
| `connect-timeout-seconds` | Connect bound (default 3, 1..300) |
| `head-timeout-seconds` | Response-head deadline after the request is sent (default 0 = none) |
| `gpu-slot` | Remote models sharing the name run one GPU-starting request at a time |
| `queue-timeout-seconds` | Longest wait for the slot and the busy check (default 900) |
| `pool-wait-seconds` | placement = pool: longest wait on the partner before a request this router can run falls back here (default 60) |
| `busy-check-url` | Optional `http://<IPv4>:<port>/path` (a ComfyUI `/queue`); wait while it reports running or pending work |
| `max-request-bytes` | 413 `request_too_large` above this body size |

The router refuses to start (exit 2) unless `--remote-allow` covers every
`url` and `busy-check-url`: comma-separated IPv4 addresses or
`a.b.c.d/n` ranges, each with an optional `:port`, or `*`; the default is
empty. Router hints (`router-*`), worker options, `api-key` and
`remote-model`/`served-model-name` are rejected in remote sections; the
id must be the same on both routers.

- Requests resolve to the model as in the table above and relay with the
  body unchanged; the client's `Authorization` is never forwarded. Video
  and async music job ids are recorded, so status, content and DELETE
  reads route back. Previews (`/v1/previews/...`) and `/models/load|unload`
  go to the remote router; the ComfyUI API and the job API use remote
  models like local ones. ComfyUI `/free` stays local.
- A remote model is always `unloaded` locally: it never counts for
  `--models-max`, LRU, idle unload, the memory guard or `memory.json`, and
  `--autoload` skips it. `/v1/models` lists it with `"remote": {"url",
  "reachable"}`, empty args and the state the remote last listed.
- A background probe asks every remote router for `GET /v1/models` every
  5 s (2 s bound). A failed connect or probe answers further requests
  with 503 `remote_unavailable` and `Retry-After: 30` at once until a probe
  works again; nothing local restarts.
- With `gpu-slot`, every POST (and `/models/load`) waits for the slot in
  arrival order; a video or async music job keeps it until it ends (the
  probe polls its status). After `queue-timeout-seconds` the answer is 503
  `remote_busy`. With `busy-check-url` the request also waits while that
  queue is busy; an unreachable check counts as free.
- WebSocket routes (`/v1/realtime`, `/v1/audio/speech/stream`) answer 501
  `remote_websocket_unsupported`.
- Two routers may name each other (load sharing both ways). Every relay
  (requests, previews, `/models/load|unload`, probes, job polls) carries
  `X-Gufo-Forwarded: <router id>` (random per process, shown as
  `remote.router_id` in `/router/status`). A request that arrives with this
  header and names a model that is remote here as well is not relayed again:
  it gets 508 `remote_loop`; local models serve it normally. A client that
  sets the header itself only refuses its own relay. Probes mark ids the
  other router lists as remote: `/v1/models` shows them unloaded with
  `"remote": {"loop": true}` and `/router/status` lists them as `remote`.
- `GET /router/status` adds a `remote` block: routers (reachable, error,
  last check, model states), models (url, busy, gpu slot, placement),
  relays in flight, gpu slots (holder, waiting), jobs holding a slot and
  the last busy-check answers.

#### Switching a model between local and remote

A local section and a `[remote/<id>]` section with the same id make the
model *switchable*: it can run here or on the other router, and
`POST /router/placement` moves it while the router keeps running. In the
remote section `modality` may be left out (it must match the local one
when given) and `placement` sets the start value:

```ini
[llm/qwen3.8-27b]
model = models/qwen3.8-27b.gguf

[remote/qwen3.8-27b]
url = http://192.168.100.56:8081
api-key-env = GUFO_REMOTE56_KEY
gpu-slot = studio56
placement = local        # local (default) | remote
```

```sh
curl -X POST http://127.0.0.1:8081/router/placement \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.8-27b", "placement": "remote"}'
curl -X POST http://127.0.0.1:8081/router/placement -d '{"all": "local"}'
curl http://127.0.0.1:8081/router/placement      # current placements
```

The answer lists every switchable model (`placement`, `local_status`,
`local_busy`, `remote_busy`, `remote_url`, `remote_reachable`) and the
ids that `changed`. Errors: 400 `invalid_placement`, 404
`model_not_found`, 409 `not_switchable`. The endpoint is protected like
`/models/load`: by the router's `--api-key` when one is set.

- New requests follow the placement at once. Nothing running is cut off:
  local requests, WebSockets and jobs finish on the local worker, relays
  finish on the remote router.
- Moved to `remote`: once no local request, socket or job has been open
  for 2 s, the supervision tick unloads the local worker (a worker that
  is still loading finishes its load first). Relays to the other router
  never count as local work.
- Moved to `local`: the model loads on the next request as usual
  (`--autoload` only loads switchable models that start `local`).
- Video and async music job reads (status, content, DELETE) and job
  previews follow the job to where it was created, whatever the
  placement is now.
- While a model is `remote`, everything above applies: gpu slot, busy
  check, 503 `remote_unavailable`, 508 `remote_loop` (a relayed request
  for a model that is `remote` here as well). `/v1/models` shows
  `"switchable": true`, `"placement"` and `"remote_url"`; the `remote`
  block (which other routers read as "relayed here") appears only while
  the placement is `remote`.
- Switchable models count for the memory guard and `memory.json` like
  local ones. The placement is not saved: after a restart the preset's
  `placement` applies again.

### Users and access

Since 2.0.3 the router front can require a login (gufo-studio-windows
only; upstream gufo has no authentication). Everything lives in the
preset's optional `[router]` section and is off by default, so an
existing preset behaves exactly as before.

```ini
[router]
auth            = off        # off | lan | all
auth-loopback   = admin      # admin | login
users-file      = nutzer\nutzer.json
pool-secret-env = GUFO_POOL_SECRET
```

| Key | Default | Meaning |
| --- | --- | --- |
| `auth` | `off` | `off`: everyone may use everything (today's behaviour); the users API and `/verwaltung/` answer from `127.0.0.1` only. `lan`: `127.0.0.1` is admin (see `auth-loopback`), everyone else logs in. `all`: the machine itself logs in too (strangers at the keyboard). |
| `auth-loopback` | `admin` | With `auth = lan`: whether `127.0.0.1`/`::1` counts as the admin `lokal` without a login (`admin`) or must log in like the LAN (`login`). With `admin` the users API and `/verwaltung/` are open from `127.0.0.1` without a login (emergency path: a forgotten admin password is reset at the machine); with `login` they need an admin session like everything else, and the emergency path is the preset (set `auth-loopback = admin`, restart). |
| `users-file` | `nutzer\nutzer.json` next to the preset | The users file; a relative path resolves against the preset's directory. `nutzer\filter-woerter.txt` and `nutzer\zugriff.log` live beside it. |
| `pool-secret-env` | none | Environment variable holding the secret both routers share. Every router-to-router call (model probe, relay, capabilities, job relay, `/router/pool/pull`) carries it as `X-Gufo-Pool-Key`; a request with a wrong or missing key is answered `401 pool_key_invalid`, which the partner's probe reports as unreachable (the B5 fallback then runs the work locally). Set the same value in both presets and both environments. |

Order per request: free paths (`/health`, `/ready`, `/login/` and its
files, `/gemeinsam/` files, the favicons) are served to anyone. Then the
router's own calls (jobs runner, figure check, ComfyUI adapter: a 256-bit
per-process key, whatever address `--host` makes them arrive from), the
partner (pool key: without `X-Gufo-User` it is the partner router itself,
trusted like the router's own calls; with it, the relayed user),
`Authorization: Bearer gufo_...` (API tokens), the `gufo_sitzung` cookie (HttpOnly,
SameSite=Lax, 12 h; "stay signed in" 30 days, stored hashed in the users
file so it survives a restart), and finally loopback as admin. Anything
else gets `401 not_authenticated` (JSON) or, for a page, `302` to
`/login/?zurueck=<page>`. A bearer that does not match any token is `401
invalid_token`; after 5 failed logins from one address within a minute
that address waits 60 s (`429 too_many_attempts`, `Retry-After`).

Roles: `admin` (everything, including the users API, model load/unload,
`POST /router/placement`, `/router/pool/pull` and every job), `nutzer`
(the enabled areas, own jobs only) and `dienst` (programs: tokens only,
no password login, rights like `nutzer`). Rights per user, all optional:

| Field | Default | Effect |
| --- | --- | --- |
| `bereiche` | all | Subset of `text`, `bild`, `video`, `stimme`, `musik`, `3d`, `ablage`, `pool`, `status`. A request outside the user's areas is `403 forbidden` ("Bereich bild ist für diesen Nutzer nicht freigegeben"); a page of a closed area is a small 403 page. |
| `modelle` | all | Allowed model ids; another model is `403 model_forbidden`. |
| `filter` | role (admin off, others on) | Word block on the prompt and negative prompt of `/v1/images/*`, `/v1/videos*` and the ComfyUI `/prompt` graph (JSON and multipart): `400 content_filtered`. The list is `nutzer\filter-woerter.txt`, one entry per line, `*` at the end matches word forms; a default list (~28 words, de/en) is written when the file is missing and edits are picked up without a restart. |
| `tageslimit` | 0 (unlimited) | POST requests on generation endpoints per UTC day, videos count twice: `429 daily_limit`. Counted in memory (a restart resets it). |
| `pool` | true | `false` pins the user's requests to this machine (`X-Gufo-Pool: local`); a job submitted so is marked `pin_local` (view `"pinned": "local"`), runs here only, is never handed to the partner and is refused `400 pool_not_allowed` when only the partner could run it. |

Areas by endpoint: text = chat, completions, responses, messages,
embeddings, rerank, realtime, `/v1/figuren`; bild = `/v1/images/*` and
the ComfyUI adapter (`/prompt`, `/upload/*`, `/history`, `/view`, `/ws`,
...); video = `/v1/videos*`; stimme = speech, transcriptions, convert,
voices, separate; musik = `/v1/audio/music*`, `/v1/audio/foley`; 3d =
`/v1/3d/*`; ablage = `/v1/jobs*`; `/v1/previews/*` needs bild or video.
`GET /v1/models`, `/router/status`, `/router/capabilities`, `GET
/router/placement` and `/v1/names` are open to every signed-in user (the
pages need them); `pool` and `status` gate the pages of those names
(`/pool/`, `/speicher/`, `/status/`, ...).

Jobs belong to a user: the job API stamps `metadata.nutzer` from
`X-Gufo-User` (set by the gate on every request it lets through; a
client's own copy is dropped), a job's inner request is checked like a
direct one (area, model, filter, limit; a `body_base64` multipart body is
decoded for that), `GET /v1/jobs` lists only the user's jobs, a read,
file, patch or delete of another user's job is `404`, bulk delete is
admin only. Admins see everything. Relayed and pulled requests carry
`X-Gufo-User` and `X-Gufo-Rights` (`rolle=...;bereiche=...;modelle=...;
filter=...;tageslimit=...;pool=...`) set by the entry router; the partner
trusts the rights when the pool key is valid (no second copy of the users
file), stores a job it takes under that user, and applies the own-jobs
rule to the relayed user's job requests itself (a relayed `nutzer` sees,
cancels or deletes only its own jobs there too; a relayed `admin` sees
all). The entry router's own jobs runner runs each job as its owner (the
job header).

Endpoints (JSON):

| Endpoint | Who | What |
| --- | --- | --- |
| `GET /router/me` | anyone | `{auth, name, rolle, quelle, bereiche, modelle, filter, tageslimit, pool, verwaltung, setup}`; `401` with `setup: true` while no admin exists. |
| `POST /login` | anyone | `{"name", "passwort", "bleiben"}` -> `200` + `Set-Cookie`, `401 login_failed`, `403 login_not_allowed` (dienst), `429 too_many_attempts`. |
| `POST /logout` | anyone | Ends the session, clears the cookie. |
| `GET/POST /router/users` | loopback or admin | List (no hashes) / create `{"name", "rolle", "passwort", "rechte"}`. |
| `GET/PUT/DELETE /router/users/<name>` | loopback or admin | Read / change `{rolle, rechte, passwort, gesperrt}` / block; `?loeschen=1` removes the user. |
| `POST /router/users/<name>/tokens` | loopback or admin | `{"name", "tage"}` -> the token `gufo_<64 hex>` once; only its SHA-256 is stored. |
| `DELETE /router/users/<name>/tokens/<token>` | loopback or admin | Revokes the token. |
| `GET /router/zugriff?n=N` | loopback or admin | Last N lines of the access log. |

Everyone else gets `404` on the users API and on `/verwaltung/`, so the
pages are invisible from outside. Writes to the users API must come as
`application/json` (`415 json_required`) and not cross-site: a request
with `Sec-Fetch-Site: cross-site`, or an `Origin` that is not this
router's `Host`, is `403 csrf_rejected`, so a web page opened on the
router machine cannot create users through the loopback admin path.
With `auth = off` the gate still drops a client's `X-Gufo-User`,
`X-Gufo-Rights` and `X-Gufo-Internal` headers.

ComfyUI graphs (`POST /prompt`) are resolved to the adapter's tasks
before they run: each task is checked like a direct request (its area
and model, the word filter on image, video and foley prompts) and all of
them count against the daily limit (videos twice). `/props`,
`/sdapi/v1/loras` and `/v1/slots` load the model they name and count as
work of that model's area (text, bild, text). Beyond the job store the
gate remembers for a day who submitted each ComfyUI prompt and each
direct video job (from the answers, in memory): `GET /history` shows a
user its own prompts, `POST /history` deletes only its own,
`/v1/videos/<id>[/content]` and the job files a `POST /v1/videos/concat`
names answer `404` for another user's; an id the gate never saw (before
a restart, or created by the router itself) is open. `/view` is not
checked (ComfyUI file names carry no owner). Idempotency keys are per
owner. The users file (`{"version": 1,
"nutzer": [...]}`) holds per user the name, role, PBKDF2-SHA256 hash
(600 000 rounds, 16-byte salt), rights, token hashes with name and
expiry, remembered session hashes, timestamps and `gesperrt`; it is
written whole after every change (temporary file + rename). Files from
older builds without some fields load with defaults. The access log
`nutzer\zugriff.log` has one tab-separated line per login, refusal and
generation request (UTC time, user, address, method and path, model,
result) and never a prompt; it rotates daily to `zugriff-<day>.log` and
keeps 30 days.

First setup: with `auth = lan` and no admin in the file the LAN gets the
login page with the hint "no admin yet"; create the first admin from the
machine itself (`/verwaltung/` on `http://127.0.0.1:<port>/` or `POST
/router/users`), then create the users and the `dienst` tokens for the
film pipeline and the watchdogs, set the pool secret on both machines and
switch `auth` on in both presets. Stages: `off` (today), `lan`, `all`;
TLS stays a reverse proxy's job.

Pages: `/login/`, `/verwaltung/` (users, rights, passwords, tokens, log;
de/en, dark mode) and `gemeinsam/anmeldung.js`, which every page loads
after `sprache.js`: it reads `/router/me`, shows the name and "Abmelden",
hides tiles and menu entries of closed areas and goes to `/login/` on
`401`.


## Endpoint Set

### Text endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/models` | List loaded model aliases and capabilities |
| `POST` | `/v1/responses` | Text/images, structured output, optional SSE streaming |
| `POST` | `/v1/chat/completions` | Main chat, streaming, image and tool API |
| `POST` | `/v1/completions` | Optional legacy text completion adapter |
| `GET` | `/health` | Process liveness and GPU context (aliases: `/v1/health`, `/healthz`, `/`) |
| `GET` | `/ready` | Model and backend readiness (aliases: `/v1/ready`, `/readyz`) |
| `GET` | `/metrics` | Prometheus-format operational metrics (text LLM serving only) |

### Other model services

| Method | Path | Condition |
| --- | --- | --- |
| `POST` | `/v1/embeddings` | EmbeddingGemma 2 serving is configured with `gufo serve embedding`.  A text server answers 501, and other servers answer 404 |
| `POST` | `/embeddings`, `/embedding` | llama.cpp's non-OpenAI embeddings form, on the same `gufo serve embedding` process; other servers answer 404 |
| `POST` | `/v1/audio/transcriptions` | An STT implementation is compiled and loaded |
| `POST` | `/v1/audio/speech` | A TTS implementation is compiled and loaded |
| `POST` | `/v1/audio/music` | ACE-Step 1.5 music serving is configured with `gufo serve music` |
| `GET` | `/v1/realtime` | `serve asr`: transcription sessions, and spoken conversations for models that hear and speak |
| `GET` | `/voice` | `serve asr` with a conversational model: a browser client of `/v1/realtime` |
| `POST` | `/v1/systemone` | A decision model is configured with `gufo serve decision` |
| `POST` | `/v1/images/generations` | Qwen-Image-2.1 image serving is configured |
| `POST` | `/v1/images/edits` | Qwen-Image-2.1 image serving is configured |
| `POST` | `/v1/videos` | A validated operator-supplied MiniMax H3 checkpoint is configured |
| `GET` | `/v1/videos/{id}` | MiniMax H3 video serving is configured |
| `GET` | `/v1/videos/{id}/content` | The requested MiniMax H3 job completed |
| `DELETE` | `/v1/videos/{id}` | MiniMax H3 video serving is configured |
| `POST` | `/v1/images/upscale` | `gufo serve upscale` (Real-ESRGAN on the NPU, GPU or CPU); sync, or `"async": true` for a job read at `/v1/images/upscale/{id}[/content]` |
| `POST` | `/v1/videos/upscale` | `gufo serve upscale`: async video job (frames through FFmpeg, audio copied); status and MP4 at `/v1/videos/{id}[/content]`, `DELETE` cancels. See [Real-ESRGAN](models/realesrgan/README.md#images-and-videos-on-the-npu-gpu-or-cpu-gufo-serve-upscale) |
| `POST` | `/v1/3d/generations` | `gufo serve objekt` (TripoSG image-to-3D, 1.4.23): JSON `{model, image (base64 or data URL, PNG with alpha), steps, guidance, seed, resolution 256\|512, max_faces, preview}` answers 202 with a job snapshot and a `Location` header |
| `GET` | `/v1/3d/generations/{id}` | Snapshot `{status, progress, phase, data[0]: {vertices, faces, bbox, closed, volume, seconds}}` |
| `GET` | `/v1/3d/generations/{id}/content` | The finished GLB (`model/gltf-binary`, `Content-Disposition: attachment`) |
| `GET` | `/v1/3d/generations/{id}/preview` | Three-view preview PNG of the mesh |
| `DELETE` | `/v1/3d/generations/{id}` | Cancels or removes the job |

As in llama.cpp, vLLM, TEI and Ollama, these routes ignore request fields they
do not know, so a client that adds one still works. Known fields whose behavior
a route cannot provide, such as transcription timestamps or video image
conditioning, are refused with 400 rather than dropped.

Use `gufo serve image --model SNAPSHOT_DIR` for
[Qwen-Image-2.1](../models/qwen-image-2.1.md). Generation accepts JSON;
editing accepts multipart PNG/JPEG references. Both return base64 PNG data
and work through llama-swap's image routes.

Qwen3-TTS serving is enabled with a dedicated model process. All three 12Hz
1.7B variants are supported; the variant is detected from the checkpoint's
`tts_model_type` and determines both the advertised model id and the request
fields that are required:

```sh
./result/bin/gufo serve --port 8080 tts \
  --model /persist/models/audio/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

Start ASR separately with `gufo serve asr --model DIR`. Both speech
commands accept `--model`, `--context`, and `--served-model-name`, plus the
shared HTTP options. Context defaults are 4096 for TTS and 1024 per chunk
for ASR. Route separate model processes through llama-swap when one public
API URL should offer both synthesis and transcription.

| Variant | Model id | Voices | Additional required fields |
| --- | --- | --- | --- |
| CustomVoice | `qwen3-tts-12hz-1.7b-customvoice` | `talker_config.spk_id` names | `voice` |
| VoiceDesign | `qwen3-tts-12hz-1.7b-voice-design` | `voice-design` | `instructions` |
| Base | `qwen3-tts-12hz-1.7b-base` | `voice-clone` | `reference_audio`, plus `reference_text` unless `voice_clone_mode` is `speaker_embedding_only` |

`GET /v1/audio/voices` lists the advertised voices. CustomVoice exposes the
speaker names in `talker_config.spk_id`, while VoiceDesign and Base expose a
single placeholder name because their timbre comes from `instructions` or
`reference_audio` per request.

A Base checkpoint can additionally advertise operator-registered named voices,
so clients select a speaker by name instead of uploading a reference clip on
every request:

```sh
./result/bin/gufo serve tts \
  --model /persist/models/audio/Qwen3-TTS-12Hz-1.7B-Base \
  --voice narrator_eng=/persist/models/audio/clear-english-voice.wav \
  --voice narrator_ita=/persist/models/audio/clear-italian-voice.wav
```

`--voice NAME=PATH` is repeatable. The reference transcript comes from
`--voice-text NAME=<transcript or path>`, which is also repeatable and may
appear before or after its matching `--voice`:

```sh
./result/bin/gufo serve tts \
  --model /persist/models/audio/Qwen3-TTS-12Hz-1.7B-Base \
  --voice narrator_ita=/persist/models/audio/clear-italian-voice.wav \
  --voice-text "narrator_ita=Questo racconto e' cresciuto..." \
  --voice narrator_eng=/persist/models/audio/clear-english-voice.wav \
  --voice-text narrator_eng=/persist/models/audio/clear-english-voice.txt
```

A `--voice-text` value that names an existing file is read for its contents;
anything else is used as the transcript itself. Resolution order is
`--voice-text`, then a `.txt` sidecar beside the WAV
(`clear-english-voice.txt` for the example above), and with neither the preset
falls back to `speaker_embedding_only` cloning, which needs no transcript.

`--voice-lang NAME=LANGUAGE` binds a language to a voice. It supplies the
default when a request omits `language`, so a caller naming an Italian voice
does not have to repeat it -- without this the request would fall back to the
global `english` default and synthesize the voice in the wrong language. An
explicit request `language` still wins, so a voice can be driven in another
language deliberately.

A transcript that does not match its reference audio is worse than no
transcript: synthesis runs to the `max_new_tokens` cap, so a short input can
return several minutes of unusable audio. Preset names join
`voice-clone` in `/v1/audio/voices`, and a request naming a preset must not
also send `reference_audio`, `reference_text`, or `voice_clone_mode` --- the
preset already supplies them. Presets require a checkpoint that supports
reference presets (such as Qwen3-TTS Base); CustomVoice selects a trained
embedding and VoiceDesign is driven by `instructions`, so neither has anything
to apply them to. A request that omits `voice` gets the model's declared
default voice (such as `voice-clone`), falling back to the first name in
`/v1/audio/voices`; a preset is never chosen implicitly.

`POST /v1/audio/speech` accepts `model`, `input`, `voice`, `response_format`,
`speed`, `language`, `instructions`, `seed`, `max_new_tokens`, `greedy`,
`stream_format`, talker `temperature`/`top_k`/`top_p`/`repetition_penalty`, and
predictor `subtalker_dosample`/`subtalker_temperature`/`subtalker_top_k`/`subtalker_top_p`.
Checkpoints that support reference conditioning additionally accept
`reference_audio`, `reference_text`, `voice_clone_mode`.
Unknown fields are ignored. `response_format` supports buffered `wav` and
streaming `pcm`; `stream_format: "sse"` emits OpenAI audio events with base64
PCM. `speed` supports `1.0` only; input is capped at 16384 UTF-8 bytes and
`max_new_tokens` at 8192 (default 3000). Output is mono 16-bit PCM at the
model's output sample rate (24 kHz for Qwen3-TTS).

`POST /v1/audio/music` (`gufo serve music`, ACE-Step 1.5 XL) accepts `prompt`
(or `tags`), `lyrics`, `duration` (1--600 s, default 30; source tasks take
the source length instead), `seed`, `steps` (1--200), `shift` (1--5),
`guidance_scale` (1--15), `cfg_interval_start` / `cfg_interval_end` (0--1),
`bpm`, `keyscale`, `timesignature`, `language`, `audio_codes`, `lm_cfg`,
`lm_temperature`, `lm_top_p`, `task`, `source_audio`, `reference_audio`,
`repaint_start`, `repaint_end`, `repaint_mode`, `repaint_strength`,
`cover_strength`, `cover_noise_strength`, `track`, `track_classes`,
`vae_sample`, `response_format`, `bitrate_kbps` and `async`. `steps`,
`shift` and `guidance_scale` default to the loaded variant's official
values: turbo 8 steps, shift 3, no guidance (guidance is always 1 on turbo,
as the official pipeline forces it); sft 50 steps and base 32 steps, both
shift 3 and APG guidance 7 (two DiT passes per step, so about twice the
time per step). The status JSON reports `variant`, `task`, `steps`,
`shift`, `guidance_scale` and `duration` as used. `response_format` is `wav` (default, 48 kHz
stereo 16-bit PCM, `audio/wav`), `mp3` (`audio/mpeg`) or `aac` (alias `m4a`,
`audio/mp4`); anything else is a 400 `unsupported_format`. `mp3` and `aac`
are transcoded from the finished WAV by FFmpeg (`ffmpeg` on the PATH or
`GUFO_FFMPEG`, started directly without a shell) at the constant bitrate
`bitrate_kbps` (integer 96--320, default 192; `-codec:a libmp3lame` or
`aac`, `-b:a <kbps>k`). The answer, and `GET /v1/audio/music/<id>/content`
in async mode, carries the format's `Content-Type` and
`Content-Disposition: attachment; filename="output.<wav|mp3|m4a>"`, so a
router job stores it as `output.mp3` (router jobs name the file from the
content type; `audio/mp4` is not mapped there yet and lands as
`output.bin`). When FFmpeg is missing or fails the job ends `failed` with
`"<format> encoding needs ffmpeg: ..."` (HTTP 500 on the content read), one
error line is logged and the temporary WAV is left in the temp directory
(its path is in the message). The job status JSON reports `response_format`
and, for `mp3` / `aac`, `bitrate_kbps`.

Model variants and files: `gufo serve music --model DIR` reads
`DIR/dit.safetensors` (the ComfyUI `acestep_v1.5_xl_turbo_bf16` file or a
converted one), `text_encoder.safetensors`, `vae.safetensors` (the Diffusers
Oobleck file, encoder and decoder), `tokenizer.json`,
`silence_latent.safetensors` and, when present, `lm.safetensors`. `--dit
sft` / `--dit base` select `DIR/dit-sft.safetensors` /
`DIR/dit-base.safetensors`, `--dit FILE` any file; `--lm none|0.6b|1.7b|4b|FILE`
picks the 5 Hz planner (`DIR/lm-<size>.safetensors`). The sampler variant
comes from the DiT file's `gufo.variant` / `model_version` metadata, else
from its file name, else `--variant turbo|sft|base`; the default model id
is `ace-step-1.5-xl-<variant>`. `tools/ace_step/convert_hf_to_gufo.py dit
HF_DIR DIR/dit-sft.safetensors` merges the Hugging Face shards (F32) into
one BF16 file with that metadata; `... lm HF_LM_DIR DIR/lm-1.7b.safetensors`
does the same for the planners (0.6B, 1.7B, 4B; the geometry is read from
the weights). One worker serves one variant; expose three model ids with
three presets (`--dit turbo|sft|base` and `--served-model-name`).

Tasks (`task`, the official ACE-Step task types): `text2music` (default),
`repaint`, `cover` and `cover-nofsq` run on every variant; `extract`,
`lego` and `complete` need the base model (400 `unsupported_task`
elsewhere, sft included, as the official UI restricts them). Every task but
text2music needs `source_audio` (a data URL, raw base64 or a public https
URL, up to 256 MB and 600 s; WAV of any PCM / float depth and channel count
is decoded directly, other formats through FFmpeg, then resampled to 48 kHz
stereo like the official `_normalize_audio_to_stereo_48k`; 400
`missing_source_audio` / `invalid_audio`). The source is VAE-encoded
(`vae_sample`, default true, draws `latent_dist.sample()` from the seed
like the official pipeline; false uses the mean) and sets the duration (a
remainder below one 40 ms latent frame is dropped). `cover` quantizes the
source latents through the FSQ audio tokenizer and the detokenizer (the
official `is_covers` round trip), `cover-nofsq` conditions on the raw
latents; `cover_strength` (0--1, default 1) switches to the plain
text2music conditions after that fraction of the steps and
`cover_noise_strength` (0--1, default 0) starts from the renoised source at
the nearest schedule value. `repaint` regenerates `repaint_start` ..
`repaint_end` seconds (`repaint_end` < 0 = the end of the source; a start
below 0 or an end past the source extend the song with silence, i.e.
outpainting) with the explicit 0/1 chunk mask, silence in the span, the
official step injection, latent crossfade and waveform splice:
`repaint_mode` `conservative` / `balanced` (default, with
`repaint_strength` 0--1, default 0.5) / `aggressive`. `lego` keeps the
source inside the span and adds the `track` stem there (`repaint_start` /
`repaint_end` as for repaint, default the whole song); `extract` pulls the
`track` stem (`vocals`, `drums`, `bass`, `guitar`, `keyboard`, `strings`,
`synth`, `brass`, `woodwinds`, `percussion`, `fx`, `backing_vocals`) out of
the source; `complete` adds `track_classes` (array of stem names) to the
source. `reference_audio` (same forms as `source_audio`) gives the timbre:
three 10 s excerpts (chosen from the seed) are VAE-encoded for the timbre
encoder, as the official `process_reference_audio` does. The 5 Hz planner
runs for text2music only (the official pipeline skips it for cover,
repaint and extract; its lego / complete planning is not ported).
Differences from the official sampler: Euler only (no Heun, SDE, DCW
correction, velocity clamp / EMA, retake or flow-edit), APG guidance as
`apg_forward` (momentum -0.75, norm threshold 2.5), and one audio
tokenizer codebook convention (ComfyUI's symmetric FSQ, which gufo's
detokenizer already used).

Live preview of a running music job: `GET /v1/audio/music/<id>/preview`
answers the newest preview as `audio/wav` (mono 24 kHz 16-bit, about 6 s
around the middle of the song, or the whole song when shorter) with the
headers `X-Gufo-Preview-Step` / `X-Gufo-Preview-Steps` (the diffusion step
it was decoded from), `X-Gufo-Preview-Seconds` (where the window starts in
the song) and `X-Gufo-Preview-Window` (its length); 204 while there is none
yet, 404 for an unknown job. The status JSON carries `preview_url` (relative,
`/v1/audio/music/<id>/preview`, `null` before the job runs) and
`preview_step` / `preview_steps` / `preview_seconds` / `preview_window` /
`preview_at` once a preview exists (`preview_step` is `null` before). The
preview is the sampler's clean estimate `x0 = x_t - t * v` of the current
step (the same point the final Euler step reaches) decoded by the VAE for
the window only, so it sounds rough early on and like the result on the last
step. Previews cost nothing unless wanted: the request opts in with
`"preview": true` (also `"always"`, `{"always": true, "every": N}` or the
image form `"gufo": {"preview": {...}}`; `every` 1..50, default 2 = every
second step plus the last), or a client fetches `/preview`, which keeps
previews on for 10 s. A preview never renders while the previous one is
younger than 3 s; each one logs `preview step=... decode_ms=...`. The
router does not relay `/preview` yet (its music job reads know only the
status and `/content` paths), so ask the music worker directly; a router job
of a music request shows the worker's id as `music_id`.

`GET /v1/audio/speech/stream` upgrades to the vLLM-Omni incremental-text
WebSocket protocol. [TTS streaming examples](../models/qwen3-tts.md#sampling)
cover per-session configuration, audio events and optional sentence/clause
segmentation. Output streaming does not change model prompt construction.
HTTP/1.1 streams use chunked transfer encoding; failed generation omits the
terminal chunk so clients can detect truncated audio. SSE also reports an error
event, while WebSocket speech reports an error without `audio.done`.

Qwen3-ASR serving uses its own model process:

```sh
./result/bin/gufo serve asr \
  --model /var/llms/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/<revision>
```

`POST /v1/audio/transcriptions` accepts OpenAI-compatible multipart fields
`file`, `model`, `language`, `prompt`, `response_format`, `temperature`, and
`max_tokens`, and `stream`. The native route is deterministic (`temperature=0`)
and supports `json`, `text`, and `verbose_json`. `stream=true` with JSON emits
OpenAI transcript delta/done events. Long uploads are split within model
capacity; `max_tokens` applies per chunk. Timestamp (`timestamp_granularities[]`)
and logprobs (`include[]`) requests are rejected; unknown fields are ignored.

`GET /v1/realtime?intent=transcription` upgrades to OpenAI's manual-commit
transcription WebSocket interface. [ASR streaming examples](../models/qwen3-asr.md#streaming-a-file)
describe PCM format, append/commit/clear, event IDs and limits. WebSockets use
the same connection limits as HTTP. Messages are
bounded to 4 MiB, queued input to 8 MiB/128 messages, and unconsumed output to
a five-second socket timeout. Native generation observes close/disconnect;
no per-server streaming flag or extra model is required.

Models with live transcription stream text deltas while audio is still being
appended, and `turn_detection: {"type": "server_vad"}` lets the server find
turns (`input_audio_buffer.speech_started`/`speech_stopped`, then the
commit). Without `intent`, models that hear and speak hold an OpenAI Realtime
conversation on the same path: audio and text input, streamed text and 24 kHz
speech replies, function tools, `response.cancel`, interruption by user
speech and `conversation.item.truncate`. The handshake selects the
`realtime` subprotocol a client offers. `GET /voice`, served when the model
holds conversations, is a browser client for it. Event-level behavior is
documented by the model's provider (for example the LFM3-Audio extension).

`POST /v1/systemone` (`gufo serve decision`) answers one JSON decision
request with the loaded model; a `model` field, when present, must name it.
Malformed JSON and requests the model rejects return 400, and requests are
serialized over the one loaded model.

The MiniMax H3 subset follows the asynchronous OpenAI-style video resource
shape and is versioned independently as `gufo.video-api.v1`. Its supported
fields, frozen presets, queue behavior, and deliberate conditioning
omissions are documented in the [H3 upstream contract](models/minimax-h3/QUALITY.md).
Create requests accept both `application/json` and OpenAI-client-compatible
`multipart/form-data`; duplicate form fields, malformed boundaries, unsupported
media types, and reference-image parts fail explicitly.
H3 accepts `seconds` as a JSON number or string (including multipart), at
`512x512` or `1344x768`. Durations from 5 seconds are converted to 24-fps frames
and rounded up to the official VAE grid (`17n + 5`), within upstream's
15-second ceiling: 5 → 124 frames, 10 → 243, 14 → 345. The maximum is
`14.375` seconds (345 frames); `15` would round to 362 frames and is rejected,
as upstream does. An explicit `gufo.frames` must match the aligned duration.
The one-second/22-frame diagnostic extension remains available.
The encoder accepts at most 4,096 tokens after tokenization and normalization.
There is no 4,096-byte prompt limit. HTTP request bodies remain bounded by
`--max-request-bytes`; token-limit violations fail the asynchronous job before
prompt-encoder weights or activations are allocated.
The bounded worker retains prompt text only in volatile queued or active
request memory, persists only its SHA-256 digest, and wipes both source and
active string storage after transfer and completion. Completed MP4s are probed
for H.264/AAC codec, geometry, rates, duration, and A/V synchronization before
atomic publication.

## Embeddings

EmbeddingGemma 2 (`google/embeddinggemma-2`) serves text, image, video and audio embeddings from its own model process.  It is a 768-dimension, BF16, 8192-token-context encoder:

```sh
./result/bin/gufo serve --port 8080 embedding --model /persist/models/embeddinggemma-2
```

`--model` must name an EmbeddingGemma 2 checkpoint directory, and `--served-model-name` overrides the default model ID `embeddinggemma-2`.  The process refuses to start on any other directory and in a build without the HIP backend.  There is no `--sessions` or `--context` option, because requests share one encoder.  Requests that wait for the device share batches of up to 8,192 tokens, except that a request with an image, video or audio runs in a batch of its own.  When the device is free, the oldest waiting request runs one batch for itself and the requests behind it.  The batch takes whole requests in arrival order until the next one does not fit or has an image, video or audio.  A lone request runs at once.  A request that fits in one batch never spans two.  A request larger than one batch is split, and after each of its batches it goes behind the requests that arrived meanwhile, so a short request waits for at most one batch of a long one.  Route this process through llama-swap when one public URL should offer embeddings alongside text, speech, or images.

`POST /v1/embeddings` follows the OpenAI request shape:

```json
{
  "model": "embeddinggemma-2",
  "input": ["task: search result | query: how do I cook rice?",
            "title: none | text: Rinse the rice, then simmer it covered."],
  "encoding_format": "float",
  "dimensions": 256
}
```

The response holds one embedding per input in request order:

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.0123, -0.0456, "..."]},
    {"object": "embedding", "index": 1, "embedding": [0.0098, 0.0311, "..."]}
  ],
  "model": "embeddinggemma-2",
  "usage": {"prompt_tokens": 41, "total_tokens": 41}
}
```

- `input` is required.  It is a string or an array of 1 to 2048 strings or typed content objects.  The server rejects empty strings.  Token-array input, meaning integers or arrays of integers, is not supported and returns 400 `unsupported_input`, so send text.
- `model` is optional.  When present it must equal the served model ID, or the request returns 404 `model_not_found`.
- `encoding_format` is `float` by default or `base64`.  Base64 carries the little-endian float32 bytes of the vector, as OpenAI's SDKs expect.  The server writes floats with enough digits to round-trip a float32 exactly.
- `dimensions` is optional and must be 128, 256, 512, or 768, which is the default.  Any other value returns 400 `invalid_dimensions`.  A smaller value keeps the leading components of the normalized vector, which is Matryoshka truncation, and renormalizes them to unit length.
- `user` is accepted and ignored when it is a string, and a non-string `user` returns 400 `invalid_parameter_type`.
- `prompt_name` is optional; see task prefixes below.
- `video_options` optionally selects video FPS, frame cap, overflow strategy, timestamps and the per-frame vision budget (`max_soft_tokens`: 70, 140, 280, 560 or 1120; 140 is the default); see [video inputs](models/embeddinggemma-2/README.md#video).
- `mm_processor_kwargs` is optional, as in vLLM.  Its only key, `max_soft_tokens`, selects the image budget of the checkpoint's `Gemma4ImageProcessor` for every image of the request: 70, 140, 280 (the default), 560, or 1120 soft tokens.  Absent, null or empty `mm_processor_kwargs` keeps the default. Any other key or value returns 400 `invalid_mm_processor_kwargs`, and a non-object returns 400 `invalid_parameter_type`. Video frames keep their separate budget, `video_options.max_soft_tokens`. The 32-image request cap is a server resource limit across all inputs, independent of the selected budget; see [image limits and memory](models/embeddinggemma-2/README.md#images).
- `truncate` is optional and `false` by default; see the token limit below.  A non-boolean `truncate` returns 400 `invalid_parameter_type`.  As in llama.cpp, TEI and vLLM, the server ignores members it does not know.
- As in llama.cpp, `embd_normalize` and `content` are accepted here too; see the native route below.  A request with `content` and no `input` gets the native response.
- Every vector is mean-pooled and L2-normalized, so the dot product of two vectors is their cosine similarity.
- Each input may hold at most 8192 tokens, counting the `<bos>` and `<eos>` markers the tokenizer adds.  By default the server refuses longer inputs instead of truncating them, with 400 `context_length_exceeded`, and the message names the input index.  With `"truncate": true` it keeps the first 8190 tokens of each longer input between the markers instead; only `--max-request-bytes` then limits a text input's length.  An input with images, video or audio is truncated only when all of its media tokens fit in the kept tokens; otherwise it is still refused.  `usage.prompt_tokens` and `usage.total_tokens` count every token embedded for every input, markers included.
- Errors use the standard error object.  Other 400 codes are `too_many_videos`, `invalid_video_input`, `invalid_video_options`, `too_many_audio`, `invalid_audio_input`, `too_many_images`, `invalid_image_input`, `parse_error`, `invalid_request_body`, `missing_input`, `empty_input`, `too_many_inputs`, `invalid_encoding_format`, `invalid_mm_processor_kwargs`, and `invalid_parameter_type`.  A client disconnect returns 499 `cancelled`, and a request whose client disconnects before its next batch does not use the device again.  A failed encode returns 500 `embedding_failed` to every request with inputs in the failed batch.  A service that is not ready returns 503 `embedding_service_unavailable`.  A GPU failure that the device probe confirms returns 503 `device_lost` and shuts the server down as described under [Lifecycle and limits](#lifecycle-and-limits).  The usual `--max-request-bytes` and request logging apply.

Typed items follow llama.cpp's form: `{"content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}},{"type":"text","text":"\n"}]}`. Each item produces one vector in the same space as text. Text parts retain their order and whitespace. PNG, JPEG and WebP use the existing image reader, including public-HTTPS checks and the shared 20 MiB / 15-second budget. Embeddings accept at most 32 images per request, 32 megapixels per image, and 32,768 pixels on either side. Image soft tokens and their markers count toward context and usage. A request with images runs in its own device batch, in arrival order, and its soft tokens count toward the 8,192-token batch. The first image request loads the vision weights in that batch. See [image embeddings](models/embeddinggemma-2/README.md#images) for exact preprocessing and index migration. Audio uses typed `input_audio` parts with raw base64, audio data URLs or public HTTPS URLs; WAV, FLAC, MP3 and Ogg are supported. Up to 32 clips of 30 seconds each share the image read budget. Actual subsampled audio tokens and their boundary markers count toward context and usage. See [audio embeddings](models/embeddinggemma-2/README.md#audio) for decoding, preprocessing, bounds and examples. Video uses typed `input_video` parts with raw base64, video data URLs or public HTTPS URLs, producing one jointly encoded vector per input. MP4/MOV and WebM/Matroska share the same media read budget. Defaults sample 1 FPS, uniformly capped at 32 frames with 140 soft tokens per frame; a request holds at most 64 selected frames and 8,960 video soft tokens. See [video embeddings](models/embeddinggemma-2/README.md#video) for `video_options`, timestamp framing, audio-track treatment and decoding limits.

**llama.cpp's `POST /embeddings`.** Clients written for llama-server's non-OpenAI route, such as Frigate 0.18's semantic search, can use `POST /embeddings` (or its older alias `POST /embedding`) on the same process.  It shares the model, tokenization, image reader, pooling, limits and errors of `/v1/embeddings`, and differs in two ways:

- The input is `input`, or llama.cpp's older `content` when `input` is absent.  Besides strings and typed content objects, an item may be `{"prompt_string": "...", "multimodal_data": ["<base64>", ...]}`: each `<__media__>` marker in the string stands for the next base64 file, in place.  As in llama.cpp, a file that starts like WAV, MP3 or FLAC is audio, as is Ogg here; anything else must be a PNG, JPEG or WebP image.  The number of markers must equal the number of files, or the request returns 400 `invalid_multimodal_data`.  Gufo always uses llama.cpp's default marker; it serves no `/props`, and clients such as Frigate fall back to that marker when `/props` is missing.
- The response is llama.cpp's bare array, one item per input in request order, with the pooled vector nested in a one-element list and no `model` or `usage`:

  ```json
  [{"index": 0, "embedding": [[0.0123, -0.0456, "..."]]}]
  ```

On both routes, `embd_normalize` is accepted only as `2`, the Euclidean default that every EmbeddingGemma 2 vector already has; other values return 400 `unsupported_embd_normalize`.  As in llama.cpp, the native form validates `encoding_format` but always returns floats.  `dimensions`, `mm_processor_kwargs`, `prompt_name`, `truncate` and `user` behave as on `/v1/embeddings`.  Token arrays and per-token (`--pooling none`) output stay unsupported, and unknown members are ignored.

**Task prefixes.** EmbeddingGemma 2 was trained with short task prefixes, and by default the server does not add them, so clients must.  Queries use `task: search result | query: {text}`, and documents use `title: none | text: {text}`, or `title: {title} | text: {text}` when a title is known.  Omitting the prefixes still works but lowers retrieval quality.  Embed a query and its documents with matching conventions.  A client that cannot change its input can send `"prompt_name"` with a name from the checkpoint's `config_sentence_transformers.json` `prompts` (for example `query` or `document`), and `--default-prompt-name NAME` applies one to requests that name none.  As the model card advises, the default skips inputs without text, so images and audio alone embed unprefixed; a requested `prompt_name` prefixes every input, as in sentence-transformers.  An empty `prompt_name` applies none, even with a default.  An unknown name returns 400 `invalid_prompt_name`, and an unknown `--default-prompt-name` fails startup.  `usage` counts the prompt's tokens.

## Internal Request Model

Text endpoints use `ChatRequest` and `SamplingConfig` at the
[backend boundary](../../src/cli/serve/text_generation_backend.hpp).
Chat messages retain roles, reasoning, tool calls and image content until the
model's tokenizer and template turn them into model input. The scheduler owns
request limits, cancellation, cache accounting and completion state.

## OpenAI input token counts

`POST /v1/chat/completions/input_tokens` and
`POST /v1/responses/input_tokens` accept the same request body and model
selection as their respective generation endpoints and return:

```json
{"object":"response.input_tokens","input_tokens":123}
```

Responses follows OpenAI's input-token-count response shape; Chat Completions
uses the equivalent llama-server extension. Counts include the model's chat
template, system/instructions, tools and tool history, reasoning settings,
structured-output instructions and supported image tokens. They describe the
complete prompt, independent of any cached prefix, and agree with Chat
`usage.prompt_tokens` or Responses `usage.input_tokens` for the same request.

Counting does not schedule inference, prefill or generation, and does not
update the continuation cache or generation metrics. It can count prompts
larger than the execution context so clients can decide how much to trim.
Image inputs still undergo the usual image loading and validation and may
populate the CPU image-preprocessing cache. Generation options are validated as
usual, but `stream: true` is accepted and still returns
one JSON count response. Output-token limits are not added to `input_tokens`.
Malformed requests return 400, a model that is not served returns 404, and a
backend without prompt-count support returns 501. As with generation, Responses
clients must supply the complete conversation; server-side history is not
resolved by these endpoints.

## Responses API Subset

`POST /v1/responses` accepts `model`, `input` as text or message arrays,
`instructions`, `max_output_tokens`, `stream`, `reasoning.effort`,
`text.format`, `tools`, `tool_choice`, `parallel_tool_calls`, `include`,
`top_logprobs`, and the shared
sampling controls. Message content supports
`input_text` and `input_image` with an `image_url` (HTTPS or a data URL).
Image uploads accept PNG, JPEG and WebP. Base64 data URLs also accept
`image/jpg`, mixed-case media types/`base64`, and parameters such as `name=`.
Clients supply the complete conversation, including prior Gufo `output` items
when retaining reasoning. Replay `function_call` items with their `call_id`,
then supply `function_call_output` items using the same ID. Function tools use
the flat `{type:"function",name,parameters,strict}` shape. The Responses API
also defines hosted tool types (`web_search`, `file_search`, `code_interpreter`,
`mcp`, ...) that only OpenAI can execute; they are accepted and skipped so
the function tools still reach the model. A `namespace` entry is not hosted:
it groups client-executed function tools for organization only, and its
functions are flattened into the function list. Their `function_call` items
keep the plain `name` and add the owning `namespace`, so clients can route
them. Function names that collide across namespaces or with top-level
functions are rejected. Mid-conversation `system` and `developer` message
items are accepted as in Chat Completions. Standard Responses request fields
with
no native effect are accepted and ignored so conforming clients interoperate
(for example Codex): `reasoning.summary`, `text.verbosity`,
`client_metadata` and `prompt_cache_key`.
`include: ["message.output_text.logprobs"]` enables generated-token scores;
other string entries in `include` are accepted without native effect.
`store` and `background` must be false when present; server-side conversations
and `previous_response_id` remain unsupported.

Function-call output accepts either a text string or an array of `input_text`
and `input_image` parts. Images stay inside the corresponding tool result;
image-only output and interleaved text/images are supported. Historical
`custom_tool_call` items and their `custom_tool_call_output` use the same
association and output parts; the custom call's free-form `input` is retained
as literal tool-history data. This does not enable custom-tool generation.
Chat Completions also accepts `image_url` parts in `role:"tool"` content.
Existing text-string outputs keep their prompt representation. Images remain
unsupported in assistant/system/developer messages. Image resolution is
model-owned: `detail` accepts `auto`, `low`, `high` and `original`. For Qwen
models, `low` scales the image down to fit within 512 × 512 pixels and keeps its
aspect ratio, so it uses at most 256 merged image tokens. The model's
65,536-pixel minimum still applies, so a very small or very narrow image can be
enlarged, but `low` never uses more pixels than `auto`. The other values behave
as `auto` and use the model's full resolution range. A `low` request and an
`auto` request for the same image share the cached prefix when both resize it to
the same size. Otherwise they share only the prompt text before the image. File
IDs and `input_file` parts remain unsupported.

Responses report `incomplete` with reason `max_output_tokens` when generation
hits its limit. Otherwise they report `completed`. `stream: true` sends typed
SSE events with consecutive `sequence_number` values: lifecycle, output items,
text/reasoning deltas and terminal status. Local reasoning is exposed as
`reasoning` items with `summary_text`; visible answers use `output_text`.
Disconnects cancel generation through the same scheduler as Chat Completions.

The official [OpenAI Python SDK](https://github.com/openai/openai-python) is
included in `nix develop`. Use the model name from `/v1/models`
(for example, `qwen` with `--served-model-name qwen`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
for event in client.responses.create(
    model="qwen", input="Hello", store=False, stream=True
):
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
    elif event.type == "response.failed":
        raise RuntimeError(event.response.error.message)
```

The other compatibility routes are deliberately limited:

| Route | Supported request | Output limit |
| --- | --- | --- |
| `/v1/completions` | One prompt string, buffered or SSE completion | `max_tokens` |
| `/v1/messages` | Text, image, thinking and tool blocks, optional text system instructions and custom tools; buffered or SSE | `max_tokens` |
| `/completion` | One prompt string, non-streaming completion | `n_predict` |

All four routes validate the loaded model, positive integer limits and shared
sampling controls. `/completion` rejects streaming; all reject multiple
candidates. Streamed Messages send Anthropic SSE events: `message_start`,
`content_block_start`/`content_block_delta`/`content_block_stop` for
`thinking` (`thinking_delta`), `text` (`text_delta`) and `tool_use`
(`input_json_delta`) blocks, then `message_delta` with `stop_reason` and the
complete usage, and `message_stop`. `message_start` reports zero usage because
prompt accounting is final only at the end. A failure after the headers sends
an `error` event. Responses and Messages honor the server's thinking defaults.
Messages accepts `thinking.type` (`enabled`, `adaptive` or `disabled`);
`adaptive` keeps the server's thinking default. With `enabled`,
`budget_tokens` caps the reasoning (see Reasoning controls), and the answer
follows. The effort stays the server's unless `output_config.effort`
sets it. Reasoning is returned as a `thinking` block
before the `text` block, with an empty `signature`, for every accepted
`thinking.display` (`summarized`, `omitted` or `updates`). Replay assistant
`thinking` blocks unchanged so later turns reuse the cached prompt.
`output_config.effort` (`low`, `medium`, `high`, `xhigh` or `max`) sets the
reasoning effort used while thinking is on; it never enables thinking. Other
`output_config` members are rejected. Messages maps custom `tools`
(`input_schema`) and `tool_choice` (`auto`, `any`, `tool`, `none`, with
`disable_parallel_tool_use`) onto the Chat tool path, so framing, schema
constraints and cache reuse match Chat Completions. Calls are returned as
`tool_use` blocks with `stop_reason: "tool_use"`; replay them unchanged with
the following `tool_result` blocks. Server tools such as `web_search` are
rejected. Messages accepts `image` blocks in user turns and in `tool_result`
content, with a `base64` source (PNG, JPEG or WebP) or an HTTPS `url` source,
under the same limits and model requirements as Chat Completions images;
images in assistant turns, `file` sources and `document` blocks are rejected.
Completions routes accept `stop`;
Messages accepts `stop_sequences`. Responses has no stop-sequence field.
`POST /v1/messages/count_tokens` accepts the same text, images, system, thinking and
tool history as Messages and returns `{"input_tokens": N}` after rendering the
model's chat template. `max_tokens` is optional; `stream: true` is rejected.
The count includes cached prompt tokens and does not generate, prefill or
change the conversation cache. Prompts may exceed the configured execution
context so clients can size them before trimming; generation still enforces
the context limit. Images use the same resizing and visual token expansion as
generation, without running the vision encoder; the matching projector must
be loaded. Models without image support reject image counts with 400.
Backends without chat counting support return 501. `/infill` still returns 501 because
suffix-conditioned infill is not implemented.

Raw Completions accepts `stream: true` and
`stream_options: {"include_usage": true}`. The final usage event reports prompt,
cached and completion token counts before the `[DONE]` sentinel. Local benchmark
clients can set `ignore_eos: true` to generate exactly `max_tokens`; context
capacity remains the hard limit. Chat Completions, Responses, Messages and
`/completion` reject `ignore_eos`: Raw Completions owns the fixed-length
contract. `GET /v1/models` reports that configured limit as `context_length` on
the loaded text model.

`ignore_eos` measures decode past the model's stop tokens, which speculative
backends do not accelerate: verification ends its accepted run at every EOS it
emits, so each crossing clips the draft window. Expect decode near the
autoregressive rate in that region, not the rate the same model reaches on an
ordinary continuation.

## Chat Completions Adapter

`POST /v1/chat/completions` accepts the common compatibility subset:

- `model`
- `messages`: `system` and `developer` messages may appear at any position.
  Agent clients such as Codex send them mid-conversation after context
  compaction or when session settings change. DeepSeek renders them in place.
  Qwen's template accepts only one leading system turn, so Qwen models hoist
  them into it in their original relative order. That changes the prompt head:
  the request that introduces such a message is prefilled again, and later
  requests that keep it reuse the whole prompt
- `max_tokens` or `max_completion_tokens`
- `temperature`
- `top_p`
- `seed`
- `stop`: null, one string, or an array of up to four strings
- `stream`
- `stream_options.include_usage`
- `return_progress` (see [Prompt progress](#prompt-progress))
- `tools` and `tool_choice` when supported. Function tools accept the nested
  Chat Completions shape and the flat Responses-style `{type,name,parameters}`
  shape. Missing or null `parameters` become `{}`. `parametersJsonSchema` is
  accepted as an alias for `parameters`. Flat definitions retain all function
  fields, including `strict`. A function name holds 1-64 printable ASCII
  characters and may use any of them except a space, `<`, `>`, `"` and `\`,
  which frame a rendered call; dotted and namespaced names such as
  `github.create_issue` are accepted. OpenAI itself documents a narrower set
  for this field, so a name outside `[A-Za-z0-9_-]` is portable to gufo but not
  to every OpenAI-compatible service. Historical names in assistant
  `tool_calls` and Responses `function_call` items are preserved verbatim,
  including Unicode and names absent from the current tools. This does not
  authorize new calls to them. History still requires a non-empty string name and
  JSON-object arguments; embedded NUL names are unsupported. Malformed history
  and invalid declarations return 400 before generation.
- shared top-k, min-p, repeat, frequency and presence sampling controls

Streaming objects use `chat.completion.chunk` and end with the compatibility
sentinel expected by common clients.

### Prompt progress

Streaming `/v1/chat/completions`, `/v1/completions` and `/v1/responses` requests accept
llama-server's `return_progress: true`. Before the first token, the stream
carries chunks with an empty delta (Chat) or empty text (Completions) and a
top-level `prompt_progress` object:

```json
{"prompt_progress": {"total": 4096, "cache": 1024, "processed": 2048, "time_ms": 850}}
```

`processed` includes cached tokens; a full cache hit reports `processed == total`.
`time_ms` measures prompt-processing wall time. Updates follow prefill chunks;
a slow client receives only the newest pending update. Responses uses
`response.in_progress` events. Omitted, false or null disables progress;
other non-boolean values return 400. Buffered requests ignore the flag.
Disabled progress adds no progress queue updates or consumer wakeups.

The adapter must not implement a second inference path. It converts messages
into the same prompt and sampling structures used by `/v1/responses`.

Frequency and presence penalties count every committed token generated in the
current request, excluding its prompt. Repetition penalties use `repeat_last_n`
and may include prompt tokens. Setting that window to zero disables only the
repetition penalty. Speculative rejection discards tentative counts; seeded
sampling replay retains independent request histories.

API ranges: temperature `[0, 2]`, top-p/min-p `[0, 1]`, frequency/presence
penalties `[-2, 2]`. Top-p zero keeps the highest-probability token (or
`min_keep` tokens). Setting temperature to zero retains configured penalties;
omitted controls keep their model/CLI defaults.

Tool calls are emitted only for declared functions when `tool_choice` allows
calling tools. With `auto`, ordinary text and reasoning remain allowed; once a
call starts, decoding constrains its name and argument format. As in llama.cpp,
a DeepSeek call block ends the output: parallel calls share one block, and no
text follows it. Other DeepSeek output, including client call markup written in
place of a native call, is returned as content. Non-strict tools
keep optional arguments optional. Open nested objects retain native syntax and
declared requirements/types, including nested fields; unsupported schema
keywords remain guidance. Unsupported property-admitting rules, including
conditional branches, leave those objects open without discarding declared
requirements. Union and untyped arguments keep the native syntax, as in
llama.cpp: when the union admits strings the value is raw text, and its typed
alternatives (such as `null` or an object) are tried before the string, so Qwen
cannot return the literal string `"null"` for a string/null union.
A model with a native call syntax (Qwen, DeepSeek) never switches to a JSON
envelope, whatever the schema, strict flag or tool choice: as in llama.cpp
`common/parsers/qwen3-coder.cpp` and `deepseek.cpp`, every call uses the chat
template's syntax and no instruction is added to the prompt. Native tags enforce
what they can carry. A string parameter whose `pattern` cannot be enforced is
raw text; other values follow the supported parts of their schema, or any JSON
value of their types when nothing can be enforced. Qwen tags carry no type, so
Qwen generates declared parameters only; DeepSeek's `string` flag also carries
wildcard fields. A value that must contain the native closing tag cannot be
written natively. Only runners without a native syntax use the JSON envelope.
Historical calls render typed argument values with the chat template's
`tojson` spelling (`", "` and `": "` separators, raw UTF-8), as llama.cpp's
Jinja runtime does. Cache reuse requires identical tokens; normalizing an
assistant's formatting can require replaying that suffix.
Constrained JSON keys follow schema order, with additional
keys last. Impossible strict schemas are rejected before generation.
`tool_choice: "required"` and named choices constrain decoding to a declared
call, with the same argument syntax as `auto`. Where
the backend cannot constrain sampling, an unmet `required` choice still returns
`tool_choice_unsatisfied` (HTTP 502, or an SSE error after streaming starts).
Stops and token limits terminate normally. Responses streams may retain a partial
call with `status: "incomplete"`; clients must execute only completed calls.
With no tools or `tool_choice:"none"`, tool markers are ordinary text and do
not interrupt reasoning or delay streaming.

Stop sequences match accepted output bytes, including reasoning and tool
markup, before streaming or response parsing. Partial prefixes are buffered;
matched sequences and subsequent text are excluded. OpenAI reports
`finish_reason: "stop"`; Messages reports `stop_reason: "stop_sequence"` and
the matched `stop_sequence`. EOS and length limits flush unmatched prefixes.
Interrupted tool calls are omitted; complete preceding calls are retained.
Usage includes the token completing the match. Each request has independent
matching state, including speculative batches; caches retain only correctly
labelled executed model state. Stops must be nonempty, at most 4 KiB each and
16 KiB combined. Messages allows up to 64 sequences.

Nullable Chat Completions defaults retain server settings, including sampling
and token limits. `logprobs: false`, empty `logit_bias`,
`response_format: {"type":"text"}` and `modalities: ["text"]` are accepted.
Token biases and audio output remain unsupported and return explicit errors.

### Generated-token log probabilities

Chat Completions accepts `logprobs: true` and an optional `top_logprobs`
integer from 0 to 20 (default 0). Each choice returns `logprobs.content`,
with `token`, `logprob`, UTF-8 `bytes`, and `top_logprobs` for each generated
token. Streaming choices carry the same metadata beside `delta`.

Raw Completions accepts an integer `logprobs` from 0 to 20. Its choices
return parallel `tokens`, `token_logprobs`, `top_logprobs`, and `text_offset`
arrays. Offsets count Unicode characters and continue across streaming chunks.
Zero requests only the selected token. Omitted/null settings disable collection;
Chat also accepts `false`. Positive Chat `top_logprobs` requires `logprobs: true`.

Responses accepts `include: ["message.output_text.logprobs"]` with optional
`top_logprobs` from 0 to 20 (omitted/null defaults to 0). It returns scores on
each `output_text` content part and on streaming `response.output_text.delta`
and `response.output_text.done` events. Content parts include UTF-8 `bytes`;
text events carry `token`, `logprob`, and `top_logprobs`. Without the include
entry, collection stays disabled even when `top_logprobs` is set.

Responses scores only tokens contributing to visible output text. Reasoning,
tool arguments, hidden framing, and terminal stop tokens receive no text
scores. A token spanning visible text and hidden framing is reported once,
with its original token spelling, on the first text span it contributes to.
Tokens sharing a split UTF-8 character are retained together when that
character is emitted. Buffered results and completed streaming content parts
carry the same scores.

```json
{"model":"gufo","input":"Hello","include":["message.output_text.logprobs"],"top_logprobs":3}
```

The formats and candidate ordering follow [vLLM v0.32.0](https://github.com/vllm-project/vllm/tree/v0.32.0).
Values use the target
model's raw full-vocabulary log-softmax, before temperature, penalties,
constraints, or sampling filters. Speculative tokens use their target verifier
rows, including residual corrections. Values below -9999 are serialized as
-9999. Chat and Completions metadata includes sampled stop/control tokens even
when their text is hidden, so concatenating metadata tokens need not reproduce
visible content.
Existing usage accounting is unchanged.

Collection adds vocabulary transfers and CPU reduction only when requested.
Metadata counts toward streaming buffer limits but not `--max-output-bytes`,
which bounds generated text; `max_tokens` bounds its size. Prompt logprobs,
full-vocabulary output and token-ID extensions are not supported. For example:

```json
{"model":"gufo","messages":[{"role":"user","content":"Hello"}],"logprobs":true,"top_logprobs":3}
```

A buffered Chat choice contains metadata like this (with `top_logprobs: 0`):

```json
{"index":0,"message":{"role":"assistant","content":"Hi"},"logprobs":{"content":[{"token":"Hi","logprob":-0.5,"bytes":[72,105],"top_logprobs":[]}]},"finish_reason":"length"}
```

For streaming Completions:

```json
{"model":"gufo","prompt":"Hello","logprobs":3,"stream":true}
```

Each SSE payload's choice carries the arrays alongside its text, for example
with `logprobs: 0`:

```json
{"index":0,"text":"Hi","logprobs":{"tokens":["Hi"],"token_logprobs":[-0.5],"top_logprobs":[{"Hi":-0.5}],"text_offset":[0]},"finish_reason":null}
```

llama.cpp sampler fields that gufo does not implement (`mirostat`,
`mirostat_tau`, `mirostat_eta`, `typical_p`, `tfs_z`, `dynatemp_range`,
`dynatemp_exponent`, `top_n_sigma`, `xtc_probability`, `xtc_threshold`,
`dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_penalty_last_n` and
`dry_sequence_breakers`) are ignored when absent, `null` or set to a value that
leaves llama.cpp's sampler chain unchanged: `mirostat <= 0`,
`typical_p >= 1`, `dynatemp_range <= 0`, `top_n_sigma <= 0`, XTC with a
clamped probability of 0 or a clamped threshold above 0.5, DRY with
`dry_multiplier` or `dry_penalty_last_n` 0. `tfs_z` (llama.cpp no longer
has TFS), the Mirostat rates and `dynatemp_exponent` take any number. A value
that would activate one of these samplers returns `unsupported_sampling`; a
non-numeric value, a `dry_allowed_length` or `dry_penalty_last_n` outside
`[0, 2147483647]` or an empty `dry_sequence_breakers` returns
`invalid_sampling`.

### Structured output

Chat Completions accepts `response_format: {"type":"json_object"}` or
`{"type":"json_schema","json_schema":{"name":"Reply","strict":true,"schema":…}}`.
Responses uses `text: {"format":{"type":"json_schema","name":"Reply","strict":true,"schema":…}}`.
The official SDK supports typed parsing and streaming through
`client.chat.completions.parse/stream(response_format=Model)` and
`client.responses.parse/stream(text_format=Model)`.

Constraints apply before target sampling in AR, DFlash2, MTP and DSpark, including
streaming, images and concurrent requests. Reasoning stays separate from JSON
and counts toward the output budget. Changing the schema changes the cache prefix.

Every schema and tool grammar uses one token vocabulary per loaded model.
Gufo prepares it automatically while the model loads, before readiness, and
every constrained request reuses it. This moves a one-time CPU cost from the
first constrained request to startup (Flash-Next: 248,320 tokens, about 30 ms).
The vocabulary stays resident even if no request is constrained. The startup
line `event=structured_output_prepared` reports `vocabulary_tokens`, `prepare_ms`
and `rss_delta_kib`. Runners without structured output skip preparation and
retain ordinary text serving; vocabulary build errors still fail startup.
Grammars and their masks are unchanged.

`</think>` ends reasoning before constrained JSON. Native tools also accept an
unquoted function header as the boundary when the model omits `</think>`.
Bare marker mentions and quoted examples remain reasoning; markers inside
arguments remain data. Buffered and streamed Chat Completions and Responses
use the same boundaries.

Parse the returned content: leading whitespace is valid JSON, and stops or token
limits can leave it incomplete. `finish_reason: "stop"` includes matched stop
sequences and does not guarantee complete JSON. Token limits return `"length"`
in Chat or `status: "incomplete"` with reason `max_output_tokens` in Responses.

Supported: objects, arrays, type unions (including nullable types), `enum`/`const`,
`anyOf`, recursive local
`$ref`/`$defs`, numeric bounds/`multipleOf`, string patterns/lengths/formats and
array length bounds. Formats include date/time, duration, email, hostname,
IP addresses and UUID. Annotations that do not constrain values (`$schema`,
`$comment`, `title`, `description`, `default`, `examples`, `deprecated`,
`readOnly` and `writeOnly`) are accepted and ignored, including in strict schemas.

The root must resolve to an object. As in llama.cpp, an object that declares
`properties` admits no other keys unless `additionalProperties` is `true` or a
schema for their values; extra keys then follow the declared properties. An
object without `properties` admits any keys unless `additionalProperties` is
`false`. Strict schemas require every property (use null for optional values).
Tool parameter schemas keep their separate rules below.
Unsupported keywords, external references and unusable cycles are rejected.
Schemas are limited to 2 MiB, 5,000 properties, 1,000 enum values and 120,000
characters in names and enum/const strings. Patterns use ECMA-262 Unicode
semantics; lookbehind, backreferences, inline flags and unbounded repetition
of assertions are unsupported.

Function `strict:true` constrains tool arguments independently of the response
schema. Chat Completions and Responses preserve the supplied parameter schema
when `strict` is omitted, `null` or `false`, matching llama.cpp's schema
handling. Optional fields stay optional; Gufo does not add them to `required`
or force the model to invent default values. To require every field, list them
in `required` and use `strict:true` with a closed object schema.
With `tool_choice: "auto"`, the model may call a tool or give a final answer;
the response schema constrains the latter. Use `"none"` for JSON answers only,
`"required"` to force a call, or select a named function.
`"required"` and a named function constrain decoding regardless of tool
strictness; `parallel_tool_calls:false` allows at most one call and constrains
it likewise. Interrupted calls are omitted.

References: [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
[function calling](https://developers.openai.com/api/docs/guides/function-calling),
[structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[JSON Schema patterns](https://json-schema.org/draft/2020-12/json-schema-validation#name-pattern)
and [llama.cpp grammar sampling](https://github.com/ggml-org/llama.cpp/blob/68d9053afd4f4d0752ced6187585f862355a40be/common/sampling.cpp).
Verify with `tests/functional/openai_sdk.py --suite tools`,
`--suite structured` or `--suite structured-limits`
(add `--vision` for an image-capable server).

## Model Discovery

`GET /v1/models` lists the configured text model and ready audio, video,
image, and embedding services. Entries provide `id`, `object`, `created` and `owned_by`; the text
entry also provides `context_length` and `architecture.input_modalities`: `["text"]`
for text-only serving or `["text", "image"]` when a vision encoder is loaded.
Image support reflects the loaded projector, not the model name; launching a
vision-capable model without its projector advertises text only. Audio/video
entries describe their capability. Send the returned model ID in requests.
Repeat `--served-model-name` to serve one model under several IDs; the
first is primary. Text routes report the ID the request named; the other
modalities report the primary ID, with video aliases acting as the root
`minimax-h3`. Every `gufo serve` modality accepts the repeated option, so
llama-swap `aliases:` entries can reach the process unchanged.
Requests naming another model return 404, except on audio and video routes,
which answer `400 invalid_model`.

## Errors

Return an OpenAI-style error object:

```json
{
  "error": {
    "message": "Unsupported field: logit_bias",
    "type": "invalid_request_error",
    "param": "logit_bias",
    "code": "unsupported_parameter"
  }
}
```

Use appropriate HTTP status codes:

- `400` invalid request or unsupported field
- `404` unknown model or endpoint object
- `409` incompatible session state
- `413` request or prompt too large
- `429` admission queue full or rate limit exceeded
- `499` internally recorded client cancellation
- `500` internal failure
- `503` model or backend unavailable, or GPU context lost (`device_lost`)

Generation stops at the request budget or context capacity and reports a
length finish reason when either limit is reached.

## llama-swap

[llama-swap](https://github.com/mostlygeek/llama-swap) can launch Gufo on
demand. Use the model readiness endpoint for its startup check:

```yaml
models:
  gufo-model:
    cmd: >
      /path/to/gufo serve --host 127.0.0.1 --port ${PORT} llm
      --model /path/to/model.gguf --served-model-name gufo-model
    checkEndpoint: /ready
```

Replace the executable and model paths. llama-swap assigns `${PORT}` and
waits for `/ready` to return HTTP 200 before forwarding the triggering
request. Gufo returns 503 while the model loads. The served model name
matches the llama-swap model ID, so clients can request `gufo-model`
through the proxy. Gufo requires no credentials on any route, so llama-swap's
default `/health` probe works too. For long cold starts, increase llama-swap's
`healthCheckTimeout` to cover model and disk-cache loading.

Gufo answers the names passed to `--served-model-name` plus its built-in
identifiers: TTS also accepts `qwen3-tts`, ASR also accepts `qwen3-asr` and
`qwen3-asr-1.7b`, and video always accepts the canonical `minimax-h3` and
its preset names; requests naming another model return 404 (`400
invalid_model` for audio and video). Repeat the option for each llama-swap
`aliases:` entry — every `gufo serve` modality accepts the repeated option —
or set `useModelName` so llama-swap rewrites the request's `model` field
instead:

```yaml
models:
  gufo-model:
    aliases: [gufo-fast]
    cmd: >
      /path/to/gufo serve --host 127.0.0.1 --port ${PORT} llm
      --model /path/to/model.gguf --served-model-name gufo-model
      --served-model-name gufo-fast
    checkEndpoint: /ready
```

To load the model without generating a response, request its readiness
endpoint through llama-swap (assuming the router listens on port 8080):

```sh
curl --fail-with-body http://127.0.0.1:8080/upstream/gufo-model/ready
```

The browser Load action and startup preloading (`hooks.onStartup.preload`)
request the upstream root `/`. Gufo answers `GET /` with the `/health` body,
503 while the model loads and 200 once ready, so both actions report success.
`checkEndpoint` controls readiness polling, not the triggering
request.

## Exposure

The server binds to `127.0.0.1` by default and answers every route without
credentials: it has no authentication layer. Any process that can reach the listener can run inference and read
metrics, so keep the bind local or place a reverse proxy in front. Select
the listen address with `--host`; it must be a numeric IPv4 or IPv6 address,
not a host name. `--host ::` is dual-stack: it accepts IPv6 clients and IPv4
clients, and an IPv4 client keeps its IPv4 address as its client ID.
`--host ::1` accepts only IPv6 loopback clients. TLS termination and any
access control belong in a reverse proxy.

A stale `Authorization` header from an old client is accepted and ignored.

Request bodies, prompts and generated text are not logged.

Tool definitions and generated tool calls are treated as data. `gufo`
never executes tools, shell commands, URLs, or generated code.

The HTTP transport currently uses a wildcard CORS origin and the same
listener for inference and metrics.

`gufo eval` sends `OPENAI_API_KEY` as a bearer credential to remote
endpoints that require one; the Gufo server itself ignores credentials.

## Lifecycle and limits

Qwen image inputs have no fixed image-count cap. Their expanded tokens must fit
the model context; image-byte, pixel, request-body and read-time budgets still
apply to the submitted history, including base64 images.

The HTTP transport bounds connection count and request-body size. The scheduler
bounds admission and output buffering and propagates client cancellation to
model runners. An in-flight GPU operation may finish before its request retires.
See the [command line](../cli.md) for supported configuration flags.

Admission groups text requests by the socket peer's IP address across chat and
compatibility endpoints. Caller-provided identity headers do not affect quotas;
clients behind the same proxy or NAT share a peer quota.

Cancellation retains the last successfully executed conversation frontier and
the immutable prompt snapshot. It does not execute a selected but unfinished
token to populate the cache. A failed model operation invalidates its mutable
state; the prompt snapshot remains available for safe replay. Image identity,
positions and speculative state participate in restoration and cache isolation.

`GET /health` reports process liveness. `GET /ready` returns 503 until a model
service is ready, then reports `status` and the active model. Both read the
recorded device state without performing GPU work on each poll. HTTP model replacement and persistent Responses
conversations are not implemented.

While `gufo serve llm` loads its model and walks the disk cache tier, the port
already answers every request, `/health` included, with 503
`{"error":{"message":"Loading model","type":"unavailable_error","code":"loading"}}`
and `Retry-After: 1`, as llama-server does. A refused connection therefore
means the process is not running; supervisors such as llama-swap, which treat
only a 200 health check as ready, keep waiting until the model is loaded. If
the address cannot be bound early, the port stays closed until the load ends.

A GPU reset (for example `amdgpu` recovering from a MES hang) permanently
invalidates the process's HIP context. The text scheduler checks it after five
seconds of continuous idle time, then every five seconds while idle. The probe
uses a preallocated four-byte buffer and private stream. Submission and polling
never wait for completion; an outstanding probe is reused. Arriving requests
interrupt the idle wait and execute without waiting for that probe. Active
inference performs no periodic device checks.

After a generation failure the scheduler also runs a bounded device probe
(at most 5 s), before invalidating request state.
Only a hard HIP error from the probe
marks the device lost. A probe still pending after 5 s may be queued behind
long kernels, so it logs `event=device_probe_timeout` and counts as usable. A
usable device leaves the original failure unchanged. A lost device logs
`event=device_lost remedy=restart reason=<driver error>` once and makes the
loss permanent for the process:

- the failing request reports `device_lost`: a 503 with
  `GPU context lost; restart required` and no `Retry-After` before the
  response starts, or the same code in the stream's terminal error event
  after it started. `/v1/responses` streams end with `response.failed`
  carrying the Responses `server_error` code, since that code enum is closed;
  the message names the device loss;
- `/health`, `/ready` and their aliases return 503 with
  `{"status":"device_lost","error":{...,"code":"device_lost"}}`;
- every text-backend `POST` route, including unimplemented stubs that would
  otherwise answer 501, returns 503 `device_lost` without reaching the device;
- `gufo serve embedding` probes the same way after a failed device batch. It
  logs `event=device_lost` once, and its failing and later `/v1/embeddings`
  requests return 503 `device_lost`;
- `gufo serve` logs `event=device_lost_shutdown`, runs the `SIGTERM` shutdown
  and exits with status 75 (`EX_TEMPFAIL`), also when an external `SIGTERM`
  stops it after the loss. Teardown can block on the dead device, so the
  process exits with status 75 after 10 s regardless.

A device call that never returns is a different failure: nothing fails, so
nothing is probed, and the server keeps answering `/health` and sending stream
heartbeats while generation has stopped. `--stall-timeout-ms MS` bounds that.
Progress means a request being admitted, a work unit starting, a pending
checkpoint capture completing or the idle wait returning. When the scheduler has work but none of those
happens for `MS`, it logs
`event=device_lost remedy=restart reason=scheduler made no progress for <MS> ms`
and follows the lost-device path above, including the exit with status 75 and
the `gufo_device_lost_total` increment. This covers a hang on the scheduler
thread, in a capture the scheduler is waiting for, and in the idle device
probe. Detection takes between `MS` and about 1.5 times `MS`. Requests already
in flight end when the process exits. The default, 0, disables the check;
other values must be at least 60000, because a deep checkpoint capture or a
post-failure device probe can legitimately hold progress for tens of seconds.
An idle server and a queued request never count as stalled. While any other
request is still progressing, a hung capture is not reported; it is once
nothing else can run. A disk-cache write that never completes is not detected
while generation continues.

Text streams defer HTTP headers until the scheduler admits the request and
starts its prompt, or until the request has waited five seconds in the queue,
whichever comes first. Chat's initial role and Responses lifecycle events are
sent with them. Failures before that point return a JSON error with an
appropriate 5xx status; device loss is 503 `device_lost`. Later failures, even
before the first token, use terminal SSE errors. Heartbeats begin once headers
commit, so long prefills and queue waits keep sending bytes. Explicit
`return_progress: true` sends headers immediately and adds live prefill
progress.

`/metrics` exposes `gufo_device_lost_total`, incremented once per confirmed
context loss, including idle detection. It stays unchanged for recoverable
errors and pending probes. The counter resets when the process restarts; the
fatal loss log and supervisor exit status remain useful when a metrics scrape
misses the brief period before exit.

`gufo diagnose` inspects system availability in a separate process. It cannot
validate the serving process's existing HIP context; use the serving health
endpoints and a supervisor restart policy for this failure mode.

The listener observes the sticky loss independently of response writes, so
a blocked streaming client cannot delay arming that watchdog. Failed resident
requests retain their state and snapshot workers until process exit, avoiding
HIP resets, frees and transfer joins on the lost context. The watchdog is armed
before logging or shutdown; the fatal exit bypasses model destructors and log
flushing. The operating system reclaims those resources.

Run the server under a supervisor that restarts on failure, such as systemd
`Restart=on-failure` or a container `restart: always`/`on-failure` policy;
a restart is the only recovery.

## Metrics

`/metrics` exposes total prompt/generated tokens and the latest prompt/decode
speeds. The token counters advance as each prefill chunk and generated token
executes, so their rates show live throughput; prompt tokens exclude cache
hits. `llamacpp:requests_processing` counts admitted requests, including cache
preparation and cleanup; `llamacpp:requests_deferred` counts requests waiting
for a session. The speed gauges retain the latest nonzero request rates.
When the scheduler finishes or cancels a request, whether or not the client
reads the result, it adds the request's cached prompt tokens
(`llamacpp:prompt_tokens_cached_total`), prefill and decode seconds
(`llamacpp:prompt_seconds_total`, `llamacpp:tokens_predicted_seconds_total`),
speculative verification rounds (`llamacpp:spec_decode_num_drafts_total`),
proposed and accepted draft tokens
(`llamacpp:spec_decode_num_draft_tokens_total`,
`llamacpp:spec_decode_num_accepted_tokens_total`), proposed and accepted
prompt-lookup copies (`gufo_spec_lookup_tokens_total`,
`gufo_spec_lookup_accepted_tokens_total`), and raises
`llamacpp:n_tokens_max` to its prompt plus generated tokens. A request counts
even if the client disconnects before reading it; requests that fail in the
scheduler, such as on a deadline or a runner error, add none of these.
`llamacpp:kv_cache_usage_ratio` is the in-flight prompt and generated tokens
over sessions times context; retained cache entries are not counted.
`Server-Timing`, generation `timings`, and Chat Completions `usage.gufo`
provide request-level measurements.

`GET /slots` (alias `/v1/slots`) lists one llama-server slot per `--sessions`
entry. A session is processing from admission, including cache preparation,
until the request ends, so the processing slots match
`llamacpp:requests_processing`; queued requests do not appear.
`id_task` is the request id, or -1 when idle. `n_prompt_tokens` is the whole
prompt, `n_prompt_tokens_cache` the tokens restored from cache, and
`n_prompt_tokens_processed` the uncached tokens prefilled so far, as in
`timings.prompt_n`. `next_token[0].n_decoded` counts generated tokens and
`n_remain` the remaining token budget. Idle slots report zero counts and
`n_remain` -1. `task_id` and `state` (0 idle, 1 processing) remain for older
clients; `prompt` is always empty.
Slots and metrics are separate snapshots; requests can advance between polls.

`GET /props` reports what a text server runs, in llama-server's shape, so
llama-server tooling can record which model, build and settings answered it.
`?model=` is accepted and not required. `model_alias` is the served model
name, `model_path` is `--model`, `build_info`
the version and revision `gufo --version` prints, `total_slots` the
`--sessions` count, `default_generation_settings.n_ctx` the per-session context
and `default_generation_settings.params` the sampling a request without
sampling fields gets, with `n_predict` and `max_tokens` -1 when no default
limit is set. `modalities.vision` is true only when a vision encoder
loaded. Fields the server cannot report are omitted rather than sent as zero.

`POST /apply-template` takes a Chat Completions body and returns
`{"prompt": "..."}`, the text the model's compiled chat template renders it
to, as llama-server's route does, so tooling can inspect, count or fingerprint
what a request becomes. `model` is optional. The server's `--think` and
`--reasoning-effort` defaults apply as in Chat, so a body that sets neither
shows what such requests run at; tool and response-format constraints add the
same text they add before generation. Nothing is tokenized, prefilled or
cached. An image renders as its single placeholder, not its expanded visual
tokens. `stream` is ignored, as in llama-server; template errors return 400
as in Chat, and backends that cannot render a template as text return 501.

`POST /tokenize` takes `{"content": "..."}` and returns `{"tokens": [...]}`,
the loaded model's token IDs for raw text. Control-token spellings become their
special tokens and nothing is prepended, as in `/completion` and llama-server's
defaults (`add_special` false, `parse_special` true). Those two fields and
`with_pieces` are accepted only at their defaults; other values return 400.
Nothing is prefilled or cached. Backends that cannot tokenize return 501.

Tokenizing an `/apply-template` result counts the rendered text as a raw
prompt. It may differ from the Chat request's token count: literal control-token
spellings in message content are preserved as text by Chat, and images expand
to visual tokens. For the Chat count, send the original request to
`/v1/chat/completions/input_tokens`.

Streaming terminal chunks always include llama.cpp-compatible `timings`, even
without `stream_options.include_usage`. `prompt_n` counts newly processed
tokens; `cache_n` counts reused tokens. llama-swap uses these fields on every
turn. Gufo-specific details stay in `usage.gufo`.

Not implemented yet: retained-cache KV metrics, a validated administrative reload/drain
interface, and in-process recovery after device reset or suspend/resume.

## Troubleshooting logs

Server lifecycle and request logs go to stderr. Each request gets an
`X-Request-ID` response header matching its `request=rN` log entries. Inference
requests log receipt and completion; streaming completion is logged after the
stream ends. Successful health/metrics and video-status polls are quiet at the
default level.

Set verbosity with `--log-level <error|warn|info|debug>` (default `info`);
`-v`/`--verbose` is shorthand for `--log-level=debug` and cannot combine with
the canonical spelling — passing both is a usage error rather than a precedence
to guess at. Each tier is a threshold, not only an addition:
`--log-level error` shows ERROR lines alone, so a 5xx response or stream
failure still logs while a 4xx refusal, which logs at WARN (a 429 admission
refusal included), is filtered out.

The threshold also covers the lifecycle lines. `event=listening` (the bound
address, auth mode and connection limits) and `event=shutdown_requested` (the
signal that asked for a stop) are INFO-tier, so `--log-level=warn` and
`--log-level=error` start and stop with no output at all. Keep the default
`info` for systemd units and CI that read the startup banner, or confirm boot
with `GET /health` or `GET /ready`.

The `debug` tier adds:

- the resolved server options as `event=options` before the model opens.
  This line and the `event=listening` banner
  are separate on purpose: the banner only appears once the model has loaded and
  the listener is accepting, so a load that fails or hangs leaves
  `event=options` as the sole record of what was asked for;
- completion logs for the polls that are quiet by default, and `event=received`
  for GET requests;
- scheduler decisions: `event=admitted` with the pending queue depth and the
  per-client depth the request joined, `event=admission_refused` naming the limit
  that rejected it and the client, and `event=backpressure` once per stream
  naming which output budget refused a token (`request_buffer` or
  `total_buffer`) with the byte counts;
- cache candidate detail behind the summarised lines: `event=candidate_skip`
  gives the entry index, its token length and the first guard that failed
  (`unavailable`, `input_identity`, `stable_prefix_boundary` or
  `token_prefix`), and `event=capture_evicts` names the checkpoint a capture is
  about to overwrite. A request that finds every slot busy emits its skip
  records once, so a long wait does not repeat them on every 10ms poll.

Loader phases stay at INFO: the weight-mapping and session-preallocation work is
HIP-only code, so deeper sub-phases there need a GPU build to verify and are not
part of this tier. The imported DeepSeek V4 Flash runtime logs through the same
threshold rather than writing to stderr directly, so its informational startup
lines (the ROCm model-cache and managed-KV lines, DSpark attachment, the shared
batch workspace) are suppressed by `--log-level=warn`/`error` too.

Prompt text and message bodies stay unlogged at every level, and debug
lines use the same escaping and redaction as the rest of the log. Only the
opt-in [content trace](#content-trace), a separate file, records content. Client
identity is the exception: scheduler admission lines name a client by the peer
address of its socket (`client_id=127.0.0.1` on the default loopback bind), so a
public `--host` writes client IP addresses into the debug tier.

Pass `--log-progress` to `gufo serve llm` to log each prefill chunk, each
50-token decode boundary, and the final decode remainder. Each line includes the
request ID, completed and total tokens, percentage, current speed and average
speed. Speculative requests also include accepted and proposed drafts. Progress
lines are INFO-tier: `--log-level=warn` or `--log-level=error` would discard
them, so the server rejects that combination at startup instead of ignoring the
flag.

### Content trace

Some failures only show in the content itself, such as tool-call markup leaking
into `content` or a history that the template renders differently. For those,
start `gufo serve llm` with `--trace <PATH>`. The server appends one JSON
object per line to `PATH` and logs `event=trace_enabled` at WARN. A new file is
created readable and writable by its owner only; an existing file keeps its
permissions. A path that cannot be opened fails startup before the model loads.

Each request to `/v1/chat/completions`, `/v1/completions`, `/v1/responses`,
`/v1/messages` or `/completion` writes three records. Every record has `time`,
`event` and the `request` id shown in the `X-Request-ID` header and in the
`request=rN` log lines:

- `request`: `method`, `path` and the `body` as received.
- `generation`, written by the scheduler shared by every text model: its
  `generation` number (the `request=` value of scheduler debug and progress
  lines), `max_tokens` after clamping to the context, the effective `sampling`,
  `prompt_tokens`, `cache` (`memory`, `disk` or `miss`), `cached_tokens`, the
  miss detail the completion log also reports, `generated_tokens`, `finish`,
  `error` when generation failed, the `prompt` decoded from its tokens with
  special tokens spelled out, and the raw `output` before reasoning and
  tool-call parsing.
- `response`: `status`, `outcome`, whether the reply was a `stream`, and the
  `body` the client received. For a stream this is the event stream as
  written, `: ping` keepalives included, so a difference between streaming and
  non-streaming parsing shows up in the trace.

A request that fails before generation, for example with invalid JSON, has
no `generation` record. Requests refused before routing (malformed HTTP or an
oversized body) and other routes are not traced.
Malformed UTF-8 is replaced with U+FFFD. A record that cannot be written in
full, for example on a full disk, is cut from the file and the first failure is
logged as `event=trace_write_failed`; a file that cannot be cut back, such as a
pipe, stops tracing with `event=trace_disabled`. To read one request:

```sh
jq 'select(.request == "r12")' trace.jsonl
```

The file holds prompts, tool results and generated text in full: treat it
like the conversations themselves and delete it when done. It is never
rotated or truncated.

Text completion logs include stop/length/cancellation, queue and first-token
latency, prefill/decode speed, execution width, memory/disk cache hits and reused
tokens, plus accepted/proposed drafts and acceptance percentage. These metrics
are logged even when a streaming client does not request a usage chunk.
Errors include a stable error code; disconnects and stream failures are marked.
On a cache miss, `cache_miss_reason` distinguishes a missing checkpoint, changed
token prefix, changed image input, and explicit cache bypass. Common-prefix and
nearest-checkpoint token counts report token agreement even when image
identity differs; matching image-placeholder tokens do not imply matching
pixels. Prompt text is not logged. Adding tools or editing system/developer
instructions near the beginning invalidates the later state: keep those inputs
stable during an agent conversation. A disconnected SSE stream may never
deliver its terminal usage chunk; proxy counters can then show zero despite
generated tokens.
The server's cancellation log retains the actual token counts.

Routine cache replacement is quiet; failed captures, disk corruption and cache
capacity refusals produce warnings. Cache capacity is a budget, not allocated
memory.

Loading logs report elapsed time, model, context and session capacity,
speculative mode and memory. `rss_mib` is process resident memory;
`host_available_mib` is available system memory. At load completion,
`gpu_device_used_mib` is device-wide HIP usage. These overlap on unified memory
and must not be added together.
With disk caching enabled, `phase=artifact_identity` identifies full-file
SHA-256 work. Digests are cached against the open file's identity, size and
modification/change timestamps; unchanged artifacts avoid another scan.
Startup qualification measures process launch through first prefill and first
token, including work deferred until the first request.

Audio summaries include model, duration and generation/transcription timings.
Video requests log a job ID linking queue, start, throttled phase progress and
completion/failure events. Video startup validates inventory; weights load
lazily in the worker. Control characters are escaped in log lines.

## Focused checks

- `json_test`: number precision, Unicode escapes, malformed input and depth limits.
- `http_server_test`: transport framing, compatibility validation,
  sampling forwarding, request logs at the default and debug tiers, and streaming
  failures without loading a model.
- `serve_cli_test`: executable help, argument wiring and rejected configurations,
  including that `--log-level` and `-v` change the emitted log rather than only
  parsing.
- `openai_chat_test`: chat parsing, streaming, images, tools and sampling controls.
- Per-model serving tests: greedy/sampled decoding, batching, cache reuse and
  cancellation. Use the affected model's benchmark README for commands and limits.

For session-boundary changes, also run a six-turn conversation: remember a fact,
force one token-limit stop, recall the fact, update it and recall the update.
Compare greedy AR/speculative token traces and repeat each sampled mode with the
same seed. Mix EOS and token-limit stops rather than testing only fixed-length
generation. This is a session/replay check, not a capability or distribution test.

2026-09-19: six-turn greedy AR/speculative traces matched for DS4/DSpark,
Flash-Next/MTP and Qwen27B Q4/Q8 with the Q4 DFlash2 draft. Speculative runs
repeated exactly at temperature 0.6, seed 7, with a 32-token turn limit.
Two simultaneous Qwen27B Q4/DFlash2 HTTP conversations also retained independent
facts through six turns, with streaming and memory-cache reuse.

DSpark's short boundary check is available without the complete serving suite:

```sh
# Set GUFO_DEEPSEEK_V4_FLASH_MODEL and GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL first.
nix develop -c build/gpu-test/tests/models/deepseek_v4_flash/ds4_serving_test --dspark-eos
```
