# Qwen3-Omni-30B-A3B-Captioner

The captioner listens to one audio clip and writes a text description of
it: speech, voices, music, noises, mood. It is a gufo-studio-windows
modality (`gufo serve omni`) and needs a HIP build; a build without it
answers `this gufo build has no Qwen3-Omni support (needs HIP)` at startup.

## Files

Two GGUF files: the text model and the matching audio projector
(`--mmproj`). The production preset of the two Strix Halo machines uses
the `mradermacher/Qwen3-Omni-30B-A3B-Captioner-GGUF` conversion:

```ini
[omni/qwen3-omni-captioner]
model = D:\llm-modelle\gufo\modelle\mradermacher\Qwen3-Omni-30B-A3B-Captioner-GGUF\Qwen3-Omni-30B-A3B-Captioner.Q4_K_M.gguf
mmproj = D:\llm-modelle\gufo\modelle\mradermacher\Qwen3-Omni-30B-A3B-Captioner-GGUF\Qwen3-Omni-30B-A3B-Captioner.mmproj-f16.gguf
max-request-bytes = 67108864
```

## Run

Options of `gufo serve omni --help` (2.0.4):

| Option | Default | Meaning |
| --- | --- | --- |
| `--model <PATH>` | required | Captioner text GGUF |
| `--mmproj <PATH>` | required | Matching projector GGUF (audio tower) |
| `--context <N>` | 8192 | Context capacity: audio tokens plus caption |
| `--max-tokens <N>` | 1024 | Caption length limit and default |
| `--served-model-name <NAME>` | `qwen3-omni-captioner` | Public API model ID |

```sh
gufo serve --host 127.0.0.1 --port 8080 omni \
  --model Qwen3-Omni-30B-A3B-Captioner.Q4_K_M.gguf \
  --mmproj Qwen3-Omni-30B-A3B-Captioner.mmproj-f16.gguf
```

## Request

`POST /v1/chat/completions` (also `/chat/completions`) in OpenAI chat
shape. The last `user` message must have a content array with exactly one
`input_audio` part (`src/cli/serve/media/omni_api.cpp`):

```json
{
  "model": "qwen3-omni-captioner",
  "messages": [{"role": "user", "content": [
    {"type": "input_audio", "input_audio": {"format": "wav", "data": "<base64 WAV>"}}
  ]}],
  "max_tokens": 512
}
```

- `format` must be `wav` (the default); `data` is canonical base64 of a
  WAV file, at most 48 MiB. Raise the server's `--max-request-bytes` for
  long clips.
- `max_tokens` or `max_completion_tokens` must be positive and are capped
  at the server's `--max-tokens`.
- Sampling: `temperature` (default 0 = greedy), `top_p` (0.95), `top_k`
  (20), `seed` (1234). `temperature` must be >= 0 and `top_p` in (0, 1].
- `stream: true` returns server-sent `chat.completion.chunk` events;
  `stream_options.include_usage` adds a final usage chunk.
- Errors (400): `missing_audio`, `unsupported_audio_format`,
  `invalid_audio`, `invalid_max_tokens`, `invalid_sampling`,
  `invalid_messages`, `invalid_json`.

The answer is a normal chat completion whose `content` is the caption,
with `usage` token counts.

Behind `gufo router`, a `[omni/<id>]` section starts this worker (fork
builds only); `[remote/<id>]` with `modality = omni` names one on another
router. See the [model router](../internal/server.md#model-router).
