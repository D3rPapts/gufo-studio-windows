# Ming-3.0-tts-0.5B (inclusionAI, 44.1 kHz)

Native speech synthesis with speaker-embedding voice cloning, emotion labels and
free-text style instructions from the official `inclusionAI/Ming-3.0-tts-0.5B`
checkpoint (MIT). The checkpoint combines a Qwen2.5-0.5B backbone, a DiT
flow-matching head that samples one continuous 64-dim audio-VAE latent per
step (12.5 steps per second of audio), an aggregator that feeds each latent
back into the backbone, a stop head, and a 24-layer sliding-window Qwen2 VAE
decoder with an ISTFT head. The CAM++ speaker encoder is read directly from
the shipped `campplus.onnx`; Python and ONNX Runtime are not runtime
requirements.

Runtimes: a CPU reference runtime (FP32 math on the BF16 weights) that matches
the Python reference stage by stage, and a gfx1151 HIP runtime for the AR loop
and the VAE decoder (text front end, CAM++ and the VAE encoder stay on the
CPU). The HIP runtime is used automatically when a GPU is present
(`GUFO_MING_BACKEND=cpu` forces the CPU, `=hip` makes a missing GPU an error);
`X-Gufo-TTS-Backend` reports `native-hip` or `native-cpu`. **The HIP runtime is
compiled but has not been run on a GPU yet** (`tests/models/ming_tts/hip_test.cpp`
is the pending check).

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Usage

```sh
hf download inclusionAI/Ming-3.0-tts-0.5B --local-dir models/Ming-3.0-tts-0.5B
./result/bin/gufo serve tts --model models/Ming-3.0-tts-0.5B \
  --voice erzaehler=reference.wav --voice-lang erzaehler=de
```

`gufo serve tts` detects the checkpoint from `config.json`
(`BailingMMNativeForConditionalGeneration`, `model_type: dense`) plus
`campplus.onnx`. The default HTTP model ID is `ming-3.0-tts-0.5b`
(`ming-3.0-tts` is accepted as an alias).

```sh
curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"ming-3.0-tts","voice":"erzaehler","language":"de",
       "emotion":"angry","input":"Das kann doch nicht dein Ernst sein!"}' \
  --output speech.wav
```

Request fields specific to Ming:

| Field | Meaning |
| --- | --- |
| `emotion` | EMOTION task. One of `happy`, `sad`, `angry`, `fear`, `disgust`, `surprise`, `neutral` (German names such as `Wut`, `Trauer`, `Angst` and the Chinese labels 高兴/悲伤/愤怒/恐惧/厌恶/惊讶/中性 also work). |
| `instructions` | Free-text style caption (FREEFORM task), e.g. a whisper description. Takes precedence over `emotion`. |
| `tempo` | BASIC task: the model's own speaking rate, `slow`, `medium` or `fast` (also `normal`, German `langsam`/`mittel`/`schnell`, or the Chinese values 慢速/中速/快速). Not the same as OpenAI `speed`, which time-stretches the finished WAV; both may be combined. |
| `pitch` | BASIC task: `low`, `medium` or `high` (also `normal`, German `tief`/`mittel`/`hoch`, or 低/中/高). |
| `volume` | BASIC task: `low`, `medium` or `high` (also `quiet`/`loud`, German `leise`/`mittel`/`laut`, or 低/中/高). |
| `dialect` | DIALECT task: `cantonese` or `chuanyu dialect` (Sichuan-Chongqing; also `chuanyu`). Chinese text only. Like the reference, a dialect request drops the `Target language:` line (the request or preset `language` is ignored). |

`tempo`, `pitch` and `volume` may be combined with each other (unset ones stay
`null` in the caption). As in the reference `tts_tasks`, each request uses one
task: BASIC controls, `dialect`, `emotion` and `instructions` cannot be mixed
(400 `conflicting_controls`). Unknown values return 400 `invalid_tempo`,
`invalid_pitch`, `invalid_volume` or `invalid_dialect`; other models reject
these fields with `unsupported_field`. The model was trained on these
controls mainly with Chinese and English speech; their effect on German has
not been measured yet.

```sh
curl -sS http://127.0.0.1:8080/v1/audio/speech   -H 'Content-Type: application/json'   -d '{"model":"ming-3.0-tts","voice":"erzaehler","language":"de",
       "tempo":"slow","pitch":"low","volume":"low",
       "input":"Es war einmal, vor langer Zeit."}'   --output slow.wav
```
| `language` | `de`, `en`, `zh`, `ja`, `ko`, `fr`, `es` (or full names). Adds the `Target language:` line; omitted means Ming's default prompt without it. |
| `reference_audio` | Base64 WAV for the CAM++ speaker embedding (no transcript needed). Embeddings are cached per reference. |
| `reference_text` | Transcript of `reference_audio`: additionally encodes the reference with the audio-VAE encoder into prompt latents (the reference's ZERO_SHOT layout). `--voice-text` does the same for presets. |
| `seed` | Seeds the flow-matching start noise. |
| `max_new_tokens` | Upper bound on AR steps per text fragment (capped at 1000; the 512-position decode window is the hard limit). |

Multi-fragment FREEFORM requests reuse the first generated fragment as voice
prompt (latents, speaker embedding and its text) for the following fragments,
as the reference does.

Text is split like the reference (`cut_text_by_semantic_length`, about 24
words per fragment); every fragment is synthesized separately, its trailing
silence trimmed to 0.3 s, and the pieces are concatenated. With
`response_format: "pcm"` each finished fragment is streamed as soon as it is
decoded.

## Not yet supported

- Verified GPU numbers (HIP runtime untested so far).
- Text normalisation for Chinese/English (the reference uses a pynini FST);
  other languages are passed through unchanged exactly like the reference.
- Temperature > 0 uses gufo's own RNG, so seeded outputs differ from PyTorch.
