# Ming-3.0-tts extension (gufo extension API 0.2)

Out-of-tree `speech_synthesis` provider for inclusionAI Ming-3.0-tts-0.5B
(model ID `ming-3.0-tts-0.5b`, 44.1 kHz). Ported from the gufo-studio-windows
1.x fork. Written with AI assistance (Claude Code).

Build: `-DGUFO_EXTENSION_DIRS=<folder that holds ming_tts/ and voxcpm2/>`.
Serve: `gufo serve tts --model <Ming-3.0-tts-0.5B> [--served-model-name ming-3.0-tts-0.5b --served-model-name ming-3.0-tts]`.

## Controls in `instructions`

API 0.2 has no request fields for Ming's controls (the 1.x fields `emotion`,
`tempo`, `pitch`, `volume`, `dialect` are ignored by the 2.0 core), so they
travel in `instructions`:

| `instructions` | Ming task |
|---|---|
| `{"tempo":"slow","pitch":"low","volume":"high"}` or `tempo=slow, pitch=low` | BASIC |
| `{"dialect":"cantonese"}` or `dialect=chuanyu dialect` | DIALECT |
| `sad`, `Wut`, `{"emotion":"happy"}` | EMOTION |
| any other text, or `{"instruction":"..."}` | FREEFORM caption |

Values: tempo slow/medium/fast (langsam/mittel/schnell), pitch and volume
low/medium/high (tief/hoch, leise/laut), dialect cantonese/chuanyu dialect,
emotion happy/sad/angry/fear/disgust/surprise/neutral (also German names).
One task per request; an invalid value fails the request (HTTP 500 from the
core, see the open API points in WAVE-B-STAND.md).

## Cloning

Reference audio always gives the CAM++ speaker embedding. With
`reference_text` (default `voice_clone_mode` "icl") the transcript also adds
the ZERO_SHOT prompt latents. Reference audio without a transcript needs
`"voice_clone_mode": "speaker_embedding_only"` (the 2.0 core requires a
transcript otherwise). Voice presets (`--voice name=file.wav`) work as before.

## Tests

- `ming_tts.controls` (CPU, no checkpoint)
- `ming_tts.oracle` (CPU, `GUFO_MING_MODEL` + `GUFO_MING_REF`)
- `ming_tts.provider` (`GUFO_MING_MODEL` + `GUFO_MING_VOICE`)
- `ming_tts.hip` (GPU)
