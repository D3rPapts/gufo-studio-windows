# IndexTTS-2.5 German extension (gufo extension API 0.2)

Out-of-tree `speech_synthesis` provider for IndexTTS-2.5 (IndexTeam) with the
German GPT fine-tune (sharrnah/index-tts-2.5-german); model ID
`indextts-2.5-german`, 22.05 kHz, voice cloning from a reference clip.
Depends on the `ming_tts` extension (safetensors reader, resampler, Kaldi
fbank, FFT, HIP kernels) and on `voxcpm2` (HIP attention and skinny GEMM).
Ported from the gufo-studio-windows 1.x fork (branch feat/indextts-141).
Written with AI assistance (Claude Code).

License note: the IndexTTS-2.5 weights use the bilibili Model Use License;
check it before commercial use.

Build: `-DGUFO_EXTENSION_DIRS=<folder that holds ming_tts/, voxcpm2/ and indextts/>`.
Serve: `gufo serve tts --model <IndexTTS-2.5-german-bundle> [--voice name=clip.wav]`.

The bundle directory holds `config.yaml` (`version: 2.5`), `german/gpt.pth`
(or `gpt.pth`), `s2mel.pth`, `codec.pth`, `feat1.pt`, `feat2.pt`,
`wav2vec2bert_stats.pt`, `multilingual_zh_ja_yue_char_del.tiktoken`,
`w2v-bert-2.0/`, `campplus/` and `bigvgan_v2_22khz_80band_256x/`.

## Request mapping

API 0.2 has no emotion field (the core drops unknown JSON members), so the
emotion goes into `instructions`, like the Ming and VoxCPM2 controls.

IndexTTS-2.5 takes an 8-value emotion vector in this order:

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| English | happy | angry | sad | afraid | disgusted | melancholic | surprised | calm |
| German | Freude | Wut | Trauer | Angst | Ekel | Melancholie | Überraschung | Ruhe |

Accepted `instructions` forms (values and weights in [0, 1.2]):

- `emotion=[0,0,0.7,0,0,0.3,0,0]` (the vector)
- `emotion=Wut:0.6, Angst:0.2; emo_alpha=0.8` (names with weights)
- `emotion=traurig` (one name, weight 1)
- `{"emotion":[0,0,0.7,0,0,0.3,0,0]}`, `{"emotion":"traurig","emo_alpha":0.8}`,
  `{"emotion":{"Wut":0.6,"Angst":0.2}}`, `{"Wut":0.6,"Angst":0.2}`
- `[0,0,0.7,0,0,0.3,0,0]`
- the 1.x `emotion` strings: `Wut`, `sad:0.6, calm:0.2`, `0,0,0.6,0,0,0.2,0,0`

Names are English or German, case-insensitive (aliases such as `traurig`,
`wütend`, `froehlich`, `neutral`). `emo_alpha` (0..1, default 1) scales the
emotion as in `infer_v2_5.py`. Key=value pairs are separated by `;` or `,`.
Free text and unknown keys are errors: IndexTTS has no text style prompt.
Without an emotion the reference clip's own emotion is used.

Other fields:

- Reference audio is required: a `--voice name=clip.wav` preset or inline
  `reference_audio` (voice `voice-clone`, at most 10 s; presets are cut at
  15 s by the engine). The transcript is not used; the 2.0 core still wants
  `reference_text` unless the request sends
  `"voice_clone_mode": "speaker_embedding_only"`.
- `language`: German only. `de`, `german`, `deutsch` and the core's fill-in
  default `english` (sent when the client names none) mean German; anything
  else fails.
- Sampling: fields the client leaves out get the `indextts_de.py` values
  (temperature 0.7, top_k 30, top_p 0.8, repetition_penalty 10). The core
  fills its Qwen3-TTS defaults (0.9, 50, 1.0, 1.05) for missing fields, so a
  client that sends exactly one of those values gets the IndexTTS value.
  `greedy` is ignored. `max_new_tokens` is capped at 1500.
- `speed` other than 1.0 is rejected by the 2.0 core.

Example:

```json
{"model": "indextts-2.5-german", "voice": "erzaehler", "language": "de",
 "input": "Ich kann nicht mehr. Es tut so weh.",
 "instructions": "emotion=[0,0,0.6,0,0,0.2,0,0]"}
```

`gufo speak --model <bundle> --voice clip.wav --text "..."` synthesizes with
the clip (no emotion: `gufo speak` has no instructions option).

Environment: `GUFO_INDEXTTS_BACKEND=cpu|hip`, `GUFO_INDEXTTS_CFM_STEPS`
(4..50, default 25), `GUFO_INDEXTTS_DIT_FAST`, `GUFO_INDEXTTS_DIT_ATTN`,
`GUFO_INDEXTTS_PROFILE` (as in 1.x).

## GPU kernel libraries

The HIP runtime uses hipBLASLt (BF16 DiT projections) and hipBLAS. Both load
their Tensile kernel libraries relative to their own DLL. The extension is
linked statically into `gufo.exe`, and the host stages the HIP runtime DLLs
plus the `rocblas/` and `hipblaslt/` data directories beside `gufo.exe`
(`gufo_stage_hip_runtime(gufo)`), so nothing changes against 1.x.
`indextts_hip_test` gets the same staging next to its own executable.

## Tests

- `indextts.request` (CPU, no checkpoint; probes `GUFO_INDEXTTS_MODEL` when set)
- `indextts.oracle` (CPU, `GUFO_INDEXTTS_MODEL` + `GUFO_INDEXTTS_REF` +
  `GUFO_INDEXTTS_VOICE`, `GUFO_INDEXTTS_STAGES` to pick stages)
- `indextts.provider` (`GUFO_INDEXTTS_MODEL` + `GUFO_INDEXTTS_VOICE`)
- `indextts.hip` (GPU)
