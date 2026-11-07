# IndexTTS-2.5 (German) in gufo

Native C++ port of [IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) with the German GPT fine-tune
[sharrnah/index-tts-2.5-german](https://huggingface.co/sharrnah/index-tts-2.5-german), used through
`gufo serve tts --model <bundle dir>` and `/v1/audio/speech`. No Python at runtime.

*Written with AI assistance (Claude Code).*

## License

Both the base model and the German fine-tune are under the **bilibili Model Use License Agreement**
(`LICENSE` in both repositories): use including commercial use is allowed; a separate license is required above
100 million monthly active users or 1 billion RMB annual revenue; high-risk uses (medical, military, ...) and using
the model to improve other commercial AI models are prohibited; Chinese law applies. The auxiliary models keep
their own licenses: w2v-BERT 2.0 (MIT), CAM++ / 3D-Speaker (Apache-2.0), BigVGAN v2 (MIT). The reference code
(index-tts / TTS-Audio-Suite) was read, not copied; the oracle script imports it only for testing.

## Bundle layout

```
<bundle>/config.yaml  gpt.pth (base, optional)  german/gpt.pth  s2mel.pth  codec.pth  feat1.pt  feat2.pt
         wav2vec2bert_stats.pt  multilingual_zh_ja_yue_char_del.tiktoken
         w2v-bert-2.0/{config.json, preprocessor_config.json, model.safetensors}
         campplus/campplus_cn_common.bin
         bigvgan_v2_22khz_80band_256x/{config.json, bigvgan_generator.pt}
```
Detection: `config.yaml` with `version: 2.5` plus the files above. `german/gpt.pth` (a complete state dict;
SHA-256 `e8298315...e89e443`) is used when present, otherwise `gpt.pth`.

## Pipeline (= IndexTTS2.infer_generator with the indextts_de.py settings)

1. Text: `prepare_german` (character cleanup, numbers/dates/times/currency/units/abbreviations to words, lower
   case, `<word|PRON>` annotations), the infer cleanup, `<|de|> ` prefix, tiktoken BPE, segments of <= 120 tokens.
2. Reference (cached per clip, cut at 15 s): torchaudio-style resampling to 22.05 / 16 kHz, librosa `soxr_hq` to
   16 kHz for the emotion branch; 80-band log-mel (s2mel prompt), CAM++ style (192), w2v-BERT 2.0 hidden
   state 17 normalised with `wav2vec2bert_stats.pt`; emotion Conformer + Perceiver -> `emo_vec`.
3. Emotion: 8-value vector `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`, normalised
   (`normalize_emo_vec`), mixed from `feat2` rows chosen by CAM++ similarity to `feat1`.
4. GPT-2 (24 x 1280) AR decoder: `[spk + emo, 0, 0] + text` prefix, repetition penalty 10, temperature 0.7,
   top-k 30, top-p 0.8, stop token 8193.
5. Semantic codec decoder -> 50 Hz features -> length regulator (x 1.72) -> DiT/CFM (25 Euler steps, CFG 0.7)
   with the reference mel as prompt -> BigVGAN v2 -> 22.05 kHz, `clamp(32767 * wav)` int16.

Sampling uses gufo's own RNG (inverse CDF over the HF-processed probabilities), so the output is a valid sample
of the same distribution, not bit-identical to PyTorch's `multinomial`. Tests inject the oracle's uniforms/noise.

## API

```
POST /v1/audio/speech
{"model": "indextts-2.5-german", "voice": "<preset>" | "voice-clone", "input": "...",
 "reference_audio": "<base64 WAV, <= 10 s>",          # with voice-clone
 "emotion": [0,0,0.6,0,0,0.2,0,0] | "Trauer" | "Wut:0.6, Angst:0.2" | {"Wut": 0.6, "Angst": 0.2},
 "language": "de", "response_format": "wav" | "pcm", "seed": 42,
 "temperature": 0.7, "top_k": 30, "top_p": 0.8, "repetition_penalty": 10}
```
Emotion names: happy/Freude/fröhlich, angry/Wut/wütend, sad/Trauer/traurig, afraid/Angst, disgusted/Ekel,
melancholic/Melancholie, surprised/Überraschung, calm/Ruhe/neutral (weights 0..1.2). Only German text is supported
(`language` other than `de` -> 400). PCM streaming sends one chunk per text segment.

## Backends

`GUFO_INDEXTTS_BACKEND=cpu` forces the CPU reference runtime (FP32, weights used in place from the mapped
files); otherwise a HIP build uses the GPU for the GPT decoder, the CFM loop and BigVGAN (BF16 weights, FP32
activations) and `=hip` makes a missing GPU an error. Reference conditioning, sampling, codec and length
regulator stay on the CPU.

## Tests

- `indextts_oracle_test` (CPU, against `reference/indextts_oracle.py` on CPU fp32)
- `indextts_serve_test` (in-process `/v1/audio/speech`)
- `indextts_hip_test` (GPU vs. oracle, speed)
