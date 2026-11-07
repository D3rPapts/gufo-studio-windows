// Mapping from the gufo extension API speech request (gufo::ext::
// SynthesisRequest, API 0.2) to the IndexTTS-2.5 engine request.
//
// IndexTTS-2.5 is steered by an 8-value emotion vector (emo_vector order
// [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]). API
// 0.2 has no emotion field and the core ignores unknown JSON members, so the
// emotion travels in the OpenAI `instructions` field (SynthesisRequest::
// instruct), as for Ming-3.0-tts and VoxCPM2:
//
//   - key=value list:  "emotion=[0,0,0.7,0,0,0.3,0,0]"
//                      "emotion=Wut:0.6, Angst:0.2; emo_alpha=0.8"
//   - a JSON object:   {"emotion":[0,0,0.7,0,0,0.3,0,0]}
//                      {"emotion":"traurig","emo_alpha":0.8}
//                      {"emotion":{"Wut":0.6,"Angst":0.2}}
//                      {"Wut":0.6,"Angst":0.2}   (the 1.x object form)
//   - a JSON array:    [0,0,0.7,0,0,0.3,0,0]
//   - the 1.x string:  "Wut", "sad:0.6, calm:0.2", "0,0,0.6,0,0,0.2,0,0"
//
// Names are English or German, case-insensitive (EmotionIndex); weights and
// vector values lie in [0, 1.2] (ParseEmotionSpec). Keys: emotion (aliases
// emo, emo_vector), emo_alpha (alias alpha, in [0, 1], default 1). Anything
// else (free text, unknown keys) is an error: IndexTTS has no text prompt.
//
// Other fields:
//   - `language`: German only. "de", "german", "deutsch" (any case) and the
//     core's fill-in default "english" (sent when the client names none) map
//     to "de"; every other language is an error.
//   - reference audio is required (preset or inline); `reference_text` and
//     `voice_clone_mode` are ignored (IndexTTS clones from the audio alone).
//     Inline clips (voice "voice-clone") may be at most 10 s, presets are cut
//     at 15 s by the engine.
//   - sampling: the core fills the Qwen3-TTS defaults (temperature 0.9,
//     top_k 50, top_p 1.0, repetition_penalty 1.05) for fields the client
//     leaves out; each field still equal to that default takes the
//     indextts_de.py value instead (0.7, 30, 0.8, 10). `greedy` is ignored.
//   - max_new_tokens is capped at 1500 (GPT max_mel_tokens).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_INDEXTTS_EXT_REQUEST_HPP_
#define GUFO_MODELS_INDEXTTS_EXT_REQUEST_HPP_

#include <gufo/ext/tts.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "indextts/synthesis.hpp"

namespace gufo::models::indextts {

// Per-request inline reference clips (voice "voice-clone").
inline constexpr std::size_t kMaximumReferenceSeconds = 10;
inline constexpr std::string_view kCloneVoice = "voice-clone";
inline constexpr std::int32_t kMaximumMelTokens = 1500;

struct InstructControls {
  std::optional<std::array<float, 8>> emotion;
  float emo_alpha{1.0F};
};

// Parses `instruct` as described above.
[[nodiscard]] bool ParseInstructControls(std::string_view instruct,
                                         InstructControls* controls,
                                         std::string* error);

// Maps the request language to the engine language ("de"); fails for every
// language the German fine-tune cannot speak.
[[nodiscard]] bool ResolveLanguage(std::string_view language, std::string* out,
                                   std::string* error);

// Sampling with the indextts_de.py defaults for fields the client left at the
// core defaults.
[[nodiscard]] SamplingOptions ResolveSampling(const ext::TtsSamplingOptions& sampling,
                                              std::size_t max_new_tokens);

// Builds the engine request: text, language, emotion controls, mono reference
// audio (inline clips at most kMaximumReferenceSeconds), seed and sampling.
[[nodiscard]] bool BuildEngineRequest(const ext::SynthesisRequest& request,
                                      SynthesisRequest* out, std::string* error);

// int16 -> float such that the truncating PCM16 encoders of the core
// (static_cast<int16_t>(x * 32767)) reproduce the engine's int16 samples
// exactly, as in 1.x.
[[nodiscard]] inline float Int16ToFloat(std::int16_t s) {
  return s == 0 ? 0.0F
                : (static_cast<float>(s) + (s > 0 ? 0.25F : -0.25F)) / 32767.0F;
}

}  // namespace gufo::models::indextts

#endif  // GUFO_MODELS_INDEXTTS_EXT_REQUEST_HPP_
