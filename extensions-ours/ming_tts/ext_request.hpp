// Mapping from the gufo extension API speech request (gufo::ext::
// SynthesisRequest, API 0.2) to the Ming-3.0-tts engine request.
//
// API 0.2 has no request fields for Ming's controls, and the core
// /v1/audio/speech handler ignores unknown members, so the controls travel in
// the OpenAI `instructions` field (SynthesisRequest::instruct):
//
//   - a JSON object:  {"tempo":"slow","pitch":"low","volume":"high"}
//                     {"dialect":"cantonese"}  {"emotion":"sad"}
//                     {"instruction":"Whisper very softly ..."}
//   - key=value list: "tempo=slow, pitch=low"  ("," or ";" separated)
//   - one of the seven emotion names: "sad", "Wut", "froehlich", ...
//   - anything else:  a free-text caption (Ming FREEFORM task).
//
// Keys: emotion; tempo (alias speed); pitch; volume; dialect; instruction
// (aliases instructions, caption). Values as accepted by EmotionLabel,
// SpeedLabel, PitchLabel, VolumeLabel and DialectLabel. One Ming task per
// request, as in the fork's 1.x API (ResolveTaskInstruction).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MING_TTS_EXT_REQUEST_HPP_
#define GUFO_MODELS_MING_TTS_EXT_REQUEST_HPP_

#include <gufo/ext/tts.hpp>

#include <string>
#include <string_view>

#include "ming_tts/synthesis.hpp"

namespace gufo::models::ming_tts {

// Controls parsed from `instructions`; empty strings are unset.
struct InstructControls {
  std::string emotion;
  std::string instruction;
  std::string speed;
  std::string pitch;
  std::string volume;
  std::string dialect;
};

// Parses `instruct` as described above. Fails on malformed JSON, a non-string
// value, an unknown key, or a list that mixes key=value parts with free text.
[[nodiscard]] bool ParseInstructControls(std::string_view instruct,
                                         InstructControls* controls,
                                         std::string* error);

// Builds the engine request: text, language, controls (validated with
// ResolveTaskInstruction and the label helpers), mono reference audio, the
// reference transcript (ZERO_SHOT prompt latents) unless the request asks for
// speaker_embedding_only, seed, the decode-step cap (max_new_tokens, at most
// 1000) and the streaming callback.
[[nodiscard]] bool BuildEngineRequest(const ext::SynthesisRequest& request,
                                      SynthesisRequest* out, std::string* error);

}  // namespace gufo::models::ming_tts

#endif  // GUFO_MODELS_MING_TTS_EXT_REQUEST_HPP_
