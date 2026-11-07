// Mapping from the gufo extension API speech request (gufo::ext::
// SynthesisRequest, API 0.2) to the VoxCPM2 engine request.
//
//   - `instructions` (instruct): free-text style / emotion, sent as the
//     "(instruction)text" prefix (VoxCPM2 controllable cloning). The 1.x
//     fork's separate "emotion" field has no API 0.2 counterpart; put it into
//     `instructions` instead.
//   - reference audio alone (or voice_clone_mode speaker_embedding_only):
//     isolated reference prefix (reference mode).
//   - reference audio + reference_text: "ultimate cloning", the same clip also
//     continues as the prompt with its transcript.
//   - no reference: zero-shot voice.
//   - `language` is ignored (VoxCPM2 detects the language from the text).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_VOXCPM2_EXT_REQUEST_HPP_
#define GUFO_MODELS_VOXCPM2_EXT_REQUEST_HPP_

#include <gufo/ext/tts.hpp>

#include <cstddef>
#include <string>

#include "voxcpm2/synthesis.hpp"

namespace gufo::models::voxcpm2 {

// VoxCPM2 encodes the whole reference clip into its 8192-position context.
inline constexpr std::size_t kMaximumReferenceSeconds = 60;

// Fails when the reference audio is longer than kMaximumReferenceSeconds.
[[nodiscard]] bool BuildEngineRequest(const ext::SynthesisRequest& request,
                                      SynthesisRequest* out, std::string* error);

}  // namespace gufo::models::voxcpm2

#endif  // GUFO_MODELS_VOXCPM2_EXT_REQUEST_HPP_
