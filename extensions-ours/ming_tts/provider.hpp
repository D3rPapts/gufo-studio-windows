// Ming-3.0-tts speech_synthesis provider entry points (tests use the factory
// directly). Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MING_TTS_PROVIDER_HPP_
#define GUFO_MODELS_MING_TTS_PROVIDER_HPP_

#include <gufo/ext/tts.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

namespace gufo::models::ming_tts {

// Public model ID (the 1.x fork's default); "ming-3.0-tts" stays reachable
// through --served-model-name.
inline constexpr std::string_view kMingModelId = "ming-3.0-tts-0.5b";

[[nodiscard]] std::unique_ptr<ext::TtsRunner> CreateMingTtsRunner(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error);

}  // namespace gufo::models::ming_tts

#endif  // GUFO_MODELS_MING_TTS_PROVIDER_HPP_
