// IndexTTS-2.5 speech_synthesis provider entry points (tests use the factory
// directly). Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_INDEXTTS_PROVIDER_HPP_
#define GUFO_MODELS_INDEXTTS_PROVIDER_HPP_

#include <gufo/ext/tts.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

namespace gufo::models::indextts {

inline constexpr std::string_view kIndexTtsModelId = "indextts-2.5-german";

[[nodiscard]] std::unique_ptr<ext::TtsRunner> CreateIndexTtsRunner(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error);

}  // namespace gufo::models::indextts

#endif  // GUFO_MODELS_INDEXTTS_PROVIDER_HPP_
