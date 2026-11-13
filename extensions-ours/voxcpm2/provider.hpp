// VoxCPM2 speech_synthesis provider entry points (tests use the factory
// directly). Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_VOXCPM2_PROVIDER_HPP_
#define GUFO_MODELS_VOXCPM2_PROVIDER_HPP_

#include <gufo/ext/tts.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

namespace gufo::models::voxcpm2 {

inline constexpr std::string_view kVoxCpm2ModelId = "voxcpm2";

[[nodiscard]] std::unique_ptr<ext::TtsRunner> CreateVoxCpm2TtsRunner(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error);

}  // namespace gufo::models::voxcpm2

#endif  // GUFO_MODELS_VOXCPM2_PROVIDER_HPP_
