#ifndef GUFO_MODELS_QWEN_HIP_DFLASH_PROMPT_LOOKUP_27B_HPP_
#define GUFO_MODELS_QWEN_HIP_DFLASH_PROMPT_LOOKUP_27B_HPP_

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace gufo::hip::qwen27 {

/// GUFO_QWEN27_LOOKUP=N caps DFlash2's own drafts at N tokens. Prompt-lookup
/// copies (always on, see QwenDFlashGpuDraftBackend::TryLookup) still run up
/// to the server's --draft-tokens. 0 or unset leaves DFlash2 uncapped.
[[nodiscard]] inline std::uint32_t LookupDFlashCap() {
  static const std::uint32_t cap = [] {
    const char* value = std::getenv("GUFO_QWEN27_LOOKUP");
    const int parsed = value != nullptr ? std::atoi(value) : 0;
    return parsed > 0 ? static_cast<std::uint32_t>(parsed) : 0U;
  }();
  return cap;
}

/// GUFO_QWEN27_LOOKUP_MIN=N: the match length a copy needs when the strict
/// gate applies (sampled requests, and greedy ones while DFlash2 drafts
/// well). Unset keeps `fallback`, upstream's kSampledMinMatch.
[[nodiscard]] inline std::size_t LookupStrictMinMatch(std::size_t fallback) {
  static const std::size_t minimum = [fallback] {
    const char* value = std::getenv("GUFO_QWEN27_LOOKUP_MIN");
    const int parsed = value != nullptr ? std::atoi(value) : 0;
    return parsed > 0 ? static_cast<std::size_t>(parsed) : fallback;
  }();
  return minimum;
}

/// GUFO_QWEN27_LOOKUP_GENERATED=1: sampled requests also copy from the text
/// they generated (upstream indexes only their starting context). A JSON
/// answer that restates its own reasoning copies from that reasoning.
[[nodiscard]] inline bool LookupIndexesGenerated() {
  static const bool enabled = [] {
    const char* value = std::getenv("GUFO_QWEN27_LOOKUP_GENERATED");
    return value != nullptr && std::atoi(value) != 0;
  }();
  return enabled;
}

}  // namespace gufo::hip::qwen27

#endif  // GUFO_MODELS_QWEN_HIP_DFLASH_PROMPT_LOOKUP_27B_HPP_
