#ifndef GUFO_MODELS_QWEN_HIP_DFLASH_PROMPT_LOOKUP_27B_HPP_
#define GUFO_MODELS_QWEN_HIP_DFLASH_PROMPT_LOOKUP_27B_HPP_

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

}  // namespace gufo::hip::qwen27

#endif  // GUFO_MODELS_QWEN_HIP_DFLASH_PROMPT_LOOKUP_27B_HPP_
