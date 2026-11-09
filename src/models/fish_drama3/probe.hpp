#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "src/ext/tts.hpp"

namespace gufo::models::fish_drama3 {

[[nodiscard]] bool ProbeFishDrama3(std::string_view model_root);
[[nodiscard]] bool ResolveFishDrama3Artifacts(std::string_view model_root,
                                              std::string* error = nullptr);

#if defined(ENGINE_ENABLE_HIP)
[[nodiscard]] std::unique_ptr<ext::TtsRunner> CreateFishDrama3Runner(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error = nullptr);
#endif

}  // namespace gufo::models::fish_drama3
