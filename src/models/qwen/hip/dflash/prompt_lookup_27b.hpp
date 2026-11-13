#ifndef GUFO_MODELS_QWEN_HIP_DFLASH_PROMPT_LOOKUP_27B_HPP_
#define GUFO_MODELS_QWEN_HIP_DFLASH_PROMPT_LOOKUP_27B_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

namespace gufo::hip::qwen27 {

// Opt-in prompt-lookup knobs for the Qwen3.8-27B DFlash2 backend. Every knob
// only changes which tokens are proposed; the verifier still accepts or
// corrects each of them, so greedy text is identical and sampled output
// keeps the target distribution. Unset knobs keep upstream's behavior.

namespace detail {
[[nodiscard]] inline int EnvInt(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr ? std::atoi(value) : 0;
}
}  // namespace detail

/// GUFO_QWEN27_LOOKUP=N caps DFlash2's own drafts at N tokens. Prompt-lookup
/// copies still run up to the server's --draft-tokens. 0 or unset leaves
/// DFlash2 uncapped.
[[nodiscard]] inline std::uint32_t LookupDFlashCap() {
  static const std::uint32_t cap = [] {
    const int parsed = detail::EnvInt("GUFO_QWEN27_LOOKUP");
    return parsed > 0 ? static_cast<std::uint32_t>(parsed) : 0U;
  }();
  return cap;
}

/// The strict gate's match length (sampled requests, and greedy ones while
/// DFlash2 drafts well), per phase of a thinking answer:
///   GUFO_QWEN27_LOOKUP_MIN=N            both phases (unset: `fallback`,
///                                       upstream's kSampledMinMatch)
///   GUFO_QWEN27_LOOKUP_MIN_REASONING=N  inside <think>...</think>
///   GUFO_QWEN27_LOOKUP_MIN_ANSWER=N     after </think> (and without thinking)
/// Reasoning paraphrases the prompt, so short matches there mostly displace
/// good DFlash2 drafts; an answer (JSON) restating its reasoning copies well
/// from short matches.
struct LookupGate {
  std::size_t reasoning;
  std::size_t answer;
  [[nodiscard]] bool PhaseAware() const { return reasoning != answer; }
};

[[nodiscard]] inline const LookupGate& LookupStrictGate(std::size_t fallback) {
  static const LookupGate gate = [fallback] {
    const auto read = [](const char* name, std::size_t otherwise) {
      const int parsed = detail::EnvInt(name);
      return parsed > 0 ? static_cast<std::size_t>(parsed) : otherwise;
    };
    const std::size_t base = read("GUFO_QWEN27_LOOKUP_MIN", fallback);
    return LookupGate{read("GUFO_QWEN27_LOOKUP_MIN_REASONING", base),
                      read("GUFO_QWEN27_LOOKUP_MIN_ANSWER", base)};
  }();
  return gate;
}

/// GUFO_QWEN27_LOOKUP_GENERATED=1: sampled requests also copy from the text
/// they generated (upstream indexes only their starting context). A JSON
/// answer that restates its own reasoning copies from that reasoning.
[[nodiscard]] inline bool LookupIndexesGenerated() {
  static const bool enabled = detail::EnvInt("GUFO_QWEN27_LOOKUP_GENERATED") != 0;
  return enabled;
}

/// Whether the context ends inside a reasoning block: the latest <think> or
/// </think> token is <think>. Depends only on the tokens, so restored
/// sessions decide like the original. A prompt without thinking ends in
/// "<think>\n\n</think>\n\n" or has neither token, which both count as answer.
template<typename TokenId>
[[nodiscard]] bool InReasoning(std::span<const TokenId> tokens,
                               TokenId think_start, TokenId think_end) {
  const auto last = std::find_if(tokens.rbegin(), tokens.rend(), [&](auto t) {
    return t == think_start || t == think_end;
  });
  return last != tokens.rend() && *last == think_start;
}

}  // namespace gufo::hip::qwen27

#endif  // GUFO_MODELS_QWEN_HIP_DFLASH_PROMPT_LOOKUP_27B_HPP_
