#ifndef GUFO_MODELS_QWEN3_OMNI_CAPTIONER_HPP_
#define GUFO_MODELS_QWEN3_OMNI_CAPTIONER_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace gufo::models::qwen3_omni {

/// One caption: a WAV file in, text out. Sampling defaults to greedy.
struct CaptionRequest {
  std::span<const std::byte> wav;
  std::size_t max_tokens{1024};
  float temperature{0.0F};
  std::size_t top_k{20};
  float top_p{0.95F};
  std::uint64_t seed{1234};
  /// Called with each decoded text piece; returning false stops decoding.
  std::function<bool(std::string_view)> on_text{};
  /// Non-empty: write PREFIX.audio.f32, PREFIX.logits.f32, PREFIX.tokens.txt.
  std::string dump_prefix{};
};

struct CaptionResult {
  std::string text;
  std::size_t prompt_tokens{0};
  std::size_t audio_tokens{0};
  std::size_t completion_tokens{0};
  double audio_seconds{0.0};
  /// "stop" (end of turn) or "length" (max_tokens reached).
  std::string finish_reason;
  double encoder_ms{0.0};
  double prefill_ms{0.0};
  double decode_ms{0.0};
  /// Set on failure when the request itself was unusable (bad WAV, audio
  /// longer than the context), as opposed to a runtime fault.
  bool invalid_request{false};
};

using CancellationCheck = std::function<bool()>;

/// Qwen3-Omni-30B-A3B-Captioner loaded once and reused across requests.
/// Calls are serialized: one caption runs at a time.
class Captioner {
public:
  struct Options {
    std::string model;
    std::string mmproj;
    /// "gpu" (HIP runtime) or "cpu" (F32 reference).
    std::string backend{"gpu"};
    std::size_t context{8192};
    unsigned threads{0};
  };

  static std::unique_ptr<Captioner> Create(const Options& options,
                                           std::string* error);
  ~Captioner();

  Captioner(const Captioner&) = delete;
  Captioner& operator=(const Captioner&) = delete;

  bool Caption(const CaptionRequest& request,
               const CancellationCheck& is_cancelled, CaptionResult* result,
               std::string* error);

  [[nodiscard]] std::size_t max_context() const;

private:
  struct Impl;
  explicit Captioner(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen3_omni

#endif  // GUFO_MODELS_QWEN3_OMNI_CAPTIONER_HPP_
