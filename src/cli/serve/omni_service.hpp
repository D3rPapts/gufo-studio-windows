#ifndef GUFO_SERVER_OMNI_SERVICE_HPP_
#define GUFO_SERVER_OMNI_SERVICE_HPP_

#include <cstddef>
#include <functional>
#include <string>
#include <utility>

#include "src/models/qwen3_omni/captioner.hpp"

namespace gufo::server {

/// Qwen3-Omni-30B-A3B-Captioner behind the HTTP server. The HIP captioner is
/// injected as `runner` by `gufo serve omni`, so the HTTP library itself does
/// not link the GPU runtime.
struct OmniServiceOptions {
  using Runner = std::function<bool(
      const models::qwen3_omni::CaptionRequest&,
      const models::qwen3_omni::CancellationCheck&,
      models::qwen3_omni::CaptionResult*, std::string*)>;

  std::string model_id{"qwen3-omni-captioner"};
  std::size_t context_tokens{0};
  /// Upper bound for a request's max_tokens; also the default.
  std::size_t max_tokens{1024};
  Runner runner;
};

class OmniService {
public:
  explicit OmniService(OmniServiceOptions options)
      : options_(std::move(options)) {}

  [[nodiscard]] bool ready() const noexcept {
    return static_cast<bool>(options_.runner);
  }
  [[nodiscard]] const std::string& model_id() const noexcept {
    return options_.model_id;
  }
  [[nodiscard]] std::size_t context_tokens() const noexcept {
    return options_.context_tokens;
  }
  [[nodiscard]] std::size_t max_tokens() const noexcept {
    return options_.max_tokens;
  }

  bool Caption(const models::qwen3_omni::CaptionRequest& request,
               const models::qwen3_omni::CancellationCheck& is_cancelled,
               models::qwen3_omni::CaptionResult* result, std::string* error) {
    return options_.runner(request, is_cancelled, result, error);
  }

private:
  OmniServiceOptions options_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_OMNI_SERVICE_HPP_
