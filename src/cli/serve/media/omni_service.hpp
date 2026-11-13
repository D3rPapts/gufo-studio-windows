#ifndef GUFO_SERVER_OMNI_SERVICE_HPP_
#define GUFO_SERVER_OMNI_SERVICE_HPP_

#include <cstddef>
#include <functional>
#include <memory>
#include <string_view>
#include <string>
#include <utility>

#include "src/cli/serve/media/media_service.hpp"
#include "src/models/qwen3_omni/captioner.hpp"

namespace gufo::server {

// gufo-studio-windows (ported from 1.4.9 to the 2.0 media services).
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

/// `gufo serve omni` as a media service (2.0): the captioner has no upstream
/// capability (text_generation takes no audio parts), so it answers its own
/// OpenAI Chat Completions route like the 1.x fork did. Defined in
/// omni_api.cpp.
class OmniMediaService final : public MediaService {
public:
  explicit OmniMediaService(std::shared_ptr<OmniService> service)
      : service_(std::move(service)) {}

  [[nodiscard]] std::string_view capability() const override {
    return "audio_caption";
  }
  [[nodiscard]] std::string_view owned_by() const override {
    return "operator-supplied-qwen";
  }
  [[nodiscard]] bool ready() const override {
    return service_ != nullptr && service_->ready();
  }
  [[nodiscard]] std::string model_id() const override {
    return service_ != nullptr ? service_->model_id() : std::string();
  }
  void Describe(json::Value* entry, bool ready) const override;
  [[nodiscard]] bool HandlesPath(std::string_view path) const override;
  [[nodiscard]] std::string_view unavailable_message() const override {
    return "Qwen3-Omni captioner is not ready";
  }
  [[nodiscard]] std::string_view unavailable_code() const override {
    return "omni_service_unavailable";
  }
  [[nodiscard]] HttpResponse Handle(const HttpRequest& request) override;

private:
  std::shared_ptr<OmniService> service_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_OMNI_SERVICE_HPP_
