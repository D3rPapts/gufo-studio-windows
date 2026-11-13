// gufo-studio-windows: HTTP services for media models without an upstream
// capability (ACE-Step music, MMAudio foley, Real-ESRGAN image upscaling).
// HttpServer asks them before its own routes (HttpServerOptions::media) and
// lists them in /ready and /v1/models. Each one could become an extension
// once upstream has a matching capability (proposal gufo-org/gufo#1066).
// Written with AI assistance (Claude Code).
#ifndef GUFO_SERVER_MEDIA_MEDIA_SERVICE_HPP_
#define GUFO_SERVER_MEDIA_MEDIA_SERVICE_HPP_

#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/core/json.hpp"

namespace gufo::server {

class MediaService {
public:
  virtual ~MediaService() = default;

  /// "music", "foley" or "upscale" (/v1/models `capability`).
  [[nodiscard]] virtual std::string_view capability() const = 0;
  /// /v1/models `owned_by`.
  [[nodiscard]] virtual std::string_view owned_by() const = 0;
  [[nodiscard]] virtual bool ready() const = 0;
  /// The default model name (/ready `model`).
  [[nodiscard]] virtual std::string model_id() const = 0;
  /// Every served model name, the default first.
  [[nodiscard]] virtual std::vector<std::string> model_ids() const {
    return {model_id()};
  }
  /// Extra fields of each /v1/models entry and of /ready (e.g. device).
  virtual void Describe(json::Value* /*entry*/, bool /*ready*/) const {}

  [[nodiscard]] virtual bool HandlesPath(std::string_view path) const = 0;
  /// 503 reply while not ready: message and error code.
  [[nodiscard]] virtual std::string_view unavailable_message() const = 0;
  [[nodiscard]] virtual std::string_view unavailable_code() const = 0;
  [[nodiscard]] virtual HttpResponse Handle(const HttpRequest& request) = 0;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_MEDIA_MEDIA_SERVICE_HPP_
