#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/image.hpp"

namespace gufo::ext {

struct ImageBuffer {
  int width{0};
  int height{0};
  std::vector<std::uint8_t> rgba;
};

struct ImageMetrics {
  double prompt_ms{0};
  double denoise_ms{0};
  double vae_ms{0};
  double total_ms{0};
  std::size_t resident_weight_bytes{0};
  int steps{0};  // Denoising steps executed after resolving the default.
};

struct ImageRequest {
  std::string prompt;
  std::vector<ImageBuffer> images;
  int width{1024};
  int height{1024};
  // Zero selects the checkpoint's default; runners reject unsupported counts.
  int steps{0};
  std::uint64_t seed{0};
  // gufo-studio-windows request extension (changes/0.2/image-request-extension):
  /// JSON object text with the request members the host does not interpret,
  /// as received: "gufo" (the extension object, also filled from multipart
  /// form fields "gufo.<name>") and "loras". Empty when neither was sent.
  std::string extension_json;
  /// Image parts outside `images`, in request order: multipart "init_image",
  /// "mask", "control_image" and "region_image[]". Inline images inside
  /// `extension_json` stay there (base64 or data URLs).
  std::vector<std::pair<std::string, ImageBuffer>> named_images;
  /// The caller asked for size "auto": `width`/`height` hold the host's guess
  /// (the last reference's aspect at ~1 MP, else 1024x1024) and a provider may
  /// derive the canvas from its own inputs instead (start or control image).
  bool size_auto{false};
};

/// Thrown by `ImageRunner::Generate` when the request does not fit into the
/// memory left now (it may fit later); the host answers HTTP 503.
class ImageResourceError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

struct ImageResult {
  ImageBuffer image;
  ImageMetrics metrics;
};

using CancellationCheck = std::function<bool()>;

/// Abstract capability runner for image generation models.
class ImageRunner {
public:
  virtual ~ImageRunner() = default;

  [[nodiscard]] virtual ImageResult Generate(
      const ImageRequest& request,
      const CancellationCheck& is_cancelled = {}) = 0;
  /// Provider state for host endpoints as JSON text, empty when the topic is
  /// unknown. "loras": the registered adapters (`GET /sdapi/v1/loras`).
  [[nodiscard]] virtual std::string Query(std::string_view topic) const {
    (void)topic;
    return {};
  }
};

/// Startup inputs for an image runner: the model root and the provider's own
/// `serve image` options in command-line order, names without dashes (for
/// example {"lora", "style=D:/x.safetensors"}). Options may repeat.
struct ImageRunnerOptions {
  std::string model_root;
  std::vector<std::pair<std::string, std::string>> options;
};

using ImageProbe = bool (*)(std::string_view model_root);
using ImageArtifactResolver = bool (*)(std::string_view model_root,
                                       std::string* error);
using ImageRunnerFactory = std::unique_ptr<ImageRunner> (*)(
    const std::string& model_root, std::string* error);
using ImageRunnerOptionsFactory = std::unique_ptr<ImageRunner> (*)(
    const ImageRunnerOptions& options, std::string* error);

struct ImageProvider {
  std::string id;
  ImageProbe probe = nullptr;
  ImageArtifactResolver resolve = nullptr;
  ImageRunnerFactory create = nullptr;
  /// Preferred over `create` when set; the host passes provider options only
  /// to a provider that sets it and rejects them otherwise.
  ImageRunnerOptionsFactory create_with_options = nullptr;
};

void RegisterImageProvider(ImageProvider provider);
const ImageProvider* FindImageProvider(std::string_view id);
const ImageProvider* ProbeImageProvider(std::string_view model_root);

}  // namespace gufo::ext
