#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gufo/ext/options.hpp"
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

// gufo-studio-windows live previews (changes/0.2/live-previews.md).
/// The sampler's current estimate of the clean latent (x0), handed to a live
/// preview. FP32 in the sampler's normalized latent space; `data` is only
/// valid during the `publish` call.
struct LatentPreview {
  int step{0};   // completed denoising steps (1-based)
  int steps{0};  // total denoising steps
  int frames{1};
  int height{0};
  int width{0};
  int channels{0};
  /// false: [frames][height][width][channels] (image tokens);
  /// true:  [channels][frames][height][width] (video tensors).
  bool channel_major{false};
  std::span<const float> data;
};

/// Optional live preview of a running generation, set by the host. `wanted`
/// is asked once per step and must be cheap; only when it returns true does
/// the provider read the latent back and call `publish`, which copies what it
/// needs. Neither changes the result.
struct PreviewSink {
  std::function<bool(int step, int steps)> wanted;
  std::function<void(const LatentPreview&)> publish;

  [[nodiscard]] bool Wants(int step, int steps) const {
    return wanted && publish && wanted(step, steps);
  }
};

/// Caller's background intent, forwarded unchanged from the HTTP request.
/// kAuto leaves the background to the model and returns its pixels as
/// produced. kOpaque asks for an opaque image; the host composites any
/// remaining alpha over white after generation, so every provider supports it.
/// kTransparent asks for a transparent background. An explicit capability
/// declaration must support it and owns model-specific conditioning. Legacy
/// runners retain the host's historical transparent prompt framing.
enum class ImageBackground : std::uint8_t { kAuto, kOpaque, kTransparent };

struct ImageRequest {
  std::string prompt;
  std::vector<ImageBuffer> images;
  int width{1024};
  int height{1024};
  // Zero selects the checkpoint's default; runners reject unsupported counts.
  int steps{0};
  std::uint64_t seed{0};
  ImageBackground background{ImageBackground::kAuto};
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
  /// Live preview of this request (empty when nobody asked for one).
  PreviewSink preview;
};

/// Request features a runner accepts beyond text-to-image generation. The
/// host rejects unsupported requests with HTTP 400 before inference. An
/// explicit empty declaration accepts neither feature.
struct ImageCapabilities {
  /// Accepts reference images (`/v1/images/edits`).
  bool edits{false};
  /// Accepts `ImageBackground::kTransparent`.
  bool transparent_background{false};
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
  /// Explicit feature declaration, or nullopt for legacy runners. Legacy
  /// runners keep edits enabled and receive the historical RGBA prompt framing
  /// for transparent requests. Declaring capabilities opts into unchanged
  /// prompts and provider-owned conditioning.
  [[nodiscard]] virtual std::optional<ImageCapabilities> Capabilities() const {
    return std::nullopt;
  }
  /// Optional provider-specific request validation, called after the host's
  /// general checks and capability checks but before `Generate`. Throw
  /// `std::invalid_argument` to reject the request with HTTP 400.
  virtual void ValidateRequest(const ImageRequest& request) const {
    (void)request;
  }
  /// Provider state for host endpoints as JSON text, empty when the topic is
  /// unknown. "loras": the registered adapters (`GET /sdapi/v1/loras`).
  [[nodiscard]] virtual std::string Query(std::string_view topic) const {
    (void)topic;
    return {};
  }
};

/// Command whose manifest options reach an image provider. A declaration for
/// the bare `serve` command also applies (see `OptionApplies`).
inline constexpr std::string_view kImageServeCommand = "serve image";

/// Startup inputs for an image runner. `options` holds only explicitly
/// supplied manifest options of the selected provider, already parsed to their
/// declared types and checked by `ValidateSelectedOptions`; existing `path`
/// values name existing files or directories. Defaults for absent options
/// (for example a sidecar beside the model) remain the provider's choice.
struct ImageRunnerOptions {
  std::string model_root;
  ParsedOptions options;
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
  /// Preferred over create when set; receives validated startup options. A
  /// provider that declares manifest options for `serve image` must set it.
  ImageRunnerOptionsFactory create_with_options = nullptr;
  /// Latent space of `PreviewSink` latents, for the host's latent->RGB map:
  /// "qwen_image_21" (64 channels, 16x) or "flux2_patched" (MOSS-Image, 32
  /// channels patched 2x2). Null: the provider publishes no previews.
  const char* preview_latent = nullptr;
};

void RegisterImageProvider(ImageProvider provider);
const ImageProvider* FindImageProvider(std::string_view id);
const ImageProvider* ProbeImageProvider(std::string_view model_root);

}  // namespace gufo::ext
