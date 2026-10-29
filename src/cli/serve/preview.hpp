#ifndef GUFO_SERVER_PREVIEW_HPP_
#define GUFO_SERVER_PREVIEW_HPP_

// Live previews of running image and video generations (like ComfyUI's
// KSampler preview). A model hands the sampler's denoised estimate x0 to a
// PreviewHook; a linear latent->RGB map (latent2rgb, factors fitted against
// each model's VAE) turns it into a small JPEG (images) or an animated WebP
// loop of all latent frames (videos) on a background thread.
//
// Previews cost nothing unless someone watches: a job renders previews only
// while a client streams /v1/previews/{id}/events, polled /v1/previews/{id}
// within the last kWatchWindow, or the request asked for them explicitly
// ("gufo": {"preview": {"always": true}}).
//
//   GET /v1/previews/{id}               newest preview (image/jpeg or
//                                       image/webp), 204 before the first one
//   GET /v1/previews/{id}?format=frame  one JPEG frame (video: the middle one)
//   GET /v1/previews/{id}?format=json   metadata + base64 image
//   GET /v1/previews/{id}?format=frame-json  the same with the JPEG frame
//   GET /v1/previews/{id}/events        SSE: one "preview" event per new
//                                       preview, "done" at the end
// Headers: X-Gufo-Preview-Step / -Steps / -Frames ("first-last" pixel
// frame) / -Seq / -State (pending, running, done, failed).

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/preview_hook.hpp"

namespace gufo::server {

struct HttpRequest;
struct HttpResponse;

enum class LatentFormat : std::uint8_t {
  kQwenImage21,   // 64 channels, 16x
  kFlux2Patched,  // MOSS-Image: 32 channels patched 2x2 to 128, 16x
  kWan22,         // 48 channels, 16x, 4x temporal
  kMinimaxH3,     // 24 channels, 16x
};

struct PreviewRgb {
  int width{0};
  int height{0};
  std::vector<std::uint8_t> rgb;  // packed RGB8
};

/// latent2rgb: one RGB image per latent frame at latent resolution
/// (rgb = x0 . W + b, then (v + 1) / 2 clamped to 0..255).
[[nodiscard]] std::vector<PreviewRgb> LatentToRgb(
    LatentFormat format, const models::LatentPreview& latent);
/// Bilinear resize so the longer side is `max_side` (up or down).
[[nodiscard]] PreviewRgb FitPreview(const PreviewRgb& image, int max_side);
[[nodiscard]] std::string EncodePreviewJpeg(const PreviewRgb& image,
                                            int quality = 85);
/// Looping animated WebP (lossy), `frame_ms` per frame.
[[nodiscard]] std::string EncodePreviewWebp(std::span<const PreviewRgb> frames,
                                            int frame_ms, int quality = 60);

struct PreviewOptions {
  /// Render even without a watcher (the request asked explicitly).
  bool always{false};
  /// Preview every N-th step (the last step is never previewed: the result
  /// follows anyway).
  int every{1};
  /// Longer side of the preview in pixels.
  int max_side{512};
  /// Video: pixel frames of the whole video (for the frame range header and
  /// the loop speed) and their rate; 1 for images.
  int pixel_frames{1};
  int fps{24};
};

/// Parses a request's "preview" extension object
/// ({"id", "always", "every", "max_side"}); throws std::invalid_argument.
/// Returns the id; fills `options` (defaults kept for absent fields).
std::string ParsePreviewRequest(const json::Value& object,
                                PreviewOptions* options);
[[nodiscard]] bool ValidPreviewId(std::string_view id) noexcept;

class PreviewHub {
public:
  static constexpr auto kWatchWindow = std::chrono::seconds(10);

  struct Snapshot {
    std::uint64_t seq{0};
    int step{0};
    int steps{0};
    int first_frame{0};
    int last_frame{0};
    std::string state;  // pending, running, done, failed
    std::string mime;   // empty before the first preview
    std::shared_ptr<const std::string> image;
    std::shared_ptr<const std::string> frame;  // JPEG; == image for images
  };

  PreviewHub();
  ~PreviewHub();
  PreviewHub(const PreviewHub&) = delete;
  PreviewHub& operator=(const PreviewHub&) = delete;

  /// The process-wide hub the image and video services publish into.
  static PreviewHub& Global();

  /// Starts a job's previews and returns the hook its model gets. A watcher
  /// that arrived before the job (same id) is kept.
  models::PreviewHook Begin(const std::string& id, LatentFormat format,
                            PreviewOptions options);
  void End(const std::string& id, bool ok);

  /// Latest state; `watch` counts as a watcher for kWatchWindow and creates
  /// a pending entry for a job that has not started yet.
  std::optional<Snapshot> Latest(const std::string& id, bool watch);
  /// An open event stream keeps the job watched until Unwatch.
  void Watch(const std::string& id);
  void Unwatch(const std::string& id);

  /// Waits until the job's preview sequence passes `seq` or it ended.
  std::optional<Snapshot> WaitNewer(const std::string& id, std::uint64_t seq,
                                    std::chrono::milliseconds timeout);

  /// Encodes queued previews; the hub's own thread runs it, tests may call
  /// it directly (returns false when nothing was queued).
  bool EncodePending();

  struct Stats {
    std::uint64_t published{0};
    std::uint64_t encoded{0};
    double encode_ms{0};
  };
  [[nodiscard]] Stats stats() const;

private:
  struct Job;
  struct State;
  std::unique_ptr<State> state_;
};

[[nodiscard]] bool IsPreviewPath(std::string_view path) noexcept;
HttpResponse HandlePreviewRequest(const HttpRequest& request, PreviewHub& hub);

}  // namespace gufo::server
#endif
