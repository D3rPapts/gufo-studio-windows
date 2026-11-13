// gufo-studio-windows: RIFE / Real-ESRGAN post-steps of finished video jobs
// ("gufo.interpolate", "gufo.upscale") and post-processing-only jobs
// ("source_video"). Upstream's video capability has no post-processing
// (proposal gufo-org/gufo#1066), so the server applies these steps itself
// after any provider's runner. Kept in one place so they can move into a
// post-process capability later.
// Written with AI assistance (Claude Code).
#ifndef GUFO_SERVER_MEDIA_VIDEO_POSTPROCESS_HPP_
#define GUFO_SERVER_MEDIA_VIDEO_POSTPROCESS_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"
#include "src/ext/video.hpp"

namespace gufo::server {

// RIFE post-processing of a finished MP4 ("gufo.interpolate"): a whole-number
// multiplier (frames (n - 1) * m + 1 at m times the rate) or, when fps_num >
// 0, a target frame rate at the same speed (e.g. 60 from 24 or 16 fps).
struct VideoInterpolation {
  int multiplier{1};  // 1 = off, 2..8
  int fps_num{0};     // > 0 replaces the multiplier
  int fps_den{1};
  [[nodiscard]] bool enabled() const { return fps_num > 0 || multiplier > 1; }
};

// Real-ESRGAN post-processing of a finished MP4 ("gufo.upscale"), after any
// interpolation: every frame upscaled into width x height (both even), the
// picture fitted by `fit` ("pad": keep the aspect ratio with black bars,
// "crop": keep it and centre-crop, "stretch").
struct VideoUpscale {
  int width{0};  // 0 = off
  int height{0};
  std::string fit{"pad"};
  // A name registered with --upscale-model; empty: the first registered.
  std::string model;
  [[nodiscard]] bool enabled() const { return width > 0 || height > 0; }
};

struct VideoPostProcess {
  VideoInterpolation interpolation;
  VideoUpscale upscale;
  [[nodiscard]] bool enabled() const {
    return interpolation.enabled() || upscale.enabled();
  }
};

/// The post-steps a video service can run (`gufo serve video --rife-model /
/// --upscale-model`). Each reads `input` and writes `output`; progress phases
/// "interpolate" and "upscale".
struct VideoPostProcessors {
  std::function<bool(const std::filesystem::path& input,
                     const std::filesystem::path& output,
                     const VideoInterpolation&, const ext::CancellationCheck&,
                     ext::VideoProgress, void*, std::string*)>
      interpolator;
  std::function<bool(const std::filesystem::path& input,
                     const std::filesystem::path& output, const VideoUpscale&,
                     const ext::CancellationCheck&, ext::VideoProgress, void*,
                     std::string*)>
      upscaler;
  /// Accepted `gufo.upscale.model` names; the first is the default.
  std::vector<std::string> upscale_models;
};

struct VideoPostProcessError {
  std::string code;
  std::string message;
};

/// Parses "gufo.interpolate" / "gufo.upscale" of a /v1/videos body and
/// removes them from `body`, so providers never see them. False with an
/// OpenAI-style error code ("invalid_interpolate", "invalid_upscale").
[[nodiscard]] bool ExtractVideoPostProcess(json::Value* body,
                                           VideoPostProcess* post,
                                           VideoPostProcessError* error);

/// True for a post-processing-only request (a "source_video" member or
/// multipart file).
[[nodiscard]] bool IsSourceVideoRequest(
    const json::Value& body, const std::vector<ext::VideoAttachment>& files);

/// Parses a post-processing-only request: {"model", "source_video" (MP4 data
/// URL or base64, or a multipart file), "gufo": {"interpolate", "upscale"}}.
/// Call after ExtractVideoPostProcess.
[[nodiscard]] bool ParseSourceVideoRequest(
    const json::Value& body, const std::vector<ext::VideoAttachment>& files,
    std::vector<std::uint8_t>* source, VideoPostProcessError* error);

/// Checks the values against what the service can run. False with a message
/// for VideoJobService::Create.
[[nodiscard]] bool ValidateVideoPostProcess(const VideoPostProcess& post,
                                            const VideoPostProcessors& tools,
                                            std::string_view output_format,
                                            std::string* error);

/// Runs interpolation, then upscaling, on `directory/output.mp4` in place
/// (staged through `generated.mp4`). Logs each step with the job id.
[[nodiscard]] bool RunVideoPostProcess(
    const VideoPostProcess& post, const VideoPostProcessors& tools,
    const std::filesystem::path& directory, const std::string& job_id,
    const ext::CancellationCheck& cancelled, ext::VideoProgress progress,
    void* progress_opaque);

/// Job progress (percent) of a post-step phase, or nullopt for other phases.
[[nodiscard]] std::optional<int> VideoPostProcessPercent(std::string_view phase,
                                                         int completed,
                                                         int total);

}  // namespace gufo::server

#endif  // GUFO_SERVER_MEDIA_VIDEO_POSTPROCESS_HPP_
