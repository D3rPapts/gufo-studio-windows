#include "src/cli/serve/video_wan22.hpp"

#include <chrono>
#include <cstdint>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/wan22/pipeline.hpp"
#endif

namespace gufo::server {

bool RunWan22Job(const minimax_h3::GenerationRequest& request,
                 const minimax_h3::CancellationToken* cancellation,
                 minimax_h3::GenerationProgress progress, void* progress_opaque,
                 minimax_h3::GenerationTelemetry* telemetry,
                 std::string* error) {
#if defined(ENGINE_ENABLE_HIP)
  namespace wan22 = models::wan22;
  using Clock = std::chrono::steady_clock;
  wan22::VideoRequest video_request;
  video_request.model_root = request.model_root;
  video_request.prompt = request.prompt;
  video_request.width = request.parameters.output_width;
  video_request.height = request.parameters.output_height;
  video_request.frames = request.parameters.frames;
  video_request.steps = request.parameters.evaluations;
  video_request.seed = request.seed;
  video_request.negative_prompt = request.negative_prompt;
  video_request.guidance = request.guidance;
  std::vector<std::uint8_t> first_frame;
  if (!request.first_frame.empty()) {
    // Scale to cover the video size, crop the centre.
    first_frame = wan22::FitImage(
        request.first_frame, request.first_frame_width,
        request.first_frame_height, video_request.width, video_request.height);
    if (first_frame.empty()) {
      if (error != nullptr)
        *error = "invalid first frame";
      return false;
    }
    video_request.first_frame = first_frame;
  }
  wan22::VideoCallbacks callbacks;
  callbacks.progress = [&](std::string_view phase, int done, int total) {
    if (progress != nullptr)
      progress(phase, done, total, progress_opaque);
  };
  callbacks.keep_going = [cancellation] {
    return cancellation == nullptr || !cancellation->IsCancelled();
  };
  wan22::DecodedVideo video;
  wan22::VideoTelemetry stages;
  if (!wan22::GenerateVideo(video_request, callbacks, nullptr, &video, false,
                            &stages, error))
    return false;
  if (cancellation != nullptr && cancellation->IsCancelled()) {
    if (error != nullptr)
      *error = "cancelled";
    return false;
  }
  callbacks.progress("media", 0, 1);
  const auto media_start = Clock::now();
  if (!wan22::WriteMp4(request.output_path, video, 24, error))
    return false;
  callbacks.progress("media", 1, 1);
  if (telemetry != nullptr) {
    telemetry->prompt_ms = stages.text_ms;
    telemetry->prompt_tokens = static_cast<std::size_t>(stages.prompt_tokens);
    telemetry->denoiser_ms = stages.transformer_load_ms + stages.denoise_ms;
    telemetry->video_vae_ms = stages.vae_load_ms + stages.vae_ms;
    telemetry->media_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - media_start)
            .count();
    telemetry->total_ms = stages.total_ms + telemetry->media_ms;
  }
  return true;
#else
  (void)request;
  (void)cancellation;
  (void)progress;
  (void)progress_opaque;
  (void)telemetry;
  if (error != nullptr)
    *error = "Wan 2.2 video generation requires a ROCm HIP-enabled build";
  return false;
#endif
}

}  // namespace gufo::server
