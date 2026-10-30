#ifndef GUFO_SERVER_VIDEO_UPSCALE_HPP_
#define GUFO_SERVER_VIDEO_UPSCALE_HPP_

#include <filesystem>
#include <string>

#include "src/cli/serve/video_jobs.hpp"
#include "src/models/minimax_h3/generation.hpp"

namespace gufo::server {

// VideoJobServiceOptions::upscaler backed by Real-ESRGAN SRVGGNetCompact
// (FP16): loads `weights` (converted realesr-animevideov3 / realesr-general-
// x4v3 safetensors) per job and upscales every frame of `input` into
// spec.width x spec.height (spec.fit) at the same frame rate, copying the
// audio into `output`. Reports progress phase "upscale".
[[nodiscard]] bool RunRealEsrganUpscale(
    const std::filesystem::path& weights, const std::filesystem::path& input,
    const std::filesystem::path& output, const VideoUpscale& spec,
    const minimax_h3::CancellationToken* cancellation,
    minimax_h3::GenerationProgress progress, void* progress_opaque,
    std::string* error);

}  // namespace gufo::server

#endif  // GUFO_SERVER_VIDEO_UPSCALE_HPP_
