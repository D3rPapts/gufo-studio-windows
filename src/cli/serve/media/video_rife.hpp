#ifndef GUFO_SERVER_VIDEO_RIFE_HPP_
#define GUFO_SERVER_VIDEO_RIFE_HPP_

#include <filesystem>
#include <string>

#include "src/cli/serve/video_jobs.hpp"
#include "src/models/minimax_h3/generation.hpp"

namespace gufo::server {

// VideoJobServiceOptions::interpolator backed by RIFE 4.7 (FP16 IFNet):
// loads `weights` (converted rife47/rife49 safetensors) per job, multiplies
// the frame rate of `input` by the multiplier (or resamples it to the target
// rate) at the same speed and copies its audio into `output`. Reports
// progress phase "interpolate".
[[nodiscard]] bool RunRifeInterpolation(
    const std::filesystem::path& weights, const std::filesystem::path& input,
    const std::filesystem::path& output, const VideoInterpolation& spec,
    const minimax_h3::CancellationToken* cancellation,
    minimax_h3::GenerationProgress progress, void* progress_opaque,
    std::string* error);

}  // namespace gufo::server

#endif  // GUFO_SERVER_VIDEO_RIFE_HPP_
