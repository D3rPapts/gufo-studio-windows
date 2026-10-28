#ifndef GUFO_SERVER_VIDEO_WAN22_HPP_
#define GUFO_SERVER_VIDEO_WAN22_HPP_

#include <string>

#include "src/models/minimax_h3/generation.hpp"

namespace gufo::server {

// Video job runner for Wan 2.2 TI2V-5B: request.parameters carries the size
// (output_width x output_height), frames and steps (evaluations); writes an
// H.264 MP4 at 24 fps to request.output_path. Loads every stage per job.
[[nodiscard]] bool RunWan22Job(
    const minimax_h3::GenerationRequest& request,
    const minimax_h3::CancellationToken* cancellation,
    minimax_h3::GenerationProgress progress, void* progress_opaque,
    minimax_h3::GenerationTelemetry* telemetry, std::string* error);

}  // namespace gufo::server

#endif  // GUFO_SERVER_VIDEO_WAN22_HPP_
