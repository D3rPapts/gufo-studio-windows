#ifndef GUFO_MODELS_WAN22_POINT_RENDER_HPP_
#define GUFO_MODELS_WAN22_POINT_RENDER_HPP_

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/models/wan22/camera_path.hpp"
#include "src/models/wan22/moge.hpp"

namespace gufo::models::wan22 {

/// Renders the start image as a coloured point cloud (one point per pixel
/// from MoGe's metric depth) along `poses` into a Uni3C guide video: z-buffer
/// splatting on the GPU, each point a square of its projected pixel
/// footprint (at least one pixel, at most 8), black where nothing lands
/// (like ComfyUI's point-cloud renders). Pixels outside MoGe's mask (sky,
/// infinity) are placed at 3x the farthest valid depth so they move like
/// distant background. Writes RGB8 [poses, height, width, 3].
bool RenderGuide(const MogeResult& geometry, std::span<const std::uint8_t> rgb,
                 const std::vector<CameraPose>& poses,
                 std::vector<std::uint8_t>* frames, std::string* error);

/// Median depth of the valid pixels (0 if none).
double SceneDepth(const MogeResult& geometry);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_POINT_RENDER_HPP_
