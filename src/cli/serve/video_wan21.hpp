#ifndef GUFO_SERVER_VIDEO_WAN21_HPP_
#define GUFO_SERVER_VIDEO_WAN21_HPP_

#include <string>

#include "src/core/json.hpp"
#include "src/models/minimax_h3/generation.hpp"
#include "src/models/wan22/camera_path.hpp"

namespace gufo::server {

// Video job runner for Wan 2.1 I2V 14B with the Uni3C camera ControlNet
// ("wan2.1-uni3c"). The model directory is a camera bundle whose
// gufo-camera.json names the parts relative to it (or absolute):
//   {"_class_name": "GufoWan21Uni3CCamera",
//    "wan21": Diffusers Wan2.1-I2V-14B-480P (vae/, image_encoder/),
//    "t5": Diffusers root with tokenizer/ and text_encoder/ (umT5-XXL),
//    "dit": Kijai FP8 checkpoint, "lora": lightx2v LoRA, "lora_strength",
//    "uni3c": Uni3C ControlNet, "moge": MoGe-2 safetensors,
//    "uni3c_strength", "shift"}
// request.parameters: size, frames (16 fps), steps. request.camera_json (a
// camera path rendered from the first frame with MoGe-2) or
// request.guide_video_path (an uploaded guide) turn Uni3C on; neither is
// plain image-to-video. Writes an H.264 MP4 at 16 fps.
[[nodiscard]] bool RunWan21Job(
    const minimax_h3::GenerationRequest& request,
    const minimax_h3::CancellationToken* cancellation,
    minimax_h3::GenerationProgress progress, void* progress_opaque,
    minimax_h3::GenerationTelemetry* telemetry, std::string* error);

// "gufo.camera" without "render_video": {"type", "amount", "speed", "ease"}
// or {"poses": [{"frame", "position": [x, y, z], "yaw", "pitch", "roll",
// "zoom"}, ...]} (positions in scene depths, angles in degrees).
[[nodiscard]] bool ParseCameraSpec(const json::Value& camera,
                                   models::wan22::CameraSpec* spec,
                                   std::string* error);

}  // namespace gufo::server

#endif  // GUFO_SERVER_VIDEO_WAN21_HPP_
