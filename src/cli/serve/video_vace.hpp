#ifndef GUFO_SERVER_VIDEO_VACE_HPP_
#define GUFO_SERVER_VIDEO_VACE_HPP_

// Written with AI assistance (Claude Code).

#include <string>

#include "src/models/minimax_h3/generation.hpp"

namespace gufo::server {

// Video job runner for Wan 2.1 VACE 14B ("wan2.1-vace-14b"): T2V 14B + VACE
// module, video editing with a control video, a mask and reference images
// (ComfyUI WanVaceToVideo + TrimVideoLatent). The model directory is a bundle
// whose gufo-vace.json names the parts relative to it (or absolute):
//   {"_class_name": "GufoWan21Vace",
//    "vae": Diffusers Wan 2.1 root with vae/ (e.g. Wan2.1-I2V-14B-480P),
//    "t5": Diffusers root with tokenizer/ and text_encoder/ (umT5-XXL),
//    "dit": Kijai T2V 14B FP8 (scaled), "vace": Kijai VACE module,
//    "lora": lightx2v T2V LoRA (optional), "lora_strength",
//    "shift" (5), "sampler" ("lcm"), "steps" (4, the API default)}
// request: parameters (size, frames at 16 fps, steps), negative_prompt and
// guidance (1 = no CFG), guide_video_path = control video (MP4),
// mask_path (MP4, or a binary PPM still used for every frame), keyframes =
// reference images (RGB8), vace_json {"strength", "mask_channel" ("red",
// "green", "blue", "luma"; default "red"), "invert_mask", "sampler",
// "shift", "lora" (false: without the bundle's LoRA)}. Writes an H.264 MP4
// at 16 fps.
[[nodiscard]] bool RunWan21VaceJob(
    const minimax_h3::GenerationRequest& request,
    const minimax_h3::CancellationToken* cancellation,
    minimax_h3::GenerationProgress progress, void* progress_opaque,
    minimax_h3::GenerationTelemetry* telemetry, std::string* error);

}  // namespace gufo::server

#endif  // GUFO_SERVER_VIDEO_VACE_HPP_
