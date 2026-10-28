#ifndef GUFO_MODELS_MINIMAX_H3_PROMPT_VISION_HPP_
#define GUFO_MODELS_MINIMAX_H3_PROMPT_VISION_HPP_

#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "src/models/minimax_h3/prompt_encoder.hpp"
#include "src/models/minimax_h3/video_vae_encoder.hpp"

namespace gufo::minimax_h3 {

/// Runs the text encoder's Qwen3-VL vision tower (an mmproj GGUF converted
/// from FL2VA/text_encoder, with its deepstack mergers) over each keyframe
/// canvas. Returns one block per image with BF16 embeddings and deepstack
/// features; `begin` is left for the caller to place in the prompt.
/// Canvas sides must be multiples of 32 (the 16-pixel patch, merged 2x2).
[[nodiscard]] bool EncodeKeyframeVision(const std::filesystem::path& projector,
                                        std::span<const KeyframeImage> images,
                                        const CancellationToken* cancellation,
                                        std::vector<PromptVisionBlock>* blocks,
                                        std::string* error = nullptr);

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_PROMPT_VISION_HPP_
