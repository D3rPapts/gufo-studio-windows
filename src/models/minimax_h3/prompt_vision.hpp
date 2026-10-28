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

/// Everything the text encoder needs: the vision tower runs over `images`
/// when `projector` is set, and `blocks` gives each image's place in
/// `tokens` (begin and grid; embeddings are filled in by the child).
struct PromptIsolationInput {
  std::filesystem::path model_root;
  std::filesystem::path manifest;
  std::filesystem::path projector;
  std::span<const KeyframeImage> images;
  std::span<const std::uint32_t> tokens;
  std::span<const PromptVisionBlock> blocks;
};

/// True on Windows HIP builds unless GUFO_H3_PROMPT_INPROCESS=1.
[[nodiscard]] bool PromptIsolationEnabled();

/// Runs the vision tower and the layer-50 text encoder in a short-lived child
/// process ("gufo video --h3-prompt-child"). Measured on gfx1151 Windows: the
/// in-process text encoder leaves about 4 GiB of device memory in use until
/// the process exits, which the DiT needs; the child returns all of it.
[[nodiscard]] bool EncodePromptIsolated(const PromptIsolationInput& input,
                                        const CancellationToken* cancellation,
                                        PromptEmbedding* embedding,
                                        PromptEncoderTelemetry* telemetry,
                                        double* vision_ms,
                                        std::string* error = nullptr);

/// Child side of EncodePromptIsolated. Returns the process exit code.
[[nodiscard]] int RunPromptChild(const std::filesystem::path& request,
                                 const std::filesystem::path& response);

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_PROMPT_VISION_HPP_
