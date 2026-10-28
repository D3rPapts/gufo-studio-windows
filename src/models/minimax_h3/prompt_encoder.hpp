#ifndef GUFO_MODELS_MINIMAX_H3_PROMPT_ENCODER_HPP_
#define GUFO_MODELS_MINIMAX_H3_PROMPT_ENCODER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"

namespace gufo::minimax_h3 {

inline constexpr std::size_t kPromptEncoderLayers = 50;
inline constexpr std::size_t kPromptEncoderHiddenSize = 5120;
inline constexpr std::size_t kPromptEncoderIntermediateSize = 25600;
inline constexpr std::size_t kPromptEncoderQueryHeads = 64;
inline constexpr std::size_t kPromptEncoderKeyValueHeads = 8;
inline constexpr std::size_t kPromptEncoderHeadDimension = 128;
inline constexpr std::size_t kPromptEncoderVocabularySize = 151936;

inline constexpr std::uint32_t kVisionStartToken = 151652;
inline constexpr std::uint32_t kVisionEndToken = 151653;
inline constexpr std::uint32_t kImagePadToken = 151655;
inline constexpr std::size_t kVisionDeepstackLayers = 3;

/// One Qwen3-VL image block of the prompt: `rows` consecutive image-pad
/// tokens starting at `begin`, laid out row-major over the merged
/// grid_height x grid_width patch grid. Embeddings are BF16 rows of width
/// kPromptEncoderHiddenSize; deepstack features are added to the same rows
/// after decoder layers 0, 1 and 2.
struct PromptVisionBlock {
  std::size_t begin{0};
  std::uint32_t grid_height{0};
  std::uint32_t grid_width{0};
  std::vector<std::uint16_t> embedding;
  std::vector<std::vector<std::uint16_t>> deepstack;
};

struct PromptEncoderOptions {
  std::size_t layer_count{kPromptEncoderLayers};
  bool prefetch_layers{true};
  bool vision_inputs{false};
  bool deepstack_inputs{false};
  // Image blocks in prompt order. Non-empty switches the rotary positions
  // to Qwen3-VL's interleaved three-axis layout.
  std::span<const PromptVisionBlock> vision_blocks;
};

/// Qwen3-VL rope positions (temporal, height, width) of every token: text
/// advances all three axes together, an image block spans its grid from the
/// current offset and the text after it resumes at max(grid) + 1.
[[nodiscard]] bool BuildPromptRopePositions(
    std::size_t tokens, std::span<const PromptVisionBlock> blocks,
    std::vector<std::array<std::uint32_t, 3>>* positions,
    std::string* error = nullptr);

struct PromptEncoderLayerTelemetry {
  std::size_t layer{0};
  std::uint64_t weight_bytes{0};
  double load_ms{0.0};
  double prefetch_wait_ms{0.0};
  double submit_ms{0.0};
  double gpu_ms{0.0};
  std::uint64_t dispatches{0};
};

struct PromptEncoderTelemetry {
  std::vector<PromptEncoderLayerTelemetry> layers;
  std::uint64_t embedding_weight_bytes{0};
  std::uint64_t activation_bytes{0};
  std::uint64_t peak_device_bytes{0};
  std::uint64_t peak_registered_host_bytes{0};
  std::uint64_t live_registered_host_bytes_after{0};
  std::uint64_t total_dispatches{0};
};

struct PromptEmbedding {
  std::size_t tokens{0};
  std::size_t width{0};
  std::vector<std::uint16_t> values;
};

[[nodiscard]] bool EncodePromptLayer50(const ModelInventory& inventory,
                                       std::span<const std::uint32_t> token_ids,
                                       const PromptEncoderOptions& options,
                                       const CancellationToken* cancellation,
                                       PromptEmbedding* output,
                                       PromptEncoderTelemetry* telemetry,
                                       std::string* error = nullptr);

[[nodiscard]] std::uint64_t PromptEncoderLiveRegisteredHostBytes() noexcept;

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_PROMPT_ENCODER_HPP_
