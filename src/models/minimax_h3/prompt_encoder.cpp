#include "src/models/minimax_h3/prompt_encoder.hpp"

#include <algorithm>

namespace gufo::minimax_h3 {

bool BuildPromptRopePositions(
    std::size_t tokens, std::span<const PromptVisionBlock> blocks,
    std::vector<std::array<std::uint32_t, 3>>* positions, std::string* error) {
  positions->clear();
  positions->reserve(tokens);
  std::uint32_t offset = 0;
  std::size_t row = 0;
  for (const PromptVisionBlock& block : blocks) {
    const std::size_t rows =
        static_cast<std::size_t>(block.grid_height) * block.grid_width;
    if (block.begin < row || rows == 0 || block.begin + rows > tokens) {
      if (error != nullptr) {
        *error = "MiniMax H3 image blocks overlap or leave the prompt";
      }
      return false;
    }
    for (; row < block.begin; ++row) {
      positions->push_back({offset, offset, offset});
      ++offset;
    }
    for (std::uint32_t y = 0; y < block.grid_height; ++y) {
      for (std::uint32_t x = 0; x < block.grid_width; ++x) {
        positions->push_back({offset, offset + y, offset + x});
      }
    }
    row += rows;
    offset += std::max(block.grid_height, block.grid_width);
  }
  for (; row < tokens; ++row) {
    positions->push_back({offset, offset, offset});
    ++offset;
  }
  return true;
}

#if !defined(ENGINE_ENABLE_HIP)
bool EncodePromptLayer50(const ModelInventory&, std::span<const std::uint32_t>,
                         const PromptEncoderOptions&, const CancellationToken*,
                         PromptEmbedding*, PromptEncoderTelemetry*,
                         std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 prompt encoder requires ENGINE_ENABLE_HIP";
  }
  return false;
}

std::uint64_t PromptEncoderLiveRegisteredHostBytes() noexcept {
  return 0;
}
#endif

}  // namespace gufo::minimax_h3
