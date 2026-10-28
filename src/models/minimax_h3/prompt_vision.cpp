#include "src/models/minimax_h3/prompt_vision.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <bit>
#include <cstdint>
#include <exception>

#include "src/core/image.hpp"
#include "src/models/qwen/vision/encoder.hpp"
#endif

namespace gufo::minimax_h3 {

#if defined(ENGINE_ENABLE_HIP)
namespace {

// The tower rounds its outputs to BF16 already; keep those bits.
bool ReadRows(const float* device, std::size_t elements,
              std::vector<std::uint16_t>* output, std::string* error) {
  std::vector<float> host(elements);
  const hipError_t status = hipMemcpy(
      host.data(), device, elements * sizeof(float), hipMemcpyDeviceToHost);
  if (status != hipSuccess) {
    *error = std::string("read H3 keyframe vision features: ") +
             hipGetErrorString(status);
    return false;
  }
  output->resize(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(host[index]);
    (*output)[index] = static_cast<std::uint16_t>(
        (bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
  }
  return true;
}

}  // namespace

bool EncodeKeyframeVision(const std::filesystem::path& projector,
                          std::span<const KeyframeImage> images,
                          const CancellationToken* cancellation,
                          std::vector<PromptVisionBlock>* blocks,
                          std::string* error) {
  std::string local_error;
  if (error == nullptr) {
    error = &local_error;
  }
  blocks->clear();
  try {
    models::qwen::vision::Encoder encoder(
        projector, static_cast<std::uint32_t>(kPromptEncoderHiddenSize));
    for (const KeyframeImage& image : images) {
      if (image.width % 32 != 0 || image.height % 32 != 0) {
        *error = "MiniMax H3 keyframe vision needs 32-pixel canvas sides";
        return false;
      }
      core::Image canvas;
      canvas.width = static_cast<std::uint32_t>(image.width);
      canvas.height = static_cast<std::uint32_t>(image.height);
      canvas.pixels.assign(image.rgb.begin(), image.rgb.end());
      const auto embedding = encoder.Encode(canvas, {}, [cancellation] {
        return cancellation != nullptr && cancellation->IsCancelled();
      });
      PromptVisionBlock block;
      block.grid_height = canvas.height / 32U;
      block.grid_width = canvas.width / 32U;
      const std::size_t elements =
          static_cast<std::size_t>(embedding->rows()) * embedding->width();
      if (embedding->rows() != block.grid_height * block.grid_width ||
          embedding->width() != kPromptEncoderHiddenSize ||
          embedding->deepstack().size() != kVisionDeepstackLayers) {
        *error =
            "MiniMax H3 vision projector must be the Qwen3-VL tower with "
            "deepstack layers 8, 16 and 24";
        return false;
      }
      if (!ReadRows(embedding->data(), elements, &block.embedding, error)) {
        return false;
      }
      for (const float* features : embedding->deepstack()) {
        block.deepstack.emplace_back();
        if (!ReadRows(features, elements, &block.deepstack.back(), error)) {
          return false;
        }
      }
      blocks->push_back(std::move(block));
    }
  } catch (const std::exception& exception) {
    *error =
        std::string("MiniMax H3 keyframe vision failed: ") + exception.what();
    return false;
  }
  return true;
}
#else
bool EncodeKeyframeVision(const std::filesystem::path&,
                          std::span<const KeyframeImage>,
                          const CancellationToken*,
                          std::vector<PromptVisionBlock>*, std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 keyframe vision requires ENGINE_ENABLE_HIP";
  }
  return false;
}
#endif

}  // namespace gufo::minimax_h3
