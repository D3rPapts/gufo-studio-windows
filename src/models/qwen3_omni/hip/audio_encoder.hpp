#ifndef GUFO_MODELS_QWEN3_OMNI_HIP_AUDIO_ENCODER_HPP_
#define GUFO_MODELS_QWEN3_OMNI_HIP_AUDIO_ENCODER_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string>

#include "src/models/qwen3_omni/model.hpp"

namespace gufo::models::qwen3_omni::hip {

/// The Qwen3-Omni AuT audio tower on the GPU. It runs Qwen3-ASR's kernels
/// (same architecture, wider) on BF16 copies of the projector GGUF's F16/F32
/// weights; BF16 is the checkpoint's own precision, so the copies are exact.
class AudioEncoder {
public:
  static std::unique_ptr<AudioEncoder> Create(const Model& model,
                                              hipStream_t stream,
                                              std::string* error);
  ~AudioEncoder();

  AudioEncoder(const AudioEncoder&) = delete;
  AudioEncoder& operator=(const AudioEncoder&) = delete;

  /// Encodes mel-major log-mel features [mel_bins, frames]. On success
  /// `*device_output` points at [tokens, output] F32 embeddings that stay
  /// valid until the next call.
  bool Encode(std::span<const float> log_mel, std::size_t frames,
              const float** device_output, std::size_t* tokens,
              std::string* error,
              const std::function<bool()>& is_cancelled = {});

  /// Device bytes held by weights.
  [[nodiscard]] std::size_t weight_bytes() const;

private:
  struct Impl;
  explicit AudioEncoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen3_omni::hip

#endif  // GUFO_MODELS_QWEN3_OMNI_HIP_AUDIO_ENCODER_HPP_
