#ifndef GUFO_MODELS_MOSS_MUSIC_HIP_AUDIO_ENCODER_HPP_
#define GUFO_MODELS_MOSS_MUSIC_HIP_AUDIO_ENCODER_HPP_

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/moss_music/model.hpp"

namespace gufo::models::moss_music::hip {

/// Device-resident results of one clip; pointers stay valid until the next
/// Encode() call. Every buffer holds F32 copies of BF16 values.
struct EncodedAudio {
  std::size_t tokens{0};
  const float* frontend{nullptr};             // [tokens, d_model] stem + positions
  std::array<const float*, 3> taps{};         // encoder layers 8/16/24 [tokens, d_model]
  const float* final{nullptr};                // after layer_norm [tokens, d_model]
  const float* adapter{nullptr};              // audio_adapter [tokens, 4096]
  std::array<const float*, 3> mergers{};      // DeepStack mergers [tokens, 4096]
};

/// The MOSS-Music audio tower on the GPU: Conv2d stem and 32 Whisper layers
/// over 400-frame chunks on Qwen3-ASR's kernels, the DeepStack taps, the
/// audio adapter and the three mergers. Weights are the GGUF's own BF16.
class AudioEncoder {
public:
  static std::unique_ptr<AudioEncoder> Create(const AudioModel& model,
                                              hipStream_t stream,
                                              std::string* error);
  ~AudioEncoder();

  AudioEncoder(const AudioEncoder&) = delete;
  AudioEncoder& operator=(const AudioEncoder&) = delete;

  /// Encodes mel-major log-mel features [mel_bins, frames].
  bool Encode(std::span<const float> log_mel, std::size_t frames,
              EncodedAudio* output, std::string* error,
              const std::function<bool()>& is_cancelled = {});

  /// Copies `count` floats from a device pointer of EncodedAudio to the host.
  bool CopyToHost(const float* device, std::size_t count,
                  std::vector<float>* host, std::string* error) const;

  [[nodiscard]] std::size_t weight_bytes() const;

private:
  struct Impl;
  explicit AudioEncoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::moss_music::hip

#endif  // GUFO_MODELS_MOSS_MUSIC_HIP_AUDIO_ENCODER_HPP_
