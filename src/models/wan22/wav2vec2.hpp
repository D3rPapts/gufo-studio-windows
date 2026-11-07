// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_WAN22_WAV2VEC2_HPP_
#define GUFO_MODELS_WAN22_WAV2VEC2_HPP_

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::wan22 {

/// The InfiniteTalk / MultiTalk audio encoder: TencentGameMate
/// chinese-wav2vec2-base (Kijai's `wav2vec2-chinese-base_fp16.safetensors`,
/// keys with or without "wav2vec2."), run as ComfyUI's AudioEncoderEncode
/// does: mono 16 kHz input without normalization, 7 convolutions (the first
/// with a per-channel GroupNorm, all GELU, no biases), feature projection,
/// grouped positional convolution (weight norm over the kernel axis), the
/// post-norm ("not stable") encoder of 12 layers (768 wide, 12 heads of 64,
/// FFN 3072, exact GELU). Everything in FP32 (rocBLAS).
class Wav2Vec2 {
public:
  static constexpr int kSampleRate = 16000;
  static constexpr int kWidth = 768;
  static constexpr int kLayers = 12;
  /// Longest input: 4000 encoder frames (80 s).
  static constexpr int kMaxFrames = 4000;

  [[nodiscard]] static std::unique_ptr<Wav2Vec2> Load(
      const std::filesystem::path& file, std::string* error);
  ~Wav2Vec2();
  Wav2Vec2(const Wav2Vec2&) = delete;
  Wav2Vec2& operator=(const Wav2Vec2&) = delete;

  /// `audio` mono 16 kHz. `layers` receives the outputs of the 12 encoder
  /// layers (ComfyUI's encoded_audio_all_layers[1:]), layer-major
  /// [12][frames][768] at 50 frames per second.
  bool Encode(std::span<const float> audio, std::vector<float>* layers,
              int* frames, std::string* error);

  /// Encoder frames for `samples` 16 kHz samples (the convolution stack).
  [[nodiscard]] static int FramesFor(std::size_t samples);
  [[nodiscard]] std::size_t weight_bytes() const;

private:
  struct Impl;
  explicit Wav2Vec2(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

/// WanInfiniteTalkToVideo's resampling of the layer outputs from 50 to 25
/// frames per second: torch linear interpolation with align_corners to
/// int(frames / 50 * 25) frames. Returns frame-major [frames25][12][768].
std::vector<float> TalkAudioFeatures(std::span<const float> layers,
                                     int frames50, int* frames25);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_WAV2VEC2_HPP_
