#ifndef GUFO_MODELS_MOSS_MUSIC_REFERENCE_HPP_
#define GUFO_MODELS_MOSS_MUSIC_REFERENCE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "src/models/moss_music/model.hpp"

/// Straightforward F32 CPU implementation of the MOSS-Music audio side. It
/// reads the mapped GGUF weights directly and follows
/// src/modeling_moss_music.py of OpenMOSS/MOSS-Music (the chunked encoder)
/// operation for operation. It is the numerical yardstick for the HIP
/// runtime, not a serving path.
namespace gufo::models::moss_music {

class ThreadPool;

/// Encoder outputs for one clip: the compact [tokens, d_model] rows after
/// the final LayerNorm and the DeepStack taps (outputs of encoder layers
/// 8/16/24, before that norm), each [tokens, d_model].
struct EncoderOutput {
  std::size_t tokens{0};
  std::vector<float> final;
  std::vector<std::vector<float>> deepstack;
};

class CpuReference {
public:
  CpuReference(const AudioModel& model, unsigned threads);
  ~CpuReference();

  CpuReference(const CpuReference&) = delete;
  CpuReference& operator=(const CpuReference&) = delete;

  /// Chunking, the Conv2d stem, stem_proj and the per-chunk sinusoidal
  /// positions over mel-major log-mel features [mel_bins, frames]. Returns
  /// [tokens, d_model]; tokens == AudioTokenCount(frames).
  [[nodiscard]] std::vector<float> EncodeFrontend(std::span<const float> log_mel,
                                                  std::size_t frames) const;

  /// The whole encoder: EncodeFrontend, 32 Whisper layers with attention
  /// inside each chunk, taps and the final LayerNorm.
  [[nodiscard]] EncoderOutput EncodeAudio(std::span<const float> log_mel,
                                          std::size_t frames) const;

  /// GatedMLP over `rows` rows of [adapter.input]: [rows, adapter.output].
  [[nodiscard]] std::vector<float> GatedMlp(const GatedMlpWeights& mlp,
                                            std::span<const float> input,
                                            std::size_t rows) const;

private:
  const AudioModel& model_;
  std::unique_ptr<ThreadPool> pool_;
};

/// Sinusoidal positions of SinusoidsPositionEmbedding: [positions, d_model]
/// rows of [sin(t * inv_i) ..., cos(t * inv_i) ...], inv_i =
/// exp(-ln(10000) / (d_model / 2 - 1) * i).
[[nodiscard]] std::vector<float> SinusoidPositions(std::size_t positions,
                                                   std::size_t d_model);

}  // namespace gufo::models::moss_music

#endif  // GUFO_MODELS_MOSS_MUSIC_REFERENCE_HPP_
