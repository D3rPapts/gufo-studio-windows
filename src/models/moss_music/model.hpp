#ifndef GUFO_MODELS_MOSS_MUSIC_MODEL_HPP_
#define GUFO_MODELS_MOSS_MUSIC_MODEL_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/moss_music/config.hpp"

/// The converted MOSS-Music audio file (`<stem>.audio.gguf`, written by
/// tools/moss_music/convert_hf_to_gufo.py): the encoder, the audio adapter
/// and the three DeepStack mergers, memory-mapped with validated views of
/// every weight. The LLM lives in its own llama.cpp `qwen3` GGUF.
namespace gufo::models::moss_music {

/// A weight in its GGUF encoding: `cols` is the reduction (innermost)
/// dimension, `rows` the output dimension. Matrices are BF16, vectors F32.
struct Weight {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::size_t row_bytes{0};

  [[nodiscard]] const void* Row(std::uint64_t row) const noexcept {
    return static_cast<const std::uint8_t*>(data) + row * row_bytes;
  }
  [[nodiscard]] const float* F32() const noexcept {
    return static_cast<const float*>(data);
  }
  [[nodiscard]] std::size_t Bytes() const noexcept {
    return static_cast<std::size_t>(rows) * row_bytes;
  }
};

/// Conv2d stem: kernels viewed as [cout] rows of cin * 9 (channel, ky, kx).
struct StemWeights {
  Weight conv1_weight, conv1_bias;  // [480, 1 * 9], [480]
  Weight conv2_weight, conv2_bias;  // [480, 480 * 9], [480]
  Weight conv3_weight, conv3_bias;  // [480, 480 * 9], [480]
  Weight proj_weight, proj_bias;    // [1280, 7680], [1280]
};

struct AudioLayerWeights {
  Weight ln1_weight, ln1_bias;
  Weight q_weight, q_bias, k_weight, v_weight, v_bias;  // k has no bias
  Weight out_weight, out_bias;
  Weight ln2_weight, ln2_bias;
  Weight fc1_weight, fc1_bias, fc2_weight, fc2_bias;
};

struct AudioWeights {
  StemWeights stem;
  std::vector<AudioLayerWeights> layers;
  Weight post_ln_weight, post_ln_bias;
};

/// GatedMLP down(silu(gate(x)) * up(x)).
struct GatedMlpWeights {
  Weight gate, up, down;  // [8192, 1280], [8192, 1280], [4096, 8192]
};

class AudioModel {
public:
  [[nodiscard]] static std::unique_ptr<AudioModel> Load(
      const std::filesystem::path& audio_gguf, std::string* error);

  [[nodiscard]] const ModelConfig& config() const { return config_; }
  [[nodiscard]] const AudioWeights& audio() const { return audio_; }
  [[nodiscard]] const GatedMlpWeights& adapter() const { return adapter_; }
  [[nodiscard]] const std::array<GatedMlpWeights, 3>& mergers() const {
    return mergers_;
  }
  [[nodiscard]] const core::GgufReader& reader() const { return *reader_; }

private:
  AudioModel() = default;

  std::unique_ptr<core::GgufReader> reader_;
  ModelConfig config_;
  AudioWeights audio_;
  GatedMlpWeights adapter_;
  std::array<GatedMlpWeights, 3> mergers_;
};

/// Decodes one weight row to F32 (F32 copy, BF16/F16 widening).
void DecodeRow(const Weight& weight, std::uint64_t row, float* out);

}  // namespace gufo::models::moss_music

#endif  // GUFO_MODELS_MOSS_MUSIC_MODEL_HPP_
