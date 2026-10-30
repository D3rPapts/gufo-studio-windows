#ifndef GUFO_MODELS_QWEN3_OMNI_MODEL_HPP_
#define GUFO_MODELS_QWEN3_OMNI_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

/// Qwen3-Omni-30B-A3B-Captioner: the thinker of Qwen3-Omni fine-tuned to
/// describe one audio clip. Weights come as llama.cpp GGUFs: the text model
/// (arch `qwen3vlmoe`, K-quants) and the projector file holding the AuT audio
/// tower (`clip.audio.projector_type` = `qwen3a`). The vision tower in the
/// projector file is ignored; the captioner accepts audio only.
namespace gufo::models::qwen3_omni {

struct AudioConfig {
  std::uint32_t hidden{1280};
  std::uint32_t ffn{5120};
  std::uint32_t layers{32};
  std::uint32_t heads{20};
  std::uint32_t output{2048};
  std::uint32_t mel_bins{128};
  std::uint32_t conv_channels{480};
  float eps{1.0e-5F};
  // Not stored in the GGUF; values of the published config.json, which match
  // Qwen3-ASR: 100-frame convolution chunks and 800-frame attention windows.
  std::uint32_t n_window{50};
  std::uint32_t n_window_infer{800};

  [[nodiscard]] std::uint32_t HeadDim() const { return hidden / heads; }
  [[nodiscard]] std::uint32_t ChunkFrames() const { return 2 * n_window; }
};

struct TextConfig {
  std::uint32_t layers{48};
  std::uint32_t hidden{2048};
  std::uint32_t heads{32};
  std::uint32_t kv_heads{4};
  std::uint32_t head_dim{128};
  std::uint32_t experts{128};
  std::uint32_t experts_used{8};
  std::uint32_t expert_ff{768};
  std::uint32_t vocab{152064};
  std::uint32_t context{65536};
  float rope_theta{1.0e6F};
  float eps{1.0e-6F};

  [[nodiscard]] std::uint32_t QDim() const { return heads * head_dim; }
  [[nodiscard]] std::uint32_t KvDim() const { return kv_heads * head_dim; }
};

/// Special tokens of the Qwen3-Omni vocabulary used by the caption prompt.
struct TokenIds {
  std::int32_t im_start{151644};
  std::int32_t im_end{151645};
  std::int32_t audio_start{151669};
  std::int32_t audio_end{151670};
  std::int32_t audio_pad{151675};
  std::int32_t endoftext{151643};
};

/// A weight in its GGUF encoding. `cols` is the reduction (innermost)
/// dimension; `rows` the output dimension; `experts` stacks whole matrices.
struct Weight {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};
  std::size_t row_bytes{0};

  [[nodiscard]] const void* Row(std::uint64_t expert,
                                std::uint64_t row) const noexcept {
    return static_cast<const std::uint8_t*>(data) +
           (expert * rows + row) * row_bytes;
  }
  [[nodiscard]] const float* F32() const noexcept {
    return static_cast<const float*>(data);
  }
  [[nodiscard]] std::size_t Bytes() const noexcept {
    return static_cast<std::size_t>(experts * rows) * row_bytes;
  }
};

struct AudioLayer {
  Weight ln1_weight, ln1_bias;
  Weight q_weight, q_bias, k_weight, k_bias, v_weight, v_bias;
  Weight out_weight, out_bias;
  Weight ln2_weight, ln2_bias;
  Weight fc1_weight, fc1_bias, fc2_weight, fc2_bias;
};

struct AudioWeights {
  Weight conv1_weight, conv1_bias;  // [480, 1, 3, 3]
  Weight conv2_weight, conv2_bias;  // [480, 480, 3, 3]
  Weight conv3_weight, conv3_bias;  // [480, 480, 3, 3]
  Weight conv_out;                  // [hidden, 480 * 16], no bias
  std::vector<AudioLayer> layers;
  Weight post_ln_weight, post_ln_bias;
  Weight proj1_weight, proj1_bias;  // [hidden, hidden]
  Weight proj2_weight, proj2_bias;  // [output, hidden]
};

struct TextLayer {
  Weight attn_norm;
  Weight q, k, v, o;
  Weight q_norm, k_norm;  // per-head RMSNorm gamma [head_dim]
  Weight ffn_norm;
  Weight router;  // F32 [experts, hidden]
  Weight gate_exps, up_exps, down_exps;
};

struct TextWeights {
  Weight token_embd;  // [vocab, hidden]
  std::vector<TextLayer> layers;
  Weight output_norm;
  Weight output;  // [vocab, hidden]
};

/// Both GGUF files, memory-mapped, with validated views of every weight the
/// captioner uses.
class Model {
public:
  [[nodiscard]] static std::unique_ptr<Model> Load(
      const std::filesystem::path& text_gguf,
      const std::filesystem::path& projector_gguf, std::string* error);

  [[nodiscard]] const AudioConfig& audio_config() const { return audio_; }
  [[nodiscard]] const TextConfig& text_config() const { return text_; }
  [[nodiscard]] const TokenIds& tokens() const { return token_ids_; }
  [[nodiscard]] const AudioWeights& audio() const { return audio_weights_; }
  [[nodiscard]] const TextWeights& text() const { return text_weights_; }
  [[nodiscard]] const core::GgufReader& text_reader() const {
    return *text_reader_;
  }
  [[nodiscard]] const core::GgufReader& projector_reader() const {
    return *projector_reader_;
  }

private:
  Model() = default;

  std::unique_ptr<core::GgufReader> text_reader_;
  std::unique_ptr<core::GgufReader> projector_reader_;
  AudioConfig audio_;
  TextConfig text_;
  TokenIds token_ids_;
  AudioWeights audio_weights_;
  TextWeights text_weights_;
};

/// Audio embeddings the frontend yields for `frames` log-mel frames: 13 per
/// full 100-frame chunk plus the convolved length of the remainder.
[[nodiscard]] std::size_t AudioTokenCount(std::size_t frames,
                                          std::uint32_t n_window = 50);

/// Bytes of one row of `cols` elements in `type`; zero when unsupported.
[[nodiscard]] std::size_t RowBytes(core::GgmlType type, std::uint64_t cols);

/// The encoded bytes of `weight`, a view into `reader`, read from the GGUF
/// file with unbuffered reads into `scratch` instead of through the mapping:
/// Windows keeps every mapped page a loader touches in the process working
/// set, so converting a whole tower through the mapping pins it in host RAM.
/// The span points into `scratch` (reused across calls), or into the mapping
/// when the reader has no file behind it. Throws when the read fails.
[[nodiscard]] std::span<const std::byte> ReadWeightBytes(
    const core::GgufReader& reader, const Weight& weight,
    std::vector<std::byte>* scratch);

}  // namespace gufo::models::qwen3_omni

#endif  // GUFO_MODELS_QWEN3_OMNI_MODEL_HPP_
