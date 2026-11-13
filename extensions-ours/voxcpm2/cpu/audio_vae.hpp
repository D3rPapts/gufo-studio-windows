// AudioVAE V2 of VoxCPM2 on the CPU: causal encoder (16 kHz -> 25 Hz latents)
// and a stateful streaming decoder (latents -> 48 kHz). Every convolution is
// causal, so decoding chunk by chunk with carried left context equals one
// full decode (the StreamingVAEDecoder contract of voxcpm).
// FP32 throughout (the checkpoint is FP32). Written with AI assistance
// (Claude Code).
#ifndef GUFO_MODELS_VOXCPM2_CPU_AUDIO_VAE_HPP_
#define GUFO_MODELS_VOXCPM2_CPU_AUDIO_VAE_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "voxcpm2/config.hpp"

namespace gufo::models::voxcpm2::cpu {

// Weight [out, in / groups, kernel] (weight norm folded), causal left padding.
struct Conv1d {
  std::int32_t in{0}, out{0}, kernel{1}, stride{1}, dilation{1}, groups{1};
  std::int32_t left_pad{0};
  std::vector<float> weight;
  std::vector<float> bias;
};

// Causal ConvTranspose1d with kernel 2 * stride; weight stored [out, in, kernel].
struct ConvTranspose1d {
  std::int32_t in{0}, out{0}, stride{1};
  std::vector<float> weight;
  std::vector<float> bias;
};

struct Snake {
  std::vector<float> alpha;
  std::vector<float> inv;  // 1 / (alpha + 1e-9)
};

struct ResidualUnit {
  Snake snake1;
  Conv1d conv1;  // depthwise k7, dilated
  Snake snake2;
  Conv1d conv2;  // 1x1
};

struct EncoderBlock {
  ResidualUnit units[3];
  Snake snake;
  Conv1d down;
};

struct DecoderBlock {
  std::vector<float> sr_scale, sr_bias;  // sample-rate conditioning (one bucket)
  Snake snake;
  ConvTranspose1d up;
  ResidualUnit units[3];
};

class AudioVae {
public:
  [[nodiscard]] static std::unique_ptr<AudioVae> Load(const std::string& model_dir,
                                                      const AudioVaeConfig& config,
                                                      std::string* error);

  [[nodiscard]] const AudioVaeConfig& config() const noexcept { return config_; }

  struct EncodeTrace {
    std::vector<float> block0;  // [channels, samples] after the first conv
    std::vector<float> block4;  // [channels, frames] after the last block
  };
  // 16 kHz mono wave (length a multiple of chunk_size) -> mu [latent, frames].
  [[nodiscard]] std::vector<float> Encode(const float* wave, std::int64_t samples,
                                          std::int64_t* frames,
                                          EncodeTrace* trace = nullptr) const;

  // Streaming decoder state (left context of every causal convolution).
  struct DecoderState {
    std::vector<std::vector<float>> history;
    bool started{false};
  };
  struct DecodeTrace {
    std::vector<float> model1;  // after the 1x1 input conv [decoder_dim, frames]
    std::vector<float> model2;  // after the first decoder block
    std::vector<float> last_block;
  };
  // z [latent, frames] (channel-major) -> frames * decode_chunk_size samples.
  [[nodiscard]] std::vector<float> DecodeChunk(DecoderState* state, const float* z,
                                               std::int64_t frames,
                                               DecodeTrace* trace = nullptr) const;

  // Layers (FP32, weight norm folded) for the GPU upload.
  [[nodiscard]] const Conv1d& enc_in() const noexcept { return enc_in_; }
  [[nodiscard]] const std::vector<EncoderBlock>& enc_blocks() const noexcept { return enc_blocks_; }
  [[nodiscard]] const Conv1d& enc_mu() const noexcept { return enc_mu_; }
  [[nodiscard]] const Conv1d& dec_depthwise() const noexcept { return dec_depthwise_; }
  [[nodiscard]] const Conv1d& dec_pointwise() const noexcept { return dec_pointwise_; }
  [[nodiscard]] const std::vector<DecoderBlock>& dec_blocks() const noexcept { return dec_blocks_; }
  [[nodiscard]] const Snake& dec_snake() const noexcept { return dec_snake_; }
  [[nodiscard]] const Conv1d& dec_out() const noexcept { return dec_out_; }

private:
  AudioVaeConfig config_;
  Conv1d enc_in_;
  std::vector<EncoderBlock> enc_blocks_;
  Conv1d enc_mu_;
  Conv1d dec_depthwise_, dec_pointwise_;
  std::vector<DecoderBlock> dec_blocks_;
  Snake dec_snake_;
  Conv1d dec_out_;
};

}  // namespace gufo::models::voxcpm2::cpu

#endif  // GUFO_MODELS_VOXCPM2_CPU_AUDIO_VAE_HPP_
