// BigVGAN v2 vocoder (nvidia/bigvgan_v2_22khz_80band_256x, MIT) on the CPU:
// weight-normed convs folded at load, transposed-conv upsamplers, AMPBlock1
// with anti-aliased SnakeBeta (Kaiser-sinc 2x up / down, replicate padding),
// clamp at the end (use_tanh_at_final=false). Time-major [T, C]. FP32.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_INDEXTTS_BIGVGAN_HPP_
#define GUFO_MODELS_INDEXTTS_BIGVGAN_HPP_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "src/models/indextts/cpu/conv.hpp"
#include "src/models/indextts/pth.hpp"

namespace gufo::models::indextts {

class BigVgan {
public:
  [[nodiscard]] static std::unique_ptr<BigVgan> Load(const std::filesystem::path& dir,
                                                     std::string* error);
  // mel [frames, 80] -> waveform [frames * 256] in [-1, 1].
  [[nodiscard]] std::vector<float> Synthesize(const float* mel, std::int64_t frames) const;

  struct Act {  // Activation1d(SnakeBeta)
    std::vector<float> alpha;  // exp(alpha)
    std::vector<float> inv_beta;  // 1 / (exp(beta) + 1e-9)
    float up[12]{};
    float down[12]{};
  };

private:
  struct ResBlock {
    std::vector<cpu::Conv1dTM> c1;
    std::vector<cpu::Conv1dTM> c2;
    std::vector<Act> acts;  // 2 per (c1, c2) pair
  };
  cpu::Conv1dTM pre_;
  std::vector<cpu::ConvTranspose1dTM> ups_;
  std::vector<ResBlock> blocks_;  // 6 stages x 3 kernels
  Act post_act_;
  cpu::Conv1dTM post_;
};

}  // namespace gufo::models::indextts

#endif  // GUFO_MODELS_INDEXTTS_BIGVGAN_HPP_
