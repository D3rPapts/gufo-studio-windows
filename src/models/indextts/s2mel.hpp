// IndexTTS-2.5 semantic-to-mel stage on the CPU: the EnhancedCodec decoder
// (codes -> 1024-dim semantic features, 2x upsampled), the s2mel
// InterpolateRegulator (length regulator) and the CFM/DiT mel generator
// (13-layer UViT DiT with adaptive RMSNorm, WaveNet head, 25 Euler steps,
// CFG 0.7). Time-major [T, C] layout throughout. FP32.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_INDEXTTS_S2MEL_HPP_
#define GUFO_MODELS_INDEXTTS_S2MEL_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/indextts/cpu/conv.hpp"
#include "src/models/indextts/pth.hpp"

namespace gufo::models::indextts {

class SemanticCodec {
public:
  [[nodiscard]] static std::unique_ptr<SemanticCodec> Load(const std::filesystem::path& codec_pth,
                                                           std::string* error);
  // EnhancedCodec.decode(codes) -> [2 * n, 1024]; `vq` / `dec` taps (optional).
  [[nodiscard]] std::vector<float> Decode(std::span<const std::int32_t> codes,
                                          std::vector<float>* vq = nullptr,
                                          std::vector<float>* dec = nullptr) const;

private:
  std::unique_ptr<PthArchive> sd_;
  std::vector<float> out_project_;  // folded weight norm [1024, 8]
  cpu::Conv1dTM embed_;
  cpu::Conv1dTM up_;
};

struct CfmOptions {
  std::int32_t steps{25};
  float cfg_rate{0.7F};
};

class S2Mel {
public:
  [[nodiscard]] static std::unique_ptr<S2Mel> Load(const std::filesystem::path& s2mel_pth,
                                                   std::string* error);
  // InterpolateRegulator(x [t, 1024], ylens) -> [ylens, 512]
  [[nodiscard]] std::vector<float> Regulate(const float* x, std::int64_t t,
                                            std::int64_t ylens) const;
  // DiT estimator for one batch element: x/prompt [t, 80], cond [t, 512],
  // style [192] -> velocity [t, 80].
  [[nodiscard]] std::vector<float> Velocity(const float* x, const float* prompt, float time,
                                            const float* style, const float* cond,
                                            std::int64_t t) const;
  // BASECFM.solve_euler with CFG: noise [t, 80] (consumed), mu [t, 512],
  // prompt mel [prompt_len, 80] -> mel [t, 80] (prompt rows are zero).
  using StepTap = std::function<void(int, const std::vector<float>& cond, const std::vector<float>& uncond)>;
  [[nodiscard]] std::vector<float> Sample(std::vector<float> noise, const float* mu,
                                          std::int64_t t, const float* prompt_mel,
                                          std::int64_t prompt_len, const float* style,
                                          const CfmOptions& options,
                                          const StepTap& velocity_tap = {}) const;

private:
  [[nodiscard]] const float* W(const std::string& name, std::int64_t numel = -1) const;
  [[nodiscard]] std::vector<float> TimeEmbed(const std::string& prefix, float time) const;

  std::unique_ptr<PthArchive> sd_;
  // length regulator convs
  cpu::Conv1dTM lr_convs_[4];
  cpu::Conv1dTM lr_out_;
  // DiT WaveNet
  std::vector<float> wn_cond_;  // folded [8192, 512]
  cpu::Conv1dTM wn_in_[8];
  std::vector<float> wn_res_[8];
  std::int64_t wn_res_out_[8]{};
  std::vector<float> final_linear_;  // folded [512, 512]
  std::vector<float> rope_cos_;
  std::vector<float> rope_sin_;
};

// torch.linspace(0, 1, n + 1) in float32.
[[nodiscard]] std::vector<float> TorchLinspace01(std::int32_t n);

}  // namespace gufo::models::indextts

#endif  // GUFO_MODELS_INDEXTTS_S2MEL_HPP_
