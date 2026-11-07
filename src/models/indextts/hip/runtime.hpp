// gfx1151 HIP runtime for IndexTTS-2.5: the GPT-2 AR decoder (KV cache on the
// device, BF16 weights, FP32 activations), the DiT/CFM mel generator (whole
// Euler loop with CFG on the device; time-only terms computed on the CPU) and
// the BigVGAN vocoder. Text, reference conditioning (CAM++, w2v-BERT, emotion
// encoder), sampling, codec decoder and length regulator stay on the CPU.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_INDEXTTS_HIP_RUNTIME_HPP_
#define GUFO_MODELS_INDEXTTS_HIP_RUNTIME_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/models/indextts/bigvgan.hpp"
#include "src/models/indextts/gpt.hpp"
#include "src/models/indextts/s2mel.hpp"

namespace gufo::models::indextts::hip {

class Runtime {
public:
  // nullptr (with `error`) when HIP is not compiled in or no device is usable.
  [[nodiscard]] static std::unique_ptr<Runtime> Create(const UnifiedVoice& gpt, const S2Mel& s2mel,
                                                       const BigVgan& vocoder, std::string* error);
  ~Runtime();

  // GPT: same contract as UnifiedVoice::Prefill / Step (logits on the host).
  void GptPrefill(const std::vector<float>& prefix, std::int64_t rows, std::int64_t max_new,
                  std::vector<float>* logits);
  void GptStep(std::int32_t token, std::vector<float>* logits);

  // S2Mel::Sample on the device (noise/mu/prompt/style on the host, mel [t, 80] back).
  [[nodiscard]] std::vector<float> CfmSample(const std::vector<float>& noise, const float* mu,
                                             std::int64_t t, const float* prompt_mel,
                                             std::int64_t prompt_len, const float* style,
                                             const CfmOptions& options);
  // One estimator evaluation (tests): x/prompt [t, 80], cond [t, 512].
  [[nodiscard]] std::vector<float> Velocity(const float* x, const float* prompt, float time,
                                            const float* style, const float* cond, std::int64_t t);

  // BigVgan::Synthesize on the device.
  [[nodiscard]] std::vector<float> Vocode(const float* mel, std::int64_t frames);

  // DiT/CFM kernels: true (default, unless GUFO_INDEXTTS_DIT_FAST=0) = CFG as batch 2, hipBLASLt BF16
  // projections and GEMM attention; false = the first port's tiled GEMM / per-row attention (A/B tests).
  void SetFastDit(bool fast);
  // With GUFO_INDEXTTS_PROFILE=1: per-category GPU times of the last CfmSample, else empty.
  [[nodiscard]] std::string LastCfmProfile() const;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::indextts::hip

#endif  // GUFO_MODELS_INDEXTTS_HIP_RUNTIME_HPP_
