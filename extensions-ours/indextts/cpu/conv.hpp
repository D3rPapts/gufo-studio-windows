// Time-major ([T, C]) 1-D convolutions for the IndexTTS-2.5 CPU runtime,
// computed as one GEMM per kernel tap (weights repacked per tap at load).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_INDEXTTS_CPU_CONV_HPP_
#define GUFO_MODELS_INDEXTTS_CPU_CONV_HPP_

#include <cstdint>
#include <vector>

namespace gufo::models::indextts::cpu {

enum class PadMode { kZero, kReflect, kReplicate };

struct Conv1dTM {
  std::int64_t cin{0}, cout{0}, kernel{1}, dilation{1}, pad_left{0}, pad_right{0};
  PadMode mode{PadMode::kZero};
  std::vector<float> taps;  // [kernel][cout][cin]
  std::vector<float> bias;  // empty: none

  // From a torch Conv1d weight [cout, cin, k] (groups 1).
  void Set(const float* weight, const float* b, std::int64_t in, std::int64_t out, std::int64_t k,
           std::int64_t dil, std::int64_t left, std::int64_t right, PadMode m = PadMode::kZero);
  // x [t, cin] -> y [t_out, cout] (stride 1).
  void Run(const float* x, std::int64_t t, std::vector<float>* y) const;
};

struct ConvTranspose1dTM {
  std::int64_t cin{0}, cout{0}, kernel{1}, stride{1}, padding{0};
  std::vector<float> taps;  // [kernel][cout][cin]
  std::vector<float> bias;
  // From a torch ConvTranspose1d weight [cin, cout, k].
  void Set(const float* weight, const float* b, std::int64_t in, std::int64_t out, std::int64_t k,
           std::int64_t s, std::int64_t p);
  void Run(const float* x, std::int64_t t, std::vector<float>* y, std::int64_t* t_out) const;
};

// Depthwise conv over time on [t, c] (weight [c, 1, k]), zero padding.
void DepthwiseTM(const float* x, std::int64_t t, std::int64_t c, const float* w, const float* b,
                 std::int64_t k, std::int64_t pad_left, float* y);

// GroupNorm(1, C) over all of [t, c] + per-channel affine.
void GroupNorm1TM(float* x, std::int64_t t, std::int64_t c, const float* gamma, const float* beta,
                  float eps);

}  // namespace gufo::models::indextts::cpu

#endif  // GUFO_MODELS_INDEXTTS_CPU_CONV_HPP_
