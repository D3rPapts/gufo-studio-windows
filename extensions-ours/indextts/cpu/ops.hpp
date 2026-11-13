// FP32 CPU kernels for the IndexTTS-2.5 reference runtime (all checkpoints are
// FP32 and are used in place from the mapped .pth files). AVX2/FMA + OpenMP.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_INDEXTTS_CPU_OPS_HPP_
#define GUFO_MODELS_INDEXTTS_CPU_OPS_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gufo::models::indextts::cpu {

// y[rows, out] = x[rows, in] * W^T + b, W row-major [out, in] (nn.Linear).
void Linear(const float* x, std::int64_t rows, std::int64_t in, const float* w,
            std::int64_t out, const float* bias, float* y);
// y[rows, out] = x[rows, in] * W + b, W row-major [in, out] (GPT-2 Conv1D).
void LinearInOut(const float* x, std::int64_t rows, std::int64_t in, const float* w,
                 std::int64_t out, const float* bias, float* y);

// Row-wise LayerNorm (biased variance, like torch). gamma/beta may be null.
void LayerNorm(const float* x, float* y, std::int64_t rows, std::int64_t dim,
               const float* gamma, const float* beta, float eps);
// x * rsqrt(mean(x^2) + eps) * w (w may be null).
void RmsNorm(const float* x, float* y, std::int64_t rows, std::int64_t dim, const float* w,
             float eps);

// 1-D convolution on channel-major data x[cin, t_in] -> y[cout, t_out] with
// explicit left/right zero padding. w[cout, cin/groups, k].
void Conv1d(const float* x, std::int64_t cin, std::int64_t t_in, const float* w,
            const float* bias, std::int64_t cout, std::int64_t kernel, std::int64_t stride,
            std::int64_t pad_left, std::int64_t pad_right, std::int64_t dilation,
            std::int64_t groups, std::vector<float>* y, std::int64_t* t_out);
// Reflect padding (torch 'reflect') of channel-major x[c, t].
std::vector<float> PadReflect(const float* x, std::int64_t c, std::int64_t t,
                              std::int64_t left, std::int64_t right);

// [rows, cols] <-> [cols, rows]
std::vector<float> Transpose(const float* x, std::int64_t rows, std::int64_t cols);

float GeluErf(float x);
float GeluTanh(float x);
float Silu(float x);
float Mish(float x);
float Sigmoid(float x);
void GeluErfInPlace(float* x, std::size_t n);
void GeluTanhInPlace(float* x, std::size_t n);
void SiluInPlace(float* x, std::size_t n);
void ReluInPlace(float* x, std::size_t n);

// In-place softmax over each row.
void SoftmaxRows(float* x, std::int64_t rows, std::int64_t cols);

// Multi-head attention for one sequence, q[q_len, heads*d], k/v[kv_len, heads*d]
// with row strides (in floats); out[q_len, heads*d]. scores *= scale. If
// `bias` is non-null it is added per head: bias[h][q][k] (q_len*kv_len per
// head). causal masks k > q + (kv_len - q_len).
void Mha(const float* q, std::int64_t q_stride, const float* k, std::int64_t k_stride,
         const float* v, std::int64_t v_stride, float* out, std::int64_t out_stride,
         std::int64_t q_len, std::int64_t kv_len, std::int64_t heads, std::int64_t head_dim,
         float scale, bool causal, const float* bias = nullptr);

// Cosine similarity / max abs diff helpers for tests.
double Cosine(const float* a, const float* b, std::size_t n);
double MaxAbsDiff(const float* a, const float* b, std::size_t n);

}  // namespace gufo::models::indextts::cpu

#endif  // GUFO_MODELS_INDEXTTS_CPU_OPS_HPP_
