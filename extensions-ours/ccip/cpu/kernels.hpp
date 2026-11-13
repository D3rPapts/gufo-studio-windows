// CPU fp32 kernels for the CCIP extension (OpenMP over rows/channels).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_CCIP_CPU_KERNELS_HPP_
#define GUFO_MODELS_CCIP_CPU_KERNELS_HPP_

#include <cstddef>

namespace gufo::models::ccip::cpu {

// C[M,N] = A[M,K] * B[K,N] (+ bias[N]); all row-major. accumulate: C += ...
void Gemm(const float* a, const float* b, const float* bias, float* c, int m, int n, int k,
          bool accumulate = false);
// C[M,N] = A[M,K] * B[N,K]^T (+ bias[N]).
void GemmBT(const float* a, const float* b, const float* bias, float* c, int m, int n, int k);

// LayerNorm over the last axis (weight only, no bias), rows x dim.
void LayerNorm(const float* x, const float* weight, float* y, int rows, int dim, float eps);

// StarReLU: scale * relu(x)^2 + bias, in place.
void StarRelu(float* x, std::size_t n, float scale, float bias);

// Softmax over the last axis, in place.
void SoftmaxRows(float* x, int rows, int cols);

// NHWC depthwise conv, square kernel, stride 1, zero pad, no bias.
void DepthwiseConvNhwc(const float* x, const float* w, float* y, int h, int wd, int c,
                       int kernel, int pad);

// Dense conv on an NHWC input (or CHW when chw_input) into NHWC output, square
// kernel; weight [out, in, k, k]. Implemented as im2col + Gemm.
void ConvToNhwc(const float* x, bool chw_input, int h, int w, int cin, const float* weight,
                const float* bias, int cout, int kernel, int stride, int pad, float* y,
                int* out_h, int* out_w);

}  // namespace gufo::models::ccip::cpu

#endif  // GUFO_MODELS_CCIP_CPU_KERNELS_HPP_
