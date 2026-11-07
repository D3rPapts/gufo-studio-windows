// Written with AI assistance (Claude Code).
#include "src/models/ccip/cpu/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace gufo::models::ccip::cpu {
namespace {
constexpr int kRowBlock = 4;
constexpr int kColBlock = 256;
}  // namespace

void Gemm(const float* a, const float* b, const float* bias, float* c, int m, int n, int k,
          bool accumulate) {
  const int row_blocks = (m + kRowBlock - 1) / kRowBlock;
  const int col_blocks = (n + kColBlock - 1) / kColBlock;
#pragma omp parallel for schedule(static)
  for (int t = 0; t < row_blocks * col_blocks; ++t) {
    const int rb = t / col_blocks;
    const int cb = t % col_blocks;
    const int i0 = rb * kRowBlock;
    const int i1 = std::min(m, i0 + kRowBlock);
    const int j0 = cb * kColBlock;
    const int j1 = std::min(n, j0 + kColBlock);
    const int nj = j1 - j0;
    alignas(64) float acc[kRowBlock][kColBlock];
    for (int r = 0; r < i1 - i0; ++r) {
      float* dst = acc[r];
      if (accumulate) {
        std::memcpy(dst, c + static_cast<std::size_t>(i0 + r) * n + j0, sizeof(float) * nj);
      } else if (bias != nullptr) {
        std::memcpy(dst, bias + j0, sizeof(float) * nj);
      } else {
        std::fill(dst, dst + nj, 0.0f);
      }
    }
    if (i1 - i0 == kRowBlock) {
      const float* a0 = a + static_cast<std::size_t>(i0) * k;
      const float* a1 = a0 + k;
      const float* a2 = a1 + k;
      const float* a3 = a2 + k;
      for (int p = 0; p < k; ++p) {
        const float* brow = b + static_cast<std::size_t>(p) * n + j0;
        const float v0 = a0[p], v1 = a1[p], v2 = a2[p], v3 = a3[p];
        float* c0 = acc[0];
        float* c1 = acc[1];
        float* c2 = acc[2];
        float* c3 = acc[3];
#pragma omp simd
        for (int j = 0; j < nj; ++j) {
          const float bv = brow[j];
          c0[j] += v0 * bv;
          c1[j] += v1 * bv;
          c2[j] += v2 * bv;
          c3[j] += v3 * bv;
        }
      }
    } else {
      for (int r = 0; r < i1 - i0; ++r) {
        const float* ar = a + static_cast<std::size_t>(i0 + r) * k;
        float* cr = acc[r];
        for (int p = 0; p < k; ++p) {
          const float* brow = b + static_cast<std::size_t>(p) * n + j0;
          const float v = ar[p];
#pragma omp simd
          for (int j = 0; j < nj; ++j) cr[j] += v * brow[j];
        }
      }
    }
    for (int r = 0; r < i1 - i0; ++r) {
      std::memcpy(c + static_cast<std::size_t>(i0 + r) * n + j0, acc[r], sizeof(float) * nj);
    }
  }
}

void GemmBT(const float* a, const float* b, const float* bias, float* c, int m, int n, int k) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < m; ++i) {
    const float* ar = a + static_cast<std::size_t>(i) * k;
    for (int j = 0; j < n; ++j) {
      const float* br = b + static_cast<std::size_t>(j) * k;
      float s = 0.0f;
#pragma omp simd reduction(+ : s)
      for (int p = 0; p < k; ++p) s += ar[p] * br[p];
      c[static_cast<std::size_t>(i) * n + j] = s + (bias != nullptr ? bias[j] : 0.0f);
    }
  }
}

void LayerNorm(const float* x, const float* weight, float* y, int rows, int dim, float eps) {
#pragma omp parallel for schedule(static)
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<std::size_t>(r) * dim;
    float* yr = y + static_cast<std::size_t>(r) * dim;
    double mean = 0.0;
    for (int i = 0; i < dim; ++i) mean += xr[i];
    mean /= dim;
    double var = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double d = xr[i] - mean;
      var += d * d;
    }
    var /= dim;
    const float inv = static_cast<float>(1.0 / std::sqrt(var + eps));
    const float m = static_cast<float>(mean);
    for (int i = 0; i < dim; ++i) yr[i] = (xr[i] - m) * inv * weight[i];
  }
}

void StarRelu(float* x, std::size_t n, float scale, float bias) {
  const auto count = static_cast<std::ptrdiff_t>(n);
#pragma omp parallel for simd schedule(static)
  for (std::ptrdiff_t i = 0; i < count; ++i) {
    const float r = x[i] > 0.0f ? x[i] : 0.0f;
    x[i] = scale * (r * r) + bias;
  }
}

void SoftmaxRows(float* x, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    float* xr = x + static_cast<std::size_t>(r) * cols;
    float mx = xr[0];
    for (int i = 1; i < cols; ++i) mx = std::max(mx, xr[i]);
    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
      xr[i] = std::exp(xr[i] - mx);
      sum += xr[i];
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < cols; ++i) xr[i] *= inv;
  }
}

void DepthwiseConvNhwc(const float* x, const float* w, float* y, int h, int wd, int c,
                       int kernel, int pad) {
  // Weight [c, 1, k, k] -> per tap [k*k][c] for contiguous channel loops.
  std::vector<float> wt(static_cast<std::size_t>(kernel) * kernel * c);
  for (int ch = 0; ch < c; ++ch) {
    for (int t = 0; t < kernel * kernel; ++t) {
      wt[static_cast<std::size_t>(t) * c + ch] = w[static_cast<std::size_t>(ch) * kernel * kernel + t];
    }
  }
#pragma omp parallel for schedule(static)
  for (int oy = 0; oy < h; ++oy) {
    std::vector<float> acc(static_cast<std::size_t>(c));
    for (int ox = 0; ox < wd; ++ox) {
      std::fill(acc.begin(), acc.end(), 0.0f);
      for (int ky = 0; ky < kernel; ++ky) {
        const int iy = oy + ky - pad;
        if (iy < 0 || iy >= h) continue;
        for (int kx = 0; kx < kernel; ++kx) {
          const int ix = ox + kx - pad;
          if (ix < 0 || ix >= wd) continue;
          const float* xp = x + (static_cast<std::size_t>(iy) * wd + ix) * c;
          const float* wp = wt.data() + static_cast<std::size_t>(ky * kernel + kx) * c;
#pragma omp simd
          for (int ch = 0; ch < c; ++ch) acc[ch] += xp[ch] * wp[ch];
        }
      }
      std::memcpy(y + (static_cast<std::size_t>(oy) * wd + ox) * c, acc.data(), sizeof(float) * c);
    }
  }
}

void ConvToNhwc(const float* x, bool chw_input, int h, int w, int cin, const float* weight,
                const float* bias, int cout, int kernel, int stride, int pad, float* y,
                int* out_h, int* out_w) {
  const int oh = (h + 2 * pad - kernel) / stride + 1;
  const int ow = (w + 2 * pad - kernel) / stride + 1;
  const int kdim = cin * kernel * kernel;
  // Patches [oh*ow, cin*k*k] in weight order (in, ky, kx).
  std::vector<float> cols(static_cast<std::size_t>(oh) * ow * kdim);
#pragma omp parallel for schedule(static)
  for (int oy = 0; oy < oh; ++oy) {
    for (int ox = 0; ox < ow; ++ox) {
      float* dst = cols.data() + (static_cast<std::size_t>(oy) * ow + ox) * kdim;
      for (int ci = 0; ci < cin; ++ci) {
        for (int ky = 0; ky < kernel; ++ky) {
          const int iy = oy * stride + ky - pad;
          for (int kx = 0; kx < kernel; ++kx) {
            const int ix = ox * stride + kx - pad;
            float v = 0.0f;
            if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
              v = chw_input ? x[(static_cast<std::size_t>(ci) * h + iy) * w + ix]
                            : x[(static_cast<std::size_t>(iy) * w + ix) * cin + ci];
            }
            *dst++ = v;
          }
        }
      }
    }
  }
  GemmBT(cols.data(), weight, bias, y, oh * ow, cout, kdim);
  *out_h = oh;
  *out_w = ow;
}

}  // namespace gufo::models::ccip::cpu
