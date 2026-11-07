#include "ming_tts/cpu/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define GUFO_MING_AVX2 1
#endif

namespace gufo::models::ming_tts::cpu {
namespace {

inline float Bf16(std::uint16_t bits) {
  const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16U;
  float value;
  std::memcpy(&value, &widened, sizeof(value));
  return value;
}

#if defined(GUFO_MING_AVX2)
inline __m256 LoadBf16x8(const std::uint16_t* src) {
  const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
  const __m256i wide = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16);
  return _mm256_castsi256_ps(wide);
}

inline float HorizontalSum(__m256 v) {
  const __m128 low = _mm256_castps256_ps128(v);
  const __m128 high = _mm256_extractf128_ps(v, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55));
  return _mm_cvtss_f32(sum);
}
#endif

// Computes y[r, j0..j0+jn) for rows [r0, r0+rn) with jn <= 4, rn <= 2.
void Tile(const Linear& layer, const float* x, std::int64_t r0, std::int64_t rn,
          std::int64_t j0, std::int64_t jn, float* y) {
  const std::int64_t in = layer.in_features;
  const std::int64_t out = layer.out_features;
  float acc[2][4] = {};
  std::int64_t k = 0;
#if defined(GUFO_MING_AVX2)
  if (jn == 4 && rn == 2) {
    __m256 a00 = _mm256_setzero_ps(), a01 = a00, a02 = a00, a03 = a00;
    __m256 a10 = a00, a11 = a00, a12 = a00, a13 = a00;
    const float* x0 = x + r0 * in;
    const float* x1 = x0 + in;
    const std::uint16_t* w0 = layer.weight + j0 * in;
    const std::uint16_t* w1 = w0 + in;
    const std::uint16_t* w2 = w1 + in;
    const std::uint16_t* w3 = w2 + in;
    for (; k + 8 <= in; k += 8) {
      const __m256 v0 = _mm256_loadu_ps(x0 + k);
      const __m256 v1 = _mm256_loadu_ps(x1 + k);
      const __m256 b0 = LoadBf16x8(w0 + k);
      const __m256 b1 = LoadBf16x8(w1 + k);
      const __m256 b2 = LoadBf16x8(w2 + k);
      const __m256 b3 = LoadBf16x8(w3 + k);
      a00 = _mm256_fmadd_ps(v0, b0, a00);
      a01 = _mm256_fmadd_ps(v0, b1, a01);
      a02 = _mm256_fmadd_ps(v0, b2, a02);
      a03 = _mm256_fmadd_ps(v0, b3, a03);
      a10 = _mm256_fmadd_ps(v1, b0, a10);
      a11 = _mm256_fmadd_ps(v1, b1, a11);
      a12 = _mm256_fmadd_ps(v1, b2, a12);
      a13 = _mm256_fmadd_ps(v1, b3, a13);
    }
    acc[0][0] = HorizontalSum(a00);
    acc[0][1] = HorizontalSum(a01);
    acc[0][2] = HorizontalSum(a02);
    acc[0][3] = HorizontalSum(a03);
    acc[1][0] = HorizontalSum(a10);
    acc[1][1] = HorizontalSum(a11);
    acc[1][2] = HorizontalSum(a12);
    acc[1][3] = HorizontalSum(a13);
  } else {
    for (std::int64_t r = 0; r < rn; ++r) {
      for (std::int64_t j = 0; j < jn; ++j) {
        __m256 a = _mm256_setzero_ps();
        const float* xr = x + (r0 + r) * in;
        const std::uint16_t* wj = layer.weight + (j0 + j) * in;
        std::int64_t kk = 0;
        for (; kk + 8 <= in; kk += 8) {
          a = _mm256_fmadd_ps(_mm256_loadu_ps(xr + kk), LoadBf16x8(wj + kk), a);
        }
        acc[r][j] = HorizontalSum(a);
      }
    }
    k = in - (in % 8);
  }
#endif
  for (std::int64_t r = 0; r < rn; ++r) {
    const float* xr = x + (r0 + r) * in;
    for (std::int64_t j = 0; j < jn; ++j) {
      const std::uint16_t* wj = layer.weight + (j0 + j) * in;
      float sum = acc[r][j];
      for (std::int64_t kk = k; kk < in; ++kk) {
        sum += xr[kk] * Bf16(wj[kk]);
      }
      const float bias = layer.bias.empty()
                             ? 0.0F
                             : layer.bias[static_cast<std::size_t>(j0 + j)];
      y[(r0 + r) * out + j0 + j] = sum + bias;
    }
  }
}

#if defined(GUFO_MING_AVX2)
// R rows (3..6) x 2 output columns: every widened weight vector feeds R FMAs,
// which keeps the small-batch DiT/aggregator GEMMs off the conversion limit.
template <int R>
void TallTile(const Linear& layer, const float* x, std::int64_t r0,
              std::int64_t j0, float* y) {
  const std::int64_t in = layer.in_features;
  const std::int64_t out = layer.out_features;
  const std::uint16_t* w0 = layer.weight + j0 * in;
  const std::uint16_t* w1 = w0 + in;
  __m256 acc0[R];
  __m256 acc1[R];
  for (int r = 0; r < R; ++r) {
    acc0[r] = _mm256_setzero_ps();
    acc1[r] = _mm256_setzero_ps();
  }
  for (std::int64_t k = 0; k < in; k += 8) {
    const __m256 b0 = LoadBf16x8(w0 + k);
    const __m256 b1 = LoadBf16x8(w1 + k);
    for (int r = 0; r < R; ++r) {
      const __m256 v = _mm256_loadu_ps(x + (r0 + r) * in + k);
      acc0[r] = _mm256_fmadd_ps(v, b0, acc0[r]);
      acc1[r] = _mm256_fmadd_ps(v, b1, acc1[r]);
    }
  }
  const float bias0 = layer.bias.empty() ? 0.0F : layer.bias[static_cast<std::size_t>(j0)];
  const float bias1 = layer.bias.empty() ? 0.0F : layer.bias[static_cast<std::size_t>(j0 + 1)];
  for (int r = 0; r < R; ++r) {
    y[(r0 + r) * out + j0] = HorizontalSum(acc0[r]) + bias0;
    y[(r0 + r) * out + j0 + 1] = HorizontalSum(acc1[r]) + bias1;
  }
}

bool TryTallTile(const Linear& layer, const float* x, std::int64_t r0,
                 std::int64_t rn, std::int64_t j0, float* y) {
  if (layer.in_features % 8 != 0) {
    return false;
  }
  switch (rn) {
    case 3:
      TallTile<3>(layer, x, r0, j0, y);
      return true;
    case 4:
      TallTile<4>(layer, x, r0, j0, y);
      return true;
    case 5:
      TallTile<5>(layer, x, r0, j0, y);
      return true;
    case 6:
      TallTile<6>(layer, x, r0, j0, y);
      return true;
    default:
      return false;
  }
}
#endif

}  // namespace

namespace {
bool g_bf16_activations = false;
}  // namespace

void SetBf16ActivationEmulation(bool enabled) { g_bf16_activations = enabled; }

void LinearForward(const Linear& layer, const float* x, std::int64_t rows,
                   float* y) {
  std::vector<float> rounded;
  if (g_bf16_activations && rows > 8) {
    rounded.assign(x, x + rows * layer.in_features);
    for (float& value : rounded) {
      std::uint32_t bits;
      std::memcpy(&bits, &value, 4);
      bits = (bits + 0x7FFFU + ((bits >> 16U) & 1U)) & 0xFFFF0000U;  // RNE
      std::memcpy(&value, &bits, 4);
    }
    x = rounded.data();
  }
  const std::int64_t out = layer.out_features;
  const std::int64_t column_blocks = (out + 3) / 4;
  // Rows are processed in panels so a panel of activations stays in cache
  // while every weight row streams past it once.
  constexpr std::int64_t kPanel = 64;
  for (std::int64_t panel = 0; panel < rows; panel += kPanel) {
    const std::int64_t panel_rows = std::min(kPanel, rows - panel);
#pragma omp parallel for schedule(static) if (column_blocks * panel_rows >= 64)
    for (std::int64_t block = 0; block < column_blocks; ++block) {
      const std::int64_t j0 = block * 4;
      const std::int64_t jn = std::min<std::int64_t>(4, out - j0);
#if defined(GUFO_MING_AVX2)
      if (jn == 4 && panel_rows >= 3 && panel_rows <= 6 &&
          TryTallTile(layer, x, panel, panel_rows, j0, y)) {
        (void)TryTallTile(layer, x, panel, panel_rows, j0 + 2, y);
        continue;
      }
#endif
      for (std::int64_t r = 0; r < panel_rows; r += 2) {
        const std::int64_t rn = std::min<std::int64_t>(2, panel_rows - r);
        Tile(layer, x, panel + r, rn, j0, jn, y);
      }
    }
  }
}

void RmsNormTo(const float* x, float* out, std::int64_t rows, std::int64_t dim,
               std::span<const float> weight, float eps) {
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* xr = x + r * dim;
    float* yr = out + r * dim;
    double sum = 0.0;
    for (std::int64_t i = 0; i < dim; ++i) {
      sum += static_cast<double>(xr[i]) * xr[i];
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(sum / dim) + eps);
    for (std::int64_t i = 0; i < dim; ++i) {
      yr[i] = weight[static_cast<std::size_t>(i)] * (xr[i] * scale);
    }
  }
}

void RmsNorm(float* x, std::int64_t rows, std::int64_t dim,
             std::span<const float> weight, float eps) {
  RmsNormTo(x, x, rows, dim, weight, eps);
}

void SiluMulInPlace(float* gate, const float* up, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    const float g = gate[i];
    gate[i] = g / (1.0F + std::exp(-g)) * up[i];
  }
}

void GeluTanhInPlace(float* x, std::size_t count) {
  constexpr float kAlpha = 0.7978845608028654F;  // sqrt(2 / pi)
  for (std::size_t i = 0; i < count; ++i) {
    const float v = x[i];
    x[i] = 0.5F * v * (1.0F + std::tanh(kAlpha * (v + 0.044715F * v * v * v)));
  }
}

void Attention(const float* q, const float* k, const float* v, float* out,
               std::int64_t q_len, std::int64_t kv_len, std::int64_t heads,
               std::int64_t kv_heads, std::int64_t head_dim, bool causal,
               std::int64_t q_offset, std::int64_t window) {
  const std::int64_t group = heads / kv_heads;
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
#pragma omp parallel if (q_len * heads >= 8)
  {
    std::vector<float> scores(static_cast<std::size_t>(kv_len));
#pragma omp for schedule(static) collapse(2)
    for (std::int64_t i = 0; i < q_len; ++i) {
      for (std::int64_t h = 0; h < heads; ++h) {
        const std::int64_t position = q_offset + i;
        std::int64_t last = kv_len - 1;
        std::int64_t first = 0;
        if (causal) {
          last = std::min(last, position);
        }
        if (window > 0) {
          first = std::max<std::int64_t>(0, position - window + 1);
        }
        const float* qi = q + (i * heads + h) * head_dim;
        const std::int64_t kvh = h / group;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::int64_t j = first; j <= last; ++j) {
          const float* kj = k + (j * kv_heads + kvh) * head_dim;
          float dot = 0.0F;
          for (std::int64_t d = 0; d < head_dim; ++d) {
            dot += qi[d] * kj[d];
          }
          dot *= scale;
          scores[static_cast<std::size_t>(j)] = dot;
          maximum = std::max(maximum, dot);
        }
        float total = 0.0F;
        for (std::int64_t j = first; j <= last; ++j) {
          const float e = std::exp(scores[static_cast<std::size_t>(j)] - maximum);
          scores[static_cast<std::size_t>(j)] = e;
          total += e;
        }
        float* oi = out + (i * heads + h) * head_dim;
        std::fill(oi, oi + head_dim, 0.0F);
        const float inverse = 1.0F / total;
        for (std::int64_t j = first; j <= last; ++j) {
          const float p = scores[static_cast<std::size_t>(j)] * inverse;
          const float* vj = v + (j * kv_heads + kvh) * head_dim;
          for (std::int64_t d = 0; d < head_dim; ++d) {
            oi[d] += p * vj[d];
          }
        }
      }
    }
  }
}

void RopeHalf(float* x, std::int64_t len, std::int64_t heads,
              std::int64_t head_dim, std::int64_t first_position, float theta) {
  const std::int64_t half = head_dim / 2;
  std::vector<float> inv_freq(static_cast<std::size_t>(half));
  for (std::int64_t i = 0; i < half; ++i) {
    // transformers: 1 / theta ** (arange(0, dim, 2).float() / dim) in fp32
    const float exponent =
        static_cast<float>(2 * i) / static_cast<float>(head_dim);
    inv_freq[static_cast<std::size_t>(i)] =
        1.0F / std::pow(theta, exponent);
  }
  for (std::int64_t t = 0; t < len; ++t) {
    const auto position = static_cast<float>(first_position + t);
    for (std::int64_t i = 0; i < half; ++i) {
      const float angle = position * inv_freq[static_cast<std::size_t>(i)];
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      for (std::int64_t h = 0; h < heads; ++h) {
        float* row = x + (t * heads + h) * head_dim;
        const float a = row[i];
        const float b = row[i + half];
        row[i] = a * c - b * s;
        row[i + half] = b * c + a * s;
      }
    }
  }
}

void RopeInterleaved(float* x, std::int64_t len, std::int64_t heads,
                     std::int64_t head_dim, std::span<const float> inv_freq) {
  const std::int64_t pairs = head_dim / 2;
  for (std::int64_t t = 0; t < len; ++t) {
    for (std::int64_t i = 0; i < pairs; ++i) {
      const float angle =
          static_cast<float>(t) * inv_freq[static_cast<std::size_t>(i)];
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      for (std::int64_t h = 0; h < heads; ++h) {
        float* row = x + (t * heads + h) * head_dim;
        const float a = row[2 * i];
        const float b = row[2 * i + 1];
        row[2 * i] = a * c - b * s;
        row[2 * i + 1] = b * c + a * s;
      }
    }
  }
}

}  // namespace gufo::models::ming_tts::cpu
