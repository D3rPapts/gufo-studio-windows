#include "src/models/indextts/cpu/ops.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace gufo::models::indextts::cpu {
namespace {

inline float HSum(__m256 v) {
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}

inline float Dot(const float* a, const float* b, std::int64_t n) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  std::int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
  }
  for (; i + 8 <= n; i += 8) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
  }
  float s = HSum(_mm256_add_ps(acc0, acc1));
  for (; i < n; ++i) {
    s += a[i] * b[i];
  }
  return s;
}

// 4 rows x 4 outputs dot-product tile.
inline void Tile4x4(const float* x, std::int64_t in, const float* w, float* acc /*16*/) {
  __m256 a[4][4];
  for (auto& r : a) {
    for (auto& c : r) {
      c = _mm256_setzero_ps();
    }
  }
  std::int64_t i = 0;
  for (; i + 8 <= in; i += 8) {
    const __m256 w0 = _mm256_loadu_ps(w + i);
    const __m256 w1 = _mm256_loadu_ps(w + in + i);
    const __m256 w2 = _mm256_loadu_ps(w + 2 * in + i);
    const __m256 w3 = _mm256_loadu_ps(w + 3 * in + i);
    for (int r = 0; r < 4; ++r) {
      const __m256 xv = _mm256_loadu_ps(x + r * in + i);
      a[r][0] = _mm256_fmadd_ps(xv, w0, a[r][0]);
      a[r][1] = _mm256_fmadd_ps(xv, w1, a[r][1]);
      a[r][2] = _mm256_fmadd_ps(xv, w2, a[r][2]);
      a[r][3] = _mm256_fmadd_ps(xv, w3, a[r][3]);
    }
  }
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      float s = HSum(a[r][c]);
      for (std::int64_t j = i; j < in; ++j) {
        s += x[r * in + j] * w[c * in + j];
      }
      acc[r * 4 + c] = s;
    }
  }
}

}  // namespace

void Linear(const float* x, std::int64_t rows, std::int64_t in, const float* w,
            std::int64_t out, const float* bias, float* y) {
  const std::int64_t row_tiles = (rows + 3) / 4;
  const std::int64_t col_tiles = (out + 3) / 4;
#pragma omp parallel for schedule(static) collapse(2) if (rows * out * in > 65536)
  for (std::int64_t rt = 0; rt < row_tiles; ++rt) {
    for (std::int64_t ct = 0; ct < col_tiles; ++ct) {
      const std::int64_t r0 = rt * 4;
      const std::int64_t c0 = ct * 4;
      if (r0 + 4 <= rows && c0 + 4 <= out) {
        float acc[16];
        Tile4x4(x + r0 * in, in, w + c0 * in, acc);
        for (int r = 0; r < 4; ++r) {
          for (int c = 0; c < 4; ++c) {
            y[(r0 + r) * out + c0 + c] = acc[r * 4 + c] + (bias != nullptr ? bias[c0 + c] : 0.0F);
          }
        }
      } else {
        for (std::int64_t r = r0; r < std::min(rows, r0 + 4); ++r) {
          for (std::int64_t c = c0; c < std::min(out, c0 + 4); ++c) {
            y[r * out + c] = Dot(x + r * in, w + c * in, in) + (bias != nullptr ? bias[c] : 0.0F);
          }
        }
      }
    }
  }
}

void LinearInOut(const float* x, std::int64_t rows, std::int64_t in, const float* w,
                 std::int64_t out, const float* bias, float* y) {
  constexpr std::int64_t kChunk = 256;
  const std::int64_t chunks = (out + kChunk - 1) / kChunk;
#pragma omp parallel for schedule(static) if (rows * out * in > 65536)
  for (std::int64_t ch = 0; ch < chunks; ++ch) {
    const std::int64_t c0 = ch * kChunk;
    const std::int64_t c1 = std::min(out, c0 + kChunk);
    const std::int64_t width = c1 - c0;
    for (std::int64_t r = 0; r < rows; ++r) {
      float* yr = y + r * out + c0;
      if (bias != nullptr) {
        std::memcpy(yr, bias + c0, static_cast<std::size_t>(width) * 4);
      } else {
        std::fill(yr, yr + width, 0.0F);
      }
      const float* xr = x + r * in;
      for (std::int64_t i = 0; i < in; ++i) {
        const __m256 xv = _mm256_set1_ps(xr[i]);
        const float* wr = w + i * out + c0;
        std::int64_t c = 0;
        for (; c + 8 <= width; c += 8) {
          _mm256_storeu_ps(yr + c, _mm256_fmadd_ps(xv, _mm256_loadu_ps(wr + c), _mm256_loadu_ps(yr + c)));
        }
        for (; c < width; ++c) {
          yr[c] += xr[i] * wr[c];
        }
      }
    }
  }
}

void LayerNorm(const float* x, float* y, std::int64_t rows, std::int64_t dim, const float* gamma,
               const float* beta, float eps) {
#pragma omp parallel for schedule(static) if (rows > 16)
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* xr = x + r * dim;
    float* yr = y + r * dim;
    double mean = 0.0;
    for (std::int64_t i = 0; i < dim; ++i) {
      mean += xr[i];
    }
    mean /= static_cast<double>(dim);
    double var = 0.0;
    for (std::int64_t i = 0; i < dim; ++i) {
      const double d = xr[i] - mean;
      var += d * d;
    }
    var /= static_cast<double>(dim);
    const auto inv = static_cast<float>(1.0 / std::sqrt(var + eps));
    const auto m = static_cast<float>(mean);
    for (std::int64_t i = 0; i < dim; ++i) {
      float v = (xr[i] - m) * inv;
      if (gamma != nullptr) {
        v *= gamma[i];
      }
      if (beta != nullptr) {
        v += beta[i];
      }
      yr[i] = v;
    }
  }
}

void RmsNorm(const float* x, float* y, std::int64_t rows, std::int64_t dim, const float* w,
             float eps) {
#pragma omp parallel for schedule(static) if (rows > 16)
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* xr = x + r * dim;
    float* yr = y + r * dim;
    double ss = 0.0;
    for (std::int64_t i = 0; i < dim; ++i) {
      ss += static_cast<double>(xr[i]) * xr[i];
    }
    const auto inv = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(dim) + eps));
    for (std::int64_t i = 0; i < dim; ++i) {
      yr[i] = xr[i] * inv * (w != nullptr ? w[i] : 1.0F);
    }
  }
}

void Conv1d(const float* x, std::int64_t cin, std::int64_t t_in, const float* w,
            const float* bias, std::int64_t cout, std::int64_t kernel, std::int64_t stride,
            std::int64_t pad_left, std::int64_t pad_right, std::int64_t dilation,
            std::int64_t groups, std::vector<float>* y, std::int64_t* t_out) {
  const std::int64_t padded = t_in + pad_left + pad_right;
  const std::int64_t span = dilation * (kernel - 1) + 1;
  if (padded < span) {
    throw std::runtime_error("conv1d input too short");
  }
  const std::int64_t to = (padded - span) / stride + 1;
  *t_out = to;
  y->assign(static_cast<std::size_t>(cout * to), 0.0F);
  const std::int64_t cin_g = cin / groups;
  const std::int64_t cout_g = cout / groups;
  // kernel==1, stride 1, groups 1: plain GEMM over time: y[co, t] = sum_ci w[co,ci] x[ci,t]
  if (kernel == 1 && stride == 1 && groups == 1 && pad_left == 0 && pad_right == 0) {
    std::vector<float> xt = Transpose(x, cin, t_in);  // [t, cin]
    std::vector<float> yt(static_cast<std::size_t>(to * cout));
    Linear(xt.data(), t_in, cin, w, cout, bias, yt.data());
    std::vector<float> back = Transpose(yt.data(), to, cout);
    *y = std::move(back);
    return;
  }
#pragma omp parallel for schedule(dynamic) if (cout * to * cin_g * kernel > 65536)
  for (std::int64_t co = 0; co < cout; ++co) {
    const std::int64_t g = co / cout_g;
    float* out = y->data() + co * to;
    const float b = bias != nullptr ? bias[co] : 0.0F;
    std::fill(out, out + to, b);
    for (std::int64_t ci = 0; ci < cin_g; ++ci) {
      const float* in = x + (g * cin_g + ci) * t_in;
      const float* wk = w + (co * cin_g + ci) * kernel;
      for (std::int64_t k = 0; k < kernel; ++k) {
        const float wv = wk[k];
        const std::int64_t offset = k * dilation - pad_left;  // input index = ot*stride + offset
        if (stride == 1) {
          const std::int64_t lo = std::max<std::int64_t>(0, -offset);
          const std::int64_t hi = std::min<std::int64_t>(to, t_in - offset);
          for (std::int64_t ot = lo; ot < hi; ++ot) {
            out[ot] += wv * in[ot + offset];
          }
        } else {
          for (std::int64_t ot = 0; ot < to; ++ot) {
            const std::int64_t it = ot * stride + offset;
            if (it >= 0 && it < t_in) {
              out[ot] += wv * in[it];
            }
          }
        }
      }
    }
  }
}

std::vector<float> PadReflect(const float* x, std::int64_t c, std::int64_t t, std::int64_t left,
                              std::int64_t right) {
  const std::int64_t tt = t + left + right;
  std::vector<float> out(static_cast<std::size_t>(c * tt));
  for (std::int64_t ch = 0; ch < c; ++ch) {
    const float* in = x + ch * t;
    float* o = out.data() + ch * tt;
    for (std::int64_t i = 0; i < tt; ++i) {
      std::int64_t src = i - left;
      if (src < 0) {
        src = -src;
      } else if (src >= t) {
        src = 2 * (t - 1) - src;
      }
      o[i] = in[src];
    }
  }
  return out;
}

std::vector<float> Transpose(const float* x, std::int64_t rows, std::int64_t cols) {
  std::vector<float> out(static_cast<std::size_t>(rows * cols));
  constexpr std::int64_t kB = 32;
#pragma omp parallel for schedule(static) if (rows * cols > 65536)
  for (std::int64_t r0 = 0; r0 < rows; r0 += kB) {
    for (std::int64_t c0 = 0; c0 < cols; c0 += kB) {
      for (std::int64_t r = r0; r < std::min(rows, r0 + kB); ++r) {
        for (std::int64_t c = c0; c < std::min(cols, c0 + kB); ++c) {
          out[static_cast<std::size_t>(c * rows + r)] = x[r * cols + c];
        }
      }
    }
  }
  return out;
}

float GeluErf(float x) { return 0.5F * x * (1.0F + std::erf(x * 0.70710678118654752F)); }
float GeluTanh(float x) {
  return 0.5F * x * (1.0F + std::tanh(0.7978845608028654F * (x + 0.044715F * x * x * x)));
}
float Silu(float x) { return x / (1.0F + std::exp(-x)); }
float Sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }
float Mish(float x) {
  const float sp = x > 20.0F ? x : std::log1p(std::exp(x));
  return x * std::tanh(sp);
}
void GeluErfInPlace(float* x, std::size_t n) {
#pragma omp parallel for schedule(static) if (n > 65536)
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
    x[i] = GeluErf(x[i]);
  }
}
void GeluTanhInPlace(float* x, std::size_t n) {
#pragma omp parallel for schedule(static) if (n > 65536)
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
    x[i] = GeluTanh(x[i]);
  }
}
void SiluInPlace(float* x, std::size_t n) {
#pragma omp parallel for schedule(static) if (n > 65536)
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
    x[i] = Silu(x[i]);
  }
}
void ReluInPlace(float* x, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = std::max(x[i], 0.0F);
  }
}

void SoftmaxRows(float* x, std::int64_t rows, std::int64_t cols) {
#pragma omp parallel for schedule(static) if (rows > 16)
  for (std::int64_t r = 0; r < rows; ++r) {
    float* xr = x + r * cols;
    float m = -std::numeric_limits<float>::infinity();
    for (std::int64_t i = 0; i < cols; ++i) {
      m = std::max(m, xr[i]);
    }
    double sum = 0.0;
    for (std::int64_t i = 0; i < cols; ++i) {
      xr[i] = std::exp(xr[i] - m);
      sum += xr[i];
    }
    const auto inv = static_cast<float>(1.0 / sum);
    for (std::int64_t i = 0; i < cols; ++i) {
      xr[i] *= inv;
    }
  }
}

void Mha(const float* q, std::int64_t q_stride, const float* k, std::int64_t k_stride,
         const float* v, std::int64_t v_stride, float* out, std::int64_t out_stride,
         std::int64_t q_len, std::int64_t kv_len, std::int64_t heads, std::int64_t head_dim,
         float scale, bool causal, const float* bias) {
  const std::int64_t shift = kv_len - q_len;
#pragma omp parallel for schedule(dynamic) collapse(2) if (q_len * kv_len * heads > 4096)
  for (std::int64_t h = 0; h < heads; ++h) {
    for (std::int64_t i = 0; i < q_len; ++i) {
      std::vector<float> s(static_cast<std::size_t>(kv_len));
      const float* qi = q + i * q_stride + h * head_dim;
      const std::int64_t limit = causal ? std::min(kv_len, i + shift + 1) : kv_len;
      float m = -std::numeric_limits<float>::infinity();
      for (std::int64_t j = 0; j < limit; ++j) {
        float sc = Dot(qi, k + j * k_stride + h * head_dim, head_dim) * scale;
        if (bias != nullptr) {
          sc += bias[(h * q_len + i) * kv_len + j];
        }
        s[static_cast<std::size_t>(j)] = sc;
        m = std::max(m, sc);
      }
      double sum = 0.0;
      for (std::int64_t j = 0; j < limit; ++j) {
        const float e = std::exp(s[static_cast<std::size_t>(j)] - m);
        s[static_cast<std::size_t>(j)] = e;
        sum += e;
      }
      const auto inv = static_cast<float>(1.0 / sum);
      float* o = out + i * out_stride + h * head_dim;
      std::fill(o, o + head_dim, 0.0F);
      for (std::int64_t j = 0; j < limit; ++j) {
        const float p = s[static_cast<std::size_t>(j)] * inv;
        const float* vj = v + j * v_stride + h * head_dim;
        for (std::int64_t d = 0; d < head_dim; ++d) {
          o[d] += p * vj[d];
        }
      }
    }
  }
}

double Cosine(const float* a, const float* b, std::size_t n) {
  double ab = 0.0;
  double aa = 0.0;
  double bb = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    ab += static_cast<double>(a[i]) * b[i];
    aa += static_cast<double>(a[i]) * a[i];
    bb += static_cast<double>(b[i]) * b[i];
  }
  if (aa == 0.0 && bb == 0.0) {
    return 1.0;
  }
  return ab / std::sqrt(aa * bb + 1e-300);
}

double MaxAbsDiff(const float* a, const float* b, std::size_t n) {
  double m = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
  }
  return m;
}

}  // namespace gufo::models::indextts::cpu
