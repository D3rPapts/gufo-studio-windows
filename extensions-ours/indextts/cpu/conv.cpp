#include "src/models/indextts/cpu/conv.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "src/models/indextts/cpu/ops.hpp"

namespace gufo::models::indextts::cpu {

void Conv1dTM::Set(const float* weight, const float* b, std::int64_t in, std::int64_t out,
                   std::int64_t k, std::int64_t dil, std::int64_t left, std::int64_t right,
                   PadMode m) {
  cin = in;
  cout = out;
  kernel = k;
  dilation = dil;
  pad_left = left;
  pad_right = right;
  mode = m;
  taps.resize(static_cast<std::size_t>(k * out * in));
  for (std::int64_t o = 0; o < out; ++o) {
    for (std::int64_t i = 0; i < in; ++i) {
      for (std::int64_t j = 0; j < k; ++j) {
        taps[static_cast<std::size_t>((j * out + o) * in + i)] = weight[(o * in + i) * k + j];
      }
    }
  }
  if (b != nullptr) {
    bias.assign(b, b + out);
  } else {
    bias.clear();
  }
}

void Conv1dTM::Run(const float* x, std::int64_t t, std::vector<float>* y) const {
  const std::int64_t tp = t + pad_left + pad_right;
  const std::int64_t span = dilation * (kernel - 1) + 1;
  const std::int64_t to = tp - span + 1;
  if (to <= 0) {
    throw std::runtime_error("conv input too short");
  }
  // padded input [tp, cin]
  std::vector<float> xp(static_cast<std::size_t>(tp * cin), 0.0F);
  for (std::int64_t r = 0; r < tp; ++r) {
    std::int64_t src = r - pad_left;
    if (src < 0 || src >= t) {
      if (mode == PadMode::kZero) {
        continue;
      }
      if (mode == PadMode::kReflect) {
        src = src < 0 ? -src : 2 * (t - 1) - src;
      } else {
        src = src < 0 ? 0 : t - 1;
      }
    }
    std::copy_n(x + src * cin, cin, xp.data() + r * cin);
  }
  y->assign(static_cast<std::size_t>(to * cout), 0.0F);
  std::vector<float> tmp(static_cast<std::size_t>(to * cout));
  for (std::int64_t j = 0; j < kernel; ++j) {
    Linear(xp.data() + j * dilation * cin, to, cin, taps.data() + j * cout * cin, cout, nullptr,
           j == 0 ? y->data() : tmp.data());
    if (j > 0) {
      float* yd = y->data();
#pragma omp parallel for schedule(static) if (to * cout > 65536)
      for (std::int64_t i = 0; i < to * cout; ++i) {
        yd[i] += tmp[static_cast<std::size_t>(i)];
      }
    }
  }
  if (!bias.empty()) {
    for (std::int64_t r = 0; r < to; ++r) {
      float* row = y->data() + r * cout;
      for (std::int64_t c = 0; c < cout; ++c) {
        row[c] += bias[static_cast<std::size_t>(c)];
      }
    }
  }
}

void ConvTranspose1dTM::Set(const float* weight, const float* b, std::int64_t in, std::int64_t out,
                            std::int64_t k, std::int64_t s, std::int64_t p) {
  cin = in;
  cout = out;
  kernel = k;
  stride = s;
  padding = p;
  taps.resize(static_cast<std::size_t>(k * out * in));
  for (std::int64_t i = 0; i < in; ++i) {
    for (std::int64_t o = 0; o < out; ++o) {
      for (std::int64_t j = 0; j < k; ++j) {
        taps[static_cast<std::size_t>((j * out + o) * in + i)] = weight[(i * out + o) * k + j];
      }
    }
  }
  if (b != nullptr) {
    bias.assign(b, b + out);
  } else {
    bias.clear();
  }
}

void ConvTranspose1dTM::Run(const float* x, std::int64_t t, std::vector<float>* y,
                            std::int64_t* t_out) const {
  const std::int64_t to = (t - 1) * stride - 2 * padding + kernel;
  *t_out = to;
  y->assign(static_cast<std::size_t>(to * cout), 0.0F);
  std::vector<float> z(static_cast<std::size_t>(t * cout));
  for (std::int64_t j = 0; j < kernel; ++j) {
    Linear(x, t, cin, taps.data() + j * cout * cin, cout, nullptr, z.data());
    float* yd = y->data();
#pragma omp parallel for schedule(static) if (t * cout > 65536)
    for (std::int64_t i = 0; i < t; ++i) {
      const std::int64_t n = i * stride + j - padding;
      if (n < 0 || n >= to) {
        continue;
      }
      const float* zr = z.data() + i * cout;
      float* yr = yd + n * cout;
      for (std::int64_t c = 0; c < cout; ++c) {
        yr[c] += zr[c];
      }
    }
  }
  if (!bias.empty()) {
    for (std::int64_t r = 0; r < to; ++r) {
      float* row = y->data() + r * cout;
      for (std::int64_t c = 0; c < cout; ++c) {
        row[c] += bias[static_cast<std::size_t>(c)];
      }
    }
  }
}

void DepthwiseTM(const float* x, std::int64_t t, std::int64_t c, const float* w, const float* b,
                 std::int64_t k, std::int64_t pad_left, float* y) {
#pragma omp parallel for schedule(static) if (t * c * k > 65536)
  for (std::int64_t r = 0; r < t; ++r) {
    float* o = y + r * c;
    for (std::int64_t ch = 0; ch < c; ++ch) {
      o[ch] = b != nullptr ? b[ch] : 0.0F;
    }
    for (std::int64_t j = 0; j < k; ++j) {
      const std::int64_t src = r - pad_left + j;
      if (src < 0 || src >= t) {
        continue;
      }
      const float* xi = x + src * c;
      for (std::int64_t ch = 0; ch < c; ++ch) {
        o[ch] += w[ch * k + j] * xi[ch];
      }
    }
  }
}

void GroupNorm1TM(float* x, std::int64_t t, std::int64_t c, const float* gamma, const float* beta,
                  float eps) {
  double sum = 0.0;
  const std::int64_t n = t * c;
  for (std::int64_t i = 0; i < n; ++i) {
    sum += x[i];
  }
  const double mean = sum / static_cast<double>(n);
  double sq = 0.0;
  for (std::int64_t i = 0; i < n; ++i) {
    const double d = x[i] - mean;
    sq += d * d;
  }
  const auto inv = static_cast<float>(1.0 / std::sqrt(sq / static_cast<double>(n) + eps));
  const auto m = static_cast<float>(mean);
  for (std::int64_t r = 0; r < t; ++r) {
    float* row = x + r * c;
    for (std::int64_t ch = 0; ch < c; ++ch) {
      row[ch] = (row[ch] - m) * inv * gamma[ch] + beta[ch];
    }
  }
}

}  // namespace gufo::models::indextts::cpu
