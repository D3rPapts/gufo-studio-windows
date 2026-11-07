#include "src/models/indextts/bigvgan.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gufo::models::indextts {
namespace {

constexpr int kUpRates[6] = {4, 4, 2, 2, 2, 2};
constexpr int kUpKernels[6] = {8, 8, 4, 4, 4, 4};
constexpr int kResKernels[3] = {3, 7, 11};
constexpr int kDilations[3] = {1, 3, 5};
constexpr std::int64_t kInitial = 1536;

BigVgan::Act LoadAct(const PthArchive& sd, const std::string& p, std::int64_t ch) {
  BigVgan::Act a;
  const float* alpha = sd.F32(p + ".act.alpha", ch);
  const float* beta = sd.F32(p + ".act.beta", ch);
  a.alpha.resize(static_cast<std::size_t>(ch));
  a.inv_beta.resize(static_cast<std::size_t>(ch));
  for (std::int64_t c = 0; c < ch; ++c) {
    a.alpha[static_cast<std::size_t>(c)] = std::exp(alpha[c]);
    a.inv_beta[static_cast<std::size_t>(c)] = 1.0F / (std::exp(beta[c]) + 0.000000001F);
  }
  std::copy_n(sd.F32(p + ".upsample.filter", 12), 12, a.up);
  std::copy_n(sd.F32(p + ".downsample.lowpass.filter", 12), 12, a.down);
  return a;
}

// Activation1d: 2x Kaiser-sinc upsample (replicate pad 5, crop 15/15),
// SnakeBeta, lowpass + 2x downsample (replicate pad 5/6). x [t, c] in place.
void RunAct(const BigVgan::Act& a, std::vector<float>* xv, std::int64_t t, std::int64_t c) {
  float* x = xv->data();
#pragma omp parallel
  {
    std::vector<float> col(static_cast<std::size_t>(t + 10));
    std::vector<float> up(static_cast<std::size_t>(2 * t + 11));
#pragma omp for schedule(static)
    for (std::int64_t ch = 0; ch < c; ++ch) {
      for (std::int64_t i = 0; i < t + 10; ++i) {
        const std::int64_t src = std::clamp<std::int64_t>(i - 5, 0, t - 1);
        col[static_cast<std::size_t>(i)] = x[src * c + ch];
      }
      const float al = a.alpha[static_cast<std::size_t>(ch)];
      const float ib = a.inv_beta[static_cast<std::size_t>(ch)];
      // upsampled (cropped) sequence of length 2t, stored with 5 left / 6 right replicate pad
      for (std::int64_t m = 0; m < 2 * t; ++m) {
        const std::int64_t n = m + 15;
        float s = 0.0F;
        const std::int64_t i_lo = std::max<std::int64_t>(0, (n - 11 + 1) / 2);
        const std::int64_t i_hi = std::min<std::int64_t>(t + 9, n / 2);
        for (std::int64_t i = i_lo; i <= i_hi; ++i) {
          const std::int64_t k = n - 2 * i;
          if (k >= 0 && k < 12) {
            s += col[static_cast<std::size_t>(i)] * a.up[k];
          }
        }
        float v = 2.0F * s;
        const float sn = std::sin(v * al);
        v = v + ib * (sn * sn);
        up[static_cast<std::size_t>(m + 5)] = v;
      }
      for (int i = 0; i < 5; ++i) {
        up[static_cast<std::size_t>(i)] = up[5];
      }
      for (int i = 0; i < 6; ++i) {
        up[static_cast<std::size_t>(2 * t + 5 + i)] = up[static_cast<std::size_t>(2 * t + 4)];
      }
      for (std::int64_t j = 0; j < t; ++j) {
        float s = 0.0F;
        const float* z = up.data() + 2 * j;
        for (int k = 0; k < 12; ++k) {
          s += z[k] * a.down[k];
        }
        x[j * c + ch] = s;
      }
    }
  }
}

}  // namespace

std::unique_ptr<BigVgan> BigVgan::Load(const std::filesystem::path& dir, std::string* error) {
  auto sd = PthArchive::Open(dir / "bigvgan_generator.pt", error);
  if (sd == nullptr) {
    return nullptr;
  }
  auto out = std::unique_ptr<BigVgan>(new BigVgan());
  try {
    const std::string g = "generator.";
    {
      const std::vector<float> w = FoldWeightNorm(*sd, g + "conv_pre");
      out->pre_.Set(w.data(), sd->F32(g + "conv_pre.bias", kInitial), 80, kInitial, 7, 1, 3, 3);
    }
    std::int64_t ch = kInitial;
    for (int i = 0; i < 6; ++i) {
      const std::string up = g + "ups." + std::to_string(i) + ".0";
      const std::vector<float> w = FoldWeightNorm(*sd, up);
      cpu::ConvTranspose1dTM ct;
      ct.Set(w.data(), sd->F32(up + ".bias", ch / 2), ch, ch / 2, kUpKernels[i], kUpRates[i],
             (kUpKernels[i] - kUpRates[i]) / 2);
      out->ups_.push_back(std::move(ct));
      ch /= 2;
      for (int j = 0; j < 3; ++j) {
        const int idx = i * 3 + j;
        const std::string rp = g + "resblocks." + std::to_string(idx);
        ResBlock rb;
        const int k = kResKernels[j];
        for (int d = 0; d < 3; ++d) {
          const int dil = kDilations[d];
          const std::string c1 = rp + ".convs1." + std::to_string(d);
          const std::vector<float> w1 = FoldWeightNorm(*sd, c1);
          cpu::Conv1dTM conv1;
          conv1.Set(w1.data(), sd->F32(c1 + ".bias", ch), ch, ch, k, dil, (k * dil - dil) / 2,
                    (k * dil - dil) / 2);
          rb.c1.push_back(std::move(conv1));
          const std::string c2 = rp + ".convs2." + std::to_string(d);
          const std::vector<float> w2 = FoldWeightNorm(*sd, c2);
          cpu::Conv1dTM conv2;
          conv2.Set(w2.data(), sd->F32(c2 + ".bias", ch), ch, ch, k, 1, (k - 1) / 2, (k - 1) / 2);
          rb.c2.push_back(std::move(conv2));
        }
        for (int a = 0; a < 6; ++a) {
          rb.acts.push_back(LoadAct(*sd, rp + ".activations." + std::to_string(a), ch));
        }
        out->blocks_.push_back(std::move(rb));
      }
    }
    out->post_act_ = LoadAct(*sd, g + "activation_post", ch);
    const std::vector<float> wp = FoldWeightNorm(*sd, g + "conv_post");
    out->post_.Set(wp.data(), nullptr, ch, 1, 7, 1, 3, 3);
  } catch (const std::exception& e) {
    if (error != nullptr) {
      *error = "BigVGAN: " + std::string(e.what());
    }
    return nullptr;
  }
  return out;
}

std::vector<float> BigVgan::Synthesize(const float* mel, std::int64_t frames) const {
  std::vector<float> x;
  pre_.Run(mel, frames, &x);
  std::int64_t t = frames;
  std::int64_t ch = kInitial;
  std::vector<float> y;
  std::vector<float> xt;
  std::vector<float> tmp;
  for (int i = 0; i < 6; ++i) {
    std::int64_t t2 = 0;
    ups_[static_cast<std::size_t>(i)].Run(x.data(), t, &y, &t2);
    x.swap(y);
    t = t2;
    ch /= 2;
    std::vector<float> acc(x.size(), 0.0F);
    for (int j = 0; j < 3; ++j) {
      const ResBlock& rb = blocks_[static_cast<std::size_t>(i * 3 + j)];
      std::vector<float> h = x;
      for (int d = 0; d < 3; ++d) {
        xt = h;
        RunAct(rb.acts[static_cast<std::size_t>(2 * d)], &xt, t, ch);
        rb.c1[static_cast<std::size_t>(d)].Run(xt.data(), t, &tmp);
        RunAct(rb.acts[static_cast<std::size_t>(2 * d + 1)], &tmp, t, ch);
        rb.c2[static_cast<std::size_t>(d)].Run(tmp.data(), t, &xt);
        for (std::size_t q = 0; q < h.size(); ++q) {
          h[q] = xt[q] + h[q];
        }
      }
      for (std::size_t q = 0; q < acc.size(); ++q) {
        acc[q] += h[q];
      }
    }
    for (std::size_t q = 0; q < acc.size(); ++q) {
      x[q] = acc[q] / 3.0F;
    }
  }
  RunAct(post_act_, &x, t, ch);
  post_.Run(x.data(), t, &y);
  for (float& v : y) {
    v = std::clamp(v, -1.0F, 1.0F);
  }
  return y;
}

}  // namespace gufo::models::indextts
