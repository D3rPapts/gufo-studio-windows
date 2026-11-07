#include "indextts/campplus.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "indextts/cpu/ops.hpp"

namespace gufo::models::indextts {
namespace {

struct Map2d {  // [c, h, w]
  std::int64_t c{0}, h{0}, w{0};
  std::vector<float> d;
};

Map2d Conv2d(const Map2d& x, const float* weight, std::int64_t cout, std::int64_t k,
             std::int64_t stride_h, std::int64_t stride_w, std::int64_t pad) {
  Map2d y;
  y.c = cout;
  y.h = (x.h + 2 * pad - k) / stride_h + 1;
  y.w = (x.w + 2 * pad - k) / stride_w + 1;
  y.d.assign(static_cast<std::size_t>(y.c * y.h * y.w), 0.0F);
#pragma omp parallel for schedule(dynamic)
  for (std::int64_t oc = 0; oc < cout; ++oc) {
    for (std::int64_t oy = 0; oy < y.h; ++oy) {
      float* out = y.d.data() + (oc * y.h + oy) * y.w;
      for (std::int64_t ic = 0; ic < x.c; ++ic) {
        for (std::int64_t ky = 0; ky < k; ++ky) {
          const std::int64_t iy = oy * stride_h - pad + ky;
          if (iy < 0 || iy >= x.h) {
            continue;
          }
          const float* in = x.d.data() + (ic * x.h + iy) * x.w;
          for (std::int64_t kx = 0; kx < k; ++kx) {
            const float wv = weight[((oc * x.c + ic) * k + ky) * k + kx];
            const std::int64_t off = kx - pad;
            for (std::int64_t ox = 0; ox < y.w; ++ox) {
              const std::int64_t ix = ox * stride_w + off;
              if (ix >= 0 && ix < x.w) {
                out[ox] += wv * in[ix];
              }
            }
          }
        }
      }
    }
  }
  return y;
}

void BatchNorm(float* x, std::int64_t c, std::int64_t inner, const PthArchive& sd,
               const std::string& p, bool affine = true) {
  const float* mean = sd.F32(p + ".running_mean", c);
  const float* var = sd.F32(p + ".running_var", c);
  const float* w = affine ? sd.F32(p + ".weight", c) : nullptr;
  const float* b = affine ? sd.F32(p + ".bias", c) : nullptr;
  for (std::int64_t ch = 0; ch < c; ++ch) {
    const float inv = 1.0F / std::sqrt(var[ch] + 1e-5F);
    float* row = x + ch * inner;
    for (std::int64_t i = 0; i < inner; ++i) {
      float v = (row[i] - mean[ch]) * inv;
      if (affine) {
        v = v * w[ch] + b[ch];
      }
      row[i] = v;
    }
  }
}

void Relu(std::vector<float>* x) { cpu::ReluInPlace(x->data(), x->size()); }

Map2d ResBlock(const Map2d& x, const PthArchive& sd, const std::string& p, std::int64_t stride) {
  Map2d y = Conv2d(x, sd.F32(p + ".conv1.weight"), 32, 3, stride, 1, 1);
  BatchNorm(y.d.data(), y.c, y.h * y.w, sd, p + ".bn1");
  Relu(&y.d);
  y = Conv2d(y, sd.F32(p + ".conv2.weight"), 32, 3, 1, 1, 1);
  BatchNorm(y.d.data(), y.c, y.h * y.w, sd, p + ".bn2");
  if (sd.Has(p + ".shortcut.0.weight")) {
    Map2d s = Conv2d(x, sd.F32(p + ".shortcut.0.weight"), 32, 1, stride, 1, 0);
    BatchNorm(s.d.data(), s.c, s.h * s.w, sd, p + ".shortcut.1");
    for (std::size_t i = 0; i < y.d.size(); ++i) {
      y.d[i] += s.d[i];
    }
  } else {
    for (std::size_t i = 0; i < y.d.size(); ++i) {
      y.d[i] += x.d[i];
    }
  }
  Relu(&y.d);
  return y;
}

std::vector<float> Conv(const std::vector<float>& x, std::int64_t cin, std::int64_t t,
                        const PthArchive& sd, const std::string& p, std::int64_t cout,
                        std::int64_t k, std::int64_t stride, std::int64_t pad,
                        std::int64_t dilation, std::int64_t* t_out) {
  std::vector<float> y;
  const float* bias = sd.Has(p + ".bias") ? sd.F32(p + ".bias", cout) : nullptr;
  cpu::Conv1d(x.data(), cin, t, sd.F32(p + ".weight", cout * cin * k), bias, cout, k, stride, pad,
              pad, dilation, 1, &y, t_out);
  return y;
}

}  // namespace

std::unique_ptr<CamPlus> CamPlus::Load(const std::filesystem::path& bin, std::string* error) {
  auto sd = PthArchive::Open(bin, error);
  if (sd == nullptr) {
    return nullptr;
  }
  if (!sd->Has("head.conv1.weight") || !sd->Has("xvector.dense.linear.weight")) {
    if (error != nullptr) {
      *error = bin.string() + " is not a CAM++ state dict";
    }
    return nullptr;
  }
  auto out = std::unique_ptr<CamPlus>(new CamPlus());
  out->sd_ = std::move(sd);
  return out;
}

std::vector<float> CamPlus::Embed(std::span<const float> fbank, std::int64_t frames,
                                  std::vector<float>* head_out) const {
  const PthArchive& sd = *sd_;
  Map2d x;
  x.c = 1;
  x.h = 80;
  x.w = frames;
  x.d.resize(static_cast<std::size_t>(80 * frames));
  for (std::int64_t t = 0; t < frames; ++t) {
    for (std::int64_t f = 0; f < 80; ++f) {
      x.d[static_cast<std::size_t>(f * frames + t)] = fbank[static_cast<std::size_t>(t * 80 + f)];
    }
  }
  x = Conv2d(x, sd.F32("head.conv1.weight"), 32, 3, 1, 1, 1);
  BatchNorm(x.d.data(), x.c, x.h * x.w, sd, "head.bn1");
  Relu(&x.d);
  x = ResBlock(x, sd, "head.layer1.0", 2);
  x = ResBlock(x, sd, "head.layer1.1", 1);
  x = ResBlock(x, sd, "head.layer2.0", 2);
  x = ResBlock(x, sd, "head.layer2.1", 1);
  x = Conv2d(x, sd.F32("head.conv2.weight"), 32, 3, 2, 1, 1);
  BatchNorm(x.d.data(), x.c, x.h * x.w, sd, "head.bn2");
  Relu(&x.d);
  std::int64_t channels = x.c * x.h;  // 32 * 10
  std::int64_t t = x.w;
  std::vector<float> h = std::move(x.d);
  if (head_out != nullptr) {
    *head_out = h;
  }
  std::int64_t t2 = 0;
  h = Conv(h, channels, t, sd, "xvector.tdnn.linear", 128, 5, 2, 2, 1, &t2);
  t = t2;
  channels = 128;
  BatchNorm(h.data(), channels, t, sd, "xvector.tdnn.nonlinear.batchnorm");
  Relu(&h);
  const int layers[3] = {12, 24, 16};
  const int dilations[3] = {1, 2, 2};
  for (int b = 0; b < 3; ++b) {
    const std::string bp = "xvector.block" + std::to_string(b + 1);
    for (int l = 0; l < layers[b]; ++l) {
      const std::string lp = bp + ".tdnnd" + std::to_string(l + 1);
      std::vector<float> y = h;
      BatchNorm(y.data(), channels, t, sd, lp + ".nonlinear1.batchnorm");
      Relu(&y);
      std::int64_t tt = 0;
      y = Conv(y, channels, t, sd, lp + ".linear1", 128, 1, 1, 0, 1, &tt);
      BatchNorm(y.data(), 128, t, sd, lp + ".nonlinear2.batchnorm");
      Relu(&y);
      // CAMLayer
      std::vector<float> local =
          Conv(y, 128, t, sd, lp + ".cam_layer.linear_local", 32, 3, 1, dilations[b], dilations[b], &tt);
      std::vector<float> ctx(static_cast<std::size_t>(128 * t));
      for (std::int64_t c = 0; c < 128; ++c) {
        const float* in = y.data() + c * t;
        float* out = ctx.data() + c * t;
        float mean = 0.0F;
        {
          double s = 0.0;
          for (std::int64_t i = 0; i < t; ++i) {
            s += in[i];
          }
          mean = static_cast<float>(s / static_cast<double>(t));
        }
        for (std::int64_t s0 = 0; s0 < t; s0 += 100) {
          const std::int64_t e = std::min(t, s0 + 100);
          double s = 0.0;
          for (std::int64_t i = s0; i < e; ++i) {
            s += in[i];
          }
          const auto seg = static_cast<float>(s / static_cast<double>(e - s0));
          for (std::int64_t i = s0; i < e; ++i) {
            out[i] = mean + seg;
          }
        }
      }
      ctx = Conv(ctx, 128, t, sd, lp + ".cam_layer.linear1", 64, 1, 1, 0, 1, &tt);
      Relu(&ctx);
      ctx = Conv(ctx, 64, t, sd, lp + ".cam_layer.linear2", 32, 1, 1, 0, 1, &tt);
      for (std::size_t i = 0; i < local.size(); ++i) {
        local[i] *= cpu::Sigmoid(ctx[i]);
      }
      h.insert(h.end(), local.begin(), local.end());
      channels += 32;
    }
    const std::string tp = "xvector.transit" + std::to_string(b + 1);
    BatchNorm(h.data(), channels, t, sd, tp + ".nonlinear.batchnorm");
    Relu(&h);
    std::int64_t tt = 0;
    h = Conv(h, channels, t, sd, tp + ".linear", channels / 2, 1, 1, 0, 1, &tt);
    channels /= 2;
  }
  BatchNorm(h.data(), channels, t, sd, "xvector.out_nonlinear.batchnorm");
  Relu(&h);
  std::vector<float> stats(static_cast<std::size_t>(2 * channels));
  for (std::int64_t c = 0; c < channels; ++c) {
    const float* row = h.data() + c * t;
    double sum = 0.0;
    for (std::int64_t i = 0; i < t; ++i) {
      sum += row[i];
    }
    const double mean = sum / static_cast<double>(t);
    double sq = 0.0;
    for (std::int64_t i = 0; i < t; ++i) {
      sq += (row[i] - mean) * (row[i] - mean);
    }
    stats[static_cast<std::size_t>(c)] = static_cast<float>(mean);
    stats[static_cast<std::size_t>(channels + c)] =
        static_cast<float>(std::sqrt(sq / static_cast<double>(t - 1)));
  }
  std::vector<float> emb(192);
  cpu::Linear(stats.data(), 1, 2 * channels, sd.F32("xvector.dense.linear.weight", 192 * 2 * channels),
              192, nullptr, emb.data());
  BatchNorm(emb.data(), 192, 1, sd, "xvector.dense.nonlinear.batchnorm", false);
  return emb;
}

}  // namespace gufo::models::indextts
