// YOLOv8n face detector (deepghs face_detect_v1.4_n), CPU fp32, NCHW.
// Topology (read off model.onnx; conv shapes/strides/pads come from the JSON):
//   0 Conv/2, 1 Conv/2, 2 C2f, 3 Conv/2, 4 C2f, 5 Conv/2, 6 C2f, 7 Conv/2, 8 C2f, 9 SPPF,
//   10 up(9), 11 cat(10,6), 12 C2f, 13 up(12), 14 cat(13,4), 15 C2f (P3),
//   16 Conv/2(15), 17 cat(16,12), 18 C2f (P4), 19 Conv/2(18), 20 cat(19,9), 21 C2f (P5),
//   22 Detect(15, 18, 21): box branch cv2.i (64 = 4 x 16 DFL bins), class branch cv3.i.
// Every conv has a fused bias; SiLU = x * sigmoid(x). Written with AI assistance (Claude Code).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <gufo/ext/json.hpp>

#include "ccip/ccip.hpp"
#include "ccip/cpu/kernels.hpp"
#include "ccip/tensors.hpp"

namespace gufo::models::ccip {
namespace {

struct Map {
  int c{0}, h{0}, w{0};
  std::vector<float> d;  // CHW
};

struct ConvW {
  const float* w{nullptr};
  const float* b{nullptr};
  int in{0}, out{0}, k{1}, s{1}, p{0};
  bool silu{false};
};

class CpuYolo final : public FaceDetector {
public:
  bool Init(const std::filesystem::path& dir, std::string* error) {
    if (!weights_.Load(dir / "face_yolov8n.safetensors", error)) return false;
    std::ifstream in(dir / "face_yolov8n.json", std::ios::binary);
    if (!in) {
      if (error) *error = "missing " + (dir / "face_yolov8n.json").string();
      return false;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    try {
      const json::Value arch = json::parse(ss.str());
      for (const auto& [name, cj] : arch.find("convs")->members()) {
        ConvW c;
        c.in = static_cast<int>(cj.member_size("inp"));
        c.out = static_cast<int>(cj.member_size("out"));
        c.k = static_cast<int>(cj.member_size("kernel"));
        c.s = static_cast<int>(cj.member_size("stride"));
        c.p = static_cast<int>(cj.member_size("pad"));
        c.silu = cj.find("silu")->as_bool();
        c.w = weights_.F32("model." + name + ".weight",
                           static_cast<std::int64_t>(c.out) * c.in * c.k * c.k).data();
        if (cj.find("bias")->as_bool()) c.b = weights_.F32("model." + name + ".bias", c.out).data();
        convs_.emplace(name, c);
      }
      for (const auto& [name, cj] : arch.find("c2f")->members()) {
        c2f_n_[std::stoi(name)] = static_cast<int>(cj.member_size("n"));
        c2f_shortcut_[std::stoi(name)] = cj.find("shortcut")->as_bool();
      }
      for (const auto& s : arch.find("strides")->items()) strides_.push_back(static_cast<float>(s.as_double()));
      reg_max_ = static_cast<int>(arch.member_size("reg_max", 16));
      classes_ = static_cast<int>(arch.find("classes")->size());
    } catch (const std::exception& e) {
      if (error) *error = std::string("face detector: ") + e.what();
      return false;
    }
    return true;
  }

  std::vector<float> Raw(const float* chw, int size, int* anchors) const override {
    Map x{3, size, size, std::vector<float>(chw, chw + static_cast<std::size_t>(3) * size * size)};
    Map m0 = Conv(x, "0.conv");
    Map m1 = Conv(m0, "1.conv");
    Map m2 = C2f(m1, 2);
    Map m3 = Conv(m2, "3.conv");
    Map m4 = C2f(m3, 4);
    Map m5 = Conv(m4, "5.conv");
    Map m6 = C2f(m5, 6);
    Map m7 = Conv(m6, "7.conv");
    Map m8 = C2f(m7, 8);
    Map m9 = Sppf(m8);
    Map m12 = C2f(Cat(Up(m9), m6), 12);
    Map m15 = C2f(Cat(Up(m12), m4), 15);
    Map m18 = C2f(Cat(Conv(m15, "16.conv"), m12), 18);
    Map m21 = C2f(Cat(Conv(m18, "19.conv"), m9), 21);
    const Map* levels[3] = {&m15, &m18, &m21};
    int total = 0;
    for (const Map* l : levels) total += l->h * l->w;
    const int rows = 4 + classes_;
    std::vector<float> out(static_cast<std::size_t>(rows) * total);
    int base = 0;
    for (int i = 0; i < 3; ++i) {
      const std::string si = std::to_string(i);
      Map box = Conv(Conv(Conv(*levels[i], "22.cv2." + si + ".0.conv"), "22.cv2." + si + ".1.conv"),
                     "22.cv2." + si + ".2");
      Map cls = Conv(Conv(Conv(*levels[i], "22.cv3." + si + ".0.conv"), "22.cv3." + si + ".1.conv"),
                     "22.cv3." + si + ".2");
      const int hw = box.h * box.w;
      const ConvW& dfl = convs_.at("22.dfl.conv");
      const float stride = strides_[static_cast<std::size_t>(i)];
      for (int y = 0; y < box.h; ++y) {
        for (int xx = 0; xx < box.w; ++xx) {
          const int a = y * box.w + xx;
          float dist[4];
          for (int side = 0; side < 4; ++side) {
            float logits[64];
            float mx = -std::numeric_limits<float>::infinity();
            for (int bin = 0; bin < reg_max_; ++bin) {
              logits[bin] = box.d[static_cast<std::size_t>(side * reg_max_ + bin) * hw + a];
              mx = std::max(mx, logits[bin]);
            }
            float sum = 0.0f;
            for (int bin = 0; bin < reg_max_; ++bin) {
              logits[bin] = std::exp(logits[bin] - mx);
              sum += logits[bin];
            }
            float e = dfl.b != nullptr ? dfl.b[0] : 0.0f;
            for (int bin = 0; bin < reg_max_; ++bin) e += dfl.w[bin] * (logits[bin] / sum);
            dist[side] = e;
          }
          const float ax = static_cast<float>(xx) + 0.5f;
          const float ay = static_cast<float>(y) + 0.5f;
          const float x1 = ax - dist[0], y1 = ay - dist[1];
          const float x2 = ax + dist[2], y2 = ay + dist[3];
          const std::size_t col = static_cast<std::size_t>(base + a);
          out[0 * static_cast<std::size_t>(total) + col] = (x1 + x2) / 2.0f * stride;
          out[1 * static_cast<std::size_t>(total) + col] = (y1 + y2) / 2.0f * stride;
          out[2 * static_cast<std::size_t>(total) + col] = (x2 - x1) * stride;
          out[3 * static_cast<std::size_t>(total) + col] = (y2 - y1) * stride;
          for (int k = 0; k < classes_; ++k) {
            const float v = cls.d[static_cast<std::size_t>(k) * hw + a];
            out[static_cast<std::size_t>(4 + k) * total + col] = 1.0f / (1.0f + std::exp(-v));
          }
        }
      }
      base += hw;
    }
    *anchors = total;
    return out;
  }

private:
  Map Conv(const Map& x, const std::string& name) const {
    const ConvW& c = convs_.at(name);
    if (c.in != x.c) throw std::runtime_error("channel mismatch at " + name);
    Map y;
    y.c = c.out;
    y.h = (x.h + 2 * c.p - c.k) / c.s + 1;
    y.w = (x.w + 2 * c.p - c.k) / c.s + 1;
    const int n = y.h * y.w;
    y.d.resize(static_cast<std::size_t>(c.out) * n);
    if (c.k == 1 && c.s == 1 && c.p == 0) {
      cpu::Gemm(c.w, x.d.data(), nullptr, y.d.data(), c.out, n, c.in);
    } else {
      const int kd = c.in * c.k * c.k;
      std::vector<float> cols(static_cast<std::size_t>(kd) * n);
#pragma omp parallel for schedule(static)
      for (int r = 0; r < kd; ++r) {
        const int ci = r / (c.k * c.k);
        const int ky = (r / c.k) % c.k;
        const int kx = r % c.k;
        float* dst = cols.data() + static_cast<std::size_t>(r) * n;
        for (int oy = 0; oy < y.h; ++oy) {
          const int iy = oy * c.s + ky - c.p;
          for (int ox = 0; ox < y.w; ++ox) {
            const int ix = ox * c.s + kx - c.p;
            dst[oy * y.w + ox] = (iy >= 0 && iy < x.h && ix >= 0 && ix < x.w)
                                     ? x.d[(static_cast<std::size_t>(ci) * x.h + iy) * x.w + ix]
                                     : 0.0f;
          }
        }
      }
      cpu::Gemm(c.w, cols.data(), nullptr, y.d.data(), c.out, n, kd);
    }
    const int cout = c.out;
    const float* bias = c.b;
    const bool silu = c.silu;
#pragma omp parallel for schedule(static)
    for (int o = 0; o < cout; ++o) {
      float* r = y.d.data() + static_cast<std::size_t>(o) * n;
      const float b = bias != nullptr ? bias[o] : 0.0f;
      for (int i = 0; i < n; ++i) {
        float v = r[i] + b;
        if (silu) v = v * (1.0f / (1.0f + std::exp(-v)));
        r[i] = v;
      }
    }
    return y;
  }

  static Map Cat(const Map& a, const Map& b) {
    if (a.h != b.h || a.w != b.w) throw std::runtime_error("concat size mismatch");
    Map y{a.c + b.c, a.h, a.w, a.d};
    y.d.insert(y.d.end(), b.d.begin(), b.d.end());
    return y;
  }

  static Map Slice(const Map& a, int c0, int c1) {
    const std::size_t hw = static_cast<std::size_t>(a.h) * a.w;
    Map y{c1 - c0, a.h, a.w, {}};
    y.d.assign(a.d.begin() + static_cast<std::ptrdiff_t>(c0 * hw),
               a.d.begin() + static_cast<std::ptrdiff_t>(c1 * hw));
    return y;
  }

  Map C2f(const Map& x, int idx) const {
    const std::string p = std::to_string(idx);
    Map t = Conv(x, p + ".cv1.conv");
    const int half = t.c / 2;
    Map out = t;  // chunks y0, y1
    Map last = Slice(t, half, t.c);
    for (int i = 0; i < c2f_n_.at(idx); ++i) {
      const std::string m = p + ".m." + std::to_string(i);
      Map b = Conv(Conv(last, m + ".cv1.conv"), m + ".cv2.conv");
      if (c2f_shortcut_.at(idx)) {
        for (std::size_t k = 0; k < b.d.size(); ++k) b.d[k] += last.d[k];
      }
      out = Cat(out, b);
      last = std::move(b);
    }
    return Conv(out, p + ".cv2.conv");
  }

  static Map MaxPool5(const Map& x) {
    Map y{x.c, x.h, x.w, std::vector<float>(x.d.size())};
#pragma omp parallel for schedule(static)
    for (int c = 0; c < x.c; ++c) {
      for (int oy = 0; oy < x.h; ++oy) {
        for (int ox = 0; ox < x.w; ++ox) {
          float m = -std::numeric_limits<float>::infinity();
          for (int ky = -2; ky <= 2; ++ky) {
            const int iy = oy + ky;
            if (iy < 0 || iy >= x.h) continue;
            for (int kx = -2; kx <= 2; ++kx) {
              const int ix = ox + kx;
              if (ix < 0 || ix >= x.w) continue;
              m = std::max(m, x.d[(static_cast<std::size_t>(c) * x.h + iy) * x.w + ix]);
            }
          }
          y.d[(static_cast<std::size_t>(c) * x.h + oy) * x.w + ox] = m;
        }
      }
    }
    return y;
  }

  Map Sppf(const Map& x) const {
    Map a = Conv(x, "9.cv1.conv");
    Map b = MaxPool5(a);
    Map c = MaxPool5(b);
    Map d = MaxPool5(c);
    return Conv(Cat(Cat(Cat(a, b), c), d), "9.cv2.conv");
  }

  static Map Up(const Map& x) {
    Map y{x.c, x.h * 2, x.w * 2, {}};
    y.d.resize(static_cast<std::size_t>(y.c) * y.h * y.w);
    for (int c = 0; c < y.c; ++c) {
      for (int oy = 0; oy < y.h; ++oy) {
        for (int ox = 0; ox < y.w; ++ox) {
          y.d[(static_cast<std::size_t>(c) * y.h + oy) * y.w + ox] =
              x.d[(static_cast<std::size_t>(c) * x.h + oy / 2) * x.w + ox / 2];
        }
      }
    }
    return y;
  }

  TensorFile weights_;
  std::map<std::string, ConvW> convs_;
  std::map<int, int> c2f_n_;
  std::map<int, bool> c2f_shortcut_;
  std::vector<float> strides_;
  int reg_max_{16};
  int classes_{1};
};

}  // namespace

std::unique_ptr<FaceDetector> FaceDetector::Load(const std::filesystem::path& dir,
                                                 std::string* error) {
  auto m = std::make_unique<CpuYolo>();
  if (!m->Init(dir, error)) return nullptr;
  return m;
}

std::vector<Face> DecodeFaces(const std::vector<float>& raw, int anchors,
                              const LetterboxInput& lb, int width, int height, float conf,
                              float iou) {
  struct Cand {
    std::array<float, 4> b;
    float s;
  };
  std::vector<Cand> cands;
  const auto r = static_cast<float>(lb.ratio);
  const auto dx = static_cast<float>(lb.dx);
  const auto dy = static_cast<float>(lb.dy);
  for (int a = 0; a < anchors; ++a) {
    const float s = raw[static_cast<std::size_t>(4) * anchors + a];
    if (!(s >= conf)) continue;
    const float cx = raw[a], cy = raw[static_cast<std::size_t>(anchors) + a];
    const float w = raw[static_cast<std::size_t>(2) * anchors + a];
    const float h = raw[static_cast<std::size_t>(3) * anchors + a];
    Cand c;
    c.b = {(cx - w / 2 - dx) / r, (cy - h / 2 - dy) / r, (cx + w / 2 - dx) / r, (cy + h / 2 - dy) / r};
    c.s = s;
    cands.push_back(c);
  }
  // Descending score (numpy argsort()[::-1]); ties keep the reversed index order.
  std::vector<int> order(cands.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int i, int j) { return cands[i].s < cands[j].s; });
  std::reverse(order.begin(), order.end());
  auto area = [](const std::array<float, 4>& b) { return (b[2] - b[0]) * (b[3] - b[1]); };
  std::vector<Face> faces;
  while (!order.empty()) {
    const int i = order.front();
    const auto& bi = cands[i].b;
    std::vector<int> rest;
    for (std::size_t t = 1; t < order.size(); ++t) {
      const auto& bj = cands[order[t]].b;
      const float iw = std::max(0.0f, std::min(bi[2], bj[2]) - std::max(bi[0], bj[0]));
      const float ih = std::max(0.0f, std::min(bi[3], bj[3]) - std::max(bi[1], bj[1]));
      const float inter = iw * ih;
      const float u = area(bi) + area(bj) - inter;
      if (inter / u < iou) rest.push_back(order[t]);
    }
    Face f;
    f.box = {std::max(0.0f, bi[0]), std::max(0.0f, bi[1]),
             std::min(static_cast<float>(width), bi[2]), std::min(static_cast<float>(height), bi[3])};
    f.score = cands[i].s;
    faces.push_back(f);
    order = std::move(rest);
  }
  return faces;
}

std::vector<Face> FaceDetector::Detect(const RgbImage& image, float conf, float iou) const {
  const LetterboxInput lb = Letterbox(image, 640);
  int anchors = 0;
  const std::vector<float> raw = Raw(lb.chw.data(), lb.size, &anchors);
  return DecodeFaces(raw, anchors, lb, image.width, image.height, conf, iou);
}

}  // namespace gufo::models::ccip
