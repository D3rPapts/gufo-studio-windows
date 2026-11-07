// Written with AI assistance (Claude Code).
#include "src/models/ccip/image_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace gufo::models::ccip {
namespace {

// Pillow libImaging/Resample.c (8 bits per channel path).
constexpr int kPrecisionBits = 32 - 8 - 2;

double BilinearFilter(double x) {
  if (x < 0.0) x = -x;
  if (x < 1.0) return 1.0 - x;
  return 0.0;
}

struct Coeffs {
  int ksize{0};
  std::vector<int> bounds;   // per output: xmin, count
  std::vector<std::int32_t> kk;  // per output: ksize fixed-point weights
};

// precompute_coeffs + normalize_coeffs_8bpc for box [in0, in1) of the input.
Coeffs Precompute(int in_size, float in0, float in1, int out_size) {
  constexpr double kSupport = 1.0;  // bilinear
  double scale = static_cast<double>(in1 - in0) / out_size;
  double filterscale = scale;
  if (filterscale < 1.0) filterscale = 1.0;
  const double support = kSupport * filterscale;
  Coeffs c;
  c.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
  c.bounds.resize(static_cast<std::size_t>(out_size) * 2);
  std::vector<double> kd(static_cast<std::size_t>(out_size) * c.ksize, 0.0);
  for (int xx = 0; xx < out_size; ++xx) {
    const double center = in0 + (xx + 0.5) * scale;
    double ww = 0.0;
    const double ss = 1.0 / filterscale;
    int xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;
    double* k = &kd[static_cast<std::size_t>(xx) * c.ksize];
    for (int x = 0; x < xmax; ++x) {
      const double w = BilinearFilter((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (int x = 0; x < xmax; ++x) {
      if (ww != 0.0) k[x] /= ww;
    }
    c.bounds[static_cast<std::size_t>(xx) * 2] = xmin;
    c.bounds[static_cast<std::size_t>(xx) * 2 + 1] = xmax;
  }
  c.kk.resize(kd.size());
  for (std::size_t i = 0; i < kd.size(); ++i) {
    c.kk[i] = kd[i] < 0 ? static_cast<std::int32_t>(-0.5 + kd[i] * (1 << kPrecisionBits))
                        : static_cast<std::int32_t>(0.5 + kd[i] * (1 << kPrecisionBits));
  }
  return c;
}

inline std::uint8_t Clip8(std::int32_t v) {
  const std::int32_t s = v >> kPrecisionBits;  // arithmetic shift like Pillow
  return static_cast<std::uint8_t>(std::clamp(s, 0, 255));
}

// ImagingResampleHorizontal_8bpc over rows [row0, row0 + rows) of `in`.
RgbImage Horizontal(const RgbImage& in, const Coeffs& ch, int out_width, int row0, int rows) {
  RgbImage cur;
  cur.width = out_width;
  cur.height = rows;
  cur.rgb.resize(static_cast<std::size_t>(cur.width) * cur.height * 3);
  for (int yy = 0; yy < cur.height; ++yy) {
    const std::uint8_t* row = &in.rgb[static_cast<std::size_t>(yy + row0) * in.width * 3];
    std::uint8_t* out = &cur.rgb[static_cast<std::size_t>(yy) * cur.width * 3];
    for (int xx = 0; xx < out_width; ++xx) {
      const int xmin = ch.bounds[static_cast<std::size_t>(xx) * 2];
      const int xmax = ch.bounds[static_cast<std::size_t>(xx) * 2 + 1];
      const std::int32_t* k = &ch.kk[static_cast<std::size_t>(xx) * ch.ksize];
      std::int32_t s0 = 1 << (kPrecisionBits - 1), s1 = s0, s2 = s0;
      for (int x = 0; x < xmax; ++x) {
        const std::uint8_t* p = row + static_cast<std::size_t>(x + xmin) * 3;
        s0 += p[0] * k[x];
        s1 += p[1] * k[x];
        s2 += p[2] * k[x];
      }
      out[xx * 3 + 0] = Clip8(s0);
      out[xx * 3 + 1] = Clip8(s1);
      out[xx * 3 + 2] = Clip8(s2);
    }
  }
  return cur;
}

// ImagingResampleVertical_8bpc.
RgbImage Vertical(const RgbImage& src, const Coeffs& cv, int out_height) {
  RgbImage outimg;
  outimg.width = src.width;
  outimg.height = out_height;
  outimg.rgb.resize(static_cast<std::size_t>(outimg.width) * out_height * 3);
  for (int yy = 0; yy < out_height; ++yy) {
    const std::int32_t* k = &cv.kk[static_cast<std::size_t>(yy) * cv.ksize];
    const int ymin = cv.bounds[static_cast<std::size_t>(yy) * 2];
    const int ymax = cv.bounds[static_cast<std::size_t>(yy) * 2 + 1];
    std::uint8_t* out = &outimg.rgb[static_cast<std::size_t>(yy) * outimg.width * 3];
    for (int xx = 0; xx < outimg.width; ++xx) {
      std::int32_t s0 = 1 << (kPrecisionBits - 1), s1 = s0, s2 = s0;
      for (int y = 0; y < ymax; ++y) {
        const std::uint8_t* p = &src.rgb[(static_cast<std::size_t>(y + ymin) * src.width + xx) * 3];
        s0 += p[0] * k[y];
        s1 += p[1] * k[y];
        s2 += p[2] * k[y];
      }
      out[xx * 3 + 0] = Clip8(s0);
      out[xx * 3 + 1] = Clip8(s1);
      out[xx * 3 + 2] = Clip8(s2);
    }
  }
  return outimg;
}

}  // namespace

RgbImage ResizeBilinear(const RgbImage& in, int out_width, int out_height) {
  if (in.width <= 0 || in.height <= 0 || out_width <= 0 || out_height <= 0) {
    throw std::invalid_argument("ResizeBilinear: empty image");
  }
  if (out_width == in.width && out_height == in.height) return in;  // Image.resize returns a copy
  const bool need_h = out_width != in.width;
  const bool need_v = out_height != in.height;
  const Coeffs ch = Precompute(in.width, 0.0f, static_cast<float>(in.width), out_width);
  Coeffs cv = Precompute(in.height, 0.0f, static_cast<float>(in.height), out_height);
  // Pillow 10+: the vertical pass runs first only when the height shrinks by
  // more than twice as many pixels as the width does.
  const int dh = in.height - out_height;
  const int dw = in.width - out_width;
  const bool horizontal_first = !(dh > 0 && dh > dw * 2);
  RgbImage cur;
  const RgbImage* src = &in;
  if (horizontal_first) {
    if (need_h) {
      // Only the source rows the vertical filter touches.
      const int ybox_first = cv.bounds[0];
      const int ybox_last = cv.bounds[static_cast<std::size_t>(out_height) * 2 - 2] +
                            cv.bounds[static_cast<std::size_t>(out_height) * 2 - 1];
      for (int i = 0; i < out_height; ++i) cv.bounds[static_cast<std::size_t>(i) * 2] -= ybox_first;
      cur = Horizontal(in, ch, out_width, ybox_first, ybox_last - ybox_first);
      src = &cur;
    }
    if (!need_v) return cur;
    return Vertical(*src, cv, out_height);
  }
  if (need_v) {
    cur = Vertical(in, cv, out_height);
    src = &cur;
  }
  if (!need_h) return cur;
  return Horizontal(*src, ch, out_width, 0, src->height);
}

RgbImage Crop(const RgbImage& in, int x0, int y0, int x1, int y1) {
  RgbImage out;
  out.width = std::max(0, x1 - x0);
  out.height = std::max(0, y1 - y0);
  out.rgb.assign(static_cast<std::size_t>(out.width) * out.height * 3, 0);
  for (int y = 0; y < out.height; ++y) {
    const int sy = y + y0;
    if (sy < 0 || sy >= in.height) continue;
    for (int x = 0; x < out.width; ++x) {
      const int sx = x + x0;
      if (sx < 0 || sx >= in.width) continue;
      for (int c = 0; c < 3; ++c) {
        out.rgb[(static_cast<std::size_t>(y) * out.width + x) * 3 + c] =
            in.rgb[(static_cast<std::size_t>(sy) * in.width + sx) * 3 + c];
      }
    }
  }
  return out;
}

std::vector<float> CcipInput(const RgbImage& image) {
  const RgbImage r = ResizeBilinear(image, kCcipSize, kCcipSize);
  const std::size_t plane = static_cast<std::size_t>(kCcipSize) * kCcipSize;
  std::vector<float> x(plane * 3);
  for (std::size_t i = 0; i < plane; ++i) {
    for (int c = 0; c < 3; ++c) {
      float v = static_cast<float>(r.rgb[i * 3 + c]) / 255.0f;
      v = (v - 0.5f) / 0.5f;
      x[c * plane + i] = v;
    }
  }
  return x;
}

LetterboxInput Letterbox(const RgbImage& image, int size) {
  LetterboxInput lb;
  lb.size = size;
  const double w = image.width;
  const double h = image.height;
  lb.ratio = std::min(size / w, size / h);
  // Python round(): half to even (default rounding mode).
  const int nw = static_cast<int>(std::nearbyint(w * lb.ratio));
  const int nh = static_cast<int>(std::nearbyint(h * lb.ratio));
  lb.dx = (size - nw) / 2;
  lb.dy = (size - nh) / 2;
  const RgbImage r = ResizeBilinear(image, nw, nh);
  const std::size_t plane = static_cast<std::size_t>(size) * size;
  lb.chw.assign(plane * 3, 114.0f / 255.0f);
  for (int y = 0; y < nh; ++y) {
    const int ty = y + lb.dy;
    if (ty < 0 || ty >= size) continue;
    for (int x = 0; x < nw; ++x) {
      const int tx = x + lb.dx;
      if (tx < 0 || tx >= size) continue;
      for (int c = 0; c < 3; ++c) {
        lb.chw[c * plane + static_cast<std::size_t>(ty) * size + tx] =
            static_cast<float>(r.rgb[(static_cast<std::size_t>(y) * nw + x) * 3 + c]) / 255.0f;
      }
    }
  }
  return lb;
}

std::array<int, 4> HeadCropBox(const std::array<float, 4>& face, int width, int height,
                               double grow) {
  // float32 arithmetic like numpy on the float32 boxes in ccip_ref.py.
  const float g = static_cast<float>(grow);
  const float cx = (face[0] + face[2]) / 2.0f;
  const float cy = (face[1] + face[3]) / 2.0f;
  const float s = std::max(face[2] - face[0], face[3] - face[1]) * g;
  const auto trunc = [](float v) { return static_cast<int>(v); };
  return {trunc(std::max(0.0f, cx - s / 2.0f)), trunc(std::max(0.0f, cy - s / 2.0f)),
          trunc(std::min(static_cast<float>(width), cx + s / 2.0f)),
          trunc(std::min(static_cast<float>(height), cy + s / 2.0f))};
}

}  // namespace gufo::models::ccip
