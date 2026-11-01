// CPU pre- and post-processing for the NPU body models (see cpu_post.hpp).
// Written with AI assistance (Claude Code).
#include "src/models/realesrgan/cpu_post.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>

// Every float operation below mirrors numpy / OpenCV one rounding at a time;
// a fused multiply-add would change the last bit.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

namespace gufo::models::realesrgan::cpu {
namespace {

// numpy: python float 1/255 times a float32 array is a float32 multiply.
constexpr float kInv255 = static_cast<float>(1.0 / 255.0);

std::uint8_t RoundToU8(float v) {
  // OpenCV saturate_cast<uchar>(float): cvRound (round half to even), clamp.
  const long r = std::lrintf(v);
  return static_cast<std::uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
}

struct AreaTap {
  int di;
  int si;
  float alpha;
};

// OpenCV imgproc/resize.cpp computeResizeAreaTab.
std::vector<AreaTap> AreaTable(int ssize, int dsize, int cn, double scale) {
  std::vector<AreaTap> tab;
  tab.reserve(static_cast<std::size_t>(ssize) * 2);
  for (int dx = 0; dx < dsize; ++dx) {
    const double fsx1 = dx * scale;
    const double fsx2 = fsx1 + scale;
    const double cell = std::min(scale, ssize - fsx1);
    int sx1 = static_cast<int>(std::ceil(fsx1));
    int sx2 = static_cast<int>(std::floor(fsx2));
    sx2 = std::min(sx2, ssize - 1);
    sx1 = std::min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3)
      tab.push_back({dx * cn, (sx1 - 1) * cn,
                     static_cast<float>((sx1 - fsx1) / cell)});
    for (int sx = sx1; sx < sx2; ++sx)
      tab.push_back({dx * cn, sx * cn, static_cast<float>(1.0 / cell)});
    if (fsx2 - sx2 > 1e-3)
      tab.push_back(
          {dx * cn, sx2 * cn,
           static_cast<float>(std::min(std::min(fsx2 - sx2, 1.0), cell) /
                              cell)});
  }
  return tab;
}

void ResizeAreaGeneral(const std::uint8_t* src, int src_w, int src_h,
                       int stride, int dst_w, int dst_h, double scale_x,
                       double scale_y, std::uint8_t* dst) {
  const std::vector<AreaTap> xtab = AreaTable(src_w, dst_w, 3, scale_x);
  const std::vector<AreaTap> ytab = AreaTable(src_h, dst_h, 1, scale_y);
  // First ytab entry of every destination row.
  std::vector<std::size_t> start(static_cast<std::size_t>(dst_h) + 1,
                                 ytab.size());
  for (std::size_t k = ytab.size(); k-- > 0;)
    start[static_cast<std::size_t>(ytab[k].di)] = k;
  const int width = dst_w * 3;
#pragma omp parallel for schedule(static)
  for (int dy = 0; dy < dst_h; ++dy) {
    std::vector<float> buf(static_cast<std::size_t>(width));
    std::vector<float> sum(static_cast<std::size_t>(width), 0.0F);
    for (std::size_t j = start[static_cast<std::size_t>(dy)];
         j < ytab.size() && ytab[j].di == dy; ++j) {
      const float beta = ytab[j].alpha;
      const std::uint8_t* S =
          src + static_cast<std::size_t>(ytab[j].si) * stride;
      std::fill(buf.begin(), buf.end(), 0.0F);
      for (const AreaTap& t : xtab) {
        const float a = t.alpha;
        const float t0 = buf[t.di] + static_cast<float>(S[t.si]) * a;
        const float t1 = buf[t.di + 1] + static_cast<float>(S[t.si + 1]) * a;
        const float t2 = buf[t.di + 2] + static_cast<float>(S[t.si + 2]) * a;
        buf[t.di] = t0;
        buf[t.di + 1] = t1;
        buf[t.di + 2] = t2;
      }
      // OpenCV starts every destination row with sum = beta * buf; adding to
      // a zero sum is the same value.
      for (int dx = 0; dx < width; ++dx) {
        const float p = beta * buf[dx];
        sum[dx] = sum[dx] + p;
      }
    }
    std::uint8_t* D = dst + static_cast<std::size_t>(dy) * dst_w * 3;
    for (int dx = 0; dx < width; ++dx)
      D[dx] = RoundToU8(sum[dx]);
  }
}

void ResizeAreaInteger(const std::uint8_t* src, int stride, int dst_w,
                       int dst_h, int sx, int sy, std::uint8_t* dst) {
  const int area = sx * sy;
  const float scale = 1.0F / static_cast<float>(area);
#pragma omp parallel for schedule(static)
  for (int dy = 0; dy < dst_h; ++dy) {
    std::uint8_t* D = dst + static_cast<std::size_t>(dy) * dst_w * 3;
    for (int dx = 0; dx < dst_w; ++dx) {
      for (int c = 0; c < 3; ++c) {
        int sum = 0;
        for (int y = 0; y < sy; ++y) {
          const std::uint8_t* S =
              src + static_cast<std::size_t>(dy * sy + y) * stride;
          for (int x = 0; x < sx; ++x)
            sum += S[(dx * sx + x) * 3 + c];
        }
        D[dx * 3 + c] = (sx == 2 && sy == 2)
                            ? static_cast<std::uint8_t>((sum + 2) >> 2)
                            : RoundToU8(static_cast<float>(sum) * scale);
      }
    }
  }
}

}  // namespace

std::vector<TilePass> PlanRowTiles(int h, int tile) {
  if (h <= 0 || tile <= 0 || tile > h)
    return {};
  if (tile == h)
    return {TilePass{0, 0, h, 0}};
  const int mid = h / 2;
  if (h % 2 != 0 || tile < mid)
    return {};
  // Rows 0..tile-1 keep 0..mid-1; rows h-tile..h-1 keep mid..h-1.
  return {TilePass{0, 0, mid, 0},
          TilePass{h - tile, mid - (h - tile), tile, mid}};
}

void PackTileInput(const std::uint8_t* rgb24, int w, int row, int rows,
                   float* chw) {
  const std::size_t plane = static_cast<std::size_t>(rows) * w;
  for (int y = 0; y < rows; ++y) {
    const std::uint8_t* src = rgb24 + static_cast<std::size_t>(row + y) * w * 3;
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < 3; ++c)
        chw[c * plane + static_cast<std::size_t>(y) * w + x] =
            static_cast<float>(src[x * 3 + c]) * kInv255;
  }
}

void StoreTileOutput(const float* tile_out, int w, int tile_rows,
                     const TilePass& pass, int h, float* body) {
  const int keep = pass.keep_end - pass.keep_begin;
  for (int c = 0; c < kBodyChannels; ++c)
    std::memcpy(body + (static_cast<std::size_t>(c) * h + pass.dst_row) * w,
                tile_out + (static_cast<std::size_t>(c) * tile_rows +
                            pass.keep_begin) * w,
                static_cast<std::size_t>(keep) * w * sizeof(float));
}

void ShuffleResidual(const float* body, const std::uint8_t* rgb24, int w,
                     int h, std::uint8_t* out24) {
  const std::size_t plane = static_cast<std::size_t>(h) * w;
  const std::size_t out_w = static_cast<std::size_t>(w) * kScale;
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    for (int i = 0; i < kScale; ++i) {
      std::uint8_t* row = out24 + (static_cast<std::size_t>(y) * kScale + i) *
                                      out_w * 3;
      for (int x = 0; x < w; ++x) {
        const std::uint8_t* px = rgb24 + (static_cast<std::size_t>(y) * w + x) * 3;
        for (int j = 0; j < kScale; ++j) {
          for (int c = 0; c < 3; ++c) {
            const float base = static_cast<float>(px[c]) * kInv255;
            const float v =
                body[static_cast<std::size_t>(c * 16 + i * 4 + j) * plane +
                     static_cast<std::size_t>(y) * w + x];
            float z = v + base;
            z = z < 0.0F ? 0.0F : (z > 1.0F ? 1.0F : z);  // np.clip
            const float scaled = z * 255.0F;
            const float rounded = scaled + 0.5F;
            row[(static_cast<std::size_t>(x) * kScale + j) * 3 + c] =
                static_cast<std::uint8_t>(rounded);  // astype: truncation
          }
        }
      }
    }
  }
}

bool ResizeAreaRgb8(const std::uint8_t* src, int src_w, int src_h,
                    int src_stride, int dst_w, int dst_h, std::uint8_t* dst,
                    std::string* error) {
  if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0 ||
      dst_w > src_w || dst_h > src_h) {
    if (error != nullptr)
      *error = "area resize needs 0 < output <= input (" +
               std::to_string(src_w) + "x" + std::to_string(src_h) + " -> " +
               std::to_string(dst_w) + "x" + std::to_string(dst_h) + ")";
    return false;
  }
  if (dst_w == src_w && dst_h == src_h) {
    for (int y = 0; y < src_h; ++y)
      std::memcpy(dst + static_cast<std::size_t>(y) * dst_w * 3,
                  src + static_cast<std::size_t>(y) * src_stride,
                  static_cast<std::size_t>(src_w) * 3);
    return true;
  }
  // cv::resize: inv_scale = dsize / ssize, scale = 1 / inv_scale.
  const double scale_x = 1.0 / (static_cast<double>(dst_w) / src_w);
  const double scale_y = 1.0 / (static_cast<double>(dst_h) / src_h);
  const int iscale_x = static_cast<int>(std::lround(scale_x));
  const int iscale_y = static_cast<int>(std::lround(scale_y));
  if (std::abs(scale_x - iscale_x) < DBL_EPSILON &&
      std::abs(scale_y - iscale_y) < DBL_EPSILON) {
    ResizeAreaInteger(src, src_stride, dst_w, dst_h, iscale_x, iscale_y, dst);
    return true;
  }
  ResizeAreaGeneral(src, src_w, src_h, src_stride, dst_w, dst_h, scale_x,
                    scale_y, dst);
  return true;
}

AreaCropPlan PlanAreaCrop(int in_w, int in_h, int out_w, int out_h) {
  AreaCropPlan p;
  p.crop_w = in_w;
  p.crop_h = in_h;
  if (in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0)
    return p;
  const std::int64_t lhs = static_cast<std::int64_t>(in_w) * out_h;
  const std::int64_t rhs = static_cast<std::int64_t>(out_w) * in_h;
  if (lhs == rhs) {  // same aspect ratio
    p.resize_w = out_w;
    p.resize_h = out_h;
    return p;
  }
  if (lhs > rhs) {  // source wider: crop columns
    if (rhs % out_h == 0) {
      p.crop_w = static_cast<int>(rhs / out_h);
      p.crop_x = (in_w - p.crop_w) / 2;
      p.resize_w = out_w;
      p.resize_h = out_h;
      return p;
    }
    p.resize_h = out_h;
    p.resize_w = static_cast<int>(std::lround(static_cast<double>(lhs) / in_h));
  } else {  // source taller: crop rows
    const std::int64_t rows = static_cast<std::int64_t>(out_h) * in_w;
    if (rows % out_w == 0) {
      p.crop_h = static_cast<int>(rows / out_w);
      p.crop_y = (in_h - p.crop_h) / 2;
      p.resize_w = out_w;
      p.resize_h = out_h;
      return p;
    }
    p.resize_w = out_w;
    p.resize_h =
        static_cast<int>(std::lround(static_cast<double>(rows) / in_w));
  }
  p.out_x = (p.resize_w - out_w) / 2;
  p.out_y = (p.resize_h - out_h) / 2;
  return p;
}

bool FitAreaCrop(const std::uint8_t* in24, int in_w, int in_h, int out_w,
                 int out_h, std::vector<std::uint8_t>* out24,
                 std::string* error) {
  const AreaCropPlan p = PlanAreaCrop(in_w, in_h, out_w, out_h);
  const std::uint8_t* src =
      in24 + (static_cast<std::size_t>(p.crop_y) * in_w + p.crop_x) * 3;
  std::vector<std::uint8_t> resized(static_cast<std::size_t>(p.resize_w) *
                                    p.resize_h * 3);
  if (!ResizeAreaRgb8(src, p.crop_w, p.crop_h, in_w * 3, p.resize_w,
                      p.resize_h, resized.data(), error))
    return false;
  if (p.resize_w == out_w && p.resize_h == out_h) {
    *out24 = std::move(resized);
    return true;
  }
  out24->resize(static_cast<std::size_t>(out_w) * out_h * 3);
  for (int y = 0; y < out_h; ++y)
    std::memcpy(out24->data() + static_cast<std::size_t>(y) * out_w * 3,
                resized.data() + (static_cast<std::size_t>(y + p.out_y) *
                                      p.resize_w + p.out_x) * 3,
                static_cast<std::size_t>(out_w) * 3);
  return true;
}

bool FitAreaPad(const std::uint8_t* in24, int in_w, int in_h, int out_w,
                int out_h, std::vector<std::uint8_t>* out24,
                std::string* error) {
  if (in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0) {
    if (error != nullptr)
      *error = "invalid size";
    return false;
  }
  const double s = std::min(static_cast<double>(out_w) / in_w,
                            static_cast<double>(out_h) / in_h);
  const int cw = std::clamp(static_cast<int>(std::lround(in_w * s)), 1, out_w);
  const int ch = std::clamp(static_cast<int>(std::lround(in_h * s)), 1, out_h);
  std::vector<std::uint8_t> resized(static_cast<std::size_t>(cw) * ch * 3);
  if (!ResizeAreaRgb8(in24, in_w, in_h, in_w * 3, cw, ch, resized.data(),
                      error))
    return false;
  out24->assign(static_cast<std::size_t>(out_w) * out_h * 3, 0);
  const int ox = (out_w - cw) / 4 * 2;
  const int oy = (out_h - ch) / 4 * 2;
  for (int y = 0; y < ch; ++y)
    std::memcpy(out24->data() +
                    (static_cast<std::size_t>(y + oy) * out_w + ox) * 3,
                resized.data() + static_cast<std::size_t>(y) * cw * 3,
                static_cast<std::size_t>(cw) * 3);
  return true;
}

}  // namespace gufo::models::realesrgan::cpu
