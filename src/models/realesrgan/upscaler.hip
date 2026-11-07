#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/realesrgan/upscaler.hpp"

// Layout: every tensor is channels-last [height, width, channels]. A tile of
// the input frame (core plus a halo of one pixel per convolution, clipped at
// the frame border) runs through the SRVGGNetCompact body:
//   x    = conv3x3(3 -> 64), PReLU
//   x    = num_conv x (conv3x3(64 -> 64), PReLU)
//   raw  = conv3x3(64 -> 3 * S * S)
//   out  = PixelShuffle(S)(raw) + nearest(input, S), clamped to [0, 1]
// Zero padding at the frame border is the convolution's own padding, and the
// halo covers the receptive field, so tiles reproduce the whole-frame result.
// The assembled S x picture (FP32) is resized to the output with separable
// antialiased Lanczos-3 (horizontal, then vertical into RGB24).

namespace gufo::models::realesrgan {
namespace {

using Half = _Float16;
using Half16 = __attribute__((ext_vector_type(16))) _Float16;
using Acc8 = __attribute__((ext_vector_type(8))) float;

constexpr int kThreads = 256;
constexpr int kInputChannels = 16;  // RGB padded for the WMMA loads
constexpr std::size_t kDefaultTilePixels = 512 * 1024;

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

bool Check(hipError_t status, const char* what, std::string* error) {
  if (status == hipSuccess)
    return true;
  SetError(error, std::string("Real-ESRGAN ") + what + ": " +
                      hipGetErrorString(status));
  return false;
}

unsigned Grid(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

// ---------------------------------------------------------------- kernels --

struct ConvArgs {
  const void* in;  // [h, w, cin]
  int h, w, cin;
  int rows;   // output channels
  int k_pad;  // weight row stride
  const void* weight;
  const float* bias;
  const float* slope;  // PReLU per output channel; nullptr: none
  int raw;             // 1: FP32 out [h, w, rows]; 0: activation type
  void* out;
};

__device__ __forceinline__ float ToFloat(float v) {
  return v;
}
__device__ __forceinline__ float ToFloat(Half v) {
  return static_cast<float>(v);
}
template<typename T>
__device__ __forceinline__ T FromFloat(float v) {
  return static_cast<T>(v);
}

// One thread per output element, FP32 accumulation: the validation path.
template<typename T>
__global__ void DirectConvKernel(ConvArgs a) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t pixels = static_cast<std::size_t>(a.h) * a.w;
  if (index >= pixels * a.rows)
    return;
  const int row = static_cast<int>(index % a.rows);
  const std::size_t p = index / a.rows;
  const int oy = static_cast<int>(p / a.w);
  const int ox = static_cast<int>(p % a.w);
  const T* in = static_cast<const T*>(a.in);
  const float* w = static_cast<const float*>(a.weight) +
                   static_cast<std::size_t>(row) * a.k_pad;
  float acc = a.bias[row];
  for (int ky = 0; ky < 3; ++ky) {
    const int iy = oy - 1 + ky;
    if (iy < 0 || iy >= a.h)
      continue;
    for (int kx = 0; kx < 3; ++kx) {
      const int ix = ox - 1 + kx;
      if (ix < 0 || ix >= a.w)
        continue;
      const T* source = in + (static_cast<std::size_t>(iy) * a.w + ix) * a.cin;
      const float* tap = w + (ky * 3 + kx) * a.cin;
      for (int c = 0; c < a.cin; ++c)
        acc += ToFloat(source[c]) * tap[c];
    }
  }
  const std::size_t o = p * a.rows + row;
  if (a.raw != 0) {
    static_cast<float*>(a.out)[o] = acc;
    return;
  }
  if (a.slope != nullptr && acc < 0.0F)
    acc *= a.slope[row];
  static_cast<T*>(a.out)[o] = FromFloat<T>(acc);
}

// Implicit-GEMM 3x3 convolution (stride 1, zero padding 1), FP16 WMMA with
// FP32 accumulation; the kernel of src/models/rife/ifnet.hip with a PReLU
// epilogue. A block computes 64 output pixels x 64 rows with four waves
// (2 x 2, each 32 x 32); K = tap * cin + channel advances 32 per stage through
// LDS while the next stage is fetched into registers. cin is a multiple of
// 16; weights are [rows padded to 64, k_pad] with zeros past 9 * cin and past
// `rows`.
constexpr int kTileM = 64, kTileN = 64, kTileK = 32, kLdsK = kTileK + 8;
constexpr int kOutStride = kTileN + 4;

__device__ __forceinline__ Half16 LoadFrag(const Half* p) {
  union {
    Half16 f;
    uint4 v[2];
  } u;
  u.v[0] = *reinterpret_cast<const uint4*>(p);
  u.v[1] = *reinterpret_cast<const uint4*>(p + 8);
  return u.f;
}

__launch_bounds__(128) __global__ void WmmaConvKernel(ConvArgs a) {
  __shared__ __align__(16) Half as[kTileM * kLdsK];
  __shared__ __align__(16) Half bs[kTileN * kLdsK];
  __shared__ __align__(16) float cs[kTileM * kOutStride];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int sl = lane & 15, wm = wave >> 1, wn = wave & 1;
  const int pixels = a.h * a.w;
  const int p0 = blockIdx.x * kTileM, r0 = blockIdx.y * kTileN;
  const int group = tid & 3, line = tid >> 2;
  int oy[2], ox[2];
  bool valid[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int p = p0 + line + 32 * i;
    valid[i] = p < pixels;
    oy[i] = p / a.w;
    ox[i] = p % a.w;
  }
  const Half* in = static_cast<const Half*>(a.in);
  const Half* weight = static_cast<const Half*>(a.weight);
  const int k_total = 9 * a.cin;
  const int stages = a.k_pad / kTileK;
  uint4 av[2], bv[2];
  const auto fetch = [&](int s) {
    const int k = s * kTileK + group * 8;
    const int tap = k / a.cin, c = k - tap * a.cin;
    const int ky = tap / 3, kx = tap - ky * 3;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      uint4 v = make_uint4(0, 0, 0, 0);
      const int iy = oy[i] - 1 + ky;
      const int ix = ox[i] - 1 + kx;
      if (valid[i] && k < k_total && iy >= 0 && iy < a.h && ix >= 0 &&
          ix < a.w)
        v = *reinterpret_cast<const uint4*>(
            in + (static_cast<std::size_t>(iy) * a.w + ix) * a.cin + c);
      av[i] = v;
      bv[i] = *reinterpret_cast<const uint4*>(
          weight + static_cast<std::size_t>(r0 + line + 32 * i) * a.k_pad + k);
    }
  };
  Acc8 acc[2][2] = {};
  fetch(0);
  for (int s = 0; s < stages; ++s) {
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      *reinterpret_cast<uint4*>(as + (line + 32 * i) * kLdsK + group * 8) =
          av[i];
      *reinterpret_cast<uint4*>(bs + (line + 32 * i) * kLdsK + group * 8) =
          bv[i];
    }
    __syncthreads();
    if (s + 1 < stages)
      fetch(s + 1);
#pragma unroll
    for (int kk = 0; kk < kTileK; kk += 16) {
      Half16 fa[2], fb[2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        fa[i] = LoadFrag(as + (wm * 32 + i * 16 + sl) * kLdsK + kk);
#pragma unroll
      for (int j = 0; j < 2; ++j)
        fb[j] = LoadFrag(bs + (wn * 32 + j * 16 + sl) * kLdsK + kk);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fa[i], fb[j],
                                                                 acc[i][j]);
    }
  }
  // D[m = pixel][n = row]: lane holds n = lane % 16, m = 2 z + lane / 16.
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int z = 0; z < 8; ++z)
        cs[(wm * 32 + i * 16 + 2 * z + (lane >> 4)) * kOutStride + wn * 32 +
           j * 16 + sl] = acc[i][j][z];
  __syncthreads();
  for (int index = tid; index < kTileM * kTileN / 8; index += 128) {
    const int m = index / (kTileN / 8), g = index % (kTileN / 8);
    const int p = p0 + m, row = r0 + g * 8;
    if (p >= pixels || row >= a.rows)
      continue;
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e)
      v[e] = cs[m * kOutStride + g * 8 + e] + a.bias[row + e];
    const std::size_t o = static_cast<std::size_t>(p) * a.rows + row;
    if (a.raw != 0) {
      float* out = static_cast<float*>(a.out) + o;
      *reinterpret_cast<float4*>(out) = make_float4(v[0], v[1], v[2], v[3]);
      *reinterpret_cast<float4*>(out + 4) =
          make_float4(v[4], v[5], v[6], v[7]);
      continue;
    }
    union {
      uint4 u;
      Half h[8];
    } w;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float x =
          a.slope != nullptr && v[e] < 0.0F ? v[e] * a.slope[row + e] : v[e];
      w.h[e] = static_cast<Half>(x);
    }
    *reinterpret_cast<uint4*>(static_cast<Half*>(a.out) + o) = w.u;
  }
}

// RGB24 frame [height, width, 3] -> tile input [th, tw, 16] at (x0, y0):
// channels 0..2 = value / 255, the rest zero.
template<typename T>
__global__ void PrepareTileKernel(const std::uint8_t* rgb, int width, int x0,
                                  int y0, int tw, int th, T* out) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(tw) * th)
    return;
  const int y = static_cast<int>(index / tw) + y0;
  const int x = static_cast<int>(index % tw) + x0;
  const std::uint8_t* s = rgb + (static_cast<std::size_t>(y) * width + x) * 3;
  T* e = out + index * kInputChannels;
  e[0] = FromFloat<T>(s[0] / 255.0F);
  e[1] = FromFloat<T>(s[1] / 255.0F);
  e[2] = FromFloat<T>(s[2] / 255.0F);
  for (int c = 3; c < kInputChannels; ++c)
    e[c] = FromFloat<T>(0.0F);
}

// Pixel shuffle of the last convolution's raw [th, tw, 3 S^2] (tile at (x0,
// y0)) plus the nearest-upsampled input, clamped, for the core [cx0, cx1) x
// [cy0, cy1) of the frame: FP32 picture [height S, width S, 3].
__global__ void AssembleKernel(const float* raw, int tw, int x0, int y0,
                               int cx0, int cy0, int cw, int ch, int S,
                               const std::uint8_t* rgb, int width,
                               float* picture) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t wide = static_cast<std::size_t>(cw) * S;
  if (index >= wide * ch * S)
    return;
  const int oy = cy0 * S + static_cast<int>(index / wide);
  const int ox = cx0 * S + static_cast<int>(index % wide);
  const int iy = oy / S, ix = ox / S, sy = oy % S, sx = ox % S;
  const float* r = raw + (static_cast<std::size_t>(iy - y0) * tw + (ix - x0)) *
                             (3 * S * S);
  const std::uint8_t* base =
      rgb + (static_cast<std::size_t>(iy) * width + ix) * 3;
  float* out = picture + (static_cast<std::size_t>(oy) * width * S + ox) * 3;
  for (int c = 0; c < 3; ++c) {
    const float v = r[c * S * S + sy * S + sx] + base[c] / 255.0F;
    out[c] = fminf(fmaxf(v, 0.0F), 1.0F);
  }
}

// Antialiased Lanczos-3 (PIL's filter: support 3 * max(1, ratio), window
// clipped to the source and renormalised).
__device__ __forceinline__ float Lanczos3(float x) {
  x = fabsf(x);
  if (x < 1e-6F)
    return 1.0F;
  if (x >= 3.0F)
    return 0.0F;
  const float pi = 3.14159265358979F;
  return 3.0F * sinf(pi * x) * sinf(pi * x / 3.0F) / (pi * pi * x * x);
}

struct Taps {
  int first, count;
  float center, filter_scale;
};

__device__ __forceinline__ Taps ResampleTaps(int u, float ratio, int size) {
  const float filter_scale = fmaxf(ratio, 1.0F);
  const float support = 3.0F * filter_scale;
  const float center = (static_cast<float>(u) + 0.5F) * ratio;
  const int first = max(0, static_cast<int>(floorf(center - support)));
  const int last = min(size, static_cast<int>(ceilf(center + support)));
  return {first, last - first, center, filter_scale};
}

// picture [H, W, 3] -> rows [H, out_w, 3]: columns outside the content
// (offset_x, content_w) are black.
__global__ void ResizeHorizontalKernel(const float* picture, int H, int W,
                                       int out_w, int content_w, int offset_x,
                                       float* rows) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(H) * out_w)
    return;
  const int y = static_cast<int>(index / out_w);
  const int u = static_cast<int>(index % out_w) - offset_x;
  float acc[3] = {0.0F, 0.0F, 0.0F};
  if (u >= 0 && u < content_w) {
    const Taps t =
        ResampleTaps(u, static_cast<float>(W) / content_w, W);
    float total = 0.0F;
    const float* line = picture + static_cast<std::size_t>(y) * W * 3;
    for (int i = 0; i < t.count; ++i) {
      const int x = t.first + i;
      const float k =
          Lanczos3((x + 0.5F - t.center) / t.filter_scale);
      total += k;
      acc[0] += k * line[x * 3 + 0];
      acc[1] += k * line[x * 3 + 1];
      acc[2] += k * line[x * 3 + 2];
    }
    const float inv = total != 0.0F ? 1.0F / total : 0.0F;
    for (float& v : acc)
      v *= inv;
  }
  float* out = rows + index * 3;
  out[0] = acc[0];
  out[1] = acc[1];
  out[2] = acc[2];
}

// rows [H, out_w, 3] -> RGB24 [out_h, out_w, 3]: lines outside the content
// (offset_y, content_h) are black.
__global__ void ResizeVerticalKernel(const float* rows, int H, int out_w,
                                     int out_h, int content_h, int offset_y,
                                     std::uint8_t* out24) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(out_h) * out_w)
    return;
  const int v = static_cast<int>(index / out_w) - offset_y;
  const int x = static_cast<int>(index % out_w);
  float acc[3] = {0.0F, 0.0F, 0.0F};
  if (v >= 0 && v < content_h) {
    const Taps t =
        ResampleTaps(v, static_cast<float>(H) / content_h, H);
    float total = 0.0F;
    for (int i = 0; i < t.count; ++i) {
      const int y = t.first + i;
      const float k = Lanczos3((y + 0.5F - t.center) / t.filter_scale);
      total += k;
      const float* s = rows + (static_cast<std::size_t>(y) * out_w + x) * 3;
      acc[0] += k * s[0];
      acc[1] += k * s[1];
      acc[2] += k * s[2];
    }
    const float inv = total != 0.0F ? 1.0F / total : 0.0F;
    for (float& value : acc)
      value *= inv;
  }
  std::uint8_t* o = out24 + index * 3;
  for (int c = 0; c < 3; ++c)
    o[c] = static_cast<std::uint8_t>(
        fminf(fmaxf(acc[c], 0.0F), 1.0F) * 255.0F + 0.5F);
}

// ------------------------------------------------------------------ host --

struct HostTensor {
  std::vector<std::size_t> shape;
  std::vector<float> data;
};

bool ReadSafetensors(const std::filesystem::path& path,
                     std::map<std::string, HostTensor>* tensors,
                     std::map<std::string, std::string>* metadata,
                     std::string* error) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    SetError(error, "cannot open Real-ESRGAN weights " + path.string());
    return false;
  }
  file.seekg(0, std::ios::end);
  const auto size = static_cast<std::uint64_t>(file.tellg());
  file.seekg(0);
  std::uint64_t header_size = 0;
  if (size < 8 || !file.read(reinterpret_cast<char*>(&header_size), 8) ||
      header_size == 0 || header_size > (16U << 20U) ||
      header_size > size - 8) {
    SetError(error, "invalid safetensors header in " + path.string());
    return false;
  }
  std::string text(static_cast<std::size_t>(header_size), '\0');
  std::vector<char> body(static_cast<std::size_t>(size - 8 - header_size));
  if (!file.read(text.data(), static_cast<std::streamsize>(text.size())) ||
      !file.read(body.data(), static_cast<std::streamsize>(body.size()))) {
    SetError(error, "short read of " + path.string());
    return false;
  }
  try {
    const json::Value header = json::parse(text);
    for (const auto& [name, entry] : header.members()) {
      if (name == "__metadata__") {
        for (const auto& [key, value] : entry.members())
          if (value.is_string())
            (*metadata)[key] = value.str();
        continue;
      }
      const json::Value* shape = entry.find("shape");
      const json::Value* offsets = entry.find("data_offsets");
      if (entry.member_str("dtype") != "F32" || shape == nullptr ||
          !shape->is_array() || offsets == nullptr || !offsets->is_array() ||
          offsets->size() != 2) {
        SetError(error, "Real-ESRGAN tensor " + name + " is not F32");
        return false;
      }
      HostTensor tensor;
      std::size_t count = 1;
      for (const auto& dim : shape->items()) {
        tensor.shape.push_back(dim.as_size());
        count *= tensor.shape.back();
      }
      const std::size_t begin = offsets->items()[0].as_size();
      const std::size_t end = offsets->items()[1].as_size();
      if (end < begin || end - begin != count * 4 || end > body.size()) {
        SetError(error, "invalid Real-ESRGAN tensor extent " + name);
        return false;
      }
      tensor.data.resize(count);
      std::memcpy(tensor.data.data(), body.data() + begin, count * 4);
      (*tensors)[name] = std::move(tensor);
    }
  } catch (const std::exception& e) {
    SetError(error, path.string() + ": " + e.what());
    return false;
  }
  return true;
}

// IEEE binary16 bits of `value`, round to nearest even (host side).
std::uint16_t HalfBits(float value) {
  std::uint32_t x = 0;
  std::memcpy(&x, &value, 4);
  const std::uint32_t sign = (x >> 16U) & 0x8000U;
  const std::uint32_t magnitude = x & 0x7FFFFFFFU;
  if (magnitude >= 0x7F800000U)  // inf / nan
    return static_cast<std::uint16_t>(sign | 0x7C00U |
                                      (magnitude > 0x7F800000U ? 0x200U : 0));
  if (magnitude >= 0x477FF000U)  // rounds to >= 65520: inf
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  if (magnitude < 0x38800000U) {  // subnormal or zero in binary16
    const float f = std::fabs(value) * 16777216.0F;  // / 2^-24
    return static_cast<std::uint16_t>(
        sign | static_cast<std::uint32_t>(std::nearbyint(f)));
  }
  const std::uint32_t rounded =
      magnitude + 0xFFFU + ((magnitude >> 13U) & 1U) - 0x38000000U;
  return static_cast<std::uint16_t>(sign | (rounded >> 13U));
}

int RoundUp(int value, int multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

struct Conv {
  int cin{0};  // padded input channels
  int rows{0};
  int k_pad{0};
  void* weight{nullptr};   // FP16 [rows padded to 64, k_pad] or FP32 [rows, k]
  float* bias{nullptr};    // [rows padded to 64]
  float* slope{nullptr};   // PReLU [rows], nullptr for the last convolution
};

// A device buffer that only grows.
struct Buffer {
  void* data{nullptr};
  std::size_t bytes{0};
  bool Reserve(std::size_t need, std::string* error) {
    if (need <= bytes)
      return true;
    if (data != nullptr)
      (void)hipFree(data);
    data = nullptr;
    bytes = 0;
    if (!Check(hipMalloc(&data, need), "allocation", error))
      return false;
    bytes = need;
    return true;
  }
  void Release() {
    if (data != nullptr)
      (void)hipFree(data);
    data = nullptr;
    bytes = 0;
  }
  template<typename T>
  T* as() const {
    return static_cast<T*>(data);
  }
};

}  // namespace

bool ParseFit(std::string_view text, Fit* fit) {
  if (text == "pad")
    *fit = Fit::kPad;
  else if (text == "crop")
    *fit = Fit::kCrop;
  else if (text == "stretch")
    *fit = Fit::kStretch;
  else
    return false;
  return true;
}

std::string_view FitName(Fit fit) {
  switch (fit) {
  case Fit::kPad:
    return "pad";
  case Fit::kCrop:
    return "crop";
  case Fit::kStretch:
    return "stretch";
  }
  return "pad";
}

Placement PlaceContent(int in_w, int in_h, int out_w, int out_h, Fit fit) {
  Placement p;
  if (in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0)
    return p;
  if (fit == Fit::kStretch) {
    p.content_w = out_w;
    p.content_h = out_h;
    return p;
  }
  const double sx = static_cast<double>(out_w) / in_w;
  const double sy = static_cast<double>(out_h) / in_h;
  const double s = fit == Fit::kPad ? std::min(sx, sy) : std::max(sx, sy);
  // Even sizes; the axis that defines the scale matches the output exactly.
  const auto even = [](double v) {
    return std::max(2, static_cast<int>(std::lround(v / 2.0)) * 2);
  };
  p.content_w = sx == s ? out_w : even(in_w * s);
  p.content_h = sy == s ? out_h : even(in_h * s);
  if (fit == Fit::kPad) {
    p.content_w = std::min(p.content_w, out_w);
    p.content_h = std::min(p.content_h, out_h);
  } else {
    p.content_w = std::max(p.content_w, out_w);
    p.content_h = std::max(p.content_h, out_h);
  }
  // Centred; even offsets keep 4:2:0 chroma sited on whole pairs.
  p.offset_x = (out_w - p.content_w) / 4 * 2;
  p.offset_y = (out_h - p.content_h) / 4 * 2;
  return p;
}

struct Upscaler::Impl {
  Precision precision{Precision::kFp16};
  int scale{4};
  int features{64};
  std::vector<Conv> convs;
  std::size_t weight_bytes{0};
  std::size_t tile_pixels{kDefaultTilePixels};
  Buffer frame24, tile_in, act[2], raw, picture, rows, out24;

  ~Impl() {
    for (Conv& c : convs) {
      (void)hipFree(c.weight);
      (void)hipFree(c.bias);
      (void)hipFree(c.slope);
    }
    for (Buffer* b : {&frame24, &tile_in, &act[0], &act[1], &raw, &picture,
                      &rows, &out24})
      b->Release();
  }

  [[nodiscard]] std::size_t ActivationBytes() const {
    return precision == Precision::kFp16 ? sizeof(Half) : sizeof(float);
  }

  bool Upload(const std::map<std::string, HostTensor>& tensors, int index,
              int cin_pad, bool prelu, std::string* error) {
    const std::string prefix = "body." + std::to_string(index);
    const auto w = tensors.find(prefix + ".weight");
    const auto b = tensors.find(prefix + ".bias");
    if (w == tensors.end() || b == tensors.end() ||
        w->second.shape.size() != 4 || w->second.shape[2] != 3 ||
        w->second.shape[3] != 3 ||
        static_cast<int>(w->second.shape[1]) > cin_pad ||
        b->second.data.size() != w->second.shape[0]) {
      SetError(error, "Real-ESRGAN weights lack the convolution " + prefix);
      return false;
    }
    const int rows = static_cast<int>(w->second.shape[0]);
    const int cin = static_cast<int>(w->second.shape[1]);
    if (rows % 8 != 0 || rows > 64) {
      SetError(error, "Real-ESRGAN convolution " + prefix +
                          " has an unsupported width");
      return false;
    }
    Conv conv;
    conv.cin = cin_pad;
    conv.rows = rows;
    const int k = 9 * cin_pad;
    const bool half = precision == Precision::kFp16;
    const int rows_pad = half ? 64 : rows;
    conv.k_pad = half ? RoundUp(k, kTileK) : k;
    std::vector<float> weight(static_cast<std::size_t>(rows_pad) * conv.k_pad,
                              0.0F);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cin; ++c)
        for (int tap = 0; tap < 9; ++tap)
          weight[static_cast<std::size_t>(r) * conv.k_pad + tap * cin_pad + c] =
              w->second.data[(static_cast<std::size_t>(r) * cin + c) * 9 + tap];
    std::vector<float> bias(64, 0.0F);
    std::copy(b->second.data.begin(), b->second.data.end(), bias.begin());
    if (half) {
      std::vector<std::uint16_t> bits(weight.size());
      for (std::size_t i = 0; i < weight.size(); ++i)
        bits[i] = HalfBits(weight[i]);
      if (!Check(hipMalloc(&conv.weight, bits.size() * 2), "weights", error) ||
          !Check(hipMemcpy(conv.weight, bits.data(), bits.size() * 2,
                           hipMemcpyHostToDevice),
                 "weights", error))
        return false;
      weight_bytes += bits.size() * 2;
    } else {
      if (!Check(hipMalloc(&conv.weight, weight.size() * 4), "weights",
                 error) ||
          !Check(hipMemcpy(conv.weight, weight.data(), weight.size() * 4,
                           hipMemcpyHostToDevice),
                 "weights", error))
        return false;
      weight_bytes += weight.size() * 4;
    }
    if (!Check(hipMalloc(reinterpret_cast<void**>(&conv.bias), 64 * 4),
               "bias", error) ||
        !Check(hipMemcpy(conv.bias, bias.data(), 64 * 4,
                         hipMemcpyHostToDevice),
               "bias", error))
      return false;
    if (prelu) {
      const auto s = tensors.find("body." + std::to_string(index + 1) +
                                  ".weight");
      if (s == tensors.end() ||
          s->second.data.size() != static_cast<std::size_t>(rows)) {
        SetError(error, "Real-ESRGAN weights lack the PReLU after " + prefix);
        return false;
      }
      if (!Check(hipMalloc(reinterpret_cast<void**>(&conv.slope), rows * 4),
                 "slope", error) ||
          !Check(hipMemcpy(conv.slope, s->second.data.data(), rows * 4,
                           hipMemcpyHostToDevice),
                 "slope", error))
        return false;
    }
    weight_bytes += 64 * 4 + (prelu ? rows * 4 : 0);
    convs.push_back(conv);
    return true;
  }

  bool RunConv(const Conv& c, const void* in, int w, int h, void* out,
               std::string* error) {
    ConvArgs a{};
    a.in = in;
    a.h = h;
    a.w = w;
    a.cin = c.cin;
    a.rows = c.rows;
    a.k_pad = c.k_pad;
    a.weight = c.weight;
    a.bias = c.bias;
    a.slope = c.slope;
    a.raw = c.slope == nullptr ? 1 : 0;
    a.out = out;
    const std::size_t pixels = static_cast<std::size_t>(w) * h;
    if (precision == Precision::kFp16) {
      const dim3 grid(static_cast<unsigned>((pixels + kTileM - 1) / kTileM),
                      static_cast<unsigned>((c.rows + kTileN - 1) / kTileN));
      WmmaConvKernel<<<grid, 128>>>(a);
    } else {
      DirectConvKernel<float><<<Grid(pixels * c.rows), kThreads>>>(a);
    }
    return Check(hipGetLastError(), "convolution", error);
  }

  // Runs the network over the frame in frame24 into picture.
  bool Network(int w, int h, std::string* error) {
    const int halo = static_cast<int>(convs.size());
    // Fewest tiles whose padded size fits the tile budget: split the longer
    // tile side until it fits (or the halo dominates).
    const auto tile_size = [&](int n, int size) {
      const int core = (size + n - 1) / n;
      return std::min(size, core + (n > 1 ? 2 * halo : 0));
    };
    int nx = 1, ny = 1;
    while (static_cast<std::size_t>(tile_size(nx, w)) * tile_size(ny, h) >
               tile_pixels &&
           (nx < w || ny < h)) {
      if ((tile_size(nx, w) >= tile_size(ny, h) && nx < w) || ny >= h)
        ++nx;
      else
        ++ny;
      if (nx > 64 || ny > 64)
        break;
    }
    const std::size_t max_tile =
        static_cast<std::size_t>(tile_size(nx, w)) * tile_size(ny, h);
    const std::size_t abytes = ActivationBytes();
    const int last_rows = convs.back().rows;
    if (!tile_in.Reserve(max_tile * kInputChannels * abytes, error) ||
        !act[0].Reserve(max_tile * features * abytes, error) ||
        !act[1].Reserve(max_tile * features * abytes, error) ||
        !raw.Reserve(max_tile * last_rows * sizeof(float), error) ||
        !picture.Reserve(static_cast<std::size_t>(w) * h * scale * scale * 3 *
                             sizeof(float),
                         error))
      return false;
    const int core_w = (w + nx - 1) / nx, core_h = (h + ny - 1) / ny;
    for (int ty = 0; ty < ny; ++ty) {
      for (int tx = 0; tx < nx; ++tx) {
        const int cx0 = tx * core_w, cx1 = std::min(w, cx0 + core_w);
        const int cy0 = ty * core_h, cy1 = std::min(h, cy0 + core_h);
        if (cx0 >= cx1 || cy0 >= cy1)
          continue;
        const int x0 = std::max(0, cx0 - halo), x1 = std::min(w, cx1 + halo);
        const int y0 = std::max(0, cy0 - halo), y1 = std::min(h, cy1 + halo);
        const int tw = x1 - x0, th = y1 - y0;
        const std::size_t pixels = static_cast<std::size_t>(tw) * th;
        if (precision == Precision::kFp16)
          PrepareTileKernel<Half><<<Grid(pixels), kThreads>>>(
              frame24.as<std::uint8_t>(), w, x0, y0, tw, th,
              tile_in.as<Half>());
        else
          PrepareTileKernel<float><<<Grid(pixels), kThreads>>>(
              frame24.as<std::uint8_t>(), w, x0, y0, tw, th,
              tile_in.as<float>());
        const void* in = tile_in.data;
        int ping = 0;
        for (std::size_t i = 0; i < convs.size(); ++i) {
          void* out = i + 1 == convs.size() ? raw.data : act[ping].data;
          if (!RunConv(convs[i], in, tw, th, out, error))
            return false;
          in = out;
          ping ^= 1;
        }
        const std::size_t count =
            static_cast<std::size_t>(cx1 - cx0) * (cy1 - cy0) * scale * scale;
        AssembleKernel<<<Grid(count), kThreads>>>(
            raw.as<float>(), tw, x0, y0, cx0, cy0, cx1 - cx0, cy1 - cy0, scale,
            frame24.as<std::uint8_t>(), w, picture.as<float>());
        if (!Check(hipGetLastError(), "assemble", error))
          return false;
      }
    }
    return true;
  }

  bool UploadFrame(const std::uint8_t* rgb24, int w, int h,
                   std::string* error) {
    const std::size_t bytes = static_cast<std::size_t>(w) * h * 3;
    return frame24.Reserve(bytes, error) &&
           Check(hipMemcpy(frame24.data, rgb24, bytes, hipMemcpyHostToDevice),
                 "upload", error);
  }
};

Upscaler::Upscaler(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Upscaler::~Upscaler() = default;

std::unique_ptr<Upscaler> Upscaler::Load(const std::filesystem::path& weights,
                                         Precision precision,
                                         std::string* error) {
  std::map<std::string, HostTensor> tensors;
  std::map<std::string, std::string> metadata;
  if (!ReadSafetensors(weights, &tensors, &metadata, error))
    return nullptr;
  if (metadata["format"] != "realesrgan-srvgg") {
    SetError(error, weights.string() +
                        " is not a converted Real-ESRGAN SRVGGNetCompact file "
                        "(tools/upscale/convert_realesrgan_pth.py)");
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->precision = precision;
  int num_conv = 0;
  try {
    num_conv = std::stoi(metadata["num_conv"]);
    impl->features = std::stoi(metadata["num_feat"]);
    impl->scale = std::stoi(metadata["upscale"]);
  } catch (...) {
    SetError(error, "Real-ESRGAN metadata lacks num_conv/num_feat/upscale");
    return nullptr;
  }
  if (num_conv < 1 || num_conv > 64 || impl->features != 64 ||
      impl->scale < 1 || impl->scale > 4) {
    SetError(error, "unsupported Real-ESRGAN architecture (64 features, "
                    "x1..x4 expected)");
    return nullptr;
  }
  // body: conv(3), PReLU, num_conv x (conv, PReLU), conv(-> 3 S^2).
  if (!impl->Upload(tensors, 0, kInputChannels, true, error))
    return nullptr;
  for (int i = 1; i <= num_conv; ++i)
    if (!impl->Upload(tensors, 2 * i, impl->features, true, error))
      return nullptr;
  if (!impl->Upload(tensors, 2 * (num_conv + 1), impl->features, false,
                    error))
    return nullptr;
  if (impl->convs.back().rows != 3 * impl->scale * impl->scale) {
    SetError(error, "Real-ESRGAN last convolution does not match the scale");
    return nullptr;
  }
  return std::unique_ptr<Upscaler>(new Upscaler(std::move(impl)));
}

void Upscaler::SetTilePixels(std::size_t pixels) {
  impl_->tile_pixels = pixels == 0 ? kDefaultTilePixels : pixels;
}

bool Upscaler::UpscaleNetwork(const std::uint8_t* rgb24, int w, int h,
                              float* out, std::string* error) {
  if (w <= 0 || h <= 0) {
    SetError(error, "Real-ESRGAN input is empty");
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(w) * h * impl_->scale *
                            impl_->scale * 3 * sizeof(float);
  return impl_->UploadFrame(rgb24, w, h, error) &&
         impl_->Network(w, h, error) &&
         Check(hipMemcpy(out, impl_->picture.data, bytes,
                         hipMemcpyDeviceToHost),
               "download", error);
}

bool Upscaler::Upscale(const std::uint8_t* rgb24, int w, int h, int out_w,
                       int out_h, Fit fit, std::uint8_t* out24,
                       std::string* error) {
  if (w <= 0 || h <= 0 || out_w <= 0 || out_h <= 0 || out_w > 16384 ||
      out_h > 16384) {
    SetError(error, "Real-ESRGAN sizes must be 1..16384");
    return false;
  }
  const Placement place = PlaceContent(w, h, out_w, out_h, fit);
  const int H = h * impl_->scale, W = w * impl_->scale;
  Impl& m = *impl_;
  if (!m.UploadFrame(rgb24, w, h, error) || !m.Network(w, h, error) ||
      !m.rows.Reserve(static_cast<std::size_t>(H) * out_w * 3 * sizeof(float),
                      error) ||
      !m.out24.Reserve(static_cast<std::size_t>(out_w) * out_h * 3, error))
    return false;
  ResizeHorizontalKernel<<<Grid(static_cast<std::size_t>(H) * out_w),
                           kThreads>>>(m.picture.as<float>(), H, W, out_w,
                                       place.content_w, place.offset_x,
                                       m.rows.as<float>());
  ResizeVerticalKernel<<<Grid(static_cast<std::size_t>(out_h) * out_w),
                         kThreads>>>(m.rows.as<float>(), H, out_w, out_h,
                                     place.content_h, place.offset_y,
                                     m.out24.as<std::uint8_t>());
  return Check(hipGetLastError(), "resize", error) &&
         Check(hipMemcpy(out24, m.out24.data,
                         static_cast<std::size_t>(out_w) * out_h * 3,
                         hipMemcpyDeviceToHost),
               "download", error);
}

int Upscaler::scale() const {
  return impl_->scale;
}
int Upscaler::convolutions() const {
  return static_cast<int>(impl_->convs.size());
}
Precision Upscaler::precision() const {
  return impl_->precision;
}
std::size_t Upscaler::weight_bytes() const {
  return impl_->weight_bytes;
}
std::size_t Upscaler::workspace_bytes() const {
  const Impl& m = *impl_;
  return m.frame24.bytes + m.tile_in.bytes + m.act[0].bytes + m.act[1].bytes +
         m.raw.bytes + m.picture.bytes + m.rows.bytes + m.out24.bytes;
}

}  // namespace gufo::models::realesrgan
