#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/rife/ifnet.hpp"

// Layout: every tensor is channels-last [height, width, channels] for a single
// frame. Full-resolution state (images, encoder features, flow, mask) is FP32;
// IFBlock activations are FP16 (WMMA path) or FP32 (direct path).
//
// IFBlock of arch 4.7 (ComfyUI-Frame-Interpolation rife_arch.py):
//   x    = interpolate(cat(inputs), 1 / scale) (+ interpolate(flow) / scale)
//   feat = lrelu(conv3x3 s2(lrelu(conv3x3 s2(x))))
//   feat = 8 x lrelu(conv3x3(feat) * beta + feat)
//   tmp  = PixelShuffle(2)(ConvTranspose2d(c, 24, 4, 2, 1)(feat))
//   flow = interpolate(tmp[:4], scale) * scale, mask = interpolate(tmp[4])
// ConvTranspose2d(k4, s2, p1) is a 3x3 convolution with 4x the output rows
// followed by depth-to-space; with the PixelShuffle it is a 3x3 convolution
// whose rows scatter into a 4x4 patch, so every convolution of the network
// runs through one 3x3 kernel.

namespace gufo::models::rife {
namespace {

using Half = _Float16;
using Half16 = __attribute__((ext_vector_type(16))) _Float16;
using Acc8 = __attribute__((ext_vector_type(8))) float;

constexpr int kThreads = 256;
constexpr int kPad = 64;          // frames are padded to multiples of 64
constexpr int kBlocks = 4;        // IFBlocks of arch 4.7
constexpr int kResConvs = 8;      // ResConv layers per IFBlock
constexpr int kMaskChannels = 5;  // flow (4) + mask (1); channel 5 unused
constexpr std::array<int, kBlocks> kBlockWidth = {192, 128, 96, 64};

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

bool Check(hipError_t status, const char* what, std::string* error) {
  if (status == hipSuccess)
    return true;
  SetError(error,
           std::string("RIFE ") + what + ": " + hipGetErrorString(status));
  return false;
}

unsigned Grid(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

// ---------------------------------------------------------------- kernels --

struct ConvArgs {
  const void* in;  // [in_h, in_w, cin]
  int in_h, in_w, cin;
  int out_h, out_w, stride;
  int rows;   // output channels (before any depth-to-space)
  int k_pad;  // weight row stride
  const void* weight;
  const float* bias;
  const void* residual;  // same layout as out (normal mode only)
  int lrelu;
  int shuffle;           // 0: out [out_h, out_w, rows] in the activation type
  int shuffle_channels;  // S > 0: FP32 out [out_h * S, out_w * S, channels]
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

__device__ __forceinline__ void ShuffleStore(const ConvArgs& a, int oy, int ox,
                                             int row, float value) {
  const int s = a.shuffle;
  const int c = row / (s * s), sy = (row / s) % s, sx = row % s;
  const std::size_t wide = static_cast<std::size_t>(a.out_w) * s;
  static_cast<float*>(a.out)[((static_cast<std::size_t>(oy) * s + sy) * wide +
                              static_cast<std::size_t>(ox) * s + sx) *
                                 a.shuffle_channels +
                             c] = value;
}

// One thread per output element, FP32 accumulation: the validation path.
template<typename T>
__global__ void DirectConvKernel(ConvArgs a) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t pixels = static_cast<std::size_t>(a.out_h) * a.out_w;
  if (index >= pixels * a.rows)
    return;
  const int row = static_cast<int>(index % a.rows);
  const std::size_t p = index / a.rows;
  const int oy = static_cast<int>(p / a.out_w);
  const int ox = static_cast<int>(p % a.out_w);
  const T* in = static_cast<const T*>(a.in);
  const float* w = static_cast<const float*>(a.weight) +
                   static_cast<std::size_t>(row) * a.k_pad;
  float acc = a.bias[row];
  for (int ky = 0; ky < 3; ++ky) {
    const int iy = oy * a.stride - 1 + ky;
    if (iy < 0 || iy >= a.in_h)
      continue;
    for (int kx = 0; kx < 3; ++kx) {
      const int ix = ox * a.stride - 1 + kx;
      if (ix < 0 || ix >= a.in_w)
        continue;
      const T* source =
          in + (static_cast<std::size_t>(iy) * a.in_w + ix) * a.cin;
      const float* tap = w + (ky * 3 + kx) * a.cin;
      for (int c = 0; c < a.cin; ++c)
        acc += ToFloat(source[c]) * tap[c];
    }
  }
  if (a.shuffle != 0) {
    ShuffleStore(a, oy, ox, row, acc);
    return;
  }
  const std::size_t o = p * a.rows + row;
  if (a.residual != nullptr)
    acc += ToFloat(static_cast<const T*>(a.residual)[o]);
  if (a.lrelu != 0 && acc < 0.0F)
    acc *= 0.2F;
  static_cast<T*>(a.out)[o] = FromFloat<T>(acc);
}

// Implicit-GEMM 3x3 convolution, FP16 WMMA with FP32 accumulation. A block
// computes 64 output pixels x 64 rows with four waves (2 x 2, each 32 x 32);
// K = tap * cin + channel advances 32 per stage through LDS while the next
// stage is fetched into registers. cin is a multiple of 16 (8 channels of a
// load never straddle a tap); weights are [rows padded to 64, k_pad] with
// zeros past 9 * cin and past `rows`.
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
  const int pixels = a.out_h * a.out_w;
  const int p0 = blockIdx.x * kTileM, r0 = blockIdx.y * kTileN;
  const int group = tid & 3, line = tid >> 2;
  int oy[2], ox[2];
  bool valid[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int p = p0 + line + 32 * i;
    valid[i] = p < pixels;
    oy[i] = p / a.out_w;
    ox[i] = p % a.out_w;
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
      const int iy = oy[i] * a.stride - 1 + ky;
      const int ix = ox[i] * a.stride - 1 + kx;
      if (valid[i] && k < k_total && iy >= 0 && iy < a.in_h && ix >= 0 &&
          ix < a.in_w)
        v = *reinterpret_cast<const uint4*>(
            in + (static_cast<std::size_t>(iy) * a.in_w + ix) * a.cin + c);
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
    if (a.shuffle != 0) {
      const int y = p / a.out_w, x = p % a.out_w;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (row + e < a.rows)
          ShuffleStore(a, y, x, row + e, v[e]);
      continue;
    }
    const std::size_t o = static_cast<std::size_t>(p) * a.rows + row;
    if (a.residual != nullptr) {
      union {
        uint4 u;
        Half h[8];
      } r;
      r.u = *reinterpret_cast<const uint4*>(
          static_cast<const Half*>(a.residual) + o);
#pragma unroll
      for (int e = 0; e < 8; ++e)
        v[e] += static_cast<float>(r.h[e]);
    }
    union {
      uint4 u;
      Half h[8];
    } w;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float x = a.lrelu != 0 && v[e] < 0.0F ? v[e] * 0.2F : v[e];
      w.h[e] = static_cast<Half>(x);
    }
    *reinterpret_cast<uint4*>(static_cast<Half*>(a.out) + o) = w.u;
  }
}

// RGB24 [h, w, 3] -> FP32 image [H, W, 4] (zero padding) and the encoder
// input [H, W, 16] (channels 0..2, the rest zero).
template<typename T>
__global__ void PrepareKernel(const std::uint8_t* rgb, int h, int w, int H,
                              int W, float4* image, T* encoder_input) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(H) * W)
    return;
  const int y = static_cast<int>(index / W), x = static_cast<int>(index % W);
  float4 v = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  if (y < h && x < w) {
    const std::uint8_t* s = rgb + (static_cast<std::size_t>(y) * w + x) * 3;
    v = make_float4(s[0] / 255.0F, s[1] / 255.0F, s[2] / 255.0F, 0.0F);
  }
  image[index] = v;
  T* e = encoder_input + index * 16;
  e[0] = FromFloat<T>(v.x);
  e[1] = FromFloat<T>(v.y);
  e[2] = FromFloat<T>(v.z);
  for (int c = 3; c < 16; ++c)
    e[c] = FromFloat<T>(0.0F);
}

__device__ __forceinline__ float4 Lerp4(float4 a, float4 b, float t) {
  return make_float4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
                     a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t);
}

// grid_sample(bilinear, padding_mode="border", align_corners=True) at pixel
// coordinates (x, y) = (column + flow_x, row + flow_y).
__device__ __forceinline__ float4 Warp(const float4* image, int H, int W,
                                       float x, float y) {
  x = fminf(fmaxf(x, 0.0F), static_cast<float>(W - 1));
  y = fminf(fmaxf(y, 0.0F), static_cast<float>(H - 1));
  const int x0 = static_cast<int>(floorf(x)), y0 = static_cast<int>(floorf(y));
  const float fx = x - x0, fy = y - y0;
  const int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
  const float4 a = image[static_cast<std::size_t>(y0) * W + x0];
  const float4 b = image[static_cast<std::size_t>(y0) * W + x1];
  const float4 c = image[static_cast<std::size_t>(y1) * W + x0];
  const float4 d = image[static_cast<std::size_t>(y1) * W + x1];
  const float wa = (1.0F - fx) * (1.0F - fy), wb = fx * (1.0F - fy);
  const float wc = (1.0F - fx) * fy, wd = fx * fy;
  return make_float4(a.x * wa + b.x * wb + c.x * wc + d.x * wd,
                     a.y * wa + b.y * wb + c.y * wc + d.y * wd,
                     a.z * wa + b.z * wb + c.z * wc + d.z * wd,
                     a.w * wa + b.w * wb + c.w * wc + d.w * wd);
}

// Bilinear source index of F.interpolate(align_corners=False) with an
// explicit scale factor: src = (dst + 0.5) * ratio - 0.5, clamped at 0.
struct Tap {
  int i0, i1;
  float lambda;
};
__device__ __forceinline__ Tap SourceTap(int dst, float ratio, int size) {
  float src = (static_cast<float>(dst) + 0.5F) * ratio - 0.5F;
  if (src < 0.0F)
    src = 0.0F;
  const int i0 = static_cast<int>(src);
  return {i0, i0 < size - 1 ? i0 + 1 : i0, src - static_cast<float>(i0)};
}

struct BlockInputArgs {
  const float4* image0;
  const float4* image1;
  const float4* feature0;
  const float4* feature1;
  const float4* flow;  // nullptr for the first block
  const float* mask;
  int H, W;  // full (padded) resolution
  int h, w;  // block input resolution
  float scale;
  float timestep;
  int cin;  // padded channels of the output
  void* out;
};

// IFBlock input at 1 / scale: block 0 takes cat(img0, img1, f0, f1, t) (15
// channels); later blocks cat(warp(img0), warp(img1), warp(f0), warp(f1), t,
// mask) (16) plus flow / scale (4), each resized from full resolution.
template<typename T>
__global__ void BlockInputKernel(BlockInputArgs a) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(a.h) * a.w)
    return;
  const int y = static_cast<int>(index / a.w),
            x = static_cast<int>(index % a.w);
  const Tap ty = SourceTap(y, a.scale, a.H), tx = SourceTap(x, a.scale, a.W);
  float v[20] = {};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int yy = q < 2 ? ty.i0 : ty.i1, xx = (q & 1) == 0 ? tx.i0 : tx.i1;
    const float wy = q < 2 ? 1.0F - ty.lambda : ty.lambda;
    const float wx = (q & 1) == 0 ? 1.0F - tx.lambda : tx.lambda;
    const float weight = wy * wx;
    const std::size_t pixel = static_cast<std::size_t>(yy) * a.W + xx;
    float4 i0, i1, f0, f1;
    if (a.flow == nullptr) {
      i0 = a.image0[pixel];
      i1 = a.image1[pixel];
      f0 = a.feature0[pixel];
      f1 = a.feature1[pixel];
    } else {
      const float4 fl = a.flow[pixel];
      const float x0 = static_cast<float>(xx) + fl.x;
      const float y0 = static_cast<float>(yy) + fl.y;
      const float x1 = static_cast<float>(xx) + fl.z;
      const float y1 = static_cast<float>(yy) + fl.w;
      i0 = Warp(a.image0, a.H, a.W, x0, y0);
      i1 = Warp(a.image1, a.H, a.W, x1, y1);
      f0 = Warp(a.feature0, a.H, a.W, x0, y0);
      f1 = Warp(a.feature1, a.H, a.W, x1, y1);
      v[15] += weight * a.mask[pixel];
      v[16] += weight * fl.x;
      v[17] += weight * fl.y;
      v[18] += weight * fl.z;
      v[19] += weight * fl.w;
    }
    v[0] += weight * i0.x;
    v[1] += weight * i0.y;
    v[2] += weight * i0.z;
    v[3] += weight * i1.x;
    v[4] += weight * i1.y;
    v[5] += weight * i1.z;
    v[6] += weight * f0.x;
    v[7] += weight * f0.y;
    v[8] += weight * f0.z;
    v[9] += weight * f0.w;
    v[10] += weight * f1.x;
    v[11] += weight * f1.y;
    v[12] += weight * f1.z;
    v[13] += weight * f1.w;
  }
  // The timestep plane is constant, its resize is exact.
  v[14] = a.timestep;
  const float inverse = 1.0F / a.scale;
  for (int c = 16; c < 20; ++c)
    v[c] *= inverse;
  const int used = a.flow == nullptr ? 15 : 20;
  T* out = static_cast<T*>(a.out) + index * a.cin;
  for (int c = 0; c < a.cin; ++c)
    out[c] = FromFloat<T>(c < used ? v[c] : 0.0F);
}

// tmp [h, w, 5] at block resolution -> full resolution: flow (+)= up(tmp[:4])
// * scale, mask = up(tmp[4]).
__global__ void UpsampleFlowKernel(const float* tmp, int h, int w, int H, int W,
                                   float scale, int first, float4* flow,
                                   float* mask) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(H) * W)
    return;
  const int y = static_cast<int>(index / W), x = static_cast<int>(index % W);
  const float ratio = 1.0F / scale;
  const Tap ty = SourceTap(y, ratio, h), tx = SourceTap(x, ratio, w);
  float v[kMaskChannels];
  const float* r00 = tmp + (static_cast<std::size_t>(ty.i0) * w + tx.i0) * 5;
  const float* r01 = tmp + (static_cast<std::size_t>(ty.i0) * w + tx.i1) * 5;
  const float* r10 = tmp + (static_cast<std::size_t>(ty.i1) * w + tx.i0) * 5;
  const float* r11 = tmp + (static_cast<std::size_t>(ty.i1) * w + tx.i1) * 5;
  // upsample_bilinear2d: h0lambda * (w0lambda * a + w1lambda * b) + ...
  const float h1 = ty.lambda, h0 = 1.0F - h1, w1 = tx.lambda, w0 = 1.0F - w1;
#pragma unroll
  for (int c = 0; c < kMaskChannels; ++c)
    v[c] = h0 * (w0 * r00[c] + w1 * r01[c]) + h1 * (w0 * r10[c] + w1 * r11[c]);
  float4 f =
      make_float4(v[0] * scale, v[1] * scale, v[2] * scale, v[3] * scale);
  if (first == 0) {
    const float4 old = flow[index];
    f = make_float4(old.x + f.x, old.y + f.y, old.z + f.z, old.w + f.w);
  }
  flow[index] = f;
  mask[index] = v[4];
}

// merged = warp(img0) * sigmoid(mask) + warp(img1) * (1 - sigmoid(mask)),
// cropped to h x w and clamped to [0, 1].
__global__ void MergeKernel(const float4* image0, const float4* image1,
                            const float4* flow, const float* mask, int H, int W,
                            int h, int w, std::uint8_t* rgb24, float* rgb) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(h) * w)
    return;
  const int y = static_cast<int>(index / w), x = static_cast<int>(index % w);
  const std::size_t pixel = static_cast<std::size_t>(y) * W + x;
  const float4 fl = flow[pixel];
  const float4 a = Warp(image0, H, W, x + fl.x, y + fl.y);
  const float4 b = Warp(image1, H, W, x + fl.z, y + fl.w);
  const float m = 1.0F / (1.0F + expf(-mask[pixel]));
  const float out[3] = {a.x * m + b.x * (1.0F - m), a.y * m + b.y * (1.0F - m),
                        a.z * m + b.z * (1.0F - m)};
  for (int c = 0; c < 3; ++c) {
    const float v = fminf(fmaxf(out[c], 0.0F), 1.0F);
    if (rgb != nullptr)
      rgb[index * 3 + c] = v;
    if (rgb24 != nullptr)
      rgb24[index * 3 + c] = static_cast<std::uint8_t>(v * 255.0F + 0.5F);
  }
}

// ------------------------------------------------------------------ host --

struct HostTensor {
  std::vector<std::size_t> shape;
  std::vector<float> data;
};

bool ReadSafetensors(const std::filesystem::path& path,
                     std::map<std::string, HostTensor>* tensors,
                     std::string* error) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    SetError(error, "cannot open RIFE weights " + path.string());
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
      if (name == "__metadata__")
        continue;
      const json::Value* shape = entry.find("shape");
      const json::Value* offsets = entry.find("data_offsets");
      if (entry.member_str("dtype") != "F32" || shape == nullptr ||
          !shape->is_array() || offsets == nullptr || !offsets->is_array() ||
          offsets->size() != 2) {
        SetError(error, "RIFE tensor " + name + " is not F32");
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
        SetError(error, "invalid RIFE tensor extent " + name);
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

// A 3x3 convolution as uploaded: rows x (9 * cin) with k = tap * cin + c.
struct Conv {
  int cin{0};  // padded input channels
  int rows{0};
  int rows_pad{0};
  int k_pad{0};
  void* weight{nullptr};  // FP16 [rows_pad, k_pad] or FP32 [rows, 9 * cin]
  float* bias{nullptr};   // [rows_pad]
};

struct HostConv {
  int cin{0};
  int rows{0};
  std::vector<float> weight;  // [rows, 9 * cin]
  std::vector<float> bias;    // [rows]
};

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

// nn.Conv2d(cin_real, rows, 3) weight [rows, cin_real, 3, 3]; beta folds a
// ResConv's per-channel scale into weight and bias.
bool BuildConv(const std::map<std::string, HostTensor>& tensors,
               const std::string& prefix, int cin_pad, const float* beta,
               HostConv* conv, std::string* error) {
  const auto w = tensors.find(prefix + ".weight");
  const auto b = tensors.find(prefix + ".bias");
  if (w == tensors.end() || b == tensors.end() || w->second.shape.size() != 4 ||
      w->second.shape[2] != 3 || w->second.shape[3] != 3 ||
      static_cast<int>(w->second.shape[1]) > cin_pad ||
      b->second.data.size() != w->second.shape[0]) {
    SetError(error, "RIFE weights lack a 3x3 convolution " + prefix);
    return false;
  }
  const int rows = static_cast<int>(w->second.shape[0]);
  const int cin = static_cast<int>(w->second.shape[1]);
  conv->cin = cin_pad;
  conv->rows = rows;
  conv->weight.assign(static_cast<std::size_t>(rows) * 9 * cin_pad, 0.0F);
  conv->bias.resize(rows);
  for (int r = 0; r < rows; ++r) {
    const float s = beta != nullptr ? beta[r] : 1.0F;
    conv->bias[r] = b->second.data[r] * s;
    for (int c = 0; c < cin; ++c)
      for (int tap = 0; tap < 9; ++tap)
        conv->weight[(static_cast<std::size_t>(r) * 9 + tap) * cin_pad + c] =
            w->second.data[(static_cast<std::size_t>(r) * cin + c) * 9 + tap] *
            s;
  }
  return true;
}

// nn.ConvTranspose2d(cin, cout, 4, 2, 1) weight [cin, cout, 4, 4] as a 3x3
// convolution whose rows scatter into an S x S patch (depth-to-space):
// row = c * S * S + qy * S + qx. S = 2 is the plain transposed convolution
// (c = cout channel, (qy, qx) its output parity); S = 4 folds a following
// PixelShuffle(2) (deconv channel = c * 4 + (qy % 2) * 2 + qx % 2, parity
// (qy / 2, qx / 2)). Only the first `channels` of c are kept.
bool BuildDeconv(const std::map<std::string, HostTensor>& tensors,
                 const std::string& prefix, int cin_pad, int s, int channels,
                 HostConv* conv, std::string* error) {
  const auto w = tensors.find(prefix + ".weight");
  const auto b = tensors.find(prefix + ".bias");
  if (w == tensors.end() || b == tensors.end() || w->second.shape.size() != 4 ||
      w->second.shape[2] != 4 || w->second.shape[3] != 4 ||
      static_cast<int>(w->second.shape[0]) > cin_pad) {
    SetError(error, "RIFE weights lack a transposed convolution " + prefix);
    return false;
  }
  const int cin = static_cast<int>(w->second.shape[0]);
  const int cout = static_cast<int>(w->second.shape[1]);
  const int rows = channels * s * s;
  if ((s == 2 && channels > cout) || (s == 4 && channels * 4 > cout) ||
      (s != 2 && s != 4)) {
    SetError(error, "RIFE transposed convolution shape " + prefix);
    return false;
  }
  conv->cin = cin_pad;
  conv->rows = rows;
  conv->weight.assign(static_cast<std::size_t>(rows) * 9 * cin_pad, 0.0F);
  conv->bias.resize(rows);
  for (int row = 0; row < rows; ++row) {
    const int c = row / (s * s), qy = (row / s) % s, qx = row % s;
    int co, ry, rx;
    if (s == 2) {
      co = c;
      ry = qy;
      rx = qx;
    } else {
      co = c * 4 + (qy % 2) * 2 + qx % 2;
      ry = qy / 2;
      rx = qx / 2;
    }
    conv->bias[row] = b->second.data[co];
    for (int dy = 0; dy < 3; ++dy) {
      const int ky = ry + 1 - 2 * (dy - 1);
      if (ky < 0 || ky > 3)
        continue;
      for (int dx = 0; dx < 3; ++dx) {
        const int kx = rx + 1 - 2 * (dx - 1);
        if (kx < 0 || kx > 3)
          continue;
        for (int ci = 0; ci < cin; ++ci)
          conv->weight[(static_cast<std::size_t>(row) * 9 + dy * 3 + dx) *
                           cin_pad +
                       ci] =
              w->second
                  .data[((static_cast<std::size_t>(ci) * cout + co) * 4 + ky) *
                            4 +
                        kx];
      }
    }
  }
  return true;
}

struct Block {
  Conv conv0a, conv0b;
  std::array<Conv, kResConvs> res;
  Conv last;
  int cin{0};
  int width{0};
};

}  // namespace

struct Interpolator::Impl {
  Precision precision{Precision::kFp16};
  hipStream_t stream{nullptr};
  std::vector<void*> allocations;
  std::size_t weight_bytes{0};
  std::size_t workspace_bytes{0};

  Conv encode0, encode1;
  std::array<Block, kBlocks> blocks;

  // Geometry.
  int w{0}, h{0}, W{0}, H{0};
  std::array<float, kBlocks> scales{};
  // Workspace.
  std::array<float4*, kSlots> image{};
  std::array<float4*, kSlots> feature{};
  float4* flow{nullptr};
  float* mask{nullptr};
  float* tmp{nullptr};
  void* x{nullptr};
  void* a{nullptr};
  void* b{nullptr};
  void* c{nullptr};
  std::uint8_t* rgb_in{nullptr};
  std::uint8_t* rgb_out{nullptr};
  float* rgb_float{nullptr};
  std::vector<void*> workspace;

  ~Impl() {
    FreeWorkspace();
    for (void* p : allocations)
      (void)hipFree(p);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }

  [[nodiscard]] std::size_t activation_size() const {
    return precision == Precision::kFp16 ? 2 : sizeof(float);
  }

  void FreeWorkspace() {
    for (void* p : workspace)
      (void)hipFree(p);
    workspace.clear();
    workspace_bytes = 0;
  }

  bool Alloc(std::size_t bytes, void** out, std::vector<void*>* list,
             std::size_t* total, std::string* error) {
    if (!Check(hipMalloc(out, std::max<std::size_t>(bytes, 16)), "hipMalloc",
               error))
      return false;
    list->push_back(*out);
    *total += bytes;
    return true;
  }

  bool Upload(const HostConv& host, Conv* conv, std::string* error) {
    conv->cin = host.cin;
    conv->rows = host.rows;
    const int k = 9 * host.cin;
    if (precision == Precision::kFp16) {
      conv->rows_pad = RoundUp(host.rows, kTileN);
      conv->k_pad = RoundUp(k, kTileK);
      std::vector<std::uint16_t> weight(
          static_cast<std::size_t>(conv->rows_pad) * conv->k_pad, 0);
      for (int r = 0; r < host.rows; ++r)
        for (int i = 0; i < k; ++i)
          weight[static_cast<std::size_t>(r) * conv->k_pad + i] =
              HalfBits(host.weight[static_cast<std::size_t>(r) * k + i]);
      if (!Alloc(weight.size() * 2, &conv->weight, &allocations, &weight_bytes,
                 error) ||
          !Check(hipMemcpy(conv->weight, weight.data(), weight.size() * 2,
                           hipMemcpyHostToDevice),
                 "weight upload", error))
        return false;
    } else {
      conv->rows_pad = host.rows;
      conv->k_pad = k;
      if (!Alloc(host.weight.size() * sizeof(float), &conv->weight,
                 &allocations, &weight_bytes, error) ||
          !Check(hipMemcpy(conv->weight, host.weight.data(),
                           host.weight.size() * sizeof(float),
                           hipMemcpyHostToDevice),
                 "weight upload", error))
        return false;
    }
    std::vector<float> bias(conv->rows_pad, 0.0F);
    std::copy(host.bias.begin(), host.bias.end(), bias.begin());
    void* bias_ptr = nullptr;
    if (!Alloc(bias.size() * sizeof(float), &bias_ptr, &allocations,
               &weight_bytes, error) ||
        !Check(hipMemcpy(bias_ptr, bias.data(), bias.size() * sizeof(float),
                         hipMemcpyHostToDevice),
               "bias upload", error))
      return false;
    conv->bias = static_cast<float*>(bias_ptr);
    return true;
  }

  bool Load(const std::filesystem::path& path, std::string* error) {
    std::map<std::string, HostTensor> tensors;
    if (!ReadSafetensors(path, &tensors, error))
      return false;
    if (tensors.count("encode.0.weight") == 0 ||
        tensors.count("block0.conv0.0.0.weight") == 0 ||
        tensors.at("block0.conv0.0.0.weight").shape[1] != 15) {
      SetError(error, path.string() +
                          " is not a RIFE 4.7 IFNet (rife47 / rife49) "
                          "state_dict");
      return false;
    }
    if (!Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
               "stream", error))
      return false;
    HostConv host;
    if (!BuildConv(tensors, "encode.0", 16, nullptr, &host, error) ||
        !Upload(host, &encode0, error) ||
        !BuildDeconv(tensors, "encode.1", 16, 2, 4, &host, error) ||
        !Upload(host, &encode1, error))
      return false;
    for (int i = 0; i < kBlocks; ++i) {
      Block& block = blocks[i];
      const std::string p = "block" + std::to_string(i);
      block.cin = i == 0 ? 16 : 32;
      block.width = kBlockWidth[i];
      const auto beta_key = [&](int k) {
        return p + ".convblock." + std::to_string(k) + ".beta";
      };
      if (!BuildConv(tensors, p + ".conv0.0.0", block.cin, nullptr, &host,
                     error) ||
          !Upload(host, &block.conv0a, error) ||
          !BuildConv(tensors, p + ".conv0.1.0", host.rows, nullptr, &host,
                     error) ||
          !Upload(host, &block.conv0b, error))
        return false;
      if (block.conv0b.rows != block.width || block.conv0a.rows % 16 != 0) {
        SetError(error, "unexpected RIFE block width in " + p);
        return false;
      }
      for (int k = 0; k < kResConvs; ++k) {
        const auto beta = tensors.find(beta_key(k));
        if (beta == tensors.end() ||
            static_cast<int>(beta->second.data.size()) != block.width) {
          SetError(error, "RIFE weights lack " + beta_key(k));
          return false;
        }
        if (!BuildConv(tensors, p + ".convblock." + std::to_string(k) + ".conv",
                       block.width, beta->second.data.data(), &host, error) ||
            !Upload(host, &block.res[k], error))
          return false;
      }
      if (!BuildDeconv(tensors, p + ".lastconv.0", block.width, 4,
                       kMaskChannels, &host, error) ||
          !Upload(host, &block.last, error))
        return false;
    }
    return true;
  }

  bool Configure(int width, int height, float scale_factor,
                 std::string* error) {
    if (width < 2 || height < 2 || width > 8192 || height > 8192) {
      SetError(error, "RIFE frame size out of range");
      return false;
    }
    if (!(scale_factor == 0.5F || scale_factor == 1.0F ||
          scale_factor == 2.0F || scale_factor == 4.0F)) {
      SetError(error, "RIFE scale_factor must be 0.5, 1, 2 or 4");
      return false;
    }
    FreeWorkspace();
    w = width;
    h = height;
    W = RoundUp(width, kPad);
    H = RoundUp(height, kPad);
    std::size_t x_bytes = 0, feature_bytes = 0, tmp_bytes = 0;
    for (int i = 0; i < kBlocks; ++i) {
      scales[i] = static_cast<float>(8 >> i) / scale_factor;
      const int bh = static_cast<int>(H / scales[i]);
      const int bw = static_cast<int>(W / scales[i]);
      if (bh * scales[i] != static_cast<float>(H) ||
          bw * scales[i] != static_cast<float>(W) || bh % 4 != 0 ||
          bw % 4 != 0) {
        SetError(error, "RIFE block resolution is not integral");
        return false;
      }
      const Block& block = blocks[i];
      x_bytes =
          std::max(x_bytes, static_cast<std::size_t>(bh) * bw * block.cin);
      feature_bytes = std::max(
          {feature_bytes,
           static_cast<std::size_t>(bh / 2) * (bw / 2) * block.conv0a.rows,
           static_cast<std::size_t>(bh / 4) * (bw / 4) * block.width});
      tmp_bytes = std::max(tmp_bytes, static_cast<std::size_t>(bh) * bw *
                                          kMaskChannels * sizeof(float));
    }
    const std::size_t full = static_cast<std::size_t>(H) * W;
    // The encoder input [H, W, 16] and its stride-2 output share x / a.
    x_bytes = std::max(x_bytes, full * 16) * activation_size();
    feature_bytes = std::max(feature_bytes, full / 4 * 16) * activation_size();
    void* p = nullptr;
    for (int s = 0; s < kSlots; ++s) {
      if (!Alloc(full * sizeof(float4), &p, &workspace, &workspace_bytes,
                 error))
        return false;
      image[s] = static_cast<float4*>(p);
      if (!Alloc(full * sizeof(float4), &p, &workspace, &workspace_bytes,
                 error))
        return false;
      feature[s] = static_cast<float4*>(p);
    }
    if (!Alloc(full * sizeof(float4), &p, &workspace, &workspace_bytes, error))
      return false;
    flow = static_cast<float4*>(p);
    if (!Alloc(full * sizeof(float), &p, &workspace, &workspace_bytes, error))
      return false;
    mask = static_cast<float*>(p);
    if (!Alloc(tmp_bytes, &p, &workspace, &workspace_bytes, error))
      return false;
    tmp = static_cast<float*>(p);
    if (!Alloc(x_bytes, &x, &workspace, &workspace_bytes, error) ||
        !Alloc(feature_bytes, &a, &workspace, &workspace_bytes, error) ||
        !Alloc(feature_bytes, &b, &workspace, &workspace_bytes, error) ||
        !Alloc(feature_bytes, &c, &workspace, &workspace_bytes, error))
      return false;
    const std::size_t pixels = static_cast<std::size_t>(w) * h;
    if (!Alloc(pixels * 3, &p, &workspace, &workspace_bytes, error))
      return false;
    rgb_in = static_cast<std::uint8_t*>(p);
    if (!Alloc(pixels * 3, &p, &workspace, &workspace_bytes, error))
      return false;
    rgb_out = static_cast<std::uint8_t*>(p);
    if (!Alloc(pixels * 3 * sizeof(float), &p, &workspace, &workspace_bytes,
               error))
      return false;
    rgb_float = static_cast<float*>(p);
    return true;
  }

  // 3x3 convolution `conv` of in [in_h, in_w, conv.cin] at `stride`.
  void Run(const Conv& conv, const void* in, int in_h, int in_w, int stride,
           const void* residual, bool lrelu, int shuffle, int channels,
           void* out) {
    ConvArgs args{};
    args.in = in;
    args.in_h = in_h;
    args.in_w = in_w;
    args.cin = conv.cin;
    args.out_h = (in_h - 1) / stride + 1;
    args.out_w = (in_w - 1) / stride + 1;
    args.stride = stride;
    args.rows = conv.rows;
    args.k_pad = conv.k_pad;
    args.weight = conv.weight;
    args.bias = conv.bias;
    args.residual = residual;
    args.lrelu = lrelu ? 1 : 0;
    args.shuffle = shuffle;
    args.shuffle_channels = channels;
    args.out = out;
    const std::size_t pixels =
        static_cast<std::size_t>(args.out_h) * args.out_w;
    if (precision == Precision::kFp16) {
      const dim3 grid(static_cast<unsigned>((pixels + kTileM - 1) / kTileM),
                      static_cast<unsigned>(conv.rows_pad / kTileN));
      WmmaConvKernel<<<grid, 128, 0, stream>>>(args);
    } else {
      DirectConvKernel<float>
          <<<Grid(pixels * conv.rows), kThreads, 0, stream>>>(args);
    }
  }

  bool SetFrame(int slot, const std::uint8_t* rgb24, std::string* error) {
    if (slot < 0 || slot >= kSlots || W == 0) {
      SetError(error, "RIFE frame slot or geometry invalid");
      return false;
    }
    const std::size_t full = static_cast<std::size_t>(H) * W;
    if (!Check(
            hipMemcpyAsync(rgb_in, rgb24, static_cast<std::size_t>(w) * h * 3,
                           hipMemcpyHostToDevice, stream),
            "frame upload", error))
      return false;
    if (precision == Precision::kFp16)
      PrepareKernel<Half><<<Grid(full), kThreads, 0, stream>>>(
          rgb_in, h, w, H, W, image[slot], static_cast<Half*>(x));
    else
      PrepareKernel<float><<<Grid(full), kThreads, 0, stream>>>(
          rgb_in, h, w, H, W, image[slot], static_cast<float*>(x));
    // encode = ConvTranspose2d(16, 4, 4, 2, 1)(Conv2d(3, 16, 3, 2, 1)(img)).
    Run(encode0, x, H, W, 2, nullptr, false, 0, 0, a);
    Run(encode1, a, H / 2, W / 2, 1, nullptr, false, 2, 4, feature[slot]);
    return Check(hipGetLastError(), "encoder", error) &&
           Check(hipStreamSynchronize(stream), "encoder", error);
  }

  bool Interpolate(int slot0, int slot1, float t, std::uint8_t* out24,
                   float* out_float, std::string* error) {
    if (slot0 < 0 || slot0 >= kSlots || slot1 < 0 || slot1 >= kSlots ||
        W == 0) {
      SetError(error, "RIFE frame slot or geometry invalid");
      return false;
    }
    const std::size_t full = static_cast<std::size_t>(H) * W;
    for (int i = 0; i < kBlocks; ++i) {
      const Block& block = blocks[i];
      const float scale = scales[i];
      const int bh = static_cast<int>(H / scale);
      const int bw = static_cast<int>(W / scale);
      BlockInputArgs in{};
      in.image0 = image[slot0];
      in.image1 = image[slot1];
      in.feature0 = feature[slot0];
      in.feature1 = feature[slot1];
      in.flow = i == 0 ? nullptr : flow;
      in.mask = mask;
      in.H = H;
      in.W = W;
      in.h = bh;
      in.w = bw;
      in.scale = scale;
      in.timestep = t;
      in.cin = block.cin;
      in.out = x;
      const std::size_t block_pixels = static_cast<std::size_t>(bh) * bw;
      if (precision == Precision::kFp16)
        BlockInputKernel<Half><<<Grid(block_pixels), kThreads, 0, stream>>>(in);
      else
        BlockInputKernel<float>
            <<<Grid(block_pixels), kThreads, 0, stream>>>(in);
      Run(block.conv0a, x, bh, bw, 2, nullptr, true, 0, 0, a);
      Run(block.conv0b, a, bh / 2, bw / 2, 2, nullptr, true, 0, 0, b);
      void* cur = b;
      void* next = c;
      for (int k = 0; k < kResConvs; ++k) {
        Run(block.res[k], cur, bh / 4, bw / 4, 1, cur, true, 0, 0, next);
        std::swap(cur, next);
      }
      Run(block.last, cur, bh / 4, bw / 4, 1, nullptr, false, 4, kMaskChannels,
          tmp);
      UpsampleFlowKernel<<<Grid(full), kThreads, 0, stream>>>(
          tmp, bh, bw, H, W, scale, i == 0 ? 1 : 0, flow, mask);
    }
    const std::size_t pixels = static_cast<std::size_t>(w) * h;
    MergeKernel<<<Grid(pixels), kThreads, 0, stream>>>(
        image[slot0], image[slot1], flow, mask, H, W, h, w,
        out24 != nullptr ? rgb_out : nullptr,
        out_float != nullptr ? rgb_float : nullptr);
    if (!Check(hipGetLastError(), "IFNet", error))
      return false;
    if (out24 != nullptr &&
        !Check(hipMemcpyAsync(out24, rgb_out, pixels * 3, hipMemcpyDeviceToHost,
                              stream),
               "frame download", error))
      return false;
    if (out_float != nullptr &&
        !Check(hipMemcpyAsync(out_float, rgb_float, pixels * 3 * sizeof(float),
                              hipMemcpyDeviceToHost, stream),
               "frame download", error))
      return false;
    return Check(hipStreamSynchronize(stream), "IFNet", error);
  }
};

Interpolator::Interpolator(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
Interpolator::~Interpolator() = default;

std::unique_ptr<Interpolator> Interpolator::Load(
    const std::filesystem::path& weights, Precision precision,
    std::string* error) {
  auto impl = std::make_unique<Impl>();
  impl->precision = precision;
  if (!impl->Load(weights, error))
    return nullptr;
  return std::unique_ptr<Interpolator>(new Interpolator(std::move(impl)));
}

bool Interpolator::Configure(int width, int height, float scale_factor,
                             std::string* error) {
  return impl_->Configure(width, height, scale_factor, error);
}

bool Interpolator::SetFrame(int slot, const std::uint8_t* rgb24,
                            std::string* error) {
  return impl_->SetFrame(slot, rgb24, error);
}

bool Interpolator::Interpolate(int slot0, int slot1, float t,
                               std::uint8_t* rgb24, float* rgb,
                               std::string* error) {
  return impl_->Interpolate(slot0, slot1, t, rgb24, rgb, error);
}

Precision Interpolator::precision() const {
  return impl_->precision;
}
std::size_t Interpolator::weight_bytes() const {
  return impl_->weight_bytes;
}
std::size_t Interpolator::workspace_bytes() const {
  return impl_->workspace_bytes;
}

}  // namespace gufo::models::rife
