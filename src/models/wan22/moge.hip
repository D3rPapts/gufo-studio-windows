#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/models/wan22/moge.hpp"

namespace gufo::models::wan22 {
namespace {

constexpr int kDim = 1024;
constexpr int kHeads = 16;
constexpr int kHeadDim = 64;
constexpr int kLayers = 24;
constexpr int kMlp = 4096;
constexpr int kPatch = 14;
constexpr int kThreads = 256;
constexpr int kTokens = 3600;  // resolution level 9 of [1200, 3600]
constexpr int kTake[4] = {5, 11, 17, 23};
constexpr int kLevelDims[5] = {1024, 256, 128, 64, 32};

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

bool Check(hipError_t status, const char* operation, std::string* error) {
  if (status == hipSuccess)
    return true;
  SetError(error, std::string("MoGe ") + operation + ": " +
                      hipGetErrorString(status));
  return false;
}

unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

__device__ __forceinline__ float BlockSum(float value, float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  const float result = shared[0];
  __syncthreads();
  return result;
}

// Layer norm of rows of 1024 (eps 1e-6).
__global__ void LayerNormKernel(const float* x, const float* weight,
                                const float* bias, float* out) {
  __shared__ float shared[kThreads];
  const float* row = x + static_cast<std::size_t>(blockIdx.x) * kDim;
  float v[4], sum = 0;
  for (int i = 0; i < 4; ++i) {
    v[i] = row[threadIdx.x + i * kThreads];
    sum += v[i];
  }
  const float mean = BlockSum(sum, shared) / kDim;
  float squares = 0;
  for (int i = 0; i < 4; ++i) {
    v[i] -= mean;
    squares += v[i] * v[i];
  }
  const float rstd = rsqrtf(BlockSum(squares, shared) / kDim + 1e-6F);
  float* target = out + static_cast<std::size_t>(blockIdx.x) * kDim;
  for (int i = 0; i < 4; ++i) {
    const int c = threadIdx.x + i * kThreads;
    target[c] = v[i] * rstd * weight[c] + bias[c];
  }
}

// x[r, c] = act(x[r, c] + bias[c]); act 0 none, 1 exact GELU, 2 ReLU.
__global__ void BiasActKernel(float* x, const float* bias, std::size_t rows,
                              int columns, int act) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * columns)
    return;
  float v = x[i] + bias[i % columns];
  if (act == 1)
    v = 0.5F * v * (1.0F + erff(v * 0.7071067811865476F));
  else if (act == 2)
    v = fmaxf(v, 0.0F);
  x[i] = v;
}

// hidden[r, c] += gamma[c] * y[r, c].
__global__ void ScaledAddKernel(float* hidden, const float* y,
                                const float* gamma, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    hidden[i] += gamma[i % kDim] * y[i];
}

__global__ void AddKernel(float* x, const float* y, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] += y[i];
}

// x[c, p] += bias[c] for NCHW maps of `pixels`.
__global__ void ChannelBiasKernel(float* x, const float* bias, int pixels,
                                  std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] += bias[i / pixels];
}

__global__ void SoftmaxKernel(float* scores, int columns, float scale) {
  __shared__ float shared[kThreads];
  float* row = scores + static_cast<std::size_t>(blockIdx.x) * columns;
  float maximum = -INFINITY;
  for (int c = threadIdx.x; c < columns; c += kThreads)
    maximum = fmaxf(maximum, row[c] * scale);
  shared[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] =
          fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    __syncthreads();
  }
  maximum = shared[0];
  __syncthreads();
  float sum = 0;
  for (int c = threadIdx.x; c < columns; c += kThreads) {
    const float e = expf(row[c] * scale - maximum);
    row[c] = e;
    sum += e;
  }
  sum = BlockSum(sum, shared);
  const float inverse = 1.0F / sum;
  for (int c = threadIdx.x; c < columns; c += kThreads)
    row[c] *= inverse;
}

// Tokens [1 + h * w, 1024] (row 0 = class) -> NCHW [1024, h, w] of the patch
// tokens.
__global__ void TokensToMapKernel(const float* tokens, int pixels, float* map) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(pixels) * kDim)
    return;
  const int c = static_cast<int>(i / pixels);
  const int p = static_cast<int>(i % pixels);
  map[i] = tokens[static_cast<std::size_t>(p) * kDim + c];
}

// NCHW 3x3 (replicate padding) or 1x1 convolution, optional ReLU on the input.
__global__ void ConvKernel(const float* in, int cin, int height, int width,
                           const float* weight, const float* bias, int cout,
                           int k, int relu_in, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t plane = static_cast<std::size_t>(height) * width;
  if (i >= plane * cout)
    return;
  const int co = static_cast<int>(i / plane);
  const int p = static_cast<int>(i % plane);
  const int y = p / width, x = p % width;
  float sum = bias[co];
  const int r = k / 2;
  for (int ci = 0; ci < cin; ++ci) {
    const float* channel = in + ci * plane;
    const float* w = weight + (static_cast<std::size_t>(co) * cin + ci) * k * k;
    for (int dy = 0; dy < k; ++dy) {
      const int sy = min(max(y + dy - r, 0), height - 1);
      for (int dx = 0; dx < k; ++dx) {
        const int sx = min(max(x + dx - r, 0), width - 1);
        float v = channel[sy * width + sx];
        if (relu_in)
          v = fmaxf(v, 0.0F);
        sum += w[dy * k + dx] * v;
      }
    }
  }
  out[i] = sum;
}

// ConvTranspose2d kernel 2, stride 2: weight [cin, cout, 2, 2].
__global__ void ConvTransposeKernel(const float* in, int cin, int height,
                                    int width, const float* weight,
                                    const float* bias, int cout, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int oh = 2 * height, ow = 2 * width;
  const std::size_t plane = static_cast<std::size_t>(oh) * ow;
  if (i >= plane * cout)
    return;
  const int co = static_cast<int>(i / plane);
  const int p = static_cast<int>(i % plane);
  const int y = p / ow, x = p % ow;
  const int sy = y / 2, sx = x / 2, ky = y % 2, kx = x % 2;
  float sum = bias[co];
  const std::size_t in_plane = static_cast<std::size_t>(height) * width;
  for (int ci = 0; ci < cin; ++ci)
    sum += in[ci * in_plane + sy * width + sx] *
           weight[((static_cast<std::size_t>(ci) * cout + co) * 2 + ky) * 2 + kx];
  out[i] = sum;
}

// Bilinear 2x upsampling, align_corners false.
__global__ void Upsample2Kernel(const float* in, int channels, int height,
                                int width, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int oh = 2 * height, ow = 2 * width;
  const std::size_t plane = static_cast<std::size_t>(oh) * ow;
  if (i >= plane * channels)
    return;
  const int c = static_cast<int>(i / plane);
  const int p = static_cast<int>(i % plane);
  const int y = p / ow, x = p % ow;
  const float ry = fmaxf(0.5F * (y + 0.5F) - 0.5F, 0.0F);
  const float rx = fmaxf(0.5F * (x + 0.5F) - 0.5F, 0.0F);
  const int y0 = static_cast<int>(ry), x0 = static_cast<int>(rx);
  const int y1 = min(y0 + 1, height - 1), x1 = min(x0 + 1, width - 1);
  const float ly = ry - y0, lx = rx - x0;
  const float* s = in + static_cast<std::size_t>(c) * height * width;
  out[i] = (1 - ly) * ((1 - lx) * s[y0 * width + x0] + lx * s[y0 * width + x1]) +
           ly * ((1 - lx) * s[y1 * width + x0] + lx * s[y1 * width + x1]);
}

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr)
        (void)hipFree(data_);
      data_ = std::exchange(other.data_, nullptr);
    }
    return *this;
  }
  ~DeviceBuffer() {
    if (data_ != nullptr)
      (void)hipFree(data_);
  }
  bool Allocate(std::size_t bytes, std::string* error) {
    if (data_ != nullptr)
      (void)hipFree(data_);
    data_ = nullptr;
    return Check(hipMalloc(&data_, std::max<std::size_t>(bytes, 256)),
                 "device allocation", error);
  }
  float* f32() const { return static_cast<float*>(data_); }

private:
  void* data_{nullptr};
};

// Torch's antialiased bilinear resampling weights for one axis.
struct Axis {
  std::vector<int> first, count;
  std::vector<double> weights;
  int stride{0};
};

Axis AntialiasAxis(int in, int out) {
  const double scale = static_cast<double>(in) / out;
  const double support = scale >= 1.0 ? scale : 1.0;
  const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
  Axis a;
  a.stride = static_cast<int>(std::ceil(support)) * 2 + 1;
  a.first.resize(out);
  a.count.resize(out);
  a.weights.assign(static_cast<std::size_t>(out) * a.stride, 0.0);
  for (int i = 0; i < out; ++i) {
    const double center = scale * (i + 0.5);
    const int xmin = std::max(static_cast<int>(center - support + 0.5), 0);
    const int xsize =
        std::min(static_cast<int>(center + support + 0.5), in) - xmin;
    double total = 0;
    for (int j = 0; j < xsize; ++j) {
      const double w =
          std::max(0.0, 1.0 - std::abs((j + xmin - center + 0.5) * invscale));
      a.weights[static_cast<std::size_t>(i) * a.stride + j] = w;
      total += w;
    }
    for (int j = 0; j < xsize; ++j)
      if (total != 0.0)
        a.weights[static_cast<std::size_t>(i) * a.stride + j] /= total;
    a.first[i] = xmin;
    a.count[i] = xsize;
  }
  return a;
}

// [C, H, W] -> [C, oh, ow], antialiased bilinear (width pass, then height).
std::vector<float> ResizeAntialias(const std::vector<float>& in, int channels,
                                   int height, int width, int oh, int ow) {
  const Axis ax = AntialiasAxis(width, ow), ay = AntialiasAxis(height, oh);
  std::vector<float> mid(static_cast<std::size_t>(channels) * height * ow);
  for (int c = 0; c < channels; ++c)
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < ow; ++x) {
        double sum = 0;
        for (int j = 0; j < ax.count[x]; ++j)
          sum += ax.weights[static_cast<std::size_t>(x) * ax.stride + j] *
                 in[(static_cast<std::size_t>(c) * height + y) * width +
                    ax.first[x] + j];
        mid[(static_cast<std::size_t>(c) * height + y) * ow + x] =
            static_cast<float>(sum);
      }
  std::vector<float> out(static_cast<std::size_t>(channels) * oh * ow);
  for (int c = 0; c < channels; ++c)
    for (int y = 0; y < oh; ++y)
      for (int x = 0; x < ow; ++x) {
        double sum = 0;
        for (int j = 0; j < ay.count[y]; ++j)
          sum += ay.weights[static_cast<std::size_t>(y) * ay.stride + j] *
                 mid[(static_cast<std::size_t>(c) * height + ay.first[y] + j) *
                         ow +
                     x];
        out[(static_cast<std::size_t>(c) * oh + y) * ow + x] =
            static_cast<float>(sum);
      }
  return out;
}

// Plain bilinear resize (align_corners false, no antialias) of [C, H, W].
std::vector<float> ResizeBilinear(const std::vector<float>& in, int channels,
                                  int height, int width, int oh, int ow) {
  std::vector<float> out(static_cast<std::size_t>(channels) * oh * ow);
  const double sy = static_cast<double>(height) / oh;
  const double sx = static_cast<double>(width) / ow;
  for (int y = 0; y < oh; ++y) {
    const double ry = std::max(sy * (y + 0.5) - 0.5, 0.0);
    const int y0 = static_cast<int>(ry), y1 = std::min(y0 + 1, height - 1);
    const double ly = ry - y0;
    for (int x = 0; x < ow; ++x) {
      const double rx = std::max(sx * (x + 0.5) - 0.5, 0.0);
      const int x0 = static_cast<int>(rx), x1 = std::min(x0 + 1, width - 1);
      const double lx = rx - x0;
      for (int c = 0; c < channels; ++c) {
        const float* s = in.data() + static_cast<std::size_t>(c) * height * width;
        out[(static_cast<std::size_t>(c) * oh + y) * ow + x] = static_cast<float>(
            (1 - ly) * ((1 - lx) * s[y0 * width + x0] + lx * s[y0 * width + x1]) +
            ly * ((1 - lx) * s[y1 * width + x0] + lx * s[y1 * width + x1]));
      }
    }
  }
  return out;
}

double Cubic1(double x, double a) {  // |x| <= 1
  return ((a + 2) * x - (a + 3)) * x * x + 1;
}
double Cubic2(double x, double a) {  // 1 < |x| < 2
  return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
}

// torch bicubic (A = -0.75, align_corners false, scale factors given) of the
// 37x37 position grid [M, M, 1024] to [h, w, 1024].
std::vector<float> InterpolatePositions(const std::vector<float>& grid, int m,
                                        int h, int w) {
  const double sy = (h + 0.1) / m, sx = (w + 0.1) / m;
  std::vector<float> out(static_cast<std::size_t>(h) * w * kDim);
  const auto coefficients = [](double t, double c[4]) {
    constexpr double a = -0.75;
    c[0] = Cubic2(t + 1, a);
    c[1] = Cubic1(t, a);
    c[2] = Cubic1(1 - t, a);
    c[3] = Cubic2(2 - t, a);
  };
  for (int y = 0; y < h; ++y) {
    const double ry = (y + 0.5) / sy - 0.5;
    const int iy = static_cast<int>(std::floor(ry));
    double cy[4];
    coefficients(ry - iy, cy);
    for (int x = 0; x < w; ++x) {
      const double rx = (x + 0.5) / sx - 0.5;
      const int ix = static_cast<int>(std::floor(rx));
      double cx[4];
      coefficients(rx - ix, cx);
      for (int c = 0; c < kDim; ++c) {
        double sum = 0;
        for (int j = 0; j < 4; ++j) {
          const int yy = std::clamp(iy - 1 + j, 0, m - 1);
          double row = 0;
          for (int i = 0; i < 4; ++i) {
            const int xx = std::clamp(ix - 1 + i, 0, m - 1);
            row += cx[i] * grid[(static_cast<std::size_t>(yy) * m + xx) * kDim + c];
          }
          sum += cy[j] * row;
        }
        out[(static_cast<std::size_t>(y) * w + x) * kDim + c] =
            static_cast<float>(sum);
      }
    }
  }
  return out;
}

// normalized_view_plane_uv: [2, height, width] (u, v).
std::vector<float> ViewPlaneUv(int width, int height, double aspect) {
  const double span_x = aspect / std::sqrt(1 + aspect * aspect);
  const double span_y = 1 / std::sqrt(1 + aspect * aspect);
  std::vector<float> uv(static_cast<std::size_t>(2) * width * height);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      const double u0 = -span_x * (width - 1) / width;
      const double v0 = -span_y * (height - 1) / height;
      const double u =
          width > 1 ? u0 + (-2 * u0) * x / (width - 1) : 0.0;
      const double v =
          height > 1 ? v0 + (-2 * v0) * y / (height - 1) : 0.0;
      uv[static_cast<std::size_t>(y) * width + x] = static_cast<float>(u);
      uv[static_cast<std::size_t>(width) * height + y * width + x] =
          static_cast<float>(v);
    }
  return uv;
}

}  // namespace

struct Moge::Impl {
  ~Impl() {
    if (blas != nullptr)
      (void)rocblas_destroy_handle(blas);
  }
  const float* W(const std::string& name) const {
    const auto it = weights.find(name);
    return it == weights.end() ? nullptr : it->second.f32();
  }
  // out[T, R] = in[T, C] . W[R, C]^T (+ bias, act).
  bool Linear(const float* in, int tokens, const std::string& name, int rows,
              int columns, float* out, int act, std::string* error);
  // out[cout, HW] = W[cout, cin] . in[cin, HW] + b (1x1 convolution).
  bool Conv1x1(const float* in, int cin, int pixels, const std::string& name,
               int cout, float* out, std::string* error);
  bool RunStack(const std::string& prefix, const std::vector<int>& res_blocks,
                std::vector<DeviceBuffer>& inputs, const std::vector<int>& in_channels,
                int base_h, int base_w, std::vector<DeviceBuffer>* outputs,
                std::string* error);

  std::map<std::string, DeviceBuffer, std::less<>> weights;
  std::map<std::string, std::vector<std::size_t>, std::less<>> shapes;
  std::vector<float> pos_grid, pos_class;  // host
  std::vector<float> scale_w0, scale_b0, scale_w1, scale_b1, scale_w2, scale_b2;
  rocblas_handle blas{nullptr};
};

Moge::Moge(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Moge::~Moge() = default;

bool Moge::Impl::Linear(const float* in, int tokens, const std::string& name,
                        int rows, int columns, float* out, int act,
                        std::string* error) {
  const float one = 1.0F, zero = 0.0F;
  if (rocblas_sgemm(blas, rocblas_operation_transpose, rocblas_operation_none,
                    rows, tokens, columns, &one, W(name + ".weight"), columns,
                    in, columns, &zero, out, rows) != rocblas_status_success) {
    SetError(error, "MoGe GEMM failed: " + name);
    return false;
  }
  BiasActKernel<<<Blocks(static_cast<std::size_t>(tokens) * rows), kThreads>>>(
      out, W(name + ".bias"), tokens, rows, act);
  return Check(hipGetLastError(), "linear", error);
}

bool Moge::Impl::Conv1x1(const float* in, int cin, int pixels,
                         const std::string& name, int cout, float* out,
                         std::string* error) {
  const float one = 1.0F, zero = 0.0F;
  if (rocblas_sgemm(blas, rocblas_operation_none, rocblas_operation_none,
                    pixels, cout, cin, &one, in, pixels, W(name + ".weight"),
                    cin, &zero, out, pixels) != rocblas_status_success) {
    SetError(error, "MoGe 1x1 convolution failed: " + name);
    return false;
  }
  ChannelBiasKernel<<<Blocks(static_cast<std::size_t>(cout) * pixels),
                      kThreads>>>(out, W(name + ".bias"), pixels,
                                  static_cast<std::size_t>(cout) * pixels);
  return Check(hipGetLastError(), "1x1 convolution", error);
}

// ConvStack.forward: per level, x = input_block(in) (+ x), residual blocks,
// output = x; between levels the resampler (conv-transpose or bilinear, then
// a replicate-padded 3x3 convolution). `inputs[i]` with `in_channels[i]`
// channels at base * 2^i; outputs are the per-level states.
bool Moge::Impl::RunStack(const std::string& prefix,
                          const std::vector<int>& res_blocks,
                          std::vector<DeviceBuffer>& inputs,
                          const std::vector<int>& in_channels, int base_h,
                          int base_w, std::vector<DeviceBuffer>* outputs,
                          std::string* error) {
  outputs->clear();
  DeviceBuffer x, t, u;
  for (int level = 0; level < 5; ++level) {
    const int h = base_h << level, w = base_w << level;
    const std::size_t pixels = static_cast<std::size_t>(h) * w;
    const int c = kLevelDims[level];
    DeviceBuffer feature;
    if (!feature.Allocate(pixels * c * 4, error) ||
        !Conv1x1(inputs[level].f32(), in_channels[level], static_cast<int>(pixels),
                 prefix + ".input_blocks." + std::to_string(level), c,
                 feature.f32(), error))
      return false;
    if (level == 0) {
      x = std::move(feature);
    } else {
      AddKernel<<<Blocks(pixels * c), kThreads>>>(x.f32(), feature.f32(),
                                                  pixels * c);
    }
    if (!t.Allocate(pixels * c * 4, error) || !u.Allocate(pixels * c * 4, error))
      return false;
    for (int r = 0; r < res_blocks[level]; ++r) {
      const std::string p = prefix + ".res_blocks." + std::to_string(level) +
                            "." + std::to_string(r) + ".layers.";
      ConvKernel<<<Blocks(pixels * c), kThreads>>>(
          x.f32(), c, h, w, W(p + "2.weight"), W(p + "2.bias"), c, 3, 1,
          t.f32());
      ConvKernel<<<Blocks(pixels * c), kThreads>>>(
          t.f32(), c, h, w, W(p + "5.weight"), W(p + "5.bias"), c, 3, 1,
          u.f32());
      AddKernel<<<Blocks(pixels * c), kThreads>>>(x.f32(), u.f32(), pixels * c);
    }
    DeviceBuffer copy;
    if (!copy.Allocate(pixels * c * 4, error) ||
        !Check(hipMemcpy(copy.f32(), x.f32(), pixels * c * 4,
                         hipMemcpyDeviceToDevice),
               "copy", error))
      return false;
    outputs->push_back(std::move(copy));
    if (level == 4)
      break;
    // Resampler to the next level.
    const int next = kLevelDims[level + 1];
    const std::size_t next_pixels = pixels * 4;
    const std::string r = prefix + ".resamplers." + std::to_string(level);
    DeviceBuffer up;
    const bool transpose = shapes.count(r + ".0.weight") != 0;
    if (!up.Allocate(next_pixels * std::max(c, next) * 4, error))
      return false;
    int up_channels;
    if (transpose) {
      ConvTransposeKernel<<<Blocks(next_pixels * next), kThreads>>>(
          x.f32(), c, h, w, W(r + ".0.weight"), W(r + ".0.bias"), next,
          up.f32());
      up_channels = next;
    } else {
      Upsample2Kernel<<<Blocks(next_pixels * c), kThreads>>>(x.f32(), c, h, w,
                                                             up.f32());
      up_channels = c;
    }
    DeviceBuffer y;
    if (!y.Allocate(next_pixels * next * 4, error))
      return false;
    ConvKernel<<<Blocks(next_pixels * next), kThreads>>>(
        up.f32(), up_channels, 2 * h, 2 * w, W(r + ".1.weight"),
        W(r + ".1.bias"), next, 3, 0, y.f32());
    x = std::move(y);
    if (!Check(hipGetLastError(), "stack", error))
      return false;
  }
  return Check(hipDeviceSynchronize(), "stack", error);
}

std::unique_ptr<Moge> Moge::Load(const std::filesystem::path& path,
                                 std::string* error) {
  std::string open_error;
  auto file = platform::OpenReadFile(path, &open_error);
  if (!platform::FileValid(file)) {
    SetError(error, "cannot open " + path.string() + ": " + open_error);
    return nullptr;
  }
  struct Closer {
    platform::PlatformFile file;
    ~Closer() { platform::CloseFile(file); }
  } closer{file};
  const auto file_size = platform::FileSizeBytes(file);
  std::uint64_t header_size = 0;
  if (!file_size || platform::ReadFileAt(file, &header_size, 8, 0) != 8 ||
      header_size == 0 || header_size > (32U << 20U) ||
      header_size > *file_size - 8) {
    SetError(error, "invalid safetensors header in " + path.string());
    return nullptr;
  }
  std::string text(static_cast<std::size_t>(header_size), '\0');
  if (platform::ReadFileAt(file, text.data(), text.size(), 8) != text.size()) {
    SetError(error, "short read of " + path.string());
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  if (rocblas_create_handle(&m.blas) != rocblas_status_success) {
    SetError(error, "MoGe rocBLAS handle");
    return nullptr;
  }
  std::vector<float> host;
  try {
    const json::Value header = json::parse(text);
    for (const auto& [name, entry] : header.members()) {
      if (name == "__metadata__")
        continue;
      const json::Value* shape = entry.find("shape");
      const json::Value* offsets = entry.find("data_offsets");
      if (entry.member_str("dtype") != "F32" || shape == nullptr ||
          offsets == nullptr || offsets->size() != 2) {
        SetError(error, "unsupported MoGe tensor " + name);
        return nullptr;
      }
      std::vector<std::size_t> dims;
      std::size_t count = 1;
      for (const auto& dim : shape->items()) {
        dims.push_back(dim.as_size());
        count *= dims.back();
      }
      const std::uint64_t begin = offsets->items()[0].as_size();
      const std::uint64_t end = offsets->items()[1].as_size();
      if (end - begin != count * 4 || 8 + header_size + end > *file_size) {
        SetError(error, "invalid MoGe tensor extent " + name);
        return nullptr;
      }
      host.resize(count);
      if (platform::ReadFileAt(file, host.data(), count * 4,
                               8 + header_size + begin) != count * 4) {
        SetError(error, "short read of MoGe tensor " + name);
        return nullptr;
      }
      m.shapes[name] = dims;
      if (name == "encoder.backbone.pos_embed") {
        m.pos_class.assign(host.begin(), host.begin() + kDim);
        m.pos_grid.assign(host.begin() + kDim, host.end());
        continue;
      }
      const auto keep = [&](std::vector<float>* target) {
        *target = host;
      };
      if (name == "scale_head.0.weight") keep(&m.scale_w0);
      if (name == "scale_head.0.bias") keep(&m.scale_b0);
      if (name == "scale_head.2.weight") keep(&m.scale_w1);
      if (name == "scale_head.2.bias") keep(&m.scale_b1);
      if (name == "scale_head.4.weight") keep(&m.scale_w2);
      if (name == "scale_head.4.bias") keep(&m.scale_b2);
      DeviceBuffer buffer;
      if (!buffer.Allocate(count * 4, error) ||
          !Check(hipMemcpy(buffer.f32(), host.data(), count * 4,
                           hipMemcpyHostToDevice),
                 "weight upload", error))
        return nullptr;
      m.weights.emplace(name, std::move(buffer));
    }
  } catch (const std::exception& e) {
    SetError(error, path.string() + ": " + e.what());
    return nullptr;
  }
  if (m.pos_grid.size() != static_cast<std::size_t>(37 * 37 * kDim) ||
      m.W("points_head.output_blocks.4.weight") == nullptr ||
      m.W("mask_head.output_blocks.4.weight") == nullptr ||
      m.scale_w2.size() != kDim) {
    SetError(error, "not a MoGe-2 ViT-L checkpoint");
    return nullptr;
  }
  return std::unique_ptr<Moge>(new Moge(std::move(impl)));
}

bool Moge::Infer(std::span<const std::uint8_t> rgb, int width, int height,
                 MogeResult* result, MogeRaw* raw, std::string* error) {
  Impl& m = *impl_;
  if (width < 16 || height < 16 ||
      rgb.size() != static_cast<std::size_t>(width) * height * 3) {
    SetError(error, "MoGe: invalid image");
    return false;
  }
  const double aspect = static_cast<double>(width) / height;
  const int bh = static_cast<int>(std::nearbyint(std::sqrt(kTokens / aspect)));
  const int bw = static_cast<int>(std::nearbyint(std::sqrt(kTokens * aspect)));
  const int ih = bh * kPatch, iw = bw * kPatch;
  const int patches = bh * bw, T = patches + 1;
  // Image [3, H, W] in [0, 1] -> antialiased bilinear to the token grid,
  // ImageNet-normalized; patch rows [patches, 3 * 14 * 14].
  std::vector<float> image(static_cast<std::size_t>(3) * width * height);
  for (int c = 0; c < 3; ++c)
    for (std::size_t i = 0; i < static_cast<std::size_t>(width) * height; ++i)
      image[c * static_cast<std::size_t>(width) * height + i] =
          rgb[i * 3 + c] / 255.0F;
  auto resized = ResizeAntialias(image, 3, height, width, ih, iw);
  constexpr float kMean[3] = {0.485F, 0.456F, 0.406F};
  constexpr float kStd[3] = {0.229F, 0.224F, 0.225F};
  constexpr int kPatchValues = 3 * kPatch * kPatch;
  std::vector<float> columns(static_cast<std::size_t>(patches) * kPatchValues);
  for (int py = 0; py < bh; ++py)
    for (int px = 0; px < bw; ++px)
      for (int c = 0; c < 3; ++c)
        for (int y = 0; y < kPatch; ++y)
          for (int x = 0; x < kPatch; ++x)
            columns[(static_cast<std::size_t>(py) * bw + px) * kPatchValues +
                    (c * kPatch + y) * kPatch + x] =
                (resized[(static_cast<std::size_t>(c) * ih + py * kPatch + y) *
                             iw +
                         px * kPatch + x] -
                 kMean[c]) /
                kStd[c];
  // Tokens: class + positions (bicubic-interpolated grid).
  const auto positions = InterpolatePositions(m.pos_grid, 37, bh, bw);
  std::vector<float> tokens(static_cast<std::size_t>(T) * kDim);
  const float* cls = m.W("encoder.backbone.cls_token");
  std::vector<float> cls_host(kDim);
  if (!Check(hipMemcpy(cls_host.data(), cls, kDim * 4, hipMemcpyDeviceToHost),
             "readback", error))
    return false;
  for (int c = 0; c < kDim; ++c)
    tokens[c] = cls_host[c] + m.pos_class[c];
  DeviceBuffer columns_d, hidden, normed, qkv, scores, attn, linear, mlp,
      features, map;
  const std::size_t TD = static_cast<std::size_t>(T) * kDim;
  if (!columns_d.Allocate(columns.size() * 4, error) ||
      !hidden.Allocate(TD * 4, error) || !normed.Allocate(TD * 4, error) ||
      !qkv.Allocate(TD * 3 * 4, error) ||
      !scores.Allocate(static_cast<std::size_t>(kHeads) * T * T * 4, error) ||
      !attn.Allocate(TD * 4, error) || !linear.Allocate(TD * 4, error) ||
      !mlp.Allocate(static_cast<std::size_t>(T) * kMlp * 4, error) ||
      !features.Allocate(static_cast<std::size_t>(patches) * kDim * 4, error) ||
      !map.Allocate(static_cast<std::size_t>(patches) * (kDim + 2) * 4, error))
    return false;
  // Patch embedding into rows 1.. of hidden, then + positions.
  if (!Check(hipMemcpy(columns_d.f32(), columns.data(), columns.size() * 4,
                       hipMemcpyHostToDevice),
             "upload", error) ||
      !m.Linear(columns_d.f32(), patches, "encoder.backbone.patch_embed.proj",
                kDim, kPatchValues, hidden.f32() + kDim, 0, error))
    return false;
  for (std::size_t i = 0; i < static_cast<std::size_t>(patches) * kDim; ++i)
    tokens[kDim + i] = positions[i];
  DeviceBuffer pos;
  if (!pos.Allocate(TD * 4, error) ||
      !Check(hipMemcpy(pos.f32(), tokens.data(), TD * 4, hipMemcpyHostToDevice),
             "upload", error) ||
      !Check(hipMemset(hidden.f32(), 0, kDim * 4), "zero", error))
    return false;
  AddKernel<<<Blocks(TD), kThreads>>>(hidden.f32(), pos.f32(), TD);
  // Transformer blocks; the normalized patch tokens of layers 5, 11, 17, 23
  // are projected (1x1) and summed.
  const float one = 1.0F, zero = 0.0F;
  std::vector<float> class_token(kDim);
  int taken = 0;
  for (int layer = 0; layer < kLayers; ++layer) {
    const std::string p = "encoder.backbone.blocks." + std::to_string(layer) + ".";
    LayerNormKernel<<<T, kThreads>>>(hidden.f32(), m.W(p + "norm1.weight"),
                                     m.W(p + "norm1.bias"), normed.f32());
    if (!m.Linear(normed.f32(), T, p + "attn.qkv", 3 * kDim, kDim, qkv.f32(),
                  0, error))
      return false;
    const float* q = qkv.f32();
    if (rocblas_sgemm_strided_batched(
            m.blas, rocblas_operation_transpose, rocblas_operation_none, T, T,
            kHeadDim, &one, q + kDim, 3 * kDim, kHeadDim, q, 3 * kDim, kHeadDim,
            &zero, scores.f32(), T, static_cast<rocblas_stride>(T) * T,
            kHeads) != rocblas_status_success) {
      SetError(error, "MoGe attention scores failed");
      return false;
    }
    SoftmaxKernel<<<kHeads * T, kThreads>>>(scores.f32(), T,
                                            1.0F / std::sqrt(64.0F));
    if (rocblas_sgemm_strided_batched(
            m.blas, rocblas_operation_none, rocblas_operation_none, kHeadDim, T,
            T, &one, q + 2 * kDim, 3 * kDim, kHeadDim, scores.f32(), T,
            static_cast<rocblas_stride>(T) * T, &zero, attn.f32(), kDim,
            kHeadDim, kHeads) != rocblas_status_success) {
      SetError(error, "MoGe attention values failed");
      return false;
    }
    if (!m.Linear(attn.f32(), T, p + "attn.proj", kDim, kDim, linear.f32(), 0,
                  error))
      return false;
    ScaledAddKernel<<<Blocks(TD), kThreads>>>(hidden.f32(), linear.f32(),
                                              m.W(p + "ls1.gamma"), TD);
    LayerNormKernel<<<T, kThreads>>>(hidden.f32(), m.W(p + "norm2.weight"),
                                     m.W(p + "norm2.bias"), normed.f32());
    if (!m.Linear(normed.f32(), T, p + "mlp.fc1", kMlp, kDim, mlp.f32(), 1,
                  error) ||
        !m.Linear(mlp.f32(), T, p + "mlp.fc2", kDim, kMlp, linear.f32(), 0,
                  error))
      return false;
    ScaledAddKernel<<<Blocks(TD), kThreads>>>(hidden.f32(), linear.f32(),
                                              m.W(p + "ls2.gamma"), TD);
    if (taken < 4 && layer == kTake[taken]) {
      LayerNormKernel<<<T, kThreads>>>(hidden.f32(),
                                       m.W("encoder.backbone.norm.weight"),
                                       m.W("encoder.backbone.norm.bias"),
                                       normed.f32());
      const std::string proj =
          "encoder.output_projections." + std::to_string(taken);
      if (rocblas_sgemm(m.blas, rocblas_operation_transpose,
                        rocblas_operation_none, kDim, patches, kDim, &one,
                        m.W(proj + ".weight"), kDim, normed.f32() + kDim, kDim,
                        taken == 0 ? &zero : &one, features.f32(),
                        kDim) != rocblas_status_success) {
        SetError(error, "MoGe projection failed");
        return false;
      }
      BiasActKernel<<<Blocks(static_cast<std::size_t>(patches) * kDim),
                      kThreads>>>(features.f32(), m.W(proj + ".bias"), patches,
                                  kDim, 0);
      if (taken == 3 &&
          !Check(hipMemcpy(class_token.data(), normed.f32(), kDim * 4,
                           hipMemcpyDeviceToHost),
                 "readback", error))
        return false;
      ++taken;
    }
    if (!Check(hipGetLastError(), "encoder", error))
      return false;
  }
  // Metric scale: MLP on the class token (host, FP64 accumulation).
  const auto dense = [](const std::vector<float>& w, const std::vector<float>& b,
                        const std::vector<double>& x, bool relu) {
    std::vector<double> y(b.size());
    for (std::size_t r = 0; r < b.size(); ++r) {
      double sum = b[r];
      for (std::size_t c = 0; c < x.size(); ++c)
        sum += static_cast<double>(w[r * x.size() + c]) * x[c];
      y[r] = relu ? std::max(sum, 0.0) : sum;
    }
    return y;
  };
  std::vector<double> x(class_token.begin(), class_token.end());
  x = dense(m.scale_w0, m.scale_b0, x, true);
  x = dense(m.scale_w1, m.scale_b1, x, true);
  x = dense(m.scale_w2, m.scale_b2, x, false);
  const double metric_scale = std::exp(x[0]);
  // Neck inputs: level 0 = features [1024, bh, bw] ++ uv, others = uv.
  std::vector<DeviceBuffer> inputs(5);
  TokensToMapKernel<<<Blocks(static_cast<std::size_t>(patches) * kDim),
                      kThreads>>>(features.f32(), patches, map.f32());
  for (int level = 0; level < 5; ++level) {
    const int h = bh << level, w = bw << level;
    const auto uv = ViewPlaneUv(w, h, aspect);
    if (level == 0) {
      if (!Check(hipMemcpy(map.f32() + static_cast<std::size_t>(patches) * kDim,
                           uv.data(), uv.size() * 4, hipMemcpyHostToDevice),
                 "upload", error))
        return false;
      inputs[0] = std::move(map);
      continue;
    }
    if (!inputs[level].Allocate(uv.size() * 4, error) ||
        !Check(hipMemcpy(inputs[level].f32(), uv.data(), uv.size() * 4,
                         hipMemcpyHostToDevice),
               "upload", error))
      return false;
  }
  std::vector<DeviceBuffer> neck, points_out, mask_out;
  if (!m.RunStack("neck", {0, 2, 2, 2, 0}, inputs, {kDim + 2, 2, 2, 2, 2}, bh,
                  bw, &neck, error))
    return false;
  const std::vector<int> dims(kLevelDims, kLevelDims + 5);
  if (!m.RunStack("points_head", {0, 1, 1, 1, 0}, neck, dims, bh, bw,
                  &points_out, error) ||
      !m.RunStack("mask_head", {0, 1, 1, 1, 0}, neck, dims, bh, bw, &mask_out,
                  error))
    return false;
  // Output 1x1 projections at the last level, then bilinear to the image.
  const int oh = bh << 4, ow = bw << 4;
  const std::size_t out_pixels = static_cast<std::size_t>(oh) * ow;
  DeviceBuffer points_d, mask_d;
  if (!points_d.Allocate(out_pixels * 3 * 4, error) ||
      !mask_d.Allocate(out_pixels * 4, error) ||
      !m.Conv1x1(points_out[4].f32(), 32, static_cast<int>(out_pixels),
                 "points_head.output_blocks.4", 3, points_d.f32(), error) ||
      !m.Conv1x1(mask_out[4].f32(), 32, static_cast<int>(out_pixels),
                 "mask_head.output_blocks.4", 1, mask_d.f32(), error))
    return false;
  std::vector<float> points_low(out_pixels * 3), mask_low(out_pixels);
  if (!Check(hipMemcpy(points_low.data(), points_d.f32(), out_pixels * 12,
                       hipMemcpyDeviceToHost),
             "readback", error) ||
      !Check(hipMemcpy(mask_low.data(), mask_d.f32(), out_pixels * 4,
                       hipMemcpyDeviceToHost),
             "readback", error))
    return false;
  const auto points_map = ResizeBilinear(points_low, 3, oh, ow, height, width);
  const auto mask_map = ResizeBilinear(mask_low, 1, oh, ow, height, width);
  const std::size_t pixels = static_cast<std::size_t>(width) * height;
  // Remap "exp": (x, y) * exp(z), exp(z); sigmoid mask.
  std::vector<float> points(pixels * 3), mask(pixels);
  for (std::size_t i = 0; i < pixels; ++i) {
    const double z = std::exp(static_cast<double>(points_map[2 * pixels + i]));
    points[i * 3] = static_cast<float>(points_map[i] * z);
    points[i * 3 + 1] = static_cast<float>(points_map[pixels + i] * z);
    points[i * 3 + 2] = static_cast<float>(z);
    mask[i] = static_cast<float>(1.0 / (1.0 + std::exp(-mask_map[i])));
  }
  if (raw != nullptr) {
    raw->base_height = bh;
    raw->base_width = bw;
    raw->points = points;
    raw->mask = mask;
    raw->metric_scale = metric_scale;
  }
  if (result == nullptr)
    return true;
  // recover_focal_shift on a 64x64 nearest downsampling: minimize
  // |f * xy / (z + s) - uv| over s with the closed-form f.
  std::vector<double> su, sv, sx, sy, sz;
  {
    const auto uv = ViewPlaneUv(width, height, aspect);
    for (int y = 0; y < 64; ++y)
      for (int xx = 0; xx < 64; ++xx) {
        const int iy = std::min(static_cast<int>(std::floor(y * height / 64.0)),
                                height - 1);
        const int ix = std::min(static_cast<int>(std::floor(xx * width / 64.0)),
                                width - 1);
        const std::size_t i = static_cast<std::size_t>(iy) * width + ix;
        if (mask[i] <= 0.5F)
          continue;
        su.push_back(uv[i]);
        sv.push_back(uv[pixels + i]);
        sx.push_back(points[i * 3]);
        sy.push_back(points[i * 3 + 1]);
        sz.push_back(points[i * 3 + 2]);
      }
  }
  const auto evaluate = [&](double s, double* focal) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < su.size(); ++i) {
      const double px = sx[i] / (sz[i] + s), py = sy[i] / (sz[i] + s);
      num += px * su[i] + py * sv[i];
      den += px * px + py * py;
    }
    const double f = den > 0 ? num / den : 1.0;
    double err = 0;
    for (std::size_t i = 0; i < su.size(); ++i) {
      const double px = sx[i] / (sz[i] + s), py = sy[i] / (sz[i] + s);
      err += (f * px - su[i]) * (f * px - su[i]) + (f * py - sv[i]) * (f * py - sv[i]);
    }
    if (focal != nullptr)
      *focal = f;
    return err;
  };
  double shift = 0, focal = 1;
  if (su.size() >= 2) {
    // Bracket s > -min(z) and golden-section search, then refine.
    double zmin = sz[0];
    for (double z : sz)
      zmin = std::min(zmin, z);
    double lo = -zmin * 0.999 + 1e-6, hi = std::max(10.0 * zmin, 10.0);
    double best = 0, best_err = evaluate(0, nullptr);
    for (int i = 0; i <= 400; ++i) {
      const double s = lo + (hi - lo) * i / 400.0;
      const double e = evaluate(s, nullptr);
      if (e < best_err) {
        best_err = e;
        best = s;
      }
    }
    double a = std::max(lo, best - (hi - lo) / 400.0);
    double b = std::min(hi, best + (hi - lo) / 400.0);
    for (int i = 0; i < 100; ++i) {
      const double m1 = a + (b - a) * 0.381966, m2 = a + (b - a) * 0.618034;
      if (evaluate(m1, nullptr) < evaluate(m2, nullptr))
        b = m2;
      else
        a = m1;
    }
    shift = 0.5 * (a + b);
    evaluate(shift, &focal);
  }
  const double fx_n = focal / 2 * std::sqrt(1 + aspect * aspect) / aspect;
  const double fy_n = focal / 2 * std::sqrt(1 + aspect * aspect);
  result->width = width;
  result->height = height;
  result->focal = focal;
  result->shift = shift;
  result->metric_scale = metric_scale;
  result->fx = fx_n * width;
  result->fy = fy_n * height;
  result->cx = 0.5 * width;
  result->cy = 0.5 * height;
  result->points.assign(pixels * 3, INFINITY);
  result->depth.assign(pixels, INFINITY);
  result->mask.assign(pixels, 0);
  for (int y = 0; y < height; ++y)
    for (int xx = 0; xx < width; ++xx) {
      const std::size_t i = static_cast<std::size_t>(y) * width + xx;
      const double depth = points[i * 3 + 2] + shift;
      if (mask[i] <= 0.5F || depth <= 0)
        continue;
      // depth_map_to_point_map with normalized intrinsics (pixel centres).
      const double u = (xx + 0.5) / width, v = (y + 0.5) / height;
      result->mask[i] = 1;
      result->depth[i] = static_cast<float>(depth * metric_scale);
      result->points[i * 3] =
          static_cast<float>((u - 0.5) / fx_n * depth * metric_scale);
      result->points[i * 3 + 1] =
          static_cast<float>((v - 0.5) / fy_n * depth * metric_scale);
      result->points[i * 3 + 2] = static_cast<float>(depth * metric_scale);
    }
  return true;
}

}  // namespace gufo::models::wan22
