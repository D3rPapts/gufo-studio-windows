#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/models/wan22/gemm.hpp"
#include "src/models/wan22/vae.hpp"

// Activations are channels-last BF16 [frames, height, width, channels].
// Causal 3x3x3 convolutions see the stream "two cached input frames ++ the
// current frames"; the cache starts as zeros, which is the zero padding of
// Diffusers' WanCausalConv3d on the first chunk.

namespace gufo::models::wan22 {
namespace {

constexpr int kZ = VaeDecoder::kLatentChannels;
constexpr int kTop = 1024;      // decoder_base_dim 256 * dim_mult[-1] 4
constexpr int kOutPadded = 16;  // conv_out rows 12 padded for the GEMM
constexpr int kThreads = 256;
constexpr std::size_t kColumnBytes = 256U << 20U;

__device__ __forceinline__ float FromBf16(std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t ToBf16(float value) {
  const unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7FFFFFFFU) > 0x7F800000U)
    return static_cast<std::uint16_t>((bits >> 16U) | 0x40U);
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >>
                                    16U);
}

struct Im2colArgs {
  const std::uint16_t* current;  // [source frames, Hs, Ws, stride]
  const std::uint16_t* history;  // [2, Hs, Ws, stride] for kt = 3
  int frames, height, width;     // output grid
  int channels;                  // channels per tap
  int stride;                    // source channels per pixel
  int source_height, source_width;
  int upsample;  // 1: nearest 2x spatial upsample of the source
  int split;     // 1: frame f = source frame f / 2, channel block f % 2
  int kt, ks;
};

// columns[row, ((dt * ks + dy) * ks + dx) * C + c], eight channels a thread.
__global__ void Im2colKernel(Im2colArgs a, std::size_t first_row,
                             std::size_t rows, std::uint16_t* columns) {
  const int groups = a.channels / 8;
  const std::size_t per_row =
      static_cast<std::size_t>(a.kt) * a.ks * a.ks * groups;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= rows * per_row)
    return;
  const std::size_t local = index / per_row;
  const int q = static_cast<int>(index % per_row);
  const int group = q % groups;
  const int tap = q / groups;
  const int dx = tap % a.ks, dy = (tap / a.ks) % a.ks, dt = tap / (a.ks * a.ks);
  const std::size_t row = first_row + local;
  const int x = static_cast<int>(row % a.width);
  const int y = static_cast<int>((row / a.width) % a.height);
  const int f =
      static_cast<int>(row / (static_cast<std::size_t>(a.width) * a.height));
  const int ef = f + dt - (a.kt - 1);
  const int yy = y + dy - a.ks / 2;
  const int xx = x + dx - a.ks / 2;
  uint4 value = make_uint4(0, 0, 0, 0);
  if (yy >= 0 && yy < a.height && xx >= 0 && xx < a.width) {
    const int sy = yy >> a.upsample, sx = xx >> a.upsample;
    const std::size_t pixel =
        static_cast<std::size_t>(sy) * a.source_width + sx;
    const std::size_t plane =
        static_cast<std::size_t>(a.source_height) * a.source_width;
    const std::uint16_t* source;
    if (ef >= 0) {
      const int sf = a.split ? ef >> 1 : ef;
      const int offset = a.split ? (ef & 1) * a.channels : 0;
      source = a.current + (sf * plane + pixel) * a.stride + offset;
    } else {
      source = a.history + ((ef + 2) * plane + pixel) * a.stride;
    }
    value = reinterpret_cast<const uint4*>(source)[group];
  }
  reinterpret_cast<uint4*>(columns + local * per_row * 8)[q] = value;
}

// out = act(x / max(|x|, 1e-12) * sqrt(C) * gamma) per pixel (F.normalize
// over channels, WanRMS_norm), FP32 math, BF16 out.
template<bool kSilu>
__global__ void NormKernel(const std::uint16_t* x, const float* gamma,
                           int channels, std::uint16_t* out) {
  __shared__ float partial[kThreads];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * channels;
  float sum = 0;
  for (int c = threadIdx.x; c < channels; c += kThreads) {
    const float v = FromBf16(x[base + c]);
    sum += v * v;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale =
      sqrtf(static_cast<float>(channels)) / fmaxf(sqrtf(partial[0]), 1e-12F);
  for (int c = threadIdx.x; c < channels; c += kThreads) {
    float v = FromBf16(x[base + c]) * scale * gamma[c];
    if (kSilu)
      v = v / (1.0F + expf(-v));
    out[base + c] = ToBf16(v);
  }
}

__global__ void AddKernel(std::uint16_t* out, const std::uint16_t* a,
                          std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = ToBf16(FromBf16(out[i]) + FromBf16(a[i]));
}

// DupUp3D shortcut added to `out` [frames, 2H, 2W, Co]: channel
// n = ((co * ft + ti) * 2 + sy) * 2 + sx of the repeat-interleaved input
// [Tin, H, W, Ci] (n / repeats); the first chunk drops ft - 1 frames.
__global__ void DupUpAddKernel(std::uint16_t* out, const std::uint16_t* in,
                               int frames, int height, int width, int co_count,
                               int ci_count, int ft, int skip) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int oh = 2 * height, ow = 2 * width;
  const std::size_t count =
      static_cast<std::size_t>(frames) * oh * ow * co_count;
  if (i >= count)
    return;
  const int co = static_cast<int>(i % co_count);
  std::size_t pixel = i / co_count;
  const int ox = static_cast<int>(pixel % ow);
  pixel /= ow;
  const int oy = static_cast<int>(pixel % oh);
  const int f = static_cast<int>(pixel / oh) + skip;
  const int t = f / ft, ti = f % ft;
  const int repeats = co_count * ft * 4 / ci_count;
  const int n = ((co * ft + ti) * 2 + (oy & 1)) * 2 + (ox & 1);
  const std::size_t source =
      ((static_cast<std::size_t>(t) * height + (oy >> 1)) * width + (ox >> 1)) *
          ci_count +
      n / repeats;
  out[i] = ToBf16(FromBf16(out[i]) + FromBf16(in[source]));
}

__global__ void WidenKernel(const std::uint16_t* x, std::size_t count,
                            float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = FromBf16(x[i]);
}

__global__ void NarrowKernel(const float* x, std::size_t count,
                             std::uint16_t* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = ToBf16(x[i]);
}

// In-place softmax(scale * row) of FP32 scores [rows, columns].
__global__ void SoftmaxKernel(float* scores, int columns, float scale) {
  __shared__ float partial[kThreads];
  float* row = scores + static_cast<std::size_t>(blockIdx.x) * columns;
  float maximum = -INFINITY;
  for (int c = threadIdx.x; c < columns; c += kThreads)
    maximum = fmaxf(maximum, row[c] * scale);
  partial[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] =
          fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    __syncthreads();
  }
  maximum = partial[0];
  __syncthreads();
  float sum = 0;
  for (int c = threadIdx.x; c < columns; c += kThreads) {
    const float e = expf(row[c] * scale - maximum);
    row[c] = e;
    sum += e;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float inverse = 1.0F / partial[0];
  for (int c = threadIdx.x; c < columns; c += kThreads)
    row[c] *= inverse;
}

// conv_out rows [frames, H, W, 16] -> clamped RGB [frames, 2H, 2W, 3]:
// unpatchify puts channel c * 4 + q1 * 2 + q2 at (2y + q2, 2x + q1).
__global__ void OutputKernel(const std::uint16_t* x, int frames, int height,
                             int width, float* rgb) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int oh = 2 * height, ow = 2 * width;
  const std::size_t count = static_cast<std::size_t>(frames) * oh * ow * 3;
  if (i >= count)
    return;
  const int c = static_cast<int>(i % 3);
  std::size_t pixel = i / 3;
  const int ox = static_cast<int>(pixel % ow);
  pixel /= ow;
  const int oy = static_cast<int>(pixel % oh);
  const int f = static_cast<int>(pixel / oh);
  const int q1 = ox & 1, q2 = oy & 1;
  const std::size_t source =
      ((static_cast<std::size_t>(f) * height + (oy >> 1)) * width + (ox >> 1)) *
          kOutPadded +
      c * 4 + q1 * 2 + q2;
  rgb[i] = fminf(fmaxf(FromBf16(x[source]), -1.0F), 1.0F);
}

unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

bool Check(hipError_t status, const char* operation, std::string* error) {
  if (status == hipSuccess)
    return true;
  SetError(error, std::string("Wan VAE ") + operation + ": " +
                      hipGetErrorString(status));
  return false;
}

std::uint16_t HostBf16(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, 4);
  if ((bits & 0x7FFFFFFFU) > 0x7F800000U)
    return static_cast<std::uint16_t>((bits >> 16U) | 0x40U);
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >>
                                    16U);
}

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        bytes_(std::exchange(other.bytes_, 0)) {}
  ~DeviceBuffer() { Release(); }
  bool Allocate(std::size_t bytes, std::string* error) {
    Release();
    bytes_ = std::max<std::size_t>(bytes, 256);
    return Check(hipMalloc(&data_, bytes_), "device allocation", error);
  }
  void Release() {
    if (data_ != nullptr)
      (void)hipFree(data_);
    data_ = nullptr;
    bytes_ = 0;
  }
  std::uint16_t* get() const { return static_cast<std::uint16_t*>(data_); }
  float* f32() const { return static_cast<float*>(data_); }
  std::size_t bytes() const { return bytes_; }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

struct Conv {
  DeviceBuffer weight;  // BF16 [rows, kt * ks * ks * cin]
  DeviceBuffer bias;    // BF16 [rows]
  int cin{0}, cout{0}, rows{0}, kt{1}, ks{1};
  DeviceBuffer history;  // [2, H, W, cin] of the last input (kt = 3)
  int history_height{0}, history_width{0};
};

struct Resnet {
  DeviceBuffer gamma1, gamma2;  // FP32
  Conv conv1, conv2;
  bool has_shortcut{false};
  Conv shortcut;
};

struct UpBlock {
  std::vector<Resnet> resnets;
  int mode{0};  // 0 none, 2 upsample2d, 3 upsample3d
  Conv time_conv, resample;
  int in_channels{0}, out_channels{0};
};

// A view of channels-last activations.
struct Tensor {
  std::uint16_t* data{nullptr};
  int frames{0}, height{0}, width{0}, channels{0};
  std::size_t count() const {
    return static_cast<std::size_t>(frames) * height * width * channels;
  }
};

}  // namespace

struct VaeDecoder::Impl {
  ~Impl() {
    if (blas != nullptr)
      (void)rocblas_destroy_handle(blas);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }

  bool ReadTensor(const std::string& name, std::vector<float>* values,
                  std::vector<std::size_t>* shape, std::string* error);
  bool LoadConv(const std::string& name, int kt, int ks, int rows_padded,
                Conv* conv, std::string* error);
  bool LoadGamma(const std::string& name, int channels, DeviceBuffer* buffer,
                 std::string* error);
  bool LoadResnet(const std::string& prefix, int cin, int cout, Resnet* resnet,
                  std::string* error);

  // Convolution of `source` (output grid frames x height x width) into
  // `out` [frames, height, width, rows].
  bool RunConv(Conv& conv, const Im2colArgs& source, std::uint16_t* out,
               std::string* error);
  bool CausalConv(Conv& conv, const Tensor& in, std::uint16_t* out,
                  std::string* error);
  bool Norm(const Tensor& in, const DeviceBuffer& gamma, bool silu,
            std::uint16_t* out);
  bool RunResnet(Resnet& resnet, const Tensor& in, std::uint16_t* out,
                 std::uint16_t* scratch_a, std::uint16_t* scratch_b,
                 std::string* error);
  bool RunAttention(const Tensor& in, std::uint16_t* out,
                    std::uint16_t* scratch_a, std::uint16_t* scratch_b,
                    std::string* error);
  bool Prepare(int height, int width, std::string* error);
  bool DecodeFrame(int frame, bool first, const std::uint16_t* latent_frame,
                   float* rgb, int* out_frames, std::string* error);

  platform::PlatformFile file{platform::kInvalidFile};
  struct Location {
    std::uint64_t offset{0};
    std::uint64_t bytes{0};
    std::vector<std::size_t> shape;
  };
  std::map<std::string, Location, std::less<>> tensors;
  std::vector<float> mean, std_dev;

  Conv post_quant, conv_in, conv_out;
  Resnet mid0, mid1;
  DeviceBuffer attn_gamma;
  Conv attn_qkv, attn_proj;
  std::array<UpBlock, 4> up;
  DeviceBuffer norm_out;
  std::size_t weight_bytes{0};

  hipStream_t stream{nullptr};
  rocblas_handle blas{nullptr};
  LinearGemm gemm;

  // Geometry-dependent state.
  int prepared_height{0}, prepared_width{0};
  std::array<DeviceBuffer, 5> pool;
  DeviceBuffer columns;
  DeviceBuffer qkv32, scores, attn32;
  DeviceBuffer rgb_device;
  DeviceBuffer latent_device;
  std::size_t activation_bytes{0};
};

bool VaeDecoder::Impl::ReadTensor(const std::string& name,
                                  std::vector<float>* values,
                                  std::vector<std::size_t>* shape,
                                  std::string* error) {
  const auto it = tensors.find(name);
  if (it == tensors.end()) {
    SetError(error, "missing VAE tensor " + name);
    return false;
  }
  *shape = it->second.shape;
  values->resize(it->second.bytes / 4);
  auto* bytes = reinterpret_cast<std::byte*>(values->data());
  for (std::uint64_t read = 0; read < it->second.bytes;) {
    const std::size_t result = platform::ReadFileAt(
        file, bytes + read,
        static_cast<std::size_t>(std::min<std::uint64_t>(
            it->second.bytes - read, std::uint64_t{64} << 20U)),
        it->second.offset + read);
    if (result == 0) {
      SetError(error, "short read of VAE tensor " + name);
      return false;
    }
    read += result;
  }
  return true;
}

// Weight [cout, cin, (kt,) kh, kw] -> BF16 rows [cout, kt, kh, kw, cin]
// (zero rows up to rows_padded); bias -> BF16.
bool VaeDecoder::Impl::LoadConv(const std::string& name, int kt, int ks,
                                int rows_padded, Conv* conv,
                                std::string* error) {
  std::vector<float> weight, bias;
  std::vector<std::size_t> shape, bias_shape;
  if (!ReadTensor(name + ".weight", &weight, &shape, error) ||
      !ReadTensor(name + ".bias", &bias, &bias_shape, error))
    return false;
  const bool three_d = shape.size() == 5;
  if ((shape.size() != 5 && shape.size() != 4) ||
      (three_d &&
       (static_cast<int>(shape[2]) != kt || static_cast<int>(shape[3]) != ks ||
        static_cast<int>(shape[4]) != ks)) ||
      (!three_d && (kt != 1 || static_cast<int>(shape[2]) != ks ||
                    static_cast<int>(shape[3]) != ks)) ||
      bias.size() != shape[0] || shape[1] % 8 != 0) {
    SetError(error, "unexpected shape of VAE tensor " + name);
    return false;
  }
  conv->cout = static_cast<int>(shape[0]);
  conv->cin = static_cast<int>(shape[1]);
  conv->rows = std::max(conv->cout, rows_padded);
  conv->kt = kt;
  conv->ks = ks;
  const int taps = kt * ks * ks;
  const std::size_t columns = static_cast<std::size_t>(taps) * conv->cin;
  std::vector<std::uint16_t> packed(columns * conv->rows, 0);
  for (int o = 0; o < conv->cout; ++o)
    for (int i = 0; i < conv->cin; ++i)
      for (int tap = 0; tap < taps; ++tap)
        packed[o * columns + static_cast<std::size_t>(tap) * conv->cin + i] =
            HostBf16(
                weight[(static_cast<std::size_t>(o) * conv->cin + i) * taps +
                       tap]);
  std::vector<std::uint16_t> packed_bias(conv->rows, 0);
  for (int o = 0; o < conv->cout; ++o)
    packed_bias[o] = HostBf16(bias[o]);
  weight_bytes += packed.size() * 2 + packed_bias.size() * 2;
  return conv->weight.Allocate(packed.size() * 2, error) &&
         conv->bias.Allocate(packed_bias.size() * 2, error) &&
         Check(hipMemcpy(conv->weight.get(), packed.data(), packed.size() * 2,
                         hipMemcpyHostToDevice),
               "weight upload", error) &&
         Check(hipMemcpy(conv->bias.get(), packed_bias.data(),
                         packed_bias.size() * 2, hipMemcpyHostToDevice),
               "weight upload", error);
}

bool VaeDecoder::Impl::LoadGamma(const std::string& name, int channels,
                                 DeviceBuffer* buffer, std::string* error) {
  std::vector<float> values;
  std::vector<std::size_t> shape;
  if (!ReadTensor(name, &values, &shape, error))
    return false;
  if (static_cast<int>(values.size()) != channels) {
    SetError(error, "unexpected shape of VAE tensor " + name);
    return false;
  }
  weight_bytes += values.size() * 4;
  return buffer->Allocate(values.size() * 4, error) &&
         Check(hipMemcpy(buffer->get(), values.data(), values.size() * 4,
                         hipMemcpyHostToDevice),
               "weight upload", error);
}

bool VaeDecoder::Impl::LoadResnet(const std::string& prefix, int cin, int cout,
                                  Resnet* resnet, std::string* error) {
  resnet->has_shortcut = cin != cout;
  if (!LoadGamma(prefix + ".norm1.gamma", cin, &resnet->gamma1, error) ||
      !LoadGamma(prefix + ".norm2.gamma", cout, &resnet->gamma2, error) ||
      !LoadConv(prefix + ".conv1", 3, 3, 0, &resnet->conv1, error) ||
      !LoadConv(prefix + ".conv2", 3, 3, 0, &resnet->conv2, error) ||
      (resnet->has_shortcut &&
       !LoadConv(prefix + ".conv_shortcut", 1, 1, 0, &resnet->shortcut, error)))
    return false;
  if (resnet->conv1.cin != cin || resnet->conv1.cout != cout ||
      resnet->conv2.cin != cout || resnet->conv2.cout != cout) {
    SetError(error, "unexpected channels in " + prefix);
    return false;
  }
  return true;
}

bool VaeDecoder::Impl::RunConv(Conv& conv, const Im2colArgs& source,
                               std::uint16_t* out, std::string* error) {
  const std::size_t rows =
      static_cast<std::size_t>(source.frames) * source.height * source.width;
  const int taps = conv.kt * conv.ks * conv.ks;
  if (taps == 1 && !source.upsample && !source.split &&
      source.stride == conv.cin)
    return gemm.Run(conv.weight.get(), conv.bias.get(), source.current, out,
                    static_cast<int>(rows), conv.rows, conv.cin, error);
  const std::size_t width = static_cast<std::size_t>(taps) * conv.cin;
  std::size_t pass = columns.bytes() / (width * 2);
  pass = std::max<std::size_t>(pass / LinearGemm::kRowChunk, 1) *
         LinearGemm::kRowChunk;
  for (std::size_t first = 0; first < rows; first += pass) {
    const std::size_t count = std::min(pass, rows - first);
    Im2colKernel<<<Blocks(count * width / 8), kThreads, 0, stream>>>(
        source, first, count, columns.get());
    if (!gemm.Run(conv.weight.get(), conv.bias.get(), columns.get(),
                  out + first * conv.rows, static_cast<int>(count), conv.rows,
                  static_cast<int>(width), error))
      return false;
  }
  return Check(hipGetLastError(), "im2col", error);
}

bool VaeDecoder::Impl::CausalConv(Conv& conv, const Tensor& in,
                                  std::uint16_t* out, std::string* error) {
  Im2colArgs args{};
  args.current = in.data;
  args.history = conv.history.get();
  args.frames = in.frames;
  args.height = args.source_height = in.height;
  args.width = args.source_width = in.width;
  args.channels = args.stride = in.channels;
  args.kt = conv.kt;
  args.ks = conv.ks;
  if (in.channels != conv.cin) {
    SetError(error, "Wan VAE channel mismatch");
    return false;
  }
  if (!RunConv(conv, args, out, error))
    return false;
  if (conv.kt != 3)
    return true;
  // history = last two frames of (history ++ input).
  const std::size_t plane =
      static_cast<std::size_t>(in.height) * in.width * in.channels * 2;
  auto* history = reinterpret_cast<std::byte*>(conv.history.get());
  const auto* current = reinterpret_cast<const std::byte*>(in.data);
  if (in.frames >= 2)
    return Check(hipMemcpyAsync(history, current + (in.frames - 2) * plane,
                                2 * plane, hipMemcpyDeviceToDevice, stream),
                 "cache update", error);
  return Check(hipMemcpyAsync(history, history + plane, plane,
                              hipMemcpyDeviceToDevice, stream),
               "cache update", error) &&
         Check(hipMemcpyAsync(history + plane, current, plane,
                              hipMemcpyDeviceToDevice, stream),
               "cache update", error);
}

bool VaeDecoder::Impl::Norm(const Tensor& in, const DeviceBuffer& gamma,
                            bool silu, std::uint16_t* out) {
  const unsigned pixels = static_cast<unsigned>(in.count() / in.channels);
  if (silu)
    NormKernel<true><<<pixels, kThreads, 0, stream>>>(in.data, gamma.f32(),
                                                      in.channels, out);
  else
    NormKernel<false><<<pixels, kThreads, 0, stream>>>(in.data, gamma.f32(),
                                                       in.channels, out);
  return true;
}

bool VaeDecoder::Impl::RunResnet(Resnet& resnet, const Tensor& in,
                                 std::uint16_t* out, std::uint16_t* scratch_a,
                                 std::uint16_t* scratch_b, std::string* error) {
  const int cout = resnet.conv1.cout;
  Tensor a = in;
  a.data = scratch_a;
  Norm(in, resnet.gamma1, true, scratch_a);
  if (!CausalConv(resnet.conv1, a, scratch_b, error))
    return false;
  Tensor b = in;
  b.data = scratch_b;
  b.channels = cout;
  a.channels = cout;
  Norm(b, resnet.gamma2, true, scratch_a);
  if (!CausalConv(resnet.conv2, a, out, error))
    return false;
  Tensor result = b;
  result.data = out;
  const std::uint16_t* shortcut = in.data;
  if (resnet.has_shortcut) {
    if (!CausalConv(resnet.shortcut, in, scratch_b, error))
      return false;
    shortcut = scratch_b;
  }
  AddKernel<<<Blocks(result.count()), kThreads, 0, stream>>>(out, shortcut,
                                                             result.count());
  return Check(hipGetLastError(), "resnet", error);
}

// Single-head spatial self-attention per frame (WanAttentionBlock).
bool VaeDecoder::Impl::RunAttention(const Tensor& in, std::uint16_t* out,
                                    std::uint16_t* scratch_a,
                                    std::uint16_t* scratch_b,
                                    std::string* error) {
  const int n = in.height * in.width;
  const int c = in.channels;
  Norm(in, attn_gamma, false, scratch_a);
  for (int f = 0; f < in.frames; ++f) {
    const std::size_t offset = static_cast<std::size_t>(f) * n * c;
    if (!gemm.Run(attn_qkv.weight.get(), attn_qkv.bias.get(),
                  scratch_a + offset, scratch_b, n, 3 * c, c, error))
      return false;
    const std::size_t qkv_count = static_cast<std::size_t>(n) * 3 * c;
    WidenKernel<<<Blocks(qkv_count), kThreads, 0, stream>>>(
        scratch_b, qkv_count, qkv32.f32());
    const float one = 1.0F, zero = 0.0F;
    const float* q = qkv32.f32();
    const float* k = q + c;
    const float* v = q + 2 * c;
    if (rocblas_sgemm(blas, rocblas_operation_transpose, rocblas_operation_none,
                      n, n, c, &one, k, 3 * c, q, 3 * c, &zero, scores.f32(),
                      n) != rocblas_status_success) {
      SetError(error, "Wan VAE attention scores GEMM failed");
      return false;
    }
    SoftmaxKernel<<<n, kThreads, 0, stream>>>(
        scores.f32(), n, 1.0F / std::sqrt(static_cast<float>(c)));
    if (rocblas_sgemm(blas, rocblas_operation_none, rocblas_operation_none, c,
                      n, n, &one, v, 3 * c, scores.f32(), n, &zero,
                      attn32.f32(), c) != rocblas_status_success) {
      SetError(error, "Wan VAE attention value GEMM failed");
      return false;
    }
    const std::size_t count = static_cast<std::size_t>(n) * c;
    NarrowKernel<<<Blocks(count), kThreads, 0, stream>>>(attn32.f32(), count,
                                                         scratch_b);
    if (!gemm.Run(attn_proj.weight.get(), attn_proj.bias.get(), scratch_b,
                  out + offset, n, c, c, error))
      return false;
  }
  AddKernel<<<Blocks(in.count()), kThreads, 0, stream>>>(out, in.data,
                                                         in.count());
  return Check(hipGetLastError(), "attention", error);
}

bool VaeDecoder::Impl::Prepare(int height, int width, std::string* error) {
  if (height != prepared_height || width != prepared_width) {
    const std::size_t hw = static_cast<std::size_t>(height) * width;
    // Largest activation: 4 frames at 8x the latent size with 512 channels
    // (or the 1024-channel time convolution output at 2x).
    const std::size_t largest =
        std::max<std::size_t>(4 * 64 * 512, 2 * 4 * 2048) * hw;
    activation_bytes = 0;
    for (auto& buffer : pool) {
      if (!buffer.Allocate(largest * 2, error))
        return false;
      activation_bytes += largest * 2;
    }
    const std::size_t widest = static_cast<std::size_t>(27) * kTop * 2;
    if (!columns.Allocate(
            std::max(kColumnBytes, LinearGemm::kRowChunk * widest), error) ||
        !qkv32.Allocate(hw * 3 * kTop * 4, error) ||
        !scores.Allocate(hw * hw * 4, error) ||
        !attn32.Allocate(hw * kTop * 4, error) ||
        !rgb_device.Allocate(4 * 256 * hw * 3 * 4, error) ||
        !latent_device.Allocate(hw * kZ * 2, error))
      return false;
    activation_bytes += columns.bytes() + qkv32.bytes() + scores.bytes() +
                        attn32.bytes() + rgb_device.bytes() +
                        latent_device.bytes();
    // Cache of every causal convolution at its resolution.
    const auto history = [&](Conv& conv, int scale) {
      if (conv.kt != 3)
        return true;
      conv.history_height = height * scale;
      conv.history_width = width * scale;
      const std::size_t bytes = 2 * hw * scale * scale * conv.cin * 2;
      activation_bytes += bytes;
      return conv.history.Allocate(bytes, error);
    };
    const auto resnet_history = [&](Resnet& resnet, int scale) {
      return history(resnet.conv1, scale) && history(resnet.conv2, scale);
    };
    if (!history(conv_in, 1) || !resnet_history(mid0, 1) ||
        !resnet_history(mid1, 1))
      return false;
    for (int b = 0; b < 4; ++b) {
      const int scale = 1 << b;
      for (auto& resnet : up[b].resnets)
        if (!resnet_history(resnet, scale))
          return false;
      if (up[b].mode == 3 && !history(up[b].time_conv, scale))
        return false;
    }
    if (!history(conv_out, 8))
      return false;
    prepared_height = height;
    prepared_width = width;
  }
  // Zero caches: the causal padding of the first chunk.
  const auto clear = [&](Conv& conv) {
    return conv.kt != 3 || Check(hipMemsetAsync(conv.history.get(), 0,
                                                conv.history.bytes(), stream),
                                 "cache reset", error);
  };
  bool ok = clear(conv_in) && clear(mid0.conv1) && clear(mid0.conv2) &&
            clear(mid1.conv1) && clear(mid1.conv2) && clear(conv_out);
  for (auto& block : up) {
    for (auto& resnet : block.resnets)
      ok = ok && clear(resnet.conv1) && clear(resnet.conv2);
    if (block.mode == 3)
      ok = ok && clear(block.time_conv);
  }
  return ok;
}

// One latent frame through post_quant_conv and the decoder; writes
// `out_frames` (1 for the first chunk, else 4) RGB frames to `rgb`.
bool VaeDecoder::Impl::DecodeFrame(int frame, bool first,
                                   const std::uint16_t* latent_frame,
                                   float* rgb, int* out_frames,
                                   std::string* error) {
  (void)frame;
  const int h = prepared_height, w = prepared_width;
  std::array<std::uint16_t*, 5> p{};
  for (int i = 0; i < 5; ++i)
    p[i] = pool[i].get();
  // Latent [h, w, 48] -> post_quant_conv -> conv_in -> mid block.
  Tensor x{p[0], 1, h, w, kZ};
  if (!CausalConv(post_quant,
                  Tensor{const_cast<std::uint16_t*>(latent_frame), 1, h, w, kZ},
                  p[0], error) ||
      !CausalConv(conv_in, x, p[1], error))
    return false;
  x = Tensor{p[1], 1, h, w, kTop};
  if (!RunResnet(mid0, x, p[0], p[2], p[3], error))
    return false;
  x.data = p[0];
  if (!RunAttention(x, p[1], p[2], p[3], error))
    return false;
  x.data = p[1];
  if (!RunResnet(mid1, x, p[0], p[2], p[3], error))
    return false;
  x.data = p[0];
  // Buffers: input I = p[0]; A, B, S1, S2 = the others.
  int in_index = 0;
  for (int b = 0; b < 4; ++b) {
    UpBlock& block = up[b];
    std::array<int, 4> other{};
    for (int i = 0, n = 0; i < 5; ++i)
      if (i != in_index)
        other[n++] = i;
    std::uint16_t* a = p[other[0]];
    std::uint16_t* bb = p[other[1]];
    std::uint16_t* s1 = p[other[2]];
    std::uint16_t* s2 = p[other[3]];
    const Tensor block_in = x;
    Tensor y = x;
    std::uint16_t* targets[3] = {a, bb, a};
    for (std::size_t r = 0; r < block.resnets.size(); ++r) {
      if (!RunResnet(block.resnets[r], y, targets[r], s1, s2, error))
        return false;
      y.data = targets[r];
      y.channels = block.out_channels;
    }
    if (block.mode == 0) {
      x = y;
      in_index = other[0];
      continue;
    }
    // Upsampler: (time conv ->) nearest 2x -> 3x3 conv into B.
    Im2colArgs args{};
    args.current = y.data;
    args.frames = y.frames;
    args.channels = args.stride = y.channels;
    args.source_height = y.height;
    args.source_width = y.width;
    if (block.mode == 3 && !first) {
      if (!CausalConv(block.time_conv, y, s1, error))
        return false;
      args.current = s1;
      args.stride = 2 * y.channels;
      args.split = 1;
      args.frames = 2 * y.frames;
    }
    args.height = 2 * y.height;
    args.width = 2 * y.width;
    args.upsample = 1;
    args.kt = 1;
    args.ks = 3;
    if (!RunConv(block.resample, args, bb, error))
      return false;
    Tensor z{bb, args.frames, args.height, args.width, block.out_channels};
    const int ft = block.mode == 3 ? 2 : 1;
    const int skip = first ? ft - 1 : 0;
    DupUpAddKernel<<<Blocks(z.count()), kThreads, 0, stream>>>(
        bb, block_in.data, z.frames, block_in.height, block_in.width,
        block.out_channels, block_in.channels, ft, skip);
    if (!Check(hipGetLastError(), "shortcut", error))
      return false;
    x = z;
    in_index = other[1];
  }
  // Head: norm_out + SiLU -> conv_out -> unpatchify, clamp.
  std::array<int, 4> other{};
  for (int i = 0, n = 0; i < 5; ++i)
    if (i != in_index)
      other[n++] = i;
  Tensor a = x;
  a.data = p[other[0]];
  Norm(x, norm_out, true, a.data);
  if (!CausalConv(conv_out, a, p[other[1]], error))
    return false;
  const std::size_t count =
      static_cast<std::size_t>(x.frames) * 4 * x.height * x.width * 3;
  OutputKernel<<<Blocks(count), kThreads, 0, stream>>>(
      p[other[1]], x.frames, x.height, x.width, rgb_device.f32());
  *out_frames = x.frames;
  return Check(hipGetLastError(), "output", error) &&
         Check(hipMemcpyAsync(rgb, rgb_device.f32(), count * 4,
                              hipMemcpyDeviceToHost, stream),
               "output download", error) &&
         Check(hipStreamSynchronize(stream), "decode", error);
}

VaeDecoder::VaeDecoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
VaeDecoder::~VaeDecoder() {
  if (impl_ != nullptr && platform::FileValid(impl_->file))
    platform::CloseFile(impl_->file);
}

std::size_t VaeDecoder::weight_bytes() const {
  return impl_->weight_bytes;
}
std::size_t VaeDecoder::activation_bytes() const {
  return impl_->activation_bytes;
}

std::unique_ptr<VaeDecoder> VaeDecoder::Load(const std::filesystem::path& dir,
                                             std::string* error) {
  std::ifstream config_file(dir / "config.json", std::ios::binary);
  std::stringstream config_text;
  config_text << config_file.rdbuf();
  json::Value config;
  try {
    config = json::parse(config_text.str());
  } catch (const std::exception& e) {
    SetError(error, "cannot read VAE config: " + std::string(e.what()));
    return nullptr;
  }
  const json::Value* mean = config.find("latents_mean");
  const json::Value* std_dev = config.find("latents_std");
  const json::Value* mult = config.find("dim_mult");
  if (config.member_str("_class_name") != "AutoencoderKLWan" ||
      config.member_double("z_dim") != kZ ||
      config.member_double("decoder_base_dim") != 256 ||
      config.member_double("patch_size") != 2 ||
      config.member_double("num_res_blocks") != 2 ||
      config.member_double("out_channels") != 12 || mean == nullptr ||
      !mean->is_array() || mean->size() != kZ || std_dev == nullptr ||
      !std_dev->is_array() || std_dev->size() != kZ || mult == nullptr ||
      !mult->is_array() || mult->size() != 4 ||
      mult->items()[0].as_size() != 1 || mult->items()[1].as_size() != 2 ||
      mult->items()[2].as_size() != 4 || mult->items()[3].as_size() != 4) {
    SetError(error, "unsupported VAE configuration (Wan 2.2 TI2V-5B only)");
    return nullptr;
  }
  const json::Value* attn = config.find("attn_scales");
  const json::Value* down = config.find("temperal_downsample");
  const json::Value* residual = config.find("is_residual");
  if ((attn != nullptr && !(attn->is_array() && attn->empty())) ||
      down == nullptr || !down->is_array() || down->size() != 3 ||
      down->items()[0].as_bool() || !down->items()[1].as_bool() ||
      !down->items()[2].as_bool() || residual == nullptr ||
      !residual->as_bool()) {
    SetError(error, "unsupported VAE configuration (Wan 2.2 TI2V-5B only)");
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  for (int c = 0; c < kZ; ++c) {
    impl->mean.push_back(static_cast<float>(mean->items()[c].as_double()));
    impl->std_dev.push_back(
        static_cast<float>(std_dev->items()[c].as_double()));
  }

  const auto path = dir / "diffusion_pytorch_model.safetensors";
  std::string open_error;
  impl->file = platform::OpenReadFile(path, &open_error);
  if (!platform::FileValid(impl->file)) {
    SetError(error, "cannot open " + path.string() + ": " + open_error);
    return nullptr;
  }
  const auto file_size = platform::FileSizeBytes(impl->file);
  std::uint64_t header_size = 0;
  if (!file_size || platform::ReadFileAt(impl->file, &header_size, 8, 0) != 8 ||
      header_size == 0 || header_size > (32U << 20U) ||
      header_size > *file_size - 8) {
    SetError(error, "invalid safetensors header in " + path.string());
    return nullptr;
  }
  std::string text(static_cast<std::size_t>(header_size), '\0');
  if (platform::ReadFileAt(impl->file, text.data(), text.size(), 8) !=
      text.size()) {
    SetError(error, "short read of " + path.string());
    return nullptr;
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
          offsets->size() != 2)
        continue;
      Impl::Location location;
      std::uint64_t bytes = 4;
      for (const auto& dim : shape->items()) {
        location.shape.push_back(dim.as_size());
        bytes *= location.shape.back();
      }
      const std::uint64_t begin = offsets->items()[0].as_size();
      const std::uint64_t end = offsets->items()[1].as_size();
      if (end < begin || end - begin != bytes ||
          8 + header_size + end > *file_size) {
        SetError(error, "invalid VAE tensor extent " + name);
        return nullptr;
      }
      location.offset = 8 + header_size + begin;
      location.bytes = bytes;
      impl->tensors[name] = std::move(location);
    }
  } catch (const std::exception& e) {
    SetError(error, path.string() + ": " + e.what());
    return nullptr;
  }

  Impl& m = *impl;
  if (!Check(hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking),
             "stream", error) ||
      !m.gemm.Init(m.stream, error))
    return nullptr;
  if (rocblas_create_handle(&m.blas) != rocblas_status_success ||
      rocblas_set_stream(m.blas, m.stream) != rocblas_status_success) {
    SetError(error, "Wan VAE rocBLAS handle");
    return nullptr;
  }
  const std::string d = "decoder.";
  if (!m.LoadConv("post_quant_conv", 1, 1, 0, &m.post_quant, error) ||
      !m.LoadConv(d + "conv_in", 3, 3, 0, &m.conv_in, error) ||
      !m.LoadResnet(d + "mid_block.resnets.0", kTop, kTop, &m.mid0, error) ||
      !m.LoadResnet(d + "mid_block.resnets.1", kTop, kTop, &m.mid1, error) ||
      !m.LoadGamma(d + "mid_block.attentions.0.norm.gamma", kTop, &m.attn_gamma,
                   error) ||
      !m.LoadConv(d + "mid_block.attentions.0.to_qkv", 1, 1, 0, &m.attn_qkv,
                  error) ||
      !m.LoadConv(d + "mid_block.attentions.0.proj", 1, 1, 0, &m.attn_proj,
                  error) ||
      !m.LoadGamma(d + "norm_out.gamma", 256, &m.norm_out, error) ||
      !m.LoadConv(d + "conv_out", 3, 3, kOutPadded, &m.conv_out, error))
    return nullptr;
  // dims [1024, 1024, 1024, 512, 256]; temporal upsampling in blocks 0, 1.
  constexpr int kDims[5] = {1024, 1024, 1024, 512, 256};
  for (int b = 0; b < 4; ++b) {
    UpBlock& block = m.up[b];
    block.in_channels = kDims[b];
    block.out_channels = kDims[b + 1];
    block.mode = b < 2 ? 3 : (b == 2 ? 2 : 0);
    const std::string prefix = d + "up_blocks." + std::to_string(b);
    block.resnets.resize(3);
    for (int r = 0; r < 3; ++r)
      if (!m.LoadResnet(prefix + ".resnets." + std::to_string(r),
                        r == 0 ? block.in_channels : block.out_channels,
                        block.out_channels, &block.resnets[r], error))
        return nullptr;
    if (block.mode == 3 && !m.LoadConv(prefix + ".upsampler.time_conv", 3, 1, 0,
                                       &block.time_conv, error))
      return nullptr;
    if (block.mode != 0 && !m.LoadConv(prefix + ".upsampler.resample.1", 1, 3,
                                       0, &block.resample, error))
      return nullptr;
  }
  platform::CloseFile(m.file);
  m.file = platform::kInvalidFile;
  m.tensors.clear();
  return std::unique_ptr<VaeDecoder>(new VaeDecoder(std::move(impl)));
}

bool VaeDecoder::Decode(std::span<const float> latent, int frames, int height,
                        int width, bool want_float, bool want_rgb8,
                        DecodedVideo* video,
                        const std::function<void(int, int)>& progress,
                        const std::function<bool()>& keep_going,
                        std::string* error) {
  Impl& m = *impl_;
  const std::size_t plane = static_cast<std::size_t>(height) * width;
  if (frames < 1 || height < 1 || width < 1 ||
      latent.size() != plane * frames * kZ) {
    SetError(error, "Wan VAE: latent size does not match its shape");
    return false;
  }
  if (!m.Prepare(height, width, error))
    return false;
  const int out_height = height * kSpatialRatio;
  const int out_width = width * kSpatialRatio;
  const int total = 1 + (frames - 1) * kTemporalRatio;
  const std::size_t frame_values =
      static_cast<std::size_t>(out_height) * out_width * 3;
  video->frames = total;
  video->height = out_height;
  video->width = out_width;
  video->rgb.clear();
  video->rgb8.clear();
  if (want_float)
    video->rgb.resize(frame_values * total);
  if (want_rgb8)
    video->rgb8.resize(frame_values * total);
  std::vector<float> chunk(frame_values * kTemporalRatio);
  std::vector<std::uint16_t> host(plane * kZ);
  int written = 0;
  for (int f = 0; f < frames; ++f) {
    if (keep_going && !keep_going()) {
      SetError(error, "cancelled");
      return false;
    }
    // De-normalize (latents * std + mean) into channels-last BF16.
    for (int c = 0; c < kZ; ++c) {
      const float* source =
          latent.data() + (static_cast<std::size_t>(c) * frames + f) * plane;
      for (std::size_t i = 0; i < plane; ++i)
        host[i * kZ + c] = HostBf16(source[i] * m.std_dev[c] + m.mean[c]);
    }
    if (!Check(hipMemcpyAsync(m.latent_device.get(), host.data(),
                              host.size() * 2, hipMemcpyHostToDevice, m.stream),
               "latent upload", error))
      return false;
    int produced = 0;
    if (!m.DecodeFrame(f, f == 0, m.latent_device.get(), chunk.data(),
                       &produced, error))
      return false;
    const std::size_t values = frame_values * produced;
    if (want_float)
      std::copy_n(chunk.data(), values,
                  video->rgb.data() + frame_values * written);
    if (want_rgb8) {
      std::uint8_t* out = video->rgb8.data() + frame_values * written;
      for (std::size_t i = 0; i < values; ++i) {
        const float unit = std::clamp(chunk[i] * 0.5F + 0.5F, 0.0F, 1.0F);
        out[i] = static_cast<std::uint8_t>(unit * 255.0F);
      }
    }
    written += produced;
    if (progress)
      progress(f + 1, frames);
  }
  return true;
}

}  // namespace gufo::models::wan22
