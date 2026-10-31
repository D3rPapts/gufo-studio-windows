// Written with AI assistance (Claude Code).
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "src/models/wan22/attention.hip.hpp"
#include "src/models/wan22/gemm.hpp"
#include "src/models/wan22/infinitetalk.hpp"
#include "src/models/wan22/tensor_file.hpp"

// ComfyUI reference: comfy/ldm/wan/model_multitalk.py (MultiTalkAudioProjModel,
// SingleStreamMultiAttention, get_attn_map_with_target) and
// comfy_extras/nodes_wan.py (WanInfiniteTalkToVideo).

namespace gufo::models::wan22 {
namespace {

constexpr int kLayers = 40;
constexpr int kHeads = 40;
constexpr int kHeadDim = 128;
constexpr int kWidth = Transformer21::kModelWidth;  // 5120
constexpr int kTokens = InfiniteTalk::kContextTokens;
constexpr int kAudio = InfiniteTalk::kAudioWidth;
constexpr int kAudioLayers = InfiniteTalk::kAudioLayers;
constexpr int kFirstInput = InfiniteTalk::kAudioWindow * kAudioLayers * kAudio;
constexpr int kLaterWindows = InfiniteTalk::kAudioWindow + 4 - 1;  // 8
constexpr int kLaterInput = kLaterWindows * kAudioLayers * kAudio;
constexpr int kHidden = 512;
constexpr float kEpsilon = 1e-5F;  // torch LayerNorm default
constexpr int kThreads = 256;
constexpr int kRowValues = kWidth / kThreads;  // 20
// SingleStreamMultiAttention: class_range 24, class_interval 4.
constexpr float kSpeaker1Low = 0.0F, kSpeaker1High = 4.0F;
constexpr float kSpeaker2Low = 20.0F, kSpeaker2High = 24.0F;
constexpr int kMaxKeys = InfiniteTalk::kMaxSpeakers * kTokens;  // 64

static_assert(kWidth == kHeads * kHeadDim && kWidth % kThreads == 0);

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

__device__ __forceinline__ float BlockSum(float value, float* shared) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
#pragma unroll
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  const float result = shared[0];
  __syncthreads();
  return result;
}

// out = bf16(layer_norm(hidden) * weight + bias), rows of 5120 (norm_x).
__global__ void NormKernel(const float* hidden, const float* weight,
                           const float* bias, std::uint16_t* out) {
  __shared__ float shared[kThreads];
  const std::size_t row = blockIdx.x;
  const float* x = hidden + row * kWidth;
  float values[kRowValues];
  float sum = 0;
#pragma unroll
  for (int i = 0; i < kRowValues; ++i) {
    values[i] = x[threadIdx.x + i * kThreads];
    sum += values[i];
  }
  const float mean = BlockSum(sum, shared) / kWidth;
  float squares = 0;
#pragma unroll
  for (int i = 0; i < kRowValues; ++i) {
    values[i] -= mean;
    squares += values[i] * values[i];
  }
  const float rstd = rsqrtf(BlockSum(squares, shared) / kWidth + kEpsilon);
#pragma unroll
  for (int i = 0; i < kRowValues; ++i) {
    const int c = threadIdx.x + i * kThreads;
    out[row * kWidth + c] = ToBf16(values[i] * rstd * weight[c] + bias[c]);
  }
}

// hidden += scale * y (y BF16).
__global__ void AddScaledKernel(const std::uint16_t* y, float scale,
                                std::size_t count, float* hidden) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    hidden[i] += scale * FromBf16(y[i]);
}

// Audio projection rows [speakers * frames, 512]: row % frames == 0 is the
// first frame (proj1 bias), the rest proj1_vf; then ReLU.
__global__ void FirstBiasReluKernel(float* x, const float* first_bias,
                                    const float* later_bias, int frames,
                                    std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const int c = static_cast<int>(i % kHidden);
  const int row = static_cast<int>(i / kHidden);
  const float v = x[i] + (row % frames == 0 ? first_bias[c] : later_bias[c]);
  x[i] = v > 0.0F ? v : 0.0F;
}

__global__ void BiasKernel(float* x, const float* bias, int width, bool relu,
                           std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float v = x[i] + bias[i % width];
  x[i] = relu && v < 0.0F ? 0.0F : v;
}

// LayerNorm(768) of the proj3 rows [speaker][frame][32 x 768] into the
// token layout [frame][speaker * 32 + token][768] (BF16 and FP32).
__global__ void TokenNormKernel(const float* x, const float* weight,
                                const float* bias, int frames, int speakers,
                                std::uint16_t* out, float* out32) {
  __shared__ float shared[kThreads];
  const int row = blockIdx.x;  // (speaker * frames + frame) * 32 + token
  const int token = row % kTokens;
  const int frame = (row / kTokens) % frames;
  const int speaker = row / (kTokens * frames);
  const float* a = x + static_cast<std::size_t>(row) * kAudio;
  float sum = 0;
  for (int c = threadIdx.x; c < kAudio; c += kThreads)
    sum += a[c];
  const float mean = BlockSum(sum, shared) / kAudio;
  float squares = 0;
  for (int c = threadIdx.x; c < kAudio; c += kThreads) {
    const float d = a[c] - mean;
    squares += d * d;
  }
  const float rstd = rsqrtf(BlockSum(squares, shared) / kAudio + kEpsilon);
  const std::size_t target =
      (static_cast<std::size_t>(frame) * speakers * kTokens +
       static_cast<std::size_t>(speaker) * kTokens + token) *
      kAudio;
  for (int c = threadIdx.x; c < kAudio; c += kThreads) {
    const float y = (a[c] - mean) * rstd * weight[c] + bias[c];
    out[target + c] = ToBf16(y);
    out32[target + c] = y;
  }
}

// MultiTalk's 1D RoPE: pairs (2p, 2p + 1) turned by position * 10000^(-2p /
// 128) (rotate_half on interleaved pairs). In place on the keys of the
// kv rows [rows][2 * 5120] (k first): audio token j of a frame with `keys`
// tokens sits at speaker 1's centre for j < keys / 2, else speaker 2's.
__global__ void RopeKeysKernel(std::uint16_t* kv, int rows, int keys) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  constexpr int kPairs = kWidth / 2;
  if (i >= static_cast<std::size_t>(rows) * kPairs)
    return;
  const int row = static_cast<int>(i / kPairs);
  const int pair = static_cast<int>(i % kPairs);
  const int p = pair % (kHeadDim / 2);
  const float position = (row % keys) < keys / 2
                             ? 0.5F * (kSpeaker1Low + kSpeaker1High)
                             : 0.5F * (kSpeaker2Low + kSpeaker2High);
  const float frequency =
      1.0F / powf(10000.0F, static_cast<float>(2 * p) / kHeadDim);
  float s, c;
  sincosf(position * frequency, &s, &c);
  std::uint16_t* x = kv + static_cast<std::size_t>(row) * 2 * kWidth + 2 * pair;
  const float a = FromBf16(x[0]), b = FromBf16(x[1]);
  x[0] = ToBf16(a * c - b * s);
  x[1] = ToBf16(b * c + a * s);
}

// Audio cross-attention of one video token (one block): its latent frame's
// `keys` audio tokens, 40 heads of 128, scale 1/sqrt(128). `positions`
// (two speakers) turns the query by the token's routing position first.
__global__ void AudioAttentionKernel(const std::uint16_t* query,
                                     const std::uint16_t* kv,
                                     const float* positions, int frame_tokens,
                                     int keys, std::uint16_t* out) {
  __shared__ float q[kWidth];
  __shared__ float scores[kHeads * kMaxKeys];
  __shared__ float turn[kHeadDim];  // cos, sin per pair
  const std::size_t token = blockIdx.x;
  const int frame = static_cast<int>(token / frame_tokens);
  const std::uint16_t* x = query + token * kWidth;
  if (positions != nullptr) {
    if (threadIdx.x < kHeadDim / 2) {
      const int p = threadIdx.x;
      const float frequency =
          1.0F / powf(10000.0F, static_cast<float>(2 * p) / kHeadDim);
      float s, c;
      sincosf(positions[token] * frequency, &s, &c);
      turn[2 * p] = c;
      turn[2 * p + 1] = s;
    }
    __syncthreads();
    for (int pair = threadIdx.x; pair < kWidth / 2; pair += kThreads) {
      const int p = pair % (kHeadDim / 2);
      const float a = FromBf16(x[2 * pair]), b = FromBf16(x[2 * pair + 1]);
      q[2 * pair] = a * turn[2 * p] - b * turn[2 * p + 1];
      q[2 * pair + 1] = b * turn[2 * p] + a * turn[2 * p + 1];
    }
  } else {
    for (int c = threadIdx.x; c < kWidth; c += kThreads)
      q[c] = FromBf16(x[c]);
  }
  __syncthreads();
  const float scale = rsqrtf(static_cast<float>(kHeadDim));
  const std::uint16_t* frame_kv =
      kv + static_cast<std::size_t>(frame) * keys * 2 * kWidth;
  for (int index = threadIdx.x; index < kHeads * keys; index += kThreads) {
    const int head = index / keys;
    const int j = index % keys;
    const std::uint16_t* k =
        frame_kv + static_cast<std::size_t>(j) * 2 * kWidth + head * kHeadDim;
    const float* qh = q + head * kHeadDim;
    float dot = 0;
#pragma unroll 8
    for (int d = 0; d < kHeadDim; ++d)
      dot += qh[d] * FromBf16(k[d]);
    scores[head * kMaxKeys + j] = dot * scale;
  }
  __syncthreads();
  if (threadIdx.x < kHeads) {
    float* row = scores + threadIdx.x * kMaxKeys;
    float maximum = -INFINITY;
    for (int j = 0; j < keys; ++j)
      maximum = fmaxf(maximum, row[j]);
    float sum = 0;
    for (int j = 0; j < keys; ++j) {
      row[j] = expf(row[j] - maximum);
      sum += row[j];
    }
    const float inverse = 1.0F / sum;
    for (int j = 0; j < keys; ++j)
      row[j] *= inverse;
  }
  __syncthreads();
  for (int c = threadIdx.x; c < kWidth; c += kThreads) {
    const int head = c / kHeadDim;
    const float* p = scores + head * kMaxKeys;
    const std::uint16_t* v = frame_kv + kWidth + c;
    float sum = 0;
    for (int j = 0; j < keys; ++j)
      sum += p[j] * FromBf16(v[static_cast<std::size_t>(j) * 2 * kWidth]);
    out[token * kWidth + c] = ToBf16(sum);
  }
}

// x_ref_attn_map: per token and speaker c the head mean of the attention
// output's channel c (the masked probability mass) / mask size.
__global__ void MapKernel(const std::uint16_t* attention, int tokens,
                          float count0, float count1, float* map) {
  const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= tokens)
    return;
  const std::uint16_t* row = attention + static_cast<std::size_t>(t) * kWidth;
  float a = 0, b = 0;
  for (int h = 0; h < kHeads; ++h) {
    a += FromBf16(row[h * kHeadDim]);
    b += FromBf16(row[h * kHeadDim + 1]);
  }
  map[t] = a / kHeads / (count0 + 1e-8F);
  map[tokens + t] = b / kHeads / (count1 + 1e-8F);
}

// Routing positions from the map (one block of 1024 threads): each speaker's
// map min-max scaled into its RoPE range, the larger map wins (ties: speaker
// 1, as argmax).
__global__ void PositionsKernel(const float* map, int tokens,
                                float* positions) {
  __shared__ float low[2][1024], high[2][1024];
  const int tid = threadIdx.x;
  float l0 = INFINITY, l1 = INFINITY, h0 = -INFINITY, h1 = -INFINITY;
  for (int t = tid; t < tokens; t += blockDim.x) {
    l0 = fminf(l0, map[t]);
    h0 = fmaxf(h0, map[t]);
    l1 = fminf(l1, map[tokens + t]);
    h1 = fmaxf(h1, map[tokens + t]);
  }
  low[0][tid] = l0;
  low[1][tid] = l1;
  high[0][tid] = h0;
  high[1][tid] = h1;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      for (int c = 0; c < 2; ++c) {
        low[c][tid] = fminf(low[c][tid], low[c][tid + stride]);
        high[c][tid] = fmaxf(high[c][tid], high[c][tid + stride]);
      }
    __syncthreads();
  }
  const float min0 = low[0][0], max0 = high[0][0];
  const float min1 = low[1][0], max1 = high[1][0];
  for (int t = tid; t < tokens; t += blockDim.x) {
    const float a = map[t], b = map[tokens + t];
    positions[t] =
        a >= b ? (a - min0) / (max0 - min0 + 1e-8F) *
                         (kSpeaker1High - kSpeaker1Low) +
                     kSpeaker1Low
               : (b - min1) / (max1 - min1 + 1e-8F) *
                         (kSpeaker2High - kSpeaker2Low) +
                     kSpeaker2Low;
  }
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
  SetError(error, std::string("InfiniteTalk ") + operation + ": " +
                      hipGetErrorString(status));
  return false;
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
    if (!Check(hipMalloc(&data_, std::max<std::size_t>(bytes, 256)),
               "device allocation", error))
      return false;
    bytes_ = bytes;
    return true;
  }
  // Grows to at least `bytes` (contents lost).
  bool Reserve(std::size_t bytes, std::string* error) {
    return bytes <= bytes_ && data_ != nullptr ? true : Allocate(bytes, error);
  }
  template<typename T>
  bool Upload(const std::vector<T>& host, std::string* error) {
    return Allocate(host.size() * sizeof(T), error) &&
           Check(hipMemcpy(data_, host.data(), host.size() * sizeof(T),
                           hipMemcpyHostToDevice),
                 "weight upload", error);
  }
  void Release() {
    if (data_ != nullptr)
      (void)hipFree(data_);
    data_ = nullptr;
    bytes_ = 0;
  }
  std::uint16_t* get() const { return static_cast<std::uint16_t*>(data_); }
  float* f32() const { return static_cast<float*>(data_); }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

struct Block {
  DeviceBuffer norm_weight, norm_bias;  // FP32 [5120]
  DeviceBuffer q, q_bias;               // BF16 [5120, 5120]
  DeviceBuffer kv, kv_bias;             // BF16 [10240, 768]
  DeviceBuffer proj, proj_bias;         // BF16 [5120, 5120]
};

}  // namespace

struct InfiniteTalk::Impl {
  ~Impl() {
    if (blas != nullptr)
      (void)rocblas_destroy_handle(blas);
  }
  bool Sgemm(const float* in, const DeviceBuffer& weight, int tokens, int n,
             int k, float* out, std::string* error) {
    const float one = 1.0F, zero = 0.0F;
    if (rocblas_sgemm(blas, rocblas_operation_transpose,
                      rocblas_operation_none, n, tokens, k, &one,
                      weight.f32(), k, in, k, &zero, out,
                      n) != rocblas_status_success) {
      SetError(error, "InfiniteTalk audio projection GEMM failed");
      return false;
    }
    return true;
  }

  std::string variant;
  std::vector<Block> blocks;
  DeviceBuffer proj1, proj1_bias, proj1_vf, proj1_vf_bias;  // FP32
  DeviceBuffer proj2, proj2_bias, proj3, proj3_bias, norm_weight, norm_bias;
  std::size_t weight_bytes{0};
  hipStream_t stream{nullptr};
  rocblas_handle blas{nullptr};
  LinearGemm gemm;
  float audio_scale{1.0F};

  // The window's audio tokens [frames][speakers * 32][768].
  DeviceBuffer audio_tokens, audio_tokens32;
  int audio_frames{0};
  int speakers{0};
  // Routing.
  std::vector<std::vector<std::uint8_t>> masks;
  int mask_height{0}, mask_width{0};
  float mask_count[2]{};
  DeviceBuffer mask_packed;  // FP16 values for LaunchAttention
  DeviceBuffer ref_key, map, positions;
  bool routed{false};  // AfterSelfAttention produced this block's positions
  // Per Forward.
  int frames{0}, frame_tokens{0}, tokens{0};
  DeviceBuffer a, b, kv;  // BF16 [tokens, 5120] x 2, [frames * keys, 10240]
};

InfiniteTalk::InfiniteTalk(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
InfiniteTalk::~InfiniteTalk() = default;

const std::string& InfiniteTalk::variant() const {
  return impl_->variant;
}

std::size_t InfiniteTalk::weight_bytes() const {
  return impl_->weight_bytes;
}

void InfiniteTalk::set_audio_scale(float scale) {
  impl_->audio_scale = scale;
}

std::unique_ptr<InfiniteTalk> InfiniteTalk::Load(
    const std::filesystem::path& file, hipStream_t stream,
    std::string* error) {
  TensorFile tensors;
  if (!tensors.Open(file, error))
    return nullptr;
  if (!tensors.Has("audio_proj.proj1.weight") ||
      tensors.Shape("audio_proj.proj1.weight") !=
          std::vector<std::size_t>{kHidden, kFirstInput} ||
      tensors.Shape("audio_proj.proj1_vf.weight") !=
          std::vector<std::size_t>{kHidden, kLaterInput} ||
      tensors.Shape("blocks.0.audio_cross_attn.proj.weight") !=
          std::vector<std::size_t>{kWidth, kWidth} ||
      !tensors.Has("blocks." + std::to_string(kLayers - 1) +
                   ".audio_cross_attn.q_linear.weight")) {
    SetError(error, file.string() +
                        " is not an InfiniteTalk / MultiTalk model patch for "
                        "Wan 2.1 14B");
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  m.variant = tensors.Metadata("model_type");
  m.stream = stream;
  if (!m.gemm.Init(stream, error))
    return nullptr;
  if (rocblas_create_handle(&m.blas) != rocblas_status_success ||
      rocblas_set_stream(m.blas, stream) != rocblas_status_success) {
    SetError(error, "InfiniteTalk rocBLAS handle");
    return nullptr;
  }
  std::vector<float> f32;
  std::vector<std::uint16_t> bf16;
  const auto load32 = [&](DeviceBuffer& buffer, const std::string& name,
                          std::size_t count) {
    if (!tensors.ReadF32(name, count, &f32, error) ||
        !buffer.Upload(f32, error))
      return false;
    m.weight_bytes += f32.size() * 4;
    return true;
  };
  const auto load16 = [&](DeviceBuffer& buffer, const std::string& name,
                          std::size_t count) {
    if (!tensors.ReadBf16(name, count, &bf16, error) ||
        !buffer.Upload(bf16, error))
      return false;
    m.weight_bytes += bf16.size() * 2;
    return true;
  };
  const std::string a = "audio_proj.";
  if (!load32(m.proj1, a + "proj1.weight",
              static_cast<std::size_t>(kHidden) * kFirstInput) ||
      !load32(m.proj1_bias, a + "proj1.bias", kHidden) ||
      !load32(m.proj1_vf, a + "proj1_vf.weight",
              static_cast<std::size_t>(kHidden) * kLaterInput) ||
      !load32(m.proj1_vf_bias, a + "proj1_vf.bias", kHidden) ||
      !load32(m.proj2, a + "proj2.weight",
              static_cast<std::size_t>(kHidden) * kHidden) ||
      !load32(m.proj2_bias, a + "proj2.bias", kHidden) ||
      !load32(m.proj3, a + "proj3.weight",
              static_cast<std::size_t>(kTokens) * kAudio * kHidden) ||
      !load32(m.proj3_bias, a + "proj3.bias",
              static_cast<std::size_t>(kTokens) * kAudio) ||
      !load32(m.norm_weight, a + "norm.weight", kAudio) ||
      !load32(m.norm_bias, a + "norm.bias", kAudio))
    return nullptr;
  m.blocks.resize(kLayers);
  const std::size_t square = static_cast<std::size_t>(kWidth) * kWidth;
  for (int layer = 0; layer < kLayers; ++layer) {
    Block& b = m.blocks[layer];
    const std::string p = "blocks." + std::to_string(layer) + ".";
    const std::string c = p + "audio_cross_attn.";
    if (!load32(b.norm_weight, p + "norm_x.weight", kWidth) ||
        !load32(b.norm_bias, p + "norm_x.bias", kWidth) ||
        !load16(b.q, c + "q_linear.weight", square) ||
        !load16(b.q_bias, c + "q_linear.bias", kWidth) ||
        !load16(b.kv, c + "kv_linear.weight",
                static_cast<std::size_t>(2 * kWidth) * kAudio) ||
        !load16(b.kv_bias, c + "kv_linear.bias", 2 * kWidth) ||
        !load16(b.proj, c + "proj.weight", square) ||
        !load16(b.proj_bias, c + "proj.bias", kWidth))
      return nullptr;
  }
  return std::unique_ptr<InfiniteTalk>(new InfiniteTalk(std::move(impl)));
}

bool InfiniteTalk::SetAudio(const std::vector<std::span<const float>>& features,
                            int feature_frames, int start, int length,
                            std::string* error) {
  Impl& m = *impl_;
  const int speakers = static_cast<int>(features.size());
  const std::size_t frame_values =
      static_cast<std::size_t>(kAudioLayers) * kAudio;
  if (speakers < 1 || speakers > kMaxSpeakers || feature_frames < 1 ||
      start < 0 || length < 1 || (length - 1) % 4 != 0) {
    SetError(error, "InfiniteTalk: 1 or 2 speakers and a window of 4k + 1 "
                    "frames are required");
    return false;
  }
  for (const auto& f : features)
    if (f.size() != frame_values * feature_frames) {
      SetError(error, "InfiniteTalk: speaker features differ in length");
      return false;
    }
  // get_audio_embeds: frames past the end repeat frame 0; a window of 5
  // around each video frame, clamped to the (padded) audio.
  const int end = start + length;
  const int padded = std::max(feature_frames, end);
  const auto frame_of = [&](int speaker, int index) {
    index = std::clamp(index, 0, padded - 1);
    if (index >= feature_frames)
      index = 0;
    return features[speaker].data() + frame_values * index;
  };
  const int groups = (length - 1) / 4;
  const int latent = 1 + groups;
  std::vector<float> first(static_cast<std::size_t>(speakers) * kFirstInput);
  std::vector<float> later(static_cast<std::size_t>(speakers) *
                           std::max(groups, 1) * kLaterInput);
  for (int s = 0; s < speakers; ++s) {
    float* target = first.data() + static_cast<std::size_t>(s) * kFirstInput;
    for (int w = 0; w < InfiniteTalk::kAudioWindow; ++w)
      std::copy_n(frame_of(s, start + w - 2), frame_values,
                  target + w * frame_values);
    // Each later group of 4 frames: windows 0..2 of its first frame, the
    // centre of the middle two, windows 2..4 of its last.
    for (int g = 0; g < groups; ++g) {
      float* row = later.data() +
                   (static_cast<std::size_t>(s) * groups + g) * kLaterInput;
      int slot = 0;
      const auto put = [&](int frame, int w) {
        std::copy_n(frame_of(s, start + frame + w - 2), frame_values,
                    row + static_cast<std::size_t>(slot++) * frame_values);
      };
      const int base = 1 + 4 * g;
      for (int w = 0; w <= 2; ++w)
        put(base, w);
      put(base + 1, 2);
      put(base + 2, 2);
      for (int w = 2; w <= 4; ++w)
        put(base + 3, w);
    }
  }
  const std::size_t rows = static_cast<std::size_t>(speakers) * latent;
  DeviceBuffer d_first, d_later, hidden, hidden2, expanded;
  if (!d_first.Upload(first, error) || !d_later.Upload(later, error) ||
      !hidden.Allocate(rows * kHidden * 4, error) ||
      !hidden2.Allocate(rows * kHidden * 4, error) ||
      !expanded.Allocate(rows * kTokens * kAudio * 4, error) ||
      !m.audio_tokens.Reserve(rows * kTokens * kAudio * 2, error) ||
      !m.audio_tokens32.Reserve(rows * kTokens * kAudio * 4, error))
    return false;
  // proj1 / proj1_vf into rows speaker * latent + (0 | 1 + group).
  for (int s = 0; s < speakers; ++s) {
    if (!m.Sgemm(d_first.f32() + static_cast<std::size_t>(s) * kFirstInput,
                 m.proj1, 1, kHidden, kFirstInput,
                 hidden.f32() + static_cast<std::size_t>(s) * latent * kHidden,
                 error))
      return false;
    if (groups > 0 &&
        !m.Sgemm(d_later.f32() +
                     static_cast<std::size_t>(s) * groups * kLaterInput,
                 m.proj1_vf, groups, kHidden, kLaterInput,
                 hidden.f32() +
                     (static_cast<std::size_t>(s) * latent + 1) * kHidden,
                 error))
      return false;
  }
  hipStream_t st = m.stream;
  FirstBiasReluKernel<<<Blocks(rows * kHidden), kThreads, 0, st>>>(
      hidden.f32(), m.proj1_bias.f32(), m.proj1_vf_bias.f32(), latent,
      rows * kHidden);
  if (!m.Sgemm(hidden.f32(), m.proj2, static_cast<int>(rows), kHidden,
               kHidden, hidden2.f32(), error))
    return false;
  BiasKernel<<<Blocks(rows * kHidden), kThreads, 0, st>>>(
      hidden2.f32(), m.proj2_bias.f32(), kHidden, true, rows * kHidden);
  if (!m.Sgemm(hidden2.f32(), m.proj3, static_cast<int>(rows),
               kTokens * kAudio, kHidden, expanded.f32(), error))
    return false;
  BiasKernel<<<Blocks(rows * kTokens * kAudio), kThreads, 0, st>>>(
      expanded.f32(), m.proj3_bias.f32(), kTokens * kAudio, false,
      rows * kTokens * kAudio);
  TokenNormKernel<<<static_cast<unsigned>(rows * kTokens), kThreads, 0, st>>>(
      expanded.f32(), m.norm_weight.f32(), m.norm_bias.f32(), latent,
      speakers, m.audio_tokens.get(), m.audio_tokens32.f32());
  m.audio_frames = latent;
  m.speakers = speakers;
  return Check(hipGetLastError(), "audio projection", error) &&
         Check(hipStreamSynchronize(st), "audio projection", error);
}

bool InfiniteTalk::DownloadAudioTokens(std::vector<float>* tokens,
                                       std::string* error) {
  Impl& m = *impl_;
  const std::size_t count = static_cast<std::size_t>(m.audio_frames) *
                            m.speakers * kTokens * kAudio;
  tokens->resize(count);
  return Check(hipMemcpyAsync(tokens->data(), m.audio_tokens32.f32(),
                              count * 4, hipMemcpyDeviceToHost, m.stream),
               "readback", error) &&
         Check(hipStreamSynchronize(m.stream), "readback", error);
}

bool InfiniteTalk::SetSpeakerMasks(
    const std::vector<std::vector<std::uint8_t>>& masks, int token_height,
    int token_width, std::string* error) {
  Impl& m = *impl_;
  m.masks.clear();
  m.mask_packed.Release();
  if (masks.empty())
    return true;
  const std::size_t area = static_cast<std::size_t>(token_height) * token_width;
  if (masks.size() != 2 || masks[0].size() != area || masks[1].size() != area) {
    SetError(error, "InfiniteTalk: two speaker masks of the token grid are "
                    "required");
    return false;
  }
  for (int c = 0; c < 2; ++c) {
    m.mask_count[c] = static_cast<float>(
        std::count_if(masks[c].begin(), masks[c].end(),
                      [](std::uint8_t v) { return v != 0; }));
    if (m.mask_count[c] == 0.0F) {
      SetError(error, "InfiniteTalk: speaker mask " + std::to_string(c + 1) +
                          " is empty at the latent resolution");
      return false;
    }
  }
  // Values [heads][area][128]: channel c = mask c, so the attention output's
  // channel c is the masked probability mass.
  std::vector<std::uint16_t> values(static_cast<std::size_t>(kHeads) * area *
                                        kHeadDim,
                                    0);
  const std::uint16_t one = FloatToBf16Bits(1.0F);
  for (int h = 0; h < kHeads; ++h)
    for (std::size_t r = 0; r < area; ++r)
      for (int c = 0; c < 2; ++c)
        if (masks[c][r] != 0)
          values[(static_cast<std::size_t>(h) * area + r) * kHeadDim + c] = one;
  DeviceBuffer staging;
  if (!staging.Upload(values, error) ||
      !m.mask_packed.Allocate(static_cast<std::size_t>(kHeads) * kHeadDim *
                                  PaddedRows(static_cast<std::uint32_t>(area)) *
                                  2,
                              error))
    return false;
  LaunchPackValues(staging.get(), m.mask_packed.get(),
                   static_cast<std::uint32_t>(area), kHeads, m.stream);
  if (!Check(hipStreamSynchronize(m.stream), "mask upload", error))
    return false;
  m.masks = masks;
  m.mask_height = token_height;
  m.mask_width = token_width;
  return true;
}

bool InfiniteTalk::Begin(const float* input, int frames, int height,
                         int width, const float* temb, hipStream_t stream,
                         std::string* error) {
  (void)input, (void)temb, (void)stream;
  Impl& m = *impl_;
  if (m.audio_frames == 0) {
    SetError(error, "InfiniteTalk: no audio set");
    return false;
  }
  if (frames != m.audio_frames) {
    SetError(error, "InfiniteTalk: the audio window has " +
                        std::to_string(m.audio_frames) +
                        " latent frames, the video " + std::to_string(frames));
    return false;
  }
  m.frames = frames;
  m.frame_tokens = (height / 2) * (width / 2);
  m.tokens = frames * m.frame_tokens;
  if (!m.masks.empty() &&
      (m.mask_height != height / 2 || m.mask_width != width / 2)) {
    SetError(error, "InfiniteTalk: speaker masks do not match the video size");
    return false;
  }
  if (m.speakers > 1 && m.masks.empty()) {
    SetError(error, "InfiniteTalk: two speakers need two speaker masks");
    return false;
  }
  const std::size_t T = m.tokens;
  const int keys = m.speakers * kTokens;
  if (!m.a.Reserve(T * kWidth * 2, error) ||
      !m.b.Reserve(T * kWidth * 2, error) ||
      !m.kv.Reserve(static_cast<std::size_t>(frames) * keys * 2 * kWidth * 2,
                    error))
    return false;
  if (m.speakers > 1 &&
      (!m.ref_key.Reserve(static_cast<std::size_t>(kHeads) * m.frame_tokens *
                              kHeadDim * 2,
                          error) ||
       !m.map.Reserve(2 * T * 4, error) ||
       !m.positions.Reserve(T * 4, error)))
    return false;
  m.routed = false;
  return true;
}

bool InfiniteTalk::AfterBlock(int layer, float* hidden, hipStream_t stream,
                              std::string* error) {
  (void)layer, (void)hidden, (void)stream, (void)error;
  return true;
}

bool InfiniteTalk::AfterSelfAttention(int layer, const std::uint16_t* query,
                                      const std::uint16_t* key, int tokens,
                                      hipStream_t stream, std::string* error) {
  (void)layer;
  Impl& m = *impl_;
  m.routed = false;
  if (m.speakers < 2)
    return true;
  if (tokens != m.tokens) {
    SetError(error, "InfiniteTalk: token count changed within a forward");
    return false;
  }
  // get_attn_map_with_target: queries of all tokens against the keys of the
  // first latent frame; masked softmax mass per speaker, head mean.
  const std::size_t S = m.frame_tokens;
  if (!Check(hipMemcpy2DAsync(m.ref_key.get(), S * kHeadDim * 2, key,
                              static_cast<std::size_t>(tokens) * kHeadDim * 2,
                              S * kHeadDim * 2, kHeads,
                              hipMemcpyDeviceToDevice, stream),
             "reference keys", error))
    return false;
  LaunchAttention(query, m.ref_key.get(), m.mask_packed.get(), m.a.get(),
                  static_cast<std::uint32_t>(tokens),
                  static_cast<std::uint32_t>(S), kHeads, stream);
  MapKernel<<<Blocks(tokens), kThreads, 0, stream>>>(
      m.a.get(), tokens, m.mask_count[0], m.mask_count[1], m.map.f32());
  PositionsKernel<<<1, 1024, 0, stream>>>(m.map.f32(), tokens,
                                          m.positions.f32());
  m.routed = true;
  return Check(hipGetLastError(), "speaker routing", error);
}

bool InfiniteTalk::AfterCrossAttention(int layer, float* hidden,
                                       hipStream_t stream, std::string* error) {
  Impl& m = *impl_;
  const Block& b = m.blocks[layer];
  const int keys = m.speakers * kTokens;
  const std::size_t count = static_cast<std::size_t>(m.tokens) * kWidth;
  if (m.speakers > 1 && !m.routed) {
    SetError(error, "InfiniteTalk: speaker routing missing for block " +
                        std::to_string(layer));
    return false;
  }
  NormKernel<<<static_cast<unsigned>(m.tokens), kThreads, 0, stream>>>(
      hidden, b.norm_weight.f32(), b.norm_bias.f32(), m.a.get());
  if (!m.gemm.Run(b.q.get(), b.q_bias.get(), m.a.get(), m.b.get(), m.tokens,
                  kWidth, kWidth, error) ||
      !m.gemm.Run(b.kv.get(), b.kv_bias.get(), m.audio_tokens.get(),
                  m.kv.get(), m.frames * keys, 2 * kWidth, kAudio, error))
    return false;
  if (m.speakers > 1)
    RopeKeysKernel<<<Blocks(static_cast<std::size_t>(m.frames) * keys *
                            (kWidth / 2)),
                     kThreads, 0, stream>>>(m.kv.get(), m.frames * keys, keys);
  AudioAttentionKernel<<<static_cast<unsigned>(m.tokens), kThreads, 0,
                         stream>>>(
      m.b.get(), m.kv.get(), m.speakers > 1 ? m.positions.f32() : nullptr,
      m.frame_tokens, keys, m.a.get());
  if (!m.gemm.Run(b.proj.get(), b.proj_bias.get(), m.a.get(), m.b.get(),
                  m.tokens, kWidth, kWidth, error))
    return false;
  AddScaledKernel<<<Blocks(count), kThreads, 0, stream>>>(
      m.b.get(), m.audio_scale, count, hidden);
  m.routed = false;
  return Check(hipGetLastError(), "audio cross-attention", error);
}

std::vector<std::uint8_t> TokenMask(std::span<const std::uint8_t> mask,
                                    int width, int height, int token_width,
                                    int token_height) {
  std::vector<std::uint8_t> out(static_cast<std::size_t>(token_width) *
                                token_height);
  if (width < 1 || height < 1 ||
      mask.size() != static_cast<std::size_t>(width) * height)
    return {};
  // torch 'nearest': source = floor(target * in / out).
  const float sy = static_cast<float>(height) / token_height;
  const float sx = static_cast<float>(width) / token_width;
  for (int y = 0; y < token_height; ++y) {
    const int ys = std::min(static_cast<int>(std::floor(y * sy)), height - 1);
    for (int x = 0; x < token_width; ++x) {
      const int xs = std::min(static_cast<int>(std::floor(x * sx)), width - 1);
      out[static_cast<std::size_t>(y) * token_width + x] =
          mask[static_cast<std::size_t>(ys) * width + xs] != 0 ? 1 : 0;
    }
  }
  return out;
}

}  // namespace gufo::models::wan22
