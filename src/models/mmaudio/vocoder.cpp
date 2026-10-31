// MMAudio for gufo: VAE decoder and BigVGAN v2 vocoder (see vocoder.hpp).
// Written with AI assistance (Claude Code).
#include "src/models/mmaudio/vocoder.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

#include "src/models/mmaudio/kernels.hpp"

namespace gufo::models::mmaudio {
namespace {

// MMAudio's DATA_MEAN_128D / DATA_STD_128D (ext/autoencoder/vae.py).
constexpr float kMelMean[128] = {
    -3.3462F,  -2.6723F,  -2.4893F,  -2.3143F,  -2.2664F,  -2.3317F,
    -2.1802F,  -2.4006F,  -2.2357F,  -2.4597F,  -2.3717F,  -2.4690F,
    -2.5142F,  -2.4919F,  -2.6610F,  -2.5047F,  -2.7483F,  -2.5926F,
    -2.7462F,  -2.7033F,  -2.7386F,  -2.8112F,  -2.7502F,  -2.9594F,
    -2.7473F,  -3.0035F,  -2.8891F,  -2.9922F,  -2.9856F,  -3.0157F,
    -3.1191F,  -2.9893F,  -3.1718F,  -3.0745F,  -3.1879F,  -3.2310F,
    -3.1424F,  -3.2296F,  -3.2791F,  -3.2782F,  -3.2756F,  -3.3134F,
    -3.3509F,  -3.3750F,  -3.3951F,  -3.3698F,  -3.4505F,  -3.4509F,
    -3.5089F,  -3.4647F,  -3.5536F,  -3.5788F,  -3.5867F,  -3.6036F,
    -3.6400F,  -3.6747F,  -3.7072F,  -3.7279F,  -3.7283F,  -3.7795F,
    -3.8259F,  -3.8447F,  -3.8663F,  -3.9182F,  -3.9605F,  -3.9861F,
    -4.0105F,  -4.0373F,  -4.0762F,  -4.1121F,  -4.1488F,  -4.1874F,
    -4.2461F,  -4.3170F,  -4.3639F,  -4.4452F,  -4.5282F,  -4.6297F,
    -4.7019F,  -4.7960F,  -4.8700F,  -4.9507F,  -5.0303F,  -5.0866F,
    -5.1634F,  -5.2342F,  -5.3242F,  -5.4053F,  -5.4927F,  -5.5712F,
    -5.6464F,  -5.7052F,  -5.7619F,  -5.8410F,  -5.9188F,  -6.0103F,
    -6.0955F,  -6.1673F,  -6.2362F,  -6.3120F,  -6.3926F,  -6.4797F,
    -6.5565F,  -6.6511F,  -6.8130F,  -6.9961F,  -7.1275F,  -7.2457F,
    -7.3576F,  -7.4663F,  -7.6136F,  -7.7469F,  -7.8815F,  -8.0132F,
    -8.1515F,  -8.3071F,  -8.4722F,  -8.7418F,  -9.3975F,  -9.6628F,
    -9.7671F,  -9.8863F,  -9.9992F,  -10.0860F, -10.1709F, -10.5418F,
    -11.2795F, -11.3861F};
constexpr float kMelStd[128] = {
    2.3804F, 2.4368F, 2.3772F, 2.3145F, 2.2803F, 2.2510F, 2.2316F, 2.2083F,
    2.1996F, 2.1835F, 2.1769F, 2.1659F, 2.1631F, 2.1618F, 2.1540F, 2.1606F,
    2.1571F, 2.1567F, 2.1612F, 2.1579F, 2.1679F, 2.1683F, 2.1634F, 2.1557F,
    2.1668F, 2.1518F, 2.1415F, 2.1449F, 2.1406F, 2.1350F, 2.1313F, 2.1415F,
    2.1281F, 2.1352F, 2.1219F, 2.1182F, 2.1327F, 2.1195F, 2.1137F, 2.1080F,
    2.1179F, 2.1036F, 2.1087F, 2.1036F, 2.1015F, 2.1068F, 2.0975F, 2.0991F,
    2.0902F, 2.1015F, 2.0857F, 2.0920F, 2.0893F, 2.0897F, 2.0910F, 2.0881F,
    2.0925F, 2.0873F, 2.0960F, 2.0900F, 2.0957F, 2.0958F, 2.0978F, 2.0936F,
    2.0886F, 2.0905F, 2.0845F, 2.0855F, 2.0796F, 2.0840F, 2.0813F, 2.0817F,
    2.0838F, 2.0840F, 2.0917F, 2.1061F, 2.1431F, 2.1976F, 2.2482F, 2.3055F,
    2.3700F, 2.4088F, 2.4372F, 2.4609F, 2.4731F, 2.4847F, 2.5072F, 2.5451F,
    2.5772F, 2.6147F, 2.6529F, 2.6596F, 2.6645F, 2.6726F, 2.6803F, 2.6812F,
    2.6899F, 2.6916F, 2.6931F, 2.6998F, 2.7062F, 2.7262F, 2.7222F, 2.7158F,
    2.7041F, 2.7485F, 2.7491F, 2.7451F, 2.7485F, 2.7233F, 2.7297F, 2.7233F,
    2.7145F, 2.6958F, 2.6788F, 2.6439F, 2.6007F, 2.4786F, 2.2469F, 2.1877F,
    2.1392F, 2.0717F, 2.0107F, 1.9676F, 1.9140F, 1.7102F, 0.9101F, 0.7164F};

constexpr float kClipAct = 256.0F;
constexpr float kMpSiluScale = 1.0F / 0.596F;
constexpr int kRowChunk = 65536;  // im2col rows per GEMM

/// conv1d (stride 1) through im2col in row chunks; `accumulate` adds to out.
bool RunConv1d(Blas& blas, DeviceBuffer& columns, const Matrix& w,
               const float* bias, const float* x, int rows, int channels,
               int kernel, int dilation, int padding, float* out,
               std::string* error, bool accumulate = false) {
  if (kernel == 1 && padding == 0)
    return blas.Linear(w, bias, x, rows, out, error, accumulate);
  for (int start = 0; start < rows; start += kRowChunk) {
    const int n = std::min(kRowChunk, rows - start);
    if (!columns.Reserve(static_cast<std::size_t>(n) * channels * kernel * 4,
                         error))
      return false;
    // Shifting the padding moves the window to output rows start..start+n.
    LaunchIm2Col(x, rows, channels, kernel, dilation, padding - start, n,
                 columns.f32(), blas.stream());
    if (!blas.Linear(w, bias, columns.f32(), n,
                     out + static_cast<std::size_t>(start) * w.rows, error,
                     accumulate))
      return false;
  }
  return true;
}

double BesselI0(double x) {
  double sum = 1.0, term = 1.0;
  for (int k = 1; k < 64; ++k) {
    term *= (x / (2.0 * k)) * (x / (2.0 * k));
    sum += term;
    if (term < 1e-17 * sum)
      break;
  }
  return sum;
}

}  // namespace

std::vector<float> KaiserSincFilter(double cutoff, double half_width,
                                    int kernel_size) {
  const bool even = kernel_size % 2 == 0;
  const int half_size = kernel_size / 2;
  const double delta_f = 4.0 * half_width;
  const double a =
      2.285 * (half_size - 1) * std::numbers::pi * delta_f + 7.95;
  double beta = 0.0;
  if (a > 50.0)
    beta = 0.1102 * (a - 8.7);
  else if (a >= 21.0)
    beta = 0.5842 * std::pow(a - 21.0, 0.4) + 0.07886 * (a - 21.0);
  std::vector<double> filter(kernel_size);
  const double n = kernel_size - 1;
  double sum = 0;
  for (int i = 0; i < kernel_size; ++i) {
    const double ratio = (i - n / 2.0) / (n / 2.0);
    const double window =
        BesselI0(beta * std::sqrt(std::max(0.0, 1.0 - ratio * ratio))) /
        BesselI0(beta);
    const double time = even ? (i - half_size) + 0.5 : i - half_size;
    const double x = 2.0 * cutoff * time;
    const double sinc =
        x == 0.0 ? 1.0
                 : std::sin(std::numbers::pi * x) / (std::numbers::pi * x);
    filter[i] = 2.0 * cutoff * window * sinc;
    sum += filter[i];
  }
  std::vector<float> out(kernel_size);
  for (int i = 0; i < kernel_size; ++i)
    out[i] = static_cast<float>(filter[i] / sum);
  return out;
}

void NormalizeMpWeight(std::vector<float>* weight, int out, float gain) {
  const std::size_t n = weight->size() / out;
  const double root = std::sqrt(static_cast<double>(n));
  for (int o = 0; o < out; ++o) {
    float* row = weight->data() + static_cast<std::size_t>(o) * n;
    double squares = 0;
    for (std::size_t i = 0; i < n; ++i)
      squares += static_cast<double>(row[i]) * row[i];
    const double norm = 1e-4 + std::sqrt(squares) / root;
    for (std::size_t i = 0; i < n; ++i)
      row[i] = static_cast<float>(row[i] / norm / root * gain);
  }
}

// ---- VAE decoder ---------------------------------------------------------------

namespace {

struct MpConv {
  Matrix w;  // [out, in * k]
  int in{0};
  int out{0};
  int k{3};
};

struct ResBlock {
  MpConv conv1, conv2;
  bool shortcut{false};
  MpConv nin;
};

struct AttnBlock {
  MpConv qkv, proj;
};

}  // namespace

struct VaeDecoder::Impl {
  ~Impl() {
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }
  bool LoadConv(const TensorFile& file, const std::string& name, int out,
                int in, int k, float gain, MpConv* conv, std::string* error);
  bool LoadRes(const TensorFile& file, const std::string& prefix, int in,
               int out, ResBlock* block, std::string* error);
  bool Conv(const MpConv& conv, const float* x, int rows, float* out,
            std::string* error);
  /// x [rows, in] -> x' [rows, out] (swaps x / spare).
  bool Res(const ResBlock& block, int rows, std::string* error);
  bool Attn(const AttnBlock& block, int rows, std::string* error);

  hipStream_t stream{nullptr};
  Blas blas;
  Precision precision{Precision::kF16};
  std::size_t weight_bytes{0};

  MpConv conv_in, conv_out, upsample;
  ResBlock mid1, mid2;
  AttnBlock mid_attn;
  std::vector<ResBlock> up2, up1, up0;  // processed in this order
  DeviceBuffer mean, stddev;

  DeviceBuffer x, spare, normed, h, h2, columns, qkv, q, k, v, attn;
  int width{0};  // channels of x
};

VaeDecoder::VaeDecoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
VaeDecoder::~VaeDecoder() = default;
std::size_t VaeDecoder::weight_bytes() const { return impl_->weight_bytes; }

bool VaeDecoder::Impl::LoadConv(const TensorFile& file, const std::string& name,
                                int out, int in, int k, float gain,
                                MpConv* conv, std::string* error) {
  std::vector<float> host;
  if (!file.Read(name, &host, error, static_cast<std::size_t>(out) * in * k))
    return false;
  NormalizeMpWeight(&host, out, gain);
  conv->in = in;
  conv->out = out;
  conv->k = k;
  if (!UploadMatrix(host, out, in * k, precision, &conv->w, error))
    return false;
  weight_bytes += conv->w.bytes();
  return true;
}

bool VaeDecoder::Impl::LoadRes(const TensorFile& file,
                               const std::string& prefix, int in, int out,
                               ResBlock* block, std::string* error) {
  if (!LoadConv(file, prefix + ".conv1.weight", out, in, 3, 1.0F,
                &block->conv1, error) ||
      !LoadConv(file, prefix + ".conv2.weight", out, out, 3, 1.0F,
                &block->conv2, error))
    return false;
  block->shortcut = in != out;
  return !block->shortcut ||
         LoadConv(file, prefix + ".nin_shortcut.weight", out, in, 1, 1.0F,
                  &block->nin, error);
}

std::unique_ptr<VaeDecoder> VaeDecoder::Load(const std::filesystem::path& path,
                                             Precision precision,
                                             std::string* error) {
  TensorFile file;
  if (!file.Open(path, error))
    return nullptr;
  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  m.precision = precision;
  if (!Check(hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking),
             "stream", error) ||
      !m.blas.Init(m.stream, error))
    return nullptr;
  std::vector<float> gain;
  if (!file.Read("decoder.learnable_gain", &gain, error, 1))
    return nullptr;
  const std::string d = "decoder.";
  if (!m.LoadConv(file, d + "conv_in.weight", 2048, kLatentDim, 3, 1.0F,
                  &m.conv_in, error) ||
      !m.LoadRes(file, d + "mid.block_1", 2048, 2048, &m.mid1, error) ||
      !m.LoadConv(file, d + "mid.attn_1.qkv.weight", 3 * 2048, 2048, 1, 1.0F,
                  &m.mid_attn.qkv, error) ||
      !m.LoadConv(file, d + "mid.attn_1.proj_out.weight", 2048, 2048, 1, 1.0F,
                  &m.mid_attn.proj, error) ||
      !m.LoadRes(file, d + "mid.block_2", 2048, 2048, &m.mid2, error) ||
      !m.LoadConv(file, d + "up.1.upsample.conv.weight", 1024, 1024, 3, 1.0F,
                  &m.upsample, error) ||
      !m.LoadConv(file, d + "conv_out.weight", kMels, 512, 3, gain[0] + 1.0F,
                  &m.conv_out, error))
    return nullptr;
  const auto level = [&](int index, int in, int out,
                         std::vector<ResBlock>* blocks) {
    blocks->resize(3);
    for (int b = 0; b < 3; ++b) {
      if (!m.LoadRes(file,
                     d + "up." + std::to_string(index) + ".block." +
                         std::to_string(b),
                     b == 0 ? in : out, out, &(*blocks)[b], error))
        return false;
    }
    return true;
  };
  if (!level(2, 2048, 2048, &m.up2) || !level(1, 2048, 1024, &m.up1) ||
      !level(0, 1024, 512, &m.up0))
    return nullptr;
  if (!UploadF32(std::vector<float>(kMelMean, kMelMean + kMels), &m.mean,
                 error) ||
      !UploadF32(std::vector<float>(kMelStd, kMelStd + kMels), &m.stddev,
                 error))
    return nullptr;
  return std::unique_ptr<VaeDecoder>(new VaeDecoder(std::move(impl)));
}

bool VaeDecoder::Impl::Conv(const MpConv& conv, const float* input, int rows,
                            float* out, std::string* error) {
  return RunConv1d(blas, columns, conv.w, nullptr, input, rows, conv.in,
                   conv.k, 1, conv.k / 2, out, error);
}

bool VaeDecoder::Impl::Res(const ResBlock& block, int rows,
                           std::string* error) {
  const std::size_t in_count = static_cast<std::size_t>(rows) * block.conv1.in;
  const std::size_t out_count =
      static_cast<std::size_t>(rows) * block.conv1.out;
  // x = pixel_norm(x); h = conv2(mp_silu(conv1(mp_silu(x))))
  LaunchPixelNorm(x.f32(), rows, block.conv1.in, normed.f32(), stream);
  if (!Check(hipMemcpyAsync(h.get(), normed.get(), in_count * 4,
                            hipMemcpyDeviceToDevice, stream),
             "copy", error))
    return false;
  LaunchSilu(h.f32(), in_count, kMpSiluScale, stream);
  if (!Conv(block.conv1, h.f32(), rows, h2.f32(), error))
    return false;
  LaunchSilu(h2.f32(), out_count, kMpSiluScale, stream);
  if (!Conv(block.conv2, h2.f32(), rows, h.f32(), error))
    return false;
  const float* skip = normed.f32();
  if (block.shortcut) {
    if (!Conv(block.nin, normed.f32(), rows, h2.f32(), error))
      return false;
    skip = h2.f32();
  }
  LaunchMpSum(skip, h.f32(), out_count, 0.3F, x.f32(), stream);
  width = block.conv1.out;
  return true;
}

bool VaeDecoder::Impl::Attn(const AttnBlock& block, int rows,
                            std::string* error) {
  const int c = block.proj.out;
  if (!Conv(block.qkv, x.f32(), rows, qkv.f32(), error))
    return false;
  LaunchVaeQkv(qkv.f32(), rows, c, q.f32(), k.f32(), v.f32(), stream);
  if (!blas.Attention(q.f32(), c, k.f32(), c, v.f32(), c, attn.f32(), c, rows,
                      rows, 1, c, 1.0F / std::sqrt(static_cast<float>(c)),
                      false, error) ||
      !Conv(block.proj, attn.f32(), rows, h.f32(), error))
    return false;
  LaunchMpSum(x.f32(), h.f32(), static_cast<std::size_t>(rows) * c, 0.3F,
              x.f32(), stream);
  return true;
}

bool VaeDecoder::Decode(const std::vector<float>& latent, int frames,
                        std::vector<float>* mel, std::string* error) {
  Impl& m = *impl_;
  if (frames <= 0 ||
      latent.size() != static_cast<std::size_t>(frames) * kLatentDim)
    return Fail(error, "VAE: latent must be [frames][40]");
  const int rows_out = 2 * frames;
  const std::size_t big = static_cast<std::size_t>(rows_out) * 2048 * 4;
  if (!m.x.Reserve(big, error) || !m.spare.Reserve(big, error) ||
      !m.normed.Reserve(big, error) || !m.h.Reserve(big, error) ||
      !m.h2.Reserve(big, error) ||
      !m.qkv.Reserve(static_cast<std::size_t>(frames) * 3 * 2048 * 4, error) ||
      !m.q.Reserve(static_cast<std::size_t>(frames) * 2048 * 4, error) ||
      !m.k.Reserve(static_cast<std::size_t>(frames) * 2048 * 4, error) ||
      !m.v.Reserve(static_cast<std::size_t>(frames) * 2048 * 4, error) ||
      !m.attn.Reserve(static_cast<std::size_t>(frames) * 2048 * 4, error) ||
      !Check(hipMemcpyAsync(m.spare.get(), latent.data(), latent.size() * 4,
                            hipMemcpyHostToDevice, m.stream),
             "latent upload", error))
    return false;
  int rows = frames;
  const auto clamp = [&] {
    LaunchClamp(m.x.f32(), static_cast<std::size_t>(rows) * m.width,
                -kClipAct, kClipAct, m.stream);
  };
  // conv_in, middle.
  if (!m.Conv(m.conv_in, m.spare.f32(), rows, m.x.f32(), error))
    return false;
  m.width = 2048;
  if (!m.Res(m.mid1, rows, error) || !m.Attn(m.mid_attn, rows, error) ||
      !m.Res(m.mid2, rows, error))
    return false;
  clamp();
  // Levels 2, 1 (then 2x upsampling), 0.
  for (const ResBlock& block : m.up2) {
    if (!m.Res(block, rows, error))
      return false;
    clamp();
  }
  for (const ResBlock& block : m.up1) {
    if (!m.Res(block, rows, error))
      return false;
    clamp();
  }
  LaunchUpsampleNearest(m.x.f32(), rows, m.width, 2, m.spare.f32(), m.stream);
  rows *= 2;
  if (!m.Conv(m.upsample, m.spare.f32(), rows, m.x.f32(), error))
    return false;
  for (const ResBlock& block : m.up0) {
    if (!m.Res(block, rows, error))
      return false;
    clamp();
  }
  // mp_silu, conv_out (gain folded), unnormalize.
  const std::size_t count = static_cast<std::size_t>(rows) * m.width;
  LaunchSilu(m.x.f32(), count, kMpSiluScale, m.stream);
  if (!m.Conv(m.conv_out, m.x.f32(), rows, m.spare.f32(), error))
    return false;
  LaunchColumnAffine(m.spare.f32(), rows, kMels, m.stddev.f32(), m.mean.f32(),
                     m.stream);
  mel->resize(static_cast<std::size_t>(rows) * kMels);
  return Check(hipMemcpyAsync(mel->data(), m.spare.get(), mel->size() * 4,
                              hipMemcpyDeviceToHost, m.stream),
               "mel readback", error) &&
         Check(hipStreamSynchronize(m.stream), "VAE decode", error);
}

// ---- BigVGAN v2 ------------------------------------------------------------------

namespace {

constexpr int kUpsampleRates[6] = {8, 4, 2, 2, 2, 2};
constexpr int kUpsampleKernels[6] = {16, 8, 4, 4, 4, 4};
constexpr int kResKernels[3] = {3, 7, 11};
constexpr int kDilations[3] = {1, 3, 5};
constexpr int kInitialChannels = 1536;

struct Snake {
  DeviceBuffer alpha;     // exp(alpha)
  DeviceBuffer inv_beta;  // 1 / (exp(beta) + 1e-9)
};

struct Conv1d {
  Matrix w;  // [out, in * k] (transposed conv: [out * k, in])
  DeviceBuffer b;
  int in{0};
  int out{0};
  int k{1};
};

struct AmpBlock {
  int kernel{3};
  Conv1d convs1[3], convs2[3];
  Snake acts[6];
};

}  // namespace

struct BigVgan::Impl {
  ~Impl() {
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }
  bool LoadConv(const TensorFile& file, const std::string& name, int out,
                int in, int k, bool bias, Conv1d* conv, std::string* error);
  bool LoadSnake(const TensorFile& file, const std::string& prefix,
                 int channels, Snake* snake, std::string* error);

  hipStream_t stream{nullptr};
  Blas blas;
  Precision precision{Precision::kF16};
  std::size_t weight_bytes{0};

  Conv1d conv_pre, conv_post;
  Conv1d ups[6];
  std::vector<AmpBlock> blocks;  // 18
  Snake post_act;
  DeviceBuffer filter;

  DeviceBuffer x, acc, r, a, c, columns;
};

BigVgan::BigVgan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
BigVgan::~BigVgan() = default;
std::size_t BigVgan::weight_bytes() const { return impl_->weight_bytes; }

bool BigVgan::Impl::LoadConv(const TensorFile& file, const std::string& name,
                             int out, int in, int k, bool bias, Conv1d* conv,
                             std::string* error) {
  std::vector<float> host;
  if (!file.Read(name + ".weight", &host, error,
                 static_cast<std::size_t>(out) * in * k))
    return false;
  conv->in = in;
  conv->out = out;
  conv->k = k;
  if (!UploadMatrix(host, out, in * k, precision, &conv->w, error))
    return false;
  weight_bytes += conv->w.bytes();
  return !bias || LoadF32(file, name + ".bias", out, &conv->b, error);
}

bool BigVgan::Impl::LoadSnake(const TensorFile& file,
                              const std::string& prefix, int channels,
                              Snake* snake, std::string* error) {
  std::vector<float> alpha, beta;
  if (!file.Read(prefix + ".alpha", &alpha, error, channels) ||
      !file.Read(prefix + ".beta", &beta, error, channels))
    return false;
  // snake_logscale: alpha = exp(alpha), beta = exp(beta).
  for (int i = 0; i < channels; ++i) {
    alpha[i] = std::exp(alpha[i]);
    beta[i] = 1.0F / (std::exp(beta[i]) + 1e-9F);
  }
  return UploadF32(alpha, &snake->alpha, error) &&
         UploadF32(beta, &snake->inv_beta, error);
}

std::unique_ptr<BigVgan> BigVgan::Load(const std::filesystem::path& path,
                                       Precision precision,
                                       std::string* error) {
  TensorFile file;
  if (!file.Open(path, error))
    return nullptr;
  if (!file.Has("conv_pre.weight") || file.Has("conv_pre.weight_g") ||
      file.Has("conv_pre.weight_v")) {
    Fail(error, path.string() +
                    ": expected a folded BigVGAN generator (run "
                    "tools/models/mmaudio/convert_bigvgan.py)");
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  m.precision = precision;
  if (!Check(hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking),
             "stream", error) ||
      !m.blas.Init(m.stream, error))
    return nullptr;
  if (!m.LoadConv(file, "conv_pre", kInitialChannels, kMels, 7, true,
                  &m.conv_pre, error))
    return nullptr;
  int channels = kInitialChannels;
  for (int i = 0; i < 6; ++i) {
    const int out = channels / 2;
    const int k = kUpsampleKernels[i];
    // ConvTranspose1d weight [in, out, k] -> [out * k, in].
    std::vector<float> host;
    const std::string name = "ups." + std::to_string(i) + ".0";
    if (!file.Read(name + ".weight", &host, error,
                   static_cast<std::size_t>(channels) * out * k))
      return nullptr;
    std::vector<float> transposed(host.size());
    for (int ci = 0; ci < channels; ++ci)
      for (int co = 0; co < out; ++co)
        for (int kk = 0; kk < k; ++kk)
          transposed[(static_cast<std::size_t>(co) * k + kk) * channels + ci] =
              host[(static_cast<std::size_t>(ci) * out + co) * k + kk];
    Conv1d& up = m.ups[i];
    up.in = channels;
    up.out = out;
    up.k = k;
    if (!UploadMatrix(transposed, out * k, channels, precision, &up.w,
                      error) ||
        !LoadF32(file, name + ".bias", out, &up.b, error))
      return nullptr;
    m.weight_bytes += up.w.bytes();
    channels = out;
    for (int j = 0; j < 3; ++j) {
      AmpBlock block;
      block.kernel = kResKernels[j];
      const std::string p =
          "resblocks." + std::to_string(i * 3 + j) + ".";
      for (int d = 0; d < 3; ++d) {
        if (!m.LoadConv(file, p + "convs1." + std::to_string(d), channels,
                        channels, block.kernel, true, &block.convs1[d],
                        error) ||
            !m.LoadConv(file, p + "convs2." + std::to_string(d), channels,
                        channels, block.kernel, true, &block.convs2[d], error))
          return nullptr;
      }
      for (int a = 0; a < 6; ++a)
        if (!m.LoadSnake(file, p + "activations." + std::to_string(a) + ".act",
                         channels, &block.acts[a], error))
          return nullptr;
      m.blocks.push_back(std::move(block));
    }
  }
  if (!m.LoadSnake(file, "activation_post.act", channels, &m.post_act,
                   error) ||
      !m.LoadConv(file, "conv_post", 1, channels, 7, false, &m.conv_post,
                  error) ||
      !UploadF32(KaiserSincFilter(0.25, 0.3, 12), &m.filter, error) ||
      !Check(hipStreamSynchronize(m.stream), "load", error))
    return nullptr;
  return std::unique_ptr<BigVgan>(new BigVgan(std::move(impl)));
}

bool BigVgan::Vocode(const std::vector<float>& mel, int frames,
                     std::vector<float>* audio, std::string* error) {
  Impl& m = *impl_;
  if (frames <= 0 || mel.size() != static_cast<std::size_t>(frames) * kMels)
    return Fail(error, "BigVGAN: mel must be [frames][128]");
  // Largest activation: rows * channels is frames * 6144 after the first
  // upsampling and frames * 12288 after each later one.
  const std::size_t largest = static_cast<std::size_t>(frames) * 12288 * 4;
  if (!m.x.Reserve(largest, error) || !m.acc.Reserve(largest, error) ||
      !m.r.Reserve(largest, error) || !m.a.Reserve(largest, error) ||
      !m.c.Reserve(std::max<std::size_t>(largest, static_cast<std::size_t>(
                                                      frames) *
                                                      kInitialChannels * 4),
                   error) ||
      !Check(hipMemcpyAsync(m.a.get(), mel.data(), mel.size() * 4,
                            hipMemcpyHostToDevice, m.stream),
             "mel upload", error))
    return false;
  int rows = frames;
  if (!RunConv1d(m.blas, m.columns, m.conv_pre.w, m.conv_pre.b.f32(),
                 m.a.f32(), rows, kMels, 7, 1, 3, m.x.f32(), error))
    return false;
  int channels = kInitialChannels;
  for (int i = 0; i < 6; ++i) {
    const Conv1d& up = m.ups[i];
    const int rate = kUpsampleRates[i];
    // Transposed conv: y = x W' [rows, out * k], then overlap-add.
    if (!m.columns.Reserve(static_cast<std::size_t>(rows) * up.out * up.k * 4,
                           error) ||
        !m.blas.Linear(up.w, nullptr, m.x.f32(), rows, m.columns.f32(), error))
      return false;
    const int rows_out = rows * rate;
    LaunchCol2ImTransposed(m.columns.f32(), rows, up.out, up.k, rate,
                           (up.k - rate) / 2, up.b.f32(), rows_out, m.x.f32(),
                           m.stream);
    rows = rows_out;
    channels = up.out;
    const std::size_t count = static_cast<std::size_t>(rows) * channels;
    if (!Check(hipMemsetAsync(m.acc.get(), 0, count * 4, m.stream), "clear",
               error))
      return false;
    for (int j = 0; j < 3; ++j) {
      const AmpBlock& block = m.blocks[i * 3 + j];
      if (!Check(hipMemcpyAsync(m.r.get(), m.x.get(), count * 4,
                                hipMemcpyDeviceToDevice, m.stream),
                 "copy", error))
        return false;
      for (int p = 0; p < 3; ++p) {
        const int d = kDilations[p];
        const int k = block.kernel;
        // xt = c2(a2(c1(a1(r)))); r += xt
        LaunchSnakeAntiAlias(m.r.f32(), rows, channels,
                             block.acts[2 * p].alpha.f32(),
                             block.acts[2 * p].inv_beta.f32(), m.filter.f32(),
                             m.a.f32(), m.stream);
        if (!RunConv1d(m.blas, m.columns, block.convs1[p].w,
                       block.convs1[p].b.f32(), m.a.f32(), rows, channels, k,
                       d, d * (k - 1) / 2, m.c.f32(), error))
          return false;
        LaunchSnakeAntiAlias(m.c.f32(), rows, channels,
                             block.acts[2 * p + 1].alpha.f32(),
                             block.acts[2 * p + 1].inv_beta.f32(),
                             m.filter.f32(), m.a.f32(), m.stream);
        if (!RunConv1d(m.blas, m.columns, block.convs2[p].w,
                       block.convs2[p].b.f32(), m.a.f32(), rows, channels, k,
                       1, (k - 1) / 2, m.r.f32(), error, /*accumulate=*/true))
          return false;
      }
      LaunchAdd(m.acc.f32(), m.r.f32(), count, m.stream);
    }
    // x = (sum of the three AMP blocks) / 3
    if (!Check(hipMemcpyAsync(m.x.get(), m.acc.get(), count * 4,
                              hipMemcpyDeviceToDevice, m.stream),
               "copy", error))
      return false;
    LaunchAffine(m.x.f32(), count, 1.0F / 3.0F, 0.0F, m.stream);
  }
  LaunchSnakeAntiAlias(m.x.f32(), rows, channels, m.post_act.alpha.f32(),
                       m.post_act.inv_beta.f32(), m.filter.f32(), m.a.f32(),
                       m.stream);
  if (!RunConv1d(m.blas, m.columns, m.conv_post.w, nullptr, m.a.f32(), rows,
                 channels, 7, 1, 3, m.c.f32(), error))
    return false;
  // use_tanh_at_final = false: clamp to [-1, 1].
  LaunchClamp(m.c.f32(), rows, -1.0F, 1.0F, m.stream);
  audio->resize(rows);
  return Check(hipMemcpyAsync(audio->data(), m.c.get(),
                              static_cast<std::size_t>(rows) * 4,
                              hipMemcpyDeviceToHost, m.stream),
               "audio readback", error) &&
         Check(hipStreamSynchronize(m.stream), "vocoder", error);
}

}  // namespace gufo::models::mmaudio
