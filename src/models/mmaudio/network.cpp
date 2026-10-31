// MMAudio for gufo: the flow-matching multimodal transformer (see
// network.hpp).
// Written with AI assistance (Claude Code).
#include "src/models/mmaudio/network.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "src/models/mmaudio/kernels.hpp"

namespace gufo::models::mmaudio {
namespace {

constexpr int D = FlowNetwork::kHidden;
constexpr int kMlpHidden = 2560;  // 256 * ceil(int(2 * 4 * 896 / 3) / 256)
constexpr int kJointBlocks = 7;
constexpr int kFusedBlocks = 14;
constexpr float kLnEps = 1e-5F;

/// conv1d (k > 1, "same" padding, as ChannelLastConv1d) or Linear (k == 1).
struct Conv {
  Matrix w;  // [out, in * k]
  DeviceBuffer b;  // FP32 bias or empty
  int in{0};
  int out{0};
  int k{1};
};

/// ConvMLP / MLP: w2(silu(w1 x) * w3 x), w1 and w3 stacked.
struct GatedMlp {
  Matrix w13;  // [2 * 2560, in * k]
  Matrix w2;   // [out, 2560 * k]
  int k{1};
};

struct SingleBlock {
  bool pre_only{false};
  Matrix ada;  // [6 * D or 2 * D, D]
  DeviceBuffer ada_b;
  Matrix qkv;  // [3 * D, D]
  DeviceBuffer qkv_b, q_norm, k_norm;
  Conv linear1;
  GatedMlp ffn;
};

struct JointBlock {
  SingleBlock latent, clip, text;
};

}  // namespace

struct FlowNetwork::Impl {
  ~Impl() {
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }

  bool LoadConv(const TensorFile& file, const std::string& name, int out,
                int in, int k, bool bias, Conv* conv, std::string* error);
  bool LoadMlp(const TensorFile& file, const std::string& prefix, int in,
               int k, GatedMlp* mlp, std::string* error);
  bool LoadBlock(const TensorFile& file, const std::string& prefix,
                 bool pre_only, int kernel, SingleBlock* block,
                 std::string* error);

  /// out[rows, conv.out] (+)= conv(x[rows, conv.in]).
  bool RunConv(const Conv& conv, const float* x, int rows, float* out,
               std::string* error, bool accumulate = false);
  bool RunMlp(const GatedMlp& mlp, const float* x, int rows, int out_width,
              float* out, std::string* error);
  /// Stream input projections: Linear / conv, SiLU, ConvMLP / MLP.
  bool RunInputProjection(const Conv& first, const GatedMlp& mlp,
                          const float* x, int rows, float* out,
                          std::string* error);
  /// LN + modulate + qkv + split (with RMSNorm / RoPE) into the joint
  /// q/k/v rows starting at `row`.
  bool PreAttention(const SingleBlock& block, const float* x, int rows,
                    const float* mod, int mod_stride, const float* rope,
                    int row, std::string* error);
  bool PostAttention(const SingleBlock& block, float* x, int rows,
                     const float* attn, const float* mod, int mod_stride,
                     std::string* error);
  bool EnsureLengths(const SequenceLengths& lengths, std::string* error);

  hipStream_t stream{nullptr};
  Blas blas;
  std::size_t weight_bytes{0};

  Conv audio_in, clip_in, sync_in, text_in;
  GatedMlp audio_mlp, clip_mlp, sync_mlp, text_mlp;
  Conv clip_cond, text_cond;
  GatedMlp global_mlp;  // MLP(896, 3584) on one vector
  DeviceBuffer sync_pos;          // [8, 768]
  Conv t_mlp0, t_mlp2;
  std::vector<JointBlock> joint;
  std::vector<SingleBlock> fused;
  Matrix final_ada;
  DeviceBuffer final_ada_b;
  Conv final_conv;
  std::vector<float> latent_mean, latent_std;
  std::vector<float> empty_clip, empty_sync, empty_text;

  // Per length set.
  SequenceLengths lengths;
  DeviceBuffer latent_rot, clip_rot;

  // Scratch.
  DeviceBuffer x, clip_x, text_x, normed, qkv, q, k, v, attn, columns,
      wide, gated, tmp, mod_latent, mod_clip, mod_text, silu_ext, extended,
      vec_a, vec_b, vec_c, t_freq, cond_in, flow_a, flow_b, state;
};

FlowNetwork::FlowNetwork(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
FlowNetwork::~FlowNetwork() = default;

std::size_t FlowNetwork::weight_bytes() const { return impl_->weight_bytes; }
hipStream_t FlowNetwork::stream() const { return impl_->stream; }

SequenceLengths LengthsFor(double duration) {
  SequenceLengths lengths;
  lengths.latent =
      static_cast<int>(std::ceil(duration * 44100.0 / 512.0 / 2.0));
  lengths.clip = static_cast<int>(duration * 8.0);
  const double frames = duration * 25.0;
  const double segments = std::floor((frames - 16.0) / 8.0) + 1.0;
  lengths.sync = segments > 0 ? static_cast<int>(segments * 16.0 / 2.0) : 0;
  return lengths;
}

std::vector<float> RopeTable(int length, int head_dim, float freq_scaling) {
  const int half = head_dim / 2;
  std::vector<float> table(static_cast<std::size_t>(length) * half * 2);
  for (int i = 0; i < half; ++i) {
    // torch: 1 / theta^(arange(0, dim, 2) / dim) in FP32, then * scaling.
    const float exponent =
        static_cast<float>(2 * i) / static_cast<float>(head_dim);
    float freq = 1.0F / std::pow(10000.0F, exponent);
    freq *= freq_scaling;
    for (int p = 0; p < length; ++p) {
      const float angle = static_cast<float>(p) * freq;
      const std::size_t at = (static_cast<std::size_t>(p) * half + i) * 2;
      table[at] = std::cos(angle);
      table[at + 1] = std::sin(angle);
    }
  }
  return table;
}

std::vector<float> TimestepFrequencies(float t) {
  constexpr int kHalf = D / 2;
  std::vector<float> out(D);
  for (int i = 0; i < kHalf; ++i) {
    const float exponent = static_cast<float>(2 * i) / static_cast<float>(D);
    // freq_scale = 10000 / max_period (1).
    const float freq = 10000.0F * (1.0F / std::pow(10000.0F, exponent));
    const float arg = t * freq;
    out[i] = std::cos(arg);
    out[kHalf + i] = std::sin(arg);
  }
  return out;
}

bool FlowNetwork::Impl::LoadConv(const TensorFile& file,
                                 const std::string& name, int out, int in,
                                 int k, bool bias, Conv* conv,
                                 std::string* error) {
  conv->in = in;
  conv->out = out;
  conv->k = k;
  if (!LoadMatrix(file, name + ".weight", out, in * k, DefaultPrecision(),
                  &conv->w, error))
    return false;
  weight_bytes += conv->w.bytes();
  return !bias || LoadF32(file, name + ".bias", out, &conv->b, error);
}

bool FlowNetwork::Impl::LoadMlp(const TensorFile& file,
                                const std::string& prefix, int in, int k,
                                GatedMlp* mlp, std::string* error) {
  std::vector<float> w1, w3, w2;
  const std::size_t inner = static_cast<std::size_t>(kMlpHidden) * in * k;
  if (!file.Read(prefix + ".w1.weight", &w1, error, inner) ||
      !file.Read(prefix + ".w3.weight", &w3, error, inner) ||
      !file.Read(prefix + ".w2.weight", &w2, error,
                 static_cast<std::size_t>(in) * kMlpHidden * k))
    return false;
  w1.insert(w1.end(), w3.begin(), w3.end());
  mlp->k = k;
  if (!UploadMatrix(w1, 2 * kMlpHidden, in * k, DefaultPrecision(), &mlp->w13,
                    error) ||
      !UploadMatrix(w2, in, kMlpHidden * k, DefaultPrecision(), &mlp->w2,
                    error))
    return false;
  weight_bytes += mlp->w13.bytes() + mlp->w2.bytes();
  return true;
}

bool FlowNetwork::Impl::LoadBlock(const TensorFile& file,
                                  const std::string& prefix, bool pre_only,
                                  int kernel, SingleBlock* block,
                                  std::string* error) {
  block->pre_only = pre_only;
  const int chunks = pre_only ? 2 : 6;
  if (!LoadMatrix(file, prefix + ".adaLN_modulation.1.weight", chunks * D, D,
                  DefaultPrecision(), &block->ada, error) ||
      !LoadF32(file, prefix + ".adaLN_modulation.1.bias", chunks * D,
               &block->ada_b, error) ||
      !LoadMatrix(file, prefix + ".attn.qkv.weight", 3 * D, D,
                  DefaultPrecision(), &block->qkv, error) ||
      !LoadF32(file, prefix + ".attn.qkv.bias", 3 * D, &block->qkv_b, error) ||
      !LoadF32(file, prefix + ".attn.q_norm.weight", kHeadDim, &block->q_norm,
               error) ||
      !LoadF32(file, prefix + ".attn.k_norm.weight", kHeadDim, &block->k_norm,
               error))
    return false;
  weight_bytes += block->ada.bytes() + block->qkv.bytes();
  if (pre_only)
    return true;
  return LoadConv(file, prefix + ".linear1", D, D, kernel, true,
                  &block->linear1, error) &&
         LoadMlp(file, prefix + ".ffn", D, kernel, &block->ffn, error);
}

std::unique_ptr<FlowNetwork> FlowNetwork::Load(
    const std::filesystem::path& path, Precision precision,
    std::string* error) {
  (void)precision;  // the loaders use DefaultPrecision(); kept for symmetry
  TensorFile file;
  if (!file.Open(path, error))
    return nullptr;
  const TensorFile::Info* probe = file.Find("audio_input_proj.0.bias");
  const TensorFile::Info* t_probe = file.Find("t_embed.mlp.0.weight");
  if (probe == nullptr || probe->count() != static_cast<std::size_t>(D) ||
      t_probe == nullptr || t_probe->shape.size() != 2 ||
      t_probe->shape[1] != D) {
    Fail(error,
         "only mmaudio_large_44k_v2 is supported (hidden 896, v2 timestep "
         "embedding)");
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  if (!Check(hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking),
             "stream", error) ||
      !m.blas.Init(m.stream, error))
    return nullptr;
  if (!m.LoadConv(file, "audio_input_proj.0", D, kLatentDim, 7, true,
                  &m.audio_in, error) ||
      !m.LoadMlp(file, "audio_input_proj.2", D, 7, &m.audio_mlp, error) ||
      !m.LoadConv(file, "clip_input_proj.0", D, kClipDim, 1, true, &m.clip_in,
                  error) ||
      !m.LoadMlp(file, "clip_input_proj.2", D, 3, &m.clip_mlp, error) ||
      !m.LoadConv(file, "sync_input_proj.0", D, kSyncDim, 7, true, &m.sync_in,
                  error) ||
      !m.LoadMlp(file, "sync_input_proj.2", D, 3, &m.sync_mlp, error) ||
      !m.LoadConv(file, "text_input_proj.0", D, kTextDim, 1, true, &m.text_in,
                  error) ||
      !m.LoadMlp(file, "text_input_proj.2", D, 1, &m.text_mlp, error) ||
      !m.LoadConv(file, "clip_cond_proj", D, D, 1, true, &m.clip_cond,
                  error) ||
      !m.LoadConv(file, "text_cond_proj", D, D, 1, true, &m.text_cond,
                  error) ||
      !LoadF32(file, "sync_pos_emb", 8 * kSyncDim, &m.sync_pos, error) ||
      !m.LoadConv(file, "t_embed.mlp.0", D, D, 1, true, &m.t_mlp0, error) ||
      !m.LoadConv(file, "t_embed.mlp.2", D, D, 1, true, &m.t_mlp2, error))
    return nullptr;
  // global_cond_mlp is an MLP (linear, k = 1) like text_input_proj.2.
  if (!m.LoadMlp(file, "global_cond_mlp", D, 1, &m.global_mlp, error))
    return nullptr;
  m.joint.resize(kJointBlocks);
  for (int i = 0; i < kJointBlocks; ++i) {
    const std::string p = "joint_blocks." + std::to_string(i);
    const bool pre_only = i == kJointBlocks - 1;
    if (!m.LoadBlock(file, p + ".latent_block", false, 3, &m.joint[i].latent,
                     error) ||
        !m.LoadBlock(file, p + ".clip_block", pre_only, 3, &m.joint[i].clip,
                     error) ||
        !m.LoadBlock(file, p + ".text_block", pre_only, 1, &m.joint[i].text,
                     error))
      return nullptr;
  }
  m.fused.resize(kFusedBlocks);
  for (int i = 0; i < kFusedBlocks; ++i)
    if (!m.LoadBlock(file, "fused_blocks." + std::to_string(i), false, 3,
                     &m.fused[i], error))
      return nullptr;
  if (!LoadMatrix(file, "final_layer.adaLN_modulation.1.weight", 2 * D, D,
                  DefaultPrecision(), &m.final_ada, error) ||
      !LoadF32(file, "final_layer.adaLN_modulation.1.bias", 2 * D,
               &m.final_ada_b, error) ||
      !m.LoadConv(file, "final_layer.conv", kLatentDim, D, 7, true,
                  &m.final_conv, error) ||
      !file.Read("latent_mean", &m.latent_mean, error, kLatentDim) ||
      !file.Read("latent_std", &m.latent_std, error, kLatentDim) ||
      !file.Read("empty_clip_feat", &m.empty_clip, error, kClipDim) ||
      !file.Read("empty_sync_feat", &m.empty_sync, error, kSyncDim) ||
      !file.Read("empty_string_feat", &m.empty_text, error,
                 static_cast<std::size_t>(kTextTokens) * kTextDim))
    return nullptr;
  if (!m.vec_a.Allocate(4 * D * 4, error) ||
      !m.vec_b.Allocate(4 * D * 4, error) ||
      !m.vec_c.Allocate(4 * D * 4, error) ||
      !m.t_freq.Allocate(D * 4, error) ||
      !m.mod_clip.Allocate(6 * D * 4, error) ||
      !m.mod_text.Allocate(6 * D * 4, error) ||
      !Check(hipStreamSynchronize(m.stream), "load", error))
    return nullptr;
  return std::unique_ptr<FlowNetwork>(new FlowNetwork(std::move(impl)));
}

bool FlowNetwork::Impl::RunConv(const Conv& conv, const float* input, int rows,
                                float* out, std::string* error,
                                bool accumulate) {
  const float* source = input;
  if (conv.k > 1) {
    if (!columns.Reserve(static_cast<std::size_t>(rows) * conv.in * conv.k * 4,
                         error))
      return false;
    LaunchIm2Col(input, rows, conv.in, conv.k, 1, conv.k / 2, rows,
                 columns.f32(), stream);
    source = columns.f32();
  }
  return blas.Linear(conv.w, conv.b.bytes() > 0 ? conv.b.f32() : nullptr,
                     source, rows, out, error, accumulate);
}

bool FlowNetwork::Impl::RunMlp(const GatedMlp& mlp, const float* input,
                               int rows, int out_width, float* out,
                               std::string* error) {
  const int in_width = mlp.w13.columns / mlp.k;
  const float* source = input;
  if (!wide.Reserve(static_cast<std::size_t>(rows) * 2 * kMlpHidden * 4,
                    error) ||
      !gated.Reserve(static_cast<std::size_t>(rows) * kMlpHidden * 4, error))
    return false;
  if (mlp.k > 1) {
    if (!columns.Reserve(static_cast<std::size_t>(rows) * kMlpHidden * mlp.k *
                             4,
                         error))
      return false;
    LaunchIm2Col(input, rows, in_width, mlp.k, 1, mlp.k / 2, rows,
                 columns.f32(), stream);
    source = columns.f32();
  }
  if (!blas.Linear(mlp.w13, nullptr, source, rows, wide.f32(), error))
    return false;
  LaunchSwiGlu(wide.f32(), rows, kMlpHidden, gated.f32(), stream);
  source = gated.f32();
  if (mlp.k > 1) {
    LaunchIm2Col(gated.f32(), rows, kMlpHidden, mlp.k, 1, mlp.k / 2, rows,
                 columns.f32(), stream);
    source = columns.f32();
  }
  (void)out_width;
  return blas.Linear(mlp.w2, nullptr, source, rows, out, error);
}

bool FlowNetwork::Impl::RunInputProjection(const Conv& first,
                                           const GatedMlp& mlp,
                                           const float* input, int rows,
                                           float* out, std::string* error) {
  if (!tmp.Reserve(static_cast<std::size_t>(rows) * D * 4, error) ||
      !RunConv(first, input, rows, tmp.f32(), error))
    return false;
  LaunchSilu(tmp.f32(), static_cast<std::size_t>(rows) * D, 1.0F, stream);
  return RunMlp(mlp, tmp.f32(), rows, D, out, error);
}

bool FlowNetwork::Impl::EnsureLengths(const SequenceLengths& wanted,
                                      std::string* error) {
  if (wanted.latent == lengths.latent && wanted.clip == lengths.clip &&
      wanted.sync == lengths.sync && latent_rot.bytes() > 0)
    return true;
  const std::vector<float> latent_table =
      RopeTable(wanted.latent, kHeadDim, 1.0F);
  const float clip_scaling = static_cast<float>(wanted.latent) /
                             static_cast<float>(std::max(wanted.clip, 1));
  const std::vector<float> clip_table =
      RopeTable(std::max(wanted.clip, 1), kHeadDim, clip_scaling);
  if (!UploadF32(latent_table, &latent_rot, error) ||
      !UploadF32(clip_table, &clip_rot, error))
    return false;
  const std::size_t total = static_cast<std::size_t>(wanted.latent) +
                            wanted.clip + kTextTokens;
  const std::size_t n = wanted.latent;
  if (!x.Reserve(n * D * 4, error) ||
      !clip_x.Reserve(static_cast<std::size_t>(std::max(wanted.clip, 1)) * D *
                          4,
                      error) ||
      !text_x.Reserve(static_cast<std::size_t>(kTextTokens) * D * 4, error) ||
      !normed.Reserve(n * D * 4 + total * D * 4, error) ||
      !qkv.Reserve(total * 3 * D * 4, error) ||
      !q.Reserve(total * D * 4, error) || !k.Reserve(total * D * 4, error) ||
      !v.Reserve(total * D * 4, error) ||
      !attn.Reserve(total * D * 4, error) ||
      !tmp.Reserve(std::max<std::size_t>(n, 8 * 64) * D * 4, error) ||
      !mod_latent.Reserve(n * 6 * D * 4, error) ||
      !silu_ext.Reserve(n * D * 4, error) ||
      !extended.Reserve(n * D * 4, error) ||
      !flow_a.Reserve(n * kLatentDim * 4, error) ||
      !flow_b.Reserve(n * kLatentDim * 4, error) ||
      !state.Reserve(n * kLatentDim * 4, error))
    return false;
  lengths = wanted;
  return true;
}

bool FlowNetwork::Prepare(const SequenceLengths& lengths,
                          const std::vector<float>* clip,
                          const std::vector<float>* sync,
                          const std::vector<float>* text, Conditions* out,
                          std::string* error) {
  Impl& m = *impl_;
  if (lengths.latent <= 0 || lengths.clip <= 0 || lengths.sync <= 0 ||
      lengths.sync % 8 != 0)
    return Fail(error, "invalid MMAudio sequence lengths");
  if ((clip != nullptr &&
       clip->size() != static_cast<std::size_t>(lengths.clip) * kClipDim) ||
      (sync != nullptr &&
       sync->size() != static_cast<std::size_t>(lengths.sync) * kSyncDim) ||
      (text != nullptr &&
       text->size() != static_cast<std::size_t>(kTextTokens) * kTextDim))
    return Fail(error, "MMAudio condition features have the wrong size");
  if (!m.EnsureLengths(lengths, error))
    return false;
  out->lengths = lengths;
  const auto upload = [&](const std::vector<float>& host, std::string* err) {
    return m.cond_in.Reserve(host.size() * 4, err) &&
           Check(hipMemcpyAsync(m.cond_in.get(), host.data(), host.size() * 4,
                                hipMemcpyHostToDevice, m.stream),
                 "condition upload", err);
  };
  const auto repeated = [](const std::vector<float>& row, int rows) {
    std::vector<float> all;
    all.reserve(row.size() * rows);
    for (int r = 0; r < rows; ++r)
      all.insert(all.end(), row.begin(), row.end());
    return all;
  };
  // CLIP stream.
  if (!out->clip.Reserve(static_cast<std::size_t>(lengths.clip) * D * 4,
                         error) ||
      !upload(clip != nullptr ? *clip : repeated(m.empty_clip, lengths.clip),
              error) ||
      !m.RunInputProjection(m.clip_in, m.clip_mlp, m.cond_in.f32(),
                            lengths.clip, out->clip.f32(), error))
    return false;
  // Synchformer stream: + sync_pos_emb per segment position, projection,
  // nearest-exact upsampling to the latent length.
  {
    if (!upload(sync != nullptr ? *sync : repeated(m.empty_sync, lengths.sync),
                error) ||
        !m.normed.Reserve(static_cast<std::size_t>(lengths.sync) * kSyncDim *
                              4,
                          error))
      return false;
    LaunchRepeatRows(m.sync_pos.f32(), 8, lengths.sync, kSyncDim,
                     m.normed.f32(), m.stream);
    LaunchAdd(m.cond_in.f32(), m.normed.f32(),
              static_cast<std::size_t>(lengths.sync) * kSyncDim, m.stream);
    DeviceBuffer projected;
    if (!projected.Allocate(static_cast<std::size_t>(lengths.sync) * D * 4,
                            error) ||
        !m.RunInputProjection(m.sync_in, m.sync_mlp, m.cond_in.f32(),
                              lengths.sync, projected.f32(), error) ||
        !out->sync.Reserve(static_cast<std::size_t>(lengths.latent) * D * 4,
                           error))
      return false;
    LaunchInterpolateNearestExact(projected.f32(), lengths.sync, D,
                                  lengths.latent, out->sync.f32(), m.stream);
    if (!Check(hipStreamSynchronize(m.stream), "sync projection", error))
      return false;
  }
  // Text stream.
  if (!out->text.Reserve(static_cast<std::size_t>(kTextTokens) * D * 4,
                         error) ||
      !upload(text != nullptr ? *text : m.empty_text, error) ||
      !m.RunInputProjection(m.text_in, m.text_mlp, m.cond_in.f32(),
                            kTextTokens, out->text.f32(), error))
    return false;
  // global = global_cond_mlp(clip_cond_proj(mean clip) +
  //                          text_cond_proj(mean text)).
  LaunchColumnMean(out->clip.f32(), lengths.clip, D, m.vec_a.f32(), m.stream);
  if (!m.blas.Linear(m.clip_cond.w, m.clip_cond.b.f32(), m.vec_a.f32(), 1,
                     m.vec_b.f32(), error))
    return false;
  LaunchColumnMean(out->text.f32(), kTextTokens, D, m.vec_a.f32(), m.stream);
  if (!m.blas.Linear(m.text_cond.w, m.text_cond.b.f32(), m.vec_a.f32(), 1,
                     m.vec_b.f32(), error, /*accumulate=*/true) ||
      !out->global.Reserve(D * 4, error) ||
      !m.RunMlp(m.global_mlp, m.vec_b.f32(), 1, D, out->global.f32(), error))
    return false;
  return Check(hipStreamSynchronize(m.stream), "conditions", error);
}

bool FlowNetwork::Impl::PreAttention(const SingleBlock& block, const float* in,
                                     int rows, const float* mod,
                                     int mod_stride, const float* rope,
                                     int row, std::string* error) {
  // modulate(norm1(x), shift_msa, scale_msa) -> qkv -> heads.
  LaunchLayerNorm(in, rows, D, nullptr, nullptr, kLnEps, mod, mod + D,
                  mod_stride, normed.f32(), stream);
  if (!blas.Linear(block.qkv, block.qkv_b.f32(), normed.f32(), rows,
                   qkv.f32(), error))
    return false;
  const std::size_t offset = static_cast<std::size_t>(row) * D;
  LaunchSplitQkv(qkv.f32(), rows, kHeads, kHeadDim, block.q_norm.f32(),
                 block.k_norm.f32(), rope, q.f32() + offset, k.f32() + offset,
                 v.f32() + offset, D, stream);
  return true;
}

bool FlowNetwork::Impl::PostAttention(const SingleBlock& block, float* state_x,
                                      int rows, const float* attention,
                                      const float* mod, int mod_stride,
                                      std::string* error) {
  if (block.pre_only)
    return true;
  // x = x + linear1(attn) * gate_msa
  if (!RunConv(block.linear1, attention, rows, tmp.f32(), error))
    return false;
  LaunchAddGated(state_x, tmp.f32(), rows, D, mod + 2 * D, mod_stride, stream);
  // x = x + ffn(modulate(norm2(x), shift_mlp, scale_mlp)) * gate_mlp
  LaunchLayerNorm(state_x, rows, D, nullptr, nullptr, kLnEps, mod + 3 * D,
                  mod + 4 * D, mod_stride, normed.f32(), stream);
  if (!RunMlp(block.ffn, normed.f32(), rows, D, tmp.f32(), error))
    return false;
  LaunchAddGated(state_x, tmp.f32(), rows, D, mod + 5 * D, mod_stride, stream);
  return true;
}

bool FlowNetwork::PredictFlow(const float* latent, float t,
                              const Conditions& conditions, float* flow,
                              std::string* error) {
  Impl& m = *impl_;
  if (!m.EnsureLengths(conditions.lengths, error))
    return false;
  const int n = conditions.lengths.latent;
  const int lc = conditions.lengths.clip;
  const int total = n + lc + kTextTokens;
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  // latent = audio_input_proj(latent)
  if (!m.RunInputProjection(m.audio_in, m.audio_mlp, latent, n, m.x.f32(),
                            error))
    return false;
  // global_c = t_embed(t) + global_cond_mlp(...); extended_c = global_c + sync.
  const std::vector<float> frequencies = TimestepFrequencies(t);
  if (!Check(hipMemcpyAsync(m.t_freq.get(), frequencies.data(), D * 4,
                            hipMemcpyHostToDevice, m.stream),
             "timestep upload", error) ||
      !m.blas.Linear(m.t_mlp0.w, m.t_mlp0.b.f32(), m.t_freq.f32(), 1,
                     m.vec_a.f32(), error))
    return false;
  LaunchSilu(m.vec_a.f32(), D, 1.0F, m.stream);
  if (!m.blas.Linear(m.t_mlp2.w, m.t_mlp2.b.f32(), m.vec_a.f32(), 1,
                     m.vec_b.f32(), error))
    return false;
  LaunchAdd(m.vec_b.f32(), conditions.global.f32(), D, m.stream);
  float* global_c = m.vec_b.f32();
  float* silu_global = m.vec_c.f32();
  LaunchSiluTo(global_c, D, silu_global, m.stream);
  if (!Check(hipMemcpyAsync(m.extended.get(), conditions.sync.get(),
                            static_cast<std::size_t>(n) * D * 4,
                            hipMemcpyDeviceToDevice, m.stream),
             "extended condition", error) ||
      !Check(hipMemcpyAsync(m.clip_x.get(), conditions.clip.get(),
                            static_cast<std::size_t>(lc) * D * 4,
                            hipMemcpyDeviceToDevice, m.stream),
             "clip stream", error) ||
      !Check(hipMemcpyAsync(m.text_x.get(), conditions.text.get(),
                            static_cast<std::size_t>(kTextTokens) * D * 4,
                            hipMemcpyDeviceToDevice, m.stream),
             "text stream", error))
    return false;
  LaunchAddRow(m.extended.f32(), global_c, n, D, m.stream);
  LaunchSiluTo(m.extended.f32(), static_cast<std::size_t>(n) * D,
               m.silu_ext.f32(), m.stream);

  for (const JointBlock& block : m.joint) {
    if (!m.blas.Linear(block.latent.ada, block.latent.ada_b.f32(),
                       m.silu_ext.f32(), n, m.mod_latent.f32(), error) ||
        !m.blas.Linear(block.clip.ada, block.clip.ada_b.f32(), silu_global, 1,
                       m.mod_clip.f32(), error) ||
        !m.blas.Linear(block.text.ada, block.text.ada_b.f32(), silu_global, 1,
                       m.mod_text.f32(), error) ||
        !m.PreAttention(block.latent, m.x.f32(), n, m.mod_latent.f32(), 6 * D,
                        m.latent_rot.f32(), 0, error) ||
        !m.PreAttention(block.clip, m.clip_x.f32(), lc, m.mod_clip.f32(), 0,
                        m.clip_rot.f32(), n, error) ||
        !m.PreAttention(block.text, m.text_x.f32(), kTextTokens,
                        m.mod_text.f32(), 0, nullptr, n + lc, error) ||
        !m.blas.Attention(m.q.f32(), D, m.k.f32(), D, m.v.f32(), D,
                          m.attn.f32(), D, total, total, kHeads, kHeadDim,
                          scale, false, error) ||
        !m.PostAttention(block.latent, m.x.f32(), n, m.attn.f32(),
                         m.mod_latent.f32(), 6 * D, error) ||
        !m.PostAttention(block.clip, m.clip_x.f32(), lc,
                         m.attn.f32() + static_cast<std::size_t>(n) * D,
                         m.mod_clip.f32(), 0, error) ||
        !m.PostAttention(block.text, m.text_x.f32(), kTextTokens,
                         m.attn.f32() + static_cast<std::size_t>(n + lc) * D,
                         m.mod_text.f32(), 0, error))
      return false;
  }
  for (const SingleBlock& block : m.fused) {
    if (!m.blas.Linear(block.ada, block.ada_b.f32(), m.silu_ext.f32(), n,
                       m.mod_latent.f32(), error) ||
        !m.PreAttention(block, m.x.f32(), n, m.mod_latent.f32(), 6 * D,
                        m.latent_rot.f32(), 0, error) ||
        !m.blas.Attention(m.q.f32(), D, m.k.f32(), D, m.v.f32(), D,
                          m.attn.f32(), D, n, n, kHeads, kHeadDim, scale,
                          false, error) ||
        !m.PostAttention(block, m.x.f32(), n, m.attn.f32(), m.mod_latent.f32(),
                         6 * D, error))
      return false;
  }
  // final_layer(latent, global_c) (MMAudio uses global_c here, issue #55).
  if (!m.blas.Linear(m.final_ada, m.final_ada_b.f32(), silu_global, 1,
                     m.mod_clip.f32(), error))
    return false;
  LaunchLayerNorm(m.x.f32(), n, D, nullptr, nullptr, kLnEps, m.mod_clip.f32(),
                  m.mod_clip.f32() + D, 0, m.normed.f32(), m.stream);
  if (!m.RunConv(m.final_conv, m.normed.f32(), n, flow, error))
    return false;
  return Check(hipGetLastError(), "predict_flow", error);
}

bool FlowNetwork::Sample(const std::vector<float>& noise, int steps,
                         float cfg_strength, const Conditions& positive,
                         const Conditions& negative,
                         std::vector<float>* latent,
                         const std::function<bool(int, int)>& progress,
                         std::string* error) {
  Impl& m = *impl_;
  const int n = positive.lengths.latent;
  const std::size_t count = static_cast<std::size_t>(n) * kLatentDim;
  if (steps < 1 || noise.size() != count || negative.lengths.latent != n ||
      negative.lengths.clip != positive.lengths.clip)
    return Fail(error, "MMAudio sampling inputs do not match");
  if (!m.EnsureLengths(positive.lengths, error) ||
      !Check(hipMemcpyAsync(m.state.get(), noise.data(), count * 4,
                            hipMemcpyHostToDevice, m.stream),
             "noise upload", error))
    return false;
  for (int i = 0; i < steps; ++i) {
    // FlowMatching (euler): linspace(0, 1, steps + 1).
    const float t = static_cast<float>(i) / static_cast<float>(steps);
    const float next = static_cast<float>(i + 1) / static_cast<float>(steps);
    const float dt = next - t;
    if (!PredictFlow(m.state.f32(), t, positive, m.flow_a.f32(), error))
      return false;
    if (cfg_strength >= 1.0F) {
      if (!PredictFlow(m.state.f32(), t, negative, m.flow_b.f32(), error))
        return false;
      // cfg * conditional + (1 - cfg) * unconditional
      LaunchAffine(m.flow_a.f32(), count, cfg_strength, 0.0F, m.stream);
      LaunchAffine(m.flow_b.f32(), count, 1.0F - cfg_strength, 0.0F,
                   m.stream);
      LaunchAdd(m.flow_a.f32(), m.flow_b.f32(), count, m.stream);
    }
    LaunchAffine(m.flow_a.f32(), count, dt, 0.0F, m.stream);
    LaunchAdd(m.state.f32(), m.flow_a.f32(), count, m.stream);
    if (!Check(hipStreamSynchronize(m.stream), "sampling step", error))
      return false;
    if (progress && !progress(i + 1, steps))
      return Fail(error, "cancelled");
  }
  latent->resize(count);
  if (!Check(hipMemcpyAsync(latent->data(), m.state.get(), count * 4,
                            hipMemcpyDeviceToHost, m.stream),
             "latent readback", error) ||
      !Check(hipStreamSynchronize(m.stream), "latent readback", error))
    return false;
  // unnormalize: x * latent_std + latent_mean
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t c = i % kLatentDim;
    (*latent)[i] = (*latent)[i] * m.latent_std[c] + m.latent_mean[c];
  }
  return true;
}

}  // namespace gufo::models::mmaudio
