// MMAudio for gufo: the Synchformer visual feature extractor (see
// synchformer.hpp).
// Written with AI assistance (Claude Code).
#include "src/models/mmaudio/synchformer.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "src/models/mmaudio/kernels.hpp"

namespace gufo::models::mmaudio {
namespace {

constexpr int kWidth = Synchformer::kWidth;
constexpr int kHeads = 12;
constexpr int kMlp = 3072;
constexpr int kBlocks = 12;
constexpr int kTime = 8;                  // tubelet positions per segment
constexpr int kPatches = 14 * 14;         // per frame
constexpr int kTokens = 1 + kTime * kPatches;  // 1569
constexpr int kTubelet = 3 * 2 * 16 * 16;      // 1536
constexpr int kAggTokens = 1 + kPatches;        // 197
constexpr int kBatch = 4;                 // segments per pass
constexpr float kEps = 1e-6F;

struct Block {
  DeviceBuffer norm1_w, norm1_b, norm2_w, norm2_b, norm3_w, norm3_b;
  Matrix qkv, proj, time_qkv, time_proj, fc1, fc2;
  DeviceBuffer qkv_b, proj_b, time_qkv_b, time_proj_b, fc1_b, fc2_b;
};

}  // namespace

struct Synchformer::Impl {
  ~Impl() {
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }

  hipStream_t stream{nullptr};
  Blas blas;
  std::size_t weight_bytes{0};

  Matrix patch;  // [768, 1536]
  DeviceBuffer patch_b;
  DeviceBuffer embed_tiled;  // [kBatch * 1569, 768]: CLS + positions
  std::vector<Block> blocks;
  DeviceBuffer norm_w, norm_b;
  // Spatial aggregation layer.
  DeviceBuffer agg_cls, agg_norm1_w, agg_norm1_b, agg_norm2_w, agg_norm2_b;
  Matrix agg_qkv, agg_out, agg_fc1, agg_fc2;
  DeviceBuffer agg_qkv_b, agg_out_b, agg_fc1_b, agg_fc2_b;

  DeviceBuffer frames, patches, hidden, normed, qkv, attn, mlp, pooled;
};

Synchformer::Synchformer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Synchformer::~Synchformer() = default;

std::size_t Synchformer::weight_bytes() const { return impl_->weight_bytes; }

std::unique_ptr<Synchformer> Synchformer::Load(
    const std::filesystem::path& path, Precision precision,
    std::string* error) {
  TensorFile file;
  if (!file.Open(path, error))
    return nullptr;
  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  if (!Check(hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking),
             "stream", error) ||
      !m.blas.Init(m.stream, error))
    return nullptr;
  const std::string v = "vfeat_extractor.";
  const auto matrix = [&](const std::string& name, int rows, int columns,
                          Matrix* out) {
    if (!LoadMatrix(file, name, rows, columns, precision, out, error))
      return false;
    m.weight_bytes += out->bytes();
    return true;
  };
  const auto vec = [&](const std::string& name, std::size_t count,
                       DeviceBuffer* out) {
    return LoadF32(file, name, count, out, error);
  };
  if (!matrix(v + "patch_embed_3d.proj.weight", kWidth, kTubelet, &m.patch) ||
      !vec(v + "patch_embed_3d.proj.bias", kWidth, &m.patch_b))
    return nullptr;
  {
    // Row 0: cls_token + pos_embed[0]; row 1 + t * 196 + n:
    // pos_embed[1 + n] + temp_embed[t] (POS_EMBED "separate").
    std::vector<float> cls, pos, temp;
    if (!file.Read(v + "cls_token", &cls, error, kWidth) ||
        !file.Read(v + "pos_embed", &pos, error,
                   static_cast<std::size_t>(kAggTokens) * kWidth) ||
        !file.Read(v + "temp_embed", &temp, error,
                   static_cast<std::size_t>(kTime) * kWidth))
      return nullptr;
    std::vector<float> table(static_cast<std::size_t>(kTokens) * kWidth);
    for (int c = 0; c < kWidth; ++c)
      table[c] = cls[c] + pos[c];
    for (int t = 0; t < kTime; ++t)
      for (int n = 0; n < kPatches; ++n)
        for (int c = 0; c < kWidth; ++c)
          table[(static_cast<std::size_t>(1 + t * kPatches + n)) * kWidth + c] =
              pos[static_cast<std::size_t>(1 + n) * kWidth + c] +
              temp[static_cast<std::size_t>(t) * kWidth + c];
    std::vector<float> tiled;
    tiled.reserve(table.size() * kBatch);
    for (int b = 0; b < kBatch; ++b)
      tiled.insert(tiled.end(), table.begin(), table.end());
    if (!UploadF32(tiled, &m.embed_tiled, error))
      return nullptr;
  }
  m.blocks.resize(kBlocks);
  for (int i = 0; i < kBlocks; ++i) {
    Block& b = m.blocks[i];
    const std::string p = v + "blocks." + std::to_string(i) + ".";
    if (!vec(p + "norm1.weight", kWidth, &b.norm1_w) ||
        !vec(p + "norm1.bias", kWidth, &b.norm1_b) ||
        !vec(p + "norm2.weight", kWidth, &b.norm2_w) ||
        !vec(p + "norm2.bias", kWidth, &b.norm2_b) ||
        !vec(p + "norm3.weight", kWidth, &b.norm3_w) ||
        !vec(p + "norm3.bias", kWidth, &b.norm3_b) ||
        !matrix(p + "attn.qkv.weight", 3 * kWidth, kWidth, &b.qkv) ||
        !vec(p + "attn.qkv.bias", 3 * kWidth, &b.qkv_b) ||
        !matrix(p + "attn.proj.weight", kWidth, kWidth, &b.proj) ||
        !vec(p + "attn.proj.bias", kWidth, &b.proj_b) ||
        !matrix(p + "timeattn.qkv.weight", 3 * kWidth, kWidth, &b.time_qkv) ||
        !vec(p + "timeattn.qkv.bias", 3 * kWidth, &b.time_qkv_b) ||
        !matrix(p + "timeattn.proj.weight", kWidth, kWidth, &b.time_proj) ||
        !vec(p + "timeattn.proj.bias", kWidth, &b.time_proj_b) ||
        !matrix(p + "mlp.fc1.weight", kMlp, kWidth, &b.fc1) ||
        !vec(p + "mlp.fc1.bias", kMlp, &b.fc1_b) ||
        !matrix(p + "mlp.fc2.weight", kWidth, kMlp, &b.fc2) ||
        !vec(p + "mlp.fc2.bias", kWidth, &b.fc2_b))
      return nullptr;
  }
  const std::string a = v + "spatial_attn_agg.";
  if (!vec(v + "norm.weight", kWidth, &m.norm_w) ||
      !vec(v + "norm.bias", kWidth, &m.norm_b) ||
      !vec(a + "cls_token", kWidth, &m.agg_cls) ||
      !vec(a + "norm1.weight", kWidth, &m.agg_norm1_w) ||
      !vec(a + "norm1.bias", kWidth, &m.agg_norm1_b) ||
      !vec(a + "norm2.weight", kWidth, &m.agg_norm2_w) ||
      !vec(a + "norm2.bias", kWidth, &m.agg_norm2_b) ||
      !matrix(a + "self_attn.in_proj_weight", 3 * kWidth, kWidth,
              &m.agg_qkv) ||
      !vec(a + "self_attn.in_proj_bias", 3 * kWidth, &m.agg_qkv_b) ||
      !matrix(a + "self_attn.out_proj.weight", kWidth, kWidth, &m.agg_out) ||
      !vec(a + "self_attn.out_proj.bias", kWidth, &m.agg_out_b) ||
      !matrix(a + "linear1.weight", kMlp, kWidth, &m.agg_fc1) ||
      !vec(a + "linear1.bias", kMlp, &m.agg_fc1_b) ||
      !matrix(a + "linear2.weight", kWidth, kMlp, &m.agg_fc2) ||
      !vec(a + "linear2.bias", kWidth, &m.agg_fc2_b))
    return nullptr;
  const std::size_t rows = static_cast<std::size_t>(kBatch) * kTokens;
  const std::size_t agg_rows =
      static_cast<std::size_t>(kBatch) * kTime * kAggTokens;
  const std::size_t max_rows = std::max(rows, agg_rows);
  if (!m.patches.Allocate(static_cast<std::size_t>(kBatch) * (kTokens - 1) *
                              kTubelet * 4,
                          error) ||
      !m.hidden.Allocate(max_rows * kWidth * 4, error) ||
      !m.normed.Allocate(max_rows * kWidth * 4, error) ||
      !m.qkv.Allocate(max_rows * 3 * kWidth * 4, error) ||
      !m.attn.Allocate(max_rows * kWidth * 4, error) ||
      !m.mlp.Allocate(max_rows * kMlp * 4, error) ||
      !m.pooled.Allocate(static_cast<std::size_t>(kBatch) * kTime * kWidth * 4,
                         error) ||
      !Check(hipStreamSynchronize(m.stream), "load", error))
    return nullptr;
  return std::unique_ptr<Synchformer>(new Synchformer(std::move(impl)));
}

bool Synchformer::Encode(const std::vector<float>& frames, int count,
                         std::vector<float>* features, std::string* error) {
  Impl& m = *impl_;
  const std::size_t frame_values = static_cast<std::size_t>(3) * kSize * kSize;
  if (count < kSegmentFrames || frames.size() != frame_values * count)
    return Fail(error, "Synchformer needs >= 16 frames of [3][224][224]");
  const int segments = (count - kSegmentFrames) / kSegmentStep + 1;
  const int used_frames = (segments - 1) * kSegmentStep + kSegmentFrames;
  if (!m.frames.Reserve(frame_values * used_frames * 4, error) ||
      !Check(hipMemcpyAsync(m.frames.get(), frames.data(),
                            frame_values * used_frames * 4,
                            hipMemcpyHostToDevice, m.stream),
             "frame upload", error))
    return false;
  features->assign(static_cast<std::size_t>(segments) * kTime * kWidth, 0.0F);
  const float scale = 1.0F / std::sqrt(64.0F);
  for (int start = 0; start < segments; start += kBatch) {
    const int batch = std::min(kBatch, segments - start);
    const int rows = batch * kTokens;
    LaunchPatchify3d(m.frames.f32() +
                         static_cast<std::size_t>(start) * kSegmentStep *
                             frame_values,
                     batch, kSegmentStep, m.patches.f32(), m.stream);
    if (!m.blas.Linear(m.patch, m.patch_b.f32(), m.patches.f32(),
                       batch * (kTokens - 1), m.normed.f32(), error))
      return false;
    // hidden = [0, tubelets] per segment + CLS / position table.
    if (!Check(hipMemsetAsync(m.hidden.get(), 0,
                              static_cast<std::size_t>(rows) * kWidth * 4,
                              m.stream),
               "clear", error))
      return false;
    for (int b = 0; b < batch; ++b)
      if (!Check(hipMemcpyAsync(
                     m.hidden.f32() +
                         (static_cast<std::size_t>(b) * kTokens + 1) * kWidth,
                     m.normed.f32() +
                         static_cast<std::size_t>(b) * (kTokens - 1) * kWidth,
                     static_cast<std::size_t>(kTokens - 1) * kWidth * 4,
                     hipMemcpyDeviceToDevice, m.stream),
                 "tubelets", error))
        return false;
    LaunchAdd(m.hidden.f32(), m.embed_tiled.f32(),
              static_cast<std::size_t>(rows) * kWidth, m.stream);
    for (const Block& blk : m.blocks) {
      // Time attention, then space attention, then the MLP.
      LaunchLayerNorm(m.hidden.f32(), rows, kWidth, blk.norm3_w.f32(),
                      blk.norm3_b.f32(), kEps, nullptr, nullptr, 0,
                      m.normed.f32(), m.stream);
      if (!m.blas.Linear(blk.time_qkv, blk.time_qkv_b.f32(), m.normed.f32(),
                         rows, m.qkv.f32(), error))
        return false;
      LaunchSubsetAttention(m.qkv.f32(), m.qkv.f32() + kWidth,
                            m.qkv.f32() + 2 * kWidth, 3 * kWidth, m.attn.f32(),
                            rows, kHeads, kTokens, kTime, kPatches,
                            KeySet::kTime, scale, m.stream);
      if (!m.blas.Linear(blk.time_proj, blk.time_proj_b.f32(), m.attn.f32(),
                         rows, m.hidden.f32(), error, true))
        return false;
      LaunchLayerNorm(m.hidden.f32(), rows, kWidth, blk.norm1_w.f32(),
                      blk.norm1_b.f32(), kEps, nullptr, nullptr, 0,
                      m.normed.f32(), m.stream);
      if (!m.blas.Linear(blk.qkv, blk.qkv_b.f32(), m.normed.f32(), rows,
                         m.qkv.f32(), error))
        return false;
      LaunchSubsetAttention(m.qkv.f32(), m.qkv.f32() + kWidth,
                            m.qkv.f32() + 2 * kWidth, 3 * kWidth, m.attn.f32(),
                            rows, kHeads, kTokens, kTime, kPatches,
                            KeySet::kSpace, scale, m.stream);
      if (!m.blas.Linear(blk.proj, blk.proj_b.f32(), m.attn.f32(), rows,
                         m.hidden.f32(), error, true))
        return false;
      LaunchLayerNorm(m.hidden.f32(), rows, kWidth, blk.norm2_w.f32(),
                      blk.norm2_b.f32(), kEps, nullptr, nullptr, 0,
                      m.normed.f32(), m.stream);
      if (!m.blas.Linear(blk.fc1, blk.fc1_b.f32(), m.normed.f32(), rows,
                         m.mlp.f32(), error))
        return false;
      LaunchGelu(m.mlp.f32(), static_cast<std::size_t>(rows) * kMlp, m.stream);
      if (!m.blas.Linear(blk.fc2, blk.fc2_b.f32(), m.mlp.f32(), rows,
                         m.hidden.f32(), error, true))
        return false;
    }
    // Final norm (the CLS token is dropped afterwards).
    LaunchLayerNorm(m.hidden.f32(), rows, kWidth, m.norm_w.f32(),
                    m.norm_b.f32(), kEps, nullptr, nullptr, 0, m.normed.f32(),
                    m.stream);
    // Spatial aggregation: per (segment, frame) [agg CLS, 196 patches].
    const int groups = batch * kTime;
    const int agg_rows = groups * kAggTokens;
    for (int g = 0; g < groups; ++g) {
      const int b = g / kTime, t = g % kTime;
      float* group = m.hidden.f32() +
                     static_cast<std::size_t>(g) * kAggTokens * kWidth;
      if (!Check(hipMemcpyAsync(group, m.agg_cls.get(), kWidth * 4,
                                hipMemcpyDeviceToDevice, m.stream),
                 "aggregation token", error) ||
          !Check(hipMemcpyAsync(
                     group + kWidth,
                     m.normed.f32() + (static_cast<std::size_t>(b) * kTokens +
                                       1 + t * kPatches) *
                                          kWidth,
                     static_cast<std::size_t>(kPatches) * kWidth * 4,
                     hipMemcpyDeviceToDevice, m.stream),
                 "aggregation input", error))
        return false;
    }
    LaunchLayerNorm(m.hidden.f32(), agg_rows, kWidth, m.agg_norm1_w.f32(),
                    m.agg_norm1_b.f32(), kEps, nullptr, nullptr, 0,
                    m.normed.f32(), m.stream);
    if (!m.blas.Linear(m.agg_qkv, m.agg_qkv_b.f32(), m.normed.f32(), agg_rows,
                       m.qkv.f32(), error))
      return false;
    LaunchSubsetAttention(m.qkv.f32(), m.qkv.f32() + kWidth,
                          m.qkv.f32() + 2 * kWidth, 3 * kWidth, m.attn.f32(),
                          agg_rows, kHeads, kAggTokens, 0, 0, KeySet::kFull,
                          scale, m.stream);
    if (!m.blas.Linear(m.agg_out, m.agg_out_b.f32(), m.attn.f32(), agg_rows,
                       m.hidden.f32(), error, true))
      return false;
    LaunchLayerNorm(m.hidden.f32(), agg_rows, kWidth, m.agg_norm2_w.f32(),
                    m.agg_norm2_b.f32(), kEps, nullptr, nullptr, 0,
                    m.normed.f32(), m.stream);
    if (!m.blas.Linear(m.agg_fc1, m.agg_fc1_b.f32(), m.normed.f32(), agg_rows,
                       m.mlp.f32(), error))
      return false;
    LaunchGelu(m.mlp.f32(), static_cast<std::size_t>(agg_rows) * kMlp,
               m.stream);
    if (!m.blas.Linear(m.agg_fc2, m.agg_fc2_b.f32(), m.mlp.f32(), agg_rows,
                       m.hidden.f32(), error, true))
      return false;
    // CLS row of every group -> features.
    if (!Check(hipMemcpy2DAsync(m.pooled.get(), kWidth * 4, m.hidden.get(),
                                static_cast<std::size_t>(kAggTokens) * kWidth *
                                    4,
                                kWidth * 4, groups, hipMemcpyDeviceToDevice,
                                m.stream),
               "aggregation output", error) ||
        !Check(hipMemcpyAsync(features->data() + static_cast<std::size_t>(
                                                     start) *
                                                     kTime * kWidth,
                              m.pooled.get(),
                              static_cast<std::size_t>(groups) * kWidth * 4,
                              hipMemcpyDeviceToHost, m.stream),
               "feature readback", error) ||
        !Check(hipStreamSynchronize(m.stream), "synchformer", error))
      return false;
  }
  return true;
}

}  // namespace gufo::models::mmaudio
