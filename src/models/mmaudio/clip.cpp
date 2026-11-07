// MMAudio for gufo: DFN5B CLIP image and text towers (see clip.hpp).
// Written with AI assistance (Claude Code).
#include "src/models/mmaudio/clip.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "src/models/mmaudio/kernels.hpp"

namespace gufo::models::mmaudio {
namespace {

constexpr int kPatch = 14;
constexpr int kGrid = 27;  // floor(384 / 14)
constexpr int kImageTokens = kGrid * kGrid + 1;  // 730
constexpr int kVisionWidth = 1280;
constexpr int kVisionHeads = 16;
constexpr int kVisionLayers = 32;
constexpr int kVisionMlp = 5120;
constexpr int kTextHeads = 16;
constexpr int kTextLayers = 24;
constexpr int kTextMlp = 4096;
constexpr int kVocabulary = 49408;
constexpr int kImageBatch = 8;
constexpr float kEps = 1e-5F;

struct Layer {
  DeviceBuffer ln1_w, ln1_b, ln2_w, ln2_b;
  Matrix qkv, out, fc1, fc2;
  DeviceBuffer qkv_b, out_b, fc1_b, fc2_b;
};

struct Tower {
  int width{0};
  int heads{0};
  int mlp{0};
  std::vector<Layer> layers;
};

bool LoadTower(const TensorFile& file, const std::string& prefix, int layers,
               int width, int mlp, Precision precision, Tower* tower,
               std::size_t* bytes, std::string* error) {
  tower->width = width;
  tower->mlp = mlp;
  tower->layers.resize(layers);
  for (int i = 0; i < layers; ++i) {
    Layer& l = tower->layers[i];
    const std::string p = prefix + "transformer.resblocks." +
                          std::to_string(i) + ".";
    if (!LoadF32(file, p + "ln_1.weight", width, &l.ln1_w, error) ||
        !LoadF32(file, p + "ln_1.bias", width, &l.ln1_b, error) ||
        !LoadF32(file, p + "ln_2.weight", width, &l.ln2_w, error) ||
        !LoadF32(file, p + "ln_2.bias", width, &l.ln2_b, error) ||
        !LoadMatrix(file, p + "attn.in_proj_weight", 3 * width, width,
                    precision, &l.qkv, error) ||
        !LoadF32(file, p + "attn.in_proj_bias", 3 * width, &l.qkv_b, error) ||
        !LoadMatrix(file, p + "attn.out_proj.weight", width, width, precision,
                    &l.out, error) ||
        !LoadF32(file, p + "attn.out_proj.bias", width, &l.out_b, error) ||
        !LoadMatrix(file, p + "mlp.c_fc.weight", mlp, width, precision,
                    &l.fc1, error) ||
        !LoadF32(file, p + "mlp.c_fc.bias", mlp, &l.fc1_b, error) ||
        !LoadMatrix(file, p + "mlp.c_proj.weight", width, mlp, precision,
                    &l.fc2, error) ||
        !LoadF32(file, p + "mlp.c_proj.bias", width, &l.fc2_b, error))
      return false;
    *bytes += l.qkv.bytes() + l.out.bytes() + l.fc1.bytes() + l.fc2.bytes();
  }
  return true;
}

}  // namespace

struct ClipEncoder::Impl {
  ~Impl() {
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }
  bool RunTower(const Tower& tower, int items, int sequence, bool causal,
                std::string* error);

  hipStream_t stream{nullptr};
  Blas blas;
  std::size_t weight_bytes{0};

  // Image tower.
  Matrix patch;  // [1280, 3 * 14 * 14]
  DeviceBuffer class_embedding, pos_tiled;  // pos repeated kImageBatch times
  DeviceBuffer ln_pre_w, ln_pre_b, ln_post_w, ln_post_b;
  Matrix proj;  // [1024, 1280] (transposed from [1280, 1024])
  Tower vision;
  // Text tower.
  DeviceBuffer token_table;  // FP32 [49408, 1024]
  DeviceBuffer text_pos, ln_final_w, ln_final_b;
  Tower text;

  // Activations.
  DeviceBuffer pixels, patches, hidden, normed, qkv, attn, mlp, pooled,
      features, ids;
};

ClipEncoder::ClipEncoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ClipEncoder::~ClipEncoder() = default;

std::size_t ClipEncoder::weight_bytes() const { return impl_->weight_bytes; }

std::unique_ptr<ClipEncoder> ClipEncoder::Load(
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
  m.vision.heads = kVisionHeads;
  m.text.heads = kTextHeads;
  std::vector<float> host;
  // Image tower embeddings.
  if (!LoadMatrix(file, "visual.conv1.weight", kVisionWidth,
                  3 * kPatch * kPatch, precision, &m.patch, error) ||
      !LoadF32(file, "visual.class_embedding", kVisionWidth,
               &m.class_embedding, error) ||
      !file.Read("visual.positional_embedding", &host, error,
                 static_cast<std::size_t>(kImageTokens) * kVisionWidth))
    return nullptr;
  {
    std::vector<float> tiled;
    tiled.reserve(host.size() * kImageBatch);
    for (int b = 0; b < kImageBatch; ++b)
      tiled.insert(tiled.end(), host.begin(), host.end());
    if (!UploadF32(tiled, &m.pos_tiled, error))
      return nullptr;
  }
  if (!LoadF32(file, "visual.ln_pre.weight", kVisionWidth, &m.ln_pre_w,
               error) ||
      !LoadF32(file, "visual.ln_pre.bias", kVisionWidth, &m.ln_pre_b, error) ||
      !LoadF32(file, "visual.ln_post.weight", kVisionWidth, &m.ln_post_w,
               error) ||
      !LoadF32(file, "visual.ln_post.bias", kVisionWidth, &m.ln_post_b,
               error) ||
      !file.Read("visual.proj", &host, error,
                 static_cast<std::size_t>(kVisionWidth) * kFeatures))
    return nullptr;
  {
    // pooled @ proj with proj [1280, 1024] -> W [1024, 1280].
    std::vector<float> transposed(host.size());
    for (int i = 0; i < kVisionWidth; ++i)
      for (int j = 0; j < kFeatures; ++j)
        transposed[static_cast<std::size_t>(j) * kVisionWidth + i] =
            host[static_cast<std::size_t>(i) * kFeatures + j];
    if (!UploadMatrix(transposed, kFeatures, kVisionWidth, precision, &m.proj,
                      error))
      return nullptr;
  }
  if (!LoadTower(file, "visual.", kVisionLayers, kVisionWidth, kVisionMlp,
                 precision, &m.vision, &m.weight_bytes, error))
    return nullptr;
  // Text tower.
  if (!LoadF32(file, "token_embedding.weight",
               static_cast<std::size_t>(kVocabulary) * kTextWidth,
               &m.token_table, error) ||
      !LoadF32(file, "positional_embedding",
               static_cast<std::size_t>(kTextTokens) * kTextWidth, &m.text_pos,
               error) ||
      !LoadF32(file, "ln_final.weight", kTextWidth, &m.ln_final_w, error) ||
      !LoadF32(file, "ln_final.bias", kTextWidth, &m.ln_final_b, error) ||
      !LoadTower(file, "", kTextLayers, kTextWidth, kTextMlp, precision,
                 &m.text, &m.weight_bytes, error))
    return nullptr;
  m.weight_bytes += m.patch.bytes() + m.proj.bytes() + m.token_table.bytes();
  // Activation buffers for the larger tower (image batch).
  const std::size_t rows = static_cast<std::size_t>(kImageBatch) * kImageTokens;
  if (!m.pixels.Allocate(static_cast<std::size_t>(kImageBatch) * 3 *
                             kImageSize * kImageSize * 4,
                         error) ||
      !m.patches.Allocate(rows * 3 * kPatch * kPatch * 4, error) ||
      !m.hidden.Allocate(rows * kVisionWidth * 4, error) ||
      !m.normed.Allocate(rows * kVisionWidth * 4, error) ||
      !m.qkv.Allocate(rows * 3 * kVisionWidth * 4, error) ||
      !m.attn.Allocate(rows * kVisionWidth * 4, error) ||
      !m.mlp.Allocate(rows * kVisionMlp * 4, error) ||
      !m.pooled.Allocate(static_cast<std::size_t>(kImageBatch) * kVisionWidth *
                             4,
                         error) ||
      !m.features.Allocate(static_cast<std::size_t>(kImageBatch) * kFeatures *
                               4,
                           error) ||
      !m.ids.Allocate(kTextTokens * sizeof(int), error) ||
      !Check(hipStreamSynchronize(m.stream), "load", error))
    return nullptr;
  return std::unique_ptr<ClipEncoder>(new ClipEncoder(std::move(impl)));
}

bool ClipEncoder::Impl::RunTower(const Tower& tower, int items, int sequence,
                                 bool causal, std::string* error) {
  const int width = tower.width;
  const int head_dim = width / tower.heads;
  const int rows = items * sequence;
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  for (const Layer& l : tower.layers) {
    LaunchLayerNorm(hidden.f32(), rows, width, l.ln1_w.f32(), l.ln1_b.f32(),
                    kEps, nullptr, nullptr, 0, normed.f32(), stream);
    if (!blas.Linear(l.qkv, l.qkv_b.f32(), normed.f32(), rows, qkv.f32(),
                     error))
      return false;
    for (int s = 0; s < items; ++s) {
      const float* q =
          qkv.f32() + static_cast<std::size_t>(s) * sequence * 3 * width;
      if (!blas.Attention(q, 3 * width, q + width, 3 * width, q + 2 * width,
                          3 * width,
                          attn.f32() +
                              static_cast<std::size_t>(s) * sequence * width,
                          width, sequence, sequence, tower.heads, head_dim,
                          scale, causal, error))
        return false;
    }
    if (!blas.Linear(l.out, l.out_b.f32(), attn.f32(), rows, hidden.f32(),
                     error, /*accumulate=*/true))
      return false;
    LaunchLayerNorm(hidden.f32(), rows, width, l.ln2_w.f32(), l.ln2_b.f32(),
                    kEps, nullptr, nullptr, 0, normed.f32(), stream);
    if (!blas.Linear(l.fc1, l.fc1_b.f32(), normed.f32(), rows, mlp.f32(),
                     error))
      return false;
    LaunchQuickGelu(mlp.f32(), static_cast<std::size_t>(rows) * tower.mlp,
                    stream);
    if (!blas.Linear(l.fc2, l.fc2_b.f32(), mlp.f32(), rows, hidden.f32(),
                     error, /*accumulate=*/true))
      return false;
  }
  return Check(hipGetLastError(), "CLIP layers", error);
}

bool ClipEncoder::EncodeImages(const std::vector<float>& pixels, int frames,
                               std::vector<float>* features,
                               std::string* error) {
  Impl& m = *impl_;
  const std::size_t frame_values =
      static_cast<std::size_t>(3) * kImageSize * kImageSize;
  if (frames <= 0 || pixels.size() != frame_values * frames)
    return Fail(error, "CLIP: pixels must be [frames][3][384][384]");
  features->assign(static_cast<std::size_t>(frames) * kFeatures, 0.0F);
  for (int start = 0; start < frames; start += kImageBatch) {
    const int batch = std::min(kImageBatch, frames - start);
    const int rows = batch * kImageTokens;
    if (!Check(hipMemcpyAsync(m.pixels.get(),
                              pixels.data() + frame_values * start,
                              frame_values * batch * 4, hipMemcpyHostToDevice,
                              m.stream),
               "pixel upload", error))
      return false;
    LaunchPatchify2d(m.pixels.f32(), batch, kImageSize, kPatch, kGrid,
                     m.patches.f32(), m.stream);
    // Patch embeddings into normed (scratch), then CLS + patches + positions.
    if (!m.blas.Linear(m.patch, nullptr, m.patches.f32(),
                       batch * (kImageTokens - 1), m.normed.f32(), error))
      return false;
    for (int b = 0; b < batch; ++b) {
      float* frame = m.hidden.f32() +
                     static_cast<std::size_t>(b) * kImageTokens * kVisionWidth;
      if (!Check(hipMemcpyAsync(frame, m.class_embedding.get(),
                                kVisionWidth * 4, hipMemcpyDeviceToDevice,
                                m.stream),
                 "class token", error) ||
          !Check(hipMemcpyAsync(
                     frame + kVisionWidth,
                     m.normed.f32() + static_cast<std::size_t>(b) *
                                          (kImageTokens - 1) * kVisionWidth,
                     static_cast<std::size_t>(kImageTokens - 1) *
                         kVisionWidth * 4,
                     hipMemcpyDeviceToDevice, m.stream),
                 "patch tokens", error))
        return false;
    }
    LaunchAdd(m.hidden.f32(), m.pos_tiled.f32(),
              static_cast<std::size_t>(rows) * kVisionWidth, m.stream);
    LaunchLayerNorm(m.hidden.f32(), rows, kVisionWidth, m.ln_pre_w.f32(),
                    m.ln_pre_b.f32(), kEps, nullptr, nullptr, 0,
                    m.hidden.f32(), m.stream);
    if (!m.RunTower(m.vision, batch, kImageTokens, false, error))
      return false;
    // CLS rows -> ln_post -> proj -> L2 normalize.
    if (!Check(hipMemcpy2DAsync(m.pooled.get(), kVisionWidth * 4,
                                m.hidden.get(),
                                static_cast<std::size_t>(kImageTokens) *
                                    kVisionWidth * 4,
                                kVisionWidth * 4, batch,
                                hipMemcpyDeviceToDevice, m.stream),
               "pooling", error))
      return false;
    LaunchLayerNorm(m.pooled.f32(), batch, kVisionWidth, m.ln_post_w.f32(),
                    m.ln_post_b.f32(), kEps, nullptr, nullptr, 0,
                    m.pooled.f32(), m.stream);
    if (!m.blas.Linear(m.proj, nullptr, m.pooled.f32(), batch,
                       m.features.f32(), error))
      return false;
    LaunchL2Normalize(m.features.f32(), batch, kFeatures, m.stream);
    if (!Check(hipMemcpyAsync(features->data() +
                                  static_cast<std::size_t>(start) * kFeatures,
                              m.features.get(),
                              static_cast<std::size_t>(batch) * kFeatures * 4,
                              hipMemcpyDeviceToHost, m.stream),
               "feature readback", error) ||
        !Check(hipStreamSynchronize(m.stream), "image encode", error))
      return false;
  }
  return true;
}

bool ClipEncoder::EncodeText(const std::vector<int>& tokens,
                             std::vector<float>* features,
                             std::string* error) {
  Impl& m = *impl_;
  if (tokens.size() != static_cast<std::size_t>(kTextTokens))
    return Fail(error, "CLIP text needs 77 tokens");
  for (const int id : tokens)
    if (id < 0 || id >= kVocabulary)
      return Fail(error, "CLIP token id out of range");
  if (!Check(hipMemcpyAsync(m.ids.get(), tokens.data(),
                            kTextTokens * sizeof(int), hipMemcpyHostToDevice,
                            m.stream),
             "token upload", error))
    return false;
  LaunchGatherRows(m.token_table.f32(), static_cast<const int*>(m.ids.get()),
                   kTextTokens, kTextWidth, m.hidden.f32(), m.stream);
  LaunchAdd(m.hidden.f32(), m.text_pos.f32(),
            static_cast<std::size_t>(kTextTokens) * kTextWidth, m.stream);
  if (!m.RunTower(m.text, 1, kTextTokens, true, error))
    return false;
  LaunchLayerNorm(m.hidden.f32(), kTextTokens, kTextWidth, m.ln_final_w.f32(),
                  m.ln_final_b.f32(), kEps, nullptr, nullptr, 0,
                  m.normed.f32(), m.stream);
  LaunchL2Normalize(m.normed.f32(), kTextTokens, kTextWidth, m.stream);
  features->resize(static_cast<std::size_t>(kTextTokens) * kTextWidth);
  return Check(hipMemcpyAsync(features->data(), m.normed.get(),
                              features->size() * 4, hipMemcpyDeviceToHost,
                              m.stream),
               "text readback", error) &&
         Check(hipStreamSynchronize(m.stream), "text encode", error);
}

}  // namespace gufo::models::mmaudio
