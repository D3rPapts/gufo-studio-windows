#include "src/models/qwen3_omni/model.hpp"

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "src/core/platform/mapped_file.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::qwen3_omni {
namespace {

using core::GgmlType;

std::string ShapeText(std::initializer_list<std::uint64_t> shape) {
  std::string text = "[";
  for (const std::uint64_t dimension : shape) {
    if (text.size() > 1) {
      text += ", ";
    }
    text += std::to_string(dimension);
  }
  return text + "]";
}

/// Looks up `name` and checks its GGUF shape (ggml order, innermost first)
/// and type. `types` empty accepts any type the row-size helper knows.
Weight Require(const core::GgufReader& reader, std::string_view name,
               std::initializer_list<std::uint64_t> shape,
               std::initializer_list<GgmlType> types = {}) {
  const core::GgufTensorInfo* info = reader.FindTensor(name);
  if (info == nullptr) {
    throw std::runtime_error("Qwen3-Omni tensor missing: " + std::string(name));
  }
  bool shape_ok = info->dimensions.size() == shape.size();
  if (shape_ok) {
    std::size_t index = 0;
    for (const std::uint64_t dimension : shape) {
      shape_ok = shape_ok && info->dimensions[index++] == dimension;
    }
  }
  if (!shape_ok) {
    throw std::runtime_error("Qwen3-Omni tensor " + std::string(name) +
                             " does not have shape " + ShapeText(shape));
  }
  bool type_ok = types.size() == 0;
  for (const GgmlType type : types) {
    type_ok = type_ok || info->type == type;
  }
  if (!type_ok) {
    throw std::runtime_error("Qwen3-Omni tensor " + std::string(name) +
                             " has unsupported type " +
                             std::string(core::ToString(info->type)));
  }
  Weight weight;
  weight.data = info->data;
  weight.type = info->type;
  weight.cols = info->dimensions[0];
  std::uint64_t outer = 1;
  for (std::size_t index = 1; index < info->dimensions.size(); ++index) {
    outer *= info->dimensions[index];
  }
  weight.rows = outer;
  weight.row_bytes = RowBytes(info->type, weight.cols);
  if (weight.row_bytes == 0 || weight.row_bytes * outer != info->size_bytes) {
    throw std::runtime_error("Qwen3-Omni tensor " + std::string(name) +
                             " has an unexpected byte size");
  }
  return weight;
}

/// A stacked expert tensor [cols, rows, experts].
Weight RequireExperts(const core::GgufReader& reader, std::string_view name,
                      std::uint64_t cols, std::uint64_t rows,
                      std::uint64_t experts) {
  Weight weight = Require(reader, name, {cols, rows, experts},
                          {GgmlType::kQ4_K, GgmlType::kQ6_K});
  weight.rows = rows;
  weight.experts = experts;
  return weight;
}

/// A 3x3 F16 convolution kernel [3, 3, cin, cout], viewed as cout rows of
/// cin * 9 elements in (channel, ky, kx) order.
Weight RequireConvolution(const core::GgufReader& reader, std::string_view name,
                          std::uint64_t cin, std::uint64_t cout) {
  Weight weight = Require(reader, name, {3, 3, cin, cout}, {GgmlType::kF16});
  weight.cols = cin * 9U;
  weight.rows = cout;
  weight.row_bytes = RowBytes(weight.type, weight.cols);
  return weight;
}

std::uint32_t MetadataU32(const core::GgufReader& reader, std::string_view key) {
  const auto value = reader.GetMetadataUint32(key);
  if (!value.has_value()) {
    throw std::runtime_error("Qwen3-Omni GGUF metadata missing: " +
                             std::string(key));
  }
  return *value;
}

float MetadataF32(const core::GgufReader& reader, std::string_view key) {
  const auto value = reader.GetMetadataFloat32(key);
  if (!value.has_value()) {
    throw std::runtime_error("Qwen3-Omni GGUF metadata missing: " +
                             std::string(key));
  }
  return *value;
}

std::string Blk(std::size_t layer, std::string_view suffix) {
  return "blk." + std::to_string(layer) + "." + std::string(suffix);
}

std::string AudioBlk(std::size_t layer, std::string_view suffix) {
  return "a.blk." + std::to_string(layer) + "." + std::string(suffix);
}

constexpr std::initializer_list<GgmlType> kQuantized = {GgmlType::kQ4_K,
                                                        GgmlType::kQ6_K};
constexpr std::initializer_list<GgmlType> kF32 = {GgmlType::kF32};
constexpr std::initializer_list<GgmlType> kF16 = {GgmlType::kF16};

}  // namespace

std::size_t RowBytes(core::GgmlType type, std::uint64_t cols) {
  return quant::EncodedSizeBytes(type, static_cast<std::size_t>(cols));
}

std::span<const std::byte> ReadWeightBytes(const core::GgufReader& reader,
                                           const Weight& weight,
                                           std::vector<std::byte>* scratch) {
  constexpr std::size_t kSector = 4096;
  const std::size_t bytes = weight.Bytes();
  const auto address = reinterpret_cast<std::uintptr_t>(weight.data);
  for (const core::GgufMappedRegion& region : reader.GetMappedRegions()) {
    const auto base = reinterpret_cast<std::uintptr_t>(region.data);
    if (region.data == nullptr || address < base ||
        address - base > region.size ||
        bytes > region.size - (address - base)) {
      continue;
    }
    if (!platform::FileValid(region.file_descriptor)) {
      break;
    }
    // The sector-aligned superset into a sector-aligned buffer, so the
    // reads bypass the file cache.
    const std::uint64_t offset = address - base;
    const std::uint64_t aligned = offset & ~std::uint64_t{kSector - 1U};
    const auto lead = static_cast<std::size_t>(offset - aligned);
    const std::size_t span = (lead + bytes + kSector - 1U) & ~(kSector - 1U);
    if (scratch->size() < span + kSector) {
      scratch->resize(span + kSector);
    }
    const auto start = reinterpret_cast<std::uintptr_t>(scratch->data());
    std::byte* buffer =
        scratch->data() + ((kSector - start % kSector) % kSector);
    if (platform::ReadFileParallel(region.file_descriptor, buffer, span,
                                   aligned) < lead + bytes) {
      throw std::runtime_error("cannot read a Qwen3-Omni weight from " +
                               region.source_path.string());
    }
    return {buffer + lead, bytes};
  }
  return {static_cast<const std::byte*>(weight.data), bytes};
}

std::size_t AudioTokenCount(std::size_t frames, std::uint32_t n_window) {
  const std::size_t chunk = 2U * n_window;
  const std::size_t remainder = frames % chunk;
  std::size_t tail = 0;
  if (remainder != 0U) {
    const std::size_t first = (remainder - 1U) / 2U + 1U;
    const std::size_t second = (first - 1U) / 2U + 1U;
    tail = (second - 1U) / 2U + 1U;
  }
  return frames / chunk * 13U + tail;
}

std::unique_ptr<Model> Model::Load(const std::filesystem::path& text_gguf,
                                   const std::filesystem::path& projector_gguf,
                                   std::string* error) {
  std::unique_ptr<Model> model(new Model());
  try {
    std::string open_error;
    model->text_reader_ = core::GgufReader::OpenFile(text_gguf, &open_error);
    if (model->text_reader_ == nullptr) {
      throw std::runtime_error("cannot open " + text_gguf.string() + ": " +
                               open_error);
    }
    model->projector_reader_ =
        core::GgufReader::OpenFile(projector_gguf, &open_error);
    if (model->projector_reader_ == nullptr) {
      throw std::runtime_error("cannot open " + projector_gguf.string() + ": " +
                               open_error);
    }
    const core::GgufReader& text = *model->text_reader_;
    const core::GgufReader& projector = *model->projector_reader_;

    if (text.GetMetadataString("general.architecture") != "qwen3vlmoe") {
      throw std::runtime_error(
          "text GGUF is not a Qwen3-Omni thinker (arch qwen3vlmoe)");
    }
    if (projector.GetMetadataString("clip.audio.projector_type") != "qwen3a") {
      throw std::runtime_error(
          "projector GGUF has no Qwen3-Omni audio tower (qwen3a)");
    }

    TextConfig& tc = model->text_;
    tc.layers = MetadataU32(text, "qwen3vlmoe.block_count");
    tc.hidden = MetadataU32(text, "qwen3vlmoe.embedding_length");
    tc.heads = MetadataU32(text, "qwen3vlmoe.attention.head_count");
    tc.kv_heads = MetadataU32(text, "qwen3vlmoe.attention.head_count_kv");
    tc.head_dim = MetadataU32(text, "qwen3vlmoe.attention.key_length");
    tc.experts = MetadataU32(text, "qwen3vlmoe.expert_count");
    tc.experts_used = MetadataU32(text, "qwen3vlmoe.expert_used_count");
    tc.expert_ff = MetadataU32(text, "qwen3vlmoe.expert_feed_forward_length");
    tc.context = MetadataU32(text, "qwen3vlmoe.context_length");
    tc.rope_theta = MetadataF32(text, "qwen3vlmoe.rope.freq_base");
    tc.eps = MetadataF32(text, "qwen3vlmoe.attention.layer_norm_rms_epsilon");
    if (MetadataU32(text, "qwen3vlmoe.attention.value_length") != tc.head_dim) {
      throw std::runtime_error("Qwen3-Omni key and value widths differ");
    }
    const auto shared =
        text.GetMetadataUint32("qwen3vlmoe.expert_shared_feed_forward_length");
    if (shared.has_value() && *shared != 0U) {
      throw std::runtime_error("Qwen3-Omni thinker must not have a shared "
                               "expert");
    }
    const core::GgufTensorInfo* embd = text.FindTensor("token_embd.weight");
    if (embd == nullptr || embd->dimensions.size() != 2) {
      throw std::runtime_error("Qwen3-Omni token_embd.weight missing");
    }
    tc.vocab = static_cast<std::uint32_t>(embd->dimensions[1]);

    AudioConfig& ac = model->audio_;
    ac.hidden = MetadataU32(projector, "clip.audio.embedding_length");
    ac.ffn = MetadataU32(projector, "clip.audio.feed_forward_length");
    ac.layers = MetadataU32(projector, "clip.audio.block_count");
    ac.heads = MetadataU32(projector, "clip.audio.attention.head_count");
    ac.output = MetadataU32(projector, "clip.audio.projection_dim");
    ac.mel_bins = MetadataU32(projector, "clip.audio.num_mel_bins");
    ac.eps = MetadataF32(projector, "clip.audio.attention.layer_norm_epsilon");
    if (ac.output != tc.hidden) {
      throw std::runtime_error("Qwen3-Omni audio projection width " +
                               std::to_string(ac.output) +
                               " does not match the thinker width " +
                               std::to_string(tc.hidden));
    }
    if (ac.mel_bins != 128U || ac.hidden % ac.heads != 0U ||
        ac.HeadDim() > 64U) {
      throw std::runtime_error("Qwen3-Omni audio tower geometry unsupported");
    }

    // Audio tower, F16 matrices with F32 biases and norms.
    const std::uint64_t h = ac.hidden;
    const std::uint64_t c = ac.conv_channels;
    AudioWeights& aw = model->audio_weights_;
    aw.conv1_weight = RequireConvolution(projector, "a.conv2d.1.weight", 1, c);
    aw.conv1_bias = Require(projector, "a.conv2d.1.bias", {1, 1, c}, kF32);
    aw.conv2_weight = RequireConvolution(projector, "a.conv2d.2.weight", c, c);
    aw.conv2_bias = Require(projector, "a.conv2d.2.bias", {1, 1, c}, kF32);
    aw.conv3_weight = RequireConvolution(projector, "a.conv2d.3.weight", c, c);
    aw.conv3_bias = Require(projector, "a.conv2d.3.bias", {1, 1, c}, kF32);
    aw.conv_out = Require(projector, "a.conv_out.weight", {c * 16U, h}, kF16);
    aw.layers.resize(ac.layers);
    for (std::size_t layer = 0; layer < ac.layers; ++layer) {
      AudioLayer& l = aw.layers[layer];
      l.ln1_weight = Require(projector, AudioBlk(layer, "ln1.weight"), {h}, kF32);
      l.ln1_bias = Require(projector, AudioBlk(layer, "ln1.bias"), {h}, kF32);
      l.q_weight = Require(projector, AudioBlk(layer, "attn_q.weight"), {h, h}, kF16);
      l.q_bias = Require(projector, AudioBlk(layer, "attn_q.bias"), {h}, kF32);
      l.k_weight = Require(projector, AudioBlk(layer, "attn_k.weight"), {h, h}, kF16);
      l.k_bias = Require(projector, AudioBlk(layer, "attn_k.bias"), {h}, kF32);
      l.v_weight = Require(projector, AudioBlk(layer, "attn_v.weight"), {h, h}, kF16);
      l.v_bias = Require(projector, AudioBlk(layer, "attn_v.bias"), {h}, kF32);
      l.out_weight =
          Require(projector, AudioBlk(layer, "attn_out.weight"), {h, h}, kF16);
      l.out_bias = Require(projector, AudioBlk(layer, "attn_out.bias"), {h}, kF32);
      l.ln2_weight = Require(projector, AudioBlk(layer, "ln2.weight"), {h}, kF32);
      l.ln2_bias = Require(projector, AudioBlk(layer, "ln2.bias"), {h}, kF32);
      l.fc1_weight = Require(projector, AudioBlk(layer, "ffn_up.weight"),
                             {h, ac.ffn}, kF16);
      l.fc1_bias = Require(projector, AudioBlk(layer, "ffn_up.bias"), {ac.ffn}, kF32);
      l.fc2_weight = Require(projector, AudioBlk(layer, "ffn_down.weight"),
                             {ac.ffn, h}, kF16);
      l.fc2_bias = Require(projector, AudioBlk(layer, "ffn_down.bias"), {h}, kF32);
    }
    aw.post_ln_weight = Require(projector, "a.post_ln.weight", {h}, kF32);
    aw.post_ln_bias = Require(projector, "a.post_ln.bias", {h}, kF32);
    aw.proj1_weight = Require(projector, "mm.a.mlp.1.weight", {h, h}, kF16);
    aw.proj1_bias = Require(projector, "mm.a.mlp.1.bias", {h}, kF32);
    aw.proj2_weight =
        Require(projector, "mm.a.mlp.2.weight", {h, ac.output}, kF16);
    aw.proj2_bias = Require(projector, "mm.a.mlp.2.bias", {ac.output}, kF32);

    // Thinker, K-quant matrices with F32 norms and router.
    const std::uint64_t d = tc.hidden;
    TextWeights& tw = model->text_weights_;
    tw.token_embd = Require(text, "token_embd.weight", {d, tc.vocab}, kQuantized);
    tw.output_norm = Require(text, "output_norm.weight", {d}, kF32);
    tw.output = Require(text, "output.weight", {d, tc.vocab}, kQuantized);
    tw.layers.resize(tc.layers);
    for (std::size_t layer = 0; layer < tc.layers; ++layer) {
      TextLayer& l = tw.layers[layer];
      l.attn_norm = Require(text, Blk(layer, "attn_norm.weight"), {d}, kF32);
      l.q = Require(text, Blk(layer, "attn_q.weight"), {d, tc.QDim()}, kQuantized);
      l.k = Require(text, Blk(layer, "attn_k.weight"), {d, tc.KvDim()},
                    kQuantized);
      l.v = Require(text, Blk(layer, "attn_v.weight"), {d, tc.KvDim()},
                    kQuantized);
      l.o = Require(text, Blk(layer, "attn_output.weight"), {tc.QDim(), d},
                    kQuantized);
      l.q_norm = Require(text, Blk(layer, "attn_q_norm.weight"), {tc.head_dim},
                         kF32);
      l.k_norm = Require(text, Blk(layer, "attn_k_norm.weight"), {tc.head_dim},
                         kF32);
      l.ffn_norm = Require(text, Blk(layer, "ffn_norm.weight"), {d}, kF32);
      l.router = Require(text, Blk(layer, "ffn_gate_inp.weight"),
                         {d, tc.experts}, kF32);
      l.gate_exps = RequireExperts(text, Blk(layer, "ffn_gate_exps.weight"), d,
                                   tc.expert_ff, tc.experts);
      l.up_exps = RequireExperts(text, Blk(layer, "ffn_up_exps.weight"), d,
                                 tc.expert_ff, tc.experts);
      l.down_exps = RequireExperts(text, Blk(layer, "ffn_down_exps.weight"),
                                   tc.expert_ff, d, tc.experts);
    }
  } catch (const std::exception& exception) {
    if (error != nullptr) {
      *error = exception.what();
    }
    return nullptr;
  }
  return model;
}

}  // namespace gufo::models::qwen3_omni
