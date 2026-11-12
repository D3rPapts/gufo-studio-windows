#include "src/models/moss_music/model.hpp"

#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string_view>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::moss_music {

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

/// The tensor `name` with GGUF dimensions `shape` (innermost first) and one
/// of `types`, as a row view.
Weight Require(const core::GgufReader& reader, std::string_view name,
               std::initializer_list<std::uint64_t> shape,
               std::initializer_list<GgmlType> types) {
  const core::GgufTensorInfo* info = reader.FindTensor(name);
  if (info == nullptr) {
    throw std::runtime_error("MOSS-Music tensor missing: " + std::string(name));
  }
  bool shape_ok = info->dimensions.size() == shape.size();
  if (shape_ok) {
    std::size_t index = 0;
    for (const std::uint64_t dimension : shape) {
      shape_ok = shape_ok && info->dimensions[index++] == dimension;
    }
  }
  if (!shape_ok) {
    throw std::runtime_error("MOSS-Music tensor " + std::string(name) +
                             " does not have shape " + ShapeText(shape));
  }
  bool type_ok = false;
  for (const GgmlType type : types) {
    type_ok = type_ok || info->type == type;
  }
  if (!type_ok) {
    throw std::runtime_error("MOSS-Music tensor " + std::string(name) +
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
  weight.row_bytes =
      quant::EncodedSizeBytes(info->type, static_cast<std::size_t>(weight.cols));
  if (weight.row_bytes == 0 || weight.row_bytes * outer != info->size_bytes) {
    throw std::runtime_error("MOSS-Music tensor " + std::string(name) +
                             " has an unexpected byte size");
  }
  return weight;
}

Weight Matrix(const core::GgufReader& reader, std::string_view name,
              std::uint64_t cols, std::uint64_t rows) {
  return Require(reader, name, {cols, rows}, {GgmlType::kBF16, GgmlType::kF16});
}

Weight Vector(const core::GgufReader& reader, std::string_view name,
              std::uint64_t size) {
  return Require(reader, name, {size}, {GgmlType::kF32});
}

/// A 3x3 kernel [cout, cin, 3, 3] (GGUF dims [3, 3, cin, cout]) viewed as
/// cout rows of cin * 9 elements in (channel, ky, kx) order.
Weight Convolution(const core::GgufReader& reader, std::string_view name,
                   std::uint64_t cin, std::uint64_t cout) {
  Weight weight =
      Require(reader, name, {3, 3, cin, cout}, {GgmlType::kBF16, GgmlType::kF16});
  weight.cols = cin * 9U;
  weight.rows = cout;
  weight.row_bytes =
      quant::EncodedSizeBytes(weight.type, static_cast<std::size_t>(weight.cols));
  return weight;
}

GatedMlpWeights LoadGatedMlp(const core::GgufReader& reader,
                             const std::string& prefix,
                             const AdapterConfig& adapter) {
  GatedMlpWeights mlp;
  mlp.gate = Matrix(reader, prefix + ".gate_proj.weight", adapter.input,
                    adapter.hidden);
  mlp.up =
      Matrix(reader, prefix + ".up_proj.weight", adapter.input, adapter.hidden);
  mlp.down = Matrix(reader, prefix + ".down_proj.weight", adapter.hidden,
                    adapter.output);
  return mlp;
}

std::uint32_t MetadataU32(const core::GgufReader& reader, std::string_view key,
                          std::uint32_t fallback) {
  return reader.GetMetadataUint32(key).value_or(fallback);
}

}  // namespace

void DecodeRow(const Weight& weight, std::uint64_t row, float* out) {
  const std::size_t cols = static_cast<std::size_t>(weight.cols);
  const void* source = weight.Row(row);
  switch (weight.type) {
    case GgmlType::kF32:
      std::memcpy(out, source, cols * sizeof(float));
      return;
    case GgmlType::kBF16: {
      const auto* bits = static_cast<const std::uint16_t*>(source);
      for (std::size_t index = 0; index < cols; ++index) {
        const std::uint32_t widened = static_cast<std::uint32_t>(bits[index])
                                      << 16;
        float value;
        std::memcpy(&value, &widened, sizeof(value));
        out[index] = value;
      }
      return;
    }
    case GgmlType::kF16: {
      const auto* half = static_cast<const std::uint16_t*>(source);
      for (std::size_t index = 0; index < cols; ++index) {
        out[index] = quant::Fp16ToFloat(half[index]);
      }
      return;
    }
    default:
      throw std::runtime_error("MOSS-Music: unsupported weight type");
  }
}

std::unique_ptr<AudioModel> AudioModel::Load(
    const std::filesystem::path& audio_gguf, std::string* error) {
  std::unique_ptr<AudioModel> model(new AudioModel());
  try {
    std::string open_error;
    model->reader_ = core::GgufReader::OpenFile(audio_gguf, &open_error);
    if (model->reader_ == nullptr) {
      throw std::runtime_error("cannot open " + audio_gguf.string() + ": " +
                               open_error);
    }
    const core::GgufReader& reader = *model->reader_;
    const auto arch = reader.GetMetadataString("general.architecture");
    if (!arch.has_value() || *arch != "moss-music") {
      throw std::runtime_error(audio_gguf.string() +
                               " is not a MOSS-Music audio GGUF");
    }
    ModelConfig& config = model->config_;
    config.audio.mel_bins =
        MetadataU32(reader, "moss-music.audio.mel_bins", config.audio.mel_bins);
    config.audio.d_model =
        MetadataU32(reader, "moss-music.audio.d_model", config.audio.d_model);
    config.audio.layers =
        MetadataU32(reader, "moss-music.audio.layers", config.audio.layers);
    config.audio.heads =
        MetadataU32(reader, "moss-music.audio.heads", config.audio.heads);
    config.audio.ffn = MetadataU32(reader, "moss-music.audio.ffn", config.audio.ffn);
    config.audio.conv_channels = MetadataU32(
        reader, "moss-music.audio.conv_channels", config.audio.conv_channels);
    config.audio.n_window =
        MetadataU32(reader, "moss-music.audio.n_window", config.audio.n_window);
    config.adapter.hidden =
        MetadataU32(reader, "moss-music.adapter.hidden", config.adapter.hidden);
    config.adapter.input = config.audio.d_model;
    config.adapter.output =
        MetadataU32(reader, "moss-music.text.hidden", config.text.hidden);
    config.text.hidden = config.adapter.output;
    if (config.audio.d_model % config.audio.heads != 0) {
      throw std::runtime_error("MOSS-Music: d_model is not a multiple of heads");
    }

    const AudioEncoderConfig& audio = config.audio;
    const std::uint64_t d = audio.d_model;
    const std::uint64_t c = audio.conv_channels;
    AudioWeights& weights = model->audio_;
    weights.stem.conv1_weight = Convolution(reader, "audio_encoder.conv1.weight", 1, c);
    weights.stem.conv1_bias = Vector(reader, "audio_encoder.conv1.bias", c);
    weights.stem.conv2_weight = Convolution(reader, "audio_encoder.conv2.weight", c, c);
    weights.stem.conv2_bias = Vector(reader, "audio_encoder.conv2.bias", c);
    weights.stem.conv3_weight = Convolution(reader, "audio_encoder.conv3.weight", c, c);
    weights.stem.conv3_bias = Vector(reader, "audio_encoder.conv3.bias", c);
    weights.stem.proj_weight = Matrix(reader, "audio_encoder.stem_proj.weight",
                                      audio.StemFlattened(), d);
    weights.stem.proj_bias = Vector(reader, "audio_encoder.stem_proj.bias", d);
    weights.layers.resize(audio.layers);
    for (std::uint32_t index = 0; index < audio.layers; ++index) {
      const std::string p = "audio_encoder.layers." + std::to_string(index);
      AudioLayerWeights& layer = weights.layers[index];
      layer.ln1_weight = Vector(reader, p + ".self_attn_layer_norm.weight", d);
      layer.ln1_bias = Vector(reader, p + ".self_attn_layer_norm.bias", d);
      layer.q_weight = Matrix(reader, p + ".self_attn.q_proj.weight", d, d);
      layer.q_bias = Vector(reader, p + ".self_attn.q_proj.bias", d);
      layer.k_weight = Matrix(reader, p + ".self_attn.k_proj.weight", d, d);
      layer.v_weight = Matrix(reader, p + ".self_attn.v_proj.weight", d, d);
      layer.v_bias = Vector(reader, p + ".self_attn.v_proj.bias", d);
      layer.out_weight = Matrix(reader, p + ".self_attn.out_proj.weight", d, d);
      layer.out_bias = Vector(reader, p + ".self_attn.out_proj.bias", d);
      layer.ln2_weight = Vector(reader, p + ".final_layer_norm.weight", d);
      layer.ln2_bias = Vector(reader, p + ".final_layer_norm.bias", d);
      layer.fc1_weight = Matrix(reader, p + ".fc1.weight", d, audio.ffn);
      layer.fc1_bias = Vector(reader, p + ".fc1.bias", audio.ffn);
      layer.fc2_weight = Matrix(reader, p + ".fc2.weight", audio.ffn, d);
      layer.fc2_bias = Vector(reader, p + ".fc2.bias", d);
    }
    weights.post_ln_weight = Vector(reader, "audio_encoder.layer_norm.weight", d);
    weights.post_ln_bias = Vector(reader, "audio_encoder.layer_norm.bias", d);

    model->adapter_ = LoadGatedMlp(reader, "audio_adapter", config.adapter);
    for (std::size_t k = 0; k < model->mergers_.size(); ++k) {
      model->mergers_[k] = LoadGatedMlp(
          reader, "deepstack_audio_merger_list." + std::to_string(k),
          config.adapter);
    }
  } catch (const std::exception& failure) {
    if (error != nullptr) {
      *error = failure.what();
    }
    return nullptr;
  }
  return model;
}

}  // namespace gufo::models::moss_music
