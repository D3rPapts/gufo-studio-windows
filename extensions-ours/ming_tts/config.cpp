#include "ming_tts/config.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <gufo/ext/json.hpp>

namespace gufo::models::ming_tts {
namespace {

namespace json = gufo::json;

void SetError(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

bool ReadInt(const json::Value& object, const char* key, std::int32_t* out) {
  const auto* value = object.find(key);
  if (value == nullptr || !value->is_number()) {
    return false;
  }
  const double number = value->as_double();
  if (!std::isfinite(number) || std::floor(number) != number) {
    return false;
  }
  *out = static_cast<std::int32_t>(number);
  return true;
}

bool ReadFloat(const json::Value& object, const char* key, float* out) {
  const auto* value = object.find(key);
  if (value == nullptr || !value->is_number()) {
    return false;
  }
  *out = static_cast<float>(value->as_double());
  return std::isfinite(*out);
}

bool ParseQwen2(const json::Value& object, Qwen2Config* out) {
  if (!object.is_object()) {
    return false;
  }
  const bool ok = ReadInt(object, "hidden_size", &out->hidden_size) &&
                  ReadInt(object, "intermediate_size", &out->intermediate_size) &&
                  ReadInt(object, "num_hidden_layers", &out->num_layers) &&
                  ReadInt(object, "num_attention_heads", &out->num_heads) &&
                  ReadInt(object, "num_key_value_heads", &out->num_kv_heads) &&
                  ReadFloat(object, "rms_norm_eps", &out->rms_norm_eps) &&
                  ReadFloat(object, "rope_theta", &out->rope_theta);
  if (!ok || out->num_heads <= 0 || out->num_kv_heads <= 0 ||
      out->hidden_size % out->num_heads != 0 ||
      out->num_heads % out->num_kv_heads != 0) {
    return false;
  }
  out->head_dim = out->hidden_size / out->num_heads;
  (void)ReadInt(object, "vocab_size", &out->vocab_size);
  // transformers 4.46 (the reference) builds the SDPA mask from
  // config.sliding_window whenever it is set, independent of
  // use_sliding_window; it only matters once a sequence exceeds the window.
  std::int32_t window = 0;
  if (ReadInt(object, "sliding_window", &window) && window > 0) {
    out->sliding_window = window;
  }
  return true;
}

bool ParseDit(const json::Value& object, DitConfig* out) {
  if (!object.is_object()) {
    return false;
  }
  const bool ok = ReadInt(object, "hidden_size", &out->hidden_size) &&
                  ReadInt(object, "depth", &out->depth) &&
                  ReadInt(object, "num_heads", &out->num_heads) &&
                  ReadInt(object, "mlp_ratio", &out->mlp_ratio);
  if (!ok || out->num_heads <= 0 || out->hidden_size % out->num_heads != 0) {
    return false;
  }
  (void)ReadInt(object, "patch_size", &out->patch_size);
  out->history_patch_size = out->patch_size;
  (void)ReadInt(object, "history_patch_size", &out->history_patch_size);
  return true;
}

}  // namespace

std::optional<ModelConfig> LoadModelConfigFromPath(const std::string& model_dir,
                                                   std::string* error) {
  const std::filesystem::path path =
      std::filesystem::path(model_dir) / "config.json";
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    SetError(error, "cannot open " + path.string());
    return std::nullopt;
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  json::Value root;
  try {
    root = json::parse(buffer.str());
  } catch (const std::exception& exception) {
    SetError(error, std::string("invalid config.json: ") + exception.what());
    return std::nullopt;
  }
  if (!root.is_object()) {
    SetError(error, "config.json root must be an object");
    return std::nullopt;
  }
  const auto* architectures = root.find("architectures");
  bool architecture_ok = false;
  if (architectures != nullptr && architectures->is_array()) {
    for (const auto& item : architectures->items()) {
      architecture_ok = architecture_ok ||
                        item.str() == "BailingMMNativeForConditionalGeneration";
    }
  }
  if (!architecture_ok || root.member_str("model_type") != "dense") {
    SetError(error, "not a Ming-3.0-tts dense checkpoint");
    return std::nullopt;
  }

  ModelConfig config;
  const auto* llm = root.find("llm_config");
  const auto* audio = root.find("audio_tokenizer_config");
  const auto* dit = root.find("ditar_config");
  const auto* aggregator = root.find("aggregator_config");
  if (llm == nullptr || !ParseQwen2(*llm, &config.llm) || audio == nullptr ||
      !audio->is_object() || dit == nullptr || !ParseDit(*dit, &config.dit) ||
      aggregator == nullptr || !ParseDit(*aggregator, &config.aggregator)) {
    SetError(error, "incomplete Ming-3.0-tts config.json");
    return std::nullopt;
  }
  // The backbone never exceeds the 512-position reference cache, far below
  // its 32768 window; keep attention global.
  config.llm.sliding_window = 0;

  const auto* dec = audio->find("dec_kwargs");
  const auto* backbone = dec != nullptr ? dec->find("backbone") : nullptr;
  if (backbone == nullptr || !ParseQwen2(*backbone, &config.vae_decoder) ||
      !ReadInt(*dec, "latent_dim", &config.latent_dim) ||
      !ReadInt(*dec, "output_dim", &config.vae_hop_size) ||
      !ReadInt(*audio, "patch_size", &config.vae_patch_size) ||
      !ReadInt(*audio, "sample_rate", &config.sample_rate)) {
    SetError(error, "incomplete Ming-3.0-tts audio_tokenizer_config");
    return std::nullopt;
  }
  const auto* enc = audio->find("enc_kwargs");
  const auto* enc_backbone = enc != nullptr ? enc->find("backbone") : nullptr;
  std::int32_t input_dim = 0;
  if (enc_backbone == nullptr || !ParseQwen2(*enc_backbone, &config.vae_encoder) ||
      !ReadInt(*enc, "input_dim", &input_dim) || input_dim != config.vae_hop_size) {
    SetError(error, "incomplete Ming-3.0-tts enc_kwargs");
    return std::nullopt;
  }
  if (config.dit.patch_size != 1 || config.dit.history_patch_size != 1 ||
      config.latent_dim <= 0 || config.vae_patch_size <= 0 ||
      config.vae_hop_size <= 0) {
    SetError(error, "unsupported Ming-3.0-tts patch layout");
    return std::nullopt;
  }
  return config;
}

bool LooksLikeMingTts(const std::string& model_dir) {
  const std::filesystem::path root(model_dir);
  std::error_code ec;
  if (!std::filesystem::is_regular_file(root / "model.safetensors", ec) ||
      !std::filesystem::is_regular_file(root / "campplus.onnx", ec)) {
    return false;
  }
  return LoadModelConfigFromPath(model_dir).has_value();
}

}  // namespace gufo::models::ming_tts
