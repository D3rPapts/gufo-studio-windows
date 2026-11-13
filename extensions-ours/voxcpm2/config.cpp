#include "voxcpm2/config.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <gufo/ext/json.hpp>

namespace gufo::models::voxcpm2 {
namespace {

namespace json = gufo::json;

void SetError(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

std::int32_t Int(const json::Value& object, const char* key, std::int32_t def) {
  const json::Value* value = object.find(key);
  return value != nullptr && value->is_number()
             ? static_cast<std::int32_t>(value->as_double())
             : def;
}

float Float(const json::Value& object, const char* key, float def) {
  const json::Value* value = object.find(key);
  return value != nullptr && value->is_number() ? static_cast<float>(value->as_double())
                                                : def;
}

bool Bool(const json::Value& object, const char* key, bool def) {
  const json::Value* value = object.find(key);
  return value != nullptr && value->is_bool() ? value->as_bool() : def;
}

std::vector<std::int32_t> Ints(const json::Value& object, const char* key,
                               std::vector<std::int32_t> def) {
  const json::Value* value = object.find(key);
  if (value == nullptr || !value->is_array()) {
    return def;
  }
  std::vector<std::int32_t> out;
  for (const auto& item : value->items()) {
    out.push_back(static_cast<std::int32_t>(item.as_double()));
  }
  return out;
}

std::vector<float> Floats(const json::Value& object, const char* key) {
  std::vector<float> out;
  const json::Value* value = object.find(key);
  if (value != nullptr && value->is_array()) {
    for (const auto& item : value->items()) {
      out.push_back(static_cast<float>(item.as_double()));
    }
  }
  return out;
}

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

}  // namespace

std::int32_t AudioVaeConfig::chunk_size() const {
  std::int32_t product = 1;
  for (const std::int32_t rate : encoder_rates) {
    product *= rate;
  }
  return product;
}

std::int32_t AudioVaeConfig::decode_chunk_size() const {
  std::int32_t product = 1;
  for (const std::int32_t rate : decoder_rates) {
    product *= rate;
  }
  return product;
}

const std::vector<float>& ModelConfig::rope_factors() const {
  return max_position_embeddings > original_max_position_embeddings ? rope_long_factor
                                                                    : rope_short_factor;
}

float ModelConfig::rope_scaling() const {
  const double scale = static_cast<double>(max_position_embeddings) /
                       static_cast<double>(original_max_position_embeddings);
  return static_cast<float>(std::sqrt(
      1.0 + std::log(scale) / std::log(static_cast<double>(original_max_position_embeddings))));
}

std::optional<ModelConfig> ParseModelConfig(const std::string& json_text,
                                            std::string* error) {
  json::Value root;
  try {
    root = json::parse(json_text);
  } catch (const std::exception& exception) {
    SetError(error, std::string("invalid VoxCPM2 config.json: ") + exception.what());
    return std::nullopt;
  }
  if (root.member_str("architecture") != "voxcpm2") {
    SetError(error, "config.json is not a VoxCPM2 config (architecture != voxcpm2)");
    return std::nullopt;
  }
  const json::Value* lm = root.find("lm_config");
  const json::Value* enc = root.find("encoder_config");
  const json::Value* dit = root.find("dit_config");
  if (lm == nullptr || enc == nullptr || dit == nullptr) {
    SetError(error, "VoxCPM2 config.json lacks lm_config/encoder_config/dit_config");
    return std::nullopt;
  }
  ModelConfig config;
  config.base_lm.hidden_size = Int(*lm, "hidden_size", 2048);
  config.base_lm.intermediate_size = Int(*lm, "intermediate_size", 6144);
  config.base_lm.num_layers = Int(*lm, "num_hidden_layers", 28);
  config.base_lm.num_heads = Int(*lm, "num_attention_heads", 16);
  config.base_lm.num_kv_heads = Int(*lm, "num_key_value_heads", 2);
  config.base_lm.head_dim =
      Int(*lm, "kv_channels", config.base_lm.hidden_size / config.base_lm.num_heads);
  config.vocab_size = Int(*lm, "vocab_size", 73448);
  config.rms_norm_eps = Float(*lm, "rms_norm_eps", 1e-5F);
  config.rope_theta = Float(*lm, "rope_theta", 10000.0F);
  config.max_position_embeddings = Int(*lm, "max_position_embeddings", 32768);
  config.use_mup = Bool(*lm, "use_mup", true);
  config.scale_emb = Float(*lm, "scale_emb", 1.0F);
  config.scale_depth = Float(*lm, "scale_depth", 1.0F);
  if (const json::Value* scaling = lm->find("rope_scaling"); scaling != nullptr) {
    config.rope_short_factor = Floats(*scaling, "short_factor");
    config.rope_long_factor = Floats(*scaling, "long_factor");
    config.original_max_position_embeddings =
        Int(*scaling, "original_max_position_embeddings", config.max_position_embeddings);
  }

  config.residual_lm = config.base_lm;
  config.residual_lm.num_layers = Int(root, "residual_lm_num_layers", 8);
  config.residual_lm.rope = !Bool(root, "residual_lm_no_rope", false);

  config.encoder = config.base_lm;
  config.encoder.hidden_size = Int(*enc, "hidden_dim", 1024);
  config.encoder.intermediate_size = Int(*enc, "ffn_dim", 4096);
  config.encoder.num_heads = Int(*enc, "num_heads", 16);
  config.encoder.num_layers = Int(*enc, "num_layers", 4);
  config.encoder.head_dim =
      Int(*enc, "kv_channels", config.encoder.hidden_size / config.encoder.num_heads);

  config.dit = config.base_lm;
  config.dit.hidden_size = Int(*dit, "hidden_dim", 1024);
  config.dit.intermediate_size = Int(*dit, "ffn_dim", 4096);
  config.dit.num_heads = Int(*dit, "num_heads", 16);
  config.dit.num_layers = Int(*dit, "num_layers", 4);
  config.dit.head_dim = Int(*dit, "kv_channels", config.dit.hidden_size / config.dit.num_heads);
  config.dit_mean_mode = Bool(*dit, "mean_mode", false);
  if (const json::Value* cfm = dit->find("cfm_config"); cfm != nullptr) {
    config.sigma_min = Float(*cfm, "sigma_min", 1e-6F);
    config.inference_cfg_rate = Float(*cfm, "inference_cfg_rate", 1.0F);
    if (cfm->member_str("solver", "euler") != "euler") {
      SetError(error, "VoxCPM2: only the euler CFM solver is supported");
      return std::nullopt;
    }
  }

  config.patch_size = Int(root, "patch_size", 4);
  config.feat_dim = Int(root, "feat_dim", 64);
  config.fsq_latent_dim = Int(root, "scalar_quantization_latent_dim", 512);
  config.fsq_scale = Int(root, "scalar_quantization_scale", 9);
  config.max_length = Int(root, "max_length", 8192);
  if (const json::Value* vae = root.find("audio_vae_config"); vae != nullptr) {
    config.vae.encoder_dim = Int(*vae, "encoder_dim", 128);
    config.vae.encoder_rates = Ints(*vae, "encoder_rates", config.vae.encoder_rates);
    config.vae.latent_dim = Int(*vae, "latent_dim", 64);
    config.vae.decoder_dim = Int(*vae, "decoder_dim", 2048);
    config.vae.decoder_rates = Ints(*vae, "decoder_rates", config.vae.decoder_rates);
    config.vae.sr_bin_boundaries =
        Ints(*vae, "sr_bin_boundaries", config.vae.sr_bin_boundaries);
    config.vae.sample_rate = Int(*vae, "sample_rate", 16000);
    config.vae.out_sample_rate = Int(*vae, "out_sample_rate", 48000);
    if (Bool(*vae, "use_noise_block", false) || !Bool(*vae, "depthwise", true) ||
        vae->member_str("cond_type", "scale_bias") != "scale_bias" ||
        Bool(*vae, "cond_out_layer", false)) {
      SetError(error, "VoxCPM2: unsupported AudioVAE variant (noise block, "
                      "non-depthwise or non scale_bias conditioning)");
      return std::nullopt;
    }
  }
  if (config.feat_dim != config.vae.latent_dim) {
    SetError(error, "VoxCPM2: feat_dim must equal the VAE latent_dim");
    return std::nullopt;
  }
  if (config.base_lm.rope &&
      static_cast<std::int32_t>(config.rope_factors().size()) * 2 != config.base_lm.head_dim) {
    SetError(error, "VoxCPM2: LongRoPE factors do not match head_dim");
    return std::nullopt;
  }
  if (config.encoder.head_dim != config.base_lm.head_dim ||
      config.dit.head_dim != config.base_lm.head_dim) {
    SetError(error, "VoxCPM2: all stacks must share one head_dim (one RoPE table)");
    return std::nullopt;
  }
  return config;
}

std::optional<ModelConfig> LoadModelConfigFromPath(const std::string& model_dir,
                                                   std::string* error) {
  const std::filesystem::path path = std::filesystem::path(model_dir) / "config.json";
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec)) {
    SetError(error, "missing " + path.string());
    return std::nullopt;
  }
  return ParseModelConfig(ReadFile(path), error);
}

bool LooksLikeVoxCpm2(const std::string& model_dir) {
  const std::filesystem::path root(model_dir);
  std::error_code ec;
  if (!std::filesystem::is_regular_file(root / "model.safetensors", ec) ||
      !std::filesystem::is_regular_file(root / "tokenizer.json", ec) ||
      (!std::filesystem::is_regular_file(root / "audiovae.pth", ec) &&
       !std::filesystem::is_regular_file(root / "audiovae.safetensors", ec))) {
    return false;
  }
  return LoadModelConfigFromPath(model_dir).has_value();
}

}  // namespace gufo::models::voxcpm2
