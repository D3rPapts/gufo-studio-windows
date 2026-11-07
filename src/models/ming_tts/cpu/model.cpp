#include "src/models/ming_tts/cpu/model.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <filesystem>
#include <numbers>
#include <stdexcept>

#include "src/models/ming_tts/cpu/fft.hpp"

namespace gufo::models::ming_tts::cpu {
namespace {

Linear LoadLinear(const SafetensorsFile& file, const std::string& prefix,
                  bool require_bias) {
  const TensorView& weight = file.Require(prefix + ".weight");
  if (weight.dtype != DType::kBF16 || weight.shape.size() != 2) {
    throw std::runtime_error(prefix + ".weight must be a 2-D BF16 tensor");
  }
  Linear layer;
  layer.weight = reinterpret_cast<const std::uint16_t*>(weight.data);
  layer.out_features = weight.shape[0];
  layer.in_features = weight.shape[1];
  if (file.Find(prefix + ".bias") != nullptr) {
    layer.bias = file.ToF32(prefix + ".bias");
    if (static_cast<std::int64_t>(layer.bias.size()) != layer.out_features) {
      throw std::runtime_error(prefix + ".bias has the wrong length");
    }
  } else if (require_bias) {
    throw std::runtime_error("missing " + prefix + ".bias");
  }
  return layer;
}

std::vector<float> LoadVector(const SafetensorsFile& file,
                              const std::string& name, std::int64_t length) {
  std::vector<float> values = file.ToF32(name);
  if (static_cast<std::int64_t>(values.size()) != length) {
    throw std::runtime_error(name + " has the wrong length");
  }
  return values;
}

Qwen2Stack LoadQwen2(const SafetensorsFile& file, const std::string& prefix,
                     const Qwen2Config& config) {
  Qwen2Stack stack;
  stack.config = config;
  for (std::int32_t index = 0; index < config.num_layers; ++index) {
    const std::string p = prefix + "layers." + std::to_string(index) + ".";
    Qwen2Layer layer;
    layer.q = LoadLinear(file, p + "self_attn.q_proj", true);
    layer.k = LoadLinear(file, p + "self_attn.k_proj", true);
    layer.v = LoadLinear(file, p + "self_attn.v_proj", true);
    layer.o = LoadLinear(file, p + "self_attn.o_proj", false);
    layer.gate = LoadLinear(file, p + "mlp.gate_proj", false);
    layer.up = LoadLinear(file, p + "mlp.up_proj", false);
    layer.down = LoadLinear(file, p + "mlp.down_proj", false);
    layer.input_norm =
        LoadVector(file, p + "input_layernorm.weight", config.hidden_size);
    layer.post_norm = LoadVector(file, p + "post_attention_layernorm.weight",
                                 config.hidden_size);
    if (layer.q.out_features != config.num_heads * config.head_dim ||
        layer.k.out_features != config.num_kv_heads * config.head_dim ||
        layer.gate.out_features != config.intermediate_size) {
      throw std::runtime_error(p + " does not match the configuration");
    }
    stack.layers.push_back(std::move(layer));
  }
  stack.final_norm = LoadVector(file, prefix + "norm.weight", config.hidden_size);
  return stack;
}

DitStack LoadDit(const SafetensorsFile& file, const std::string& prefix,
                 const DitConfig& config) {
  DitStack stack;
  stack.config = config;
  for (std::int32_t index = 0; index < config.depth; ++index) {
    const std::string p = prefix + "blocks." + std::to_string(index) + ".";
    DitBlock block;
    block.q = LoadLinear(file, p + "attn.to_q", true);
    block.k = LoadLinear(file, p + "attn.to_k", true);
    block.v = LoadLinear(file, p + "attn.to_v", true);
    block.out = LoadLinear(file, p + "attn.to_out.0", true);
    block.ff_in = LoadLinear(file, p + "mlp.ff.0.0", true);
    block.ff_out = LoadLinear(file, p + "mlp.ff.2", true);
    block.norm1 = LoadVector(file, p + "norm1.weight", config.hidden_size);
    block.norm2 = LoadVector(file, p + "norm2.weight", config.hidden_size);
    stack.blocks.push_back(std::move(block));
  }
  const std::int32_t head_dim = config.hidden_size / config.num_heads;
  stack.inv_freq =
      LoadVector(file, prefix + "rotary_embed.inv_freq", head_dim / 2);
  stack.final_norm =
      LoadVector(file, prefix + "final_layer.norm_final.weight",
                 config.hidden_size);
  stack.final_linear = LoadLinear(file, prefix + "final_layer.linear", true);
  return stack;
}

float SiLU(float x) { return x / (1.0F + std::exp(-x)); }

}  // namespace

std::unique_ptr<Model> Model::Load(const std::string& model_dir,
                                   std::string* error) {
  auto set_error = [error](const std::string& message) {
    if (error != nullptr) {
      *error = message;
    }
  };
  auto config = LoadModelConfigFromPath(model_dir, error);
  if (!config.has_value()) {
    return nullptr;
  }
  auto model = std::unique_ptr<Model>(new Model());
  model->config_ = *config;
  model->weights_ = SafetensorsFile::Open(
      std::filesystem::path(model_dir) / "model.safetensors", error);
  if (model->weights_ == nullptr) {
    return nullptr;
  }
  try {
    const SafetensorsFile& file = *model->weights_;
    const ModelConfig& c = model->config_;
    const TensorView& embed = file.Require("model.model.embed_tokens.weight");
    if (embed.dtype != DType::kBF16 || embed.shape.size() != 2 ||
        embed.shape[1] != c.llm.hidden_size) {
      throw std::runtime_error("unexpected embed_tokens layout");
    }
    model->config_.llm.vocab_size = static_cast<std::int32_t>(embed.shape[0]);
    model->embed_tokens_ = reinterpret_cast<const std::uint16_t*>(embed.data);
    model->llm_ = LoadQwen2(file, "model.model.", c.llm);
    model->spk_head_ = LoadLinear(file, "spk_head", true);
    model->stop_head_ = LoadLinear(file, "stop_head", true);

    FlowHead& flow = model->flow_;
    flow.stack = LoadDit(file, "flowloss.cfm.model.", c.dit);
    flow.time_mlp0 = LoadLinear(file, "flowloss.cfm.model.t_embedder.time_mlp.0", true);
    flow.time_mlp2 = LoadLinear(file, "flowloss.cfm.model.t_embedder.time_mlp.2", true);
    flow.x_embedder = LoadLinear(file, "flowloss.cfm.model.x_embedder", true);
    flow.c_embedder =
        LoadLinear(file, "flowloss.cfm.model.c_embedder.cond_embedder", true);

    Aggregator& aggregator = model->aggregator_;
    aggregator.stack = LoadDit(file, "linear_proj_audio.", c.aggregator);
    aggregator.x_embedder = LoadLinear(file, "linear_proj_audio.x_embedder", true);
    aggregator.cls = LoadVector(file, "linear_proj_audio.word_embedder.weight",
                                c.aggregator.hidden_size);

    VaeDecoder& vae = model->vae_;
    vae.stack = LoadQwen2(file, "audio.decoder.decoder.", c.vae_decoder);
    vae.fc1 = LoadLinear(file, "audio.decoder.fc1", true);
    vae.head_out = LoadLinear(file, "audio.decoder.head.out", true);
    vae.window = LoadVector(file, "audio.decoder.head.istft.window",
                            static_cast<std::int64_t>(c.vae_hop_size) * 4);
    if (vae.head_out.out_features != c.vae_hop_size * 4 + 2) {
      throw std::runtime_error("unexpected ISTFT head width");
    }
    VaeEncoder& encoder = model->encoder_;
    encoder.backbone = LoadQwen2(file, "audio.encoder.encoder.", c.vae_encoder);
    Qwen2Config aggregator_config = c.vae_encoder;
    aggregator_config.num_layers = c.vae_encoder_aggregator_layers;
    encoder.aggregator = LoadQwen2(file, "audio.encoder.aggregator.", aggregator_config);
    encoder.fc1 = LoadLinear(file, "audio.encoder.fc1", false);
    encoder.fc2 = LoadLinear(file, "audio.encoder.fc2", true);
    encoder.fc3 = LoadLinear(file, "audio.encoder.fc3", true);
    encoder.cls = LoadVector(file, "audio.encoder.cls_embed", c.vae_encoder.hidden_size);
  } catch (const std::exception& exception) {
    set_error(std::string("Ming-3.0-tts weights: ") + exception.what());
    return nullptr;
  }
  model->ResetCache();
  return model;
}

void Model::Embed(std::span<const std::int64_t> ids, float* out) const {
  const std::int64_t hidden = config_.llm.hidden_size;
  for (std::size_t row = 0; row < ids.size(); ++row) {
    const std::int64_t id = ids[row];
    if (id < 0 || id >= config_.llm.vocab_size) {
      throw std::out_of_range("token id outside the vocabulary");
    }
    const std::uint16_t* src = embed_tokens_ + id * hidden;
    for (std::int64_t i = 0; i < hidden; ++i) {
      out[static_cast<std::int64_t>(row) * hidden + i] = Bf16ToF32(src[i]);
    }
  }
}

void Model::ProjectSpeaker(std::span<const float> embedding, float* out) const {
  if (static_cast<std::int64_t>(embedding.size()) != spk_head_.in_features) {
    throw std::invalid_argument("speaker embedding must have 192 values");
  }
  LinearForward(spk_head_, embedding.data(), 1, out);
}

void Model::ResetCache() {
  cache_k_.assign(static_cast<std::size_t>(llm_.config.num_layers), {});
  cache_v_.assign(static_cast<std::size_t>(llm_.config.num_layers), {});
  cache_length_ = 0;
}

void Model::Qwen2Forward(const Qwen2Stack& stack, float* hidden,
                         std::int64_t rows, std::int64_t first_position,
                         bool use_cache,
                         std::vector<std::vector<float>>* layer_outputs,
                         bool release_pages) const {
  const Qwen2Config& c = stack.config;
  const std::int64_t h = c.hidden_size;
  const std::int64_t q_width = static_cast<std::int64_t>(c.num_heads) * c.head_dim;
  const std::int64_t kv_width =
      static_cast<std::int64_t>(c.num_kv_heads) * c.head_dim;
  std::vector<float> normed(static_cast<std::size_t>(rows * h));
  std::vector<float> q(static_cast<std::size_t>(rows * q_width));
  std::vector<float> k(static_cast<std::size_t>(rows * kv_width));
  std::vector<float> v(static_cast<std::size_t>(rows * kv_width));
  std::vector<float> attn(static_cast<std::size_t>(rows * q_width));
  std::vector<float> projected(static_cast<std::size_t>(rows * h));
  std::vector<float> gate(static_cast<std::size_t>(rows * c.intermediate_size));
  std::vector<float> up(static_cast<std::size_t>(rows * c.intermediate_size));
  for (std::size_t index = 0; index < stack.layers.size(); ++index) {
    const Qwen2Layer& layer = stack.layers[index];
    RmsNormTo(hidden, normed.data(), rows, h, layer.input_norm, c.rms_norm_eps);
    LinearForward(layer.q, normed.data(), rows, q.data());
    LinearForward(layer.k, normed.data(), rows, k.data());
    LinearForward(layer.v, normed.data(), rows, v.data());
    RopeHalf(q.data(), rows, c.num_heads, c.head_dim, first_position,
             c.rope_theta);
    RopeHalf(k.data(), rows, c.num_kv_heads, c.head_dim, first_position,
             c.rope_theta);
    const float* keys = k.data();
    const float* values = v.data();
    std::int64_t kv_len = rows;
    std::int64_t q_offset = 0;
    if (use_cache) {
      auto& ck = cache_k_[index];
      auto& cv = cache_v_[index];
      ck.insert(ck.end(), k.begin(), k.end());
      cv.insert(cv.end(), v.begin(), v.end());
      keys = ck.data();
      values = cv.data();
      kv_len = static_cast<std::int64_t>(ck.size()) / kv_width;
      q_offset = first_position;
    }
    Attention(q.data(), keys, values, attn.data(), rows, kv_len, c.num_heads,
              c.num_kv_heads, c.head_dim, /*causal=*/true, q_offset,
              c.sliding_window);
    LinearForward(layer.o, attn.data(), rows, projected.data());
    for (std::int64_t i = 0; i < rows * h; ++i) {
      hidden[i] += projected[static_cast<std::size_t>(i)];
    }
    RmsNormTo(hidden, normed.data(), rows, h, layer.post_norm, c.rms_norm_eps);
    LinearForward(layer.gate, normed.data(), rows, gate.data());
    LinearForward(layer.up, normed.data(), rows, up.data());
    SiluMulInPlace(gate.data(), up.data(), gate.size());
    LinearForward(layer.down, gate.data(), rows, projected.data());
    for (std::int64_t i = 0; i < rows * h; ++i) {
      hidden[i] += projected[static_cast<std::size_t>(i)];
    }
    if (layer_outputs != nullptr) {
      layer_outputs->emplace_back(hidden, hidden + rows * h);
    }
    if (release_pages) {
      // A stack read once per call (the prompt encoder): keep only the
      // current layer resident, not the whole stack (#829).
      for (const Linear* linear : {&layer.q, &layer.k, &layer.v, &layer.o,
                                   &layer.gate, &layer.up, &layer.down}) {
        ReleaseHostPages(linear->weight,
                         static_cast<std::size_t>(linear->out_features *
                                                  linear->in_features) *
                             sizeof(std::uint16_t));
      }
    }
  }
}

void Model::BackboneForward(const float* embeddings, std::int64_t rows,
                            float* normed,
                            std::vector<std::vector<float>>* layer_outputs) {
  const std::int64_t h = llm_.config.hidden_size;
  std::vector<float> hidden(embeddings, embeddings + rows * h);
  Qwen2Forward(llm_, hidden.data(), rows, cache_length_, /*use_cache=*/true,
               layer_outputs);
  cache_length_ += rows;
  RmsNormTo(hidden.data(), normed, rows, h, llm_.final_norm,
            llm_.config.rms_norm_eps);
}

float Model::StopProbability(const float* z) const {
  float logits[2];
  LinearForward(stop_head_, z, 1, logits);
  return 1.0F / (1.0F + std::exp(logits[0] - logits[1]));
}

void Model::DitForward(const DitStack& stack, float* x, std::int64_t batch,
                       std::int64_t seq) const {
  const std::int64_t d = stack.config.hidden_size;
  const std::int64_t heads = stack.config.num_heads;
  const std::int64_t head_dim = d / heads;
  const std::int64_t rows = batch * seq;
  const std::int64_t inner = d * stack.config.mlp_ratio;
  std::vector<float> normed(static_cast<std::size_t>(rows * d));
  std::vector<float> q(normed.size()), k(normed.size()), v(normed.size());
  std::vector<float> attn(normed.size()), projected(normed.size());
  std::vector<float> ff(static_cast<std::size_t>(rows * inner));
  for (const DitBlock& block : stack.blocks) {
    RmsNormTo(x, normed.data(), rows, d, block.norm1, 1e-6F);
    LinearForward(block.q, normed.data(), rows, q.data());
    LinearForward(block.k, normed.data(), rows, k.data());
    LinearForward(block.v, normed.data(), rows, v.data());
    for (std::int64_t b = 0; b < batch; ++b) {
      const std::int64_t offset = b * seq * d;
      RopeInterleaved(q.data() + offset, seq, heads, head_dim, stack.inv_freq);
      RopeInterleaved(k.data() + offset, seq, heads, head_dim, stack.inv_freq);
      Attention(q.data() + offset, k.data() + offset, v.data() + offset,
                attn.data() + offset, seq, seq, heads, heads, head_dim,
                /*causal=*/false, 0, 0);
    }
    LinearForward(block.out, attn.data(), rows, projected.data());
    for (std::int64_t i = 0; i < rows * d; ++i) {
      x[i] += projected[static_cast<std::size_t>(i)];
    }
    RmsNormTo(x, normed.data(), rows, d, block.norm2, 1e-6F);
    LinearForward(block.ff_in, normed.data(), rows, ff.data());
    GeluTanhInPlace(ff.data(), ff.size());
    LinearForward(block.ff_out, ff.data(), rows, projected.data());
    for (std::int64_t i = 0; i < rows * d; ++i) {
      x[i] += projected[static_cast<std::size_t>(i)];
    }
  }
}

void Model::FlowVelocity(const float* x, float t, const float* cond,
                         const float* history, float* cond_uncond) const {
  const std::int64_t d = flow_.stack.config.hidden_size;
  const std::int64_t latent = config_.latent_dim;
  // SinusPositionEmbedding(256), scale 1000, in fp32 like the reference.
  constexpr std::int64_t kFreqDim = 256;
  constexpr std::int64_t kHalf = kFreqDim / 2;
  const auto step = static_cast<float>(std::log(10000.0) / (kHalf - 1));
  float sinus[kFreqDim];
  const float scaled_t = 1000.0F * t;
  for (std::int64_t i = 0; i < kHalf; ++i) {
    const float freq = std::exp(static_cast<float>(i) * -step);
    const float arg = scaled_t * freq;
    sinus[i] = std::sin(arg);
    sinus[i + kHalf] = std::cos(arg);
  }
  std::vector<float> t_hidden(static_cast<std::size_t>(d));
  std::vector<float> t_vec(static_cast<std::size_t>(d));
  LinearForward(flow_.time_mlp0, sinus, 1, t_hidden.data());
  for (float& value : t_hidden) {
    value = SiLU(value);
  }
  LinearForward(flow_.time_mlp2, t_hidden.data(), 1, t_vec.data());

  std::vector<float> c_cond(static_cast<std::size_t>(d));
  LinearForward(flow_.c_embedder, cond, 1, c_cond.data());
  std::vector<float> xs(static_cast<std::size_t>(2 * latent));
  std::memcpy(xs.data(), history, sizeof(float) * latent);
  std::memcpy(xs.data() + latent, x, sizeof(float) * latent);
  std::vector<float> x_emb(static_cast<std::size_t>(2 * d));
  LinearForward(flow_.x_embedder, xs.data(), 2, x_emb.data());

  // [batch 2, seq 3, d]: [t + c, history, x]; the unconditional branch uses
  // c = 0, i.e. only the c_embedder bias.
  std::vector<float> seq(static_cast<std::size_t>(2 * 3 * d));
  for (std::int64_t b = 0; b < 2; ++b) {
    float* base = seq.data() + b * 3 * d;
    for (std::int64_t i = 0; i < d; ++i) {
      const float c = b == 0 ? c_cond[static_cast<std::size_t>(i)]
                             : flow_.c_embedder.bias[static_cast<std::size_t>(i)];
      base[i] = t_vec[static_cast<std::size_t>(i)] + c;
    }
    std::memcpy(base + d, x_emb.data(), sizeof(float) * d);
    std::memcpy(base + 2 * d, x_emb.data() + d, sizeof(float) * d);
  }
  DitForward(flow_.stack, seq.data(), 2, 3);
  std::vector<float> last(static_cast<std::size_t>(2 * d));
  for (std::int64_t b = 0; b < 2; ++b) {
    RmsNormTo(seq.data() + (b * 3 + 2) * d, last.data() + b * d, 1, d,
              flow_.stack.final_norm, 1e-6F);
  }
  LinearForward(flow_.stack.final_linear, last.data(), 2, cond_uncond);
}

std::vector<float> Model::FlowTimesteps(std::int32_t steps) {
  static const std::vector<std::pair<int, std::vector<int>>> kEpss = {
      {5, {0, 2, 4, 8, 16, 32}},
      {6, {0, 2, 4, 6, 8, 16, 32}},
      {7, {0, 2, 4, 6, 8, 16, 24, 32}},
      {10, {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32}},
      {12, {0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32}},
      {16, {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32}},
  };
  std::vector<float> t;
  for (const auto& [count, table] : kEpss) {
    if (count == steps) {
      for (const int value : table) {
        t.push_back((1.0F / 32.0F) * static_cast<float>(value));
      }
    }
  }
  if (t.empty()) {
    for (std::int32_t i = 0; i <= steps; ++i) {
      t.push_back(static_cast<float>(i) / static_cast<float>(steps));
    }
  }
  // Sway sampling, coefficient -1: t + c * (cos(pi/2 t) - 1 + t).
  constexpr float kHalfPi = static_cast<float>(std::numbers::pi / 2.0);
  for (float& value : t) {
    value = value + -1.0F * (std::cos(kHalfPi * value) - 1.0F + value);
  }
  return t;
}

void Model::SampleLatent(const float* cond, const float* history,
                         const float* noise, const float* sde_noise,
                         const FlowOptions& options, float* latent) const {
  const std::int64_t n = config_.latent_dim;
  const std::vector<float> t = FlowTimesteps(options.steps);
  std::vector<float> y(noise, noise + n);
  std::vector<float> velocity(static_cast<std::size_t>(2 * n));
  for (std::int32_t s = 0; s < options.steps; ++s) {
    const float dt = t[static_cast<std::size_t>(s) + 1] - t[static_cast<std::size_t>(s)];
    FlowVelocity(y.data(), t[static_cast<std::size_t>(s)], cond, history,
                 velocity.data());
    for (std::int64_t i = 0; i < n; ++i) {
      const float pred = velocity[static_cast<std::size_t>(i)];
      const float null_pred = velocity[static_cast<std::size_t>(n + i)];
      y[static_cast<std::size_t>(i)] += (pred + (pred - null_pred) * options.cfg) * dt;
    }
    if (options.temperature > 0.0F && sde_noise != nullptr) {
      const float scale = options.sigma * std::sqrt(options.temperature) *
                          std::sqrt(std::fabs(dt));
      for (std::int64_t i = 0; i < n; ++i) {
        y[static_cast<std::size_t>(i)] += scale * sde_noise[s * n + i];
      }
    }
  }
  std::memcpy(latent, y.data(), sizeof(float) * n);
}

void Model::Aggregate(const float* latent, float* out) const {
  const std::int64_t d = aggregator_.stack.config.hidden_size;
  std::vector<float> seq(static_cast<std::size_t>(2 * d));
  std::memcpy(seq.data(), aggregator_.cls.data(), sizeof(float) * d);
  LinearForward(aggregator_.x_embedder, latent, 1, seq.data() + d);
  DitForward(aggregator_.stack, seq.data(), 1, 2);
  std::vector<float> normed(static_cast<std::size_t>(d));
  RmsNormTo(seq.data(), normed.data(), 1, d, aggregator_.stack.final_norm, 1e-6F);
  LinearForward(aggregator_.stack.final_linear, normed.data(), 1, out);
}

std::vector<float> Model::DecodeLatents(const float* latents,
                                        std::int64_t frames,
                                        VaeTrace* trace) const {
  const std::int64_t h = vae_.stack.config.hidden_size;
  const std::int64_t factor = config_.vae_patch_size;
  const std::int64_t hop = config_.vae_hop_size;
  const std::int64_t n_fft = hop * 4;
  const std::int64_t bins = n_fft / 2 + 1;
  std::vector<float> fc1(static_cast<std::size_t>(frames * h));
  LinearForward(vae_.fc1, latents, frames, fc1.data());

  // nn.Upsample(scale_factor, mode='linear', align_corners=False)
  const std::int64_t steps = frames * factor;
  std::vector<float> hidden(static_cast<std::size_t>(steps * h));
  const float scale = 1.0F / static_cast<float>(factor);
  for (std::int64_t o = 0; o < steps; ++o) {
    float source = scale * (static_cast<float>(o) + 0.5F) - 0.5F;
    source = std::max(source, 0.0F);
    const auto i0 = static_cast<std::int64_t>(source);
    const std::int64_t i1 = i0 + (i0 < frames - 1 ? 1 : 0);
    const float lambda1 = source - static_cast<float>(i0);
    const float lambda0 = 1.0F - lambda1;
    for (std::int64_t c = 0; c < h; ++c) {
      hidden[static_cast<std::size_t>(o * h + c)] =
          lambda0 * fc1[static_cast<std::size_t>(i0 * h + c)] +
          lambda1 * fc1[static_cast<std::size_t>(i1 * h + c)];
    }
  }
  if (trace != nullptr) {
    trace->fc1 = fc1;
    trace->upsampled = hidden;
  }
  std::vector<std::vector<float>> layers;
  Qwen2Forward(vae_.stack, hidden.data(), steps, 0, /*use_cache=*/false,
               trace != nullptr ? &layers : nullptr);
  RmsNorm(hidden.data(), steps, h, vae_.stack.final_norm,
          vae_.stack.config.rms_norm_eps);
  std::vector<float> head(static_cast<std::size_t>(steps * (n_fft + 2)));
  LinearForward(vae_.head_out, hidden.data(), steps, head.data());
  if (trace != nullptr) {
    trace->hidden_l0 = layers.empty() ? std::vector<float>{} : layers.front();
    trace->hidden_last = hidden;
    trace->head_out = head;
  }

  // ISTFT with "same" padding (vocos style): irfft * window, overlap-add,
  // trim (win - hop) / 2 on both sides, divide by the window envelope.
  const std::int64_t out_size = (steps - 1) * hop + n_fft;
  std::vector<double> signal(static_cast<std::size_t>(out_size), 0.0);
  std::vector<double> envelope(static_cast<std::size_t>(out_size), 0.0);
  const RealInverseFft irfft(static_cast<std::size_t>(n_fft));
  std::vector<std::complex<double>> spectrum(static_cast<std::size_t>(bins));
  std::vector<double> frame(static_cast<std::size_t>(n_fft));
  for (std::int64_t t = 0; t < steps; ++t) {
    const float* row = head.data() + t * (n_fft + 2);
    for (std::int64_t k = 0; k < bins; ++k) {
      float magnitude = std::exp(row[k]);
      magnitude = std::min(magnitude, 100.0F);
      const float phase = row[bins + k];
      spectrum[static_cast<std::size_t>(k)] = std::complex<double>(
          static_cast<double>(magnitude * std::cos(phase)),
          static_cast<double>(magnitude * std::sin(phase)));
    }
    irfft.Run(spectrum.data(), frame.data());
    for (std::int64_t n = 0; n < n_fft; ++n) {
      const double w = vae_.window[static_cast<std::size_t>(n)];
      signal[static_cast<std::size_t>(t * hop + n)] += frame[static_cast<std::size_t>(n)] * w;
      envelope[static_cast<std::size_t>(t * hop + n)] += w * w;
    }
  }
  const std::int64_t pad = (n_fft - hop) / 2;
  std::vector<float> audio(static_cast<std::size_t>(out_size - 2 * pad));
  for (std::size_t i = 0; i < audio.size(); ++i) {
    audio[i] = static_cast<float>(signal[i + static_cast<std::size_t>(pad)] /
                                  envelope[i + static_cast<std::size_t>(pad)]);
  }
  return audio;
}

std::vector<float> Model::EncodePosterior(const float* wave,
                                          std::int64_t samples,
                                          std::int64_t* patches) const {
  const std::int64_t hop = config_.vae_hop_size;
  const std::int64_t patch = config_.vae_patch_size;
  const std::int64_t h = encoder_.backbone.config.hidden_size;
  // get_frames: ceil(len / hop) frames of `hop` samples, zero padded.
  const std::int64_t frames = (samples + hop - 1) / hop;
  std::vector<float> framed(static_cast<std::size_t>(frames * hop), 0.0F);
  std::copy(wave, wave + samples, framed.begin());
  std::vector<float> x1(static_cast<std::size_t>(frames * h));
  std::vector<float> x2(static_cast<std::size_t>(frames * h));
  LinearForward(encoder_.fc1, framed.data(), frames, x1.data());
  LinearForward(encoder_.fc2, x1.data(), frames, x2.data());
  Qwen2Forward(encoder_.backbone, x2.data(), frames, 0, /*use_cache=*/false,
               nullptr, /*release_pages=*/true);
  RmsNorm(x2.data(), frames, h, encoder_.backbone.final_norm,
          encoder_.backbone.config.rms_norm_eps);
  // pad_patch_insert_cls: groups of `patch` frames (zero padded) + CLS.
  *patches = (frames + patch - 1) / patch;
  const std::int64_t group = patch + 1;
  std::vector<float> seq(static_cast<std::size_t>(*patches * group * h), 0.0F);
  for (std::int64_t p = 0; p < *patches; ++p) {
    for (std::int64_t f = 0; f < patch; ++f) {
      const std::int64_t frame = p * patch + f;
      if (frame < frames) {
        std::copy(x2.begin() + frame * h, x2.begin() + (frame + 1) * h,
                  seq.begin() + (p * group + f) * h);
      }
    }
    std::copy(encoder_.cls.begin(), encoder_.cls.end(),
              seq.begin() + (p * group + patch) * h);
  }
  Qwen2Forward(encoder_.aggregator, seq.data(), *patches * group, 0,
               /*use_cache=*/false, nullptr, /*release_pages=*/true);
  std::vector<float> cls_rows(static_cast<std::size_t>(*patches * h));
  for (std::int64_t p = 0; p < *patches; ++p) {
    RmsNormTo(seq.data() + (p * group + patch) * h, cls_rows.data() + p * h, 1, h,
              encoder_.aggregator.final_norm,
              encoder_.aggregator.config.rms_norm_eps);
  }
  std::vector<float> params(
      static_cast<std::size_t>(*patches * encoder_.fc3.out_features));
  LinearForward(encoder_.fc3, cls_rows.data(), *patches, params.data());
  return params;
}

std::vector<float> Model::SamplePosterior(const std::vector<float>& params,
                                          const float* noise) const {
  const std::int64_t d = config_.latent_dim;
  const auto patches = static_cast<std::int64_t>(params.size()) / (2 * d);
  std::vector<float> latent(static_cast<std::size_t>(patches * d));
  for (std::int64_t p = 0; p < patches; ++p) {
    for (std::int64_t i = 0; i < d; ++i) {
      const float mean = params[static_cast<std::size_t>(p * 2 * d + i)];
      const float scale = params[static_cast<std::size_t>(p * 2 * d + d + i)];
      // softplus with PyTorch's threshold 20
      const float softplus = scale > 20.0F ? scale : std::log1p(std::exp(scale));
      latent[static_cast<std::size_t>(p * d + i)] =
          mean + (softplus + 1e-4F) * noise[p * d + i];
    }
  }
  return latent;
}

}  // namespace gufo::models::ming_tts::cpu
