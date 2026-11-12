#ifndef GUFO_MODELS_MOSS_MUSIC_MODEL_DESCRIPTION_HPP_
#define GUFO_MODELS_MOSS_MUSIC_MODEL_DESCRIPTION_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "src/models/moss_music/config.hpp"

/// Every weight of MOSS-Music-8B-Instruct: its Hugging Face name (as in
/// model.safetensors.index.json), its gufo name, and its shape derived from
/// the configuration. The loader validates the converted files against this
/// table and tools/moss_music/convert_hf_to_gufo.py carries the same mapping.
///
/// Converted layout:
///  * `<stem>.audio.gguf` - audio encoder, audio_adapter and the three
///    DeepStack mergers, BF16 matrices and F32 vectors, Hugging Face names.
///  * `<stem>.llm-*.gguf` - the Qwen3-8B LLM as a llama.cpp `qwen3` GGUF
///    (names `blk.N.attn_q.weight`, ...), produced from the renamed
///    `language_model.*` tensors by llama.cpp's converter and quantizer.
namespace gufo::models::moss_music {

enum class Component : std::uint8_t {
  kAudioEncoder,
  kAudioAdapter,
  kDeepStackMerger,
  kLanguageModel,
};

struct TensorDescription {
  std::string hf_name;
  std::string gufo_name;
  Component component{Component::kAudioEncoder};
  /// PyTorch order: [out, in, ...] for weights, [out] for biases/norms.
  std::vector<std::uint64_t> shape;
  /// A non-trainable buffer (embed_positions.inv_timescales), present in the
  /// checkpoint but outside its parameter count; gufo recomputes it.
  bool buffer{false};

  [[nodiscard]] std::size_t Elements() const {
    std::size_t count = 1;
    for (const std::uint64_t dim : shape) {
      count *= static_cast<std::size_t>(dim);
    }
    return count;
  }
};

/// Parameters the checkpoint's index reports (metadata.total_parameters),
/// which excludes the sinusoid buffer; every tensor is stored as BF16, so
/// metadata.total_size is 2 * (this + 640).
inline constexpr std::size_t kCheckpointParameters = 9'052'463'488;
inline constexpr std::size_t kCheckpointBytes = 18'104'928'256;

namespace detail {

inline void Add(std::vector<TensorDescription>* out, std::string hf,
                std::string gufo, Component component,
                std::vector<std::uint64_t> shape, bool buffer = false) {
  out->push_back(TensorDescription{std::move(hf), std::move(gufo), component,
                                   std::move(shape), buffer});
}

inline void AddGatedMlp(std::vector<TensorDescription>* out,
                        const std::string& prefix, Component component,
                        const AdapterConfig& adapter) {
  Add(out, prefix + ".gate_proj.weight", prefix + ".gate_proj.weight",
      component, {adapter.hidden, adapter.input});
  Add(out, prefix + ".up_proj.weight", prefix + ".up_proj.weight", component,
      {adapter.hidden, adapter.input});
  Add(out, prefix + ".down_proj.weight", prefix + ".down_proj.weight",
      component, {adapter.output, adapter.hidden});
}

}  // namespace detail

/// All tensors of the checkpoint in a stable order: encoder, adapter,
/// mergers, language model.
[[nodiscard]] inline std::vector<TensorDescription> DescribeTensors(
    const ModelConfig& config = {}) {
  using detail::Add;
  std::vector<TensorDescription> out;
  const AudioEncoderConfig& audio = config.audio;
  const std::uint64_t d = audio.d_model;
  const std::uint64_t c = audio.conv_channels;
  constexpr Component kEnc = Component::kAudioEncoder;

  // Conv2d stem: [out_channels, in_channels, 3, 3], stride 2, padding 1.
  Add(&out, "audio_encoder.conv1.weight", "audio_encoder.conv1.weight", kEnc,
      {c, 1, 3, 3});
  Add(&out, "audio_encoder.conv1.bias", "audio_encoder.conv1.bias", kEnc, {c});
  Add(&out, "audio_encoder.conv2.weight", "audio_encoder.conv2.weight", kEnc,
      {c, c, 3, 3});
  Add(&out, "audio_encoder.conv2.bias", "audio_encoder.conv2.bias", kEnc, {c});
  Add(&out, "audio_encoder.conv3.weight", "audio_encoder.conv3.weight", kEnc,
      {c, c, 3, 3});
  Add(&out, "audio_encoder.conv3.bias", "audio_encoder.conv3.bias", kEnc, {c});
  // Linear(480 * 16 -> 1280) over the time-major flattened conv3 output.
  Add(&out, "audio_encoder.stem_proj.weight", "audio_encoder.stem_proj.weight",
      kEnc, {d, audio.StemFlattened()});
  Add(&out, "audio_encoder.stem_proj.bias", "audio_encoder.stem_proj.bias",
      kEnc, {d});
  // Sinusoid timescales exp(-log(10000) / (d/2 - 1) * i), i < d/2.
  Add(&out, "audio_encoder.embed_positions.inv_timescales",
      "audio_encoder.embed_positions.inv_timescales", kEnc, {d / 2}, true);

  // WhisperEncoderLayer x 32: pre-LN attention (k_proj without bias) and
  // pre-LN GELU feed-forward.
  for (std::uint32_t layer = 0; layer < audio.layers; ++layer) {
    const std::string p = "audio_encoder.layers." + std::to_string(layer);
    Add(&out, p + ".self_attn_layer_norm.weight",
        p + ".self_attn_layer_norm.weight", kEnc, {d});
    Add(&out, p + ".self_attn_layer_norm.bias", p + ".self_attn_layer_norm.bias",
        kEnc, {d});
    Add(&out, p + ".self_attn.q_proj.weight", p + ".self_attn.q_proj.weight",
        kEnc, {d, d});
    Add(&out, p + ".self_attn.q_proj.bias", p + ".self_attn.q_proj.bias", kEnc,
        {d});
    Add(&out, p + ".self_attn.k_proj.weight", p + ".self_attn.k_proj.weight",
        kEnc, {d, d});
    Add(&out, p + ".self_attn.v_proj.weight", p + ".self_attn.v_proj.weight",
        kEnc, {d, d});
    Add(&out, p + ".self_attn.v_proj.bias", p + ".self_attn.v_proj.bias", kEnc,
        {d});
    Add(&out, p + ".self_attn.out_proj.weight",
        p + ".self_attn.out_proj.weight", kEnc, {d, d});
    Add(&out, p + ".self_attn.out_proj.bias", p + ".self_attn.out_proj.bias",
        kEnc, {d});
    Add(&out, p + ".final_layer_norm.weight", p + ".final_layer_norm.weight",
        kEnc, {d});
    Add(&out, p + ".final_layer_norm.bias", p + ".final_layer_norm.bias", kEnc,
        {d});
    Add(&out, p + ".fc1.weight", p + ".fc1.weight", kEnc, {audio.ffn, d});
    Add(&out, p + ".fc1.bias", p + ".fc1.bias", kEnc, {audio.ffn});
    Add(&out, p + ".fc2.weight", p + ".fc2.weight", kEnc, {d, audio.ffn});
    Add(&out, p + ".fc2.bias", p + ".fc2.bias", kEnc, {d});
  }
  Add(&out, "audio_encoder.layer_norm.weight", "audio_encoder.layer_norm.weight",
      kEnc, {d});
  Add(&out, "audio_encoder.layer_norm.bias", "audio_encoder.layer_norm.bias",
      kEnc, {d});
  // out_proj is nn.Identity (output_dim == d_model): no weight.

  detail::AddGatedMlp(&out, "audio_adapter", Component::kAudioAdapter,
                      config.adapter);
  for (std::uint32_t k = 0; k < config.text.deepstack_inject_layers; ++k) {
    detail::AddGatedMlp(&out, "deepstack_audio_merger_list." + std::to_string(k),
                        Component::kDeepStackMerger, config.adapter);
  }

  // Qwen3-8B dense, llama.cpp `qwen3` names on the gufo side.
  const TextConfig& text = config.text;
  const std::uint64_t h = text.hidden;
  constexpr Component kLm = Component::kLanguageModel;
  Add(&out, "language_model.embed_tokens.weight", "token_embd.weight", kLm,
      {text.vocab, h});
  for (std::uint32_t layer = 0; layer < text.layers; ++layer) {
    const std::string p = "language_model.layers." + std::to_string(layer);
    const std::string g = "blk." + std::to_string(layer);
    Add(&out, p + ".input_layernorm.weight", g + ".attn_norm.weight", kLm, {h});
    Add(&out, p + ".self_attn.q_proj.weight", g + ".attn_q.weight", kLm,
        {text.QDim(), h});
    Add(&out, p + ".self_attn.k_proj.weight", g + ".attn_k.weight", kLm,
        {text.KvDim(), h});
    Add(&out, p + ".self_attn.v_proj.weight", g + ".attn_v.weight", kLm,
        {text.KvDim(), h});
    Add(&out, p + ".self_attn.o_proj.weight", g + ".attn_output.weight", kLm,
        {h, text.QDim()});
    Add(&out, p + ".self_attn.q_norm.weight", g + ".attn_q_norm.weight", kLm,
        {text.head_dim});
    Add(&out, p + ".self_attn.k_norm.weight", g + ".attn_k_norm.weight", kLm,
        {text.head_dim});
    Add(&out, p + ".post_attention_layernorm.weight", g + ".ffn_norm.weight",
        kLm, {h});
    Add(&out, p + ".mlp.gate_proj.weight", g + ".ffn_gate.weight", kLm,
        {text.intermediate, h});
    Add(&out, p + ".mlp.up_proj.weight", g + ".ffn_up.weight", kLm,
        {text.intermediate, h});
    Add(&out, p + ".mlp.down_proj.weight", g + ".ffn_down.weight", kLm,
        {h, text.intermediate});
  }
  Add(&out, "language_model.norm.weight", "output_norm.weight", kLm, {h});
  // tie_word_embeddings is false: lm_head is its own matrix.
  Add(&out, "lm_head.weight", "output.weight", kLm, {text.vocab, h});
  return out;
}

/// Sum of elements, optionally without buffers (the index's count).
[[nodiscard]] inline std::size_t ParameterCount(
    const std::vector<TensorDescription>& tensors, bool include_buffers) {
  std::size_t count = 0;
  for (const TensorDescription& tensor : tensors) {
    if (include_buffers || !tensor.buffer) {
      count += tensor.Elements();
    }
  }
  return count;
}

}  // namespace gufo::models::moss_music

#endif  // GUFO_MODELS_MOSS_MUSIC_MODEL_DESCRIPTION_HPP_
