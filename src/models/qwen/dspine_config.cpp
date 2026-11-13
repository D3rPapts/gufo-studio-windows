#include "src/models/qwen/dspine_config.hpp"

#include <stdexcept>
#include <string_view>
#include <variant>

namespace gufo::models::qwen {
namespace {

std::uint32_t U32(const core::GgufReader& r, std::string_view key) {
  const auto v = r.GetMetadataUint32(key);
  if (!v) throw std::runtime_error("missing " + std::string(key));
  return *v;
}

float F32(const core::GgufReader& r, std::string_view key) {
  const auto v = r.GetMetadataFloat32(key);
  if (!v) throw std::runtime_error("missing " + std::string(key));
  return *v;
}

}  // namespace

std::optional<DSpineConfig> ReadDSpineConfig(const core::GgufReader& r,
                                             std::string* error) {
  try {
    if (r.GetMetadataString("general.architecture").value_or("") != "dspine")
      throw std::runtime_error("not a dspine GGUF");
    DSpineConfig c;
    c.layers = U32(r, "dspine.block_count");
    c.hidden = U32(r, "dspine.embedding_length");
    c.ffn = U32(r, "dspine.feed_forward_length");
    c.heads = U32(r, "dspine.attention.head_count");
    c.kv_heads = U32(r, "dspine.attention.head_count_kv");
    c.head_dim = U32(r, "dspine.attention.key_length");
    c.block_size = U32(r, "dspine.block_size");
    c.sliding_window = U32(r, "dspine.attention.sliding_window");
    c.mask_token = U32(r, "dspine.mask_token_id");
    c.code_size = U32(r, "dspine.bridge.code_size");
    c.rank = U32(r, "dspine.bridge.rank");
    c.head_topk = U32(r, "dspine.second_pass.head_topk");
    c.vocab = U32(r, "dspine.vocab_size");
    c.rms_eps = F32(r, "dspine.attention.layer_norm_rms_epsilon");
    c.bridge_eps = F32(r, "dspine.bridge.eps");
    c.rope_theta = F32(r, "dspine.rope.freq_base");
    const auto* tl = r.FindMetadata("dspine.target_layers");
    if (!tl) throw std::runtime_error("missing dspine.target_layers");
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&tl->value))
      for (auto x : *u) c.target_layers.push_back(static_cast<std::uint32_t>(x));
    else if (const auto* s = std::get_if<std::vector<std::int64_t>>(&tl->value))
      for (auto x : *s) {
        if (x < 0) throw std::runtime_error("negative dspine.target_layers");
        c.target_layers.push_back(static_cast<std::uint32_t>(x));
      }
    if (c.target_layers.empty())
      throw std::runtime_error("empty dspine.target_layers");
    for (std::size_t i = 1; i < c.target_layers.size(); ++i)
      if (c.target_layers[i] <= c.target_layers[i - 1])
        throw std::runtime_error("dspine.target_layers must increase");
    if (c.layers == 0 || c.hidden == 0 || c.heads == 0 || c.kv_heads == 0 ||
        c.heads % c.kv_heads != 0 || c.head_dim == 0 || c.head_dim % 2 != 0 ||
        c.block_size < 2 || c.code_size == 0 || c.rank == 0 ||
        c.head_topk == 0 || c.head_topk > 32 || c.vocab == 0 ||
        c.mask_token >= c.vocab)
      throw std::runtime_error("invalid dspine topology");
    return c;
  } catch (const std::exception& e) {
    if (error) *error = std::string("dspine: ") + e.what();
    return std::nullopt;
  }
}

}  // namespace gufo::models::qwen
