#ifndef GUFO_MODELS_WAN22_TRANSFORMER_HPP_
#define GUFO_MODELS_WAN22_TRANSFORMER_HPP_

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::wan22 {

/// One call of the Wan 2.2 TI2V-5B transformer: latent [48, frames, height,
/// width] (height and width even), one timestep per patch token (or a single
/// timestep for all tokens) and the text states [text_rows, 4096] as the
/// pipeline passes them (umT5 output zero-padded to 512 rows).
struct TransformerInput {
  std::span<const float> latent;
  int frames{0};
  int height{0};
  int width{0};
  std::span<const float> timesteps;
  std::span<const float> text;
  int text_rows{0};
  /// >= 0: use the text states cached by PrepareText(text_slot) instead of
  /// `text` / `text_rows` (which may then be empty).
  int text_slot{-1};
};

/// Optional intermediates for oracle comparison, all FP32 row-major. The
/// modulation rows belong to the first token's timestep.
struct TransformerProbes {
  std::vector<float> patch;          // [tokens, 3072] patch embedding
  std::vector<float> temb;           // [3072]
  std::vector<float> timestep_proj;  // [6 * 3072]
  std::vector<float> text_states;    // [text_rows, 3072]
  std::vector<float> block0_attn1;   // [tokens, 3072] self-attention output
  std::vector<float> block0_attn2;   // [tokens, 3072] cross-attention output
  std::vector<float> block0_ffn;     // [tokens, 3072] feed-forward output
  std::vector<float> block0;         // [tokens, 3072] residual after block 0
  std::vector<float> block_last;     // [tokens, 3072] residual after block 29
};

/// WanTransformer3DModel of Wan 2.2 TI2V-5B (Diffusers `transformer/`, FP32
/// files uploaded as BF16; time embedder, modulation tables and the cross
/// attention norm stay FP32 as in Diffusers). Arithmetic follows the BF16
/// Diffusers reference except for the residual stream, which stays FP32;
/// GEMMs, norms and softmax accumulate in FP32.
class Transformer {
public:
  static constexpr int kChannels = 48;
  static constexpr int kTextWidth = 4096;
  static constexpr int kModelWidth = 3072;

  /// Loads `<diffusers root>/transformer`. Returns null and sets `error` on
  /// failure.
  [[nodiscard]] static std::unique_ptr<Transformer> Load(
      const std::filesystem::path& transformer_dir, std::string* error);
  ~Transformer();
  Transformer(const Transformer&) = delete;
  Transformer& operator=(const Transformer&) = delete;

  /// Writes the noise prediction [48, frames, height, width] (BF16 values as
  /// FP32). `probes` may be null.
  bool Forward(const TransformerInput& input, std::vector<float>* noise,
               TransformerProbes* probes, std::string* error);

  /// Number of text slots PrepareText can fill (prompt and negative prompt).
  static constexpr int kTextSlots = 2;
  /// Runs the text-dependent part of the model once for `text` [text_rows,
  /// 4096] (the text embedding and every block's cross-attention keys and
  /// values, which do not change over the denoising steps) and keeps it in
  /// `slot` for Forward calls with TransformerInput::text_slot = slot.
  bool PrepareText(int slot, std::span<const float> text, int text_rows,
                   std::string* error);

  /// Device bytes held for weights.
  [[nodiscard]] std::size_t weight_bytes() const;
  /// Device bytes held for activations (grows with the largest input).
  [[nodiscard]] std::size_t activation_bytes() const;
  /// Milliseconds of the transformer blocks of the last Forward, measured
  /// with device events (self-attention kernels only and everything).
  [[nodiscard]] double last_attention_ms() const;
  [[nodiscard]] double last_blocks_ms() const;

private:
  struct Impl;
  explicit Transformer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_TRANSFORMER_HPP_
