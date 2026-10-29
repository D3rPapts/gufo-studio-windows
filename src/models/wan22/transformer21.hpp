#ifndef GUFO_MODELS_WAN22_TRANSFORMER21_HPP_
#define GUFO_MODELS_WAN22_TRANSFORMER21_HPP_

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::wan22 {

/// One call of the Wan 2.1 I2V 14B transformer: the model input
/// [36, frames, height, width] (16 noisy latent + 4 first-frame mask + 16
/// condition latent channels, height and width even), one timestep for all
/// tokens, and a context slot filled by PrepareContext.
struct Transformer21Input {
  std::span<const float> latent;
  int frames{0};
  int height{0};
  int width{0};
  float timestep{0};
  int context_slot{0};
};

/// Optional intermediates for oracle comparison (FP32 row-major).
struct Transformer21Probes {
  std::vector<float> temb;    // [5120]
  std::vector<float> block0;  // [tokens, 5120] residual after block 0
};

/// Wan 2.1 image-to-video 14B (dim 5120, 40 blocks, 40 heads of 128, FFN
/// 13824, CLIP image cross-attention) from Kijai's FP8 file
/// (`*_fp8_e4m3fn_scaled_KJ.safetensors`: block linears FP8 e4m3 with one FP32
/// `scale_weight` per tensor, everything else F16/F32, original Wan keys).
/// The FP8 matrices stay FP8 on the device (gfx1151 has no FP8 matrix
/// cores) and are unpacked to BF16 right before each GEMM; an optional LoRA
/// (lightx2v format: lora_up/lora_down, `diff`, `diff_b`, no alpha) is added
/// exactly: its low-rank product into the unpacked BF16 matrix (FP32
/// accumulation), its full differences into norms, biases and the F16
/// matrices at load time. Activations BF16, residual stream, norms,
/// modulation and softmax FP32.
class Transformer21 {
public:
  static constexpr int kInChannels = 36;
  static constexpr int kOutChannels = 16;
  static constexpr int kModelWidth = 5120;
  static constexpr int kTextWidth = 4096;
  static constexpr int kImageWidth = 1280;
  static constexpr int kImageTokens = 257;
  static constexpr int kContextSlots = 2;

  /// `lora` may be empty. Returns null and sets `error` on failure.
  [[nodiscard]] static std::unique_ptr<Transformer21> Load(
      const std::filesystem::path& dit_file,
      const std::filesystem::path& lora_file, float lora_strength,
      std::string* error);
  ~Transformer21();
  Transformer21(const Transformer21&) = delete;
  Transformer21& operator=(const Transformer21&) = delete;

  /// Embeds the text states [text_rows, 4096] (umT5 output, zero-padded to
  /// 512 rows by the pipeline) and the CLIP image states [257, 1280] and
  /// caches every block's cross-attention keys and values in `slot`.
  /// `context_text` / `context_image` (may be null) receive the embedded
  /// context [rows, 5120] for probing.
  bool PrepareContext(int slot, std::span<const float> text, int text_rows,
                      std::span<const float> image,
                      std::vector<float>* context_text,
                      std::vector<float>* context_image, std::string* error);

  /// Writes the flow prediction [16, frames, height, width]. `probes` may be
  /// null.
  bool Forward(const Transformer21Input& input, std::vector<float>* noise,
               Transformer21Probes* probes, std::string* error);

  [[nodiscard]] std::size_t weight_bytes() const;
  [[nodiscard]] std::size_t activation_bytes() const;
  /// Milliseconds of the last Forward's blocks and of their self-attention
  /// kernels (device events).
  [[nodiscard]] double last_blocks_ms() const;
  [[nodiscard]] double last_attention_ms() const;

private:
  struct Impl;
  explicit Transformer21(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_TRANSFORMER21_HPP_
