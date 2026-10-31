#ifndef GUFO_MODELS_WAN22_TRANSFORMER21_HPP_
#define GUFO_MODELS_WAN22_TRANSFORMER21_HPP_

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::wan22 {

/// Extra per-block conditioning (the Uni3C ControlNet): Begin sees the model
/// input and the time embedding once per Forward, AfterBlock may add to the
/// FP32 residual stream [tokens, 5120] after each block. All device work is
/// ordered on `stream` (the transformer's).
/// Device state of one Forward for hooks that run their own Wan blocks (the
/// VACE context blocks). `hidden` is valid only inside BeginBlocks; all other
/// pointers stay valid until the Forward returns. The workspace buffers are
/// the transformer's activations: free inside BeginBlocks and AfterBlock (the
/// block before has finished on the shared stream), overwritten by the next
/// block. All values FP32 unless noted.
struct Transformer21BlockState {
  int frames{0}, height{0}, width{0};  // latent grid (height, width even)
  int tokens{0};                       // frames * (height / 2) * (width / 2)
  const float* hidden{nullptr};        // [tokens, 5120] after the patch
                                       // embedding
  const float* temb{nullptr};          // [5120]
  const float* modulation{nullptr};    // [6, 5120] time projection (e0)
  const float* rope{nullptr};          // (cos, sin) [tokens, 64, 2]
  const std::uint16_t* text{nullptr};  // BF16 [text_rows, 5120] embedded text
  int text_rows{0};
  int context_slot{0};
  std::uint64_t context_serial{0};  // new for every PrepareContext
  // Workspace, BF16 (packed: FP16).
  std::uint16_t* weights{nullptr};    // [2 * 13824 * 5120]
  std::uint16_t* normed{nullptr};     // [tokens, 5120]
  std::uint16_t* linear{nullptr};     // [tokens, 3 * 5120]
  std::uint16_t* query{nullptr};      // [tokens, 5120]
  std::uint16_t* key{nullptr};        // [tokens, 5120]
  std::uint16_t* value{nullptr};      // [tokens, 5120]
  std::uint16_t* packed{nullptr};     // [PaddedRows(tokens) * 5120]
  std::uint16_t* attention{nullptr};  // [tokens, 5120]
  std::uint16_t* ffn{nullptr};        // [ffn_rows, 13824]
  int ffn_rows{0};
};

class Transformer21Hook {
public:
  virtual ~Transformer21Hook() = default;
  /// `input` FP32 [36, frames, height, width], `temb` FP32 [5120].
  virtual bool Begin(const float* input, int frames, int height, int width,
                     const float* temb, hipStream_t stream,
                     std::string* error) = 0;
  virtual bool AfterBlock(int layer, float* hidden, hipStream_t stream,
                          std::string* error) = 0;
  /// Optional, right after Begin: the state above (VACE).
  virtual bool BeginBlocks(const Transformer21BlockState& state,
                           hipStream_t stream, std::string* error) {
    (void)state;
    (void)stream;
    (void)error;
    return true;
  }
  /// Optional (InfiniteTalk speaker routing): right after a block's
  /// self-attention, its RMS-normalized, rotated queries and keys, BF16
  /// head-major [40][tokens][128]. Read-only; valid until the hook returns.
  virtual bool AfterSelfAttention(int layer, const std::uint16_t* query,
                                  const std::uint16_t* key, int tokens,
                                  hipStream_t stream, std::string* error) {
    (void)layer, (void)query, (void)key, (void)tokens, (void)stream,
        (void)error;
    return true;
  }
  /// Optional (InfiniteTalk audio cross-attention): after the text/image
  /// cross-attention residual and before the feed-forward; may add to the
  /// FP32 residual stream [tokens, 5120].
  virtual bool AfterCrossAttention(int layer, float* hidden,
                                   hipStream_t stream, std::string* error) {
    (void)layer, (void)hidden, (void)stream, (void)error;
    return true;
  }
};

/// One call of the Wan 2.1 I2V 14B transformer: the model input
/// [36, frames, height, width] (16 noisy latent + 4 first-frame mask + 16
/// condition latent channels, height and width even), one timestep for all
/// tokens, and a context slot filled by PrepareContext. A text-to-video
/// checkpoint (in_channels() 16) takes the noisy latent [16, frames, height,
/// width] only.
struct Transformer21Input {
  std::span<const float> latent;
  int frames{0};
  int height{0};
  int width{0};
  float timestep{0};
  int context_slot{0};
  Transformer21Hook* hook{nullptr};
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
/// The text-to-video 14B (Kijai's `Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ`, the
/// base of VACE) loads as well: no img_emb / k_img / v_img, 16 input
/// channels (patch embedding zero-padded to 36), text-only cross-attention.
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
  /// 512 rows by the pipeline) and the CLIP image states [257, 1280] (empty
  /// for text_only()) and caches every block's cross-attention keys and
  /// values in `slot`.
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
  /// 36 (image-to-video) or 16 (text-to-video checkpoint).
  [[nodiscard]] int in_channels() const;
  [[nodiscard]] bool text_only() const;
  /// Milliseconds of the last Forward's blocks and of their self-attention
  /// kernels (device events).
  [[nodiscard]] double last_blocks_ms() const;
  [[nodiscard]] double last_attention_ms() const;
  /// The stream all device work runs on (for hooks sharing it).
  [[nodiscard]] hipStream_t stream() const;

private:
  struct Impl;
  explicit Transformer21(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_TRANSFORMER21_HPP_
