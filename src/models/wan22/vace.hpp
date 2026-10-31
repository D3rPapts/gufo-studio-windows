#ifndef GUFO_MODELS_WAN22_VACE_HPP_
#define GUFO_MODELS_WAN22_VACE_HPP_

// Written with AI assistance (Claude Code).

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/wan22/transformer21.hpp"

namespace gufo::models::wan22 {

/// Wan 2.1 VACE 14B context blocks (Kijai's
/// `Wan2_1-VACE-module-14B_fp8_e4m3fn_scaled_KJ.safetensors`, or a BF16/F16
/// module with the same `vace_*` keys) on top of the Wan 2.1 T2V 14B
/// transformer, as ComfyUI's VaceWanModel runs them:
///   c = vace_patch_embedding(context)            [96 -> 5120, patch 1x2x2]
///   c = before_proj(c) + x                       (x: DiT tokens after the
///                                                 patch embedding)
///   after DiT block i = n * (40 / blocks), n < blocks (0, 5, ..., 35):
///     c = WanAttentionBlock_n(c)                 (t2v: text cross-attention
///                                                 only, same e0 and RoPE)
///     x += strength * after_proj_n(c)
/// The context [96, frames, height, width] is the VAE latent of the inactive
/// video (16, normalized), of the reactive video (16, normalized) and the
/// 64-channel pixel-unshuffled mask (vace_context.hpp). Block weights
/// stay FP8 and are unpacked to BF16 right before each GEMM into the
/// transformer's workspace (Transformer21BlockState), like the DiT's own.
class Vace : public Transformer21Hook {
public:
  static constexpr int kContextChannels = 96;
  /// Uses `stream` (the transformer's) for all device work.
  [[nodiscard]] static std::unique_ptr<Vace> Load(
      const std::filesystem::path& file, hipStream_t stream,
      std::string* error);
  ~Vace() override;
  Vace(const Vace&) = delete;
  Vace& operator=(const Vace&) = delete;

  /// The VACE context [96, frames, height, width] in the DiT's latent
  /// geometry (frames include any reference-image frames in front).
  bool SetContext(std::span<const float> context, int frames, int height,
                  int width, std::string* error);
  /// ComfyUI's WanVaceToVideo strength (vace_strength), default 1.
  void set_strength(float strength);
  /// Number of VACE blocks (8 for 14B) and the DiT block after which block
  /// `n` runs (n * 40 / blocks).
  [[nodiscard]] int blocks() const;
  [[nodiscard]] int dit_layer(int block) const;
  /// Keeps the first hint (after_proj of VACE block 0, before the strength)
  /// of the next Forward ([tokens, 5120]) for oracle checks.
  void set_probe(bool probe);
  [[nodiscard]] const std::vector<float>& hint_first() const;
  [[nodiscard]] std::size_t weight_bytes() const;

  bool Begin(const float* input, int frames, int height, int width,
             const float* temb, hipStream_t stream,
             std::string* error) override;
  bool BeginBlocks(const Transformer21BlockState& state, hipStream_t stream,
                   std::string* error) override;
  bool AfterBlock(int layer, float* hidden, hipStream_t stream,
                  std::string* error) override;

private:
  struct Impl;
  explicit Vace(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_VACE_HPP_
