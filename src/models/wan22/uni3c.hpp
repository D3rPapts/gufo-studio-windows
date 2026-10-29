#ifndef GUFO_MODELS_WAN22_UNI3C_HPP_
#define GUFO_MODELS_WAN22_UNI3C_HPP_

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/wan22/transformer21.hpp"

namespace gufo::models::wan22 {

/// Uni3C camera ControlNet for Wan 2.1 14B (Kijai's
/// `Wan21_Uni3C_controlnet_fp16.safetensors`), as ComfyUI's
/// WanUni3CControlnetApply runs it: control input = the first 20 channels of
/// the DiT input (noisy latent + mask) and the VAE-encoded guide video (16,
/// normalized); patch embedding 36 -> 5120, proj_in -> 1024, 20 blocks
/// (LayerNorm-zero modulated by the DiT's time embedding, self-attention with
/// 16 heads of 64, QK RMS norm, 3D RoPE [24, 20, 20], GELU-tanh FFN 4096);
/// after DiT block i < 20 it adds strength * proj_out[i](control state).
/// Heads of 64 run through the 128-wide attention kernel with zero padding.
class Uni3C : public Transformer21Hook {
public:
  /// Uses `stream` (the transformer's) for all device work.
  [[nodiscard]] static std::unique_ptr<Uni3C> Load(
      const std::filesystem::path& file, hipStream_t stream,
      std::string* error);
  ~Uni3C() override;
  Uni3C(const Uni3C&) = delete;
  Uni3C& operator=(const Uni3C&) = delete;

  /// The guide latent [16, frames, height, width] (normalized, the DiT's
  /// latent geometry).
  bool SetGuide(std::span<const float> latent, int frames, int height,
                int width, std::string* error);
  void set_strength(float strength);
  /// Keeps the residuals of blocks 0 and 19 of the next Forward ([tokens,
  /// 5120]) for oracle checks.
  void set_probe(bool probe);
  [[nodiscard]] const std::vector<float>& residual_first() const;
  [[nodiscard]] const std::vector<float>& residual_last() const;
  [[nodiscard]] std::size_t weight_bytes() const;

  bool Begin(const float* input, int frames, int height, int width,
             const float* temb, hipStream_t stream,
             std::string* error) override;
  bool AfterBlock(int layer, float* hidden, hipStream_t stream,
                  std::string* error) override;

private:
  struct Impl;
  explicit Uni3C(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_UNI3C_HPP_
