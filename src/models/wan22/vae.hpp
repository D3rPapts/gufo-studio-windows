#ifndef GUFO_MODELS_WAN22_VAE_HPP_
#define GUFO_MODELS_WAN22_VAE_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::wan22 {

/// Decoded video: frame-major, row-major, interleaved RGB.
struct DecodedVideo {
  int frames{0};
  int height{0};
  int width{0};
  std::vector<float> rgb;          // [-1, 1] after the final clamp (optional)
  std::vector<std::uint8_t> rgb8;  // trunc(255 * (x / 2 + 0.5)) (optional)
};

/// Decoder of the Wan 2.2 VAE (Diffusers AutoencoderKLWan, z_dim 48,
/// decoder_base_dim 256, patch 2: 16x spatial and 4x temporal expansion).
/// Weights are uploaded as BF16; convolutions are BF16 GEMMs with FP32
/// accumulation (im2col in bounded row passes), norms and softmax in FP32.
/// The decode runs one latent frame at a time with causal caches, exactly
/// like Diffusers' untiled `_decode`, so no launch grows with the length of
/// the video and every GEMM is split into short launches.
class VaeDecoder {
public:
  static constexpr int kLatentChannels = 48;
  static constexpr int kSpatialRatio = 16;
  static constexpr int kTemporalRatio = 4;

  /// Loads `<diffusers root>/vae` (config.json and the safetensors file).
  [[nodiscard]] static std::unique_ptr<VaeDecoder> Load(
      const std::filesystem::path& vae_dir, std::string* error);
  ~VaeDecoder();
  VaeDecoder(const VaeDecoder&) = delete;
  VaeDecoder& operator=(const VaeDecoder&) = delete;

  /// Decodes model-space latents [48, frames, height, width] (FP32, as the
  /// transformer produces them; the latents_mean/std de-normalization is
  /// applied here). Fills `video->rgb` and/or `video->rgb8` as requested.
  /// `progress(done, total)` is called after every latent frame; returning
  /// false from `keep_going` cancels.
  bool Decode(std::span<const float> latent, int frames, int height, int width,
              bool want_float, bool want_rgb8, DecodedVideo* video,
              const std::function<void(int, int)>& progress,
              const std::function<bool()>& keep_going, std::string* error);

  [[nodiscard]] std::size_t weight_bytes() const;
  [[nodiscard]] std::size_t activation_bytes() const;

private:
  struct Impl;
  explicit VaeDecoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_VAE_HPP_
