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
/// decoder_base_dim 256, patch 2: 16x spatial and 4x temporal expansion), or
/// of the Wan 2.1 VAE (z_dim 16, base_dim 96, 8x spatial, 4x temporal), chosen
/// by the config. The constants below are the Wan 2.2 values.
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

  /// Latent channels and spatial expansion of the loaded VAE.
  [[nodiscard]] int latent_channels() const;
  [[nodiscard]] int spatial_ratio() const;

  /// Decodes model-space latents [z, frames, height, width] (FP32, as the
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

/// Encoder of the same VAE for one image (the image-to-video first frame),
/// as Diffusers' untiled `_encode` of a single frame: every causal 3x3x3
/// convolution sees two zero frames before the image, so only its last
/// temporal tap is loaded, and the temporal downsampling convolutions are
/// skipped (Diffusers only caches on the first chunk). BF16 weights and
/// activations, FP32 accumulation, attention and norms in FP32.
class VaeEncoder {
public:
  /// Loads the encoder half of `<diffusers root>/vae`.
  [[nodiscard]] static std::unique_ptr<VaeEncoder> Load(
      const std::filesystem::path& vae_dir, std::string* error);
  ~VaeEncoder();
  VaeEncoder(const VaeEncoder&) = delete;
  VaeEncoder& operator=(const VaeEncoder&) = delete;

  /// `rgb` is [height, width, 3] in [-1, 1] (height, width multiples of 16).
  /// Writes the posterior mean [48, height / 16, width / 16] (the "argmax"
  /// latent), with `normalize` as (mean - latents_mean) / latents_std, the
  /// model-space condition of the image-to-video pipeline.
  bool Encode(std::span<const float> rgb, int height, int width, bool normalize,
              std::vector<float>* latent, std::string* error);

  /// Wan 2.1 only: encodes a video `rgb` [frames, height, width, 3] in
  /// [-1, 1] (frames = 4k + 1, height, width multiples of 16) like Diffusers'
  /// untiled `_encode` (first frame, then chunks of four with causal caches).
  /// Writes the posterior mean [16, 1 + (frames - 1) / 4, height / 8,
  /// width / 8], normalized as above when `normalize`. With the Wan 2.1 VAE,
  /// `Encode` is this with one frame.
  bool EncodeVideo(std::span<const float> rgb, int frames, int height,
                   int width, bool normalize, std::vector<float>* latent,
                   std::string* error);

  [[nodiscard]] int latent_channels() const;

  [[nodiscard]] std::size_t weight_bytes() const;
  [[nodiscard]] std::size_t activation_bytes() const;

private:
  struct Impl;
  explicit VaeEncoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_VAE_HPP_
