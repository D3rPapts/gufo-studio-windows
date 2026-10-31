// MMAudio for gufo: the 44.1 kHz VAE decoder (latent -> log-mel) and the
// BigVGAN v2 vocoder (log-mel -> waveform).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MMAUDIO_VOCODER_HPP_
#define GUFO_MODELS_MMAUDIO_VOCODER_HPP_

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "src/models/mmaudio/common.hpp"

namespace gufo::models::mmaudio {

/// MMAudio's VAE_44k decoder (`decoder.*` of mmaudio_vae_44k_fp16):
/// EDM2 magnitude-preserving 1-D U-Net (512 x (1, 2, 4), one 2x upsampling,
/// attention in the middle), weights normalized at load like
/// remove_weight_norm(), output unnormalized with the 128-band mel
/// statistics.
class VaeDecoder {
public:
  static constexpr int kLatentDim = 40;
  static constexpr int kMels = 128;

  [[nodiscard]] static std::unique_ptr<VaeDecoder> Load(
      const std::filesystem::path& path, Precision precision,
      std::string* error);
  ~VaeDecoder();
  VaeDecoder(const VaeDecoder&) = delete;
  VaeDecoder& operator=(const VaeDecoder&) = delete;

  /// `latent` [frames][40] (unnormalized network output) -> `mel`
  /// [2 * frames][128] (natural-log mel, channel-last).
  bool Decode(const std::vector<float>& latent, int frames,
              std::vector<float>* mel, std::string* error);
  [[nodiscard]] std::size_t weight_bytes() const;

private:
  struct Impl;
  explicit VaeDecoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

/// nvidia/bigvgan_v2_44khz_128band_512x generator with weight norm folded
/// (tools/models/mmaudio/convert_bigvgan.py writes the safetensors file):
/// conv_pre 128 -> 1536, six transposed-conv upsamplings (8, 4, 2, 2, 2, 2)
/// each followed by three AMP blocks (kernels 3, 7, 11, dilations 1, 3, 5)
/// with anti-aliased SnakeBeta, then SnakeBeta, conv_post, clamp(-1, 1).
class BigVgan {
public:
  static constexpr int kMels = 128;
  static constexpr int kHop = 512;

  [[nodiscard]] static std::unique_ptr<BigVgan> Load(
      const std::filesystem::path& path, Precision precision,
      std::string* error);
  ~BigVgan();
  BigVgan(const BigVgan&) = delete;
  BigVgan& operator=(const BigVgan&) = delete;

  /// `mel` [frames][128] -> `audio` [frames * 512] in [-1, 1].
  bool Vocode(const std::vector<float>& mel, int frames,
              std::vector<float>* audio, std::string* error);
  [[nodiscard]] std::size_t weight_bytes() const;

private:
  struct Impl;
  explicit BigVgan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

/// kaiser_sinc_filter1d(cutoff, half_width, kernel_size) of BigVGAN's
/// alias-free activation (FP64 on the host, returned as FP32).
std::vector<float> KaiserSincFilter(double cutoff, double half_width,
                                    int kernel_size);

/// Folds EDM2 MPConv1D weight normalization: each output row w_o becomes
/// w_o / (1e-4 + |w_o| / sqrt(n)) / sqrt(n) * gain with n = in * kernel.
void NormalizeMpWeight(std::vector<float>* weight, int out, float gain);

}  // namespace gufo::models::mmaudio

#endif  // GUFO_MODELS_MMAUDIO_VOCODER_HPP_
