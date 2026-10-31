#ifndef GUFO_MODELS_MINIMAX_H3_VIDEO_VAE_ENCODER_HPP_
#define GUFO_MODELS_MINIMAX_H3_VIDEO_VAE_ENCODER_HPP_

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"

namespace gufo::minimax_h3 {

inline constexpr int kH3VideoVaeMomentChannels = 48;
inline constexpr int kH3VideoVaeEncoderTile = 256;
inline constexpr int kH3VideoVaeEncoderTileOverlap = 64;
inline constexpr float kH3KeyframeNoiseAugmentation = 0.999F;
inline constexpr std::uint64_t kH3KeyframeEncodeSeed = 42;

/// One keyframe already placed on the generation canvas.
struct KeyframeImage {
  int width{0};
  int height{0};
  /// Packed RGB8, row-major, width * height * 3 bytes.
  std::span<const std::uint8_t> rgb;
};

struct VideoVaeEncoderTelemetry {
  double load_ms{0.0};
  double encode_ms{0.0};
  int tiles{0};
  std::uint64_t weight_bytes{0};
  std::uint64_t scratch_bytes{0};
};

/// Posterior moments of one keyframe: 48 channels (24 means, then 24 log
/// variances), channel-major over latent_height x latent_width.
struct KeyframeMoments {
  int latent_height{0};
  int latent_width{0};
  std::vector<float> moments;
};

/// Runs the causal 3D CNN encoder of the H3 VisualVAE on single frames. A
/// single frame reduces every causal 3x3x3 convolution to its last temporal
/// slice, so the encoder is a 2D CNN here. Tiles follow the reference
/// 256-pixel/64-pixel-overlap layout and are blended in latent space.
[[nodiscard]] bool EncodeKeyframeMoments(const ModelInventory& inventory,
                                         std::span<const KeyframeImage> images,
                                         const CancellationToken* cancellation,
                                         std::vector<KeyframeMoments>* output,
                                         VideoVaeEncoderTelemetry* telemetry,
                                         std::string* error = nullptr);

/// Encodes a guide clip (MiniMaxH3AddGuide with a multi-frame image batch):
/// `frames` = 17k + 5 RGB8 canvases of width x height, frame-major, through
/// the causal 3D encoder like the reference's encode_temporal (17-frame
/// chunks, the last padded by repeating its final frame, the last 3 latent
/// frames of the concatenation dropped). Writes 5k + 2 latent frames of
/// moments, oldest first. Loads the encoder with all temporal taps.
[[nodiscard]] bool EncodeGuideClipMoments(
    const ModelInventory& inventory, int width, int height,
    std::span<const std::uint8_t> rgb_frames, int frames,
    const CancellationToken* cancellation, std::vector<KeyframeMoments>* output,
    VideoVaeEncoderTelemetry* telemetry, std::string* error = nullptr);

/// Turns posterior moments into the conditioning latent of the reference:
/// sample the posterior under `encode_seed`, round to float16, normalize with
/// the VAE statistics, then hold the anchor at t = 0.999
/// (x = t * latent + (1 - t) * noise, noise drawn under `noise_seed`).
/// Writes 24 x latent_height x latent_width floats.
[[nodiscard]] bool BuildKeyframeConditioning(const ModelInventory& inventory,
                                             const KeyframeMoments& moments,
                                             std::uint64_t encode_seed,
                                             std::uint64_t noise_seed,
                                             std::vector<float>* latent,
                                             std::string* error = nullptr);

/// The conditioning latent of a guide clip: BuildKeyframeConditioning per
/// latent frame (posterior seed `encode_seed + t`, anchor noise seed
/// `noise_seed + t`), interleaved channel-major as
/// [24][latent frames][latent_height][latent_width].
[[nodiscard]] bool BuildGuideClipConditioning(
    const ModelInventory& inventory, std::span<const KeyframeMoments> moments,
    std::uint64_t encode_seed, std::uint64_t noise_seed,
    std::vector<float>* latent, std::string* error = nullptr);

/// Places an RGB8 image on a width x height canvas: scale to cover, Lanczos-3
/// resample, center crop (the reference's follower-keyframe arithmetic).
[[nodiscard]] std::vector<std::uint8_t> FitKeyframeToCanvas(
    std::span<const std::uint8_t> rgb, int source_width, int source_height,
    int width, int height);

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_VIDEO_VAE_ENCODER_HPP_
