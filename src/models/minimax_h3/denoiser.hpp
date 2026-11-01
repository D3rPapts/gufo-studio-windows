#ifndef GUFO_MODELS_MINIMAX_H3_DENOISER_HPP_
#define GUFO_MODELS_MINIMAX_H3_DENOISER_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/preview_hook.hpp"
#include "src/models/minimax_h3/sampling.hpp"

namespace gufo::minimax_h3 {

class LoraAdapter;

inline constexpr std::size_t kH3TextConditionWidth = 5120;
inline constexpr std::size_t kH3VideoLatentChannels = 24;
inline constexpr std::size_t kH3VideoPatchWidth = 96;
inline constexpr std::size_t kH3AudioLatentChannels = 32;
inline constexpr std::size_t kH3AudioTracks = 2;

struct DenoiserOptions {
  GenerationGeometry geometry;
  int evaluations{0};
  std::size_t text_rows{0};
  int active_blocks{50};
  int reuse_interval{1};
  bool row_parallel_attention{true};
  // Merged into the token refiner and DiT projections at load; must outlive
  // the session's creation.
  const LoraAdapter* lora{nullptr};
};

/// An audio guide (MiniMaxH3AddGuide audio): the normalized AudioVAE
/// posterior mean `[32][2][latent_frames]` that starts at pixel frame
/// `frame` of the target video. It is held clean for every step.
struct AudioGuideLatent {
  int frame{0};
  int latent_frames{0};
  std::vector<float> latent;
};

struct DenoiserConditioning {
  std::span<const std::uint16_t> layer50;
  std::span<const std::uint8_t> tags;
  // Image guides (FL2VA keyframes and intermediate guide frames): one
  // normalized, noise-augmented latent frame (24 x latent_height x
  // latent_width) per guide, in packed order, and the pixel frame each one
  // is anchored at.
  std::span<const std::vector<float>> keyframe_latents;
  std::span<const int> keyframe_frames;
  // Latent frames per image guide (empty: one each). A guide clip
  // (MiniMaxH3AddGuide with a 17k + 5 frame batch) has 5k + 2, and its
  // latent is [24][latent frames][latent_height][latent_width].
  std::span<const int> keyframe_latent_frames;
  std::span<const AudioGuideLatent> audio_guides;
};

struct DenoiserVelocity {
  std::vector<float> video;
  std::vector<float> audio;
};

[[nodiscard]] std::optional<double> DenoiserGateScore(
    std::span<const std::uint16_t> modulation, std::uint32_t time_rows,
    std::string* error = nullptr);
[[nodiscard]] std::optional<std::vector<std::size_t>>
SelectGateRankedDenoiserBlocks(std::span<const double> gate_scores,
                               int active_blocks, std::string* error = nullptr);

struct DenoiserTelemetry {
  double setup_ms{0.0};
  double text_refiner_ms{0.0};
  double adaln_precompute_ms{0.0};
  // The AdaLN modulations came from the stage cache (GUFO_H3_CACHE_DIR).
  bool adaln_cache_hit{false};
  double core_load_ms{0.0};
  std::vector<double> forward_ms;
  double sampler_ms{0.0};
  // Live preview read-backs: count and time spent in the denoiser loop
  // (device->host copy, x0 estimate, unpatchify, hand-over).
  double preview_ms{0.0};
  int previews{0};
  std::uint64_t core_weight_bytes{0};
  std::uint64_t persistent_bytes{0};
  std::uint64_t scratch_bytes{0};
  std::uint64_t peak_live_bytes{0};
  std::uint64_t cumulative_allocation_bytes{0};
  std::uint64_t core_gemm_plan_validations{0};
  std::uint64_t minor_page_faults{0};
  std::uint64_t major_page_faults{0};
  std::uint64_t swap_bytes{0};
  std::uint64_t dispatches{0};
  std::uintptr_t scratch_address{0};
  bool row_parallel_attention{true};
  std::string first_non_finite;
};

using DenoiserProgress = void (*)(int completed_steps, int total_steps,
                                  void* opaque);

class DenoiserSession {
public:
  ~DenoiserSession();

  DenoiserSession(const DenoiserSession&) = delete;
  DenoiserSession& operator=(const DenoiserSession&) = delete;
  DenoiserSession(DenoiserSession&&) noexcept;
  DenoiserSession& operator=(DenoiserSession&&) noexcept;

  [[nodiscard]] static std::unique_ptr<DenoiserSession> Create(
      const ModelInventory& inventory, const DenoiserOptions& options,
      const DenoiserConditioning& conditioning,
      const CancellationToken* cancellation, DenoiserTelemetry* telemetry,
      std::string* error = nullptr);

  [[nodiscard]] bool Forward(int step, std::span<const float> video_latent,
                             std::span<const float> audio_latent,
                             const CancellationToken* cancellation,
                             DenoiserVelocity* velocity,
                             DenoiserTelemetry* telemetry,
                             std::string* error = nullptr);

  [[nodiscard]] bool Denoise(std::span<float> video_latent,
                             std::span<float> audio_latent,
                             const CancellationToken* cancellation,
                             DenoiserTelemetry* telemetry,
                             std::string* error = nullptr);

  [[nodiscard]] bool DenoiseWithProgress(std::span<float> video_latent,
                                         std::span<float> audio_latent,
                                         const CancellationToken* cancellation,
                                         DenoiserProgress progress,
                                         void* progress_opaque,
                                         DenoiserTelemetry* telemetry,
                                         std::string* error = nullptr);

  /// Live preview of the denoised video estimate before each Euler update
  /// (channel-major [24][latent frames][height][width]); reads only.
  void SetPreview(models::PreviewHook preview);

  [[nodiscard]] const GenerationGeometry& geometry() const noexcept;
  [[nodiscard]] const PackedLayout& layout() const noexcept;
  [[nodiscard]] const SigmaSchedule& schedule() const noexcept;
  [[nodiscard]] std::span<const std::uint16_t> block0_modulation()
      const noexcept;
  [[nodiscard]] std::span<const std::uint16_t> refined_text() const noexcept;

private:
  struct Impl;
  explicit DenoiserSession(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_DENOISER_HPP_
