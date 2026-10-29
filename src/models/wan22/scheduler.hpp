#ifndef GUFO_MODELS_WAN22_SCHEDULER_HPP_
#define GUFO_MODELS_WAN22_SCHEDULER_HPP_

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::wan22 {

/// Settings of Diffusers' UniPCMultistepScheduler that Wan 2.2 TI2V-5B uses
/// (`scheduler/scheduler_config.json`): flow sigmas with a static shift,
/// flow prediction converted to x0 (predict_x0), the B(h) = expm1 variant
/// ("bh2"), linspace spacing and a zero final sigma.
struct UniPcConfig {
  int train_timesteps{1000};
  double flow_shift{5.0};
  int solver_order{2};
  bool lower_order_final{true};
};

/// Reads and validates scheduler_config.json; only the settings above may
/// vary, every other option must have the value the port implements.
[[nodiscard]] std::optional<UniPcConfig> ReadUniPcConfig(
    const std::filesystem::path& path, std::string* error = nullptr);

/// Host-side UniPC multistep sampler (FP32 arithmetic like the Diffusers
/// reference). One Step per denoising step, as WanPipeline calls it.
class UniPcScheduler {
public:
  UniPcScheduler(const UniPcConfig& config, int steps);

  /// sigmas()[i] belongs to step i; one extra trailing 0.
  [[nodiscard]] const std::vector<float>& sigmas() const { return sigmas_; }
  /// Integer timesteps (truncated sigma * 1000) the transformer receives.
  [[nodiscard]] const std::vector<std::int64_t>& timesteps() const {
    return timesteps_;
  }
  [[nodiscard]] int step_index() const { return step_index_; }

  /// Advances `sample` (in place) by one step given the (guided) flow
  /// prediction of the transformer for the current step. The prediction is
  /// treated as BF16 values, as in the reference.
  void Step(std::span<const float> model_output, std::span<float> sample);

private:
  UniPcConfig config_;
  std::vector<float> sigmas_;
  std::vector<std::int64_t> timesteps_;
  std::vector<float> lambdas_;
  std::vector<float> outputs_[2];  // converted x0 predictions, oldest first
  bool has_output_[2]{false, false};
  std::vector<float> last_sample_;
  bool has_last_sample_{false};
  int step_index_{0};
  int lower_order_nums_{0};
  int this_order_{1};
};

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_SCHEDULER_HPP_
