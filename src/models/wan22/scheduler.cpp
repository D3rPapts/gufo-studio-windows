#include "src/models/wan22/scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

#include "src/core/json.hpp"

namespace gufo::models::wan22 {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

// Round-to-nearest-even BF16, returned as FP32 (what a BF16 torch op yields).
float Bf16(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, 4);
  if ((bits & 0x7FFFFFFFU) > 0x7F800000U)
    return value;
  bits = (bits + 0x7FFFU + ((bits >> 16U) & 1U)) & 0xFFFF0000U;
  float result;
  std::memcpy(&result, &bits, 4);
  return result;
}

// lambda = log(alpha) - log(sigma) with alpha = 1 - sigma (flow sigmas);
// +inf for the final sigma 0.
float Lambda(float sigma) {
  return std::log(1.0F - sigma) - std::log(sigma);
}

}  // namespace

std::optional<UniPcConfig> ReadUniPcConfig(const std::filesystem::path& path,
                                           std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    SetError(error, "cannot read " + path.string());
    return std::nullopt;
  }
  std::stringstream text;
  text << in.rdbuf();
  json::Value root;
  try {
    root = json::parse(text.str());
  } catch (const std::exception& e) {
    SetError(error, path.string() + ": " + e.what());
    return std::nullopt;
  }
  const auto expect_string = [&](const char* key, const char* value) {
    if (root.member_str(key) == value)
      return true;
    SetError(error, std::string("unsupported scheduler setting ") + key +
                        " = '" + root.member_str(key) + "'");
    return false;
  };
  const auto expect_bool = [&](const char* key, bool value) {
    const json::Value* item = root.find(key);
    if (item != nullptr && item->is_bool() && item->as_bool() == value)
      return true;
    SetError(error, std::string("unsupported scheduler setting ") + key);
    return false;
  };
  if (!expect_string("_class_name", "UniPCMultistepScheduler") ||
      !expect_string("prediction_type", "flow_prediction") ||
      !expect_string("solver_type", "bh2") ||
      !expect_string("final_sigmas_type", "zero") ||
      !expect_string("timestep_spacing", "linspace") ||
      !expect_bool("use_flow_sigmas", true) ||
      !expect_bool("predict_x0", true) ||
      !expect_bool("use_dynamic_shifting", false) ||
      !expect_bool("thresholding", false) ||
      !expect_bool("use_karras_sigmas", false) ||
      !expect_bool("use_exponential_sigmas", false) ||
      !expect_bool("use_beta_sigmas", false))
    return std::nullopt;
  const json::Value* corrector = root.find("disable_corrector");
  const json::Value* solver_p = root.find("solver_p");
  const json::Value* terminal = root.find("shift_terminal");
  if ((corrector != nullptr &&
       !(corrector->is_array() && corrector->empty())) ||
      (solver_p != nullptr && !solver_p->is_null()) ||
      (terminal != nullptr && !terminal->is_null())) {
    SetError(error,
             "unsupported scheduler setting (disable_corrector, solver_p or "
             "shift_terminal)");
    return std::nullopt;
  }
  UniPcConfig config;
  config.train_timesteps =
      static_cast<int>(root.member_size("num_train_timesteps", 1000));
  config.flow_shift = root.member_double("flow_shift", 1.0);
  config.solver_order = static_cast<int>(root.member_size("solver_order", 2));
  const json::Value* lower = root.find("lower_order_final");
  config.lower_order_final =
      lower == nullptr || !lower->is_bool() || lower->as_bool();
  if (config.train_timesteps < 2 || !(config.flow_shift > 0) ||
      config.solver_order < 1 || config.solver_order > 2) {
    SetError(error, "unsupported scheduler order, shift or timesteps");
    return std::nullopt;
  }
  return config;
}

UniPcScheduler::UniPcScheduler(const UniPcConfig& config, int steps)
    : config_(config) {
  steps = std::max(steps, 1);
  // np.linspace(1, 1 / N, steps + 1)[:-1] in float64, then the static shift.
  const double start = 1.0;
  const double stop = 1.0 / config.train_timesteps;
  const double delta = (stop - start) / steps;
  std::vector<double> sigmas(steps);
  for (int i = 0; i < steps; ++i) {
    const double s = start + i * delta;
    sigmas[i] = config.flow_shift * s / (1.0 + (config.flow_shift - 1.0) * s);
  }
  if (std::fabs(sigmas[0] - 1.0) < 1e-6)
    sigmas[0] -= 1e-6;
  for (int i = 0; i < steps; ++i) {
    timesteps_.push_back(
        static_cast<std::int64_t>(sigmas[i] * config.train_timesteps));
    sigmas_.push_back(static_cast<float>(sigmas[i]));
  }
  sigmas_.push_back(0.0F);
  for (const float sigma : sigmas_)
    lambdas_.push_back(Lambda(sigma));
}

void UniPcScheduler::Step(std::span<const float> model_output,
                          std::span<float> sample) {
  const int k = step_index_;
  const int steps = static_cast<int>(timesteps_.size());
  if (k >= steps || model_output.size() != sample.size())
    return;
  const std::size_t count = sample.size();

  // x0 = sample - bf16(sigma * output): the BF16 prediction times the FP32
  // scalar stays BF16 in torch; the subtraction promotes to FP32.
  const float sigma = sigmas_[k];
  std::vector<float> converted(count);
  for (std::size_t i = 0; i < count; ++i)
    converted[i] = sample[i] - Bf16(sigma * Bf16(model_output[i]));

  // UniC corrector of the previous predictor step.
  if (k > 0 && has_last_sample_) {
    const int order = this_order_;
    const float sigma_t = sigmas_[k], sigma_s0 = sigmas_[k - 1];
    const float alpha_t = 1.0F - sigma_t;
    const float lambda_s0 = lambdas_[k - 1];
    const float h = lambdas_[k] - lambda_s0;
    const float hh = -h;
    const float h_phi_1 = std::expm1(hh);
    const float b_h = std::expm1(hh);
    float rk = 1.0F;
    if (order == 2)
      rk = (lambdas_[k - 2] - lambda_s0) / h;
    float h_phi_k = h_phi_1 / hh - 1.0F;
    float factorial = 1.0F;
    float b[2]{};
    for (int i = 1; i <= order; ++i) {
      b[i - 1] = h_phi_k * factorial / b_h;
      factorial *= static_cast<float>(i + 1);
      h_phi_k = h_phi_k / hh - 1.0F / factorial;
    }
    float rho_d1 = 0.0F, rho_t = 0.5F;
    if (order == 2) {
      // R = [[1, 1], [rk, 1]], rhos = R^-1 b.
      rho_d1 = (b[0] - b[1]) / (1.0F - rk);
      rho_t = b[0] - rho_d1;
    }
    const float scale_x = sigma_t / sigma_s0;
    const float scale_m = alpha_t * h_phi_1;
    const float scale_c = alpha_t * b_h;
    const std::vector<float>& m0 = outputs_[1];
    const std::vector<float>& m1 = outputs_[0];
    for (std::size_t i = 0; i < count; ++i) {
      const float base = scale_x * last_sample_[i] - scale_m * m0[i];
      float correction = rho_t * (converted[i] - m0[i]);
      if (order == 2)
        correction += rho_d1 * ((m1[i] - m0[i]) / rk);
      sample[i] = base - scale_c * correction;
    }
  }

  outputs_[0].swap(outputs_[1]);
  has_output_[0] = has_output_[1];
  outputs_[1].swap(converted);
  has_output_[1] = true;

  int order = config_.solver_order;
  if (config_.lower_order_final)
    order = std::min(order, steps - k);
  this_order_ = std::min(order, lower_order_nums_ + 1);
  last_sample_.assign(sample.begin(), sample.end());
  has_last_sample_ = true;

  // UniP predictor to the next sigma.
  {
    const float sigma_t = sigmas_[k + 1], sigma_s0 = sigmas_[k];
    const float alpha_t = 1.0F - sigma_t;
    const float lambda_s0 = lambdas_[k];
    const float h = lambdas_[k + 1] - lambda_s0;
    const float hh = -h;
    const float h_phi_1 = std::expm1(hh);
    const float b_h = std::expm1(hh);
    float rk = 1.0F;
    if (this_order_ == 2)
      rk = (lambdas_[k - 1] - lambda_s0) / h;
    const float scale_x = sigma_t / sigma_s0;
    const float scale_m = alpha_t * h_phi_1;
    const float scale_p = alpha_t * b_h;
    const std::vector<float>& m0 = outputs_[1];
    const std::vector<float>& m1 = outputs_[0];
    for (std::size_t i = 0; i < count; ++i) {
      float value = scale_x * sample[i] - scale_m * m0[i];
      if (this_order_ == 2)
        value -= scale_p * (0.5F * ((m1[i] - m0[i]) / rk));
      sample[i] = value;
    }
  }
  if (lower_order_nums_ < config_.solver_order)
    ++lower_order_nums_;
  ++step_index_;
}

}  // namespace gufo::models::wan22
