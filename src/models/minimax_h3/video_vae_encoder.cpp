#include "src/models/minimax_h3/video_vae_encoder.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <exception>
#include <limits>
#include <numbers>

#include "src/models/minimax_h3/json.hpp"
#include "src/models/minimax_h3/sampling.hpp"

namespace gufo::minimax_h3 {
namespace {

constexpr int kLatentChannels = kH3VideoVaeMomentChannels / 2;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

// Round a float to the nearest IEEE half (ties to even) and widen it back,
// matching torch's `.to(torch.float16).float()`.
float RoundTripHalf(float value) {
  if (!std::isfinite(value)) {
    return value;
  }
  const float magnitude = std::fabs(value);
  if (magnitude >= 65520.0F) {
    return std::copysign(std::numeric_limits<float>::infinity(), value);
  }
  if (magnitude < 6.103515625e-05F) {
    // Subnormal halves are multiples of 2^-24.
    const float step = 5.9604644775390625e-08F;
    return std::nearbyint(value / step) * step;
  }
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  // Keep 10 mantissa bits: round the low 13 bits to nearest even.
  const std::uint32_t lower = bits & 0x1FFFU;
  std::uint32_t upper = bits & ~0x1FFFU;
  if (lower > 0x1000U || (lower == 0x1000U && (upper & 0x2000U) != 0U)) {
    upper += 0x2000U;
  }
  return std::bit_cast<float>(upper);
}

double LanczosKernel(double x) {
  constexpr double kSupport = 3.0;
  if (x == 0.0) {
    return 1.0;
  }
  if (x <= -kSupport || x >= kSupport) {
    return 0.0;
  }
  const double pi_x = std::numbers::pi * x;
  return kSupport * std::sin(pi_x) * std::sin(pi_x / kSupport) / (pi_x * pi_x);
}

// PIL-style separable Lanczos-3 weights for one axis.
struct AxisWeights {
  std::vector<int> first;
  std::vector<int> count;
  std::vector<double> weights;
  int stride{0};
};

AxisWeights BuildAxisWeights(int source, int target, double offset,
                             double extent) {
  const double scale = extent / target;
  const double filter_scale = std::max(scale, 1.0);
  const double support = 3.0 * filter_scale;
  AxisWeights axis;
  axis.stride = static_cast<int>(std::ceil(support)) * 2 + 1;
  axis.first.resize(static_cast<std::size_t>(target));
  axis.count.resize(static_cast<std::size_t>(target));
  axis.weights.assign(static_cast<std::size_t>(target) * axis.stride, 0.0);
  for (int index = 0; index < target; ++index) {
    const double center = offset + (index + 0.5) * scale;
    int lower = static_cast<int>(center - support + 0.5);
    int upper = static_cast<int>(center + support + 0.5);
    lower = std::max(lower, 0);
    upper = std::min(upper, source);
    double total = 0.0;
    const int taps = std::min(upper - lower, axis.stride);
    for (int tap = 0; tap < taps; ++tap) {
      const double weight =
          LanczosKernel((lower + tap - center + 0.5) / filter_scale);
      axis.weights[static_cast<std::size_t>(index) * axis.stride + tap] =
          weight;
      total += weight;
    }
    if (total != 0.0) {
      for (int tap = 0; tap < taps; ++tap) {
        axis.weights[static_cast<std::size_t>(index) * axis.stride + tap] /=
            total;
      }
    }
    axis.first[static_cast<std::size_t>(index)] = lower;
    axis.count[static_cast<std::size_t>(index)] = taps;
  }
  return axis;
}

std::uint8_t ClampByte(double value) {
  return static_cast<std::uint8_t>(
      std::clamp(std::lround(value), long{0}, long{255}));
}

}  // namespace

std::vector<std::uint8_t> FitKeyframeToCanvas(std::span<const std::uint8_t> rgb,
                                              int source_width,
                                              int source_height, int width,
                                              int height) {
  if (source_width <= 0 || source_height <= 0 || width <= 0 || height <= 0 ||
      rgb.size() !=
          static_cast<std::size_t>(source_width) * source_height * 3U) {
    return {};
  }
  if (source_width == width && source_height == height) {
    return {rgb.begin(), rgb.end()};
  }
  // Cover the canvas, then crop the centre. The crop is expressed as a source
  // window so that one resampling pass produces the canvas directly.
  const double scale = std::max(static_cast<double>(width) / source_width,
                                static_cast<double>(height) / source_height);
  const int resized_width =
      std::max(width, static_cast<int>(std::lround(source_width * scale)));
  const int resized_height =
      std::max(height, static_cast<int>(std::lround(source_height * scale)));
  const int left = std::max(0, (resized_width - width) / 2);
  const int top = std::max(0, (resized_height - height) / 2);
  const double scale_x = static_cast<double>(source_width) / resized_width;
  const double scale_y = static_cast<double>(source_height) / resized_height;
  const AxisWeights columns =
      BuildAxisWeights(source_width, width, left * scale_x, width * scale_x);
  const AxisWeights rows =
      BuildAxisWeights(source_height, height, top * scale_y, height * scale_y);

  std::vector<double> horizontal(static_cast<std::size_t>(source_height) *
                                 width * 3U);
  for (int y = 0; y < source_height; ++y) {
    for (int x = 0; x < width; ++x) {
      std::array<double, 3> sum{};
      const int first = columns.first[static_cast<std::size_t>(x)];
      for (int tap = 0; tap < columns.count[static_cast<std::size_t>(x)];
           ++tap) {
        const double weight =
            columns.weights[static_cast<std::size_t>(x) * columns.stride + tap];
        const std::size_t source =
            (static_cast<std::size_t>(y) * source_width + first + tap) * 3U;
        for (int channel = 0; channel < 3; ++channel) {
          sum[static_cast<std::size_t>(channel)] +=
              weight * rgb[source + channel];
        }
      }
      for (int channel = 0; channel < 3; ++channel) {
        horizontal[(static_cast<std::size_t>(y) * width + x) * 3U + channel] =
            sum[static_cast<std::size_t>(channel)];
      }
    }
  }
  std::vector<std::uint8_t> output(static_cast<std::size_t>(width) * height *
                                   3U);
  for (int y = 0; y < height; ++y) {
    const int first = rows.first[static_cast<std::size_t>(y)];
    for (int x = 0; x < width; ++x) {
      std::array<double, 3> sum{};
      for (int tap = 0; tap < rows.count[static_cast<std::size_t>(y)]; ++tap) {
        const double weight =
            rows.weights[static_cast<std::size_t>(y) * rows.stride + tap];
        const std::size_t source =
            (static_cast<std::size_t>(first + tap) * width + x) * 3U;
        for (int channel = 0; channel < 3; ++channel) {
          sum[static_cast<std::size_t>(channel)] +=
              weight * horizontal[source + channel];
        }
      }
      for (int channel = 0; channel < 3; ++channel) {
        output[(static_cast<std::size_t>(y) * width + x) * 3U + channel] =
            ClampByte(sum[static_cast<std::size_t>(channel)]);
      }
    }
  }
  return output;
}

bool BuildKeyframeConditioning(const ModelInventory& inventory,
                               const KeyframeMoments& moments,
                               std::uint64_t encode_seed,
                               std::uint64_t noise_seed,
                               std::vector<float>* latent, std::string* error) {
  const std::size_t plane = static_cast<std::size_t>(moments.latent_height) *
                            static_cast<std::size_t>(moments.latent_width);
  if (latent == nullptr || plane == 0 ||
      moments.moments.size() != plane * kH3VideoVaeMomentChannels) {
    SetError(error, "invalid MiniMax H3 keyframe moments");
    return false;
  }
  std::array<float, kLatentChannels> mean{};
  std::array<float, kLatentChannels> deviation{};
  try {
    const json::Value config = json::ParseFile(
        inventory.model_root() / "FL2VA/video_vae/config.json", 1U << 20U);
    const json::Value* mean_value = config.Find("latents_mean");
    const json::Value* std_value = config.Find("latents_std");
    if (mean_value == nullptr || std_value == nullptr ||
        !mean_value->IsArray() || !std_value->IsArray() ||
        mean_value->AsArray().size() != mean.size() ||
        std_value->AsArray().size() != deviation.size()) {
      throw json::Error("latent normalization shape differs");
    }
    for (std::size_t index = 0; index < mean.size(); ++index) {
      mean[index] = static_cast<float>(mean_value->AsArray()[index].AsDouble());
      deviation[index] =
          static_cast<float>(std_value->AsArray()[index].AsDouble());
      if (!(deviation[index] > 0.0F)) {
        throw json::Error("latent normalization contains invalid values");
      }
    }
  } catch (const std::exception& exception) {
    SetError(error, "MiniMax H3 VisualVAE config failed: " +
                        std::string(exception.what()));
    return false;
  }
  latent->assign(plane * kLatentChannels, 0.0F);
  NormalRng posterior(encode_seed);
  NormalRng augmentation(noise_seed);
  for (int channel = 0; channel < kLatentChannels; ++channel) {
    for (std::size_t index = 0; index < plane; ++index) {
      const float mu =
          moments.moments[static_cast<std::size_t>(channel) * plane + index];
      const float log_variance = std::clamp(
          moments.moments[static_cast<std::size_t>(channel + kLatentChannels) *
                              plane +
                          index],
          -30.0F, 20.0F);
      const float sample =
          mu + std::exp(0.5F * log_variance) * posterior.NextNormal();
      const float normalized =
          (RoundTripHalf(sample) - mean[static_cast<std::size_t>(channel)]) /
          deviation[static_cast<std::size_t>(channel)];
      (*latent)[static_cast<std::size_t>(channel) * plane + index] =
          kH3KeyframeNoiseAugmentation * normalized +
          (1.0F - kH3KeyframeNoiseAugmentation) * augmentation.NextNormal();
    }
  }
  return true;
}

bool BuildGuideClipConditioning(const ModelInventory& inventory,
                                std::span<const KeyframeMoments> moments,
                                std::uint64_t encode_seed,
                                std::uint64_t noise_seed,
                                std::vector<float>* latent,
                                std::string* error) {
  if (latent == nullptr || moments.empty()) {
    SetError(error, "invalid MiniMax H3 guide clip moments");
    return false;
  }
  const int latent_height = moments.front().latent_height;
  const int latent_width = moments.front().latent_width;
  const std::size_t plane = static_cast<std::size_t>(latent_height) *
                            static_cast<std::size_t>(latent_width);
  const std::size_t time = moments.size();
  latent->assign(plane * time * static_cast<std::size_t>(kLatentChannels),
                 0.0F);
  std::vector<float> frame;
  for (std::size_t t = 0; t < time; ++t) {
    if (moments[t].latent_height != latent_height ||
        moments[t].latent_width != latent_width ||
        !BuildKeyframeConditioning(inventory, moments[t], encode_seed + t,
                                   noise_seed + t, &frame, error)) {
      if (error != nullptr && error->empty()) {
        *error = "MiniMax H3 guide clip latent frames differ in size";
      }
      return false;
    }
    for (std::size_t channel = 0;
         channel < static_cast<std::size_t>(kLatentChannels); ++channel) {
      std::copy_n(frame.begin() + static_cast<std::ptrdiff_t>(channel * plane),
                  plane,
                  latent->begin() + static_cast<std::ptrdiff_t>(
                                        (channel * time + t) * plane));
    }
  }
  return true;
}

#if !defined(ENGINE_ENABLE_HIP)
bool EncodeKeyframeMoments(const ModelInventory&,
                           std::span<const KeyframeImage>,
                           const CancellationToken*,
                           std::vector<KeyframeMoments>*,
                           VideoVaeEncoderTelemetry*, std::string* error) {
  SetError(error, "MiniMax H3 keyframe encoding requires ENGINE_ENABLE_HIP");
  return false;
}

bool EncodeGuideClipMoments(const ModelInventory&, int, int,
                            std::span<const std::uint8_t>, int,
                            const CancellationToken*,
                            std::vector<KeyframeMoments>*,
                            VideoVaeEncoderTelemetry*, std::string* error) {
  SetError(error, "MiniMax H3 guide clip encoding requires ENGINE_ENABLE_HIP");
  return false;
}
#endif

}  // namespace gufo::minimax_h3
