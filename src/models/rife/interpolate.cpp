#include "src/models/rife/interpolate.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <vector>

#include "src/models/rife/video_io.hpp"

namespace gufo::models::rife {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

double Milliseconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - since)
      .count();
}

}  // namespace

bool InterpolateVideoFile(Interpolator& interpolator,
                          const std::filesystem::path& input,
                          const std::filesystem::path& output,
                          const InterpolateVideoOptions& options,
                          InterpolateVideoStats* stats, std::string* error) {
  const auto start = std::chrono::steady_clock::now();
  InterpolateVideoStats local;
  InterpolateVideoStats& s = stats != nullptr ? *stats : local;
  s = {};
  if (options.multiplier < 2 || options.multiplier > 8) {
    SetError(error, "RIFE multiplier must be 2..8");
    return false;
  }
  VideoInfo info;
  if (!ProbeVideo(input, &info, error) ||
      !interpolator.Configure(info.width, info.height, options.scale_factor,
                              error))
    return false;
  s.width = info.width;
  s.height = info.height;
  long long num = static_cast<long long>(info.fps_num) * options.multiplier;
  long long den = info.fps_den;
  const bool same_speed = options.fps_num <= 0;
  if (!same_speed) {
    num = options.fps_num;
    den = std::max(1, options.fps_den);
  }
  const long long g = std::gcd(num, den);
  s.fps_num = static_cast<int>(num / g);
  s.fps_den = static_cast<int>(den / g);
  s.audio = info.has_audio && options.keep_audio && same_speed;

  FrameReader reader;
  FrameWriter writer;
  if (!reader.Open(input, info.width, info.height, error) ||
      !writer.Open(output, info.width, info.height, s.fps_num, s.fps_den,
                   s.audio ? input : std::filesystem::path{}, error))
    return false;
  const std::size_t frame_bytes =
      static_cast<std::size_t>(info.width) * info.height * 3;
  std::vector<std::uint8_t> frame(frame_bytes), middle(frame_bytes);
  const int total_estimate =
      info.frames > 0 ? (info.frames - 1) * options.multiplier + 1 : 0;
  const auto fail = [&](const std::string& message) {
    writer.Abort();
    reader.Close();
    SetError(error, message);
    return false;
  };
  std::string step_error;
  int slot = 0;
  bool end = false;
  while (true) {
    if (options.cancelled && options.cancelled())
      return fail("RIFE interpolation cancelled");
    if (!reader.Read(frame.data(), &end, &step_error)) {
      if (end)
        break;
      return fail(step_error);
    }
    auto timer = std::chrono::steady_clock::now();
    if (!interpolator.SetFrame(slot, frame.data(), &step_error))
      return fail(step_error);
    s.encode_ms += Milliseconds(timer);
    if (s.input_frames > 0) {
      for (int k = 1; k < options.multiplier; ++k) {
        timer = std::chrono::steady_clock::now();
        if (!interpolator.Interpolate(
                1 - slot, slot,
                static_cast<float>(k) / static_cast<float>(options.multiplier),
                middle.data(), nullptr, &step_error))
          return fail(step_error);
        s.interpolate_ms += Milliseconds(timer);
        if (!writer.Write(middle.data(), &step_error))
          return fail(step_error);
        ++s.output_frames;
      }
    }
    if (!writer.Write(frame.data(), &step_error))
      return fail(step_error);
    ++s.output_frames;
    ++s.input_frames;
    slot = 1 - slot;
    if (options.progress)
      options.progress(s.output_frames,
                       std::max(total_estimate, s.output_frames));
  }
  reader.Close();
  if (s.input_frames < 2)
    return fail("RIFE needs a video with at least two frames");
  if (!writer.Finish(error))
    return false;
  s.total_seconds = Milliseconds(start) / 1000.0;
  return true;
}

}  // namespace gufo::models::rife
