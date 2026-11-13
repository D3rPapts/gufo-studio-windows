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


// Target frame rate mode (InterpolateVideoOptions::target_fps_num > 0). The
// reader keeps the last two decoded frames; the two encoder slots remember
// which source frame they hold, so every source frame is encoded at most once.
bool ResampleVideo(Interpolator& interpolator, const VideoInfo& info,
                   const std::filesystem::path& input,
                   const std::filesystem::path& output,
                   const InterpolateVideoOptions& options,
                   InterpolateVideoStats& s, std::string* error) {
  // p_j = j * a / b source frames.
  const long long a = static_cast<long long>(info.fps_num) *
                      std::max(1, options.target_fps_den);
  const long long b =
      static_cast<long long>(info.fps_den) * options.target_fps_num;
  if (a <= 0 || b <= 0) {
    SetError(error, "RIFE target frame rate needs a known input frame rate");
    return false;
  }
  {
    const long long g =
        std::gcd(static_cast<long long>(options.target_fps_num),
                 static_cast<long long>(std::max(1, options.target_fps_den)));
    s.fps_num = static_cast<int>(options.target_fps_num / g);
    s.fps_den = static_cast<int>(std::max(1, options.target_fps_den) / g);
  }
  s.audio = info.has_audio && options.keep_audio;
  FrameReader reader;
  FrameWriter writer;
  if (!reader.Open(input, info.width, info.height, error) ||
      !writer.Open(output, info.width, info.height, s.fps_num, s.fps_den,
                   s.audio ? input : std::filesystem::path{}, error))
    return false;
  const std::size_t frame_bytes =
      static_cast<std::size_t>(info.width) * info.height * 3;
  // Host ring of the last two decoded frames: frame k lives in buffer k & 1.
  std::vector<std::uint8_t> host[2] = {std::vector<std::uint8_t>(frame_bytes),
                                       std::vector<std::uint8_t>(frame_bytes)};
  std::vector<std::uint8_t> middle(frame_bytes);
  long long read = 0;  // frames decoded so far
  bool end = false;
  long long slot_frame[Interpolator::kSlots] = {-1, -1};
  const long long total_estimate =
      info.frames > 0 ? (static_cast<long long>(info.frames) * b + a - 1) / a
                      : 0;
  const auto fail = [&](const std::string& message) {
    writer.Abort();
    reader.Close();
    SetError(error, message);
    return false;
  };
  std::string step_error;
  // Decodes up to and including source frame k (unless the input ends).
  const auto read_through = [&](long long k) {
    while (!end && read <= k) {
      if (!reader.Read(host[read & 1].data(), &end, &step_error)) {
        if (end)
          break;
        return false;
      }
      ++read;
    }
    return true;
  };
  // Encodes source frame k into a slot not holding `keep`; returns the slot.
  const auto encode = [&](long long k, long long keep, int* slot) {
    for (int i = 0; i < Interpolator::kSlots; ++i)
      if (slot_frame[i] == k) {
        *slot = i;
        return true;
      }
    const int target = slot_frame[0] == keep ? 1 : 0;
    const auto timer = std::chrono::steady_clock::now();
    if (!interpolator.SetFrame(target, host[k & 1].data(), &step_error))
      return false;
    s.encode_ms += Milliseconds(timer);
    slot_frame[target] = k;
    *slot = target;
    return true;
  };
  for (long long j = 0;; ++j) {
    if (options.cancelled && options.cancelled())
      return fail("RIFE interpolation cancelled");
    const long long position = j * a;
    const long long i = position / b;
    const long long remainder = position % b;
    if (!read_through(remainder == 0 ? i : i + 1))
      return fail(step_error);
    if (read <= i)
      break;  // past the last source frame
    const std::uint8_t* frame = host[i & 1].data();
    if (remainder != 0 && read > i + 1) {
      int slot0 = 0, slot1 = 0;
      if (!encode(i, i + 1, &slot0) || !encode(i + 1, i, &slot1))
        return fail(step_error);
      const auto timer = std::chrono::steady_clock::now();
      if (!interpolator.Interpolate(
              slot0, slot1,
              static_cast<float>(static_cast<double>(remainder) /
                                 static_cast<double>(b)),
              middle.data(), nullptr, &step_error))
        return fail(step_error);
      s.interpolate_ms += Milliseconds(timer);
      frame = middle.data();
      ++s.interpolated_frames;
    }
    if (!writer.Write(frame, &step_error))
      return fail(step_error);
    ++s.output_frames;
    if (options.progress)
      options.progress(s.output_frames,
                       static_cast<int>(std::max<long long>(
                           total_estimate, s.output_frames)));
  }
  reader.Close();
  s.input_frames = static_cast<int>(read);
  if (s.input_frames < 2)
    return fail("RIFE needs a video with at least two frames");
  return writer.Finish(error);
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
  if (options.target_fps_num <= 0 &&
      (options.multiplier < 2 || options.multiplier > 8)) {
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
  if (options.target_fps_num > 0) {
    if (!ResampleVideo(interpolator, info, input, output, options, s, error))
      return false;
    s.total_seconds = Milliseconds(start) / 1000.0;
    return true;
  }
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
        ++s.interpolated_frames;
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
