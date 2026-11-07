#include "src/models/realesrgan/upscale_video.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <vector>

#include "src/models/rife/video_io.hpp"

namespace gufo::models::realesrgan {
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

bool UpscaleVideoFile(Upscaler& upscaler, const std::filesystem::path& input,
                      const std::filesystem::path& output,
                      const UpscaleVideoOptions& options,
                      UpscaleVideoStats* stats, std::string* error) {
  namespace io = models::rife;
  const auto start = std::chrono::steady_clock::now();
  UpscaleVideoStats local;
  UpscaleVideoStats& s = stats != nullptr ? *stats : local;
  s = {};
  if (options.width < 2 || options.height < 2 || options.width > 8192 ||
      options.height > 8192) {
    SetError(error, "upscale size must be 2..8192 on each side");
    return false;
  }
  io::VideoInfo info;
  if (!io::ProbeVideo(input, &info, error))
    return false;
  s.in_width = info.width;
  s.in_height = info.height;
  s.fps_num = info.fps_num > 0 ? info.fps_num : 24;
  s.fps_den = info.fps_num > 0 ? info.fps_den : 1;
  s.audio = info.has_audio && options.keep_audio;
  s.placement =
      PlaceContent(info.width, info.height, options.width, options.height,
                   options.fit);
  io::FrameReader reader;
  io::FrameWriter writer;
  writer.SetCrf(options.crf);
  if (!reader.Open(input, info.width, info.height, error) ||
      !writer.Open(output, options.width, options.height, s.fps_num, s.fps_den,
                   s.audio ? input : std::filesystem::path{}, error))
    return false;
  std::vector<std::uint8_t> frame(static_cast<std::size_t>(info.width) *
                                  info.height * 3);
  std::vector<std::uint8_t> out(static_cast<std::size_t>(options.width) *
                                options.height * 3);
  const auto fail = [&](const std::string& message) {
    writer.Abort();
    reader.Close();
    SetError(error, message);
    return false;
  };
  std::string extension = output.extension().string();
  std::ranges::transform(extension, extension.begin(), [](char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  });
  const bool single_image = extension == ".png";
  std::string step_error;
  bool end = false;
  while (true) {
    if (options.cancelled && options.cancelled())
      return fail("upscaling cancelled");
    if (!reader.Read(frame.data(), &end, &step_error)) {
      if (end)
        break;
      return fail(step_error);
    }
    const auto timer = std::chrono::steady_clock::now();
    if (!upscaler.Upscale(frame.data(), info.width, info.height, options.width,
                          options.height, options.fit, out.data(),
                          &step_error))
      return fail(step_error);
    s.upscale_ms += Milliseconds(timer);
    if (!writer.Write(out.data(), &step_error))
      return fail(step_error);
    ++s.frames;
    if (options.progress)
      options.progress(s.frames, std::max(info.frames, s.frames));
    if (single_image)
      break;
  }
  reader.Close();
  if (s.frames < 1)
    return fail("the input has no frames");
  if (!writer.Finish(error))
    return false;
  s.total_seconds = Milliseconds(start) / 1000.0;
  return true;
}

}  // namespace gufo::models::realesrgan
