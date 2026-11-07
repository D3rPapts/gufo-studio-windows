#include "src/cli/video/video_interpolate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/core/json.hpp"
#include "src/models/rife/interpolate.hpp"
#include "src/models/rife/video_io.hpp"

namespace gufo::cli {
namespace {

namespace rife = models::rife;

struct Options {
  std::filesystem::path model;
  std::filesystem::path input;
  std::filesystem::path output;
  int multiplier{2};
  float scale{1.0F};
  std::string fps;
  std::string target_fps;
  bool no_audio{false};
  bool fp32{false};
  std::filesystem::path reference;
  int bench{0};
};

bool ParseFps(const std::string& text, int* num, int* den) {
  const std::size_t slash = text.find('/');
  try {
    *num = std::stoi(text.substr(0, slash));
    *den = slash == std::string::npos ? 1 : std::stoi(text.substr(slash + 1));
  } catch (...) {
    return false;
  }
  return *num > 0 && *den > 0;
}

// Reads frames 0..count-1 of `path`.
bool ReadFrames(const std::filesystem::path& path, const rife::VideoInfo& info,
                int count, std::vector<std::vector<std::uint8_t>>* frames,
                std::string* error) {
  rife::FrameReader reader;
  if (!reader.Open(path, info.width, info.height, error))
    return false;
  const std::size_t bytes =
      static_cast<std::size_t>(info.width) * info.height * 3;
  while (static_cast<int>(frames->size()) < count) {
    std::vector<std::uint8_t> frame(bytes);
    bool end = false;
    if (!reader.Read(frame.data(), &end, error)) {
      if (end)
        *error = "video has fewer frames than requested";
      return false;
    }
    frames->push_back(std::move(frame));
  }
  return true;
}

// Compares against the PyTorch reference written by
// tools-local/rife_reference.py: manifest.json lists {pair, t, file} entries,
// each an FP32 [height, width, 3] dump of the clamped IFNet output.
int RunReference(rife::Interpolator& interpolator, const Options& options,
                 const rife::VideoInfo& info) {
  std::ifstream manifest_file(options.reference / "manifest.json");
  std::stringstream text;
  text << manifest_file.rdbuf();
  json::Value manifest;
  try {
    manifest = json::parse(text.str());
  } catch (const std::exception& e) {
    std::cerr << "Error: manifest.json: " << e.what() << "\n";
    return 1;
  }
  const json::Value* entries = manifest.find("entries");
  if (entries == nullptr || !entries->is_array()) {
    std::cerr << "Error: manifest.json has no entries\n";
    return 1;
  }
  int last = 0;
  for (const auto& entry : entries->items())
    last = std::max(last, static_cast<int>(entry.member_size("pair", 0)) + 1);
  std::vector<std::vector<std::uint8_t>> frames;
  std::string error;
  if (!ReadFrames(options.input, info, last + 1, &frames, &error)) {
    std::cerr << "Error: " << error << "\n";
    return 1;
  }
  const std::size_t count =
      static_cast<std::size_t>(info.width) * info.height * 3;
  std::vector<float> ours(count), theirs(count);
  std::ostringstream report;
  report << "{\"precision\":\"" << (options.fp32 ? "fp32" : "fp16")
         << "\",\"entries\":[";
  double worst_psnr = 1e9, worst_cos = 1.0;
  bool first = true;
  for (const auto& entry : entries->items()) {
    const int pair = static_cast<int>(entry.member_size("pair", 0));
    const double t = entry.member_double("t", 0.5);
    const std::string file = entry.member_str("file", "");
    std::ifstream in(options.reference / file, std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(theirs.data()),
                 static_cast<std::streamsize>(count * sizeof(float)))) {
      std::cerr << "Error: cannot read " << file << "\n";
      return 1;
    }
    if (!interpolator.SetFrame(0, frames[pair].data(), &error) ||
        !interpolator.SetFrame(1, frames[pair + 1].data(), &error) ||
        !interpolator.Interpolate(0, 1, static_cast<float>(t), nullptr,
                                  ours.data(), &error)) {
      std::cerr << "Error: " << error << "\n";
      return 1;
    }
    double se = 0, dot = 0, na = 0, nb = 0, max_abs = 0, se8 = 0;
    std::size_t off8 = 0;
    for (std::size_t i = 0; i < count; ++i) {
      const double a = ours[i], b = theirs[i];
      se += (a - b) * (a - b);
      dot += a * b;
      na += a * a;
      nb += b * b;
      max_abs = std::max(max_abs, std::abs(a - b));
      const int a8 = static_cast<int>(a * 255.0 + 0.5);
      const int b8 = static_cast<int>(b * 255.0 + 0.5);
      se8 += static_cast<double>(a8 - b8) * (a8 - b8);
      off8 += a8 != b8 ? 1 : 0;
    }
    const double mse = se / static_cast<double>(count);
    const double psnr = mse > 0 ? 10.0 * std::log10(1.0 / mse) : 999.0;
    const double mse8 = se8 / static_cast<double>(count);
    const double psnr8 =
        mse8 > 0 ? 10.0 * std::log10(255.0 * 255.0 / mse8) : 999.0;
    const double cosine = dot / std::sqrt(na * nb);
    worst_psnr = std::min(worst_psnr, psnr);
    worst_cos = std::min(worst_cos, cosine);
    std::cout << std::fixed << std::setprecision(4) << "pair " << pair << " t "
              << t << ": PSNR " << std::setprecision(2) << psnr << " dB (8-bit "
              << psnr8 << " dB, " << std::setprecision(3)
              << 100.0 * static_cast<double>(off8) / count
              << "% values differ)  cosine " << std::setprecision(7) << cosine
              << "  max|d| " << std::setprecision(4) << max_abs << "\n";
    report << (first ? "" : ",") << "{\"pair\":" << pair << ",\"t\":" << t
           << ",\"psnr\":" << psnr << ",\"psnr8\":" << psnr8
           << ",\"cosine\":" << std::setprecision(8) << cosine
           << ",\"max_abs\":" << max_abs << "}";
    first = false;
  }
  report << "],\"worst_psnr\":" << worst_psnr
         << ",\"worst_cosine\":" << worst_cos << "}\n";
  std::ofstream(options.reference /
                (std::string("gufo-compare-") +
                 (options.fp32 ? "fp32" : "fp16") + ".json"))
      << report.str();
  std::cout << "worst PSNR " << std::setprecision(2) << worst_psnr
            << " dB, worst cosine " << std::setprecision(7) << worst_cos
            << "\n";
  return 0;
}

int RunBench(rife::Interpolator& interpolator, const Options& options,
             const rife::VideoInfo& info) {
  std::vector<std::vector<std::uint8_t>> frames;
  std::string error;
  if (!ReadFrames(options.input, info, 2, &frames, &error) ||
      !interpolator.SetFrame(0, frames[0].data(), &error) ||
      !interpolator.SetFrame(1, frames[1].data(), &error)) {
    std::cerr << "Error: " << error << "\n";
    return 1;
  }
  std::vector<std::uint8_t> out(frames[0].size());
  const auto time = [&](int runs, double* worst) {
    double total = 0;
    for (int i = 0; i < runs; ++i) {
      const auto start = std::chrono::steady_clock::now();
      if (!interpolator.Interpolate(0, 1, 0.5F, out.data(), nullptr, &error))
        return -1.0;
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
      total += ms;
      *worst = std::max(*worst, ms);
    }
    return total / runs;
  };
  double worst = 0;
  if (time(3, &worst) < 0) {
    std::cerr << "Error: " << error << "\n";
    return 1;
  }
  worst = 0;
  const double mean = time(options.bench, &worst);
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < options.bench; ++i)
    (void)interpolator.SetFrame(i & 1, frames[i & 1].data(), &error);
  const double encode = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count() /
                        options.bench;
  std::cout << std::fixed << std::setprecision(2) << "bench " << info.width
            << "x" << info.height << " " << (options.fp32 ? "fp32" : "fp16")
            << ": " << mean << " ms per interpolated frame (worst " << worst
            << " ms), " << encode << " ms per input frame (upload + encoder)\n";
  return 0;
}

}  // namespace

int RunVideoInterpolate(std::span<const char* const> args) {
  Options options;
  ArgParser parser("gufo video-interpolate",
                   "RIFE 4.7 frame interpolation (rife47 / rife49 IFNet, "
                   "as ComfyUI's RIFE VFI node)");
  parser.AddOption("-m", "--model", "SAFETENSORS",
                   "RIFE weights (tools/rife/convert_rife_pth.py; default "
                   "$GUFO_RIFE_MODEL)",
                   "Input & Output", &options.model);
  parser.AddOption("-i", "--input", "MP4", "Input video", "Input & Output",
                   &options.input);
  parser.AddOption("-o", "--output", "MP4",
                   "Output video (H.264, audio copied)", "Input & Output",
                   &options.output);
  parser.AddOption("-x", "--multiplier", "N",
                   "Frames per input frame interval, 2..8 (default: 2)",
                   "Interpolation", &options.multiplier);
  parser.AddOption("", "--scale", "F",
                   "RIFE scale_factor 0.5, 1, 2 or 4 (default: 1; 0.5 for 4K)",
                   "Interpolation", &options.scale);
  parser.AddOption("", "--fps", "N[/D]",
                   "Output frame rate (default: input rate x multiplier; any "
                   "other rate is slow/fast motion without audio)",
                   "Interpolation", &options.fps);
  parser.AddOption("", "--target-fps", "N[/D]",
                   "Resample to this frame rate at the same speed (e.g. 60 "
                   "from 24 or 16 fps; RIFE at each output frame's time, "
                   "audio kept); replaces --multiplier and --fps",
                   "Interpolation", &options.target_fps);
  parser.AddFlag("", "--no-audio", "Drop the audio track", "Interpolation",
                 &options.no_audio);
  parser.AddFlag("", "--fp32", "FP32 direct convolutions (validation)",
                 "Diagnostics", &options.fp32);
  parser.AddOption("", "--reference", "DIR",
                   "Compare against a PyTorch reference dump (no output)",
                   "Diagnostics", &options.reference);
  parser.AddOption("", "--bench", "N",
                   "Time N interpolations of the first frame pair (no output)",
                   "Diagnostics", &options.bench);
  std::string error;
  if (!parser.Parse(args, &error)) {
    std::cerr << "Error: " << error << "\n";
    parser.PrintHelp();
    return 2;
  }
  if (parser.IsHelpRequested()) {
    parser.PrintHelp();
    return 0;
  }
  if (options.model.empty()) {
    if (const char* env = std::getenv("GUFO_RIFE_MODEL"))
      options.model = env;
  }
  const bool diagnostic = !options.reference.empty() || options.bench > 0;
  if (options.model.empty() || options.input.empty() ||
      (options.output.empty() && !diagnostic)) {
    std::cerr << "Error: --model, --input and --output are required\n";
    return 2;
  }
  rife::InterpolateVideoOptions run;
  run.multiplier = options.multiplier;
  run.scale_factor = options.scale;
  run.keep_audio = !options.no_audio;
  if (!options.fps.empty() &&
      !ParseFps(options.fps, &run.fps_num, &run.fps_den)) {
    std::cerr << "Error: --fps must be N or N/D\n";
    return 2;
  }
  if (!options.target_fps.empty() &&
      (!options.fps.empty() || !ParseFps(options.target_fps,
                                         &run.target_fps_num,
                                         &run.target_fps_den))) {
    std::cerr << "Error: --target-fps must be N or N/D (without --fps)\n";
    return 2;
  }
  auto interpolator = rife::Interpolator::Load(
      options.model,
      options.fp32 ? rife::Precision::kFp32 : rife::Precision::kFp16, &error);
  if (!interpolator) {
    std::cerr << "Error: " << error << "\n";
    return 1;
  }
  if (diagnostic) {
    rife::VideoInfo info;
    if (!rife::ProbeVideo(options.input, &info, &error) ||
        !interpolator->Configure(info.width, info.height, options.scale,
                                 &error)) {
      std::cerr << "Error: " << error << "\n";
      return 1;
    }
    std::cerr << "RIFE " << info.width << "x" << info.height << ", weights "
              << interpolator->weight_bytes() / 1024 << " KiB, workspace "
              << interpolator->workspace_bytes() / (1024 * 1024) << " MiB\n";
    return !options.reference.empty()
               ? RunReference(*interpolator, options, info)
               : RunBench(*interpolator, options, info);
  }
  int last_percent = -1;
  run.progress = [&](int done, int total) {
    const int percent = total > 0 ? done * 100 / total : 0;
    if (percent / 10 != last_percent / 10) {
      std::cerr << "frames " << done << " / " << total << "\n";
      last_percent = percent;
    }
  };
  rife::InterpolateVideoStats stats;
  if (!rife::InterpolateVideoFile(*interpolator, options.input, options.output,
                                  run, &stats, &error)) {
    std::cerr << "Error: " << error << "\n";
    return 1;
  }
  const int created = stats.interpolated_frames;
  std::cout << std::fixed << std::setprecision(2) << stats.input_frames
            << " -> " << stats.output_frames << " frames at " << stats.fps_num
            << "/" << stats.fps_den << " fps, " << stats.width << "x"
            << stats.height << (stats.audio ? ", audio copied" : ", no audio")
            << "\nIFNet " << (created > 0 ? stats.interpolate_ms / created : 0)
            << " ms per new frame, encoder "
            << stats.encode_ms / std::max(1, stats.input_frames)
            << " ms per input frame, total " << stats.total_seconds << " s\n"
            << options.output.string() << "\n";
  return 0;
}

}  // namespace gufo::cli
