#include "src/cli/video/video_upscale.hpp"

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
#include "src/models/realesrgan/upscale_video.hpp"
#include "src/models/rife/video_io.hpp"

namespace gufo::cli {
namespace {

namespace esr = models::realesrgan;
namespace io = models::rife;

struct Options {
  std::filesystem::path model;
  std::filesystem::path input;
  std::filesystem::path output;
  std::string size{"1920x1200"};
  std::string fit{"pad"};
  int crf{18};
  int tile{0};
  bool no_audio{false};
  bool fp32{false};
  std::filesystem::path reference;
  int bench{0};
};

bool ParseSize(const std::string& text, int* width, int* height) {
  const std::size_t x = text.find('x');
  if (x == std::string::npos)
    return false;
  try {
    *width = std::stoi(text.substr(0, x));
    *height = std::stoi(text.substr(x + 1));
  } catch (...) {
    return false;
  }
  return *width > 0 && *height > 0;
}

bool ReadFrames(const std::filesystem::path& path, const io::VideoInfo& info,
                int count, std::vector<std::vector<std::uint8_t>>* frames,
                std::string* error) {
  io::FrameReader reader;
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

// Compares the network output against a PyTorch dump: manifest.json lists
// {frame, file} entries, each an FP32 [h * scale, w * scale, 3] output of
// the clamped SRVGGNetCompact for that input frame.
int RunReference(esr::Upscaler& upscaler, const Options& options,
                 const io::VideoInfo& info) {
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
    last = std::max(last, static_cast<int>(entry.member_size("frame", 0)));
  std::vector<std::vector<std::uint8_t>> frames;
  std::string error;
  if (!ReadFrames(options.input, info, last + 1, &frames, &error)) {
    std::cerr << "Error: " << error << "\n";
    return 1;
  }
  const std::size_t count = static_cast<std::size_t>(info.width) *
                            info.height * upscaler.scale() * upscaler.scale() *
                            3;
  std::vector<float> ours(count), theirs(count);
  const char* precision = options.fp32 ? "fp32" : "fp16";
  std::ostringstream report;
  report << "{\"precision\":\"" << precision
         << "\",\"tile_pixels\":" << options.tile << ",\"entries\":[";
  double worst_psnr = 1e9, worst_cos = 1.0;
  bool first = true;
  for (const auto& entry : entries->items()) {
    const int frame = static_cast<int>(entry.member_size("frame", 0));
    const std::string file = entry.member_str("file", "");
    std::ifstream in(options.reference / file, std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(theirs.data()),
                 static_cast<std::streamsize>(count * sizeof(float)))) {
      std::cerr << "Error: cannot read " << file << "\n";
      return 1;
    }
    const auto start = std::chrono::steady_clock::now();
    if (!upscaler.UpscaleNetwork(frames[frame].data(), info.width, info.height,
                                 ours.data(), &error)) {
      std::cerr << "Error: " << error << "\n";
      return 1;
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
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
    std::cout << std::fixed << "frame " << frame << ": PSNR "
              << std::setprecision(2) << psnr << " dB (8-bit " << psnr8
              << " dB, " << std::setprecision(3)
              << 100.0 * static_cast<double>(off8) / count
              << "% values differ)  cosine " << std::setprecision(7) << cosine
              << "  max|d| " << std::setprecision(4) << max_abs << "  ("
              << std::setprecision(1) << ms << " ms)\n";
    report << (first ? "" : ",") << "{\"frame\":" << frame
           << ",\"psnr\":" << std::setprecision(4) << psnr
           << ",\"psnr8\":" << psnr8 << ",\"cosine\":" << std::setprecision(8)
           << cosine << ",\"max_abs\":" << max_abs << "}";
    first = false;
  }
  report << "],\"worst_psnr\":" << worst_psnr
         << ",\"worst_cosine\":" << worst_cos << "}\n";
  std::ofstream(options.reference /
                (std::string("gufo-compare-") + precision + "-tile" +
                 std::to_string(options.tile) + ".json"))
      << report.str();
  std::cout << "worst PSNR " << std::setprecision(2) << worst_psnr
            << " dB, worst cosine " << std::setprecision(7) << worst_cos
            << "\n";
  return 0;
}

int RunBench(esr::Upscaler& upscaler, const Options& options,
             const io::VideoInfo& info, int width, int height, esr::Fit fit) {
  std::vector<std::vector<std::uint8_t>> frames;
  std::string error;
  if (!ReadFrames(options.input, info, 1, &frames, &error)) {
    std::cerr << "Error: " << error << "\n";
    return 1;
  }
  std::vector<std::uint8_t> out(static_cast<std::size_t>(width) * height * 3);
  std::vector<float> net(static_cast<std::size_t>(info.width) * info.height *
                         upscaler.scale() * upscaler.scale() * 3);
  const auto time = [&](int runs, bool network_only, double* worst) {
    double total = 0;
    for (int i = 0; i < runs; ++i) {
      const auto start = std::chrono::steady_clock::now();
      const bool ok =
          network_only
              ? upscaler.UpscaleNetwork(frames[0].data(), info.width,
                                        info.height, net.data(), &error)
              : upscaler.Upscale(frames[0].data(), info.width, info.height,
                                 width, height, fit, out.data(), &error);
      if (!ok)
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
  if (time(2, false, &worst) < 0) {
    std::cerr << "Error: " << error << "\n";
    return 1;
  }
  worst = 0;
  const double mean = time(options.bench, false, &worst);
  double worst_net = 0;
  const double net_mean = time(options.bench, true, &worst_net);
  if (mean < 0 || net_mean < 0) {
    std::cerr << "Error: " << error << "\n";
    return 1;
  }
  std::cout << std::fixed << std::setprecision(2) << "bench " << info.width
            << "x" << info.height << " -> " << width << "x" << height << " ("
            << esr::FitName(fit) << ") " << (options.fp32 ? "fp32" : "fp16")
            << ", " << upscaler.convolutions() << " convolutions: " << mean
            << " ms per frame (worst " << worst
            << " ms; upload + network + resize + download), network + x"
            << upscaler.scale() << " FP32 download " << net_mean << " ms\n";
  return 0;
}

}  // namespace

int RunVideoUpscale(std::span<const char* const> args) {
  Options options;
  ArgParser parser("gufo video-upscale",
                   "Real-ESRGAN upscaling (SRVGGNetCompact: "
                   "realesr-animevideov3, realesr-general-x4v3)");
  parser.AddOption("-m", "--model", "SAFETENSORS",
                   "Real-ESRGAN weights (tools/upscale/"
                   "convert_realesrgan_pth.py; default $GUFO_UPSCALE_MODEL)",
                   "Input & Output", &options.model);
  parser.AddOption("-i", "--input", "FILE", "Input video or image",
                   "Input & Output", &options.input);
  parser.AddOption("-o", "--output", "FILE",
                   "Output video (H.264, audio copied) or .png (first frame)",
                   "Input & Output", &options.output);
  parser.AddOption("-s", "--size", "WxH", "Output size (default: 1920x1200)",
                   "Upscaling", &options.size);
  parser.AddOption("", "--fit", "MODE",
                   "pad (keep aspect, black bars; default), crop (keep "
                   "aspect, centre crop) or stretch",
                   "Upscaling", &options.fit);
  parser.AddOption("", "--crf", "N", "H.264 CRF of the output (default: 18)",
                   "Upscaling", &options.crf);
  parser.AddOption("", "--tile", "PIXELS",
                   "Largest network tile in input pixels incl. halo "
                   "(default: 524288)",
                   "Upscaling", &options.tile);
  parser.AddFlag("", "--no-audio", "Drop the audio track", "Upscaling",
                 &options.no_audio);
  parser.AddFlag("", "--fp32", "FP32 direct convolutions (validation)",
                 "Diagnostics", &options.fp32);
  parser.AddOption("", "--reference", "DIR",
                   "Compare the network against a PyTorch dump (no output)",
                   "Diagnostics", &options.reference);
  parser.AddOption("", "--bench", "N",
                   "Time N upscales of the first frame (no output)",
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
    if (const char* env = std::getenv("GUFO_UPSCALE_MODEL"))
      options.model = env;
  }
  const bool diagnostic = !options.reference.empty() || options.bench > 0;
  if (options.model.empty() || options.input.empty() ||
      (options.output.empty() && !diagnostic)) {
    std::cerr << "Error: --model, --input and --output are required\n";
    return 2;
  }
  esr::UpscaleVideoOptions run;
  if (!ParseSize(options.size, &run.width, &run.height)) {
    std::cerr << "Error: --size must be WIDTHxHEIGHT\n";
    return 2;
  }
  if (!esr::ParseFit(options.fit, &run.fit)) {
    std::cerr << "Error: --fit must be pad, crop or stretch\n";
    return 2;
  }
  run.crf = options.crf;
  run.keep_audio = !options.no_audio;
  auto upscaler = esr::Upscaler::Load(
      options.model,
      options.fp32 ? esr::Precision::kFp32 : esr::Precision::kFp16, &error);
  if (!upscaler) {
    std::cerr << "Error: " << error << "\n";
    return 1;
  }
  upscaler->SetTilePixels(static_cast<std::size_t>(std::max(0, options.tile)));
  if (diagnostic) {
    io::VideoInfo info;
    if (!io::ProbeVideo(options.input, &info, &error)) {
      std::cerr << "Error: " << error << "\n";
      return 1;
    }
    std::cerr << "Real-ESRGAN x" << upscaler->scale() << ", "
              << upscaler->convolutions() << " convolutions, input "
              << info.width << "x" << info.height << ", weights "
              << upscaler->weight_bytes() / 1024 << " KiB\n";
    const int code = !options.reference.empty()
                         ? RunReference(*upscaler, options, info)
                         : RunBench(*upscaler, options, info, run.width,
                                    run.height, run.fit);
    std::cerr << "workspace " << upscaler->workspace_bytes() / (1024 * 1024)
              << " MiB\n";
    return code;
  }
  int last_percent = -1;
  run.progress = [&](int done, int total) {
    const int percent = total > 0 ? done * 100 / total : 0;
    if (percent / 10 != last_percent / 10) {
      std::cerr << "frames " << done << " / " << total << "\n";
      last_percent = percent;
    }
  };
  esr::UpscaleVideoStats stats;
  if (!esr::UpscaleVideoFile(*upscaler, options.input, options.output, run,
                             &stats, &error)) {
    std::cerr << "Error: " << error << "\n";
    return 1;
  }
  std::cout << std::fixed << std::setprecision(2) << stats.frames << " frames "
            << stats.in_width << "x" << stats.in_height << " -> " << run.width
            << "x" << run.height << " (" << esr::FitName(run.fit)
            << ": picture " << stats.placement.content_w << "x"
            << stats.placement.content_h << " at " << stats.placement.offset_x
            << "," << stats.placement.offset_y << ") at " << stats.fps_num
            << "/" << stats.fps_den << " fps"
            << (stats.audio ? ", audio copied" : ", no audio") << "\nupscale "
            << stats.upscale_ms / std::max(1, stats.frames)
            << " ms per frame, total " << stats.total_seconds << " s, "
            << "workspace " << upscaler->workspace_bytes() / (1024 * 1024)
            << " MiB\n"
            << options.output.string() << "\n";
  return 0;
}

}  // namespace gufo::cli
