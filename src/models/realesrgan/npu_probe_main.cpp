// npu-upscale-probe: one image through a Real-ESRGAN body ONNX on the NPU
// (VitisAI from Windows ML) or the ONNX Runtime CPU provider, with Take
// One's pre/post-processing; optional pixel comparison with a reference PNG.
//   npu-upscale-probe --device npu|cpu --model body896q_h288_h3.onnx
//     --input vorher.png [--output out.ppm] [--size 1920x1080] [--fit crop]
//     [--compare nachher.png] [--repeat N] [--threads N] [--npu-cache DIR]
//     [--onnxruntime DLL] [--npu-ep DLL] [--npu-config JSON]
// Written with AI assistance (Claude Code).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/realesrgan/ort_upscaler.hpp"

namespace re = gufo::models::realesrgan;

namespace {

int Usage() {
  std::cerr << "usage: npu-upscale-probe --device npu|cpu --model FILE.onnx "
               "--input IN.png [--output OUT.ppm] [--size WxH] "
               "[--fit crop|pad|stretch] [--compare REF.png] [--repeat N] "
               "[--threads N] [--npu-cache DIR] [--onnxruntime DLL] "
               "[--npu-ep DLL] [--npu-config JSON]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  re::OrtUpscalerOptions options;
  std::string model, input, output, compare, size, fit_name = "crop";
  int repeat = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (i + 1 >= argc)
      return Usage();
    const std::string value = argv[++i];
    if (arg == "--device") {
      if (!re::ParseOrtDevice(value, &options.device))
        return Usage();
    } else if (arg == "--model") {
      model = value;
    } else if (arg == "--input") {
      input = value;
    } else if (arg == "--output") {
      output = value;
    } else if (arg == "--compare") {
      compare = value;
    } else if (arg == "--size") {
      size = value;
    } else if (arg == "--fit") {
      fit_name = value;
    } else if (arg == "--repeat") {
      repeat = std::max(1, std::atoi(value.c_str()));
    } else if (arg == "--threads") {
      options.threads = std::atoi(value.c_str());
    } else if (arg == "--npu-cache") {
      options.cache_dir = value;
    } else if (arg == "--onnxruntime") {
      options.runtime_library = value;
    } else if (arg == "--npu-ep") {
      options.ep_library = value;
    } else if (arg == "--npu-config") {
      options.ep_config_file = value;
    } else {
      return Usage();
    }
  }
  re::AreaFit fit = re::AreaFit::kCrop;
  int out_w = 0, out_h = 0;
  if (model.empty() || input.empty() || !re::ParseAreaFit(fit_name, &fit))
    return Usage();
  if (!size.empty() &&
      std::sscanf(size.c_str(), "%dx%d", &out_w, &out_h) != 2)
    return Usage();

  gufo::core::Image image;
  try {
    image = gufo::core::DecodeImage(gufo::core::ReadImageFile(input));
  } catch (const std::exception& e) {
    std::cerr << "input: " << e.what() << '\n';
    return 1;
  }
  const int w = static_cast<int>(image.width), h = static_cast<int>(image.height);
  std::string error;
  const auto load_start = std::chrono::steady_clock::now();
  auto upscaler = re::OrtUpscaler::Load(
      options, {{"body", std::filesystem::path(model), {}}}, &error);
  if (upscaler == nullptr) {
    std::cerr << "load: " << error << '\n';
    return 1;
  }
  std::cout << "device " << re::OrtDeviceName(upscaler->device())
            << "\nonnxruntime " << upscaler->runtime_version() << " "
            << upscaler->runtime_library().string() << '\n';
  if (upscaler->device() == re::OrtDevice::kNpu)
    std::cout << "vitisai ep " << upscaler->ep_version() << " "
              << upscaler->ep_library().string() << '\n';
  for (const re::OrtModelInfo& m : upscaler->models())
    std::cout << "model " << m.path.filename().string() << " width "
              << m.width << " pass rows " << m.tile_rows << " cache_key "
              << m.cache_key << " load_ms "
              << std::lround(m.load_seconds * 1000) << '\n';
  std::cout << "load_total_ms "
            << std::lround(std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - load_start)
                               .count() *
                           1000)
            << '\n';
  std::vector<std::uint8_t> out;
  for (int r = 0; r < repeat; ++r) {
    re::UpscaleTimings t;
    if (!upscaler->Upscale("", image.pixels.data(), w, h, out_w, out_h, fit,
                           &out, &t, &error)) {
      std::cerr << "upscale: " << error << '\n';
      return 1;
    }
    std::cout << "run " << r + 1 << ": body_ms "
              << std::lround(t.body_seconds * 1000) << " post_ms "
              << std::lround(t.post_seconds * 1000) << " passes " << t.passes
              << '\n';
  }
  const int ow = out_w == 0 ? w * 4 : out_w;
  const int oh = out_h == 0 ? h * 4 : out_h;
  if (!output.empty()) {
    // Binary PPM (P6): no encoder dependency; the pixels are what counts.
    std::ofstream file(output, std::ios::binary);
    file << "P6\n" << ow << ' ' << oh << "\n255\n";
    file.write(reinterpret_cast<const char*>(out.data()),
               static_cast<std::streamsize>(out.size()));
    std::cout << "wrote " << output << " " << ow << "x" << oh << '\n';
  }
  if (!compare.empty()) {
    gufo::core::Image ref;
    try {
      ref = gufo::core::DecodeImage(gufo::core::ReadImageFile(compare));
    } catch (const std::exception& e) {
      std::cerr << "compare: " << e.what() << '\n';
      return 1;
    }
    if (static_cast<int>(ref.width) != ow || static_cast<int>(ref.height) != oh) {
      std::cerr << "compare: reference is " << ref.width << "x" << ref.height
                << ", output " << ow << "x" << oh << '\n';
      return 1;
    }
    std::size_t differ = 0;
    int max_diff = 0;
    double se = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
      const int d = static_cast<int>(out[i]) - static_cast<int>(ref.pixels[i]);
      differ += d != 0;
      max_diff = std::max(max_diff, std::abs(d));
      se += static_cast<double>(d) * d;
    }
    const double mse = se / static_cast<double>(out.size());
    std::cout << "compare: differing values " << differ << " of "
              << out.size() << ", max diff " << max_diff << ", PSNR "
              << (mse == 0 ? std::string("inf")
                           : std::to_string(10 * std::log10(255.0 * 255.0 / mse)))
              << " dB\n";
    return differ == 0 ? 0 : 3;
  }
  return 0;
}
