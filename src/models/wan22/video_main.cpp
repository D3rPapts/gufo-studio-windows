// gufo-wan-video: text to video with Wan 2.2 TI2V-5B.
//
//   gufo-wan-video --model ROOT --prompt TEXT [--negative TEXT]
//                  [--width W] [--height H] [--frames N] [--steps S]
//                  [--guidance G] [--seed N] [--fps N] [--out FILE.mp4]
//                  [--image FILE] [--oracle DIR [--replay]]
//   gufo-wan-video --model ROOT --vae-oracle DIR
//   gufo-wan-video --model ROOT --encoder-oracle DIR
//   gufo-wan-video --scheduler-oracle DIR [--model ROOT]
//   gufo-wan-video --model ROOT --vae-bench FRAMES HEIGHT WIDTH
//
// `--oracle` takes the initial noise from DIR/noise.npy and compares the
// latents after every step and the decoded video with the Diffusers run of
// tools-local/wan_pipe_oracle.py. `--scheduler-oracle` replays the reference
// noise predictions through the UniPC scheduler (no GPU). `--vae-oracle`
// decodes DIR/vae_in.npy and compares with the untiled DIR/vae_out.npy.
// `--image` makes it image-to-video from FILE (PNG/JPEG/WebP, fitted to the
// video size); with `--oracle` (tools-local/wan_i2v_oracle.py) the final
// blended latents are compared too; `--replay` starts every step from the
// reference sample and compares each step's guided noise prediction.
// `--encoder-oracle` encodes DIR/enc*_in.npy and compares the posterior means
// with DIR/enc*_mu.npy.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/wan22/pipeline.hpp"
#include "src/models/wan22/scheduler.hpp"
#include "src/models/wan22/vae.hpp"

namespace {

namespace wan22 = gufo::models::wan22;
using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point begin) {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

// A little-endian, C-order FP32 .npy array.
bool ReadNpy(const std::filesystem::path& path, std::vector<float>* values,
             std::vector<std::size_t>* shape) {
  std::ifstream in(path, std::ios::binary);
  char magic[8];
  if (!in.read(magic, 8) || std::memcmp(magic, "\x93NUMPY", 6) != 0)
    return false;
  std::uint32_t header_size = 0;
  if (magic[6] == 1) {
    std::uint16_t size16 = 0;
    in.read(reinterpret_cast<char*>(&size16), 2);
    header_size = size16;
  } else {
    in.read(reinterpret_cast<char*>(&header_size), 4);
  }
  std::string header(header_size, '\0');
  if (!in.read(header.data(), header_size) ||
      header.find("'descr': '<f4'") == std::string::npos ||
      header.find("'fortran_order': False") == std::string::npos)
    return false;
  const auto open = header.find('(');
  const auto close = header.find(')');
  if (open == std::string::npos || close == std::string::npos)
    return false;
  shape->clear();
  std::size_t count = 1;
  for (std::size_t pos = open + 1; pos < close;) {
    const auto end = std::min(header.find(',', pos), close);
    const std::string item = header.substr(pos, end - pos);
    if (item.find_first_of("0123456789") != std::string::npos) {
      shape->push_back(std::stoull(item));
      count *= shape->back();
    }
    pos = end + 1;
  }
  values->resize(count);
  return static_cast<bool>(in.read(reinterpret_cast<char*>(values->data()),
                                   static_cast<std::streamsize>(count * 4)));
}

struct Comparison {
  double cosine{0};
  double max_error{0};
  double mean_error{0};
};

Comparison Compare(const float* a, const float* b, std::size_t count) {
  double dot = 0, a2 = 0, b2 = 0, error = 0, sum = 0;
  for (std::size_t i = 0; i < count; ++i) {
    dot += double{a[i]} * b[i];
    a2 += double{a[i]} * a[i];
    b2 += double{b[i]} * b[i];
    const double e = std::abs(double{a[i]} - b[i]);
    error = std::max(error, e);
    sum += e;
  }
  return {dot / std::sqrt(a2 * b2), error,
          sum / std::max<std::size_t>(count, 1)};
}

void Print(const std::string& name, const Comparison& c) {
  std::cout << std::defaultfloat << std::setprecision(7) << name << ": cosine "
            << c.cosine << ", max abs " << c.max_error << ", mean abs "
            << c.mean_error << "\n";
}

// [frames, H, W, 3] interleaved -> [3, frames, H, W].
std::vector<float> ChannelFirst(const wan22::DecodedVideo& video) {
  const std::size_t plane =
      static_cast<std::size_t>(video.height) * video.width;
  std::vector<float> out(video.rgb.size());
  for (int f = 0; f < video.frames; ++f)
    for (std::size_t p = 0; p < plane; ++p)
      for (int c = 0; c < 3; ++c)
        out[(static_cast<std::size_t>(c) * video.frames + f) * plane + p] =
            video.rgb[(f * plane + p) * 3 + c];
  return out;
}

// Per-frame cosine of channel-first videos; returns the minimum.
double FrameCosines(const std::vector<float>& ours,
                    const std::vector<float>& reference, int frames,
                    std::size_t plane, bool print) {
  double minimum = 1;
  for (int f = 0; f < frames; ++f) {
    std::vector<float> a, b;
    for (int c = 0; c < 3; ++c) {
      const std::size_t offset =
          (static_cast<std::size_t>(c) * frames + f) * plane;
      a.insert(a.end(), ours.begin() + offset, ours.begin() + offset + plane);
      b.insert(b.end(), reference.begin() + offset,
               reference.begin() + offset + plane);
    }
    const Comparison c = Compare(a.data(), b.data(), a.size());
    if (print)
      Print("  frame " + std::to_string(f), c);
    minimum = std::min(minimum, c.cosine);
  }
  return minimum;
}

int SchedulerOracle(const std::filesystem::path& dir,
                    const std::filesystem::path& model) {
  const auto config = wan22::ReadUniPcConfig(
      model.empty() ? std::filesystem::path(
                          "D:/llm-modelle/Wan-AI/Wan2.2-TI2V-5B-Diffusers/"
                          "scheduler/scheduler_config.json")
                    : model / "scheduler" / "scheduler_config.json");
  std::vector<float> sigmas, timesteps, sample;
  std::vector<std::size_t> shape;
  if (!config || !ReadNpy(dir / "sigmas.npy", &sigmas, &shape) ||
      !ReadNpy(dir / "timesteps.npy", &timesteps, &shape) ||
      !ReadNpy(dir / "step00_sample.npy", &sample, &shape)) {
    std::cerr << "cannot read scheduler config or oracle\n";
    return 1;
  }
  const int steps = static_cast<int>(timesteps.size());
  wan22::UniPcScheduler scheduler(*config, steps);
  bool pass = scheduler.sigmas().size() == sigmas.size();
  double sigma_error = 0;
  for (std::size_t i = 0; pass && i < sigmas.size(); ++i)
    sigma_error = std::max(sigma_error,
                           std::abs(double{scheduler.sigmas()[i]} - sigmas[i]));
  for (int i = 0; pass && i < steps; ++i)
    pass = scheduler.timesteps()[i] == static_cast<std::int64_t>(timesteps[i]);
  std::cout << "timesteps " << (pass ? "EQUAL" : "DIFFERENT")
            << ", max sigma error " << sigma_error << "\n";
  double min_cosine = 1;
  for (int i = 0; i < steps; ++i) {
    std::vector<float> noise, expected;
    char name[32];
    std::snprintf(name, sizeof(name), "step%02d_noise.npy", i);
    if (!ReadNpy(dir / name, &noise, &shape))
      return 1;
    scheduler.Step(noise, sample);
    std::snprintf(name, sizeof(name), "step%02d_sample.npy", i + 1);
    const std::filesystem::path next =
        i + 1 < steps ? dir / name : dir / "latents_final.npy";
    if (!ReadNpy(next, &expected, &shape) || expected.size() != sample.size())
      return 1;
    const Comparison c = Compare(sample.data(), expected.data(), sample.size());
    Print("after step " + std::to_string(i), c);
    min_cosine = std::min(min_cosine, c.cosine);
  }
  pass = pass && sigma_error < 1e-6 && min_cosine >= 0.99999;
  std::cout << (pass ? "PASS" : "FAIL") << "\n";
  return pass ? 0 : 1;
}

int VaeOracle(const std::filesystem::path& model,
              const std::filesystem::path& dir) {
  std::vector<float> latent, reference;
  std::vector<std::size_t> shape, reference_shape;
  if (!ReadNpy(dir / "vae_in.npy", &latent, &shape) || shape.size() != 4 ||
      !ReadNpy(dir / "vae_out.npy", &reference, &reference_shape)) {
    std::cerr << "cannot read VAE oracle in " << dir << "\n";
    return 1;
  }
  std::string error;
  auto start = Clock::now();
  auto vae = wan22::VaeDecoder::Load(model / "vae", &error);
  if (!vae) {
    std::cerr << "vae load: " << error << "\n";
    return 1;
  }
  std::cout << "vae load " << Seconds(start) << " s, weights "
            << vae->weight_bytes() / 1048576.0 << " MiB\n";
  wan22::DecodedVideo video;
  for (int run = 0; run < 2; ++run) {
    start = Clock::now();
    if (!vae->Decode(latent, static_cast<int>(shape[1]),
                     static_cast<int>(shape[2]), static_cast<int>(shape[3]),
                     true, false, &video, nullptr, nullptr, &error)) {
      std::cerr << "decode: " << error << "\n";
      return 1;
    }
    std::cout << "decode run " << run << ": " << Seconds(start) << " s ("
              << video.frames << " frames " << video.width << "x"
              << video.height << "), activations "
              << vae->activation_bytes() / 1048576.0 << " MiB\n";
  }
  const auto ours = ChannelFirst(video);
  if (ours.size() != reference.size()) {
    std::cerr << "size mismatch " << ours.size() << " vs " << reference.size()
              << "\n";
    return 1;
  }
  const Comparison c = Compare(ours.data(), reference.data(), ours.size());
  Print("vae_out", c);
  const double minimum =
      FrameCosines(ours, reference, video.frames,
                   static_cast<std::size_t>(video.height) * video.width, true);
  const bool pass = c.cosine >= 0.999 && minimum >= 0.999;
  std::cout << (pass ? "PASS" : "FAIL") << "\n";
  return pass ? 0 : 1;
}

int EncoderOracle(const std::filesystem::path& model,
                  const std::filesystem::path& dir) {
  std::string error;
  auto start = Clock::now();
  auto encoder = wan22::VaeEncoder::Load(model / "vae", &error);
  if (!encoder) {
    std::cerr << "encoder load: " << error << "\n";
    return 1;
  }
  std::cout << "encoder load " << Seconds(start) << " s, weights "
            << encoder->weight_bytes() / 1048576.0 << " MiB\n";
  bool pass = true;
  for (const char* name : {"enc", "enc2"}) {
    std::vector<float> image, reference;
    std::vector<std::size_t> shape, reference_shape;
    if (!ReadNpy(dir / (std::string(name) + "_in.npy"), &image, &shape) ||
        shape.size() != 3 || shape[0] != 3 ||
        !ReadNpy(dir / (std::string(name) + "_mu.npy"), &reference,
                 &reference_shape)) {
      std::cerr << "cannot read " << name << " oracle in " << dir << "\n";
      return 1;
    }
    const int height = static_cast<int>(shape[1]);
    const int width = static_cast<int>(shape[2]);
    const std::size_t plane = static_cast<std::size_t>(height) * width;
    std::vector<float> rgb(plane * 3);
    for (int c = 0; c < 3; ++c)
      for (std::size_t i = 0; i < plane; ++i)
        rgb[i * 3 + c] = image[c * plane + i];
    std::vector<float> latent;
    for (int run = 0; run < 2; ++run) {
      start = Clock::now();
      if (!encoder->Encode(rgb, height, width, false, &latent, &error)) {
        std::cerr << "encode: " << error << "\n";
        return 1;
      }
      std::cout << name << " " << width << "x" << height << " encode run "
                << run << ": " << Seconds(start) * 1000 << " ms, activations "
                << encoder->activation_bytes() / 1048576.0 << " MiB\n";
    }
    if (latent.size() != reference.size()) {
      std::cerr << "size mismatch " << latent.size() << " vs "
                << reference.size() << "\n";
      return 1;
    }
    const Comparison c =
        Compare(latent.data(), reference.data(), latent.size());
    Print(std::string(name) + "_mu", c);
    pass = pass && c.cosine >= 0.999;
    if (std::string(name) == "enc" &&
        ReadNpy(dir / "condition.npy", &reference, &reference_shape) &&
        encoder->Encode(rgb, height, width, true, &latent, &error) &&
        latent.size() == reference.size())
      Print("condition",
            Compare(latent.data(), reference.data(), latent.size()));
  }
  std::cout << (pass ? "PASS" : "FAIL") << "\n";
  return pass ? 0 : 1;
}

int VaeBench(const std::filesystem::path& model, int frames, int height,
             int width) {
  std::string error;
  auto vae = wan22::VaeDecoder::Load(model / "vae", &error);
  if (!vae) {
    std::cerr << "vae load: " << error << "\n";
    return 1;
  }
  const auto latent = wan22::GaussianNoise(
      static_cast<std::size_t>(48) * frames * height * width, 7);
  wan22::DecodedVideo video;
  auto start = Clock::now();
  auto last = start;
  if (!vae->Decode(
          latent, frames, height, width, false, true, &video,
          [&](int done, int total) {
            std::cout << "  latent frame " << done << "/" << total << " "
                      << Seconds(last) << " s\n";
            last = Clock::now();
          },
          nullptr, &error)) {
    std::cerr << "decode: " << error << "\n";
    return 1;
  }
  std::cout << "vae decode " << video.frames << " frames " << video.width << "x"
            << video.height << ": " << Seconds(start) << " s, activations "
            << vae->activation_bytes() / 1048576.0 << " MiB\n";
  return 0;
}

int Usage() {
  std::cerr
      << "usage: gufo-wan-video --model ROOT --prompt TEXT [--negative TEXT] "
         "[--width W] [--height H] [--frames N] [--steps S] [--guidance G] "
         "[--seed N] [--fps N] [--out FILE.mp4] [--image FILE] "
         "[--oracle DIR [--replay]]\n"
         "       gufo-wan-video --model ROOT --vae-oracle DIR\n"
         "       gufo-wan-video --model ROOT --encoder-oracle DIR\n"
         "       gufo-wan-video --scheduler-oracle DIR [--model ROOT]\n"
         "       gufo-wan-video --model ROOT --vae-bench FRAMES HEIGHT WIDTH\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  wan22::VideoRequest request;
  std::string out = "wan-video.mp4", oracle, vae_oracle, scheduler_oracle;
  std::string image_path, encoder_oracle;
  bool replay = false;
  int fps = 24;
  int bench[3]{};
  bool have_prompt = false;
  for (int i = 1; i < argc; ++i) {
    const std::string name = argv[i];
    const bool more = i + 1 < argc;
    if (name == "--model" && more)
      request.model_root = argv[++i];
    else if (name == "--prompt" && more) {
      request.prompt = argv[++i];
      have_prompt = true;
    } else if (name == "--negative" && more)
      request.negative_prompt = argv[++i];
    else if (name == "--width" && more)
      request.width = std::atoi(argv[++i]);
    else if (name == "--height" && more)
      request.height = std::atoi(argv[++i]);
    else if (name == "--frames" && more)
      request.frames = std::atoi(argv[++i]);
    else if (name == "--steps" && more)
      request.steps = std::atoi(argv[++i]);
    else if (name == "--guidance" && more)
      request.guidance = static_cast<float>(std::atof(argv[++i]));
    else if (name == "--seed" && more)
      request.seed = std::strtoull(argv[++i], nullptr, 10);
    else if (name == "--fps" && more)
      fps = std::atoi(argv[++i]);
    else if (name == "--out" && more)
      out = argv[++i];
    else if (name == "--replay")
      replay = true;
    else if (name == "--oracle" && more)
      oracle = argv[++i];
    else if (name == "--vae-oracle" && more)
      vae_oracle = argv[++i];
    else if (name == "--encoder-oracle" && more)
      encoder_oracle = argv[++i];
    else if (name == "--image" && more)
      image_path = argv[++i];
    else if (name == "--scheduler-oracle" && more)
      scheduler_oracle = argv[++i];
    else if (name == "--vae-bench" && i + 3 < argc)
      for (int& value : bench)
        value = std::atoi(argv[++i]);
    else
      return Usage();
  }
  if (!scheduler_oracle.empty())
    return SchedulerOracle(scheduler_oracle, request.model_root);
  if (request.model_root.empty())
    return Usage();
  if (!vae_oracle.empty())
    return VaeOracle(request.model_root, vae_oracle);
  if (!encoder_oracle.empty())
    return EncoderOracle(request.model_root, encoder_oracle);
  if (bench[0] > 0)
    return VaeBench(request.model_root, bench[0], bench[1], bench[2]);
  if (!have_prompt)
    return Usage();

  std::vector<float> noise;
  std::vector<std::size_t> shape;
  const std::filesystem::path dir(oracle);
  if (!oracle.empty()) {
    if (!ReadNpy(dir / "noise.npy", &noise, &shape) || shape.size() != 4) {
      std::cerr << "cannot read " << (dir / "noise.npy") << "\n";
      return 1;
    }
    request.noise = noise;
  }
  std::vector<std::uint8_t> first_frame;
  if (!image_path.empty()) {
    try {
      const auto image =
          gufo::core::DecodeImage(gufo::core::ReadImageFile(image_path));
      first_frame = wan22::FitImage(image.pixels, static_cast<int>(image.width),
                                    static_cast<int>(image.height),
                                    request.width, request.height);
      std::cout << "first frame " << image_path << " (" << image.width << "x"
                << image.height << ")\n";
    } catch (const std::exception& e) {
      std::cerr << "cannot read " << image_path << ": " << e.what() << "\n";
      return 1;
    }
    request.first_frame = first_frame;
  }
  std::cout << "prompt \"" << request.prompt << "\", " << request.width << "x"
            << request.height << ", " << request.frames << " frames, "
            << request.steps << " steps, guidance " << request.guidance
            << ", seed " << request.seed
            << (oracle.empty() ? "" : " (oracle noise)") << "\n";
  double min_step_cosine = 1;
  wan22::VideoCallbacks callbacks;
  callbacks.progress = [](std::string_view phase, int done, int total) {
    if (phase != "denoiser")
      std::cout << "  " << phase << " " << done << "/" << total << "\n";
  };
  callbacks.on_step = [&](int step, std::span<const float> latents) {
    if (oracle.empty())
      return;
    char name[32];
    std::snprintf(name, sizeof(name), "step%02d_sample.npy", step + 1);
    std::vector<float> expected;
    std::vector<std::size_t> expected_shape;
    // Image-to-video: latents_final.npy is blended, compared further down.
    const auto path = std::filesystem::exists(dir / name) || !image_path.empty()
                          ? dir / name
                          : dir / "latents_final.npy";
    if (!ReadNpy(path, &expected, &expected_shape) ||
        expected.size() != latents.size())
      return;
    const Comparison c =
        Compare(latents.data(), expected.data(), latents.size());
    Print("latents after step " + std::to_string(step), c);
    min_step_cosine = std::min(min_step_cosine, c.cosine);
  };
  // --replay: every step starts from the reference sample (teacher forcing),
  // so the guided noise prediction of each step is compared on its own.
  double min_noise_cosine = 1;
  if (replay && !oracle.empty()) {
    callbacks.before_step = [&](int step, std::vector<float>& latents) {
      char name[32];
      std::snprintf(name, sizeof(name), "step%02d_sample.npy", step);
      std::vector<float> sample;
      std::vector<std::size_t> sample_shape;
      if (ReadNpy(dir / name, &sample, &sample_shape) &&
          sample.size() == latents.size())
        latents = sample;
    };
    callbacks.on_noise = [&](int step, std::span<const float> noise) {
      char name[32];
      std::snprintf(name, sizeof(name), "step%02d_noise.npy", step);
      std::vector<float> expected;
      std::vector<std::size_t> expected_shape;
      if (!ReadNpy(dir / name, &expected, &expected_shape) ||
          expected.size() != noise.size())
        return;
      const Comparison c = Compare(noise.data(), expected.data(), noise.size());
      Print("replayed noise of step " + std::to_string(step), c);
      if (image_path.empty()) {
        min_noise_cosine = std::min(min_noise_cosine, c.cosine);
        return;
      }
      // Image-to-video replaces latent frame 0, so its prediction is unused:
      // compare frames 1.. on their own.
      const int frames = (request.frames - 1) / 4 + 1;
      const std::size_t plane =
          static_cast<std::size_t>(request.height / 16) * (request.width / 16);
      std::vector<float> a, b;
      for (int ch = 0; ch < wan22::VaeDecoder::kLatentChannels; ++ch) {
        const std::size_t offset =
            (static_cast<std::size_t>(ch) * frames + 1) * plane;
        const std::size_t count = static_cast<std::size_t>(frames - 1) * plane;
        a.insert(a.end(), noise.begin() + offset,
                 noise.begin() + offset + count);
        b.insert(b.end(), expected.begin() + offset,
                 expected.begin() + offset + count);
      }
      const Comparison used = Compare(a.data(), b.data(), a.size());
      Print("  frames 1.. of step " + std::to_string(step), used);
      min_noise_cosine = std::min(min_noise_cosine, used.cosine);
    };
  }
  wan22::DecodedVideo video;
  wan22::VideoTelemetry telemetry;
  std::string error;
  std::vector<float> final_latents;
  if (!wan22::GenerateVideo(request, callbacks, &final_latents, &video,
                            !oracle.empty(), &telemetry, &error)) {
    std::cerr << "generate: " << error << "\n";
    return 1;
  }
  const auto encode_start = Clock::now();
  if (!wan22::WriteMp4(out, video, fps, &error)) {
    std::cerr << "mp4: " << error << "\n";
    return 1;
  }
  const double mp4_s = Seconds(encode_start);
  double step_sum = 0, forward_sum = 0;
  for (double ms : telemetry.step_ms)
    step_sum += ms;
  for (double ms : telemetry.forward_ms)
    forward_sum += ms;
  const double steps_done =
      static_cast<double>(std::max<std::size_t>(telemetry.step_ms.size(), 1));
  std::cout << std::fixed << std::setprecision(2) << "tokens "
            << telemetry.prompt_tokens << " / negative "
            << telemetry.negative_tokens << "\nper step: transformer "
            << forward_sum / 1000 / steps_done << " s ("
            << telemetry.forward_ms.size() << " calls), rest "
            << (step_sum - forward_sum) / 1000 / steps_done << " s\ntext "
            << telemetry.text_ms / 1000 << " s, image "
            << telemetry.image_ms / 1000 << " s, transformer load "
            << telemetry.transformer_load_ms / 1000 << " s, denoise "
            << telemetry.denoise_ms / 1000 << " s ("
            << step_sum / 1000 /
                   std::max<std::size_t>(telemetry.step_ms.size(), 1)
            << " s/step), vae load " << telemetry.vae_load_ms / 1000
            << " s, vae " << telemetry.vae_ms / 1000 << " s, mp4 " << mp4_s
            << " s, total " << telemetry.total_ms / 1000 + mp4_s
            << " s\npeak device memory in use "
            << telemetry.peak_device_bytes / 1073741824.0 << " GiB\nwrote "
            << out << " (" << video.frames << " frames " << video.width << "x"
            << video.height << " @ " << fps << " fps)\n";
  if (oracle.empty())
    return 0;
  std::vector<float> reference;
  if (ReadNpy(dir / "latents_final.npy", &reference, &shape) &&
      reference.size() == final_latents.size())
    Print("final latents",
          Compare(final_latents.data(), reference.data(), reference.size()));
  if (!ReadNpy(dir / "video.npy", &reference, &shape))
    return 1;
  const auto ours = ChannelFirst(video);
  if (ours.size() != reference.size()) {
    std::cerr << "video size mismatch\n";
    return 1;
  }
  Print("video (vs tiled Diffusers decode)",
        Compare(ours.data(), reference.data(), ours.size()));
  const double min_frame =
      FrameCosines(ours, reference, video.frames,
                   static_cast<std::size_t>(video.height) * video.width, false);
  if (replay)
    std::cout << std::defaultfloat << "min replayed noise cosine "
              << min_noise_cosine << "\n";
  std::cout << std::defaultfloat << "min latent cosine over steps "
            << min_step_cosine << ", min frame cosine " << min_frame << "\n";
  return 0;
}
