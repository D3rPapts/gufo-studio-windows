// gufo-mmaudio: video / text to audio with MMAudio large_44k_v2.
// Written with AI assistance (Claude Code).
//
//   gufo-mmaudio --model ROOT --prompt TEXT [--negative TEXT]
//       [--video F.mp4 [--consecutive 1] [--force-rate R] | --image F.png]
//       [--duration 8] [--steps 25] [--cfg 4.5] [--seed 0]
//       [--mask-away-clip 1] --out F.wav
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/mmaudio/pipeline.hpp"
#include "src/models/mmaudio/preprocess.hpp"
#include "src/models/mmaudio/video_input.hpp"

namespace mm = gufo::models::mmaudio;

int main(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i + 1 < argc; i += 2)
    args[argv[i]] = argv[i + 1];
  if (!args.count("--model") || !args.count("--out")) {
    std::fprintf(stderr,
                 "usage: gufo-mmaudio --model ROOT --prompt TEXT [--negative "
                 "TEXT] [--video F [--consecutive 1] [--force-rate R] | "
                 "--image F] [--duration S] [--steps N] [--cfg X] [--seed N] "
                 "[--mask-away-clip 1] --out F.wav\n");
    return 2;
  }
  std::string error;
  mm::ModelPaths paths;
  if (!mm::ModelPaths::FromRoot(args["--model"], &paths, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  auto start = std::chrono::steady_clock::now();
  auto pipeline = mm::FoleyPipeline::Load(paths, &error);
  if (!pipeline) {
    std::fprintf(stderr, "load: %s\n", error.c_str());
    return 1;
  }
  std::printf(
      "loaded in %.1f s (%.2f GB weights)\n",
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count(),
      pipeline->weight_bytes() / 1e9);
  mm::FoleyRequest request;
  request.prompt = args["--prompt"];
  request.negative_prompt = args["--negative"];
  if (args.count("--duration"))
    request.duration = std::stod(args["--duration"]);
  if (args.count("--steps"))
    request.steps = std::stoi(args["--steps"]);
  if (args.count("--cfg"))
    request.cfg_strength = std::stof(args["--cfg"]);
  if (args.count("--seed"))
    request.seed = std::stoull(args["--seed"]);
  request.mask_away_clip = args["--mask-away-clip"] == "1";
  start = std::chrono::steady_clock::now();
  if (args.count("--video")) {
    mm::FrameOptions options;
    options.consecutive = args["--consecutive"] == "1";
    if (args.count("--force-rate"))
      options.force_rate = std::stod(args["--force-rate"]);
    if (!mm::LoadVideoFrames(args["--video"], options, &request, &error)) {
      std::fprintf(stderr, "video: %s\n", error.c_str());
      return 1;
    }
    std::printf("video: %d CLIP / %d sync frames, %.3f s, %.1f s\n",
                request.clip_frames, request.sync_frames, request.duration,
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start)
                    .count());
  } else if (args.count("--image")) {
    try {
      const auto image =
          gufo::core::DecodeImage(gufo::core::ReadImageFile(args["--image"]));
      mm::LoadImageFrame(image.pixels, static_cast<int>(image.width),
                         static_cast<int>(image.height), &request);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "image: %s\n", e.what());
      return 1;
    }
  }
  mm::FoleyTelemetry t;
  std::vector<float> audio;
  start = std::chrono::steady_clock::now();
  if (!pipeline->Generate(request, &audio, &t, nullptr, &error)) {
    std::fprintf(stderr, "generate: %s\n", error.c_str());
    return 1;
  }
  const double total =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  std::printf(
      "%.2f s audio (%zu samples): features %.2f s, sampling %.2f s, decode "
      "%.2f s, total %.2f s\n",
      request.duration, audio.size(), t.features_seconds, t.sampling_seconds,
      t.decode_seconds, total);
  const std::string wav = mm::EncodeMonoWav(audio, mm::kSampleRate);
  std::ofstream(args["--out"], std::ios::binary)
      .write(wav.data(), static_cast<std::streamsize>(wav.size()));
  std::printf("wrote %s\n", args["--out"].c_str());
  return 0;
}
