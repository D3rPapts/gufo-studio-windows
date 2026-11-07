// gufo-ace: text-to-music with ACE-Step 1.5 XL turbo.
//
//   gufo-ace --model ROOT --caption "..." [--lyrics "..." | --lyrics-file F]
//       [--duration 30] [--seed 42] [--bpm 120] [--key "A minor"]
//       [--timesignature 4] [--language en] [--steps 8] [--repeat N]
//       --out song.wav
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "src/models/ace_step/pipeline.hpp"
#include "src/models/ace_step/vae.hpp"

namespace ace = gufo::models::ace_step;

int main(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i + 1 < argc; i += 2)
    args[argv[i]] = argv[i + 1];
  if (!args.count("--model") || !args.count("--caption") ||
      !args.count("--out")) {
    std::fprintf(stderr,
                 "usage: gufo-ace --model ROOT --caption TEXT [--lyrics TEXT | "
                 "--lyrics-file F] [--duration S] [--seed N] [--bpm N] [--key "
                 "K] [--timesignature N] [--language L] [--steps N] "
                 "[--repeat N] --out F.wav\n");
    return 2;
  }
  std::string error;
  ace::ModelPaths paths;
  if (!ace::ModelPaths::FromRoot(args["--model"], &paths, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  auto start = std::chrono::steady_clock::now();
  auto pipeline = ace::MusicPipeline::Load(paths, &error);
  if (!pipeline) {
    std::fprintf(stderr, "load: %s\n", error.c_str());
    return 1;
  }
  std::printf(
      "loaded in %.1f s\n",
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count());
  ace::MusicRequest request;
  request.caption = args["--caption"];
  if (args.count("--lyrics-file")) {
    std::ifstream in(args["--lyrics-file"], std::ios::binary);
    request.lyrics.assign(std::istreambuf_iterator<char>(in), {});
  } else if (args.count("--lyrics")) {
    request.lyrics = args["--lyrics"];
  }
  if (args.count("--duration"))
    request.duration = std::stod(args["--duration"]);
  if (args.count("--seed"))
    request.seed = std::stoull(args["--seed"]);
  if (args.count("--steps"))
    request.steps = std::stoi(args["--steps"]);
  if (args.count("--bpm"))
    request.bpm = args["--bpm"];
  if (args.count("--key"))
    request.keyscale = args["--key"];
  if (args.count("--timesignature"))
    request.timesignature = args["--timesignature"];
  if (args.count("--language"))
    request.language = args["--language"];
  const int repeat = args.count("--repeat") ? std::stoi(args["--repeat"]) : 1;
  std::vector<float> audio;
  for (int r = 0; r < repeat; ++r) {
    ace::MusicTelemetry t;
    start = std::chrono::steady_clock::now();
    if (!pipeline->Generate(request, &audio, &t, nullptr, &error)) {
      std::fprintf(stderr, "generate: %s\n", error.c_str());
      return 1;
    }
    const double total =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::printf(
        "%.1f s audio (%d frames, %d text / %d lyric tokens): text %.2f s, "
        "conditions %.2f s, diffusion %.2f s, vae %.2f s, total %.2f s "
        "(%.1fx real time)\n",
        request.duration, t.frames, t.text_tokens, t.lyric_tokens,
        t.text_seconds, t.condition_seconds, t.diffusion_seconds, t.vae_seconds,
        total, request.duration / total);
  }
  const std::string wav = ace::EncodeStereoWav(audio, ace::kSampleRate);
  std::ofstream(args["--out"], std::ios::binary)
      .write(wav.data(), std::streamsize(wav.size()));
  std::printf("wrote %s\n", args["--out"].c_str());
  return 0;
}
