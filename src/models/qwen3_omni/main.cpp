// gufo-omni: captions one audio file with Qwen3-Omni-30B-A3B-Captioner.
//
//   gufo-omni --model Captioner.Q4_K_M.gguf --mmproj Captioner.mmproj-f16.gguf
//             --audio clip.wav [--backend gpu|cpu] [--max-tokens N]
//             [--context N] [--threads N] [--temperature T --top-k K
//             --top-p P --seed S] [--dump PREFIX]
//
// `--backend cpu` runs the F32 CPU reference (no GPU memory needed). `--dump`
// writes PREFIX.audio.f32 (audio embeddings), PREFIX.logits.f32 (logits after
// the prompt) and PREFIX.tokens.txt (generated ids), so two backends can be
// compared offline. Sampling defaults to greedy.

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen3_omni/captioner.hpp"

namespace {

namespace omni = gufo::models::qwen3_omni;
using Clock = std::chrono::steady_clock;

struct Arguments {
  std::string model;
  std::string mmproj;
  std::string audio;
  std::string backend{"gpu"};
  std::string dump;
  std::size_t max_tokens{1024};
  std::size_t context{8192};
  unsigned threads{0};
  float temperature{0.0F};
  std::size_t top_k{20};
  float top_p{0.95F};
  std::uint64_t seed{1234};
};

void Usage() {
  std::cerr << "usage: gufo-omni --model TEXT.gguf --mmproj MMPROJ.gguf "
               "--audio FILE.wav [--backend gpu|cpu] [--max-tokens N] "
               "[--context N] [--threads N] [--temperature T] [--top-k K] "
               "[--top-p P] [--seed S] [--dump PREFIX]\n";
}

bool Parse(int argc, char** argv, Arguments* arguments) {
  for (int index = 1; index < argc; ++index) {
    const std::string name = argv[index];
    if (index + 1 >= argc) {
      return false;
    }
    const std::string value = argv[++index];
    if (name == "--model") {
      arguments->model = value;
    } else if (name == "--mmproj") {
      arguments->mmproj = value;
    } else if (name == "--audio") {
      arguments->audio = value;
    } else if (name == "--backend") {
      arguments->backend = value;
    } else if (name == "--dump") {
      arguments->dump = value;
    } else if (name == "--max-tokens") {
      arguments->max_tokens = std::stoul(value);
    } else if (name == "--context") {
      arguments->context = std::stoul(value);
    } else if (name == "--threads") {
      arguments->threads = static_cast<unsigned>(std::stoul(value));
    } else if (name == "--temperature") {
      arguments->temperature = std::stof(value);
    } else if (name == "--top-k") {
      arguments->top_k = std::stoul(value);
    } else if (name == "--top-p") {
      arguments->top_p = std::stof(value);
    } else if (name == "--seed") {
      arguments->seed = std::stoull(value);
    } else {
      return false;
    }
  }
  return !arguments->model.empty() && !arguments->mmproj.empty() &&
         !arguments->audio.empty() &&
         (arguments->backend == "gpu" || arguments->backend == "cpu");
}

double Seconds(Clock::time_point begin) {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

double PerSecond(std::size_t count, double milliseconds) {
  return milliseconds > 0.0 ? 1000.0 * static_cast<double>(count) / milliseconds
                            : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
  Arguments arguments;
  if (!Parse(argc, argv, &arguments)) {
    Usage();
    return 2;
  }
  const auto begin = Clock::now();
  std::string error;
  const auto captioner = omni::Captioner::Create(
      {.model = arguments.model,
       .mmproj = arguments.mmproj,
       .backend = arguments.backend,
       .context = arguments.context,
       .threads = arguments.threads},
      &error);
  if (captioner == nullptr) {
    std::cerr << "error: " << error << '\n';
    return 1;
  }
  std::cerr << "loaded in " << Seconds(begin) << " s\n";

  std::ifstream file(arguments.audio, std::ios::binary);
  if (!file) {
    std::cerr << "error: cannot read " << arguments.audio << '\n';
    return 1;
  }
  const std::vector<char> raw((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
  omni::CaptionResult result;
  const bool ok = captioner->Caption(
      {.wav = std::as_bytes(std::span(raw)),
       .max_tokens = arguments.max_tokens,
       .temperature = arguments.temperature,
       .top_k = arguments.top_k,
       .top_p = arguments.top_p,
       .seed = arguments.seed,
       .on_text =
           [](std::string_view piece) {
             std::cout << piece << std::flush;
             return true;
           },
       .dump_prefix = arguments.dump},
      {}, &result, &error);
  std::cout << '\n';
  if (!ok) {
    std::cerr << "error: " << error << '\n';
    return 1;
  }
  std::cerr << "audio: " << result.audio_seconds << " s, "
            << result.audio_tokens << " tokens, encoder "
            << result.encoder_ms / 1000.0 << " s\n"
            << "prefill: " << result.prompt_tokens << " tokens in "
            << result.prefill_ms / 1000.0 << " s ("
            << PerSecond(result.prompt_tokens, result.prefill_ms)
            << " tok/s)\n"
            << "decode: " << result.completion_tokens << " tokens in "
            << result.decode_ms / 1000.0 << " s ("
            << PerSecond(result.completion_tokens, result.decode_ms)
            << " tok/s), finish " << result.finish_reason << '\n';
  return 0;
}
