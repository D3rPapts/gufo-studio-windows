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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "src/core/utf8.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen3_asr/audio.hpp"
#include "src/models/qwen3_omni/hip/audio_encoder.hpp"
#include "src/models/qwen3_omni/hip/device_buffer.hpp"
#include "src/models/qwen3_omni/hip/thinker.hpp"
#include "src/models/qwen3_omni/model.hpp"
#include "src/models/qwen3_omni/reference.hpp"

namespace {

namespace omni = gufo::models::qwen3_omni;
namespace asr = gufo::models::qwen3_asr;
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

void WriteFloats(const std::string& path, const std::vector<float>& values) {
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
}

/// Encoder and thinker behind one interface, so the driver is backend-blind.
class Backend {
public:
  virtual ~Backend() = default;
  virtual std::vector<float> Encode(std::span<const float> log_mel,
                                    std::size_t frames) = 0;
  virtual void Forward(std::span<const float> embeddings, std::size_t count,
                       std::vector<float>* logits) = 0;
};

class CpuBackend final : public Backend {
public:
  CpuBackend(const omni::Model& model, unsigned threads)
      : reference_(model, threads) {}
  std::vector<float> Encode(std::span<const float> log_mel,
                            std::size_t frames) override {
    return reference_.EncodeAudio(log_mel, frames);
  }
  void Forward(std::span<const float> embeddings, std::size_t count,
               std::vector<float>* logits) override {
    reference_.Forward(embeddings, count, logits);
  }

private:
  omni::CpuReference reference_;
};

class GpuBackend final : public Backend {
public:
  GpuBackend(const omni::Model& model, std::size_t context)
      : width_(model.audio_config().output) {
    omni::hip::RequireHip(hipStreamCreateWithFlags(&stream_,
                                                   hipStreamNonBlocking),
                          "hipStreamCreate Qwen3-Omni");
    std::string error;
    const auto begin = Clock::now();
    encoder_ = omni::hip::AudioEncoder::Create(model, stream_, &error);
    if (encoder_ == nullptr) {
      throw std::runtime_error("audio encoder: " + error);
    }
    thinker_ = omni::hip::Thinker::Create(model, context, stream_, &error);
    if (thinker_ == nullptr) {
      throw std::runtime_error("thinker: " + error);
    }
    std::cerr << "gpu weights: audio "
              << (encoder_->weight_bytes() >> 20) << " MiB, thinker "
              << (thinker_->weight_bytes() >> 20) << " MiB, uploaded in "
              << Seconds(begin) << " s\n";
  }
  ~GpuBackend() override {
    thinker_.reset();
    encoder_.reset();
    (void)hipStreamDestroy(stream_);
  }
  std::vector<float> Encode(std::span<const float> log_mel,
                            std::size_t frames) override {
    const float* device = nullptr;
    std::size_t tokens = 0;
    std::string error;
    if (!encoder_->Encode(log_mel, frames, &device, &tokens, &error)) {
      throw std::runtime_error(error);
    }
    std::vector<float> host(tokens * width_);
    omni::hip::RequireHip(hipMemcpy(host.data(), device,
                                    host.size() * sizeof(float),
                                    hipMemcpyDeviceToHost),
                          "hipMemcpy Qwen3-Omni audio embeddings");
    return host;
  }
  void Forward(std::span<const float> embeddings, std::size_t count,
               std::vector<float>* logits) override {
    std::string error;
    if (!thinker_->Forward(embeddings, count, logits, &error)) {
      throw std::runtime_error(error);
    }
  }

private:
  std::size_t width_;
  hipStream_t stream_{nullptr};
  std::unique_ptr<omni::hip::AudioEncoder> encoder_;
  std::unique_ptr<omni::hip::Thinker> thinker_;
};

/// Greedy at temperature 0; otherwise top-k, then top-p, then a draw.
std::int32_t Sample(const std::vector<float>& logits, const Arguments& args,
                    std::mt19937_64* rng) {
  if (args.temperature <= 0.0F) {
    return static_cast<std::int32_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
  }
  std::vector<std::int32_t> order(logits.size());
  std::iota(order.begin(), order.end(), 0);
  const std::size_t k = std::clamp<std::size_t>(args.top_k, 1, order.size());
  std::partial_sort(order.begin(), order.begin() + k, order.end(),
                    [&](std::int32_t a, std::int32_t b) {
                      return logits[a] > logits[b];
                    });
  std::vector<double> probabilities(k);
  double total = 0.0;
  for (std::size_t index = 0; index < k; ++index) {
    probabilities[index] =
        std::exp((logits[order[index]] - logits[order[0]]) / args.temperature);
    total += probabilities[index];
  }
  std::size_t kept = 0;
  double cumulative = 0.0;
  while (kept < k) {
    cumulative += probabilities[kept++] / total;
    if (cumulative >= args.top_p) {
      break;
    }
  }
  std::discrete_distribution<std::size_t> draw(probabilities.begin(),
                                               probabilities.begin() + kept);
  return order[draw(*rng)];
}

}  // namespace

int main(int argc, char** argv) {
  Arguments arguments;
  if (!Parse(argc, argv, &arguments)) {
    Usage();
    return 2;
  }
  if (arguments.threads == 0U) {
    arguments.threads = std::max(1U, std::thread::hardware_concurrency() / 2U);
  }
  try {
    auto begin = Clock::now();
    std::string error;
    const auto model =
        omni::Model::Load(arguments.model, arguments.mmproj, &error);
    if (model == nullptr) {
      throw std::runtime_error(error);
    }
    const auto tokenizer = gufo::tokenization::QwenTokenizer::CreateFromGguf(
        model->text_reader(), &error);
    if (tokenizer == nullptr) {
      throw std::runtime_error("tokenizer: " + error);
    }
    std::cerr << "loaded in " << Seconds(begin) << " s\n";

    std::ifstream file(arguments.audio, std::ios::binary);
    if (!file) {
      throw std::runtime_error("cannot read " + arguments.audio);
    }
    const std::vector<char> raw((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
    asr::AudioBuffer buffer;
    if (!asr::DecodeWav(std::as_bytes(std::span(raw)), &buffer, &error)) {
      throw std::runtime_error(error);
    }
    const std::vector<float> waveform = asr::ResampleMono16k(buffer);
    const asr::LogMelFeatures features = asr::ComputeLogMelFeatures(waveform);
    std::cerr << "audio: " << static_cast<double>(waveform.size()) / 16000.0
              << " s, " << features.frames << " mel frames\n";

    std::unique_ptr<Backend> backend;
    if (arguments.backend == "cpu") {
      backend = std::make_unique<CpuBackend>(*model, arguments.threads);
    } else {
      backend = std::make_unique<GpuBackend>(*model, arguments.context);
    }

    begin = Clock::now();
    const std::vector<float> audio =
        backend->Encode(features.values, features.frames);
    const std::size_t hidden = model->text_config().hidden;
    const std::size_t audio_tokens = audio.size() / hidden;
    std::cerr << "audio encoder: " << audio_tokens << " tokens in "
              << Seconds(begin) << " s\n";

    // Captioner prompt: one user turn holding only the audio.
    const omni::TokenIds& ids = model->tokens();
    std::vector<std::int32_t> prompt;
    const auto append = [&](std::string_view text) {
      for (const auto id : tokenizer->Encode(text)) {
        prompt.push_back(static_cast<std::int32_t>(id));
      }
    };
    append("<|im_start|>user\n<|audio_start|>");
    prompt.insert(prompt.end(), audio_tokens, ids.audio_pad);
    append("<|audio_end|><|im_end|>\n<|im_start|>assistant\n");
    if (std::count(prompt.begin(), prompt.end(), ids.audio_pad) !=
            static_cast<std::ptrdiff_t>(audio_tokens) ||
        prompt.front() != ids.im_start) {
      throw std::runtime_error(
          "prompt special tokens do not match the vocabulary");
    }

    std::vector<float> embeddings(prompt.size() * hidden);
    std::size_t next_audio = 0;
    for (std::size_t index = 0; index < prompt.size(); ++index) {
      float* row = embeddings.data() + index * hidden;
      if (prompt[index] == ids.audio_pad) {
        std::copy_n(audio.data() + next_audio++ * hidden, hidden, row);
      } else {
        omni::EmbedToken(*model, prompt[index], row);
      }
    }

    begin = Clock::now();
    std::vector<float> logits;
    backend->Forward(embeddings, prompt.size(), &logits);
    const double prefill = Seconds(begin);
    std::cerr << "prefill: " << prompt.size() << " tokens in " << prefill
              << " s (" << static_cast<double>(prompt.size()) / prefill
              << " tok/s)\n";
    if (!arguments.dump.empty()) {
      WriteFloats(arguments.dump + ".audio.f32", audio);
      WriteFloats(arguments.dump + ".logits.f32", logits);
    }

    begin = Clock::now();
    std::mt19937_64 rng(arguments.seed);
    gufo::core::Utf8Decoder utf8;
    std::vector<float> row(hidden);
    std::vector<std::int32_t> generated;
    while (generated.size() < arguments.max_tokens) {
      const std::int32_t next = Sample(logits, arguments, &rng);
      if (next == ids.im_end || next == ids.endoftext) {
        break;
      }
      generated.push_back(next);
      std::cout << utf8.Push(tokenizer->DecodeTokenCopy(
                       static_cast<gufo::tokenization::TokenId>(next)))
                << std::flush;
      omni::EmbedToken(*model, next, row.data());
      backend->Forward(row, 1, &logits);
    }
    std::cout << utf8.Push("", true) << '\n';
    const double seconds = Seconds(begin);
    std::cerr << "decode: " << generated.size() << " tokens in " << seconds
              << " s ("
              << (seconds > 0.0
                      ? static_cast<double>(generated.size()) / seconds
                      : 0.0)
              << " tok/s)\n";
    if (!arguments.dump.empty()) {
      std::ofstream tokens(arguments.dump + ".tokens.txt");
      for (const std::int32_t id : generated) {
        tokens << id << '\n';
      }
    }
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return 1;
  }
  return 0;
}
