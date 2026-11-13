// Ming-3.0-tts speech_synthesis provider (gufo extension API 0.2): the
// TtsRunner behind `gufo serve tts` and the `gufo speak` driver. The engine
// runs the AR loop and VAE decoder on HIP when this target is built with it
// (GUFO_MING_BACKEND=cpu|hip overrides), otherwise on the CPU reference
// runtime. Written with AI assistance (Claude Code).
#include <gufo/ext/generated/ming_tts_constants.hpp>
#include <gufo/ext/json.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/tts.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "ming_tts/ext_request.hpp"
#include "ming_tts/provider.hpp"
#include "ming_tts/synthesis.hpp"
#include "ming_tts/wav.hpp"

namespace gufo::models::ming_tts {

bool ProbeMingTts(std::string_view model_root);
bool ResolveMingTtsArtifacts(std::string_view model_root, std::string* error);

namespace {

class MingTtsRunner final : public ext::TtsRunner {
public:
  explicit MingTtsRunner(std::unique_ptr<Engine> engine)
      : engine_(std::move(engine)) {}

  bool Synthesize(const ext::SynthesisRequest& request,
                  const ext::CancellationCheck& is_cancelled,
                  ext::SynthesisResult* result, std::string* error) override {
    SynthesisRequest ming;
    if (!BuildEngineRequest(request, &ming, error)) {
      return false;
    }
    SynthesisResult out;
    if (!engine_->Generate(ming, is_cancelled, &out, error)) {
      return false;
    }
    if (result != nullptr) {
      result->sample_rate = out.sample_rate;
      // One pseudo code group per AR step, so the core's X-Gufo-Codec-Steps
      // header reports the decode steps as in the 1.x fork.
      result->code_groups = 1;
      result->codes.assign(
          static_cast<std::size_t>(std::accumulate(
              out.fragment_steps.begin(), out.fragment_steps.end(),
              std::int64_t{0})),
          0);
      result->sample_count = out.samples.size();
      result->samples = std::move(out.samples);
    }
    return true;
  }

  std::vector<std::string> Voices() const override {
    return {"default", "voice-clone"};
  }

  ext::TtsVariant Variant() const override { return ext::TtsVariant::kBase; }

  ext::TtsConditioningCapabilities Conditioning() const override {
    return {
        .unconditioned = true,
        .require_voice = false,
        .default_voice = "default",
        .reference_audio = ext::TtsConditioningMode::kOptional,
        .reference_presets = true,
        .instructions = ext::TtsConditioningMode::kOptional,
    };
  }

  std::uint32_t SampleRate() const override { return engine_->sample_rate(); }

  std::string ModelId() const override { return std::string(kMingModelId); }

private:
  std::unique_ptr<Engine> engine_;
};

std::unique_ptr<ext::TtsRunner> CreateRunner(const std::string& model_root,
                                             std::size_t maximum_context_tokens,
                                             std::string* error) {
  (void)maximum_context_tokens;  // fixed 512-position KV cache per fragment
  auto engine = Engine::Create(model_root, error);
  if (engine == nullptr) {
    return nullptr;
  }
  return std::make_unique<MingTtsRunner>(std::move(engine));
}

int RunSpeak(const ext::SpeakOptions& o) {
  if (o.mode != "tts") {
    std::cerr << "Error: Ming-3.0-tts supports only speak mode tts\n";
    return 2;
  }
  if (o.text.empty()) {
    std::cerr << "Error: Ming-3.0-tts speak requires non-empty --text\n";
    return 2;
  }
  ext::SynthesisRequest request;
  request.text = o.text;
  request.language = o.language;
  request.sampling.seed = static_cast<std::uint32_t>(o.seed);
  request.max_new_tokens = o.max_frames;
  // --voice: a reference WAV for speaker-embedding cloning.
  if (!o.voice.empty()) {
    std::ifstream in(o.voice, std::ios::binary);
    const std::string bytes{std::istreambuf_iterator<char>(in),
                            std::istreambuf_iterator<char>()};
    WavAudio audio;
    std::string wav_error;
    if (bytes.empty()) {
      std::cerr << "Error: cannot read --voice " << o.voice << "\n";
      return 2;
    }
    if (!DecodeWav(std::span(reinterpret_cast<const std::byte*>(bytes.data()),
                             bytes.size()),
                   &audio, &wav_error)) {
      std::cerr << "Error: --voice " << o.voice << ": " << wav_error << "\n";
      return 2;
    }
    request.reference_audio = {.sample_rate = audio.sample_rate,
                               .channels = audio.channels,
                               .samples = std::move(audio.samples)};
    request.speaker_embedding_only = true;
  }
  std::string error;
  const auto load_begin = std::chrono::steady_clock::now();
  auto runner = CreateRunner(o.model, o.context, &error);
  if (runner == nullptr) {
    std::cerr << "Error loading Ming-3.0-tts: " << error << "\n";
    return 1;
  }
  const double load_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - load_begin)
                             .count();
  const auto begin = std::chrono::steady_clock::now();
  ext::SynthesisResult result;
  if (!runner->Synthesize(request, o.cancelled, &result, &error)) {
    std::cerr << "Error generating speech: " << error << "\n";
    return (o.cancelled && o.cancelled()) ? 130 : 1;
  }
  const double total_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - begin)
                              .count();
  if (!o.output_path.empty()) {
    const std::string wav = EncodeWavPcm16(result.samples, result.sample_rate);
    std::ofstream out(o.output_path, std::ios::binary);
    out.write(wav.data(), static_cast<std::streamsize>(wav.size()));
    if (!out) {
      std::cerr << "Error: cannot write " << o.output_path << "\n";
      return 1;
    }
  }
  const double seconds = result.sample_rate == 0
                             ? 0.0
                             : static_cast<double>(result.samples.size()) /
                                   result.sample_rate;
  if (o.format == "json") {
    json::Value report = json::Value::object();
    report["schema"] = "gufo.ming-tts.speech.v1";
    report["text"] = o.text;
    report["steps"] = static_cast<std::int64_t>(result.codes.size());
    report["audio_seconds"] = seconds;
    json::Value timings = json::Value::object();
    timings["load_ms"] = load_ms;
    timings["total_ms"] = total_ms;
    report["timings"] = std::move(timings);
    std::cout << report.dump() << "\n";
  } else {
    std::cerr << "Ming-3.0-tts: " << seconds << " s audio in " << total_ms
              << " ms" << (o.output_path.empty() ? "" : ", wrote ")
              << o.output_path << "\n";
  }
  return 0;
}

}  // namespace

std::unique_ptr<ext::TtsRunner> CreateMingTtsRunner(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error) {
  return CreateRunner(model_root, maximum_context_tokens, error);
}

}  // namespace gufo::models::ming_tts

void ming_tts_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_ming_tts::kProvider_ming_tts);
  using namespace gufo::models::ming_tts;
  registry.AddTtsProvider({
      .id = "ming-tts",
      .probe = &ProbeMingTts,
      .resolve = &ResolveMingTtsArtifacts,
      .create = &CreateMingTtsRunner,
      .speak_driver = &RunSpeak,
  });
}
