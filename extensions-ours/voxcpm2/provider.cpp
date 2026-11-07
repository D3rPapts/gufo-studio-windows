// VoxCPM2 speech_synthesis provider (gufo extension API 0.2): the TtsRunner
// behind `gufo serve tts` and the `gufo speak` driver. The AR loop (and the
// AudioVAE) run on HIP when this target is built with it (GUFO_VOXCPM_BACKEND
// and GUFO_VOXCPM_VAE override), otherwise on the CPU reference runtime.
// Written with AI assistance (Claude Code).
#include <gufo/ext/generated/voxcpm2_constants.hpp>
#include <gufo/ext/json.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/tts.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "ming_tts/wav.hpp"
#include "voxcpm2/ext_request.hpp"
#include "voxcpm2/provider.hpp"
#include "voxcpm2/synthesis.hpp"

namespace gufo::models::voxcpm2 {

bool ProbeVoxCpm2(std::string_view model_root);
bool ResolveVoxCpm2Artifacts(std::string_view model_root, std::string* error);

namespace {

class VoxCpm2TtsRunner final : public ext::TtsRunner {
public:
  explicit VoxCpm2TtsRunner(std::unique_ptr<Engine> engine)
      : engine_(std::move(engine)) {}

  bool Synthesize(const ext::SynthesisRequest& request,
                  const ext::CancellationCheck& is_cancelled,
                  ext::SynthesisResult* result, std::string* error) override {
    SynthesisRequest vox;
    if (!BuildEngineRequest(request, &vox, error)) {
      return false;
    }
    SynthesisResult out;
    if (!engine_->Generate(vox, is_cancelled, &out, error)) {
      return false;
    }
    if (result != nullptr) {
      result->sample_rate = out.sample_rate;
      // One pseudo code group per AR step (X-Gufo-Codec-Steps), as in 1.x.
      result->code_groups = 1;
      result->codes.assign(static_cast<std::size_t>(out.steps), 0);
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

  std::string ModelId() const override { return std::string(kVoxCpm2ModelId); }

private:
  std::unique_ptr<Engine> engine_;
};

std::unique_ptr<ext::TtsRunner> CreateRunner(const std::string& model_root,
                                             std::size_t maximum_context_tokens,
                                             std::string* error) {
  (void)maximum_context_tokens;  // the engine sizes its own 8192-position context
  auto engine = Engine::Create(model_root, error);
  if (engine == nullptr) {
    return nullptr;
  }
  return std::make_unique<VoxCpm2TtsRunner>(std::move(engine));
}

int RunSpeak(const ext::SpeakOptions& o) {
  if (o.mode != "tts") {
    std::cerr << "Error: VoxCPM2 supports only speak mode tts\n";
    return 2;
  }
  if (o.text.empty()) {
    std::cerr << "Error: VoxCPM2 speak requires non-empty --text\n";
    return 2;
  }
  ext::SynthesisRequest request;
  request.text = o.text;
  request.sampling.seed = static_cast<std::uint32_t>(o.seed);
  request.max_new_tokens = o.max_frames;
  // --voice: a reference WAV (reference mode cloning).
  if (!o.voice.empty()) {
    std::ifstream in(o.voice, std::ios::binary);
    const std::string bytes{std::istreambuf_iterator<char>(in),
                            std::istreambuf_iterator<char>()};
    ming_tts::WavAudio audio;
    std::string wav_error;
    if (bytes.empty()) {
      std::cerr << "Error: cannot read --voice " << o.voice << "\n";
      return 2;
    }
    if (!ming_tts::DecodeWav(
            std::span(reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()),
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
    std::cerr << "Error loading VoxCPM2: " << error << "\n";
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
    const std::string wav =
        ming_tts::EncodeWavPcm16(result.samples, result.sample_rate);
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
    report["schema"] = "gufo.voxcpm2.speech.v1";
    report["text"] = o.text;
    report["steps"] = static_cast<std::int64_t>(result.codes.size());
    report["audio_seconds"] = seconds;
    json::Value timings = json::Value::object();
    timings["load_ms"] = load_ms;
    timings["total_ms"] = total_ms;
    report["timings"] = std::move(timings);
    std::cout << report.dump() << "\n";
  } else {
    std::cerr << "VoxCPM2: " << seconds << " s audio in " << total_ms << " ms"
              << (o.output_path.empty() ? "" : ", wrote ") << o.output_path
              << "\n";
  }
  return 0;
}

}  // namespace

std::unique_ptr<ext::TtsRunner> CreateVoxCpm2TtsRunner(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error) {
  return CreateRunner(model_root, maximum_context_tokens, error);
}

}  // namespace gufo::models::voxcpm2

void voxcpm2_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_voxcpm2::kProvider_voxcpm2);
  using namespace gufo::models::voxcpm2;
  registry.AddTtsProvider({
      .id = "voxcpm2",
      .probe = &ProbeVoxCpm2,
      .resolve = &ResolveVoxCpm2Artifacts,
      .create = &CreateVoxCpm2TtsRunner,
      .speak_driver = &RunSpeak,
  });
}
