// IndexTTS-2.5 speech_synthesis provider (gufo extension API 0.2): the
// TtsRunner behind `gufo serve tts` and the `gufo speak` driver. GPT, CFM and
// BigVGAN run on HIP when this target is built with it and a GPU is found
// (GUFO_INDEXTTS_BACKEND=cpu|hip overrides), otherwise on the CPU reference
// runtime. Output 22.05 kHz. Written with AI assistance (Claude Code).
#include <gufo/ext/generated/indextts_constants.hpp>
#include <gufo/ext/json.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/tts.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "indextts/ext_request.hpp"
#include "indextts/provider.hpp"
#include "indextts/synthesis.hpp"
#include "ming_tts/wav.hpp"

namespace gufo::models::indextts {

bool ProbeIndexTts(std::string_view model_root);
bool ResolveIndexTtsArtifacts(std::string_view model_root, std::string* error);

namespace {

class IndexTtsRunner final : public ext::TtsRunner {
public:
  explicit IndexTtsRunner(std::unique_ptr<Engine> engine) : engine_(std::move(engine)) {}

  bool Synthesize(const ext::SynthesisRequest& request,
                  const ext::CancellationCheck& is_cancelled,
                  ext::SynthesisResult* result, std::string* error) override {
    SynthesisRequest req;
    if (!BuildEngineRequest(request, &req, error)) {
      return false;
    }
    bool stopped = false;
    Engine::ChunkFn on_chunk;
    if (request.on_audio) {
      on_chunk = [&](std::span<const std::int16_t> pcm) {
        std::vector<float> chunk(pcm.size());
        std::ranges::transform(pcm, chunk.begin(), Int16ToFloat);
        if (!stopped && !request.on_audio(chunk)) {
          stopped = true;
        }
      };
    }
    const auto cancelled = [&] { return stopped || (is_cancelled && is_cancelled()); };
    SynthesisResult res;
    if (!engine_->Generate(req, cancelled, on_chunk, &res, error)) {
      return false;
    }
    if (result != nullptr) {
      result->sample_rate = static_cast<std::uint32_t>(kOutputRate);
      // One pseudo code group per GPT step (X-Gufo-Codec-Steps), as in 1.x.
      result->code_groups = 1;
      result->codes.assign(static_cast<std::size_t>(std::max(res.gpt_steps, 0)), 0);
      result->sample_count = res.samples.size();
      result->samples.clear();
      if (!request.on_audio) {  // streaming results do not buffer the waveform
        result->samples.resize(res.samples.size());
        std::ranges::transform(res.samples, result->samples.begin(), Int16ToFloat);
      }
    }
    return true;
  }

  std::vector<std::string> Voices() const override { return {std::string(kCloneVoice)}; }

  ext::TtsVariant Variant() const override { return ext::TtsVariant::kBase; }

  // Every synthesis clones a voice: a --voice preset or inline reference audio.
  ext::TtsConditioningCapabilities Conditioning() const override {
    return {
        .unconditioned = false,
        .require_voice = false,
        .default_voice = std::string(kCloneVoice),
        .reference_audio = ext::TtsConditioningMode::kRequired,
        .reference_presets = true,
        .instructions = ext::TtsConditioningMode::kOptional,
    };
  }

  std::uint32_t SampleRate() const override { return static_cast<std::uint32_t>(kOutputRate); }

  std::string ModelId() const override { return std::string(kIndexTtsModelId); }

private:
  std::unique_ptr<Engine> engine_;
};

std::unique_ptr<ext::TtsRunner> CreateRunner(const std::string& model_root,
                                             std::size_t maximum_context_tokens,
                                             std::string* error) {
  (void)maximum_context_tokens;  // the GPT sizes its own context (max_mel_tokens)
  auto engine = Engine::Create(std::filesystem::path(model_root), error);
  if (engine == nullptr) {
    return nullptr;
  }
  return std::make_unique<IndexTtsRunner>(std::move(engine));
}

// `gufo speak --model <bundle> --voice <reference.wav> --text "..."`. The
// emotion cannot be passed here (SpeakOptions has no instructions field);
// sampling uses the indextts_de.py defaults.
int RunSpeak(const ext::SpeakOptions& o) {
  if (o.mode != "tts") {
    std::cerr << "Error: IndexTTS-2.5 supports only speak mode tts\n";
    return 2;
  }
  if (o.text.empty()) {
    std::cerr << "Error: IndexTTS-2.5 speak requires non-empty --text\n";
    return 2;
  }
  if (o.voice.empty()) {
    std::cerr << "Error: IndexTTS-2.5 speak requires --voice <reference WAV>\n";
    return 2;
  }
  ext::SynthesisRequest request;
  request.text = o.text;
  request.language = "de";
  request.speaker = "preset";  // a local file: no 10 s inline limit
  request.sampling.seed = static_cast<std::uint32_t>(o.seed);
  request.max_new_tokens = o.max_frames;
  {
    std::ifstream in(o.voice, std::ios::binary);
    const std::string bytes{std::istreambuf_iterator<char>(in),
                            std::istreambuf_iterator<char>()};
    if (bytes.empty()) {
      std::cerr << "Error: cannot read --voice " << o.voice << "\n";
      return 2;
    }
    ming_tts::WavAudio audio;
    std::string wav_error;
    if (!ming_tts::DecodeWav(
            std::span(reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()),
            &audio, &wav_error)) {
      std::cerr << "Error: --voice " << o.voice << ": " << wav_error << "\n";
      return 2;
    }
    request.reference_audio = {.sample_rate = audio.sample_rate,
                               .channels = audio.channels,
                               .samples = std::move(audio.samples)};
  }
  std::string error;
  const auto load_begin = std::chrono::steady_clock::now();
  auto runner = CreateRunner(o.model, o.context, &error);
  if (runner == nullptr) {
    std::cerr << "Error loading IndexTTS-2.5: " << error << "\n";
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
    const std::string wav = ming_tts::EncodeWavPcm16(result.samples, result.sample_rate);
    std::ofstream out(o.output_path, std::ios::binary);
    out.write(wav.data(), static_cast<std::streamsize>(wav.size()));
    if (!out) {
      std::cerr << "Error: cannot write " << o.output_path << "\n";
      return 1;
    }
  }
  const double seconds = result.sample_rate == 0
                             ? 0.0
                             : static_cast<double>(result.samples.size()) / result.sample_rate;
  if (o.format == "json") {
    json::Value report = json::Value::object();
    report["schema"] = "gufo.indextts.speech.v1";
    report["text"] = o.text;
    report["steps"] = static_cast<std::int64_t>(result.codes.size());
    report["audio_seconds"] = seconds;
    json::Value timings = json::Value::object();
    timings["load_ms"] = load_ms;
    timings["total_ms"] = total_ms;
    report["timings"] = std::move(timings);
    std::cout << report.dump() << "\n";
  } else {
    std::cerr << "IndexTTS-2.5: " << seconds << " s audio in " << total_ms << " ms"
              << (o.output_path.empty() ? "" : ", wrote ") << o.output_path << "\n";
  }
  return 0;
}

}  // namespace

std::unique_ptr<ext::TtsRunner> CreateIndexTtsRunner(const std::string& model_root,
                                                     std::size_t maximum_context_tokens,
                                                     std::string* error) {
  return CreateRunner(model_root, maximum_context_tokens, error);
}

}  // namespace gufo::models::indextts

void indextts_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(gufo::ext::generated::extension_indextts::kProvider_indextts);
  using namespace gufo::models::indextts;
  registry.AddTtsProvider({
      .id = "indextts",
      .probe = &ProbeIndexTts,
      .resolve = &ResolveIndexTtsArtifacts,
      .create = &CreateIndexTtsRunner,
      .speak_driver = &RunSpeak,
  });
}
