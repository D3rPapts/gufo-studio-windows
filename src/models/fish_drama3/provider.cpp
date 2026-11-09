// Fish Audio Drama-3 speech_synthesis provider (HIP). Request mapping as in the
// gufo-studio-windows 1.4.9 TTS service: optional reference voice + transcript,
// sampling, frame budget, streaming callback; 44.1 kHz mono out, 10 code groups.
// Written with AI assistance (Claude Code).
#include <gufo/ext/generated/fish_drama3_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/tts.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "config.hpp"
#include "probe.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "hip/synthesis_runtime.hpp"
#endif

namespace gufo::models::fish_drama3 {

#if defined(ENGINE_ENABLE_HIP)
namespace {

constexpr std::uint32_t kSampleRate = 44100;

// Interleaved multi-channel reference audio to mono (the runtime expects mono).
std::vector<float> Mono(const ext::TtsAudioBuffer& audio) {
  if (audio.channels <= 1) {
    return audio.samples;
  }
  const std::size_t channels = audio.channels;
  std::vector<float> mono(audio.samples.size() / channels);
  for (std::size_t i = 0; i < mono.size(); ++i) {
    float sum = 0.0F;
    for (std::size_t c = 0; c < channels; ++c) {
      sum += audio.samples[i * channels + c];
    }
    mono[i] = sum / static_cast<float>(channels);
  }
  return mono;
}

class FishDrama3TtsRunner final : public ext::TtsRunner {
public:
  explicit FishDrama3TtsRunner(std::unique_ptr<hip::SynthesisHipRuntime> runtime)
      : runtime_(std::move(runtime)) {}

  bool Synthesize(const ext::SynthesisRequest& request,
                  const ext::CancellationCheck& is_cancelled,
                  ext::SynthesisResult* result, std::string* error) override {
    SynthesisRequest d3_req;
    d3_req.text = request.text;
    d3_req.reference_audio_mono = Mono(request.reference_audio);
    d3_req.reference_sample_rate = request.reference_audio.sample_rate > 0
                                       ? request.reference_audio.sample_rate
                                       : kSampleRate;
    d3_req.reference_text = request.reference_text;
    d3_req.sampling.temperature = request.sampling.temperature;
    d3_req.sampling.top_p = request.sampling.top_p;
    d3_req.sampling.top_k = static_cast<int>(request.sampling.top_k);
    d3_req.sampling.seed = request.sampling.seed;
    d3_req.sampling.max_new_frames = request.max_new_tokens;
    d3_req.on_audio = request.on_audio;

    SynthesisResult d3_res;
    if (!runtime_->Generate(d3_req, is_cancelled, &d3_res, error)) {
      return false;
    }
    if (result != nullptr) {
      result->sample_rate = d3_res.sample_rate;
      result->code_groups = d3_res.code_groups;
      result->samples = std::move(d3_res.samples);
      result->sample_count = d3_res.sample_count;
      // Runtime codes are [codebooks, frames]; the service wants frame-major.
      const auto frames = static_cast<std::size_t>(
          std::max<std::int64_t>(0, d3_res.codes.frames));
      const auto groups = static_cast<std::size_t>(
          std::max<std::int64_t>(0, d3_res.codes.codebooks));
      result->codes.resize(frames * groups);
      for (std::size_t f = 0; f < frames; ++f) {
        for (std::size_t g = 0; g < groups; ++g) {
          result->codes[f * groups + g] = d3_res.codes.codes[g * frames + f];
        }
      }
    }
    return true;
  }

  std::vector<std::string> Voices() const override {
    return {"default", "voice-clone"};
  }

  ext::TtsVariant Variant() const override { return ext::TtsVariant::kBase; }

  // As in 1.4.9: synthesis without a reference works ("default"), a reference
  // voice or a --voice preset is optional, instructions are accepted.
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

  std::uint32_t SampleRate() const override { return kSampleRate; }

  std::string ModelId() const override { return "fish-drama-3"; }

private:
  std::unique_ptr<hip::SynthesisHipRuntime> runtime_;
};

}  // namespace

std::unique_ptr<ext::TtsRunner> CreateFishDrama3Runner(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error) {
  auto runtime = hip::SynthesisHipRuntime::Create(
      model_root, maximum_context_tokens, error);
  if (!runtime) {
    return nullptr;
  }
  return std::make_unique<FishDrama3TtsRunner>(std::move(runtime));
}
#endif

}  // namespace gufo::models::fish_drama3

void fish_drama3_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_fish_drama3::kProvider_fish_drama_3);
#if defined(ENGINE_ENABLE_HIP)
  using namespace gufo::models::fish_drama3;
  registry.AddTtsProvider({
      .id = "fish-drama-3",
      .probe = &ProbeFishDrama3,
      .resolve = &ResolveFishDrama3Artifacts,
      .create = &CreateFishDrama3Runner,
  });
#endif
}
