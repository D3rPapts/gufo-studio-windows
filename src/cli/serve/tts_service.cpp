#include "src/cli/serve/tts_service.hpp"

#include <algorithm>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "src/core/cancellable_gate.hpp"
#include "src/models/fish_drama3/config.hpp"
#include "src/models/fish_drama3/hip/synthesis_runtime.hpp"
#include "src/models/ming_tts/config.hpp"
#include "src/models/ming_tts/synthesis.hpp"
#include "src/models/qwen3_tts/config.hpp"
#include "src/models/qwen3_tts/hip/synthesis_runtime.hpp"

namespace gufo::server {
namespace {

std::string ModelId(models::qwen3_tts::ModelVariant variant) {
  switch (variant) {
    case models::qwen3_tts::ModelVariant::kBase:
      return "qwen3-tts-12hz-1.7b-base";
    case models::qwen3_tts::ModelVariant::kVoiceDesign:
      return "qwen3-tts-12hz-1.7b-voice-design";
    case models::qwen3_tts::ModelVariant::kCustomVoice:
      return "qwen3-tts-12hz-1.7b-customvoice";
    case models::qwen3_tts::ModelVariant::kUnsupported:
      break;
  }
  return {};
}

}  // namespace

struct TtsService::Impl {
  explicit Impl(TtsServiceOptions supplied) : options(std::move(supplied)) {
    if (!options.model_root.empty() &&
        models::ming_tts::LooksLikeMingTts(options.model_root.string())) {
      InitMing();
      return;
    }
    if (!options.model_root.empty() &&
        models::fish_drama3::LooksLikeFishDrama3(options.model_root.string())) {
      is_drama3 = true;
      sample_rate = 44100;
      options.variant = models::qwen3_tts::ModelVariant::kBase;
      if (options.model_id.empty()) {
        options.model_id = "fish-drama-3";
      }
      if (options.voices.empty()) {
        options.voices.push_back("default");
        options.voices.push_back("voice-clone");
      }
      for (const auto& [name, unused] : options.voice_presets) {
        (void)unused;
        if (std::ranges::find(options.voices, name) == options.voices.end()) {
          options.voices.push_back(name);
        }
      }
      if (!options.runner) {
        drama3_runtime = models::fish_drama3::hip::SynthesisHipRuntime::Create(
            options.model_root.string(), options.native_context_tokens,
            &initialization_error);
        if (drama3_runtime == nullptr) {
          return;
        }
        options.runner =
            [this](const models::qwen3_tts::SynthesisRequest& request,
                   const models::qwen3_tts::CancellationCheck& is_cancelled,
                   models::qwen3_tts::SynthesisResult* result,
                   std::string* error) {
              models::fish_drama3::SynthesisRequest d3_req;
              d3_req.text = request.text;
              d3_req.reference_audio_mono = request.reference_audio.samples;
              d3_req.reference_sample_rate =
                  request.reference_audio.sample_rate > 0
                      ? request.reference_audio.sample_rate
                      : 44100U;
              d3_req.reference_text = request.reference_text;
              d3_req.sampling.temperature = request.sampling.temperature;
              d3_req.sampling.top_p = request.sampling.top_p;
              d3_req.sampling.top_k = static_cast<int>(request.sampling.top_k);
              d3_req.sampling.seed =
                  static_cast<std::uint32_t>(request.sampling.seed);
              d3_req.sampling.max_new_frames = request.max_new_tokens;
              d3_req.on_audio = request.on_audio;

              models::fish_drama3::SynthesisResult d3_res;
              if (!drama3_runtime->Generate(d3_req, is_cancelled, &d3_res,
                                            error)) {
                return false;
              }
              if (result != nullptr) {
                result->sample_rate = d3_res.sample_rate;
                result->code_groups = d3_res.code_groups;
                result->samples = std::move(d3_res.samples);
                result->sample_count = d3_res.sample_count;
                const auto frames = static_cast<std::size_t>(
                    std::max<std::int64_t>(0, d3_res.codes.frames));
                const auto groups = static_cast<std::size_t>(
                    std::max<std::int64_t>(0, d3_res.codes.codebooks));
                result->codes.resize(frames * groups);
                for (std::size_t f = 0; f < frames; ++f) {
                  for (std::size_t g = 0; g < groups; ++g) {
                    result->codes[f * groups + g] =
                        d3_res.codes.codes[g * frames + f];
                  }
                }
              }
              return true;
            };
      }
      std::ranges::sort(options.voices);
      ready = true;
      return;
    }

    std::optional<models::qwen3_tts::ModelConfig> config;
    if (options.validate_model) {
      config = models::qwen3_tts::LoadModelConfigFromPath(
          options.model_root.string());
      if (!config.has_value() ||
          !models::qwen3_tts::IsSupportedModelConfig(*config)) {
        initialization_error =
            "TTS service requires a supported Qwen3-TTS 1.7B or Fish Drama-3 "
            "model";
        return;
      }
      options.variant = config->variant;
      if (options.voices.empty() &&
          options.variant == models::qwen3_tts::ModelVariant::kCustomVoice) {
        for (const auto& [name, unused] : config->talker.spk_id) {
          (void)unused;
          options.voices.push_back(name);
        }
      }
    }
    if (options.model_id.empty()) {
      options.model_id = ModelId(options.variant);
    }
    if (options.voices.empty()) {
      if (options.variant == models::qwen3_tts::ModelVariant::kVoiceDesign) {
        options.voices.push_back("voice-design");
      } else if (options.variant == models::qwen3_tts::ModelVariant::kBase) {
        options.voices.push_back("voice-clone");
      }
    }
    if (!options.voice_presets.empty() &&
        options.variant != models::qwen3_tts::ModelVariant::kBase) {
      initialization_error = "voice presets require a Base checkpoint";
      return;
    }
    for (const auto& [name, unused] : options.voice_presets) {
      (void)unused;
      if (std::ranges::find(options.voices, name) == options.voices.end()) {
        options.voices.push_back(name);
      }
    }
    if (!options.runner) {
      native_runtime = models::qwen3_tts::hip::SynthesisHipRuntime::Create(
          options.model_root.string(), options.native_context_tokens,
          options.variant, &initialization_error);
      if (native_runtime == nullptr) {
        return;
      }
      options.runner =
          [this](const models::qwen3_tts::SynthesisRequest& request,
                 const models::qwen3_tts::CancellationCheck& is_cancelled,
                 models::qwen3_tts::SynthesisResult* result,
                 std::string* error) {
            return native_runtime->Generate(request, is_cancelled, result,
                                            error);
          };
    }
    std::ranges::sort(options.voices);
    ready = true;
  }

  // Ming-3.0-tts: speaker-embedding cloning (CAM++), EMOTION labels and
  // free-text instructions; CPU reference runtime.
  void InitMing() {
    is_ming = true;
    sample_rate = 44100;
    options.variant = models::qwen3_tts::ModelVariant::kBase;
    if (options.model_id.empty()) {
      options.model_id = "ming-3.0-tts-0.5b";
    }
    if (options.voices.empty()) {
      options.voices.push_back("default");
      options.voices.push_back("voice-clone");
    }
    for (const auto& [name, unused] : options.voice_presets) {
      (void)unused;
      if (std::ranges::find(options.voices, name) == options.voices.end()) {
        options.voices.push_back(name);
      }
    }
    if (!options.runner) {
      ming_engine = models::ming_tts::Engine::Create(options.model_root.string(),
                                                     &initialization_error);
      if (ming_engine == nullptr) {
        return;
      }
      options.runner =
          [this](const models::qwen3_tts::SynthesisRequest& request,
                 const models::qwen3_tts::CancellationCheck& is_cancelled,
                 models::qwen3_tts::SynthesisResult* result,
                 std::string* error) {
            models::ming_tts::SynthesisRequest ming;
            ming.text = request.text;
            ming.language = request.language;
            ming.emotion = request.emotion;
            ming.instruction = request.instruct;
            const auto& audio = request.reference_audio;
            if (!audio.samples.empty() && audio.channels > 0) {
              const std::size_t frames = audio.samples.size() / audio.channels;
              ming.reference_audio.resize(frames);
              for (std::size_t f = 0; f < frames; ++f) {
                double sum = 0.0;
                for (std::uint32_t c = 0; c < audio.channels; ++c) {
                  sum += audio.samples[f * audio.channels + c];
                }
                ming.reference_audio[f] =
                    static_cast<float>(sum / static_cast<double>(audio.channels));
              }
              ming.reference_sample_rate = audio.sample_rate;
            }
            ming.seed = request.sampling.seed;
            ming.max_decode_steps = static_cast<std::int32_t>(
                std::min<std::size_t>(request.max_new_tokens, 1000));
            ming.on_audio = request.on_audio;
            models::ming_tts::SynthesisResult ming_result;
            if (!ming_engine->Generate(ming, is_cancelled, &ming_result, error)) {
              return false;
            }
            if (result != nullptr) {
              result->sample_rate = ming_result.sample_rate;
              result->code_groups = 1;
              result->codes.assign(
                  static_cast<std::size_t>(std::accumulate(
                      ming_result.fragment_steps.begin(),
                      ming_result.fragment_steps.end(), std::int64_t{0})),
                  0);
              result->sample_count = ming_result.samples.size();
              result->samples = std::move(ming_result.samples);
            }
            return true;
          };
    }
    std::ranges::sort(options.voices);
    ready = true;
  }

  TtsServiceOptions options;
  std::unique_ptr<models::ming_tts::Engine> ming_engine;
  std::unique_ptr<models::qwen3_tts::hip::SynthesisHipRuntime> native_runtime;
  std::unique_ptr<models::fish_drama3::hip::SynthesisHipRuntime> drama3_runtime;
  std::string initialization_error;
  core::CancellableGate generation_gate;
  bool is_drama3{false};
  bool is_ming{false};
  std::uint32_t sample_rate{24000};
  bool ready{false};
};

TtsService::TtsService(TtsServiceOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

TtsService::~TtsService() = default;

bool TtsService::ready() const noexcept {
  return impl_->ready;
}

std::string TtsService::initialization_error() const {
  return impl_->initialization_error;
}

std::string TtsService::model_id() const {
  return impl_->options.model_id;
}

std::string TtsService::backend_name() const {
  return impl_->is_ming ? "native-cpu" : "native-hip";
}

std::vector<std::string> TtsService::voices() const {
  return impl_->options.voices;
}

const TtsVoicePreset* TtsService::voice_preset(const std::string& name) const {
  const auto found = impl_->options.voice_presets.find(name);
  return found == impl_->options.voice_presets.end() ? nullptr : &found->second;
}

models::qwen3_tts::ModelVariant TtsService::variant() const noexcept {
  return impl_->options.variant;
}

bool TtsService::is_fish_drama3() const noexcept {
  return impl_->is_drama3;
}

bool TtsService::is_ming_tts() const noexcept {
  return impl_->is_ming;
}

std::uint32_t TtsService::sample_rate() const noexcept {
  return impl_->sample_rate;
}

bool TtsService::Synthesize(
    const models::qwen3_tts::SynthesisRequest& request,
    const models::qwen3_tts::CancellationCheck& is_cancelled,
    models::qwen3_tts::SynthesisResult* result, std::string* error) {
  if (!impl_->ready) {
    if (error != nullptr) {
      *error = impl_->initialization_error;
    }
    return false;
  }
  auto lease = impl_->generation_gate.Acquire(is_cancelled, error);
  if (!lease) {
    if (result)
      *result = {};
    return false;
  }
  return impl_->options.runner(request, is_cancelled, result, error);
}

}  // namespace gufo::server
