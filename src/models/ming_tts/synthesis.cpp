#include "src/models/ming_tts/synthesis.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>

#include "src/models/ming_tts/cpu/model.hpp"
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/ming_tts/hip/model.hpp"
#endif
#include "src/models/ming_tts/prompt.hpp"
#include "src/models/ming_tts/speaker.hpp"
#include "src/models/ming_tts/tokenizer.hpp"

namespace gufo::models::ming_tts {
namespace {

// The reference runtime decodes with a 512-position static KV cache.
constexpr std::int64_t kMaxCacheLength = 512;

void SetError(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

std::uint64_t HashSamples(std::span<const float> samples, std::uint32_t rate) {
  std::uint64_t hash = 1469598103934665603ULL ^ rate;
  for (const float value : samples) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, 4);
    hash = (hash ^ bits) * 1099511628211ULL;
  }
  return hash ^ samples.size();
}

}  // namespace

struct Engine::Impl {
  std::unique_ptr<cpu::Model> model;
  // GPU runtime when built with HIP and a device is present; the CPU model
  // keeps the text/speaker/encoder work and is the fallback backend.
  std::unique_ptr<Backend> gpu;
  [[nodiscard]] Backend& run() { return gpu ? *gpu : static_cast<Backend&>(*model); }
  std::unique_ptr<CamPlusPlus> speaker;
  Tokenizer tokenizer;
  std::mutex speaker_mutex;
  std::map<std::uint64_t, std::vector<float>> speaker_cache;
};

std::unique_ptr<Engine> Engine::Create(const std::string& model_dir,
                                       std::string* error) {
  auto engine = std::unique_ptr<Engine>(new Engine());
  engine->impl_ = std::make_unique<Impl>();
  Impl& impl = *engine->impl_;
  impl.model = cpu::Model::Load(model_dir, error);
  if (impl.model == nullptr) {
    return nullptr;
  }
  impl.speaker = CamPlusPlus::Load(
      std::filesystem::path(model_dir) / "campplus.onnx", error);
  if (impl.speaker == nullptr) {
    return nullptr;
  }
  if (!Tokenizer::Load(model_dir, &impl.tokenizer, error)) {
    return nullptr;
  }
  // GUFO_MING_BACKEND=cpu forces the CPU runtime; =hip makes a missing GPU
  // runtime an error instead of a fallback.
  const char* backend_env = std::getenv("GUFO_MING_BACKEND");
  const std::string backend = backend_env != nullptr ? backend_env : "";
#if defined(ENGINE_ENABLE_HIP)
  if (backend != "cpu") {
    std::string gpu_error;
    impl.gpu = hip::Model::Create(*impl.model, &gpu_error);
    if (impl.gpu == nullptr && backend == "hip") {
      SetError(error, gpu_error);
      return nullptr;
    }
  }
#else
  if (backend == "hip") {
    SetError(error, "this build has no HIP runtime");
    return nullptr;
  }
#endif
  // prompt_contract.validate_tokenizer: the trained special-token ids.
  for (const auto& [token, id] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"<|im_start|>", kImStartId},
           {"<|im_end|>", kImEndId},
           {"<|vision_start|>", kVisionStartId},
           {"<|vision_pad|>", kVisionPadId},
           {"<audioPatch>", kAudioPatchId},
           {"<audio>", kAudioStartId}}) {
    std::vector<std::uint32_t> ids;
    if (!impl.tokenizer.Encode(token, &ids, error) || ids.size() != 1 ||
        ids[0] != id) {
      SetError(error, std::string("Ming-3.0-tts tokenizer does not match the "
                                  "trained layout for ") + token);
      return nullptr;
    }
  }
  // Whatever loading touched leaves the working set (#829); the GPU uploads
  // already dropped their pages tensor by tensor.
  impl.model->ReleaseHostPages();
  return engine;
}

bool ResolveTaskInstruction(const SynthesisRequest& request,
                            std::optional<std::string>* instruction,
                            std::string* error) {
  instruction->reset();
  const bool basic =
      !request.speed.empty() || !request.pitch.empty() || !request.volume.empty();
  const bool dialect = !request.dialect.empty();
  if (!basic && !dialect) {
    return true;
  }
  if ((basic && dialect) || !request.emotion.empty() ||
      !request.instruction.empty()) {
    SetError(error,
             "Ming controls are one task per request: use either speed/pitch/"
             "volume, dialect, emotion or a free-text instruction");
    return false;
  }
  StructuredControls controls;
  if (dialect) {
    controls.dialect = DialectLabel(request.dialect);
    if (!controls.dialect.has_value()) {
      SetError(error, "unknown Ming dialect '" + request.dialect +
                          "' (cantonese, chuanyu dialect)");
      return false;
    }
    *instruction = BuildStructuredInstruction(controls, InstructionSchema::kEnglish);
    return true;
  }
  const auto resolve = [&](const std::string& value, auto label_of,
                           std::optional<std::string>* slot, const char* name,
                           const char* allowed) {
    if (value.empty()) {
      return true;
    }
    *slot = label_of(value);
    if (!slot->has_value()) {
      SetError(error, std::string("unknown Ming ") + name + " '" + value + "' (" +
                          allowed + ")");
      return false;
    }
    return true;
  };
  if (!resolve(request.speed, SpeedLabel, &controls.speed, "speed",
               "slow, medium, fast") ||
      !resolve(request.pitch, PitchLabel, &controls.pitch, "pitch",
               "low, medium, high") ||
      !resolve(request.volume, VolumeLabel, &controls.volume, "volume",
               "low, medium, high")) {
    return false;
  }
  *instruction = BuildStructuredInstruction(controls);
  return true;
}

Engine::~Engine() = default;

const char* Engine::backend_name() const noexcept {
  return impl_->gpu ? impl_->gpu->name() : "cpu";
}

std::uint32_t Engine::sample_rate() const noexcept {
  return static_cast<std::uint32_t>(impl_->model->config().sample_rate);
}

bool Engine::Generate(const SynthesisRequest& request,
                      const std::function<bool()>& is_cancelled,
                      SynthesisResult* result, std::string* error) {
  Impl& impl = *impl_;
  cpu::Model& model = *impl.model;
  Backend& run = impl.run();
  const ModelConfig& config = model.config();
  const std::int64_t hidden = config.llm.hidden_size;
  const std::int64_t latent_dim = config.latent_dim;
  SynthesisResult local;
  SynthesisResult& out = result != nullptr ? *result : local;
  out = {};
  out.sample_rate = static_cast<std::uint32_t>(config.sample_rate);

  // --- task -> instruction
  std::optional<std::string> instruction;
  std::optional<std::string> task_instruction;
  if (!ResolveTaskInstruction(request, &task_instruction, error)) {
    return false;
  }
  const bool dialect_task = !request.dialect.empty();
  if (!request.instruction.empty()) {
    instruction = request.instruction;  // FREEFORM
  } else if (task_instruction.has_value()) {
    instruction = std::move(task_instruction);  // BASIC or DIALECT
  } else if (!request.emotion.empty()) {
    const auto label = EmotionLabel(request.emotion);
    if (!label.has_value()) {
      SetError(error, "unknown Ming emotion '" + request.emotion +
                          "' (happy, sad, angry, fear, disgust, surprise, neutral)");
      return false;
    }
    StructuredControls controls;
    controls.emotion = *label;
    instruction = BuildStructuredInstruction(controls);
  }

  // --- speaker embedding
  std::vector<float> speaker;
  try {
    if (!request.speaker_embedding.empty()) {
      if (request.speaker_embedding.size() != 192) {
        SetError(error, "speaker_embedding must have 192 values");
        return false;
      }
      speaker = request.speaker_embedding;
    } else if (!request.reference_audio.empty()) {
      if (request.reference_sample_rate == 0) {
        SetError(error, "reference audio needs a sample rate");
        return false;
      }
      const std::uint64_t key =
          HashSamples(request.reference_audio, request.reference_sample_rate);
      std::lock_guard<std::mutex> lock(impl.speaker_mutex);
      const auto found = impl.speaker_cache.find(key);
      if (found != impl.speaker_cache.end()) {
        speaker = found->second;
      } else {
        speaker = impl.speaker->EmbedWaveform(request.reference_audio,
                                               request.reference_sample_rate);
        impl.speaker_cache.emplace(key, speaker);
      }
    }
  } catch (const std::exception& exception) {
    SetError(error, std::string("speaker embedding failed: ") + exception.what());
    return false;
  }
  out.speaker_embedding = speaker;

  std::mt19937_64 rng(request.seed);
  std::normal_distribution<float> normal(0.0F, 1.0F);

  // --- prompt latents (VAE encoder): only together with a transcript.
  std::vector<float> prompt_latents;
  std::optional<std::string> prompt_text;
  try {
    if (!request.reference_text.empty() && !request.reference_audio.empty()) {
      prompt_latents = EncodePrompt(request.reference_audio,
                                    request.reference_sample_rate,
                                    request.prompt_noise, &rng);
      prompt_text = request.reference_text;
    }
  } catch (const std::exception& exception) {
    SetError(error, std::string("prompt encoding failed: ") + exception.what());
    return false;
  }

  // --- text
  std::vector<std::string> fragments;
  if (request.raw_text) {
    fragments.push_back(request.text);
  } else {
    fragments = SplitTextFragments(request.text, request.max_fragment_length);
  }
  if (fragments.empty()) {
    SetError(error, "text produced no synthesizable fragments");
    return false;
  }
  out.fragments = fragments;
  // DIALECT does not use a language ID (tts_tasks.build_task_generation_kwargs).
  const std::string prompt =
      BuildTtsPrompt(dialect_task ? std::string_view{} : std::string_view(request.language));
  // MingTTS: split FREEFORM (STYLE) requests reuse the first generated
  // fragment as voice prompt (latents + speaker embedding + its text).
  const bool reuse_first_fragment =
      !request.instruction.empty() && fragments.size() > 1;
  cpu::FlowOptions flow;
  flow.cfg = request.cfg;
  flow.sigma = request.sigma;
  flow.temperature = request.temperature;

  for (std::size_t index = 0; index < fragments.size(); ++index) {
    const std::size_t num_speakers = speaker.empty() ? 0 : 1;
    std::vector<float> projected(static_cast<std::size_t>(hidden));
    if (num_speakers != 0) {
      model.ProjectSpeaker(speaker, projected.data());
    }
    const std::size_t latent_steps =
        prompt_text.has_value() ? prompt_latents.size() / static_cast<std::size_t>(latent_dim) : 0;
    InputLayout layout;
    if (!BuildInputIds(impl.tokenizer, prompt, fragments[index], num_speakers,
                       instruction, &layout, error, prompt_text, latent_steps)) {
      return false;
    }
    const auto rows = static_cast<std::int64_t>(layout.ids.size());
    if (rows >= kMaxCacheLength - 4) {
      SetError(error, "prompt too long for the 512-position decode window");
      return false;
    }
    std::vector<std::int64_t> ids(layout.ids.begin(), layout.ids.end());
    std::vector<float> embeddings(static_cast<std::size_t>(rows * hidden));
    model.Embed(ids, embeddings.data());
    for (const std::size_t position : layout.speaker_positions) {
      std::copy(projected.begin(), projected.end(),
                embeddings.begin() + static_cast<std::ptrdiff_t>(position) * hidden);
    }
    for (std::size_t i = 0; i < latent_steps; ++i) {
      run.AggregateLatent(prompt_latents.data() + i * static_cast<std::size_t>(latent_dim),
                      embeddings.data() + (layout.latent_position + static_cast<std::int64_t>(i)) * hidden);
    }
    run.ResetBackbone();
    std::vector<float> normed(static_cast<std::size_t>(rows * hidden));
    run.RunBackbone(embeddings.data(), rows, normed.data());
    std::vector<float> z(normed.end() - hidden, normed.end());

    const std::int64_t budget =
        std::min<std::int64_t>(request.max_decode_steps, kMaxCacheLength - rows);
    std::vector<float> history(static_cast<std::size_t>(latent_dim), 0.0F);
    if (latent_steps > 0) {  // the last prompt latent seeds the history
      history.assign(prompt_latents.end() - latent_dim, prompt_latents.end());
    }
    std::vector<float> noise(static_cast<std::size_t>(latent_dim));
    std::vector<float> sde(static_cast<std::size_t>(flow.steps * latent_dim));
    std::vector<float> latent(static_cast<std::size_t>(latent_dim));
    std::vector<float> latents;
    std::vector<float> next(static_cast<std::size_t>(hidden));
    std::int64_t step = 0;
    for (; step < budget; ++step) {
      if (is_cancelled && is_cancelled()) {
        SetError(error, "cancelled");
        return false;
      }
      if (request.noise_source) {
        request.noise_source(index, step, noise.data());
      } else {
        for (float& value : noise) {
          value = normal(rng);
        }
      }
      if (flow.temperature > 0.0F) {
        for (float& value : sde) {
          value = normal(rng);
        }
      }
      run.SampleFlow(z.data(), history.data(), noise.data(), sde.data(),
                         flow, latent.data());
      latents.insert(latents.end(), latent.begin(), latent.end());
      if (run.StopAt(z.data()) > 0.5F && step > 3) {
        ++step;
        break;
      }
      run.AggregateLatent(latent.data(), next.data());
      run.RunBackbone(next.data(), 1, z.data());
      history = latent;
    }
    const std::int64_t frames = static_cast<std::int64_t>(latents.size()) / latent_dim;
    out.fragment_steps.push_back(frames);
    std::vector<float> audio = run.DecodeAudio(latents.data(), frames);
    if (request.trim_silence) {
      audio = TrimTrailingSilence(audio, config.sample_rate);
    }
    if (request.on_audio && !request.on_audio(audio)) {
      SetError(error, "cancelled");
      return false;
    }
    out.samples.insert(out.samples.end(), audio.begin(), audio.end());
    out.latents = std::move(latents);
    if (reuse_first_fragment && index == 0) {
      try {
        speaker = impl.speaker->EmbedWaveform(audio, config.sample_rate);
        prompt_latents = EncodePrompt(audio, static_cast<std::uint32_t>(config.sample_rate),
                                      {}, &rng);
        prompt_text = fragments[0];
      } catch (const std::exception& exception) {
        SetError(error, std::string("fragment prompt failed: ") + exception.what());
        return false;
      }
    }
  }
  return true;
}

std::vector<float> Engine::EncodePrompt(std::span<const float> audio,
                                        std::uint32_t sample_rate,
                                        std::span<const float> noise,
                                        std::mt19937_64* rng) {
  const cpu::Model& model = *impl_->model;
  const ModelConfig& config = model.config();
  if (sample_rate == 0) {
    throw std::invalid_argument("reference audio needs a sample rate");
  }
  std::vector<float> wave =
      ResampleLikeTorchaudio(audio, sample_rate, config.sample_rate);
  // Front padding to hop * patch: the prompt tail seeds the first step.
  const auto align = static_cast<std::size_t>(config.vae_hop_size) *
                     static_cast<std::size_t>(config.vae_patch_size);
  if (const std::size_t rem = wave.size() % align; rem != 0) {
    wave.insert(wave.begin(), align - rem, 0.0F);
  }
  std::int64_t patches = 0;
  const std::vector<float> params =
      model.EncodePosterior(wave.data(), static_cast<std::int64_t>(wave.size()), &patches);
  if (impl_->gpu != nullptr) {
    // The audio encoder runs on the CPU from the mapping; with the decoder on
    // the GPU nothing else reads those pages until the next prompt (#829).
    model.ReleaseHostPages();
  }
  std::vector<float> eps(static_cast<std::size_t>(patches * config.latent_dim));
  if (noise.size() == eps.size()) {
    std::copy(noise.begin(), noise.end(), eps.begin());
  } else {
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (float& value : eps) {
      value = normal(*rng);
    }
  }
  return model.SamplePosterior(params, eps.data());
}

std::vector<float> TrimTrailingSilence(std::span<const float> speech,
                                       std::int64_t sample_rate) {
  const auto frame = static_cast<std::int64_t>(static_cast<double>(sample_rate) * 0.1);
  const auto tail = static_cast<std::int64_t>(static_cast<double>(sample_rate) * 0.3);
  const auto length = static_cast<std::int64_t>(speech.size());
  if (length == 0) {
    return {};
  }
  if (frame <= 0 || length < frame) {
    return {speech.begin(), speech.begin() + std::min(length, tail)};
  }
  const std::int64_t frames = (length - frame) / frame + 1;
  std::int64_t index = frames - 1;
  for (; index >= 0; --index) {
    double sum = 0.0;
    for (std::int64_t i = 0; i < frame; ++i) {
      sum += std::fabs(speech[static_cast<std::size_t>(index * frame + i)]);
    }
    if (static_cast<float>(sum / static_cast<double>(frame)) > 1e-3F) {
      break;
    }
  }
  const std::int64_t usable = frames * frame;
  const std::int64_t keep =
      index < 0 ? std::min(usable, tail) : std::min(usable, index * frame + frame + tail);
  return {speech.begin(), speech.begin() + keep};
}

}  // namespace gufo::models::ming_tts
