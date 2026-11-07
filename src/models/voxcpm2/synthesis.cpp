#include "src/models/voxcpm2/synthesis.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>

#include "src/models/ming_tts/speaker.hpp"
#include "src/models/voxcpm2/config.hpp"
#include "src/models/voxcpm2/cpu/audio_vae.hpp"
#include "src/models/voxcpm2/cpu/model.hpp"
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/voxcpm2/hip/audio_vae.hpp"
#include "src/models/voxcpm2/hip/model.hpp"
#endif
#include "src/models/voxcpm2/vae_runner.hpp"
#include "src/models/voxcpm2/tokenizer.hpp"

namespace gufo::models::voxcpm2 {
namespace {

void SetError(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

std::uint64_t HashSamples(std::span<const float> samples, std::uint32_t rate, bool left) {
  std::uint64_t hash = 1469598103934665603ULL ^ rate ^ (left ? 0x9E3779B97F4A7C15ULL : 0);
  for (const float value : samples) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, 4);
    hash = (hash ^ bits) * 1099511628211ULL;
  }
  return hash ^ samples.size();
}

// The CPU AudioVAE behind the runner interface.
class CpuVaeRunner final : public VaeRunner {
public:
  explicit CpuVaeRunner(const cpu::AudioVae& vae) : vae_(vae) {}
  [[nodiscard]] const char* name() const noexcept override { return "cpu"; }
  [[nodiscard]] std::vector<float> Encode(const float* wave, std::int64_t samples,
                                          std::int64_t* frames) override {
    return vae_.Encode(wave, samples, frames);
  }
  void ResetDecoder() override { state_ = {}; }
  [[nodiscard]] std::vector<float> DecodeChunk(const float* z, std::int64_t frames) override {
    return vae_.DecodeChunk(&state_, z, frames);
  }

private:
  const cpu::AudioVae& vae_;
  cpu::AudioVae::DecoderState state_;
};

}  // namespace

std::string NormalizeWhitespace(const std::string& text) {
  std::string out;
  bool in_space = false;
  for (std::size_t i = 0; i < text.size();) {
    const auto c = static_cast<unsigned char>(text[i]);
    std::size_t width = 0;
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') {
      width = 1;
    } else if (c == 0xC2 && i + 1 < text.size() &&
               static_cast<unsigned char>(text[i + 1]) == 0xA0) {
      width = 2;  // U+00A0
    } else if (c == 0xE3 && i + 2 < text.size() &&
               static_cast<unsigned char>(text[i + 1]) == 0x80 &&
               static_cast<unsigned char>(text[i + 2]) == 0x80) {
      width = 3;  // U+3000
    }
    if (width != 0) {
      if (!in_space) {
        out += ' ';
      }
      in_space = true;
      i += width;
    } else {
      out += text[i++];
      in_space = false;
    }
  }
  return out;
}

struct Engine::Impl {
  std::unique_ptr<cpu::Model> model;
  std::unique_ptr<cpu::AudioVae> vae;
  std::unique_ptr<Backend> gpu;
  [[nodiscard]] Backend& run() { return gpu ? *gpu : static_cast<Backend&>(*model); }
  std::unique_ptr<VaeRunner> vae_cpu;
  std::unique_ptr<VaeRunner> vae_gpu;
  [[nodiscard]] VaeRunner& vae_run() { return vae_gpu ? *vae_gpu : *vae_cpu; }
  Tokenizer tokenizer;
  std::mutex cache_mutex;
  std::map<std::uint64_t, std::vector<float>> audio_cache;
};

std::unique_ptr<Engine> Engine::Create(const std::string& model_dir, std::string* error) {
  auto engine = std::unique_ptr<Engine>(new Engine());
  engine->impl_ = std::make_unique<Impl>();
  Impl& impl = *engine->impl_;
  impl.model = cpu::Model::Load(model_dir, error);
  if (impl.model == nullptr) {
    return nullptr;
  }
  impl.vae = cpu::AudioVae::Load(model_dir, impl.model->config().vae, error);
  if (impl.vae == nullptr) {
    return nullptr;
  }
  if (!Tokenizer::Load(model_dir, &impl.tokenizer, error)) {
    return nullptr;
  }
  impl.vae_cpu = std::make_unique<CpuVaeRunner>(*impl.vae);
  // GUFO_VOXCPM_BACKEND=cpu forces the CPU runtime; =hip makes a missing GPU
  // runtime an error instead of a fallback.
  const char* backend_env = std::getenv("GUFO_VOXCPM_BACKEND");
  const std::string backend = backend_env != nullptr ? backend_env : "";
#if defined(ENGINE_ENABLE_HIP)
  if (backend != "cpu") {
    std::string gpu_error;
    impl.gpu = hip::Model::Create(*impl.model, &gpu_error);
    if (impl.gpu == nullptr && backend == "hip") {
      SetError(error, gpu_error);
      return nullptr;
    }
    // GUFO_VOXCPM_VAE=cpu keeps the AudioVAE on the CPU next to a GPU AR loop.
    const char* vae_env = std::getenv("GUFO_VOXCPM_VAE");
    if (impl.gpu != nullptr && !(vae_env != nullptr && std::string(vae_env) == "cpu")) {
      std::string vae_error;
      impl.vae_gpu = hip::AudioVae::Create(*impl.vae, &vae_error);
    }
  }
#else
  if (backend == "hip") {
    SetError(error, "this build has no HIP runtime");
    return nullptr;
  }
#endif
  // Whatever loading touched (headers, the delta-time MLP) leaves the working
  // set; the GPU uploads already dropped their pages tensor by tensor.
  impl.model->ReleaseHostPages();
  return engine;
}

Engine::~Engine() = default;

std::uint32_t Engine::sample_rate() const noexcept {
  return static_cast<std::uint32_t>(impl_->model->config().vae.out_sample_rate);
}

const char* Engine::backend_name() const noexcept {
  return impl_->gpu ? impl_->gpu->name() : "cpu";
}

const char* Engine::vae_backend_name() const noexcept {
  return impl_->vae_gpu ? impl_->vae_gpu->name() : "cpu";
}

void Engine::SetSkinnyGemm(bool enabled) {
#if defined(ENGINE_ENABLE_HIP)
  if (auto* model = dynamic_cast<hip::Model*>(impl_->gpu.get()); model != nullptr) {
    model->SetSkinnyGemm(enabled);
  }
#else
  (void)enabled;
#endif
}

void Engine::SetWmma(bool enabled) {
#if defined(ENGINE_ENABLE_HIP)
  if (auto* model = dynamic_cast<hip::Model*>(impl_->gpu.get()); model != nullptr) {
    model->SetWmma(enabled);
  }
#else
  (void)enabled;
#endif
}

void Engine::SetSkinnyMaxRows(std::int64_t rows) {
#if defined(ENGINE_ENABLE_HIP)
  if (auto* model = dynamic_cast<hip::Model*>(impl_->gpu.get()); model != nullptr) {
    model->SetSkinnyMaxRows(rows);
  }
#else
  (void)rows;
#endif
}

void Engine::ClearAudioCache() {
  std::lock_guard<std::mutex> lock(impl_->cache_mutex);
  impl_->audio_cache.clear();
}

void Engine::SetVaeOnGpu(bool enabled) {
  if (!enabled) {
    impl_->vae_gpu.reset();
  }
#if defined(ENGINE_ENABLE_HIP)
  else if (impl_->vae_gpu == nullptr) {
    std::string error;
    impl_->vae_gpu = hip::AudioVae::Create(*impl_->vae, &error);
  }
#endif
}

std::vector<float> Engine::EncodeAudio(std::span<const float> audio, std::uint32_t sample_rate,
                                       bool pad_left) {
  const ModelConfig& config = impl_->model->config();
  if (sample_rate == 0 || audio.empty()) {
    throw std::invalid_argument("prompt audio needs samples and a sample rate");
  }
  const std::uint64_t key = HashSamples(audio, sample_rate, pad_left);
  {
    std::lock_guard<std::mutex> lock(impl_->cache_mutex);
    const auto found = impl_->audio_cache.find(key);
    if (found != impl_->audio_cache.end()) {
      return found->second;
    }
  }
  std::vector<float> wave =
      static_cast<std::int64_t>(sample_rate) == config.vae.sample_rate
          ? std::vector<float>(audio.begin(), audio.end())
          : ming_tts::ResampleLikeTorchaudio(audio, sample_rate, config.vae.sample_rate);
  const auto patch_len = static_cast<std::size_t>(config.patch_size) *
                         static_cast<std::size_t>(config.vae.chunk_size());
  if (const std::size_t rem = wave.size() % patch_len; rem != 0) {
    if (pad_left) {
      wave.insert(wave.begin(), patch_len - rem, 0.0F);
    } else {
      wave.insert(wave.end(), patch_len - rem, 0.0F);
    }
  }
  std::int64_t frames = 0;
  const std::vector<float> mu =
      impl_->vae_run().Encode(wave.data(), static_cast<std::int64_t>(wave.size()), &frames);
  // mu [latent, frames] -> view(latent, patches, patch).permute(1, 2, 0)
  const std::int64_t d = config.feat_dim;
  const std::int64_t p = config.patch_size;
  const std::int64_t patches = frames / p;
  std::vector<float> out(static_cast<std::size_t>(patches * p * d));
  for (std::int64_t n = 0; n < patches; ++n) {
    for (std::int64_t j = 0; j < p; ++j) {
      for (std::int64_t c = 0; c < d; ++c) {
        out[static_cast<std::size_t>((n * p + j) * d + c)] =
            mu[static_cast<std::size_t>(c * frames + n * p + j)];
      }
    }
  }
  std::lock_guard<std::mutex> lock(impl_->cache_mutex);
  if (impl_->audio_cache.size() > 16) {
    impl_->audio_cache.clear();
  }
  impl_->audio_cache.emplace(key, out);
  return out;
}

bool Engine::Generate(const SynthesisRequest& request, const std::function<bool()>& is_cancelled,
                      SynthesisResult* result, std::string* error) {
  Impl& impl = *impl_;
  const ModelConfig& config = impl.model->config();
  Backend& run = impl.run();
  SynthesisResult local;
  SynthesisResult& out = result != nullptr ? *result : local;
  out = {};
  out.sample_rate = sample_rate();
  const std::int64_t p = config.patch_size;
  const std::int64_t d = config.feat_dim;
  const std::int64_t h = config.base_lm.hidden_size;
  const std::int64_t patch_values = p * d;

  std::string target = request.text;
  if (!request.instruction.empty()) {
    target = "(" + request.instruction + ")" + target;
  }
  target = NormalizeWhitespace(target);
  if (target.find_first_not_of(' ') == std::string::npos) {
    SetError(error, "target text must be a non-empty string");
    return false;
  }
  const bool has_prompt = !request.prompt_audio.empty();
  if (has_prompt && request.prompt_text.empty()) {
    SetError(error, "prompt audio needs its transcript (prompt_text)");
    return false;
  }
  if (request.cfg <= 0.0F || request.timesteps < 1 || request.timesteps > 100) {
    SetError(error, "cfg must be > 0 and timesteps in [1, 100]");
    return false;
  }

  // --- prompt audio -> latent patches
  std::vector<float> ref_patches;
  std::vector<float> prompt_patches;
  try {
    if (!request.reference_audio.empty()) {
      ref_patches = EncodeAudio(request.reference_audio, request.reference_sample_rate, false);
    }
    if (has_prompt) {
      prompt_patches = EncodeAudio(request.prompt_audio, request.prompt_sample_rate, true);
    }
  } catch (const std::exception& exception) {
    SetError(error, std::string("prompt audio encoding failed: ") + exception.what());
    return false;
  }
  const auto ref_len = static_cast<std::int64_t>(ref_patches.size()) / patch_values;
  const auto prompt_len = static_cast<std::int64_t>(prompt_patches.size()) / patch_values;

  // --- layout (_generate_with_prompt_cache)
  const std::string text = has_prompt ? request.prompt_text + target : target;
  std::vector<std::int64_t> text_ids = impl.tokenizer.Encode(text);
  text_ids.push_back(kAudioStartToken);
  std::vector<std::int64_t> ids;
  std::vector<std::uint8_t> audio_mask;
  std::vector<float> feats;
  auto push_rows = [&](std::int64_t id, std::uint8_t audio, const float* patch) {
    ids.push_back(id);
    audio_mask.push_back(audio);
    if (patch != nullptr) {
      feats.insert(feats.end(), patch, patch + patch_values);
    } else {
      feats.insert(feats.end(), static_cast<std::size_t>(patch_values), 0.0F);
    }
  };
  if (ref_len > 0) {
    push_rows(kRefAudioStartToken, 0, nullptr);
    for (std::int64_t i = 0; i < ref_len; ++i) {
      push_rows(0, 1, ref_patches.data() + i * patch_values);
    }
    push_rows(kRefAudioEndToken, 0, nullptr);
  }
  for (const std::int64_t id : text_ids) {
    push_rows(id, 0, nullptr);
  }
  for (std::int64_t i = 0; i < prompt_len; ++i) {
    push_rows(0, 1, prompt_patches.data() + i * patch_values);
  }
  out.input_ids = ids;
  const auto rows = static_cast<std::int64_t>(ids.size());
  const auto target_len = static_cast<std::int64_t>(impl.tokenizer.Encode(target).size());
  const std::int64_t max_len = std::min<std::int64_t>(
      static_cast<std::int64_t>(static_cast<double>(target_len) *
                                    request.retry_badcase_ratio_threshold +
                                10.0),
      request.max_len);
  if (rows + max_len > config.max_length) {
    SetError(error, "prompt plus generation exceeds the 8192-position context");
    return false;
  }
  // Continuation: the last (up to 3) prompt patches warm up the decoder.
  const std::int64_t context_len = prompt_len > 0 ? std::min<std::int64_t>(3, prompt_len) : 0;

  FlowOptions flow;
  flow.cfg = request.cfg;
  flow.timesteps = request.timesteps;
  const bool streaming = static_cast<bool>(request.on_audio);
  const std::int32_t attempts_allowed =
      (request.retry_badcase && !streaming) ? std::max(1, request.retry_badcase_max_times) : 1;
  std::mt19937_64 rng(request.seed);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> lm_hidden(static_cast<std::size_t>(h));
  std::vector<float> residual_hidden(static_cast<std::size_t>(h));
  std::vector<float> noise(static_cast<std::size_t>(patch_values));
  std::vector<float> patch(static_cast<std::size_t>(patch_values));
  std::vector<float> prev(static_cast<std::size_t>(patch_values));
  std::vector<float> generated;
  VaeRunner& vae = impl.vae_run();
  auto decode_patches = [&](const float* patches, std::int64_t count) {
    std::vector<float> z(static_cast<std::size_t>(d * count * p));
    for (std::int64_t n = 0; n < count; ++n) {
      for (std::int64_t j = 0; j < p; ++j) {
        for (std::int64_t c = 0; c < d; ++c) {
          z[static_cast<std::size_t>(c * count * p + n * p + j)] =
              patches[(n * p + j) * d + c];
        }
      }
    }
    return vae.DecodeChunk(z.data(), count * p);
  };

  std::int32_t attempt = 0;
  for (; attempt < attempts_allowed; ++attempt) {
    run.Prefill(ids.data(), feats.data(), audio_mask.data(), rows, lm_hidden.data(),
                residual_hidden.data());
    std::copy(feats.end() - patch_values, feats.end(), prev.begin());
    generated.clear();
    vae.ResetDecoder();
    if (streaming && context_len > 0) {
      (void)decode_patches(prompt_patches.data() + (prompt_len - context_len) * patch_values,
                           context_len);
    }
    for (std::int64_t step = 0; step < max_len; ++step) {
      if (is_cancelled && is_cancelled()) {
        SetError(error, "cancelled");
        return false;
      }
      if (request.noise_source) {
        request.noise_source(attempt, step, noise.data());
      } else {
        for (float& value : noise) {
          value = normal(rng);
        }
      }
      run.SamplePatch(lm_hidden.data(), residual_hidden.data(), prev.data(), noise.data(), flow,
                      patch.data());
      generated.insert(generated.end(), patch.begin(), patch.end());
      prev = patch;
      if (streaming) {
        const std::vector<float> audio = decode_patches(patch.data(), 1);
        out.samples.insert(out.samples.end(), audio.begin(), audio.end());
        if (!request.on_audio(audio)) {
          SetError(error, "cancelled");
          return false;
        }
      }
      float logits[2];
      run.StopLogits(lm_hidden.data(), logits);
      if (step > request.min_len && logits[1] > logits[0]) {
        break;
      }
      if (step + 1 < max_len) {
        run.Advance(patch.data(), lm_hidden.data(), residual_hidden.data());
      }
    }
    const auto steps = static_cast<std::int64_t>(generated.size()) / patch_values;
    const bool badcase =
        static_cast<double>(steps) >=
        static_cast<double>(target_len) * request.retry_badcase_ratio_threshold;
    if (!badcase || attempt + 1 >= attempts_allowed) {
      break;
    }
  }
  out.attempts = std::min(attempt + 1, attempts_allowed);
  out.steps = static_cast<std::int64_t>(generated.size()) / patch_values;
  if (!streaming) {
    vae.ResetDecoder();
    if (context_len > 0) {
      (void)decode_patches(prompt_patches.data() + (prompt_len - context_len) * patch_values,
                           context_len);
    }
    if (out.steps > 0) {
      out.samples = decode_patches(generated.data(), out.steps);
    }
  }
  out.patches = std::move(generated);
  return true;
}

}  // namespace gufo::models::voxcpm2
