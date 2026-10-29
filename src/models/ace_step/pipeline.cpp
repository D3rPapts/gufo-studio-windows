#include "src/models/ace_step/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

#include "src/models/ace_step/model.hpp"
#include "src/models/ace_step/tokenizer.hpp"
#include "src/models/ace_step/vae.hpp"
#include "src/models/ace_step/weights.hpp"

namespace gufo::models::ace_step {
namespace {

constexpr int kMaxTextTokens = 256;
constexpr int kMaxLyricTokens = 2048;
constexpr std::uint32_t kEndOfText = 151643;
constexpr double kMaxSeconds = 600.0;

double Since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

std::string OrNa(const std::string& value) {
  return value.empty() ? "N/A" : value;
}

void AppendU16(std::string* out, std::uint16_t value) {
  out->push_back(static_cast<char>(value & 0xFFU));
  out->push_back(static_cast<char>(value >> 8U));
}

void AppendU32(std::string* out, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8)
    out->push_back(static_cast<char>((value >> shift) & 0xFFU));
}

}  // namespace

bool ModelPaths::FromRoot(const std::filesystem::path& root, ModelPaths* paths,
                          std::string* error) {
  paths->dit = root / "dit.safetensors";
  paths->text = root / "text_encoder.safetensors";
  paths->vae = root / "vae.safetensors";
  paths->tokenizer = root / "tokenizer.json";
  paths->silence = root / "silence_latent.safetensors";
  for (const auto* path : {&paths->dit, &paths->text, &paths->vae,
                           &paths->tokenizer, &paths->silence}) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(*path, ec))
      return Fail(error, "ACE-Step model file missing: " + path->string());
  }
  std::error_code ec;
  if (std::filesystem::is_regular_file(root / "lm.safetensors", ec))
    paths->lm = root / "lm.safetensors";
  return true;
}

std::string MusicTextPrompt(const MusicRequest& r) {
  const std::string metas =
      "- bpm: " + OrNa(r.bpm) + "\n- timesignature: " + OrNa(r.timesignature) +
      "\n- keyscale: " + OrNa(r.keyscale) +
      "\n- duration: " + std::to_string(static_cast<int>(r.duration)) +
      " seconds\n";
  return "# Instruction\nFill the audio semantic mask based on the given "
         "conditions:\n\n# Caption\n" +
         r.caption + "\n\n# Metas\n" + metas + "<|endoftext|>\n";
}

std::string MusicLyricPrompt(const MusicRequest& r) {
  const std::string lyrics = r.lyrics.empty() ? "[Instrumental]" : r.lyrics;
  return "# Languages\n" +
         (r.language.empty() ? std::string("en") : r.language) +
         "\n\n# Lyric\n" + lyrics + "<|endoftext|>";
}

namespace {

constexpr const char* kLmSystem =
    "<|im_start|>system\n# Instruction\nGenerate audio semantic tokens based "
    "on the given conditions:\n\n<|im_end|>\n<|im_start|>user\n";

std::string Instrumental(const std::string& lyrics) {
  return lyrics.empty() ? "[Instrumental]" : lyrics;
}

}  // namespace

std::string MusicLmPrompt(const MusicRequest& r) {
  // Sorted YAML of the user metadata, as ComfyUI writes it.
  std::string metas;
  if (!r.bpm.empty())
    metas += "bpm: " + r.bpm + "\n";
  metas +=
      "duration: " + std::to_string(static_cast<int>(std::ceil(r.duration))) +
      "\n";
  if (!r.keyscale.empty())
    metas += "keyscale: " + r.keyscale + "\n";
  if (!r.timesignature.empty())
    metas += "timesignature: " + r.timesignature + "\n";
  metas.pop_back();
  return std::string(kLmSystem) + "# Caption\n" + r.caption + "\n\n# Lyric\n" +
         Instrumental(r.lyrics) +
         "\n<|im_end|>\n<|im_start|>assistant\n<think>\n" + metas +
         "\n</think>\n\n";
}

std::string MusicLmUnconditionalPrompt() {
  return std::string(kLmSystem) +
         "NO USER INPUT<|im_end|>\n<|im_start|>assistant\n<think>\n\n"
         "</think>\n\n";
}

std::vector<float> MusicNoise(std::uint64_t seed, std::size_t count) {
  std::mt19937_64 engine(seed);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> noise(count);
  for (float& value : noise)
    value = normal(engine);
  return noise;
}

std::string EncodeStereoWav(std::span<const float> planar, int sample_rate) {
  const std::size_t samples = planar.size() / 2;
  const auto data_bytes = static_cast<std::uint32_t>(samples * 4);
  std::string out;
  out.reserve(44U + data_bytes);
  out.append("RIFF", 4);
  AppendU32(&out, 36U + data_bytes);
  out.append("WAVEfmt ", 8);
  AppendU32(&out, 16);
  AppendU16(&out, 1);  // PCM
  AppendU16(&out, 2);
  AppendU32(&out, static_cast<std::uint32_t>(sample_rate));
  AppendU32(&out, static_cast<std::uint32_t>(sample_rate) * 4U);
  AppendU16(&out, 4);
  AppendU16(&out, 16);
  out.append("data", 4);
  AppendU32(&out, data_bytes);
  for (std::size_t i = 0; i < samples; ++i)
    for (int ch = 0; ch < 2; ++ch) {
      const float v = std::clamp(planar[ch * samples + i], -1.0F, 1.0F);
      AppendU16(&out, static_cast<std::uint16_t>(static_cast<std::int16_t>(
                          std::lround(v * 32767.0F))));
    }
  return out;
}

struct MusicPipeline::Impl {
  std::unique_ptr<Tokenizer> tokenizer;
  std::unique_ptr<TextEncoder> text;
  std::unique_ptr<Dit> dit;
  std::unique_ptr<VaeDecoder> vae;
  std::unique_ptr<CodeLm> lm;
  std::vector<float> silence;  // [frames, 64]
  int silence_frames{0};

  // Silence latent frames [0, frames), repeating the tail when longer.
  std::vector<float> Silence(int frames) const {
    std::vector<float> out(std::size_t(frames) * kLatentChannels);
    for (int f = 0; f < frames; ++f) {
      const int s = std::min(f, silence_frames - 1);
      std::copy_n(silence.begin() + std::size_t(s) * kLatentChannels,
                  kLatentChannels,
                  out.begin() + std::size_t(f) * kLatentChannels);
    }
    return out;
  }
};

MusicPipeline::MusicPipeline(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
MusicPipeline::~MusicPipeline() = default;

std::unique_ptr<MusicPipeline> MusicPipeline::Load(const ModelPaths& paths,
                                                   std::string* error) {
  auto impl = std::make_unique<Impl>();
  try {
    impl->tokenizer = std::make_unique<Tokenizer>(paths.tokenizer);
  } catch (const std::exception& e) {
    Fail(error, std::string("ACE-Step tokenizer: ") + e.what());
    return nullptr;
  }
  {
    SafetensorsFile file;
    const auto* tensor = file.Open(paths.silence, nullptr, error)
                             ? file.Find("silence_latent")
                             : nullptr;
    if (tensor == nullptr || tensor->shape.size() != 2 ||
        tensor->shape[1] != kLatentChannels ||
        tensor->shape[0] < std::size_t(kTimbreFrames) ||
        !file.ReadHost("silence_latent", tensor->count(), &impl->silence,
                       error)) {
      if (error != nullptr && error->empty())
        *error = "invalid ACE-Step silence latent " + paths.silence.string();
      return nullptr;
    }
    impl->silence_frames = static_cast<int>(tensor->shape[0]);
  }
  impl->text = TextEncoder::Load(paths.text, error);
  if (!impl->text)
    return nullptr;
  impl->dit = Dit::Load(paths.dit, error);
  if (!impl->dit)
    return nullptr;
  impl->vae = VaeDecoder::Load(paths.vae, error);
  if (!impl->vae)
    return nullptr;
  if (!paths.lm.empty()) {
    impl->lm = CodeLm::Load(paths.lm, error);
    if (!impl->lm)
      return nullptr;
  }
  return std::unique_ptr<MusicPipeline>(new MusicPipeline(std::move(impl)));
}

bool MusicPipeline::Generate(const MusicRequest& request,
                             std::vector<float>* audio,
                             MusicTelemetry* telemetry,
                             const MusicProgress& progress,
                             std::string* error) {
  Impl& m = *impl_;
  MusicTelemetry local;
  MusicTelemetry& tm = telemetry != nullptr ? *telemetry : local;
  if (!(request.duration >= 1.0 && request.duration <= kMaxSeconds))
    return Fail(error, "duration must be 1..600 seconds");
  if (request.steps < 1 || request.steps > 50)
    return Fail(error, "steps must be 1..50");
  auto report = [&](double fraction, std::string_view phase) {
    return !progress || progress(fraction, phase);
  };
  auto start = std::chrono::steady_clock::now();
  std::vector<std::uint32_t> text_ids, lyric_ids;
  try {
    text_ids = m.tokenizer->Encode(MusicTextPrompt(request));
    lyric_ids = m.tokenizer->Encode(MusicLyricPrompt(request));
  } catch (const std::exception& e) {
    return Fail(error, std::string("ACE-Step tokenizer: ") + e.what());
  }
  // The Qwen3-Embedding tokenizer appends <|endoftext|> after truncation.
  if (text_ids.size() > kMaxTextTokens - 1)
    text_ids.resize(kMaxTextTokens - 1);
  if (lyric_ids.size() > kMaxLyricTokens - 1)
    lyric_ids.resize(kMaxLyricTokens - 1);
  text_ids.push_back(kEndOfText);
  lyric_ids.push_back(kEndOfText);
  tm.text_tokens = static_cast<int>(text_ids.size());
  tm.lyric_tokens = static_cast<int>(lyric_ids.size());
  std::vector<float> text, lyric, conditions;
  if (!m.text->Encode(text_ids, &text, error) ||
      !m.text->Embed(lyric_ids, &lyric, error))
    return false;
  tm.text_seconds = Since(start);
  if (!report(0.05, "conditions"))
    return Fail(error, "cancelled");
  start = std::chrono::steady_clock::now();
  const auto timbre = m.Silence(kTimbreFrames);
  if (!m.dit->EncodeConditions(text, tm.text_tokens, lyric, tm.lyric_tokens,
                               timbre, kTimbreFrames, &conditions, error))
    return false;
  const int frames = static_cast<int>(
      std::lround(request.duration * kSampleRate / kSamplesPerFrame));
  tm.frames = frames;
  // Source latents: the 5 Hz LM's planned audio codes through the
  // detokenizer, or silence (plain text-to-music); all-ones chunk mask.
  std::vector<float> source;
  if (m.lm != nullptr && request.audio_codes) {
    const auto lm_start = std::chrono::steady_clock::now();
    std::vector<std::uint32_t> cond, uncond;
    try {
      cond = m.tokenizer->Encode(MusicLmPrompt(request));
      uncond = m.tokenizer->Encode(MusicLmUnconditionalPrompt());
    } catch (const std::exception& e) {
      return Fail(error, std::string("ACE-Step tokenizer: ") + e.what());
    }
    CodeSampling sampling;
    sampling.cfg = request.lm_cfg;
    sampling.temperature = request.lm_temperature;
    sampling.top_p = request.lm_top_p;
    sampling.seed = request.seed;
    const int count = static_cast<int>(request.duration * 5);
    std::vector<int> codes;
    if (!m.lm->Generate(
            cond, uncond, count, sampling, &codes,
            [&](int done) {
              return report(0.05 + 0.4 * done / std::max(count, 1), "codes");
            },
            error) ||
        !m.dit->CodeHints(codes, frames, &source, error))
      return false;
    tm.codes = count;
    tm.lm_seconds = Since(lm_start);
  } else {
    source = m.Silence(frames);
  }
  std::vector<float> context(std::size_t(frames) * 2 * kLatentChannels, 1.0F);
  for (int f = 0; f < frames; ++f)
    std::copy_n(source.begin() + std::size_t(f) * kLatentChannels,
                kLatentChannels,
                context.begin() + std::size_t(f) * 2 * kLatentChannels);
  const int rows = static_cast<int>(conditions.size() / kConditionWidth);
  if (!m.dit->SetConditions(conditions, rows, context, frames, error))
    return false;
  tm.condition_seconds = Since(start);
  start = std::chrono::steady_clock::now();
  auto x = MusicNoise(request.seed, std::size_t(frames) * kLatentChannels);
  const auto times = TurboTimesteps(request.steps, request.shift);
  std::vector<float> v;
  for (std::size_t s = 0; s < times.size(); ++s) {
    if (!report(0.5 + 0.35 * double(s) / times.size(), "diffusion"))
      return Fail(error, "cancelled");
    if (!m.dit->Velocity(x, times[s], &v, nullptr, error))
      return false;
    const float dt = s + 1 < times.size() ? times[s] - times[s + 1] : times[s];
    for (std::size_t i = 0; i < x.size(); ++i)
      x[i] -= v[i] * dt;
  }
  tm.diffusion_seconds = Since(start);
  if (!report(0.85, "vae"))
    return Fail(error, "cancelled");
  start = std::chrono::steady_clock::now();
  if (!m.vae->Decode(x, frames, audio, error))
    return false;
  float peak = 0;
  for (float value : *audio)
    peak = std::max(peak, std::fabs(value));
  if (peak > 1.0F)
    for (float& value : *audio)
      value /= peak;
  tm.vae_seconds = Since(start);
  report(1.0, "done");
  return true;
}

}  // namespace gufo::models::ace_step
