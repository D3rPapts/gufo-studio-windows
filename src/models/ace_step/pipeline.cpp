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

}  // namespace

// The host side (paths, prompts, noise, WAV) is in prompt.cpp.

struct MusicPipeline::Impl {
  std::unique_ptr<Tokenizer> tokenizer;
  std::unique_ptr<TextEncoder> text;
  std::unique_ptr<Dit> dit;
  std::unique_ptr<VaeDecoder> vae;
  std::filesystem::path lm_path, dit_path;
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
  // The 8.4 GB LM is loaded per request and released before the DiT runs,
  // so only one large model holds device memory at a time.
  impl->lm_path = paths.lm;
  impl->dit_path = paths.dit;
  return std::unique_ptr<MusicPipeline>(new MusicPipeline(std::move(impl)));
}

namespace {

// Decodes the preview window of the clean estimate x0 = x - t * v (see
// MusicCleanEstimate) and hands the mono WAV to the sink. A failed decode
// only drops this preview; the generation itself goes on.
void PublishPreview(VaeDecoder& vae, const MusicPreviewSink& sink,
                    std::span<const float> x, std::span<const float> v,
                    float t, int frames, int step, int steps) {
  const auto start = std::chrono::steady_clock::now();
  const PreviewWindow window =
      MusicPreviewWindow(frames, sink.window_seconds, kPreviewMarginFrames);
  if (window.decode_count <= 0)
    return;
  const std::size_t offset =
      std::size_t(window.decode_first) * kLatentChannels;
  const std::size_t length =
      std::size_t(window.decode_count) * kLatentChannels;
  std::vector<float> x0;
  MusicCleanEstimate(x.subspan(offset, length), v.subspan(offset, length), t,
                     &x0);
  std::vector<float> audio;
  std::string error;
  if (!vae.Decode(x0, window.decode_count, &audio, &error))
    return;
  // Cut the margin: the decoded slice is planar [2][decode_count * 1920].
  const std::size_t decoded = audio.size() / 2;
  const auto skip = std::size_t(window.skip_samples());
  const auto keep = std::size_t(window.samples());
  if (skip + keep > decoded)
    return;
  std::vector<float> planar(2 * keep);
  std::copy_n(audio.begin() + skip, keep, planar.begin());
  std::copy_n(audio.begin() + decoded + skip, keep, planar.begin() + keep);
  MusicPreview preview;
  preview.step = step;
  preview.steps = steps;
  preview.start_seconds = window.start_seconds();
  preview.window_seconds = window.seconds();
  preview.wav = EncodePreviewWav(planar);
  preview.decode_ms = Since(start) * 1000.0;
  if (sink.publish)
    sink.publish(std::move(preview));
}

}  // namespace

bool MusicPipeline::Generate(const MusicRequest& request,
                             std::vector<float>* audio,
                             MusicTelemetry* telemetry,
                             const MusicProgress& progress,
                             std::string* error,
                             const MusicPreviewSink* preview) {
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
  const int frames = static_cast<int>(
      std::lround(request.duration * kSampleRate / kSamplesPerFrame));
  tm.frames = frames;
  // Source latents: the 5 Hz LM's planned audio codes through the
  // detokenizer, or silence (plain text-to-music); all-ones chunk mask.
  // The LM and the DiT never hold device memory at the same time.
  std::vector<int> codes;
  const bool plan = !m.lm_path.empty() && request.audio_codes;
  if (plan) {
    const auto lm_start = std::chrono::steady_clock::now();
    m.dit.reset();
    auto lm = CodeLm::Load(m.lm_path, error);
    if (!lm)
      return false;
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
    if (!lm->Generate(
            cond, uncond, count, sampling, &codes,
            [&](int done) {
              return report(0.05 + 0.4 * done / std::max(count, 1), "codes");
            },
            error))
      return false;
    lm.reset();
    tm.codes = count;
    tm.lm_seconds = Since(lm_start);
  }
  start = std::chrono::steady_clock::now();
  if (m.dit == nullptr) {
    m.dit = Dit::Load(m.dit_path, error);
    if (!m.dit)
      return false;
  }
  const auto timbre = m.Silence(kTimbreFrames);
  if (!m.dit->EncodeConditions(text, tm.text_tokens, lyric, tm.lyric_tokens,
                               timbre, kTimbreFrames, &conditions, error))
    return false;
  std::vector<float> source;
  if (plan) {
    if (!m.dit->CodeHints(codes, frames, &source, error))
      return false;
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
  // Live preview: the clean estimate of the current step is decoded (a short
  // window only) when the step is a candidate, someone wants it and the
  // previous preview is old enough; its time counts as diffusion time.
  const auto preview_interval =
      preview != nullptr
          ? std::chrono::duration<double>(preview->min_interval_seconds)
          : std::chrono::duration<double>(0);
  bool had_preview = false;
  auto last_preview = std::chrono::steady_clock::now();
  for (std::size_t s = 0; s < times.size(); ++s) {
    if (!report(0.5 + 0.35 * double(s) / times.size(), "diffusion"))
      return Fail(error, "cancelled");
    if (!m.dit->Velocity(x, times[s], &v, nullptr, error))
      return false;
    const int step = static_cast<int>(s) + 1;
    const int steps = static_cast<int>(times.size());
    if (preview != nullptr && preview->wanted &&
        MusicPreviewStep(step, steps, preview->every) &&
        (!had_preview ||
         std::chrono::steady_clock::now() - last_preview >= preview_interval) &&
        preview->wanted()) {
      PublishPreview(*m.vae, *preview, x, v, times[s], frames, step, steps);
      had_preview = true;
      last_preview = std::chrono::steady_clock::now();
    }
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
