// Written with AI assistance (Claude Code).
#include "src/cli/serve/video_talk.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <system_error>
#include <utility>

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/wan22/pipeline.hpp"
#include "src/models/wan22/talk21.hpp"
#endif

namespace gufo::server {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

constexpr double kPi = 3.14159265358979323846;

// The wav2vec2 convolution stack: encoder frames for 16 kHz samples.
int EncoderFrames(std::size_t samples) {
  static constexpr int kKernels[7] = {10, 3, 3, 3, 3, 2, 2};
  static constexpr int kStrides[7] = {5, 2, 2, 2, 2, 2, 2};
  long long length = static_cast<long long>(samples);
  for (int i = 0; i < 7; ++i) {
    if (length < kKernels[i])
      return 0;
    length = (length - kKernels[i]) / kStrides[i] + 1;
  }
  return static_cast<int>(length);
}

std::size_t Samples16k(const minimax_h3::AudioWaveform& audio) {
  if (audio.sample_rate <= 0)
    return 0;
  return static_cast<std::size_t>(
      (static_cast<std::uint64_t>(audio.samples) * 16000U +
       static_cast<std::uint64_t>(audio.sample_rate) - 1U) /
      static_cast<std::uint64_t>(audio.sample_rate));
}

// 16-bit PCM WAV of channel-major float PCM.
bool WriteWav(const std::filesystem::path& path, int channels, int rate,
              const std::vector<float>& pcm, std::string* error) {
  const std::size_t frames = channels > 0 ? pcm.size() / channels : 0;
  std::string data(frames * channels * 2, '\0');
  for (std::size_t i = 0; i < frames; ++i)
    for (int c = 0; c < channels; ++c) {
      const float v = std::clamp(pcm[c * frames + i], -1.0F, 1.0F);
      const auto s = static_cast<std::int16_t>(std::lround(v * 32767.0F));
      std::memcpy(&data[(i * channels + c) * 2], &s, 2);
    }
  const auto u32 = [](std::uint32_t v) {
    std::string out(4, '\0');
    std::memcpy(out.data(), &v, 4);
    return out;
  };
  const auto u16 = [](std::uint16_t v) {
    std::string out(2, '\0');
    std::memcpy(out.data(), &v, 2);
    return out;
  };
  std::ofstream file(path, std::ios::binary);
  file << "RIFF" << u32(static_cast<std::uint32_t>(36 + data.size()))
       << "WAVEfmt " << u32(16) << u16(1)
       << u16(static_cast<std::uint16_t>(channels))
       << u32(static_cast<std::uint32_t>(rate))
       << u32(static_cast<std::uint32_t>(rate * channels * 2))
       << u16(static_cast<std::uint16_t>(channels * 2)) << u16(16) << "data"
       << u32(static_cast<std::uint32_t>(data.size())) << data;
  if (!file) {
    SetError(error, "cannot write " + path.string());
    return false;
  }
  return true;
}

}  // namespace

int TalkFramesForAudio(const minimax_h3::AudioWaveform& audio) {
  const int frames50 = EncoderFrames(Samples16k(audio));
  return static_cast<int>(static_cast<double>(frames50) / 50.0 * 25.0);
}

std::vector<float> TalkMono16k(const minimax_h3::AudioWaveform& audio) {
  const std::size_t n = static_cast<std::size_t>(std::max(audio.samples, 0));
  const int channels = std::max(audio.channels, 1);
  std::vector<float> mono(n, 0.0F);
  for (int c = 0; c < channels; ++c)
    for (std::size_t i = 0; i < n && c * n + i < audio.pcm.size(); ++i)
      mono[i] += audio.pcm[c * n + i] / static_cast<float>(channels);
  if (audio.sample_rate == 16000 || n == 0)
    return mono;
  // Hann-windowed sinc, 32 zero crossings each side, cutoff at the lower
  // Nyquist rate.
  const std::size_t out = Samples16k(audio);
  const double ratio = static_cast<double>(audio.sample_rate) / 16000.0;
  const double cutoff = std::min(1.0, 1.0 / ratio);
  constexpr int kZeros = 32;
  const double half = kZeros / cutoff;
  std::vector<float> result(out, 0.0F);
  // Taps depend only on the fractional source position: one table per phase
  // (a single one for 32 kHz).
  struct Taps {
    long long lead{0};
    std::vector<double> weights;
  };
  std::map<double, Taps> phases;
  for (std::size_t j = 0; j < out; ++j) {
    const double centre = static_cast<double>(j) * ratio;
    const double base = std::floor(centre);
    const double frac = centre - base;
    auto found = phases.find(frac);
    if (found == phases.end()) {
      Taps taps;
      taps.lead = static_cast<long long>(std::ceil(frac - half));
      const long long tail = static_cast<long long>(std::floor(frac + half));
      for (long long k = taps.lead; k <= tail; ++k) {
        const double d = static_cast<double>(k) - frac;
        const double x = d * cutoff;
        const double sinc = x == 0.0 ? 1.0 : std::sin(kPi * x) / (kPi * x);
        const double window = 0.5 + 0.5 * std::cos(kPi * d / half);
        taps.weights.push_back(sinc * window * cutoff);
      }
      found = phases.emplace(frac, std::move(taps)).first;
    }
    const Taps& taps = found->second;
    double sum = 0, weights = 0;
    for (std::size_t k = 0; k < taps.weights.size(); ++k) {
      const long long i = static_cast<long long>(base) + taps.lead +
                          static_cast<long long>(k);
      if (i < 0 || i >= static_cast<long long>(n))
        continue;
      sum += taps.weights[k] * mono[static_cast<std::size_t>(i)];
      weights += taps.weights[k];
    }
    // Normalized by the taps' sum (unity gain, also where the kernel is cut
    // at the edges).
    result[j] = weights != 0.0 ? static_cast<float>(sum / weights) : 0.0F;
  }
  return result;
}

bool ParseTalkOptions(const json::Value& talk, std::string* normalized,
                      std::string* error) {
  if (!talk.is_object()) {
    SetError(error, "'gufo.talk' must be an object");
    return false;
  }
  for (const auto& [key, value] : talk.members()) {
    (void)value;
    if (key != "audio_scale" && key != "audio_mode" && key != "motion_frames") {
      SetError(error, "'gufo.talk' may contain only 'audio_scale', "
                      "'audio_mode' and 'motion_frames'");
      return false;
    }
  }
  json::Value out = json::Value::object();
  if (const json::Value* scale = talk.find("audio_scale")) {
    const double v = scale->is_number() ? scale->as_double(NAN) : NAN;
    if (!std::isfinite(v) || v < -10.0 || v > 10.0) {
      SetError(error, "'gufo.talk.audio_scale' must be -10..10");
      return false;
    }
    out["audio_scale"] = v;
  }
  if (const json::Value* mode = talk.find("audio_mode")) {
    if (!mode->is_string() || (mode->str() != "add" && mode->str() != "para")) {
      SetError(error, "'gufo.talk.audio_mode' must be \"add\" or \"para\"");
      return false;
    }
    out["audio_mode"] = mode->str();
  }
  if (const json::Value* motion = talk.find("motion_frames")) {
    const double v = motion->is_number() ? motion->as_double(NAN) : NAN;
    if (!std::isfinite(v) || std::floor(v) != v || v < 1 || v > 33 ||
        (static_cast<int>(v) - 1) % 4 != 0) {
      SetError(error, "'gufo.talk.motion_frames' must be 4k + 1 in 1..33");
      return false;
    }
    out["motion_frames"] = static_cast<int>(v);
  }
  *normalized = out.dump();
  return true;
}

bool RunWan21TalkJob(const minimax_h3::GenerationRequest& request,
                     const minimax_h3::CancellationToken* cancellation,
                     minimax_h3::GenerationProgress progress,
                     void* progress_opaque,
                     minimax_h3::GenerationTelemetry* telemetry,
                     std::string* error) {
#if defined(ENGINE_ENABLE_HIP)
  namespace wan22 = models::wan22;
  using Clock = std::chrono::steady_clock;
  json::Value manifest;
  json::Value options = json::Value::object();
  try {
    std::ifstream input(request.model_root / "gufo-talk.json",
                        std::ios::binary);
    std::stringstream text;
    text << input.rdbuf();
    manifest = json::parse(text.str());
    if (!request.talk_json.empty())
      options = json::parse(request.talk_json);
  } catch (const std::exception& e) {
    SetError(error, std::string("gufo-talk.json / gufo.talk: ") + e.what());
    return false;
  }
  const auto part = [&](const char* key) {
    const std::filesystem::path value = manifest.member_str(key);
    return value.empty() || value.is_absolute() ? value
                                                : request.model_root / value;
  };
  const int speakers = static_cast<int>(request.audio_guides.size());
  if (speakers < 1 || speakers > 2 ||
      (speakers == 2 && request.speaker_masks.size() != 2)) {
    SetError(error, "InfiniteTalk needs one or two speakers (two with masks)");
    return false;
  }
  wan22::Talk21Request job;
  job.wan21_root = part("wan21");
  job.t5_root = part("t5");
  job.dit_file = part("dit");
  job.lora_file = part("lora");
  job.lora_strength =
      static_cast<float>(manifest.member_double("lora_strength", 1.0));
  job.talk_file = speakers == 2 && !part("talk_multi").empty()
                      ? part("talk_multi")
                      : part("talk");
  job.wav2vec2_file = part("wav2vec2");
  job.shift = static_cast<float>(manifest.member_double("shift", 5.0));
  job.window = static_cast<int>(manifest.member_double("window", 81));
  job.motion_frames = static_cast<int>(options.member_double(
      "motion_frames", manifest.member_double("motion_frames", 9)));
  job.audio_scale =
      static_cast<float>(options.member_double("audio_scale", 1.0));
  job.audio_mode = options.member_str("audio_mode", "add");
  job.prompt = request.prompt;
  job.width = request.parameters.output_width;
  job.height = request.parameters.output_height;
  job.frames = request.parameters.frames;
  job.steps = request.parameters.evaluations;
  job.seed = request.seed;
  const std::vector<std::uint8_t> first_frame =
      wan22::FitImage(request.first_frame, request.first_frame_width,
                      request.first_frame_height, job.width, job.height);
  if (first_frame.empty()) {
    SetError(error, "invalid first frame");
    return false;
  }
  job.first_frame = first_frame;
  for (int s = 0; s < speakers; ++s) {
    wan22::TalkSpeaker speaker;
    speaker.audio = TalkMono16k(request.audio_guides[s].waveform);
    if (speakers == 2) {
      // The mask is fitted like the start image (cover, centre crop) so it
      // lines up with the picture; bright pixels are the speaker.
      const auto& image = request.speaker_masks[s];
      const auto fitted = wan22::FitImage(image.rgb, image.width, image.height,
                                          job.width, job.height);
      if (fitted.empty()) {
        SetError(error, "invalid speaker mask " + std::to_string(s + 1));
        return false;
      }
      speaker.mask_width = job.width;
      speaker.mask_height = job.height;
      speaker.mask.resize(static_cast<std::size_t>(job.width) * job.height);
      for (std::size_t i = 0; i < speaker.mask.size(); ++i) {
        const int sum = fitted[3 * i] + fitted[3 * i + 1] + fitted[3 * i + 2];
        speaker.mask[i] = sum >= 3 * 128 ? 255 : 0;
      }
    }
    job.speakers.push_back(std::move(speaker));
  }
  // The soundtrack as the video uses it: in turn ("add", speaker 2 from the
  // frame where speaker 1's features end) or mixed ("para"), 32 kHz stereo.
  const std::filesystem::path soundtrack =
      request.output_path.string() + ".gufo-partial-talk.wav";
  {
    constexpr int kRate = 32000;
    std::vector<std::vector<float>> tracks;
    std::vector<std::size_t> offsets;
    std::size_t length = 0;
    double start_frames = 0;
    for (int s = 0; s < speakers; ++s) {
      const auto& wave = request.audio_guides[s].waveform;
      if (wave.sample_rate != kRate || wave.channels != 2) {
        SetError(error, "speaker audio must be decoded to 32 kHz stereo");
        return false;
      }
      const std::size_t offset =
          job.audio_mode == "add"
              ? static_cast<std::size_t>(std::lround(start_frames / 25.0 *
                                                     kRate))
              : 0;
      offsets.push_back(offset);
      tracks.push_back(wave.pcm);
      length = std::max(length, offset + static_cast<std::size_t>(wave.samples));
      start_frames += TalkFramesForAudio(wave);
    }
    std::vector<float> mix(2 * length, 0.0F);
    for (int s = 0; s < speakers; ++s) {
      const std::size_t n = tracks[s].size() / 2;
      for (int c = 0; c < 2; ++c)
        for (std::size_t i = 0; i < n; ++i)
          mix[c * length + offsets[s] + i] += tracks[s][c * n + i];
    }
    if (!WriteWav(soundtrack, 2, kRate, mix, error))
      return false;
  }
  wan22::I2V21Callbacks callbacks;
  callbacks.progress = [&](const std::string& phase, int done, int total) {
    if (progress != nullptr)
      progress(phase, done, total, progress_opaque);
  };
  callbacks.keep_going = [cancellation] {
    return cancellation == nullptr || !cancellation->IsCancelled();
  };
  callbacks.preview = request.preview;
  wan22::DecodedVideo video;
  wan22::Talk21Telemetry stages;
  std::error_code ignored;
  if (!wan22::GenerateTalk21(job, callbacks, &video, &stages, error)) {
    std::filesystem::remove(soundtrack, ignored);
    return false;
  }
  if (cancellation != nullptr && cancellation->IsCancelled()) {
    std::filesystem::remove(soundtrack, ignored);
    SetError(error, "cancelled");
    return false;
  }
  callbacks.progress("media", 0, 1);
  const auto media_start = Clock::now();
  const bool written = wan22::WriteMp4WithAudio(request.output_path, video, 25,
                                                soundtrack, error);
  std::filesystem::remove(soundtrack, ignored);
  if (!written)
    return false;
  callbacks.progress("media", 1, 1);
  if (telemetry != nullptr) {
    telemetry->prompt_ms = stages.text_ms;
    telemetry->keyframe_ms = stages.clip_ms + stages.condition_ms;
    telemetry->audio_guide_ms = stages.audio_ms;
    telemetry->audio_guides = static_cast<std::size_t>(speakers);
    telemetry->audio_guide_frames = static_cast<std::size_t>(stages.audio_frames);
    double denoise = 0;
    for (double ms : stages.step_ms)
      denoise += ms;
    telemetry->denoiser_ms =
        stages.dit_load_ms + stages.context_ms + stages.motion_ms + denoise;
    telemetry->denoiser.preview_ms = stages.preview_ms;
    telemetry->denoiser.previews = stages.previews;
    telemetry->video_vae_ms = stages.decode_ms;
    telemetry->media_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - media_start)
            .count();
    telemetry->total_ms = stages.total_ms + telemetry->media_ms;
  }
  return true;
#else
  (void)request;
  (void)cancellation;
  (void)progress;
  (void)progress_opaque;
  (void)telemetry;
  SetError(error, "Wan 2.1 InfiniteTalk requires a ROCm HIP-enabled build");
  return false;
#endif
}

}  // namespace gufo::server
