// Written with AI assistance (Claude Code).
#include "src/models/wan22/talk21.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <utility>

#include "src/models/wan22/clip_vision.hpp"
#include "src/models/wan22/infinitetalk.hpp"
#include "src/models/wan22/pipeline.hpp"
#include "src/models/wan22/text_encoder.hpp"
#include "src/models/wan22/tokenizer.hpp"
#include "src/models/wan22/transformer21.hpp"
#include "src/models/wan22/wav2vec2.hpp"

namespace gufo::models::wan22 {
namespace {

using Clock = std::chrono::steady_clock;

double Ms(Clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin)
      .count();
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

void TrackDevice(std::size_t* peak) {
  std::size_t free = 0, total = 0;
  if (hipMemGetInfo(&free, &total) == hipSuccess)
    *peak = std::max(*peak, total - std::min(free, total));
}

int SpeakerFrames(const TalkSpeaker& speaker) {
  // int(T50 / 50 * 25), WanInfiniteTalkToVideo's linear_interpolation.
  const int frames50 = Wav2Vec2::FramesFor(speaker.audio.size());
  return static_cast<int>(static_cast<double>(frames50) / 50.0 * 25.0);
}

// The window length: `window`, or the whole (shorter) video rounded up to
// 4k + 1.
int WindowLength(int frames, int window) {
  if (frames >= window)
    return window;
  return 1 + 4 * ((std::max(frames, 1) - 1 + 3) / 4);
}

}  // namespace

int TalkAudioFrames(const std::vector<TalkSpeaker>& speakers,
                    const std::string& audio_mode) {
  int total = 0;
  for (const TalkSpeaker& speaker : speakers)
    total = audio_mode == "para" ? std::max(total, SpeakerFrames(speaker))
                                 : total + SpeakerFrames(speaker);
  return total;
}

int TalkWindows(int frames, int window, int motion_frames) {
  const int length = WindowLength(frames, window);
  if (frames <= length)
    return 1;
  const int step = length - motion_frames;
  return 1 + (frames - length + step - 1) / step;
}

bool GenerateTalk21(const Talk21Request& request,
                    const I2V21Callbacks& callbacks, DecodedVideo* video,
                    Talk21Telemetry* telemetry, std::string* error) {
  Talk21Telemetry local;
  Talk21Telemetry& tel = telemetry != nullptr ? *telemetry : local;
  const auto total_start = Clock::now();
  const int w = request.width, h = request.height;
  const int speakers = static_cast<int>(request.speakers.size());
  if (w % 16 != 0 || h % 16 != 0 || w < 16 || h < 16 || request.steps < 1 ||
      request.window < 5 || (request.window - 1) % 4 != 0 ||
      request.motion_frames < 1 || (request.motion_frames - 1) % 4 != 0 ||
      request.motion_frames >= request.window ||
      request.first_frame.size() != static_cast<std::size_t>(w) * h * 3) {
    SetError(error,
             "InfiniteTalk: width/height must be multiples of 16, window and "
             "motion frames 4k+1 (motion < window), and the first frame must "
             "be at the video size");
    return false;
  }
  if (speakers < 1 || speakers > InfiniteTalk::kMaxSpeakers ||
      (request.audio_mode != "add" && request.audio_mode != "para")) {
    SetError(error, "InfiniteTalk: one or two speakers, audio_mode add or "
                    "para");
    return false;
  }
  for (const TalkSpeaker& speaker : request.speakers)
    if (Wav2Vec2::FramesFor(speaker.audio.size()) < 2 ||
        (speakers > 1 &&
         (speaker.mask_width < 1 || speaker.mask_height < 1 ||
          speaker.mask.size() != static_cast<std::size_t>(speaker.mask_width) *
                                     speaker.mask_height))) {
      SetError(error, "InfiniteTalk: every speaker needs audio (and with two "
                      "speakers a mask)");
      return false;
    }
  const auto progress = [&](const std::string& phase, int done, int total) {
    if (callbacks.progress)
      callbacks.progress(phase, done, total);
  };
  const auto cancelled = [&] {
    if (callbacks.keep_going && !callbacks.keep_going()) {
      SetError(error, "cancelled");
      return true;
    }
    return false;
  };
  const auto dump = [&](const std::string& name,
                        const std::vector<float>& values,
                        std::vector<std::size_t> shape) {
    if (callbacks.dump)
      callbacks.dump(name, values, shape);
  };
  const int audio_frames = TalkAudioFrames(request.speakers,
                                           request.audio_mode);
  tel.audio_frames = audio_frames;
  const int total_frames = request.frames > 0 ? request.frames : audio_frames;
  if (total_frames < 1) {
    SetError(error, "InfiniteTalk: the audio is too short");
    return false;
  }
  const int length = WindowLength(total_frames, request.window);
  const int windows = TalkWindows(total_frames, request.window,
                                  request.motion_frames);
  const int lf = 1 + (length - 1) / 4, lh = h / 8, lw = w / 8;
  const std::size_t plane = static_cast<std::size_t>(lf) * lh * lw;
  const std::size_t frame_plane = static_cast<std::size_t>(lh) * lw;
  const auto shape16 = std::vector<std::size_t>{
      16, static_cast<std::size_t>(lf), static_cast<std::size_t>(lh),
      static_cast<std::size_t>(lw)};

  // 1. Prompt (umT5, 512 rows: the states of the real tokens, then zeros).
  std::vector<float> text;
  {
    auto start = Clock::now();
    const auto tokenizer = Tokenizer::Load(
        request.t5_root / "tokenizer" / "tokenizer.json", error);
    if (!tokenizer)
      return false;
    auto encoder = TextEncoder::Load(request.t5_root / "text_encoder", error);
    if (!encoder)
      return false;
    TrackDevice(&tel.peak_device_bytes);
    const auto ids = tokenizer->Encode(CleanPrompt(request.prompt));
    std::vector<float> states;
    if (!encoder->Encode(ids, &states, error))
      return false;
    text.assign(TextEncoder::kMaxTokens * TextEncoder::kModelWidth, 0.0F);
    std::copy(states.begin(), states.end(), text.begin());
    tel.text_ms = Ms(start);
    progress("prompt", 1, 1);
  }
  if (cancelled())
    return false;

  // 2. CLIP image states.
  std::vector<float> clip_states;
  {
    auto start = Clock::now();
    auto clip = ClipVision::Load(request.wan21_root / "image_encoder", error);
    if (!clip)
      return false;
    const auto pixels = ClipVision::Preprocess(request.first_frame, w, h);
    if (!clip->Encode(pixels, &clip_states, error))
      return false;
    TrackDevice(&tel.peak_device_bytes);
    tel.clip_ms = Ms(start);
  }

  // 3. Audio features [frames25][12][768] per speaker, placed in turn
  // ("add": speaker 2 starts where speaker 1 ends, zeros elsewhere) or
  // together ("para"), all of `audio_frames`.
  std::vector<std::vector<float>> features(speakers);
  {
    auto start = Clock::now();
    auto encoder = Wav2Vec2::Load(request.wav2vec2_file, error);
    if (!encoder)
      return false;
    const std::size_t frame_values =
        static_cast<std::size_t>(Wav2Vec2::kLayers) * Wav2Vec2::kWidth;
    int offset = 0;
    for (int s = 0; s < speakers; ++s) {
      std::vector<float> layers;
      int frames50 = 0, frames25 = 0;
      if (!encoder->Encode(request.speakers[s].audio, &layers, &frames50,
                           error))
        return false;
      const std::vector<float> own =
          TalkAudioFeatures(layers, frames50, &frames25);
      features[s].assign(frame_values * audio_frames, 0.0F);
      const int place = request.audio_mode == "add" ? offset : 0;
      std::copy_n(own.begin(),
                  frame_values * std::min(frames25, audio_frames - place),
                  features[s].begin() + frame_values * place);
      offset += frames25;
      dump("audio_features_" + std::to_string(s), features[s],
           {static_cast<std::size_t>(audio_frames),
            static_cast<std::size_t>(Wav2Vec2::kLayers),
            static_cast<std::size_t>(Wav2Vec2::kWidth)});
    }
    TrackDevice(&tel.peak_device_bytes);
    tel.audio_ms = Ms(start);
  }
  if (cancelled())
    return false;

  // 4. VAE condition: the start image, then grey, encoded and normalized.
  auto vae_encoder = VaeEncoder::Load(request.wan21_root / "vae", error);
  if (!vae_encoder)
    return false;
  std::vector<float> condition;
  {
    auto start = Clock::now();
    const std::size_t frame = static_cast<std::size_t>(w) * h * 3;
    std::vector<float> input(frame * length, 0.0F);
    for (std::size_t i = 0; i < frame; ++i)
      input[i] = static_cast<float>(request.first_frame[i]) / 127.5F - 1.0F;
    if (!vae_encoder->EncodeVideo(input, length, h, w, true, &condition,
                                  error))
      return false;
    TrackDevice(&tel.peak_device_bytes);
    tel.condition_ms = Ms(start);
  }
  dump("condition", condition, shape16);

  // 5. Transformer, InfiniteTalk adapter, VAE decoder (kept for all windows).
  auto start = Clock::now();
  auto dit = Transformer21::Load(request.dit_file, request.lora_file,
                                 request.lora_strength, error);
  if (!dit)
    return false;
  auto talk = InfiniteTalk::Load(request.talk_file, dit->stream(), error);
  if (!talk)
    return false;
  talk->set_audio_scale(request.audio_scale);
  if (speakers > 1) {
    std::vector<std::vector<std::uint8_t>> masks;
    for (const TalkSpeaker& speaker : request.speakers) {
      masks.push_back(TokenMask(speaker.mask, speaker.mask_width,
                                speaker.mask_height, lw / 2, lh / 2));
      if (masks.back().empty()) {
        SetError(error, "InfiniteTalk: invalid speaker mask");
        return false;
      }
    }
    if (!talk->SetSpeakerMasks(masks, lh / 2, lw / 2, error))
      return false;
  }
  auto decoder = VaeDecoder::Load(request.wan21_root / "vae", error);
  if (!decoder)
    return false;
  tel.dit_load_ms = Ms(start);
  start = Clock::now();
  if (!dit->PrepareContext(0, text, TextEncoder::kMaxTokens, clip_states,
                           nullptr, nullptr, error))
    return false;
  tel.context_ms = Ms(start);
  TrackDevice(&tel.peak_device_bytes);

  // Model input: 16 latent channels, 4 mask channels (1 on the first
  // latent frame), 16 condition channels.
  std::vector<float> input(36 * plane, 0.0F);
  for (int c = 0; c < 4; ++c)
    std::fill_n(input.begin() + (16 + c) * plane, frame_plane, 1.0F);
  std::copy(condition.begin(), condition.end(), input.begin() + 20 * plane);
  std::vector<double> sigmas;
  for (int i = 0; i < request.steps; ++i) {
    const double t = 1.0 - static_cast<double>(i) / request.steps;
    sigmas.push_back(request.shift * t / (1.0 + (request.shift - 1.0) * t));
  }
  sigmas.push_back(0.0);
  std::vector<std::span<const float>> feature_spans;
  for (const auto& f : features)
    feature_spans.emplace_back(f);

  const std::size_t frame_bytes = static_cast<std::size_t>(w) * h * 3;
  video->frames = 0;
  video->width = w;
  video->height = h;
  video->rgb.clear();
  video->rgb8.clear();
  video->rgb8.reserve(frame_bytes * total_frames);
  std::vector<float> motion_rgb;  // last decoded frames, [-1, 1]
  int produced = 0;
  const int total_steps = windows * request.steps;
  for (int window = 0; window < windows; ++window) {
    if (cancelled())
      return false;
    // Motion latents pinned at the front of every model input: the start
    // image's latent frame, then the re-encoded last decoded frames.
    std::vector<float> motion;
    int motion_latents = 1;
    int audio_start = 0;
    if (window == 0) {
      motion.resize(16 * frame_plane);
      for (int c = 0; c < 16; ++c)
        std::copy_n(condition.begin() + c * plane, frame_plane,
                    motion.begin() + c * frame_plane);
    } else {
      auto motion_start = Clock::now();
      audio_start = produced - request.motion_frames;
      if (!vae_encoder->EncodeVideo(motion_rgb, request.motion_frames, h, w,
                                    true, &motion, error))
        return false;
      motion_latents = 1 + (request.motion_frames - 1) / 4;
      tel.motion_ms += Ms(motion_start);
    }
    if (!talk->SetAudio(feature_spans, audio_frames, audio_start, length,
                        error))
      return false;
    const auto pin = [&](std::vector<float>& latents) {
      for (int c = 0; c < 16; ++c)
        std::copy_n(motion.begin() +
                        static_cast<std::size_t>(c) * motion_latents *
                            frame_plane,
                    motion_latents * frame_plane,
                    latents.begin() + c * plane);
    };
    const std::uint64_t seed = request.seed + static_cast<std::uint64_t>(window);
    std::vector<float> latents = GaussianNoise(16 * plane, seed);
    std::vector<float> velocity;
    for (int step = 0; step < request.steps; ++step) {
      if (cancelled())
        return false;
      const auto step_start = Clock::now();
      pin(latents);
      std::copy(latents.begin(), latents.end(), input.begin());
      Transformer21Input in;
      in.latent = input;
      in.frames = lf;
      in.height = lh;
      in.width = lw;
      in.timestep = static_cast<float>(1000.0 * sigmas[step]);
      in.hook = talk.get();
      if (!dit->Forward(in, &velocity, nullptr, error))
        return false;
      TrackDevice(&tel.peak_device_bytes);
      if (window == 0) {
        const std::string tag = "step" + std::to_string(step);
        dump(tag + "_input", input,
             {36, static_cast<std::size_t>(lf), static_cast<std::size_t>(lh),
              static_cast<std::size_t>(lw)});
        dump(tag + "_velocity", velocity, shape16);
      }
      const float sigma = static_cast<float>(sigmas[step]);
      const float next = static_cast<float>(sigmas[step + 1]);
      const int done = window * request.steps + step + 1;
      if (callbacks.preview.Wants(done, total_steps)) {
        const auto preview_start = Clock::now();
        std::vector<float> x0(latents.size());
        for (std::size_t i = 0; i < latents.size(); ++i)
          x0[i] = latents[i] - sigma * velocity[i];
        callbacks.preview.publish({.step = done,
                                   .steps = total_steps,
                                   .frames = lf,
                                   .height = lh,
                                   .width = lw,
                                   .channels = 16,
                                   .channel_major = true,
                                   .data = x0});
        tel.preview_ms += Ms(preview_start);
        ++tel.previews;
      }
      // LCM: x0 = x - sigma * v; re-noise to the next sigma.
      std::vector<float> fresh;
      if (next > 0.0F)
        fresh = GaussianNoise(latents.size(), seed + 1 + step);
      for (std::size_t i = 0; i < latents.size(); ++i) {
        const float x0 = latents[i] - sigma * velocity[i];
        latents[i] = next > 0.0F ? (1.0F - next) * x0 + next * fresh[i] : x0;
      }
      tel.step_ms.push_back(Ms(step_start));
      progress("denoiser", done, total_steps);
    }
    // Continuations decode with the motion latents in front (ComfyUI's
    // InfiniteTalkOuterSampleWrapper), then drop those frames.
    if (window > 0)
      pin(latents);
    if (window == 0)
      dump("final", latents, shape16);
    if (cancelled())
      return false;
    const auto decode_start = Clock::now();
    const bool more = window + 1 < windows;
    DecodedVideo part;
    if (!decoder->Decode(
            latents, lf, lh, lw, more, true, &part,
            [&](int done, int total) {
              progress("video_vae", window * total + done, windows * total);
            },
            callbacks.keep_going, error))
      return false;
    tel.decode_ms += Ms(decode_start);
    TrackDevice(&tel.peak_device_bytes);
    const int skip = window == 0 ? 0 : request.motion_frames;
    const int keep = std::min(part.frames - skip, total_frames - produced);
    video->rgb8.insert(video->rgb8.end(),
                       part.rgb8.begin() + frame_bytes * skip,
                       part.rgb8.begin() + frame_bytes * (skip + keep));
    produced += keep;
    if (more) {
      // The next window's motion frames: the last ones of everything so
      // far (ComfyUI's previous_frames[-motion_frame_count:]).
      motion_rgb.assign(part.rgb.end() -
                            static_cast<std::ptrdiff_t>(frame_bytes) *
                                request.motion_frames,
                        part.rgb.end());
    }
    ++tel.windows;
  }
  video->frames = produced;
  tel.total_ms = Ms(total_start);
  return true;
}

}  // namespace gufo::models::wan22
