#include "src/models/wan22/pipeline21.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <utility>

#include "src/models/wan22/clip_vision.hpp"
#include "src/models/wan22/moge.hpp"
#include "src/models/wan22/pipeline.hpp"
#include "src/models/wan22/point_render.hpp"
#include "src/models/wan22/text_encoder.hpp"
#include "src/models/wan22/tokenizer.hpp"
#include "src/models/wan22/transformer21.hpp"
#include "src/models/wan22/uni3c.hpp"

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

// RGB8 frames [count, h, w, 3] -> FP32 [-1, 1] video of `frames` frames:
// cut, or the last frame repeated (ComfyUI's Uni3C patch).
std::vector<float> GuideVideo(std::span<const std::uint8_t> rgb, int count,
                              int frames, int height, int width) {
  const std::size_t frame = static_cast<std::size_t>(height) * width * 3;
  std::vector<float> video(frame * frames);
  for (int f = 0; f < frames; ++f) {
    const std::uint8_t* source = rgb.data() + frame * std::min(f, count - 1);
    for (std::size_t i = 0; i < frame; ++i)
      video[frame * f + i] = static_cast<float>(source[i]) / 127.5F - 1.0F;
  }
  return video;
}

}  // namespace

bool RenderCameraGuide(const I2V21Request& request,
                       std::vector<std::uint8_t>* frames,
                       I2V21Telemetry* telemetry, std::string* error) {
  I2V21Telemetry local;
  I2V21Telemetry& tel = telemetry != nullptr ? *telemetry : local;
  auto start = Clock::now();
  auto moge = Moge::Load(request.moge_file, error);
  if (!moge)
    return false;
  MogeResult geometry;
  if (!moge->Infer(request.first_frame, request.width, request.height,
                   &geometry, nullptr, error))
    return false;
  moge.reset();
  tel.depth_ms = Ms(start);
  start = Clock::now();
  tel.scene_depth = SceneDepth(geometry);
  tel.fx = geometry.fx;
  tel.metric_scale = geometry.metric_scale;
  std::vector<CameraPose> poses;
  if (!BuildCameraPath(request.camera, request.frames, tel.scene_depth, &poses,
                       error) ||
      !RenderGuide(geometry, request.first_frame, poses, frames, error))
    return false;
  tel.render_ms = Ms(start);
  return true;
}

bool GenerateI2V21(const I2V21Request& request, const I2V21Callbacks& callbacks,
                   std::vector<float>* final_latents, DecodedVideo* video,
                   I2V21Telemetry* telemetry, std::string* error) {
  I2V21Telemetry local;
  I2V21Telemetry& tel = telemetry != nullptr ? *telemetry : local;
  const auto total_start = Clock::now();
  const int w = request.width, h = request.height, n = request.frames;
  if (w % 16 != 0 || h % 16 != 0 || w < 16 || h < 16 || n < 1 ||
      (n - 1) % 4 != 0 || request.steps < 1 ||
      request.first_frame.size() != static_cast<std::size_t>(w) * h * 3) {
    SetError(error,
             "Wan 2.1 I2V: width/height must be multiples of 16, frames 4k+1, "
             "and the first frame must be at the video size");
    return false;
  }
  const auto dump = [&](const std::string& name,
                        const std::vector<float>& values,
                        std::vector<std::size_t> shape) {
    if (callbacks.dump)
      callbacks.dump(name, values, shape);
  };
  const auto progress = [&](const std::string& phase, int done, int total) {
    if (callbacks.progress)
      callbacks.progress(phase, done, total);
  };
  const int lf = 1 + (n - 1) / 4, lh = h / 8, lw = w / 8;
  const std::size_t plane = static_cast<std::size_t>(lf) * lh * lw;

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
  dump("text", text, {TextEncoder::kMaxTokens, TextEncoder::kModelWidth});

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
  dump("clip_states", clip_states, {ClipVision::kTokens, ClipVision::kWidth});

  // 3. VAE condition: first frame, then zeros, encoded and normalized; and
  // the guide video for Uni3C.
  std::vector<float> condition, guide_latent;
  {
    auto start = Clock::now();
    auto encoder = VaeEncoder::Load(request.wan21_root / "vae", error);
    if (!encoder)
      return false;
    const std::size_t frame = static_cast<std::size_t>(w) * h * 3;
    std::vector<float> input(frame * n, 0.0F);
    for (std::size_t i = 0; i < frame; ++i)
      input[i] = static_cast<float>(request.first_frame[i]) / 127.5F - 1.0F;
    if (!encoder->EncodeVideo(input, n, h, w, true, &condition, error))
      return false;
    TrackDevice(&tel.peak_device_bytes);
    tel.condition_ms = Ms(start);
    if (!request.uni3c_file.empty()) {
      std::span<const std::uint8_t> guide = request.guide;
      int guide_frames = request.guide_frames;
      std::vector<std::uint8_t> rendered;
      if (guide.empty() && !request.moge_file.empty()) {
        if (!RenderCameraGuide(request, &rendered, &tel, error))
          return false;
        guide = rendered;
        guide_frames = n;
        if (callbacks.rendered_guide)
          callbacks.rendered_guide(rendered, n);
      }
      start = Clock::now();
      if (guide_frames < 1 || guide.size() != frame * guide_frames) {
        SetError(error, "Wan 2.1 I2V: guide video must be at the video size");
        return false;
      }
      input = GuideVideo(guide, guide_frames, n, h, w);
      if (!encoder->EncodeVideo(input, n, h, w, true, &guide_latent, error))
        return false;
      tel.guide_ms = Ms(start);
    }
  }
  dump("condition", condition,
       {16, static_cast<std::size_t>(lf), static_cast<std::size_t>(lh),
        static_cast<std::size_t>(lw)});
  if (!guide_latent.empty())
    dump("guide_latent", guide_latent,
         {16, static_cast<std::size_t>(lf), static_cast<std::size_t>(lh),
          static_cast<std::size_t>(lw)});

  // 4. Transformer (+ Uni3C) and LCM sampling.
  std::vector<float> latents;
  {
    auto start = Clock::now();
    auto dit = Transformer21::Load(request.dit_file, request.lora_file,
                                   request.lora_strength, error);
    if (!dit)
      return false;
    std::unique_ptr<Uni3C> uni3c;
    if (!request.uni3c_file.empty()) {
      uni3c = Uni3C::Load(request.uni3c_file, dit->stream(), error);
      if (!uni3c || !uni3c->SetGuide(guide_latent, lf, lh, lw, error))
        return false;
      uni3c->set_strength(request.uni3c_strength);
    }
    tel.dit_load_ms = Ms(start);
    start = Clock::now();
    if (!dit->PrepareContext(0, text, TextEncoder::kMaxTokens, clip_states,
                             nullptr, nullptr, error))
      return false;
    tel.context_ms = Ms(start);
    // Model input: 16 latent channels, 4 mask channels (1 on the first
    // latent frame), 16 condition channels.
    std::vector<float> input(36 * plane, 0.0F);
    const std::size_t frame_plane = static_cast<std::size_t>(lh) * lw;
    for (int c = 0; c < 4; ++c)
      std::fill_n(input.begin() + (16 + c) * plane, frame_plane, 1.0F);
    std::copy(condition.begin(), condition.end(), input.begin() + 20 * plane);
    latents = GaussianNoise(16 * plane, request.seed);
    dump("noise0", latents,
         {16, static_cast<std::size_t>(lf), static_cast<std::size_t>(lh),
          static_cast<std::size_t>(lw)});
    std::vector<double> sigmas;
    for (int i = 0; i < request.steps; ++i) {
      const double t = 1.0 - static_cast<double>(i) / request.steps;
      sigmas.push_back(request.shift * t / (1.0 + (request.shift - 1.0) * t));
    }
    sigmas.push_back(0.0);
    std::vector<float> velocity;
    for (int step = 0; step < request.steps; ++step) {
      const auto step_start = Clock::now();
      std::copy(latents.begin(), latents.end(), input.begin());
      Transformer21Input in;
      in.latent = input;
      in.frames = lf;
      in.height = lh;
      in.width = lw;
      in.timestep = static_cast<float>(1000.0 * sigmas[step]);
      in.hook = uni3c.get();
      if (!dit->Forward(in, &velocity, nullptr, error))
        return false;
      TrackDevice(&tel.peak_device_bytes);
      const std::string tag = "step" + std::to_string(step);
      dump(tag + "_input", input,
           {36, static_cast<std::size_t>(lf), static_cast<std::size_t>(lh),
            static_cast<std::size_t>(lw)});
      dump(tag + "_velocity", velocity,
           {16, static_cast<std::size_t>(lf), static_cast<std::size_t>(lh),
            static_cast<std::size_t>(lw)});
      // LCM: x0 = x - sigma * v; re-noise to the next sigma.
      const float sigma = static_cast<float>(sigmas[step]);
      const float next = static_cast<float>(sigmas[step + 1]);
      std::vector<float> fresh;
      if (next > 0.0F) {
        fresh = GaussianNoise(latents.size(), request.seed + 1 + step);
        dump(tag + "_fresh", fresh,
             {16, static_cast<std::size_t>(lf), static_cast<std::size_t>(lh),
              static_cast<std::size_t>(lw)});
      }
      for (std::size_t i = 0; i < latents.size(); ++i) {
        const float x0 = latents[i] - sigma * velocity[i];
        latents[i] = next > 0.0F ? (1.0F - next) * x0 + next * fresh[i] : x0;
      }
      tel.step_ms.push_back(Ms(step_start));
      progress("denoiser", step + 1, request.steps);
    }
  }
  dump("final", latents,
       {16, static_cast<std::size_t>(lf), static_cast<std::size_t>(lh),
        static_cast<std::size_t>(lw)});
  if (final_latents != nullptr)
    *final_latents = latents;

  // 5. VAE decode.
  if (video != nullptr) {
    const auto start = Clock::now();
    auto decoder = VaeDecoder::Load(request.wan21_root / "vae", error);
    if (!decoder ||
        !decoder->Decode(
            latents, lf, lh, lw, false, true, video,
            [&](int done, int total) { progress("video_vae", done, total); },
            nullptr, error))
      return false;
    TrackDevice(&tel.peak_device_bytes);
    tel.decode_ms = Ms(start);
  }
  tel.total_ms = Ms(total_start);
  return true;
}

}  // namespace gufo::models::wan22
