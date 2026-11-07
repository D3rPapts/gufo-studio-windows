// Written with AI assistance (Claude Code).
#include "src/models/wan22/pipeline_vace.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <utility>

#include "src/models/wan22/pipeline.hpp"
#include "src/models/wan22/text_encoder.hpp"
#include "src/models/wan22/tokenizer.hpp"
#include "src/models/wan22/transformer21.hpp"
#include "src/models/wan22/vace.hpp"
#include "src/models/wan22/vace_context.hpp"

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

}  // namespace

bool GenerateVace21(const Vace21Request& request,
                    const I2V21Callbacks& callbacks,
                    std::vector<float>* final_latents, DecodedVideo* video,
                    Vace21Telemetry* telemetry, std::string* error) {
  Vace21Telemetry local;
  Vace21Telemetry& tel = telemetry != nullptr ? *telemetry : local;
  const auto total_start = Clock::now();
  const int w = request.width, h = request.height, n = request.frames;
  const std::size_t frame_pixels = static_cast<std::size_t>(w) * h;
  if (w % 16 != 0 || h % 16 != 0 || w < 16 || h < 16 || n < 1 ||
      (n - 1) % 4 != 0 || request.steps < 1 ||
      (request.sampler != "lcm" && request.sampler != "euler") ||
      !(request.guidance >= 1.0F) || request.control_frames < 0 ||
      request.control.size() != frame_pixels * 3 * request.control_frames ||
      request.mask_frames < 0 ||
      request.mask.size() != frame_pixels * request.mask_frames) {
    SetError(error,
             "Wan 2.1 VACE: width/height must be multiples of 16, frames "
             "4k+1, sampler lcm or euler, guidance >= 1, and the control "
             "video and mask must be at the video size");
    return false;
  }
  for (const auto& reference : request.references)
    if (reference.size() != frame_pixels * 3) {
      SetError(error, "Wan 2.1 VACE: reference images must be at the video "
                      "size");
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
  const auto cancelled = [&] {
    if (callbacks.keep_going && !callbacks.keep_going()) {
      SetError(error, "cancelled");
      return true;
    }
    return false;
  };
  const bool cfg = request.guidance > 1.0F;
  const int refs = static_cast<int>(request.references.size());
  const int lf = 1 + (n - 1) / 4, lh = h / 8, lw = w / 8;
  const int frames = refs + lf;  // latent frames incl. references
  const std::size_t plane = static_cast<std::size_t>(lh) * lw;
  const auto shape16 = [&](int f) {
    return std::vector<std::size_t>{16, static_cast<std::size_t>(f),
                                    static_cast<std::size_t>(lh),
                                    static_cast<std::size_t>(lw)};
  };

  // 1. Prompts (umT5, 512 rows: the states of the real tokens, then zeros).
  std::vector<float> text, negative;
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
    const auto encode = [&](const std::string& prompt,
                            std::vector<float>* out) {
      std::vector<float> states;
      if (!encoder->Encode(tokenizer->Encode(CleanPrompt(prompt)), &states,
                           error))
        return false;
      out->assign(TextEncoder::kMaxTokens * TextEncoder::kModelWidth, 0.0F);
      std::copy_n(states.begin(), std::min(states.size(), out->size()),
                  out->begin());
      return true;
    };
    if (!encode(request.prompt, &text) ||
        (cfg && !encode(request.negative_prompt, &negative)))
      return false;
    tel.text_ms = Ms(start);
    progress("prompt", 1, 1);
  }
  dump("text", text, {TextEncoder::kMaxTokens, TextEncoder::kModelWidth});
  if (cfg)
    dump("negative_text", negative,
         {TextEncoder::kMaxTokens, TextEncoder::kModelWidth});
  if (cancelled())
    return false;

  // 2. VACE context: control video and mask (ComfyUI pads the video with
  // grey and the mask with ones), split, VAE-encoded; reference latents.
  std::vector<float> context;
  {
    auto start = Clock::now();
    std::vector<float> mask;
    if (request.mask_frames > 0) {
      mask.assign(frame_pixels * n, 1.0F);
      const std::size_t take =
          std::min(request.mask_frames, n) * frame_pixels;
      std::copy_n(request.mask.begin(), take, mask.begin());
    }
    std::vector<float> inactive_rgb, reactive_rgb;
    SplitVaceControl(request.control, std::min(request.control_frames, n),
                     mask, n, h, w, &inactive_rgb, &reactive_rgb);
    auto encoder = VaeEncoder::Load(request.vae_root / "vae", error);
    if (!encoder)
      return false;
    std::vector<float> inactive, reactive;
    if (!encoder->EncodeVideo(inactive_rgb, n, h, w, true, &inactive, error))
      return false;
    std::vector<float>().swap(inactive_rgb);
    if (!encoder->EncodeVideo(reactive_rgb, n, h, w, true, &reactive, error))
      return false;
    std::vector<float>().swap(reactive_rgb);
    std::vector<std::vector<float>> reference_latents(refs);
    for (int r = 0; r < refs; ++r) {
      std::vector<float> rgb(frame_pixels * 3);
      for (std::size_t i = 0; i < rgb.size(); ++i)
        rgb[i] = static_cast<float>(request.references[r][i]) / 127.5F - 1.0F;
      if (!encoder->EncodeVideo(rgb, 1, h, w, true, &reference_latents[r],
                                error))
        return false;
    }
    TrackDevice(&tel.peak_device_bytes);
    VaceContextInputs inputs;
    inputs.frames = n;
    inputs.height = h;
    inputs.width = w;
    inputs.inactive = inactive;
    inputs.reactive = reactive;
    inputs.mask = mask;
    for (const auto& latent : reference_latents)
      inputs.references.emplace_back(latent);
    context = BuildVaceContext(inputs);
    if (context.empty()) {
      SetError(error, "Wan 2.1 VACE: cannot build the VACE context");
      return false;
    }
    tel.condition_ms = Ms(start);
    progress("condition", 1, 1);
  }
  dump("vace_context", context,
       {static_cast<std::size_t>(Vace::kContextChannels),
        static_cast<std::size_t>(frames), static_cast<std::size_t>(lh),
        static_cast<std::size_t>(lw)});
  if (cancelled())
    return false;

  // 3. Transformer (T2V 14B) + VACE blocks and sampling.
  std::vector<float> latents;
  {
    auto start = Clock::now();
    auto dit = Transformer21::Load(request.dit_file, request.lora_file,
                                   request.lora_strength, error);
    if (!dit)
      return false;
    if (!dit->text_only()) {
      SetError(error, "Wan 2.1 VACE needs the text-to-video 14B checkpoint, "
                      "not " + request.dit_file.string());
      return false;
    }
    auto vace = Vace::Load(request.vace_file, dit->stream(), error);
    if (!vace || !vace->SetContext(context, frames, lh, lw, error))
      return false;
    std::vector<float>().swap(context);
    vace->set_strength(request.strength);
    tel.dit_load_ms = Ms(start);
    TrackDevice(&tel.peak_device_bytes);
    start = Clock::now();
    if (!dit->PrepareContext(0, text, TextEncoder::kMaxTokens, {}, nullptr,
                             nullptr, error) ||
        (cfg && !dit->PrepareContext(1, negative, TextEncoder::kMaxTokens, {},
                                     nullptr, nullptr, error)))
      return false;
    tel.context_ms = Ms(start);
    latents = GaussianNoise(16 * frames * plane, request.seed);
    dump("noise0", latents, shape16(frames));
    std::vector<double> sigmas;
    for (int i = 0; i < request.steps; ++i) {
      const double t = 1.0 - static_cast<double>(i) / request.steps;
      sigmas.push_back(request.shift * t / (1.0 + (request.shift - 1.0) * t));
    }
    sigmas.push_back(0.0);
    std::vector<float> velocity, uncond;
    for (int step = 0; step < request.steps; ++step) {
      if (cancelled())
        return false;
      const auto step_start = Clock::now();
      Transformer21Input in;
      in.latent = latents;
      in.frames = frames;
      in.height = lh;
      in.width = lw;
      in.timestep = static_cast<float>(1000.0 * sigmas[step]);
      in.hook = vace.get();
      in.context_slot = 0;
      if (!dit->Forward(in, &velocity, nullptr, error))
        return false;
      if (cfg) {
        in.context_slot = 1;
        if (!dit->Forward(in, &uncond, nullptr, error))
          return false;
        for (std::size_t i = 0; i < velocity.size(); ++i)
          velocity[i] = uncond[i] + request.guidance * (velocity[i] - uncond[i]);
      }
      TrackDevice(&tel.peak_device_bytes);
      dump("step" + std::to_string(step) + "_velocity", velocity,
           shape16(frames));
      const float sigma = static_cast<float>(sigmas[step]);
      const float next = static_cast<float>(sigmas[step + 1]);
      if (callbacks.preview.Wants(step + 1, request.steps)) {
        // x0 = x - sigma * v of the video frames (references left out).
        const auto preview_start = Clock::now();
        std::vector<float> x0(16 * lf * plane);
        for (int c = 0; c < 16; ++c)
          for (std::size_t i = 0; i < lf * plane; ++i) {
            const std::size_t at = (c * frames + refs) * plane + i;
            x0[c * lf * plane + i] = latents[at] - sigma * velocity[at];
          }
        callbacks.preview.publish({.step = step + 1,
                                   .steps = request.steps,
                                   .frames = lf,
                                   .height = lh,
                                   .width = lw,
                                   .channels = 16,
                                   .channel_major = true,
                                   .data = x0});
        tel.preview_ms += Ms(preview_start);
        ++tel.previews;
      }
      if (request.sampler == "euler") {
        for (std::size_t i = 0; i < latents.size(); ++i)
          latents[i] += (next - sigma) * velocity[i];
      } else {
        std::vector<float> fresh;
        if (next > 0.0F)
          fresh = GaussianNoise(latents.size(), request.seed + 1 + step);
        for (std::size_t i = 0; i < latents.size(); ++i) {
          const float x0 = latents[i] - sigma * velocity[i];
          latents[i] = next > 0.0F ? (1.0F - next) * x0 + next * fresh[i] : x0;
        }
      }
      tel.step_ms.push_back(Ms(step_start));
      progress("denoiser", step + 1, request.steps);
    }
  }
  // TrimVideoLatent: drop the reference frames.
  if (refs > 0) {
    std::vector<float> trimmed(16 * lf * plane);
    for (int c = 0; c < 16; ++c)
      std::copy_n(latents.begin() + (c * frames + refs) * plane, lf * plane,
                  trimmed.begin() + c * lf * plane);
    latents = std::move(trimmed);
  }
  dump("final", latents, shape16(lf));
  if (final_latents != nullptr)
    *final_latents = latents;

  // 4. VAE decode.
  if (cancelled())
    return false;
  if (video != nullptr) {
    const auto start = Clock::now();
    auto decoder = VaeDecoder::Load(request.vae_root / "vae", error);
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
