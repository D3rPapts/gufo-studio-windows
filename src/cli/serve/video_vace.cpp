// Written with AI assistance (Claude Code).
#include "src/cli/serve/video_vace.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>

#include "src/core/json.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/rife/video_io.hpp"
#include "src/models/wan22/pipeline.hpp"
#include "src/models/wan22/pipeline_vace.hpp"
#endif

namespace gufo::server {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

#if defined(ENGINE_ENABLE_HIP)
// Up to `max_frames` frames of an MP4, each fitted to width x height like the
// first frame of the other Wan runners (cover, centre crop).
bool ReadFittedVideo(const std::filesystem::path& path, int width, int height,
                     int max_frames, std::vector<std::uint8_t>* frames,
                     int* count, std::string* error) {
  models::rife::VideoInfo info;
  if (!models::rife::ProbeVideo(path, &info, error))
    return false;
  if (info.width < 16 || info.height < 16 || info.width > 4096 ||
      info.height > 4096) {
    SetError(error, "a VACE video has an unsupported size");
    return false;
  }
  models::rife::FrameReader reader;
  if (!reader.Open(path, info.width, info.height, error))
    return false;
  const std::size_t frame = static_cast<std::size_t>(width) * height * 3;
  std::vector<std::uint8_t> source(static_cast<std::size_t>(info.width) *
                                   info.height * 3);
  frames->clear();
  *count = 0;
  for (int f = 0; f < max_frames; ++f) {
    bool end = false;
    if (!reader.Read(source.data(), &end, error)) {
      if (!end)
        return false;
      break;
    }
    const auto fitted = models::wan22::FitImage(source, info.width,
                                                info.height, width, height);
    if (fitted.size() != frame) {
      SetError(error, "cannot scale a VACE video");
      return false;
    }
    frames->insert(frames->end(), fitted.begin(), fitted.end());
    ++*count;
  }
  reader.Close();
  if (*count == 0) {
    SetError(error, "a VACE video has no frames");
    return false;
  }
  return true;
}

// Binary PPM (P6, maxval 255) as written by the video API for still masks.
bool ReadPpm(const std::filesystem::path& path, std::vector<std::uint8_t>* rgb,
             int* width, int* height, std::string* error) {
  std::ifstream input(path, std::ios::binary);
  std::string magic;
  int maxval = 0;
  input >> magic >> *width >> *height >> maxval;
  input.get();
  if (!input || magic != "P6" || maxval != 255 || *width < 1 ||
      *height < 1 || *width > 8192 || *height > 8192) {
    SetError(error, "invalid VACE mask image");
    return false;
  }
  rgb->resize(static_cast<std::size_t>(*width) * *height * 3);
  input.read(reinterpret_cast<char*>(rgb->data()),
             static_cast<std::streamsize>(rgb->size()));
  if (!input) {
    SetError(error, "short VACE mask image");
    return false;
  }
  return true;
}
#endif

}  // namespace

bool RunWan21VaceJob(const minimax_h3::GenerationRequest& request,
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
  {
    std::ifstream input(request.model_root / "gufo-vace.json",
                        std::ios::binary);
    std::stringstream text;
    text << input.rdbuf();
    try {
      manifest = json::parse(text.str());
      if (!request.vace_json.empty())
        options = json::parse(request.vace_json);
    } catch (const std::exception& e) {
      SetError(error, std::string("gufo-vace.json / gufo.vace: ") + e.what());
      return false;
    }
  }
  const auto part = [&](const char* key) {
    const std::filesystem::path value = manifest.member_str(key);
    return value.empty() || value.is_absolute() ? value
                                                : request.model_root / value;
  };
  wan22::Vace21Request job;
  job.vae_root = part("vae");
  job.t5_root = part("t5");
  job.dit_file = part("dit");
  job.vace_file = part("vace");
  job.lora_file = part("lora");
  job.lora_strength =
      static_cast<float>(manifest.member_double("lora_strength", 1.0));
  if (const json::Value* lora = options.find("lora");
      lora != nullptr && lora->is_bool() && !lora->as_bool())
    job.lora_file.clear();
  job.shift = static_cast<float>(options.member_double(
      "shift", manifest.member_double("shift", 5.0)));
  job.sampler = options.member_str("sampler",
                                   manifest.member_str("sampler", "lcm"));
  job.strength = static_cast<float>(options.member_double("strength", 1.0));
  job.prompt = request.prompt;
  job.negative_prompt = request.negative_prompt;
  job.guidance = request.guidance;
  job.width = request.parameters.output_width;
  job.height = request.parameters.output_height;
  job.frames = request.parameters.frames;
  job.steps = request.parameters.evaluations;
  job.seed = request.seed;
  const int w = job.width, h = job.height;
  const std::size_t frame_pixels = static_cast<std::size_t>(w) * h;

  const auto report = [&](const std::string& phase, int done, int total) {
    if (progress != nullptr)
      progress(phase, done, total, progress_opaque);
  };
  report("inputs", 0, 1);
  std::vector<std::uint8_t> control;
  if (!request.guide_video_path.empty()) {
    if (!ReadFittedVideo(request.guide_video_path, w, h, job.frames, &control,
                         &job.control_frames, error))
      return false;
    job.control = control;
  }
  std::vector<float> mask;
  if (!request.mask_path.empty()) {
    const std::string channel = options.member_str("mask_channel", "red");
    const json::Value* invert_value = options.find("invert_mask");
    const bool invert = invert_value != nullptr && invert_value->is_bool() &&
                        invert_value->as_bool();
    const auto value = [&](const std::uint8_t* rgb) {
      float v;
      if (channel == "green")
        v = rgb[1];
      else if (channel == "blue")
        v = rgb[2];
      else if (channel == "luma")
        v = 0.299F * rgb[0] + 0.587F * rgb[1] + 0.114F * rgb[2];
      else
        v = rgb[0];
      v /= 255.0F;
      return invert ? 1.0F - v : v;
    };
    std::vector<std::uint8_t> rgb;
    int count = 0;
    if (request.mask_path.extension() == ".ppm") {
      int mw = 0, mh = 0;
      std::vector<std::uint8_t> still;
      if (!ReadPpm(request.mask_path, &still, &mw, &mh, error))
        return false;
      rgb = wan22::FitImage(still, mw, mh, w, h);
      if (rgb.size() != frame_pixels * 3) {
        SetError(error, "cannot scale the VACE mask image");
        return false;
      }
      // One still for every frame.
      mask.resize(frame_pixels * job.frames);
      for (std::size_t p = 0; p < frame_pixels; ++p)
        mask[p] = value(rgb.data() + 3 * p);
      for (int f = 1; f < job.frames; ++f)
        std::copy_n(mask.begin(), frame_pixels,
                    mask.begin() + f * frame_pixels);
      job.mask_frames = job.frames;
    } else {
      if (!ReadFittedVideo(request.mask_path, w, h, job.frames, &rgb, &count,
                           error))
        return false;
      mask.resize(frame_pixels * count);
      for (std::size_t p = 0; p < mask.size(); ++p)
        mask[p] = value(rgb.data() + 3 * p);
      job.mask_frames = count;
    }
    job.mask = mask;
  }
  std::vector<std::vector<std::uint8_t>> references;
  for (const auto& keyframe : request.keyframes) {
    references.push_back(wan22::FitImage(keyframe.rgb, keyframe.width,
                                         keyframe.height, w, h));
    if (references.back().size() != frame_pixels * 3) {
      SetError(error, "invalid VACE reference image");
      return false;
    }
  }
  for (const auto& reference : references)
    job.references.emplace_back(reference);
  report("inputs", 1, 1);

  wan22::I2V21Callbacks callbacks;
  callbacks.progress = report;
  callbacks.keep_going = [cancellation] {
    return cancellation == nullptr || !cancellation->IsCancelled();
  };
  callbacks.preview = request.preview;
  wan22::DecodedVideo video;
  wan22::Vace21Telemetry stages;
  if (!wan22::GenerateVace21(job, callbacks, nullptr, &video, &stages, error))
    return false;
  if (cancellation != nullptr && cancellation->IsCancelled()) {
    SetError(error, "cancelled");
    return false;
  }
  report("media", 0, 1);
  const auto media_start = Clock::now();
  if (!wan22::WriteMp4(request.output_path, video, 16, error))
    return false;
  report("media", 1, 1);
  if (telemetry != nullptr) {
    telemetry->prompt_ms = stages.text_ms;
    telemetry->keyframe_ms = stages.condition_ms;
    double denoise = 0;
    for (double ms : stages.step_ms)
      denoise += ms;
    telemetry->denoiser_ms = stages.dit_load_ms + stages.context_ms + denoise;
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
  SetError(error, "Wan 2.1 VACE video generation requires a ROCm "
                  "HIP-enabled build");
  return false;
#endif
}

}  // namespace gufo::server
