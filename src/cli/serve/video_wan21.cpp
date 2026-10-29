#include "src/cli/serve/video_wan21.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/rife/video_io.hpp"
#include "src/models/wan22/pipeline.hpp"
#include "src/models/wan22/pipeline21.hpp"
#endif

namespace gufo::server {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

bool Number(const json::Value* value, double low, double high, double* out) {
  if (value == nullptr)
    return true;
  if (!value->is_number())
    return false;
  const double v = value->as_double(NAN);
  if (!std::isfinite(v) || v < low || v > high)
    return false;
  *out = v;
  return true;
}

}  // namespace

bool ParseCameraSpec(const json::Value& camera, models::wan22::CameraSpec* spec,
                     std::string* error) {
  if (!camera.is_object()) {
    SetError(error, "'gufo.camera' must be an object");
    return false;
  }
  *spec = {};
  if (const json::Value* poses = camera.find("poses")) {
    if (camera.contains("type") || !poses->is_array() || poses->empty() ||
        poses->size() > 81) {
      SetError(error,
               "'gufo.camera.poses' must be a non-empty array (at most 81) "
               "without 'type'");
      return false;
    }
    spec->type = "poses";
    for (const json::Value& pose : poses->items()) {
      if (!pose.is_object()) {
        SetError(error, "each camera pose must be an object");
        return false;
      }
      models::wan22::CameraKeyframe k;
      double frame = 0;
      if (!Number(pose.find("frame"), 0, 1000, &frame) ||
          std::floor(frame) != frame) {
        SetError(error, "camera pose 'frame' must be an integer 0..1000");
        return false;
      }
      k.frame = static_cast<int>(frame);
      if (const json::Value* position = pose.find("position")) {
        if (!position->is_array() || position->size() != 3) {
          SetError(error, "camera pose 'position' must be [x, y, z]");
          return false;
        }
        for (int j = 0; j < 3; ++j)
          if (!Number(&position->items()[j], -10, 10, &k.position[j])) {
            SetError(error,
                     "camera pose positions are scene depths in -10..10");
            return false;
          }
      }
      if (!Number(pose.find("yaw"), -180, 180, &k.yaw) ||
          !Number(pose.find("pitch"), -90, 90, &k.pitch) ||
          !Number(pose.find("roll"), -180, 180, &k.roll) ||
          !Number(pose.find("zoom"), 0.25, 4, &k.zoom)) {
        SetError(error, "camera pose yaw/pitch/roll are degrees, zoom 0.25..4");
        return false;
      }
      spec->keyframes.push_back(k);
    }
    std::sort(spec->keyframes.begin(), spec->keyframes.end(),
              [](const auto& a, const auto& b) { return a.frame < b.frame; });
    return true;
  }
  const json::Value* type = camera.find("type");
  if (type == nullptr || !type->is_string()) {
    SetError(error, "'gufo.camera' needs 'type', 'poses' or 'render_video'");
    return false;
  }
  spec->type = type->str();
  if (!Number(camera.find("amount"), 0, 360, &spec->amount) ||
      !Number(camera.find("speed"), 0.05, 4, &spec->speed)) {
    SetError(error, "'gufo.camera.amount' must be 0..360, 'speed' 0.05..4");
    return false;
  }
  if (const json::Value* ease = camera.find("ease")) {
    if (!ease->is_string()) {
      SetError(error, "'gufo.camera.ease' must be a string");
      return false;
    }
    spec->ease = ease->str();
  }
  // Validate the type and ease with a dry run.
  std::vector<models::wan22::CameraPose> poses;
  return models::wan22::BuildCameraPath(*spec, 5, 1.0, &poses, error);
}

bool RunWan21Job(const minimax_h3::GenerationRequest& request,
                 const minimax_h3::CancellationToken* cancellation,
                 minimax_h3::GenerationProgress progress, void* progress_opaque,
                 minimax_h3::GenerationTelemetry* telemetry,
                 std::string* error) {
#if defined(ENGINE_ENABLE_HIP)
  namespace wan22 = models::wan22;
  using Clock = std::chrono::steady_clock;
  // Camera bundle manifest.
  json::Value manifest;
  {
    std::ifstream input(request.model_root / "gufo-camera.json",
                        std::ios::binary);
    std::stringstream text;
    text << input.rdbuf();
    try {
      manifest = json::parse(text.str());
    } catch (const std::exception& e) {
      SetError(error, std::string("gufo-camera.json: ") + e.what());
      return false;
    }
  }
  const auto part = [&](const char* key) {
    const std::filesystem::path value = manifest.member_str(key);
    return value.empty() || value.is_absolute() ? value
                                                : request.model_root / value;
  };
  wan22::I2V21Request job;
  job.wan21_root = part("wan21");
  job.t5_root = part("t5");
  job.dit_file = part("dit");
  job.lora_file = part("lora");
  job.lora_strength =
      static_cast<float>(manifest.member_double("lora_strength", 1.0));
  job.shift = static_cast<float>(manifest.member_double("shift", 5.0));
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
  std::vector<std::uint8_t> guide;
  if (!request.guide_video_path.empty()) {
    // Uploaded guide: decoded by FFmpeg, each frame fitted to the video size
    // like the first frame (cover, centre crop).
    models::rife::VideoInfo info;
    if (!models::rife::ProbeVideo(request.guide_video_path, &info, error))
      return false;
    if (info.width < 16 || info.height < 16 || info.width > 4096 ||
        info.height > 4096) {
      SetError(error, "the render video has an unsupported size");
      return false;
    }
    models::rife::FrameReader reader;
    if (!reader.Open(request.guide_video_path, info.width, info.height, error))
      return false;
    const std::size_t frame =
        static_cast<std::size_t>(job.width) * job.height * 3;
    std::vector<std::uint8_t> source(static_cast<std::size_t>(info.width) *
                                     info.height * 3);
    for (int f = 0; f < job.frames; ++f) {
      bool end = false;
      if (!reader.Read(source.data(), &end, error)) {
        if (!end)
          return false;
        break;
      }
      const auto fitted = wan22::FitImage(source, info.width, info.height,
                                          job.width, job.height);
      if (fitted.size() != frame) {
        SetError(error, "cannot scale the render video");
        return false;
      }
      guide.insert(guide.end(), fitted.begin(), fitted.end());
    }
    reader.Close();
    if (guide.empty()) {
      SetError(error, "the render video has no frames");
      return false;
    }
    job.uni3c_file = part("uni3c");
    job.guide = guide;
    job.guide_frames = static_cast<int>(guide.size() / frame);
  } else if (!request.camera_json.empty()) {
    json::Value camera;
    try {
      camera = json::parse(request.camera_json);
    } catch (const std::exception& e) {
      SetError(error, std::string("camera: ") + e.what());
      return false;
    }
    if (!ParseCameraSpec(camera, &job.camera, error))
      return false;
    job.uni3c_file = part("uni3c");
    job.moge_file = part("moge");
  }
  job.uni3c_strength =
      static_cast<float>(manifest.member_double("uni3c_strength", 1.0));
  wan22::I2V21Callbacks callbacks;
  callbacks.progress = [&](const std::string& phase, int done, int total) {
    if (progress != nullptr)
      progress(phase, done, total, progress_opaque);
  };
  callbacks.keep_going = [cancellation] {
    return cancellation == nullptr || !cancellation->IsCancelled();
  };
  wan22::DecodedVideo video;
  wan22::I2V21Telemetry stages;
  if (!wan22::GenerateI2V21(job, callbacks, nullptr, &video, &stages, error))
    return false;
  if (cancellation != nullptr && cancellation->IsCancelled()) {
    SetError(error, "cancelled");
    return false;
  }
  callbacks.progress("media", 0, 1);
  const auto media_start = Clock::now();
  if (!wan22::WriteMp4(request.output_path, video, 16, error))
    return false;
  callbacks.progress("media", 1, 1);
  if (telemetry != nullptr) {
    telemetry->prompt_ms = stages.text_ms;
    telemetry->keyframe_ms = stages.clip_ms + stages.condition_ms +
                             stages.guide_ms + stages.depth_ms +
                             stages.render_ms;
    double denoise = 0;
    for (double ms : stages.step_ms)
      denoise += ms;
    telemetry->denoiser_ms = stages.dit_load_ms + stages.context_ms + denoise;
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
  SetError(error, "Wan 2.1 video generation requires a ROCm HIP-enabled build");
  return false;
#endif
}

}  // namespace gufo::server
