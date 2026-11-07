// Written with AI assistance (Claude Code).
// Ported from the 1.x fork (src/cli/serve/video_wan22.cpp, video_wan21.cpp,
// video_vace.cpp, video_talk.cpp). Behaviour unchanged; the fork's
// minimax_h3::GenerationRequest became wan22::JobRequest and the telemetry
// the 0.2 ext::VideoTelemetry (stage details in extra_json).
#include "src/models/wan22/runners.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <system_error>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/wan22/pipeline.hpp"
#include "src/models/wan22/pipeline21.hpp"
#include "src/models/wan22/pipeline_vace.hpp"
#include "src/models/wan22/talk21.hpp"
#include "src/models/wan22/video_io.hpp"

namespace gufo::models::wan22 {
namespace {

using Clock = std::chrono::steady_clock;

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

double Since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

bool ReadManifest(const JobRequest& job, const char* file,
                  json::Value* manifest, std::string* error) {
  std::ifstream input(job.model_root / file, std::ios::binary);
  std::stringstream text;
  text << input.rdbuf();
  try {
    *manifest = json::parse(text.str());
    return true;
  } catch (const std::exception& e) {
    SetError(error, std::string(file) + ": " + e.what());
    return false;
  }
}

// A part of a bundle: relative to the bundle directory, or absolute.
std::filesystem::path Part(const JobRequest& job, const json::Value& manifest,
                           const char* key) {
  const std::filesystem::path value = manifest.member_str(key);
  return value.empty() || value.is_absolute() ? value : job.model_root / value;
}

I2V21Callbacks Wan21Callbacks(const JobRequest& job,
                              const JobCallbacks& callbacks) {
  I2V21Callbacks out;
  out.progress = [&callbacks](const std::string& phase, int done, int total) {
    if (callbacks.progress)
      callbacks.progress(phase, done, total);
  };
  out.keep_going = [&callbacks] {
    return !callbacks.keep_going || callbacks.keep_going();
  };
  out.preview = job.preview;
  return out;
}

bool Cancelled(const JobCallbacks& callbacks, std::string* error) {
  if (callbacks.keep_going && !callbacks.keep_going()) {
    SetError(error, "cancelled");
    return true;
  }
  return false;
}

void Report(const JobCallbacks& callbacks, const char* phase, int done,
            int total) {
  if (callbacks.progress)
    callbacks.progress(phase, done, total);
}

// Binary PPM (P6, maxval 255) as written by the request parser for still
// masks.
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

std::vector<std::uint8_t> FitFirstFrame(const JobRequest& job) {
  return FitImage(job.first_frame.rgb, job.first_frame.width,
                  job.first_frame.height, job.width, job.height);
}

}  // namespace

bool ReadFittedVideo(const std::filesystem::path& path, int width, int height,
                     int max_frames, std::vector<std::uint8_t>* frames,
                     int* count, std::string* error) {
  video_io::VideoInfo info;
  if (!video_io::ProbeVideo(path, &info, error))
    return false;
  if (info.width < 16 || info.height < 16 || info.width > 4096 ||
      info.height > 4096) {
    SetError(error, "a video input has an unsupported size");
    return false;
  }
  video_io::FrameReader reader;
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
    const auto fitted =
        FitImage(source, info.width, info.height, width, height);
    if (fitted.size() != frame) {
      SetError(error, "cannot scale a video input");
      return false;
    }
    frames->insert(frames->end(), fitted.begin(), fitted.end());
    ++*count;
  }
  reader.Close();
  if (*count == 0) {
    SetError(error, "a video input has no frames");
    return false;
  }
  return true;
}

bool RunWan22Job(const JobRequest& job, const JobCallbacks& callbacks,
                 ext::VideoTelemetry* telemetry, std::string* error) {
  VideoRequest video_request;
  video_request.model_root = job.model_root;
  video_request.prompt = job.prompt;
  video_request.width = job.width;
  video_request.height = job.height;
  video_request.frames = job.frames;
  video_request.steps = job.steps;
  video_request.seed = job.seed;
  video_request.negative_prompt = job.negative_prompt;
  video_request.guidance = job.guidance;
  std::vector<std::uint8_t> first_frame;
  if (!job.first_frame.rgb.empty()) {
    // Scale to cover the video size, crop the centre.
    first_frame = FitFirstFrame(job);
    if (first_frame.empty()) {
      SetError(error, "invalid first frame");
      return false;
    }
    video_request.first_frame = first_frame;
  }
  VideoCallbacks pipeline;
  pipeline.progress = [&callbacks](std::string_view phase, int done,
                                   int total) {
    if (callbacks.progress)
      callbacks.progress(std::string(phase), done, total);
  };
  pipeline.preview = job.preview;
  pipeline.keep_going = [&callbacks] {
    return !callbacks.keep_going || callbacks.keep_going();
  };
  DecodedVideo video;
  VideoTelemetry stages;
  if (!GenerateVideo(video_request, pipeline, nullptr, &video, false, &stages,
                     error))
    return false;
  if (Cancelled(callbacks, error))
    return false;
  Report(callbacks, "media", 0, 1);
  const auto media_start = Clock::now();
  if (!WriteMp4(job.output_path, video, Fps(Family::kWan22), error))
    return false;
  Report(callbacks, "media", 1, 1);
  if (telemetry != nullptr) {
    telemetry->prompt_ms = stages.text_ms;
    telemetry->prompt_tokens = static_cast<std::size_t>(stages.prompt_tokens);
    telemetry->denoise_ms = stages.transformer_load_ms + stages.denoise_ms;
    telemetry->video_vae_ms = stages.vae_load_ms + stages.vae_ms;
    telemetry->media_ms = Since(media_start);
    telemetry->total_ms = stages.total_ms + telemetry->media_ms;
  }
  return true;
}

bool RunWan21Job(const JobRequest& request, const JobCallbacks& callbacks,
                 ext::VideoTelemetry* telemetry, std::string* error) {
  json::Value manifest;
  if (!ReadManifest(request, "gufo-camera.json", &manifest, error))
    return false;
  I2V21Request job;
  job.wan21_root = Part(request, manifest, "wan21");
  job.t5_root = Part(request, manifest, "t5");
  job.dit_file = Part(request, manifest, "dit");
  job.lora_file = Part(request, manifest, "lora");
  job.lora_strength =
      static_cast<float>(manifest.member_double("lora_strength", 1.0));
  job.shift = static_cast<float>(manifest.member_double("shift", 5.0));
  job.prompt = request.prompt;
  job.width = request.width;
  job.height = request.height;
  job.frames = request.frames;
  job.steps = request.steps;
  job.seed = request.seed;
  const std::vector<std::uint8_t> first_frame = FitFirstFrame(request);
  if (first_frame.empty()) {
    SetError(error, "invalid first frame");
    return false;
  }
  job.first_frame = first_frame;
  std::vector<std::uint8_t> guide;
  if (!request.guide_video_path.empty()) {
    // Uploaded guide: decoded by FFmpeg, each frame fitted to the video size
    // like the first frame (cover, centre crop).
    int count = 0;
    if (!ReadFittedVideo(request.guide_video_path, job.width, job.height,
                         job.frames, &guide, &count, error))
      return false;
    job.uni3c_file = Part(request, manifest, "uni3c");
    job.guide = guide;
    job.guide_frames = count;
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
    job.uni3c_file = Part(request, manifest, "uni3c");
    job.moge_file = Part(request, manifest, "moge");
  }
  job.uni3c_strength =
      static_cast<float>(manifest.member_double("uni3c_strength", 1.0));
  const I2V21Callbacks pipeline = Wan21Callbacks(request, callbacks);
  DecodedVideo video;
  I2V21Telemetry stages;
  if (!GenerateI2V21(job, pipeline, nullptr, &video, &stages, error))
    return false;
  if (Cancelled(callbacks, error))
    return false;
  Report(callbacks, "media", 0, 1);
  const auto media_start = Clock::now();
  if (!WriteMp4(request.output_path, video, Fps(Family::kWan21Uni3C), error))
    return false;
  Report(callbacks, "media", 1, 1);
  if (telemetry != nullptr) {
    double denoise = 0;
    for (double ms : stages.step_ms)
      denoise += ms;
    json::Value extra = json::Value::object();
    extra["keyframe_ms"] = stages.clip_ms + stages.condition_ms +
                           stages.guide_ms + stages.depth_ms +
                           stages.render_ms;
    extra["preview_ms"] = stages.preview_ms;
    extra["previews"] = stages.previews;
    telemetry->prompt_ms = stages.text_ms;
    telemetry->denoise_ms = stages.dit_load_ms + stages.context_ms + denoise;
    telemetry->video_vae_ms = stages.decode_ms;
    telemetry->media_ms = Since(media_start);
    telemetry->total_ms = stages.total_ms + telemetry->media_ms;
    extra["total_ms"] = telemetry->total_ms;
    telemetry->extra_json = extra.dump();
  }
  return true;
}

bool RunWan21VaceJob(const JobRequest& request, const JobCallbacks& callbacks,
                     ext::VideoTelemetry* telemetry, std::string* error) {
  json::Value manifest;
  json::Value options = json::Value::object();
  if (!ReadManifest(request, "gufo-vace.json", &manifest, error))
    return false;
  try {
    if (!request.vace_json.empty())
      options = json::parse(request.vace_json);
  } catch (const std::exception& e) {
    SetError(error, std::string("gufo.vace: ") + e.what());
    return false;
  }
  Vace21Request job;
  job.vae_root = Part(request, manifest, "vae");
  job.t5_root = Part(request, manifest, "t5");
  job.dit_file = Part(request, manifest, "dit");
  job.vace_file = Part(request, manifest, "vace");
  job.lora_file = Part(request, manifest, "lora");
  job.lora_strength =
      static_cast<float>(manifest.member_double("lora_strength", 1.0));
  if (const json::Value* lora = options.find("lora");
      lora != nullptr && lora->is_bool() && !lora->as_bool())
    job.lora_file.clear();
  job.shift = static_cast<float>(
      options.member_double("shift", manifest.member_double("shift", 5.0)));
  job.sampler =
      options.member_str("sampler", manifest.member_str("sampler", "lcm"));
  job.strength = static_cast<float>(options.member_double("strength", 1.0));
  job.prompt = request.prompt;
  job.negative_prompt = request.negative_prompt;
  job.guidance = request.guidance;
  job.width = request.width;
  job.height = request.height;
  job.frames = request.frames;
  job.steps = request.steps;
  job.seed = request.seed;
  const int w = job.width, h = job.height;
  const std::size_t frame_pixels = static_cast<std::size_t>(w) * h;

  Report(callbacks, "inputs", 0, 1);
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
      rgb = FitImage(still, mw, mh, w, h);
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
  for (const auto& image : request.references) {
    references.push_back(FitImage(image.rgb, image.width, image.height, w, h));
    if (references.back().size() != frame_pixels * 3) {
      SetError(error, "invalid VACE reference image");
      return false;
    }
  }
  for (const auto& reference : references)
    job.references.emplace_back(reference);
  Report(callbacks, "inputs", 1, 1);

  const I2V21Callbacks pipeline = Wan21Callbacks(request, callbacks);
  DecodedVideo video;
  Vace21Telemetry stages;
  if (!GenerateVace21(job, pipeline, nullptr, &video, &stages, error))
    return false;
  if (Cancelled(callbacks, error))
    return false;
  Report(callbacks, "media", 0, 1);
  const auto media_start = Clock::now();
  if (!WriteMp4(request.output_path, video, Fps(Family::kWan21Vace), error))
    return false;
  Report(callbacks, "media", 1, 1);
  if (telemetry != nullptr) {
    double denoise = 0;
    for (double ms : stages.step_ms)
      denoise += ms;
    json::Value extra = json::Value::object();
    extra["keyframe_ms"] = stages.condition_ms;
    extra["preview_ms"] = stages.preview_ms;
    extra["previews"] = stages.previews;
    telemetry->prompt_ms = stages.text_ms;
    telemetry->denoise_ms = stages.dit_load_ms + stages.context_ms + denoise;
    telemetry->video_vae_ms = stages.decode_ms;
    telemetry->media_ms = Since(media_start);
    telemetry->total_ms = stages.total_ms + telemetry->media_ms;
    extra["total_ms"] = telemetry->total_ms;
    telemetry->extra_json = extra.dump();
  }
  return true;
}

bool RunWan21TalkJob(const JobRequest& request, const JobCallbacks& callbacks,
                     ext::VideoTelemetry* telemetry, std::string* error) {
  json::Value manifest;
  json::Value options = json::Value::object();
  if (!ReadManifest(request, "gufo-talk.json", &manifest, error))
    return false;
  try {
    if (!request.talk_json.empty())
      options = json::parse(request.talk_json);
  } catch (const std::exception& e) {
    SetError(error, std::string("gufo.talk: ") + e.what());
    return false;
  }
  const int speakers = static_cast<int>(request.speakers.size());
  if (speakers < 1 || speakers > 2 ||
      (speakers == 2 && request.speaker_masks.size() != 2)) {
    SetError(error, "InfiniteTalk needs one or two speakers (two with masks)");
    return false;
  }
  Talk21Request job;
  job.wan21_root = Part(request, manifest, "wan21");
  job.t5_root = Part(request, manifest, "t5");
  job.dit_file = Part(request, manifest, "dit");
  job.lora_file = Part(request, manifest, "lora");
  job.lora_strength =
      static_cast<float>(manifest.member_double("lora_strength", 1.0));
  job.talk_file = speakers == 2 && !Part(request, manifest, "talk_multi").empty()
                      ? Part(request, manifest, "talk_multi")
                      : Part(request, manifest, "talk");
  job.wav2vec2_file = Part(request, manifest, "wav2vec2");
  job.shift = static_cast<float>(manifest.member_double("shift", 5.0));
  job.window = static_cast<int>(manifest.member_double("window", 81));
  job.motion_frames = static_cast<int>(options.member_double(
      "motion_frames", manifest.member_double("motion_frames", 9)));
  job.audio_scale =
      static_cast<float>(options.member_double("audio_scale", 1.0));
  job.audio_mode = options.member_str("audio_mode", "add");
  job.prompt = request.prompt;
  job.width = request.width;
  job.height = request.height;
  job.frames = request.frames;
  job.steps = request.steps;
  job.seed = request.seed;
  const std::vector<std::uint8_t> first_frame = FitFirstFrame(request);
  if (first_frame.empty()) {
    SetError(error, "invalid first frame");
    return false;
  }
  job.first_frame = first_frame;
  // Video to video: the input video ("gufo.talk.video") fitted to the video
  // size like the start image, frame for frame (no frame-rate conversion);
  // up to one window past the output, which the last window may reach into.
  job.denoise = static_cast<float>(options.member_double("denoise", 1.0));
  std::vector<std::uint8_t> input_video;
  if (job.denoise < 1.0F) {
    if (request.guide_video_path.empty()) {
      SetError(error, "'gufo.talk.denoise' below 1 needs 'gufo.talk.video'");
      return false;
    }
    if (!ReadFittedVideo(request.guide_video_path, job.width, job.height,
                         (job.frames > 0 ? job.frames : 1500) + job.window,
                         &input_video, &job.input_video_frames, error))
      return false;
    job.input_video = input_video;
  }
  for (int s = 0; s < speakers; ++s) {
    TalkSpeaker speaker;
    speaker.audio = TalkMono16k(request.speakers[s]);
    if (speakers == 2) {
      // The mask is fitted like the start image (cover, centre crop) so it
      // lines up with the picture; bright pixels are the speaker.
      const auto& image = request.speaker_masks[s];
      const auto fitted = FitImage(image.rgb, image.width, image.height,
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
    constexpr int kRate = kSpeechSampleRate;
    std::vector<std::vector<float>> tracks;
    std::vector<std::size_t> offsets;
    std::size_t length = 0;
    double start_frames = 0;
    for (int s = 0; s < speakers; ++s) {
      const auto& wave = request.speakers[s];
      if (wave.sample_rate != kRate || wave.channels != 2) {
        SetError(error, "speaker audio must be decoded to 32 kHz stereo");
        return false;
      }
      const std::size_t offset =
          job.audio_mode == "add"
              ? static_cast<std::size_t>(
                    std::lround(start_frames / 25.0 * kRate))
              : 0;
      offsets.push_back(offset);
      tracks.push_back(wave.pcm);
      length =
          std::max(length, offset + static_cast<std::size_t>(wave.samples));
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
  const I2V21Callbacks pipeline = Wan21Callbacks(request, callbacks);
  DecodedVideo video;
  Talk21Telemetry stages;
  std::error_code ignored;
  if (!GenerateTalk21(job, pipeline, &video, &stages, error)) {
    std::filesystem::remove(soundtrack, ignored);
    return false;
  }
  if (Cancelled(callbacks, error)) {
    std::filesystem::remove(soundtrack, ignored);
    return false;
  }
  Report(callbacks, "media", 0, 1);
  const auto media_start = Clock::now();
  const bool written = WriteMp4WithAudio(
      request.output_path, video, Fps(Family::kWan21Talk), soundtrack, error);
  std::filesystem::remove(soundtrack, ignored);
  if (!written)
    return false;
  Report(callbacks, "media", 1, 1);
  if (telemetry != nullptr) {
    double denoise = 0;
    for (double ms : stages.step_ms)
      denoise += ms;
    json::Value extra = json::Value::object();
    extra["keyframe_ms"] = stages.clip_ms + stages.condition_ms;
    extra["audio_guide_ms"] = stages.audio_ms;
    extra["audio_guides"] = speakers;
    extra["audio_guide_frames"] = stages.audio_frames;
    extra["preview_ms"] = stages.preview_ms;
    extra["previews"] = stages.previews;
    telemetry->prompt_ms = stages.text_ms;
    telemetry->denoise_ms =
        stages.dit_load_ms + stages.context_ms + stages.motion_ms + denoise;
    telemetry->video_vae_ms = stages.decode_ms;
    telemetry->media_ms = Since(media_start);
    telemetry->total_ms = stages.total_ms + telemetry->media_ms;
    extra["total_ms"] = telemetry->total_ms;
    telemetry->extra_json = extra.dump();
  }
  return true;
}

}  // namespace gufo::models::wan22
