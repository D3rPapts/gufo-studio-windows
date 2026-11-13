// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_WAN22_RUNNERS_HPP_
#define GUFO_MODELS_WAN22_RUNNERS_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <gufo/ext/video.hpp>
#include <string>
#include <vector>

#include "src/models/wan22/request.hpp"

// HIP job runners of the Wan family (ported from the 1.x fork's
// src/cli/serve/video_wan22.cpp, video_wan21.cpp, video_vace.cpp and
// video_talk.cpp). Each writes an H.264 MP4 to job.output_path.
namespace gufo::models::wan22 {

struct JobCallbacks {
  std::function<void(const std::string& phase, int done, int total)> progress;
  std::function<bool()> keep_going;
};

/// Wan 2.2 TI2V-5B (24 fps): text- or image-to-video with negative prompt
/// and CFG. Loads every stage per job.
[[nodiscard]] bool RunWan22Job(const JobRequest& job,
                               const JobCallbacks& callbacks,
                               ext::VideoTelemetry* telemetry,
                               std::string* error);

/// Wan 2.1 I2V 14B + Uni3C (16 fps). The bundle's gufo-camera.json names
/// the parts: {"wan21", "t5", "dit", "lora", "lora_strength", "uni3c",
/// "moge", "uni3c_strength", "shift"}. A camera path (rendered from the first
/// frame with MoGe-2) or an uploaded render video turn Uni3C on; neither is
/// plain image-to-video.
[[nodiscard]] bool RunWan21Job(const JobRequest& job,
                               const JobCallbacks& callbacks,
                               ext::VideoTelemetry* telemetry,
                               std::string* error);

/// Wan 2.1 VACE 14B (16 fps). gufo-vace.json: {"vae", "t5", "dit", "vace",
/// "lora", "lora_strength", "shift", "sampler", "steps"}.
[[nodiscard]] bool RunWan21VaceJob(const JobRequest& job,
                                   const JobCallbacks& callbacks,
                                   ext::VideoTelemetry* telemetry,
                                   std::string* error);

/// Wan 2.1 I2V 14B + InfiniteTalk (25 fps, up to 60 s in windows of 81
/// frames continued from the last 9). gufo-talk.json: {"wan21", "t5", "dit",
/// "lora", "lora_strength", "talk", "talk_multi", "wav2vec2", "shift",
/// "window", "motion_frames"}. The MP4 carries the speakers' audio as the
/// video uses it.
[[nodiscard]] bool RunWan21TalkJob(const JobRequest& job,
                                   const JobCallbacks& callbacks,
                                   ext::VideoTelemetry* telemetry,
                                   std::string* error);

/// Up to `max_frames` frames of an MP4, each fitted to width x height like
/// the first frame (cover, centre crop), RGB8 one after another; `count` is
/// how many were read (at least one).
[[nodiscard]] bool ReadFittedVideo(const std::filesystem::path& path,
                                   int width, int height, int max_frames,
                                   std::vector<std::uint8_t>* frames,
                                   int* count, std::string* error);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_RUNNERS_HPP_
