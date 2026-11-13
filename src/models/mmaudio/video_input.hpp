// MMAudio for gufo: decoding a video / image file into FoleyRequest frames.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MMAUDIO_VIDEO_INPUT_HPP_
#define GUFO_MODELS_MMAUDIO_VIDEO_INPUT_HPP_

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

#include "src/models/mmaudio/pipeline.hpp"

namespace gufo::models::mmaudio {

/// How the frames reach the model.
struct FrameOptions {
  /// false: MMAudio's own resampling to 8 / 25 fps by time (default).
  /// true: ComfyUI-MMAudio's MMAudioSampler on a VHS_LoadVideo IMAGE batch:
  /// consecutive frames after the VHS options below.
  bool consecutive{false};
  // VHS_LoadVideo emulation (consecutive mode only).
  double force_rate{0};  // 0 = the source rate
  int skip_first_frames{0};
  int select_every_nth{1};
  int frame_load_cap{0};  // 0 = no cap
};

/// Decodes `path` with FFmpeg (rife::FrameReader), selects and preprocesses
/// the frames for `request->duration` and stores them in `request`; the
/// duration is shortened when the video is shorter (as the originals do).
bool LoadVideoFrames(const std::filesystem::path& path,
                     const FrameOptions& options, FoleyRequest* request,
                     std::string* error);

/// One still image (RGB8) as MMAudio's image_input.
void LoadImageFrame(std::span<const std::uint8_t> rgb, int width, int height,
                    FoleyRequest* request);

}  // namespace gufo::models::mmaudio

#endif  // GUFO_MODELS_MMAUDIO_VIDEO_INPUT_HPP_
