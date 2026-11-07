// MMAudio for gufo: decoding a video / image file into FoleyRequest frames
// (see video_input.hpp).
// Written with AI assistance (Claude Code).
#include "src/models/mmaudio/video_input.hpp"

#include <algorithm>
#include <vector>

#include "src/models/mmaudio/preprocess.hpp"
#include "src/models/rife/video_io.hpp"

namespace gufo::models::mmaudio {

bool LoadVideoFrames(const std::filesystem::path& path,
                     const FrameOptions& options, FoleyRequest* request,
                     std::string* error) {
  rife::VideoInfo info;
  if (!rife::ProbeVideo(path, &info, error))
    return false;
  if (info.width < 16 || info.height < 16 || info.width > 8192 ||
      info.height > 8192 || info.fps_num <= 0 || info.fps_den <= 0)
    return Fail(error, "the video has an unsupported size or frame rate");
  const double source_fps =
      static_cast<double>(info.fps_num) / static_cast<double>(info.fps_den);
  rife::FrameReader reader;
  if (!reader.Open(path, info.width, info.height, error))
    return false;
  VideoSampler sampler(request->duration, source_fps,
                       options.consecutive ? VideoSampler::Mode::kConsecutive
                                           : VideoSampler::Mode::kResample);
  std::vector<std::uint8_t> frame(static_cast<std::size_t>(info.width) *
                                  info.height * 3);
  // VHS_LoadVideo: force_rate resampling (a source frame repeats or drops by
  // time), then skip_first_frames, select_every_nth, frame_load_cap.
  const double rate = options.consecutive && options.force_rate > 0
                          ? options.force_rate
                          : 0.0;
  double next_time = 0;
  long long emitted = 0, kept = 0;
  const int nth = std::max(1, options.select_every_nth);
  bool more = true;
  for (int index = 0; more; ++index) {
    bool end = false;
    if (!reader.Read(frame.data(), &end, error)) {
      if (!end)
        return false;
      break;
    }
    int copies = 1;
    if (rate > 0) {
      copies = 0;
      const double time = index / source_fps;
      while (time >= next_time) {
        ++copies;
        next_time += 1.0 / rate;
      }
    }
    for (int c = 0; c < copies && more; ++c) {
      const long long position = emitted++;
      if (options.consecutive) {
        if (position < options.skip_first_frames)
          continue;
        if ((position - options.skip_first_frames) % nth != 0)
          continue;
        if (options.frame_load_cap > 0 && kept >= options.frame_load_cap) {
          more = false;
          break;
        }
        ++kept;
      }
      more = sampler.Add(frame, info.width, info.height);
    }
  }
  reader.Close();
  VideoSampler::Result result = sampler.Finish();
  if (result.clip_frames < 1 || result.sync_frames < 16)
    return Fail(error, "the video is too short for MMAudio (needs 16 frames "
                       "at 25 fps)");
  request->duration = result.duration;
  request->clip_pixels = std::move(result.clip);
  request->sync_pixels = std::move(result.sync);
  request->clip_frames = result.clip_frames;
  request->sync_frames = result.sync_frames;
  request->image_input = false;
  return true;
}

void LoadImageFrame(std::span<const std::uint8_t> rgb, int width, int height,
                    FoleyRequest* request) {
  request->clip_pixels = ClipPixels(rgb, width, height);
  request->clip_frames = 1;
  request->sync_pixels.clear();
  request->sync_frames = 0;
  request->image_input = true;
}

}  // namespace gufo::models::mmaudio
