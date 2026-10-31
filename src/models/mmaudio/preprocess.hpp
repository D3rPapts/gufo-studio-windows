// MMAudio for gufo: video frame selection and resizing for the CLIP and
// Synchformer inputs (CPU only, no device code).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MMAUDIO_PREPROCESS_HPP_
#define GUFO_MODELS_MMAUDIO_PREPROCESS_HPP_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::mmaudio {

inline constexpr double kClipFps = 8.0;
inline constexpr double kSyncFps = 25.0;
inline constexpr int kClipSize = 384;
inline constexpr int kSyncSize = 224;

/// PIL-style separable bicubic resampling (a = -0.5, antialiased when
/// shrinking, 8-bit rounding after each pass) of interleaved RGB8, which is
/// what torchvision's antialiased bicubic Resize does on uint8 images.
std::vector<std::uint8_t> ResizeBicubic(std::span<const std::uint8_t> rgb,
                                        int width, int height, int out_width,
                                        int out_height);

/// MMAudio's CLIP transform: squash to 384 x 384, scale to [0, 1], then
/// CLIP's mean / std -> planar [3][384][384].
std::vector<float> ClipPixels(std::span<const std::uint8_t> rgb, int width,
                              int height);
/// MMAudio's Synchformer transform: shorter side to 224 (torchvision
/// Resize(224)), centre crop 224, (x - 0.5) / 0.5 -> planar [3][224][224].
std::vector<float> SyncPixels(std::span<const std::uint8_t> rgb, int width,
                              int height);

/// Picks the CLIP (8 fps) and Synchformer (25 fps) frames of a decoded video
/// and preprocesses them on worker threads.
///   kResample   MMAudio's read_frames: a frame at time t = index / fps is
///               used (and repeated) while t >= the next sample time of each
///               rate; frames after `duration` are ignored.
///   kConsecutive ComfyUI-MMAudio's MMAudioSampler on an IMAGE batch: the
///               first int(8 * duration) frames feed CLIP and the first
///               int(25 * duration) the Synchformer, at the source rate.
/// Finish() shortens the duration when there were too few frames, exactly
/// like either original.
class VideoSampler {
public:
  enum class Mode { kResample, kConsecutive };

  struct Result {
    std::vector<float> clip;  // [clip_frames][3][384][384]
    std::vector<float> sync;  // [sync_frames][3][224][224]
    int clip_frames{0};
    int sync_frames{0};
    double duration{0};
  };

  VideoSampler(double duration, double source_fps, Mode mode);
  ~VideoSampler();
  VideoSampler(const VideoSampler&) = delete;
  VideoSampler& operator=(const VideoSampler&) = delete;

  /// Feeds the next decoded frame (RGB8). Returns false once no further
  /// frame is needed.
  bool Add(std::span<const std::uint8_t> rgb, int width, int height);
  Result Finish();

private:
  void Queue(std::shared_ptr<const std::vector<std::uint8_t>> frame,
             int width, int height, bool clip, int slot);

  double duration_;
  double source_fps_;
  Mode mode_;
  int index_{0};
  double next_[2]{0, 0};
  int clip_target_{0};
  int sync_target_{0};
  // Deques keep the slots in place while workers write them.
  std::deque<std::vector<float>> clip_;
  std::deque<std::vector<float>> sync_;
  std::vector<std::future<void>> pending_;
  unsigned workers_{1};
};

/// Seeded standard normal noise (SplitMix64 + Box-Muller); deterministic
/// across platforms, but not torch.randn's sequence.
std::vector<float> GaussianNoise(std::uint64_t seed, std::size_t count);

/// 16-bit PCM mono WAV of samples in [-1, 1].
std::string EncodeMonoWav(std::span<const float> samples, int sample_rate);

}  // namespace gufo::models::mmaudio

#endif  // GUFO_MODELS_MMAUDIO_PREPROCESS_HPP_
