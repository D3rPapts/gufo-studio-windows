#ifndef GUFO_MODELS_MINIMAX_H3_SAMPLING_HPP_
#define GUFO_MODELS_MINIMAX_H3_SAMPLING_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::minimax_h3 {

inline constexpr int kH3CanvasMultiple = 32;
inline constexpr std::int64_t kH3MaximumPixels = 768LL * 1344LL;
inline constexpr int kH3FramesPerSecond = 24;
inline constexpr int kH3AudioLatentFramesPerSecond = 40;
inline constexpr int kH3VaeSpatialRatio = 16;
inline constexpr int kH3MaximumFrames = 345;
inline constexpr int kH3MaximumEvaluations = 1000;
// Keyframe anchors are held just short of clean for every step.
inline constexpr float kH3ConditionTime = 0.999F;
// Audio guides are held exactly clean (the reference's audio condition
// timestep is 1.0, without noise augmentation).
inline constexpr float kH3AudioConditionTime = 1.0F;
// Image guides (first, last, and intermediate keyframes) per request.
inline constexpr std::size_t kH3MaximumKeyframes = 8;
// Audio guides per request.
inline constexpr std::size_t kH3MaximumAudioGuides = 2;

struct TemporalShape {
  int frames{0};
  int video_latent_frames{0};
  int audio_latent_frames{0};
};

struct GenerationGeometry {
  int width{0};
  int height{0};
  int latent_width{0};
  int latent_height{0};
  TemporalShape temporal;
};

[[nodiscard]] int AlignFrameCount(int requested_frames) noexcept;
[[nodiscard]] int VideoLatentFrameCount(int frame_count) noexcept;
[[nodiscard]] int VideoEncoderLatentFrameCount(int frame_count) noexcept;
[[nodiscard]] std::optional<GenerationGeometry> ResolveGenerationGeometry(
    int width, int height, int requested_frames, std::string* error = nullptr);

struct SigmaSchedule {
  std::vector<float> video;
  std::vector<float> audio;

  [[nodiscard]] int evaluations() const noexcept {
    return video.empty() ? 0 : static_cast<int>(video.size()) - 1;
  }
};

[[nodiscard]] std::optional<SigmaSchedule> BuildServingSchedule(
    int evaluations, std::string* error = nullptr);

struct TimeRowSchedule {
  std::vector<std::uint32_t> video_rows;
  std::vector<std::uint32_t> audio_rows;
  std::uint32_t time_rows{0};
  // Time row of the keyframe conditioning rows (t = kH3ConditionTime), or
  // kNoConditionRow when the request has no keyframes.
  static constexpr std::uint32_t kNoConditionRow = 0xFFFFFFFFU;
  std::uint32_t condition_row{kNoConditionRow};
  // Time row of the audio guide rows (t = kH3AudioConditionTime), or
  // kNoConditionRow when the request has no audio guide.
  std::uint32_t audio_condition_row{kNoConditionRow};
};

[[nodiscard]] std::optional<TimeRowSchedule> BuildTimeRowSchedule(
    const SigmaSchedule& schedule, std::string* error = nullptr);
/// Appends the constant keyframe time row after the schedule's rows.
void AppendConditionTimeRow(TimeRowSchedule* rows) noexcept;
/// Appends the constant audio guide time row after the schedule's rows.
void AppendAudioConditionTimeRow(TimeRowSchedule* rows) noexcept;

struct Position3D {
  double temporal{0.0};
  double height{0.0};
  double width{0.0};
};

enum class SegmentKind : std::uint8_t {
  kText,
  kAudioTarget,
  kVideoTarget,
  kVideoCondition,
  kAudioCondition,
};

/// Which end of the generated video an FL2VA keyframe is anchored to.
enum class KeyframeAnchor : std::uint8_t {
  kFirst,
  kLast,
};

[[nodiscard]] std::string_view ToString(SegmentKind kind) noexcept;

struct LayoutSegment {
  std::size_t begin{0};
  std::size_t end{0};
  SegmentKind kind{SegmentKind::kText};
};

struct PackedLayout {
  std::vector<Position3D> positions;
  std::vector<LayoutSegment> segments;
  std::size_t text_rows{0};
  // Image guide rows (one latent frame per keyframe, 5k + 2 per guide
  // clip).
  std::size_t condition_rows{0};
  // Audio guide rows (two stereo tracks per audio latent frame).
  std::size_t audio_condition_rows{0};
  std::size_t audio_target_rows{0};
  std::size_t video_target_rows{0};
  int video_latent_frames{0};
  int latent_height{0};
  int latent_width{0};
  int audio_latent_frames{0};

  [[nodiscard]] std::size_t rows() const noexcept { return positions.size(); }
};

[[nodiscard]] std::optional<PackedLayout> BuildTextOnlyLayout(
    std::size_t text_rows, const GenerationGeometry& geometry,
    std::string* error = nullptr);
/// `[text | keyframe conditions | target audio | target video]`: one latent
/// frame of rows per anchor, in the given order, placed on the rotary time
/// of the first or the last generated frame as in the reference.
[[nodiscard]] std::optional<PackedLayout> BuildKeyframeLayout(
    std::size_t text_rows, const GenerationGeometry& geometry,
    std::span<const KeyframeAnchor> anchors, std::string* error = nullptr);

/// Pixel frame index of a first/last anchor in a video of `frames` frames.
[[nodiscard]] int KeyframeAnchorFrame(KeyframeAnchor anchor,
                                      int frames) noexcept;

/// An audio guide in the packed sequence: `latent_frames` audio latent
/// frames that start at pixel frame `frame` of the target video.
struct AudioGuidePlacement {
  int frame{0};
  int latent_frames{0};
};

/// Largest audio guide (in audio latent frames) that still fits behind pixel
/// frame `frame`, as in the reference (floor(audio_t - 5/3 * frame)); 0 when
/// the frame is past the end of the audio track.
[[nodiscard]] int MaximumAudioGuideFrames(const GenerationGeometry& geometry,
                                          int frame) noexcept;

/// `[text | image guides | audio guides | target audio | target video]` as in
/// the reference's MiniMaxH3AddGuide: every image guide is one latent frame
/// of rows on the rotary time `text + 5/3 * frame`, every audio guide's
/// stereo rows start on that time and advance one per latent frame. Rows of
/// a full attention sequence may be packed in any order, so the guides are
/// grouped by modality rather than interleaved per guide.
[[nodiscard]] std::optional<PackedLayout> BuildGuideLayout(
    std::size_t text_rows, const GenerationGeometry& geometry,
    std::span<const int> keyframe_frames,
    std::span<const AudioGuidePlacement> audio_guides,
    std::string* error = nullptr);

/// An image guide in the packed sequence: `latent_frames` VAE latent frames
/// whose first one sits on pixel frame `frame` of the target video. A single
/// image is one latent frame; a guide clip of 17k + 5 pixel frames (the
/// reference's MiniMaxH3AddGuide with a multi-frame image batch) is
/// 5k + 2 latent frames, laid out on the target's own temporal grid
/// (5/3 * {1, 4, 4, 4, 4} rotary units per latent frame) from that origin.
struct ImageGuidePlacement {
  int frame{0};
  int latent_frames{1};
  // Pixel frames the guide covers (1 for an image); frame + pixel_frames
  // must fit in the video.
  int pixel_frames{1};
};

/// Pixel frames MiniMaxH3AddGuide keeps of a `available`-frame image batch:
/// 1 below 5 frames (first image only), otherwise the largest 17k + 5 that
/// is not above `available`.
[[nodiscard]] int GuideClipPixelFrames(int available) noexcept;
/// VAE latent frames of a guide of `pixel_frames` (1 -> 1, 17k+5 -> 5k+2).
[[nodiscard]] int GuideClipLatentFrames(int pixel_frames) noexcept;

/// BuildGuideLayout with image guides that may span several latent frames.
[[nodiscard]] std::optional<PackedLayout> BuildImageGuideLayout(
    std::size_t text_rows, const GenerationGeometry& geometry,
    std::span<const ImageGuidePlacement> image_guides,
    std::span<const AudioGuidePlacement> audio_guides,
    std::string* error = nullptr);
[[nodiscard]] std::optional<std::vector<std::uint32_t>> BuildModulationRowMap(
    const TimeRowSchedule& time_rows, int step, const PackedLayout& layout,
    std::span<const std::uint8_t> text_tags = {}, std::string* error = nullptr);
[[nodiscard]] std::string PackedLayoutSha256(const PackedLayout& layout);
[[nodiscard]] std::string ModulationRowMapSha256(
    std::span<const std::uint32_t> rows);

[[nodiscard]] bool PatchifyVideo(std::span<const float> latent, int channels,
                                 int time, int height, int width,
                                 std::span<float> rows,
                                 std::string* error = nullptr);
[[nodiscard]] bool UnpatchifyVideo(std::span<const float> rows, int channels,
                                   int time, int height, int width,
                                   std::span<float> latent,
                                   std::string* error = nullptr);
[[nodiscard]] bool PackAudio(std::span<const float> latent, int channels,
                             int time, std::span<float> rows,
                             std::string* error = nullptr);
[[nodiscard]] bool UnpackAudio(std::span<const float> rows, int channels,
                               int time, std::span<float> latent,
                               std::string* error = nullptr);

class NormalRng {
public:
  explicit NormalRng(std::uint64_t seed) noexcept;

  [[nodiscard]] std::uint32_t NextU32() noexcept;
  [[nodiscard]] float NextNormal() noexcept;
  void Fill(std::span<float> output) noexcept;

private:
  std::uint64_t state_{0};
  std::uint64_t increment_{0};
  float spare_{0.0F};
  bool has_spare_{false};
};

struct InitialNoise {
  std::vector<float> video;
  std::vector<float> audio;
};

[[nodiscard]] std::optional<InitialNoise> BuildInitialNoise(
    std::uint64_t seed, std::size_t video_elements, int audio_channels,
    int audio_time, std::string* error = nullptr);

struct EulerStepPlan {
  bool evaluate{false};
  int last_evaluated{-1};
  int previous_evaluated{-1};
  float video_sigma_from_timestep{0.0F};
  float audio_sigma_from_timestep{0.0F};
  float video_ratio{0.0F};
  float audio_ratio{0.0F};
  float video_extrapolation{0.0F};
  float audio_extrapolation{0.0F};
};

[[nodiscard]] float VelocityExtrapolationRatio(float current_sigma,
                                               float last_sigma,
                                               float previous_sigma,
                                               bool have_previous) noexcept;
[[nodiscard]] std::optional<std::vector<std::uint8_t>> BuildReuseSelection(
    int steps, int reuse_interval, std::string* error = nullptr);
[[nodiscard]] std::optional<std::vector<EulerStepPlan>> BuildEulerPlan(
    const SigmaSchedule& schedule, int reuse_interval,
    std::string* error = nullptr);

// sample_f32, last_f32, and previous_f32 must point to device memory.
// stream is a hipStream_t passed opaquely to keep the public host header free
// from ROCm headers. The update remains in device F32 between DiT evaluations.
[[nodiscard]] bool HipEulerUpdate(
    void* sample_f32, std::size_t sample_elements, std::size_t sample_offset,
    const void* last_f32, const void* previous_f32,
    std::size_t velocity_elements, float sigma_from_timestep, float ratio,
    float extrapolation, void* stream, std::string* error = nullptr);

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_SAMPLING_HPP_
