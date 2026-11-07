// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_WAN22_REQUEST_HPP_
#define GUFO_MODELS_WAN22_REQUEST_HPP_

#include <cstdint>
#include <filesystem>
#include <gufo/ext/video.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/wan22/camera_path.hpp"
#include "src/models/wan22/preview_hook.hpp"
#include "src/models/wan22/talk_audio.hpp"

// The Wan video family as gufo video providers (capability video_generation):
// request parsing, validation and the runner's decoded job. CPU only; the
// HIP runners live in runners.cpp. Request formats are the 1.x fork's
// /v1/videos formats unchanged (the router's ComfyUI adapter sends them).
namespace gufo::models::wan22 {

enum class Family : std::uint8_t {
  kWan22,       // Wan 2.2 TI2V 5B, 24 fps
  kWan21Uni3C,  // Wan 2.1 I2V 14B + lightx2v + Uni3C camera, 16 fps
  kWan21Vace,   // Wan 2.1 T2V 14B + VACE, 16 fps
  kWan21Talk,   // Wan 2.1 I2V 14B + InfiniteTalk, 25 fps
};

inline constexpr std::string_view kWan22ModelId = "wan2.2-ti2v-5b";
inline constexpr std::string_view kWan22Preset = "wan22";
inline constexpr std::string_view kWan21ModelId = "wan2.1-uni3c";
inline constexpr std::string_view kWan21Preset = "wan21";
inline constexpr std::string_view kWan21VaceModelId = "wan2.1-vace-14b";
inline constexpr std::string_view kWan21VacePreset = "wan21-vace";
inline constexpr std::string_view kWan21TalkModelId = "wan2.1-infinitetalk-14b";
inline constexpr std::string_view kWan21TalkPreset = "wan21-talk";

[[nodiscard]] std::string_view ModelId(Family family) noexcept;
[[nodiscard]] std::string_view Preset(Family family) noexcept;
/// Frames per second of the written MP4.
[[nodiscard]] int Fps(Family family) noexcept;

/// `root/model_index.json` describes a single-transformer WanPipeline.
[[nodiscard]] bool IsWan22ModelDirectory(const std::filesystem::path& root);
/// Bundles: gufo-camera.json / gufo-vace.json / gufo-talk.json with their
/// `_class_name` (GufoWan21Uni3CCamera, GufoWan21Vace,
/// GufoWan21InfiniteTalk) name the parts.
[[nodiscard]] bool IsWan21CameraDirectory(const std::filesystem::path& root);
[[nodiscard]] bool IsWan21VaceDirectory(const std::filesystem::path& root);
[[nodiscard]] bool IsWan21TalkDirectory(const std::filesystem::path& root);

/// "WxH".
[[nodiscard]] bool ParseVideoSize(std::string_view size, int* width,
                                  int* height);

/// Size, frames and steps per family (preset must match the family).
[[nodiscard]] bool ValidateParameters(Family family,
                                      const ext::VideoParameters& parameters,
                                      std::string* error = nullptr);
/// HTTP "seconds" to frames: Wan 2.2 24 fps 4k + 1 (0.2..5 s), Wan 2.1
/// camera / VACE 16 fps 4k + 1 (0.25..5 s), InfiniteTalk round(s * 25)
/// (0.04..60 s).
[[nodiscard]] std::optional<int> ResolveDurationFrames(
    Family family, std::string_view seconds, std::string* error = nullptr);

/// "gufo.camera" without "render_video": {"type", "amount", "speed", "ease"}
/// or {"poses": [{"frame", "position": [x, y, z], "yaw", "pitch", "roll",
/// "zoom"}, ...]} (positions in scene depths, angles in degrees).
[[nodiscard]] bool ParseCameraSpec(const json::Value& camera, CameraSpec* spec,
                                   std::string* error);

/// Parses one `POST /v1/videos` request of `family` (see request.cpp for the
/// per-family formats). Decodes and checks every image and audio input; the
/// encoded bytes travel as attachments (names: "first_frame",
/// "guide_video", "mask", "reference_image_<n>", "audio_<n>",
/// "speaker_mask_<n>"), the options as JSON.
[[nodiscard]] bool PrepareRequest(Family family,
                                  const ext::VideoRequestInput& input,
                                  ext::VideoPreparedRequest* prepared,
                                  ext::VideoRequestError* error);

/// A decoded RGB8 image.
struct Image {
  int width{0};
  int height{0};
  std::vector<std::uint8_t> rgb;
};

/// What a runner needs, decoded from an `ext::VideoRequest` the provider
/// prepared (field names follow the 1.x fork's GenerationRequest).
struct JobRequest {
  Family family{Family::kWan22};
  std::filesystem::path model_root;
  std::filesystem::path output_path;
  std::string prompt;
  std::uint64_t seed{42};
  int width{0};
  int height{0};
  int frames{0};
  int steps{0};
  // Start image (Wan 2.2: optional image-to-video; Wan 2.1 camera and talk:
  // required), RGB8 at its own size.
  Image first_frame;
  std::string negative_prompt;
  double guidance{5.0};
  // Uni3C: "gufo.camera" as JSON, or a render video in guide_video_path.
  std::string camera_json;
  // Uni3C render video, VACE control video, InfiniteTalk input video.
  std::filesystem::path guide_video_path;
  // VACE: options JSON, mask file (".mp4" or ".ppm"), reference images.
  std::string vace_json;
  std::filesystem::path mask_path;
  std::vector<Image> references;
  // InfiniteTalk: normalized "gufo.talk" options, one or two speakers and
  // (two speakers) their masks.
  std::string talk_json;
  std::vector<AudioWaveform> speakers;
  std::vector<Image> speaker_masks;
  // Live preview hook (unset: gufo 2.0 has no preview service yet).
  PreviewHook preview;
};

/// Decodes `request` (as prepared by PrepareRequest) into `job`; video and
/// mask attachments are written next to `request.output_path` (guide.mp4,
/// mask.mp4 / mask.ppm) and removed with the job directory.
[[nodiscard]] bool BuildJobRequest(Family family,
                                   const ext::VideoRequest& request,
                                   JobRequest* job, std::string* error);

/// parameters.json of a job: preset, size, frames, steps, seed, prompt hash.
[[nodiscard]] std::string ParametersJson(const ext::VideoParameters& parameters,
                                         std::uint64_t seed,
                                         std::string_view prompt_sha256);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_REQUEST_HPP_
