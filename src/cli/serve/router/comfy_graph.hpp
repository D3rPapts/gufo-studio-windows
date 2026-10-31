#ifndef GUFO_CLI_SERVE_ROUTER_COMFY_GRAPH_HPP_
#define GUFO_CLI_SERVE_ROUTER_COMFY_GRAPH_HPP_

// Pure half of the ComfyUI adapter: reads a ComfyUI API-format workflow
// ({"<node id>": {"class_type": .., "inputs": {..}}}), recognizes the node
// graphs gufo can serve and turns each into a task for a gufo endpoint.
// No I/O here, so the mapping is unit-testable without a router.
//
// Supported families (each keyed by the node that produces the media):
//   image  KSampler <- TextEncodeQwenImage21 / CLIPTextEncode (Qwen-Image)
//          -> /v1/images/generations, or /v1/images/edits with references;
//          a Turbo UNETLoader file -> qwen-image-2.1-turbo (8 steps only)
//   h3     KSampler <- MiniMaxH3ImageToVideo (+ MiniMaxH3AddGuide, whose
//          image / audio may come from LoadVideo -> GetVideoComponents as a
//          guide clip with its soundtrack) -> /v1/videos (minimax-h3)
//   wan    KSampler <- Wan22ImageToVideoLatent -> /v1/videos (wan2.2-ti2v-5b)
//   speech FB_Qwen3TTSVoiceDesign / VoiceClone / CustomVoice
//          -> /v1/audio/speech
//   music  KSampler <- EmptyAceStep1.5LatentAudio + TextEncodeAceStepAudio1.5
//          -> /v1/audio/music (ace-step-1.5-xl-turbo)
//   camera KSampler <- WanImageToVideo with the model through
//          WanUni3CControlnetApply (render_video from LoadVideo or a
//          GufoCameraMove node), or WanCameraImageToVideo <- WanCameraEmbedding
//          / GufoCameraMove -> /v1/videos (wan2.1-uni3c, gufo.camera)
//   vace   KSampler <- WanVaceToVideo (control_video from LoadVideo /
//          VHS_LoadVideo, control_masks through ImageToMask / InvertMask or
//          LoadImageMask, reference_image from LoadImage), TrimVideoLatent
//          after it -> /v1/videos (wan2.1-vace-14b, gufo.vace)
//   talk   KSampler <- WanInfiniteTalkToVideo (audio from LoadAudio ->
//          AudioEncoderEncode, two speakers with mask_1 / mask_2; chained
//          windows via previous_frames become one job)
//          -> /v1/videos (wan2.1-infinitetalk-14b, gufo.talk, 25 fps)
// A new family adds its node classes to the tables in comfy_graph.cpp, a
// task type here and an executor in comfy_adapter.cpp.
//
// GufoCameraMove is gufo's own node (no ComfyUI counterpart): inputs
// camera ("orbit_left", "dolly_in", ... as gufo.camera.type), speed, ease
// ("linear" | "ease_in_out") and optional amount (< 0 = type default). It
// may feed WanUni3CControlnetApply.render_video or
// WanCameraImageToVideo.camera_conditions.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::router::comfy {

/// Router model ids each node family is sent to.
struct ModelIds {
  std::string image{"qwen-image-2.1"};
  /// Qwen-Image graphs whose UNETLoader / checkpoint name contains "turbo"
  /// (IsQwenImageTurboFile). They need exactly 8 steps and denoise 1; empty
  /// = such graphs fail validation instead of running on `image`.
  std::string image_turbo{"qwen-image-2.1-turbo"};
  std::string h3{"minimax-h3"};
  std::string wan{"wan2.2-ti2v-5b"};
  std::string tts_custom{"qwen3-tts"};
  std::string tts_design{"qwen3-tts-voicedesign"};
  std::string tts_clone{"qwen3-tts-clone"};
  std::string music{"ace-step-1.5-xl-turbo"};
  std::string camera{"wan2.1-uni3c"};
  /// Post-processing only (RIFE / upscale of a loaded video): a video server
  /// with --rife-model. When `available` is set and lacks it, `h3` serves.
  std::string process{"gufo-rife"};
  std::string vace{"wan2.1-vace-14b"};
  std::string talk{"wan2.1-infinitetalk-14b"};
  std::string animate{"wan2.2-animate-14b"};
  /// The router's model ids. When not empty, a Turbo graph whose
  /// `image_turbo` is not among them fails validation at the loader node
  /// (no silent fallback to `image`). Empty = not checked.
  std::vector<std::string> available;
};

/// True for a Qwen-Image Turbo checkpoint file name ("turbo" anywhere in it,
/// any case), e.g. "qwen_image_2.1_turbo_bf16.safetensors".
bool IsQwenImageTurboFile(std::string_view name);

/// A file a Load* node names, e.g. "sub/name.png [output]".
struct FileRef {
  std::string name;    // relative path inside `folder`, '/' separated
  std::string folder;  // "input", "output" or "temp"
  bool operator==(const FileRef&) const = default;
};

enum class OutputKind { kImage, kAudio, kVideo };

/// One Save*/Preview* node that receives the task's media.
struct OutputTarget {
  std::string node_id;
  std::string class_type;
  OutputKind kind{OutputKind::kImage};
  std::string prefix;  // filename_prefix
  bool temp{false};    // Preview* nodes write to the temp folder
};

enum class SizeSource {
  kExplicit,        // width/height from an empty latent
  kFirstReference,  // first reference image scaled to resolution^2 pixels
  kCanvas,          // size of the image a VAEEncode encodes
  kSquare,          // resolution x resolution
};

struct ImageTask {
  std::string model;
  std::string prompt;
  std::string negative_prompt;  // gufo.negative_prompt
  std::uint64_t seed{0};
  int steps{0};
  std::optional<double> guidance;  // KSampler cfg -> gufo.guidance
  /// denoise < 1: image to image from `init_image` with gufo.strength.
  std::optional<double> strength;
  FileRef init_image;
  /// SetLatentNoiseMask / VAEEncodeForInpaint: masked editing of
  /// `init_image`; sent as the multipart "mask" with gufo.mask_channel.
  FileRef mask;
  std::string mask_channel;  // "alpha", "red", "green" or "blue"
  SizeSource size_source{SizeSource::kSquare};
  int width{0};
  int height{0};
  int resolution{1024};
  FileRef canvas;
  std::vector<FileRef> references;
  int batch{1};
  /// LoraLoader(ModelOnly) chain -> top-level "loras" [{name, scale}], in
  /// application order; names via LoraName, empty = field not sent.
  std::vector<std::pair<std::string, double>> loras;
};

struct SpeechTask {
  std::string model;
  std::string voice;
  std::string text;
  std::string instructions;
  std::string language;  // gufo spelling ("german", "auto"); empty = model default
  std::optional<std::uint64_t> seed;
  std::optional<FileRef> reference_audio;
  std::string reference_text;
  bool embedding_only{false};
  json::Value sampling = json::Value::object();  // extra sampling fields
};

/// FB_Qwen3TTSDialogueInference: "Role: text" lines, each spoken with the
/// role's clone reference (FB_Qwen3TTSRoleBank) and joined with silence.
struct DialogueTask {
  std::vector<SpeechTask> lines;  // one clone request per script line
  double pause_seconds{0.5};      // silence between lines
  bool merge{true};               // false: one output file per line
};

/// ACE-Step 1.5 text to music (/v1/audio/music).
struct MusicTask {
  std::string model;
  std::string tags;    // -> prompt
  std::string lyrics;
  double duration{30.0};  // EmptyAceStep1.5LatentAudio seconds
  std::uint64_t seed{0};
  int steps{8};
  int bpm{0};  // 0 = not sent
  std::string timesignature;
  std::string language;
  std::string keyscale;
  std::optional<bool> audio_codes;  // generate_audio_codes
  std::optional<double> lm_cfg;
  std::optional<double> lm_temperature;
  std::optional<double> lm_top_p;
};

/// The /v1/audio/music body for `task` (synchronous, WAV).
json::Value MusicRequestBody(const MusicTask& task);

/// kProcess: no generation, a loaded video is only interpolated / upscaled
/// (LoadVideo -> GetVideoComponents -> RIFE VFI / ImageUpscaleWithModel).
enum class VideoFamily { kH3, kWan, kCamera, kProcess, kVace, kTalk, kAnimate };

/// Frames per second a family generates (16 camera / VACE, 25 talk, 24 else).
int VideoFamilyFps(VideoFamily family);

/// Real-ESRGAN upscaling of the finished video -> gufo.upscale.
struct VideoUpscale {
  int width{0};
  int height{0};
  std::string fit{"stretch"};  // "pad", "crop" or "stretch"
  std::string model;           // e.g. "realesr-animevideov3"
  bool operator==(const VideoUpscale&) const = default;
};

/// A preset camera move -> gufo.camera {type, amount, speed, ease}.
struct CameraMove {
  std::string type{"static"};
  std::optional<double> amount;
  double speed{1.0};
  std::string ease{"linear"};
};

struct VideoTask {
  VideoFamily family{VideoFamily::kH3};
  std::string model;
  std::string prompt;
  std::string negative_prompt;
  int width{0};
  int height{0};
  int frames{0};
  std::uint64_t seed{0};
  int steps{0};
  std::optional<double> guidance;
  std::optional<FileRef> first_frame;
  std::optional<FileRef> last_frame;
  std::string lora;
  std::optional<double> lora_strength;
  int fps{24};
  int interpolate{1};  // RIFE multiplier from a "RIFE VFI" node (gufo 1..8)
  /// Target frame rate fps_num / fps_den (gufo.interpolate {"fps"}) when a
  /// RIFE VFI feeds an output with another frame rate; 0 = none.
  int fps_num{0};
  int fps_den{1};
  /// ImageUpscaleWithModel (+ ImageScale / ImageScaleBy) after the decode.
  std::optional<VideoUpscale> upscale;
  /// H3 MiniMaxH3AddGuide images between first and last frame (frame index,
  /// negative = from the end) -> gufo.keyframes.
  std::vector<std::pair<int, FileRef>> keyframes;
  /// H3 MiniMaxH3AddGuide audio -> audio_reference at
  /// gufo.audio_reference_frame.
  std::optional<FileRef> audio_reference;
  int audio_reference_frame{0};
  /// The audio comes from GetVideoComponents (output 1) of a LoadVideo:
  /// `audio_reference` is that video and the executor extracts its
  /// soundtrack (ffmpeg, FLAC) before sending it.
  bool audio_reference_from_video{false};
  /// H3 MiniMaxH3AddGuide image from GetVideoComponents (output 0) of a
  /// LoadVideo: a guide clip (e.g. the last frames of the previous part),
  /// sent as gufo.guide_video {frame, video} by the executor.
  std::optional<FileRef> guide_video;
  int guide_video_frame{0};
  /// Camera family: a preset move or a rendered guide video (LoadVideo),
  /// sent as gufo.camera.render_video by the executor.
  std::optional<CameraMove> camera;
  std::optional<FileRef> render_video;
  /// kProcess: the LoadVideo file, sent as source_video.
  std::optional<FileRef> source_video;
  /// VACE family (gufo.vace, the files added by the executor): control
  /// video, mask (a video or a still for every frame) with its channel,
  /// reference images, strength, sampler / shift, and whether the bundle's
  /// step-distill LoRA applies (false: the graph has no LoRA).
  std::optional<FileRef> control_video;
  std::optional<FileRef> vace_mask;
  std::string vace_mask_channel;  // "red", "green", "blue"
  bool vace_invert_mask{false};
  std::vector<FileRef> reference_images;
  std::optional<double> vace_strength;
  std::string vace_sampler;  // "lcm" or "euler"; empty = bundle default
  std::optional<double> vace_shift;
  bool vace_lora{true};
  /// Talk family (InfiniteTalk): the speakers' audio (one or two LoadAudio
  /// files), with two speakers their masks and the mask channel ("red",
  /// "green", "blue", or "inverted_alpha": ComfyUI's alpha masks), sent
  /// as gufo.talk {audio, speaker_masks} by the executor.
  std::vector<FileRef> talk_audio;
  std::vector<FileRef> talk_masks;
  std::vector<std::string> talk_mask_channels;
  double audio_scale{1.0};
  int motion_frames{9};
};

/// The /v1/videos body for `task` without the keyframes (the executor adds
/// them as data URLs).
json::Value VideoRequestBody(const VideoTask& task);

struct Task {
  std::string node_id;  // the sampler / TTS node producing the media
  std::string class_type;
  std::variant<ImageTask, SpeechTask, VideoTask, DialogueTask, MusicTask> spec;
  std::vector<OutputTarget> outputs;
};

struct Plan {
  std::vector<Task> tasks;
  std::vector<std::string> warnings;
};

/// ComfyUI's /prompt error shape: {"error": {type, message, details},
/// "node_errors": {...}}.
struct PlanError {
  std::string type;
  std::string message;
  std::string details;
  json::Value node_errors = json::Value::object();
};

bool BuildPlan(const json::Value& prompt, const ModelIds& models, Plan* plan,
               PlanError* error);

/// Every node class a workflow may contain.
const std::vector<std::string>& SupportedClassTypes();
bool IsOutputClass(std::string_view class_type);

/// "sub/name.png [output]" -> {"sub/name.png", "output"}.
FileRef ParseFileRef(std::string_view value, std::string_view default_folder);

/// A relative path below a served folder with '/' separators, or nullopt for
/// absolute paths, drive letters and ".." segments. Accepts '\' separators.
std::optional<std::string> SafeRelative(std::string_view path);

/// ComfyUI's reference resize: about resolution^2 pixels, multiples of 32,
/// aspect ratio kept.
std::pair<int, int> ScaleToArea(int width, int height, int resolution);

/// Width and height of a PNG or JPEG.
std::optional<std::pair<int, int>> ImageDimensions(std::string_view bytes);

/// MiniMax H3 frame counts follow 17k + 5; ComfyUI snaps lengths up.
int SnapH3Frames(int length);

std::string Base64Encode(std::string_view bytes);
std::optional<std::string> Base64Decode(std::string_view text);

struct MultipartPart {
  std::string name;
  std::string filename;
  std::string content_type;
  std::string data;
};
bool ParseMultipart(std::string_view content_type, std::string_view body,
                    std::vector<MultipartPart>* parts, std::string* error);
std::string BuildMultipart(const std::vector<MultipartPart>& parts,
                           std::string_view boundary);

/// The "gufo" extension object for an image request (negative prompt,
/// guidance, strength); empty when there is nothing to send.
json::Value ImageExtension(const ImageTask& task);

/// The "loras" array for an image request ([{"name", "scale"}]); null when
/// the graph has no LoRA node, so the worker's --lora-default set applies.
json::Value ImageLoras(const ImageTask& task);

/// The gufo adapter name for a ComfyUI lora_name: the file name without
/// folders and without ".safetensors" / ".sft" ("style/Anime.safetensors" ->
/// "Anime"). Register the file under that name with `--lora NAME=PATH`.
std::string LoraName(std::string_view lora_name);

/// 16-bit PCM WAV pieces for joining dialogue lines.
struct Wav {
  int sample_rate{0};
  int channels{0};
  std::string pcm;  // interleaved little-endian int16
};
std::optional<Wav> ParseWav(std::string_view bytes);
std::string BuildWav(const Wav& wav);
/// Joins `parts` (same format) with `pause_seconds` of silence between them.
std::optional<std::string> JoinWavs(const std::vector<std::string>& parts,
                                    double pause_seconds);

/// ComfyUI output naming: prefix "a/b/c" -> subfolder "a/b", base "c".
struct SaveName {
  std::string subfolder;
  std::string base;
};
std::optional<SaveName> SplitPrefix(std::string_view prefix);
/// Next free counter after the highest "<base>_<digits>" among `existing`.
int NextCounter(const std::vector<std::string>& existing,
                std::string_view base);
/// "<base>_00001_.png" (images, video) or "<base>_00001.flac" (audio).
std::string OutputFilename(std::string_view base, int counter, OutputKind kind,
                           std::string_view extension);

}  // namespace gufo::router::comfy

#endif  // GUFO_CLI_SERVE_ROUTER_COMFY_GRAPH_HPP_
