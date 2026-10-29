#ifndef GUFO_CLI_SERVE_ROUTER_COMFY_GRAPH_HPP_
#define GUFO_CLI_SERVE_ROUTER_COMFY_GRAPH_HPP_

// Pure half of the ComfyUI adapter: reads a ComfyUI API-format workflow
// ({"<node id>": {"class_type": .., "inputs": {..}}}), recognizes the node
// graphs gufo can serve and turns each into a task for a gufo endpoint.
// No I/O here, so the mapping is unit-testable without a router.
//
// Supported families (each keyed by the node that produces the media):
//   image  KSampler <- TextEncodeQwenImage21 / CLIPTextEncode (Qwen-Image)
//          -> /v1/images/generations, or /v1/images/edits with references
//   h3     KSampler <- MiniMaxH3ImageToVideo (+ MiniMaxH3AddGuide)
//          -> /v1/videos (minimax-h3)
//   wan    KSampler <- Wan22ImageToVideoLatent -> /v1/videos (wan2.2-ti2v-5b)
//   speech FB_Qwen3TTSVoiceDesign / VoiceClone / CustomVoice
//          -> /v1/audio/speech
// A new family (RIFE, ACE-Step) adds its node classes to the tables in
// comfy_graph.cpp, a task type here and an executor in comfy_adapter.cpp.

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
  std::string h3{"minimax-h3"};
  std::string wan{"wan2.2-ti2v-5b"};
  std::string tts_custom{"qwen3-tts"};
  std::string tts_design{"qwen3-tts-voicedesign"};
  std::string tts_clone{"qwen3-tts-clone"};
};

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
  std::string negative_prompt;  // not sent: gufo Qwen-Image has none
  std::uint64_t seed{0};
  int steps{0};
  SizeSource size_source{SizeSource::kSquare};
  int width{0};
  int height{0};
  int resolution{1024};
  FileRef canvas;
  std::vector<FileRef> references;
  int batch{1};
};

struct SpeechTask {
  std::string model;
  std::string voice;
  std::string text;
  std::string instructions;
  std::string language;  // gufo spelling ("german"); empty = model default
  std::optional<std::uint64_t> seed;
  std::optional<FileRef> reference_audio;
  std::string reference_text;
  bool embedding_only{false};
  json::Value sampling = json::Value::object();  // extra sampling fields
};

enum class VideoFamily { kH3, kWan };

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
};

/// The /v1/videos body for `task` without the keyframes (the executor adds
/// them as data URLs).
json::Value VideoRequestBody(const VideoTask& task);

struct Task {
  std::string node_id;  // the sampler / TTS node producing the media
  std::string class_type;
  std::variant<ImageTask, SpeechTask, VideoTask> spec;
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
