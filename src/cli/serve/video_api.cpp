#include "src/cli/serve/video_api.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "src/cli/serve/audio_stream.hpp"
#include "src/cli/serve/video_wan21.hpp"
#include "src/core/image.hpp"
#include "src/core/json.hpp"
#include "src/models/minimax_h3/audio_input.hpp"

namespace gufo::server {
namespace {

constexpr std::string_view kVideosPath = "/v1/videos";
constexpr double kMaximumExactJsonInteger = 9007199254740991.0;
constexpr std::size_t kMaximumMultipartHeadersBytes = 16U << 10U;
constexpr std::size_t kMaximumMultipartFields = 16;

HttpResponse Error(int status, std::string reason, std::string message,
                   std::string code) {
  json::Value root = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = std::move(message);
  error["type"] = "invalid_request_error";
  error["code"] = std::move(code);
  root["error"] = std::move(error);
  return {
      .status = status,
      .reason = std::move(reason),
      .body = root.dump(),
      .headers = {},
      .streaming_body = {},
  };
}

json::Value SnapshotValue(const VideoJobSnapshot& snapshot) {
  json::Value root = json::Value::object();
  root["id"] = snapshot.id;
  root["object"] = "video";
  root["schema"] = std::string(kVideoApiSchema);
  root["created_at"] = static_cast<long long>(snapshot.created_at);
  root["status"] = std::string(ToString(snapshot.status));
  root["progress"] = snapshot.progress;
  root["model"] = snapshot.model;
  root["size"] = snapshot.size;
  root["seconds"] = snapshot.seconds;
  root["output_format"] = snapshot.output_format;
  if (snapshot.completed_at > 0) {
    root["completed_at"] = static_cast<long long>(snapshot.completed_at);
  } else {
    root["completed_at"] = json::Value();
  }
  if (snapshot.expires_at > 0) {
    root["expires_at"] = static_cast<long long>(snapshot.expires_at);
  } else {
    root["expires_at"] = json::Value();
  }
  if (snapshot.error_code.empty()) {
    root["error"] = json::Value();
  } else {
    json::Value error = json::Value::object();
    error["code"] = snapshot.error_code;
    error["message"] = snapshot.error_message;
    root["error"] = std::move(error);
  }
  return root;
}

HttpResponse SnapshotResponse(int status, std::string reason,
                              const VideoJobSnapshot& snapshot) {
  return {
      .status = status,
      .reason = std::move(reason),
      .body = SnapshotValue(snapshot).dump(),
      .headers = {},
      .streaming_body = {},
      .log_details = "job=" + snapshot.id +
                     " state=" + std::string(ToString(snapshot.status)) +
                     " model=" + snapshot.model + " size=" + snapshot.size,
  };
}

bool HasOnlyMembers(const json::Value& value,
                    std::initializer_list<std::string_view> allowed) {
  return std::ranges::all_of(value.members(), [&](const auto& member) {
    return std::ranges::find(allowed, member.first) != allowed.end();
  });
}

// "gufo.preview": live preview options of the job (its id is the job id).
std::optional<HttpResponse> ParseVideoPreview(const json::Value& extension,
                                              PreviewOptions* preview) {
  const json::Value* value = extension.find("preview");
  if (value == nullptr)
    return std::nullopt;
  try {
    if (!ParsePreviewRequest(*value, preview).empty())
      throw std::invalid_argument(
          "gufo.preview.id is the job id for videos; leave it out");
  } catch (const std::invalid_argument& error) {
    return Error(400, "Bad Request", error.what(), "invalid_preview");
  }
  return std::nullopt;
}

std::string_view TrimAscii(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())) != 0) {
    value.remove_prefix(1);
  }
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.remove_suffix(1);
  }
  return value;
}

bool EqualCaseInsensitive(std::string_view left, std::string_view right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (std::tolower(static_cast<unsigned char>(left[index])) !=
        std::tolower(static_cast<unsigned char>(right[index]))) {
      return false;
    }
  }
  return true;
}

std::vector<std::string_view> HeaderSegments(std::string_view value,
                                             bool* valid) {
  std::vector<std::string_view> result;
  std::size_t begin = 0;
  bool quoted = false;
  bool escaped = false;
  for (std::size_t index = 0; index < value.size(); ++index) {
    const char character = value[index];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (quoted && character == '\\') {
      escaped = true;
      continue;
    }
    if (character == '"') {
      quoted = !quoted;
    } else if (character == ';' && !quoted) {
      result.push_back(TrimAscii(value.substr(begin, index - begin)));
      begin = index + 1;
    }
  }
  if (quoted || escaped) {
    *valid = false;
    return {};
  }
  result.push_back(TrimAscii(value.substr(begin)));
  *valid = true;
  return result;
}

bool DecodeHeaderParameter(std::string_view value, std::string* output) {
  value = TrimAscii(value);
  if (value.empty()) {
    output->clear();
    return true;
  }
  if (value.front() != '"') {
    if (value.find_first_of("\"\r\n") != std::string_view::npos) {
      return false;
    }
    *output = value;
    return true;
  }
  if (value.size() < 2 || value.back() != '"') {
    return false;
  }
  output->clear();
  const std::string_view inner = value.substr(1, value.size() - 2);
  bool escaped = false;
  for (const char character : inner) {
    if (escaped) {
      output->push_back(character);
      escaped = false;
    } else if (character == '\\') {
      escaped = true;
    } else if (character == '"' || character == '\r' || character == '\n') {
      return false;
    } else {
      output->push_back(character);
    }
  }
  return !escaped;
}

bool HeaderParameter(std::string_view value, std::string_view parameter,
                     std::string* output, bool* present) {
  bool valid = false;
  const std::vector<std::string_view> segments = HeaderSegments(value, &valid);
  if (!valid || segments.empty()) {
    return false;
  }
  *present = false;
  for (std::size_t index = 1; index < segments.size(); ++index) {
    const std::size_t equals = segments[index].find('=');
    if (equals == std::string_view::npos) {
      return false;
    }
    const std::string_view name = TrimAscii(segments[index].substr(0, equals));
    if (!EqualCaseInsensitive(name, parameter)) {
      continue;
    }
    if (*present ||
        !DecodeHeaderParameter(segments[index].substr(equals + 1), output)) {
      return false;
    }
    *present = true;
  }
  return true;
}

bool IsMultipartFormData(std::string_view content_type) {
  const std::size_t semicolon = content_type.find(';');
  return EqualCaseInsensitive(TrimAscii(content_type.substr(0, semicolon)),
                              "multipart/form-data");
}

std::optional<std::string> MultipartBoundary(std::string_view content_type) {
  std::string boundary;
  bool present = false;
  if (!HeaderParameter(content_type, "boundary", &boundary, &present) ||
      !present || boundary.empty() || boundary.size() > 200 ||
      boundary.find_first_of("\r\n") != std::string::npos) {
    return std::nullopt;
  }
  return boundary;
}

bool IsKeyframeField(std::string_view name) {
  return name == "input_reference" || name == "first_frame" ||
         name == "last_frame" || name == "audio_reference" ||
         name == "render_video" || name.starts_with("keyframe_");
}

// Uploaded keyframe files are returned in `files`; every other part becomes a
// string member (or the parsed `gufo` object) of the result.
std::optional<json::Value> ParseMultipartBody(
    const HttpRequest& request, std::map<std::string, std::string>* files,
    std::string* error) {
  const auto boundary = MultipartBoundary(request.header("content-type"));
  if (!boundary.has_value()) {
    *error = "multipart video request has an invalid boundary";
    return std::nullopt;
  }
  const std::string delimiter = "--" + *boundary;
  const std::string_view body = request.body;
  if (!body.starts_with(delimiter)) {
    *error = "multipart video request does not start with its boundary";
    return std::nullopt;
  }

  json::Value result = json::Value::object();
  std::size_t cursor = delimiter.size();
  std::size_t fields = 0;
  while (true) {
    if (body.substr(cursor).starts_with("--")) {
      cursor += 2;
      if (body.substr(cursor).starts_with("\r\n")) {
        cursor += 2;
      }
      if (cursor != body.size()) {
        *error = "multipart video request has data after its final boundary";
        return std::nullopt;
      }
      return result;
    }
    if (!body.substr(cursor).starts_with("\r\n")) {
      *error = "multipart video request has an invalid boundary separator";
      return std::nullopt;
    }
    cursor += 2;
    const std::size_t headers_end = body.find("\r\n\r\n", cursor);
    if (headers_end == std::string_view::npos ||
        headers_end - cursor > kMaximumMultipartHeadersBytes) {
      *error = "multipart video request has invalid part headers";
      return std::nullopt;
    }

    std::string content_disposition;
    std::string_view headers = body.substr(cursor, headers_end - cursor);
    while (!headers.empty()) {
      const std::size_t newline = headers.find("\r\n");
      const std::string_view line = headers.substr(0, newline);
      const std::size_t colon = line.find(':');
      if (colon == std::string_view::npos) {
        *error = "multipart video request has a malformed part header";
        return std::nullopt;
      }
      const std::string_view name = TrimAscii(line.substr(0, colon));
      const std::string_view value = TrimAscii(line.substr(colon + 1));
      if (EqualCaseInsensitive(name, "content-disposition")) {
        if (!content_disposition.empty()) {
          *error = "multipart video request repeats Content-Disposition";
          return std::nullopt;
        }
        content_disposition = value;
      }
      if (newline == std::string_view::npos) {
        break;
      }
      headers.remove_prefix(newline + 2);
    }

    bool disposition_valid = false;
    const std::vector<std::string_view> disposition_segments =
        HeaderSegments(content_disposition, &disposition_valid);
    std::string field_name;
    std::string filename;
    bool name_present = false;
    bool filename_present = false;
    if (!disposition_valid || disposition_segments.empty() ||
        !EqualCaseInsensitive(disposition_segments.front(), "form-data") ||
        !HeaderParameter(content_disposition, "name", &field_name,
                         &name_present) ||
        !HeaderParameter(content_disposition, "filename", &filename,
                         &filename_present) ||
        !name_present || field_name.empty()) {
      *error = "multipart video request has invalid Content-Disposition";
      return std::nullopt;
    }

    const std::size_t content_begin = headers_end + 4;
    const std::string next_boundary = "\r\n" + delimiter;
    const std::size_t content_end = body.find(next_boundary, content_begin);
    if (content_end == std::string_view::npos) {
      *error = "multipart video request is missing a closing boundary";
      return std::nullopt;
    }
    if (++fields > kMaximumMultipartFields || result.contains(field_name) ||
        files->contains(field_name)) {
      *error = fields > kMaximumMultipartFields
                   ? "multipart video request contains too many fields"
                   : "multipart video request repeats a field";
      return std::nullopt;
    }
    const std::string_view content =
        body.substr(content_begin, content_end - content_begin);
    if (field_name == "gufo" && !filename_present) {
      try {
        json::Value extension = json::parse(std::string(content));
        if (!extension.is_object()) {
          *error = "multipart 'gufo' field must contain a JSON object";
          return std::nullopt;
        }
        result[field_name] = std::move(extension);
      } catch (const std::exception&) {
        *error = "multipart 'gufo' field must contain valid JSON";
        return std::nullopt;
      }
    } else if (filename_present && IsKeyframeField(field_name)) {
      (*files)[field_name] = std::string(content);
    } else {
      result[field_name] =
          filename_present ? std::string() : std::string(content);
    }
    cursor = content_end + next_boundary.size();
  }
}

std::optional<std::uint64_t> JsonUnsigned(const json::Value* value) {
  if (value == nullptr || !value->is_number()) {
    return std::nullopt;
  }
  const double number = value->as_double(-1.0);
  if (!std::isfinite(number) || number < 0.0 ||
      number > kMaximumExactJsonInteger || std::floor(number) != number) {
    return std::nullopt;
  }
  return static_cast<std::uint64_t>(number);
}

// A signed exact integer in int range (guide frames count from the end when
// negative).
std::optional<int> JsonSignedFrame(const json::Value* value) {
  if (value == nullptr || !value->is_number()) {
    return std::nullopt;
  }
  const double number = value->as_double(0.5);
  if (!std::isfinite(number) || std::floor(number) != number ||
      number < -100000.0 || number > 100000.0) {
    return std::nullopt;
  }
  return static_cast<int>(number);
}

// An audio guide: a base64 data URL ("data:audio/flac;base64,...") or raw
// base64 of a WAV/FLAC/Ogg/MP3 file, or the uploaded file's bytes.
std::optional<minimax_h3::AudioWaveform> ReadGuideAudio(
    const json::Value* value, const std::string* file, std::string* error) {
  std::string bytes;
  if (file != nullptr) {
    bytes = *file;
  } else {
    if (value == nullptr || !value->is_string() || value->str().empty()) {
      *error = "must be an audio file or a base64 (data URL) string";
      return std::nullopt;
    }
    std::string_view text = value->str();
    if (text.starts_with("data:")) {
      const std::size_t marker = text.find(";base64,");
      if (marker == std::string_view::npos) {
        *error = "data URL must be base64 encoded";
        return std::nullopt;
      }
      text.remove_prefix(marker + 8);
    }
    if (!DecodeAudioBase64(text, minimax_h3::kH3MaximumGuideAudioBytes,
                           &bytes)) {
      *error = "is not valid base64 or exceeds 32 MiB";
      return std::nullopt;
    }
  }
  minimax_h3::AudioWaveform waveform;
  if (!minimax_h3::DecodeGuideAudio(
          std::span(reinterpret_cast<const std::uint8_t*>(bytes.data()),
                    bytes.size()),
          &waveform, error)) {
    return std::nullopt;
  }
  return waveform;
}

std::optional<int> JsonFrame(const json::Value* value) {
  const auto parsed = JsonUnsigned(value);
  if (!parsed.has_value() ||
      *parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return std::nullopt;
  }
  return static_cast<int>(*parsed);
}

bool IsUnsupportedConditioning(const json::Value& body) {
  static constexpr std::string_view kUnsupported[] = {"references"};
  return std::ranges::any_of(kUnsupported, [&](std::string_view key) {
    return body.contains(std::string(key));
  });
}

// A keyframe is an uploaded file, a base64 data URL, raw base64, a public
// HTTPS URL, or an object {"image_url": <string or {"url": string}>}.
std::optional<minimax_h3::GenerationKeyframe> ReadKeyframe(
    const json::Value* value, const std::string* file,
    minimax_h3::KeyframeAnchor anchor, core::ImageReadBudget& budget,
    std::string* error) {
  try {
    std::vector<std::uint8_t> bytes;
    if (file != nullptr) {
      if (file->size() > budget.remaining_bytes) {
        throw std::invalid_argument("keyframe images exceed the size limit");
      }
      budget.remaining_bytes -= file->size();
      bytes.assign(file->begin(), file->end());
    } else {
      const json::Value* source = value;
      if (source != nullptr && source->is_object()) {
        source = source->find("image_url");
        if (source != nullptr && source->is_object()) {
          source = source->find("url");
        }
      }
      if (source == nullptr || !source->is_string() || source->str().empty()) {
        throw std::invalid_argument(
            "must be an image file, a data URL, base64, an HTTPS URL or "
            "{\"image_url\": ...}");
      }
      const std::string& text = source->str();
      bytes = text.starts_with("data:") || text.starts_with("https://") ||
                      text.starts_with("http://")
                  ? core::ReadImageUrl(text, budget)
                  : core::ReadImageUrl("data:image/png;base64," + text, budget);
    }
    core::Image image = core::DecodeImage(bytes);
    return minimax_h3::GenerationKeyframe{
        .anchor = anchor,
        .width = static_cast<int>(image.width),
        .height = static_cast<int>(image.height),
        .rgb = std::move(image.pixels),
    };
  } catch (const std::exception& exception) {
    *error = exception.what();
    return std::nullopt;
  }
}

// Wan 2.2 image-to-video: one first frame with its own image budget.
std::optional<core::Image> ReadFirstFrame(const json::Value* value,
                                          const std::string* file,
                                          std::string* error) {
  core::ImageReadBudget budget;
  auto keyframe = ReadKeyframe(value, file, minimax_h3::KeyframeAnchor::kFirst,
                               budget, error);
  if (!keyframe.has_value()) {
    return std::nullopt;
  }
  return core::Image{.width = static_cast<std::uint32_t>(keyframe->width),
                     .height = static_cast<std::uint32_t>(keyframe->height),
                     .pixels = std::move(keyframe->rgb)};
}

// "gufo.interpolate": RIFE applied to the finished MP4, either a whole-number
// multiplier (1 = off, 2..8; 4 turns 24 fps into 96 fps at the same speed) or
// {"fps": N} / {"fps": "N/D"}, a target frame rate at the same speed (60 from
// H3's 24 or Wan's 16 fps; 1..240).
std::optional<VideoInterpolation> JsonInterpolate(const json::Value* value) {
  if (value != nullptr && value->is_object()) {
    const json::Value* fps = value->find("fps");
    if (!HasOnlyMembers(*value, {"fps"}) || fps == nullptr) {
      return std::nullopt;
    }
    VideoInterpolation spec;
    if (fps->is_string()) {
      const std::string& text = fps->str();
      const std::size_t slash = text.find('/');
      const auto parse = [](std::string_view part, int* out) {
        const auto result =
            std::from_chars(part.data(), part.data() + part.size(), *out);
        return result.ec == std::errc{} &&
               result.ptr == part.data() + part.size();
      };
      const std::string_view whole(text);
      if (!parse(whole.substr(0, slash), &spec.fps_num) ||
          (slash != std::string::npos &&
           !parse(whole.substr(slash + 1), &spec.fps_den))) {
        return std::nullopt;
      }
    } else {
      const auto parsed = JsonFrame(fps);
      if (!parsed.has_value()) {
        return std::nullopt;
      }
      spec.fps_num = *parsed;
    }
    if (spec.fps_num < 1 || spec.fps_den < 1 || spec.fps_den > 1001 ||
        spec.fps_num > 240 * spec.fps_den) {
      return std::nullopt;
    }
    return spec;
  }
  const auto parsed = JsonFrame(value);
  if (!parsed.has_value() || *parsed < 1 || *parsed > 8) {
    return std::nullopt;
  }
  return VideoInterpolation{.multiplier = *parsed};
}

// "gufo.upscale": Real-ESRGAN applied to the finished (interpolated) MP4:
// "WxH" or {"width", "height", "fit" ("pad" default, "crop", "stretch"),
// "model" (a name registered with --upscale-model)}. The job service checks
// the values.
std::optional<VideoUpscale> JsonUpscale(const json::Value* value) {
  VideoUpscale spec;
  if (value == nullptr) {
    return std::nullopt;
  }
  if (value->is_string()) {
    if (!ParseVideoSize(value->str(), &spec.width, &spec.height)) {
      return std::nullopt;
    }
    return spec;
  }
  if (!value->is_object() ||
      !HasOnlyMembers(*value, {"width", "height", "fit", "model"})) {
    return std::nullopt;
  }
  const auto width = JsonFrame(value->find("width"));
  const auto height = JsonFrame(value->find("height"));
  if (!width.has_value() || !height.has_value() || *width < 2 ||
      *height < 2) {
    return std::nullopt;
  }
  spec.width = *width;
  spec.height = *height;
  for (const char* key : {"fit", "model"}) {
    if (const json::Value* text = value->find(key)) {
      if (!text->is_string() || text->str().empty() ||
          text->str().size() > 128) {
        return std::nullopt;
      }
      (std::string_view(key) == "fit" ? spec.fit : spec.model) = text->str();
    }
  }
  return spec;
}

// Wan 2.2 TI2V-5B: {"model", "prompt", "size" (default "1280x704"),
// "seconds" (0.2..5 at 24 fps), "input_reference" or "first_frame" (image-
// to-video; the image is scaled to cover the size and centre-cropped),
// "gufo": {"seed", "steps" (default 50), "frames" (4k + 1, must match
// seconds), "negative_prompt" (default ""), "guidance" (default 5; <= 1
// disables classifier-free guidance), "interpolate" (RIFE multiplier or
// {"fps": N}), "upscale" (Real-ESRGAN, see JsonUpscale)}}.
// MP4 only.
HttpResponse CreateWan22Video(const json::Value& body,
                              const std::map<std::string, std::string>& files,
                              VideoJobService& service) {
  if (body.contains("last_frame") || files.contains("last_frame") ||
      body.contains("references")) {
    return Error(400, "Bad Request",
                 "Wan 2.2 supports only a first frame ('input_reference' or "
                 "'first_frame')",
                 "unsupported_input_reference");
  }
  if (!HasOnlyMembers(body, {"model", "prompt", "size", "seconds", "gufo",
                             "input_reference", "first_frame"})) {
    return Error(400, "Bad Request",
                 "request contains an unsupported video field",
                 "unsupported_field");
  }
  const auto has_field = [&](const char* name) {
    return body.contains(name) || files.contains(name);
  };
  if (has_field("input_reference") && has_field("first_frame")) {
    return Error(400, "Bad Request",
                 "'input_reference' and 'first_frame' both set the first frame",
                 "invalid_input_reference");
  }
  std::optional<core::Image> first_frame;
  for (const char* name : {"input_reference", "first_frame"}) {
    if (!has_field(name)) {
      continue;
    }
    const auto file = files.find(name);
    std::string image_error;
    first_frame = ReadFirstFrame(body.find(name),
                                 file == files.end() ? nullptr : &file->second,
                                 &image_error);
    if (!first_frame.has_value()) {
      return Error(400, "Bad Request",
                   "'" + std::string(name) + "' " + image_error,
                   "invalid_input_reference");
    }
  }
  const json::Value* model_value = body.find("model");
  const json::Value* prompt_value = body.find("prompt");
  const json::Value* size_value = body.find("size");
  if (model_value == nullptr || !model_value->is_string() ||
      prompt_value == nullptr || !prompt_value->is_string() ||
      (size_value != nullptr && !size_value->is_string())) {
    return Error(400, "Bad Request",
                 "'model', 'prompt', and optional 'size' must be strings",
                 "invalid_parameter_type");
  }
  const std::string model = model_value->str();
  const std::string prompt = prompt_value->str();
  const std::string size = body.member_str("size", "1280x704");
  const json::Value* seconds_value = body.find("seconds");
  std::string seconds = "5";
  if (seconds_value != nullptr && seconds_value->is_string()) {
    seconds = seconds_value->str();
  } else if (seconds_value != nullptr && seconds_value->is_number()) {
    seconds = seconds_value->dump();
  }
  if (model != service.model_id()) {
    return Error(400, "Bad Request", "unsupported video model",
                 "invalid_model");
  }
  if (prompt.empty()) {
    return Error(400, "Bad Request", "'prompt' is required",
                 "missing_required_parameter");
  }
  std::string duration_error;
  const auto duration_frames =
      ResolveWan22DurationFrames(seconds, &duration_error);
  if (!duration_frames) {
    return Error(400, "Bad Request", duration_error, "invalid_seconds");
  }
  minimax_h3::GenerationParameters parameters;
  parameters.preset = std::string(kWan22Preset);
  parameters.frames = *duration_frames;
  parameters.evaluations = 50;
  parameters.mux = true;
  parameters.decode_audio = false;
  if (!ParseVideoSize(size, &parameters.output_width,
                      &parameters.output_height)) {
    return Error(400, "Bad Request", "'size' must be WIDTHxHEIGHT",
                 "invalid_size");
  }
  parameters.internal_width = parameters.output_width;
  parameters.internal_height = parameters.output_height;
  std::uint64_t seed = 42;
  std::string negative_prompt;
  double guidance = 5.0;
  VideoInterpolation interpolate;
  VideoUpscale upscale;
  PreviewOptions preview = VideoJobRequest{}.preview;
  if (const json::Value* extension = body.find("gufo")) {
    if (!extension->is_object() ||
        !HasOnlyMembers(*extension,
                        {"seed", "steps", "frames", "negative_prompt",
                         "guidance", "interpolate", "upscale", "preview"})) {
      return Error(400, "Bad Request",
                   "'gufo' may contain only 'seed', 'steps', 'frames', "
                   "'negative_prompt', 'guidance', 'interpolate', "
                   "'upscale' and 'preview'",
                   "invalid_strix_options");
    }
    if (const auto failed = ParseVideoPreview(*extension, &preview)) {
      return *failed;
    }
    if (extension->contains("interpolate")) {
      const auto parsed = JsonInterpolate(extension->find("interpolate"));
      if (!parsed.has_value()) {
        return Error(400, "Bad Request",
                     "'gufo.interpolate' must be 1..8 or {\"fps\": 1..240}",
                     "invalid_interpolate");
      }
      interpolate = *parsed;
    }
    if (extension->contains("upscale")) {
      const auto parsed = JsonUpscale(extension->find("upscale"));
      if (!parsed.has_value()) {
        return Error(400, "Bad Request",
                     "'gufo.upscale' must be \"WxH\" or {\"width\", "
                     "\"height\", \"fit\", \"model\"}",
                     "invalid_upscale");
      }
      upscale = *parsed;
    }
    if (extension->contains("negative_prompt")) {
      const json::Value* value = extension->find("negative_prompt");
      if (!value->is_string() || value->str().find('\0') != std::string::npos) {
        return Error(400, "Bad Request",
                     "'gufo.negative_prompt' must be a string",
                     "invalid_negative_prompt");
      }
      negative_prompt = value->str();
    }
    if (extension->contains("guidance")) {
      const json::Value* value = extension->find("guidance");
      guidance = value->is_number() ? value->as_double(-1.0) : -1.0;
      if (!std::isfinite(guidance) || guidance < 0.0 || guidance > 20.0) {
        return Error(400, "Bad Request",
                     "'gufo.guidance' must be a number in 0..20",
                     "invalid_guidance");
      }
    }
    if (extension->contains("seed")) {
      const auto parsed = JsonUnsigned(extension->find("seed"));
      if (!parsed.has_value()) {
        return Error(400, "Bad Request",
                     "'gufo.seed' must be a non-negative exact integer",
                     "invalid_seed");
      }
      seed = *parsed;
    }
    if (extension->contains("steps")) {
      const auto parsed = JsonFrame(extension->find("steps"));
      if (!parsed.has_value()) {
        return Error(400, "Bad Request", "'gufo.steps' must be 1..100",
                     "invalid_steps");
      }
      parameters.evaluations = *parsed;
    }
    if (extension->contains("frames")) {
      const auto parsed = JsonFrame(extension->find("frames"));
      if (!parsed.has_value() || *parsed != *duration_frames) {
        return Error(400, "Bad Request",
                     "'gufo.frames' must match the 'seconds' duration",
                     "invalid_frames");
      }
    }
  }
  std::string parameter_error;
  if (!ValidateWan22Parameters(parameters, &parameter_error)) {
    return Error(400, "Bad Request", parameter_error, "invalid_parameters");
  }
  const VideoJobCreateResult created = service.Create({
      .model = model,
      .prompt = prompt,
      .size = size,
      .seconds = seconds,
      .output_format = "mp4",
      .parameters = std::move(parameters),
      .seed = seed,
      .first_frame_width =
          first_frame ? static_cast<int>(first_frame->width) : 0,
      .first_frame_height =
          first_frame ? static_cast<int>(first_frame->height) : 0,
      .first_frame = first_frame ? std::move(first_frame->pixels)
                                 : std::vector<std::uint8_t>(),
      .negative_prompt = std::move(negative_prompt),
      .guidance = guidance,
      .interpolate = interpolate.multiplier,
      .interpolate_fps_num = interpolate.fps_num,
      .interpolate_fps_den = interpolate.fps_den,
      .upscale = std::move(upscale),
      .preview = preview,
  });
  if (created.result == VideoJobResult::kQueueFull) {
    HttpResponse response =
        Error(429, "Too Many Requests", created.error, "video_queue_full");
    response.headers.emplace_back("Retry-After", "1");
    return response;
  }
  if (created.result != VideoJobResult::kOk || !created.job.has_value()) {
    const int status = created.result == VideoJobResult::kIoError ? 503 : 400;
    return Error(
        status, status == 503 ? "Service Unavailable" : "Bad Request",
        created.error,
        status == 503 ? "video_service_unavailable" : "invalid_video_request");
  }
  HttpResponse response = SnapshotResponse(202, "Accepted", *created.job);
  response.headers.emplace_back("Location", "/v1/videos/" + created.job->id);
  return response;
}

// Standard or URL-safe base64 (padding optional); nullopt when invalid.
std::optional<std::vector<std::uint8_t>> DecodeBase64(std::string_view text) {
  std::vector<std::uint8_t> out;
  out.reserve(text.size() * 3 / 4);
  std::uint32_t buffer = 0;
  int bits = 0;
  for (char c : text) {
    int v;
    if (c >= 'A' && c <= 'Z')
      v = c - 'A';
    else if (c >= 'a' && c <= 'z')
      v = c - 'a' + 26;
    else if (c >= '0' && c <= '9')
      v = c - '0' + 52;
    else if (c == '+' || c == '-')
      v = 62;
    else if (c == '/' || c == '_')
      v = 63;
    else if (c == '=')
      break;
    else if (c == '\r' || c == '\n' || c == ' ')
      continue;
    else
      return std::nullopt;
    buffer = (buffer << 6U) | static_cast<std::uint32_t>(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<std::uint8_t>((buffer >> bits) & 0xFFU));
    }
  }
  return out;
}

// Wan 2.1 I2V 14B + Uni3C ("wan2.1-uni3c"): {"model", "prompt", "size"
// (default "832x480", multiples of 16, at most 832x480 pixels), "seconds"
// (0.25..5 at 16 fps, default 5), "first_frame" or "input_reference"
// (required), "gufo": {"seed", "steps" (default 4), "frames",
// "interpolate", "camera": {"type", "amount", "speed", "ease"} |
// {"poses": [...]} | {"render_video": MP4 data URL or base64}}}; a
// multipart upload may carry the guide as a "render_video" file instead.
// Without a camera it is plain image-to-video.
HttpResponse CreateWan21Video(const json::Value& body,
                              const std::map<std::string, std::string>& files,
                              VideoJobService& service) {
  if (body.contains("last_frame") || files.contains("last_frame") ||
      body.contains("references")) {
    return Error(400, "Bad Request",
                 "Wan 2.1 supports only a first frame ('first_frame' or "
                 "'input_reference')",
                 "unsupported_input_reference");
  }
  if (!HasOnlyMembers(body,
                      {"model", "prompt", "size", "seconds", "gufo",
                       "input_reference", "first_frame", "render_video"})) {
    return Error(400, "Bad Request",
                 "request contains an unsupported video field",
                 "unsupported_field");
  }
  const auto has_field = [&](const char* name) {
    return body.contains(name) || files.contains(name);
  };
  if (has_field("input_reference") == has_field("first_frame")) {
    return Error(400, "Bad Request",
                 "exactly one of 'first_frame' and 'input_reference' is "
                 "required",
                 "invalid_input_reference");
  }
  std::optional<core::Image> first_frame;
  for (const char* name : {"input_reference", "first_frame"}) {
    if (!has_field(name))
      continue;
    const auto file = files.find(name);
    std::string image_error;
    first_frame = ReadFirstFrame(body.find(name),
                                 file == files.end() ? nullptr : &file->second,
                                 &image_error);
    if (!first_frame.has_value()) {
      return Error(400, "Bad Request",
                   "'" + std::string(name) + "' " + image_error,
                   "invalid_input_reference");
    }
  }
  const json::Value* model_value = body.find("model");
  const json::Value* prompt_value = body.find("prompt");
  const json::Value* size_value = body.find("size");
  if (model_value == nullptr || !model_value->is_string() ||
      prompt_value == nullptr || !prompt_value->is_string() ||
      (size_value != nullptr && !size_value->is_string())) {
    return Error(400, "Bad Request",
                 "'model', 'prompt', and optional 'size' must be strings",
                 "invalid_parameter_type");
  }
  const std::string model = model_value->str();
  if (model != service.model_id()) {
    return Error(400, "Bad Request", "unsupported video model",
                 "invalid_model");
  }
  const std::string prompt = prompt_value->str();
  if (prompt.find('\0') != std::string::npos) {
    return Error(400, "Bad Request", "'prompt' must not contain NUL",
                 "invalid_prompt");
  }
  const std::string size = body.member_str("size", "832x480");
  const json::Value* seconds_value = body.find("seconds");
  std::string seconds = "5";
  if (seconds_value != nullptr && seconds_value->is_string()) {
    seconds = seconds_value->str();
  } else if (seconds_value != nullptr && seconds_value->is_number()) {
    seconds = seconds_value->dump();
  }
  std::string duration_error;
  const auto duration_frames =
      ResolveWan21DurationFrames(seconds, &duration_error);
  if (!duration_frames) {
    return Error(400, "Bad Request", duration_error, "invalid_seconds");
  }
  minimax_h3::GenerationParameters parameters;
  parameters.preset = std::string(kWan21Preset);
  parameters.frames = *duration_frames;
  parameters.evaluations = 4;
  parameters.mux = true;
  parameters.decode_audio = false;
  if (!ParseVideoSize(size, &parameters.output_width,
                      &parameters.output_height)) {
    return Error(400, "Bad Request", "'size' must be WIDTHxHEIGHT",
                 "invalid_size");
  }
  parameters.internal_width = parameters.output_width;
  parameters.internal_height = parameters.output_height;
  std::uint64_t seed = 42;
  VideoInterpolation interpolate;
  VideoUpscale upscale;
  std::string camera_json;
  std::vector<std::uint8_t> guide_video;
  if (const auto upload = files.find("render_video"); upload != files.end()) {
    guide_video.assign(upload->second.begin(), upload->second.end());
  }
  if (const json::Value* extension = body.find("gufo")) {
    if (!extension->is_object() ||
        !HasOnlyMembers(*extension,
                        {"seed", "steps", "frames", "interpolate", "upscale",
                         "camera"})) {
      return Error(400, "Bad Request",
                   "'gufo' may contain only 'seed', 'steps', 'frames', "
                   "'interpolate', 'upscale' and 'camera'",
                   "invalid_strix_options");
    }
    if (extension->contains("interpolate")) {
      const auto parsed = JsonInterpolate(extension->find("interpolate"));
      if (!parsed.has_value()) {
        return Error(400, "Bad Request",
                     "'gufo.interpolate' must be 1..8 or {\"fps\": 1..240}",
                     "invalid_interpolate");
      }
      interpolate = *parsed;
    }
    if (extension->contains("upscale")) {
      const auto parsed = JsonUpscale(extension->find("upscale"));
      if (!parsed.has_value()) {
        return Error(400, "Bad Request",
                     "'gufo.upscale' must be \"WxH\" or {\"width\", "
                     "\"height\", \"fit\", \"model\"}",
                     "invalid_upscale");
      }
      upscale = *parsed;
    }
    if (extension->contains("seed")) {
      const auto parsed = JsonUnsigned(extension->find("seed"));
      if (!parsed.has_value()) {
        return Error(400, "Bad Request",
                     "'gufo.seed' must be a non-negative exact integer",
                     "invalid_seed");
      }
      seed = *parsed;
    }
    if (extension->contains("steps")) {
      const auto parsed = JsonFrame(extension->find("steps"));
      if (!parsed.has_value()) {
        return Error(400, "Bad Request", "'gufo.steps' must be 1..20",
                     "invalid_steps");
      }
      parameters.evaluations = *parsed;
    }
    if (extension->contains("frames")) {
      const auto parsed = JsonFrame(extension->find("frames"));
      if (!parsed.has_value() || *parsed != *duration_frames) {
        return Error(400, "Bad Request",
                     "'gufo.frames' must match the 'seconds' duration",
                     "invalid_frames");
      }
    }
    if (const json::Value* camera = extension->find("camera")) {
      if (!guide_video.empty()) {
        return Error(400, "Bad Request",
                     "'gufo.camera' and an uploaded 'render_video' exclude "
                     "each other",
                     "invalid_camera");
      }
      if (camera->is_object() && camera->contains("render_video")) {
        const json::Value* video = camera->find("render_video");
        if (camera->size() != 1 || !video->is_string()) {
          return Error(400, "Bad Request",
                       "'gufo.camera.render_video' must be the only camera "
                       "field and an MP4 data URL or base64 string",
                       "invalid_camera");
        }
        std::string_view text = video->str();
        if (text.starts_with("data:")) {
          const auto comma = text.find(',');
          if (comma == std::string_view::npos ||
              text.substr(0, comma).find(";base64") == std::string_view::npos) {
            return Error(400, "Bad Request",
                         "'gufo.camera.render_video' data URL must be base64",
                         "invalid_camera");
          }
          text.remove_prefix(comma + 1);
        }
        auto decoded = DecodeBase64(text);
        if (!decoded.has_value() || decoded->size() < 64) {
          return Error(400, "Bad Request",
                       "'gufo.camera.render_video' is not valid base64 video",
                       "invalid_camera");
        }
        guide_video = std::move(*decoded);
      } else {
        models::wan22::CameraSpec spec;
        std::string camera_error;
        if (!ParseCameraSpec(*camera, &spec, &camera_error)) {
          return Error(400, "Bad Request", camera_error, "invalid_camera");
        }
        camera_json = camera->dump();
      }
    }
  }
  std::string parameter_error;
  if (!ValidateWan21Parameters(parameters, &parameter_error)) {
    return Error(400, "Bad Request", parameter_error, "invalid_parameters");
  }
  const VideoJobCreateResult created = service.Create({
      .model = model,
      .prompt = prompt,
      .size = size,
      .seconds = seconds,
      .output_format = "mp4",
      .parameters = std::move(parameters),
      .seed = seed,
      .first_frame_width = static_cast<int>(first_frame->width),
      .first_frame_height = static_cast<int>(first_frame->height),
      .first_frame = std::move(first_frame->pixels),
      .interpolate = interpolate.multiplier,
      .interpolate_fps_num = interpolate.fps_num,
      .interpolate_fps_den = interpolate.fps_den,
      .upscale = std::move(upscale),
      .camera_json = std::move(camera_json),
      .guide_video = std::move(guide_video),
  });
  if (created.result == VideoJobResult::kQueueFull) {
    HttpResponse response =
        Error(429, "Too Many Requests", created.error, "video_queue_full");
    response.headers.emplace_back("Retry-After", "1");
    return response;
  }
  if (created.result != VideoJobResult::kOk || !created.job.has_value()) {
    const int status = created.result == VideoJobResult::kIoError ? 503 : 400;
    return Error(
        status, status == 503 ? "Service Unavailable" : "Bad Request",
        created.error,
        status == 503 ? "video_service_unavailable" : "invalid_video_request");
  }
  HttpResponse response = SnapshotResponse(202, "Accepted", *created.job);
  response.headers.emplace_back("Location", "/v1/videos/" + created.job->id);
  return response;
}

HttpResponse CreateVideo(const HttpRequest& request, VideoJobService& service) {
  json::Value body;
  std::map<std::string, std::string> files;
  const std::string content_type = request.header("content-type");
  if (IsMultipartFormData(content_type)) {
    std::string multipart_error;
    auto parsed = ParseMultipartBody(request, &files, &multipart_error);
    if (!parsed.has_value()) {
      return Error(400, "Bad Request", std::move(multipart_error),
                   "invalid_multipart_form");
    }
    body = std::move(*parsed);
  } else {
    if (!content_type.empty() &&
        !EqualCaseInsensitive(
            TrimAscii(content_type.substr(0, content_type.find(';'))),
            "application/json")) {
      return Error(415, "Unsupported Media Type",
                   "video requests must use application/json or "
                   "multipart/form-data",
                   "unsupported_media_type");
    }
    try {
      body = json::parse(request.body);
    } catch (const std::exception&) {
      return Error(400, "Bad Request", "request body must be valid JSON",
                   "parse_error");
    }
  }
  if (!body.is_object()) {
    return Error(400, "Bad Request", "request body must be a JSON object",
                 "invalid_body");
  }
  if (service.family() == VideoModelFamily::kWan22) {
    return CreateWan22Video(body, files, service);
  }
  if (service.family() == VideoModelFamily::kWan21Uni3C) {
    return CreateWan21Video(body, files, service);
  }
  if (IsUnsupportedConditioning(body)) {
    return Error(400, "Bad Request",
                 "image and ordered-reference conditioning are not supported",
                 "unsupported_input_reference");
  }
  if (!HasOnlyMembers(body, {"model", "prompt", "size", "seconds", "gufo",
                             "input_reference", "first_frame", "last_frame",
                             "audio_reference"})) {
    return Error(400, "Bad Request",
                 "request contains an unsupported video field",
                 "unsupported_field");
  }

  const json::Value* model_value = body.find("model");
  const json::Value* prompt_value = body.find("prompt");
  const json::Value* size_value = body.find("size");
  if (model_value == nullptr || !model_value->is_string() ||
      prompt_value == nullptr || !prompt_value->is_string() ||
      (size_value != nullptr && !size_value->is_string())) {
    return Error(400, "Bad Request",
                 "'model', 'prompt', and optional 'size' must be strings",
                 "invalid_parameter_type");
  }
  const std::string model = model_value->str();
  const std::string prompt = prompt_value->str();
  const std::string size = body.member_str("size", "512x512");
  const json::Value* seconds_value = body.find("seconds");
  std::string seconds;
  if (seconds_value != nullptr && seconds_value->is_string()) {
    seconds = seconds_value->str();
  } else if (seconds_value != nullptr && seconds_value->is_number()) {
    seconds = seconds_value->dump();
  }
  if (model.empty() || prompt.empty() || seconds.empty()) {
    return Error(400, "Bad Request",
                 "'model', 'prompt', and 'seconds' are required",
                 "missing_required_parameter");
  }
  std::string duration_error;
  const auto duration_frames =
      minimax_h3::ResolveDurationFrames(seconds, &duration_error);
  if (!duration_frames) {
    return Error(400, "Bad Request", duration_error, "invalid_seconds");
  }
  std::string preset = size == "1344x768" ? "exact-1344x768" : "exact";
  std::string model_preset;
  if (model == "minimax-h3-exact") {
    model_preset = "exact";
  } else if (model == "minimax-h3-fast") {
    model_preset = "fast";
  } else if (model == "minimax-h3-aggressive") {
    model_preset = "aggressive";
  } else if (model == "minimax-h3-dev") {
    model_preset = "dev";
  } else if (model == "minimax-h3-fullres") {
    model_preset = "exact-1344x768";
  } else if (model != "minimax-h3") {
    return Error(400, "Bad Request", "unsupported video model",
                 "invalid_model");
  }
  if (!model_preset.empty()) {
    preset = model_preset;
  }
  std::string output_format = "mp4";
  std::uint64_t seed = 42;
  std::optional<int> frames;
  std::optional<int> selected_frame;
  std::optional<int> steps;
  std::string lora;
  std::optional<double> lora_strength;
  VideoInterpolation interpolate;
  VideoUpscale upscale;
  PreviewOptions preview = VideoJobRequest{}.preview;
  bool preset_explicit = false;
  int audio_reference_frame = 0;
  const json::Value* guide_keyframes = nullptr;
  if (const json::Value* extension = body.find("gufo")) {
    if (!extension->is_object() ||
        !HasOnlyMembers(*extension, {"preset", "seed", "frames",
                                     "output_format", "selected_frame", "steps",
                                     "lora", "lora_strength", "interpolate",
                                     "upscale",
                                     "keyframes", "audio_reference_frame",
                                     "preview"})) {
      return Error(400, "Bad Request",
                   "'gufo' must contain only supported video options",
                   "invalid_strix_options");
    }
    if (const auto failed = ParseVideoPreview(*extension, &preview)) {
      return *failed;
    }
    if (extension->contains("preset")) {
      if (!extension->find("preset")->is_string()) {
        return Error(400, "Bad Request", "'gufo.preset' must be a string",
                     "invalid_preset");
      }
      preset = extension->member_str("preset");
      preset_explicit = true;
    }
    if (extension->contains("output_format") &&
        !extension->find("output_format")->is_string()) {
      return Error(400, "Bad Request", "'gufo.output_format' must be a string",
                   "invalid_output_format");
    }
    output_format = extension->member_str("output_format", output_format);
    if (extension->contains("seed")) {
      const auto parsed = JsonUnsigned(extension->find("seed"));
      if (!parsed.has_value()) {
        return Error(400, "Bad Request",
                     "'gufo.seed' must be a non-negative exact integer",
                     "invalid_seed");
      }
      seed = *parsed;
    }
    if (extension->contains("frames")) {
      frames = JsonFrame(extension->find("frames"));
      if (!frames.has_value()) {
        return Error(400, "Bad Request",
                     "'gufo.frames' must be a positive legal frame count",
                     "invalid_frames");
      }
    }
    if (extension->contains("steps")) {
      steps = JsonFrame(extension->find("steps"));
      if (!steps.has_value() || *steps < 2 ||
          *steps > minimax_h3::kH3MaximumEvaluations) {
        return Error(400, "Bad Request",
                     "'gufo.steps' must be an integer in 2..1000",
                     "invalid_steps");
      }
    }
    if (extension->contains("lora")) {
      const json::Value* value = extension->find("lora");
      if (!value->is_string() || value->str().empty()) {
        return Error(400, "Bad Request",
                     "'gufo.lora' must name a LoRA the server registered",
                     "invalid_lora");
      }
      lora = value->str();
    }
    if (extension->contains("lora_strength")) {
      const json::Value* value = extension->find("lora_strength");
      if (!value->is_number() || !std::isfinite(value->as_double(0.0)) ||
          std::fabs(value->as_double(0.0)) > 4.0) {
        return Error(400, "Bad Request",
                     "'gufo.lora_strength' must be a number in -4..4",
                     "invalid_lora");
      }
      lora_strength = value->as_double(1.0);
    }
    if (extension->contains("interpolate")) {
      const auto parsed = JsonInterpolate(extension->find("interpolate"));
      if (!parsed.has_value()) {
        return Error(400, "Bad Request",
                     "'gufo.interpolate' must be 1..8 or {\"fps\": 1..240}",
                     "invalid_interpolate");
      }
      interpolate = *parsed;
    }
    if (extension->contains("upscale")) {
      const auto parsed = JsonUpscale(extension->find("upscale"));
      if (!parsed.has_value()) {
        return Error(400, "Bad Request",
                     "'gufo.upscale' must be \"WxH\" or {\"width\", "
                     "\"height\", \"fit\", \"model\"}",
                     "invalid_upscale");
      }
      upscale = *parsed;
    }
    if (extension->contains("keyframes")) {
      guide_keyframes = extension->find("keyframes");
      if (!guide_keyframes->is_array()) {
        return Error(400, "Bad Request",
                     "'gufo.keyframes' must be an array of {\"frame\", "
                     "\"image\"} objects",
                     "invalid_keyframes");
      }
    }
    if (extension->contains("audio_reference_frame")) {
      const auto parsed =
          JsonSignedFrame(extension->find("audio_reference_frame"));
      if (!parsed.has_value()) {
        return Error(400, "Bad Request",
                     "'gufo.audio_reference_frame' must be an integer",
                     "invalid_audio_reference");
      }
      audio_reference_frame = *parsed;
    }
    if (extension->contains("selected_frame")) {
      selected_frame = JsonFrame(extension->find("selected_frame"));
      if (!selected_frame.has_value()) {
        return Error(400, "Bad Request",
                     "'gufo.selected_frame' must be a non-negative integer",
                     "invalid_selected_frame");
      }
    }
  }
  if (preset_explicit && !model_preset.empty() && preset != model_preset) {
    return Error(400, "Bad Request",
                 "model alias and 'gufo.preset' must select the same preset",
                 "preset_model_mismatch");
  }

  // "exact" at any other accepted canvas (e.g. FilmStudio's 832x512) is
  // the "exact-WxH" preset; the fixed presets keep their own sizes.
  if (preset == "exact" && size != "512x512" && size != "256x256") {
    if (!minimax_h3::ParseCanvasSize(size).has_value()) {
      return Error(400, "Bad Request",
                   "'size' must be WxH with multiples of 32, 256..1344 per "
                   "side and at most 768*1344 pixels",
                   "invalid_size");
    }
    preset = "exact-" + size;
  }
  std::string preset_error;
  auto parameters = minimax_h3::ResolveGenerationPreset(preset, &preset_error);
  if (!parameters.has_value()) {
    return Error(400, "Bad Request", preset_error, "invalid_preset");
  }
  parameters->frames = frames.value_or(*duration_frames);
  if (steps.has_value()) {
    parameters->evaluations = *steps;
  }

  // FL2VA keyframes: `input_reference` (OpenAI) and `first_frame` are the
  // same first-frame anchor; `last_frame` anchors the end.
  std::vector<minimax_h3::GenerationKeyframe> keyframes;
  std::string guide_error;
  {
    core::ImageReadBudget budget;
    const auto field = [&](const char* name) -> bool {
      return body.contains(name) || files.contains(name);
    };
    if (field("input_reference") && field("first_frame")) {
      return Error(400, "Bad Request",
                   "'input_reference' and 'first_frame' both set the first "
                   "frame",
                   "invalid_input_reference");
    }
    for (const auto& [name, anchor] :
         {std::pair{"input_reference", minimax_h3::KeyframeAnchor::kFirst},
          std::pair{"first_frame", minimax_h3::KeyframeAnchor::kFirst},
          std::pair{"last_frame", minimax_h3::KeyframeAnchor::kLast}}) {
      if (!field(name)) {
        continue;
      }
      const auto file = files.find(name);
      std::string image_error;
      auto keyframe = ReadKeyframe(
          body.find(name), file == files.end() ? nullptr : &file->second,
          anchor, budget, &image_error);
      if (!keyframe.has_value()) {
        return Error(400, "Bad Request",
                     "'" + std::string(name) + "' " + image_error,
                     "invalid_input_reference");
      }
      keyframes.push_back(std::move(*keyframe));
    }
    // Guide frames (MiniMaxH3AddGuide image): "gufo.keyframes" entries
    // {"frame": N, "image": <image>} and multipart files "keyframe_<N>".
    // Negative frames count from the end (-1 = last frame).
    const auto add_guide = [&](int frame, const json::Value* value,
                               const std::string* file,
                               const std::string& label) -> bool {
      std::string image_error;
      auto keyframe =
          ReadKeyframe(value, file, minimax_h3::KeyframeAnchor::kFirst, budget,
                       &image_error);
      if (!keyframe.has_value()) {
        guide_error = "'" + label + "' " + image_error;
        return false;
      }
      keyframe->frame = frame;
      keyframe->vision = false;
      keyframes.push_back(std::move(*keyframe));
      return true;
    };
    if (guide_keyframes != nullptr) {
      std::size_t index = 0;
      for (const json::Value& entry : guide_keyframes->items()) {
        const std::string label =
            "gufo.keyframes[" + std::to_string(index++) + "]";
        const auto frame = entry.is_object()
                               ? JsonSignedFrame(entry.find("frame"))
                               : std::nullopt;
        if (!frame.has_value() || !HasOnlyMembers(entry, {"frame", "image"})) {
          return Error(400, "Bad Request",
                       "'" + label +
                           "' must be {\"frame\": integer, \"image\": "
                           "image}",
                       "invalid_keyframes");
        }
        if (!add_guide(*frame, entry.find("image"), nullptr, label)) {
          return Error(400, "Bad Request", guide_error, "invalid_keyframes");
        }
      }
    }
    for (const auto& [name, content] : files) {
      if (!name.starts_with("keyframe_")) {
        continue;
      }
      int frame = 0;
      const std::string_view digits = std::string_view(name).substr(9);
      const auto parsed =
          std::from_chars(digits.data(), digits.data() + digits.size(), frame);
      if (digits.empty() || parsed.ec != std::errc{} ||
          parsed.ptr != digits.data() + digits.size()) {
        return Error(400, "Bad Request",
                     "multipart guide frames are named keyframe_<frame>",
                     "invalid_keyframes");
      }
      if (!add_guide(frame, nullptr, &content, name)) {
        return Error(400, "Bad Request", guide_error, "invalid_keyframes");
      }
    }
  }
  // One image per pixel frame, inside the video.
  {
    std::vector<int> used;
    for (const minimax_h3::GenerationKeyframe& keyframe : keyframes) {
      int frame = keyframe.frame == minimax_h3::kKeyframeAtAnchor
                      ? minimax_h3::KeyframeAnchorFrame(keyframe.anchor,
                                                        parameters->frames)
                      : keyframe.frame;
      if (frame < 0) {
        frame += parameters->frames;
      }
      if (frame < 0 || frame >= parameters->frames) {
        return Error(400, "Bad Request",
                     "guide frame " + std::to_string(keyframe.frame) +
                         " is outside the video's " +
                         std::to_string(parameters->frames) + " frames",
                     "invalid_keyframes");
      }
      if (std::ranges::find(used, frame) != used.end()) {
        return Error(400, "Bad Request",
                     "two images anchor frame " + std::to_string(frame),
                     "invalid_keyframes");
      }
      used.push_back(frame);
    }
    if (keyframes.size() > minimax_h3::kH3MaximumKeyframes) {
      return Error(400, "Bad Request",
                   "at most " +
                       std::to_string(minimax_h3::kH3MaximumKeyframes) +
                       " keyframes and guide frames",
                   "invalid_keyframes");
    }
  }
  // Audio anchor (MiniMaxH3AddGuide audio): "audio_reference" as a base64
  // (data URL) string or a multipart file, at "gufo.audio_reference_frame".
  std::vector<minimax_h3::GenerationAudioGuide> audio_guides;
  if (body.contains("audio_reference") || files.contains("audio_reference")) {
    const auto file = files.find("audio_reference");
    std::string audio_error;
    auto waveform = ReadGuideAudio(
        body.find("audio_reference"),
        file == files.end() ? nullptr : &file->second, &audio_error);
    if (!waveform.has_value()) {
      return Error(400, "Bad Request", "'audio_reference' " + audio_error,
                   "invalid_audio_reference");
    }
    int frame = audio_reference_frame;
    if (frame < 0) {
      frame += parameters->frames;
    }
    if (frame < 0 || frame >= parameters->frames) {
      return Error(400, "Bad Request",
                   "'gufo.audio_reference_frame' is outside the video",
                   "invalid_audio_reference");
    }
    audio_guides.push_back({.frame = frame, .waveform = std::move(*waveform)});
  }
  if (output_format == "ppm") {
    if (parameters->preset != "development-256" || size != "256x256") {
      return Error(400, "Bad Request",
                   "PPM output requires preset 'dev' and size '256x256'",
                   "invalid_output_format");
    }
    parameters->selected_frames = {
        selected_frame.value_or(parameters->frames / 2)};
  } else if (output_format == "mp4") {
    if (!parameters->mux || selected_frame.has_value()) {
      return Error(400, "Bad Request",
                   "MP4 output requires a supported preset, duration, and "
                   "resolution combination",
                   "invalid_output_format");
    }
  } else {
    return Error(400, "Bad Request",
                 "'gufo.output_format' must be 'mp4' or 'ppm'",
                 "invalid_output_format");
  }
  std::string parameter_error;
  if (!minimax_h3::ValidateGenerationParameters(*parameters,
                                                &parameter_error)) {
    return Error(400, "Bad Request", parameter_error, "invalid_frames");
  }
  if (parameters->frames != *duration_frames) {
    return Error(400, "Bad Request",
                 "'gufo.frames' must match the aligned 'seconds' duration",
                 "invalid_frames");
  }

  const VideoJobCreateResult created = service.Create({
      .model = model,
      .prompt = prompt,
      .size = size,
      .seconds = seconds,
      .output_format = output_format,
      .parameters = std::move(*parameters),
      .seed = seed,
      .keyframes = std::move(keyframes),
      .audio_guides = std::move(audio_guides),
      .lora = lora,
      .lora_strength = lora_strength,
      .steps_explicit = steps.has_value(),
      .interpolate = interpolate.multiplier,
      .interpolate_fps_num = interpolate.fps_num,
      .interpolate_fps_den = interpolate.fps_den,
      .upscale = std::move(upscale),
      .preview = preview,
  });
  if (created.result == VideoJobResult::kQueueFull) {
    HttpResponse response =
        Error(429, "Too Many Requests", created.error, "video_queue_full");
    response.headers.emplace_back("Retry-After", "1");
    return response;
  }
  if (created.result != VideoJobResult::kOk || !created.job.has_value()) {
    const int status = created.result == VideoJobResult::kIoError ? 503 : 400;
    return Error(
        status, status == 503 ? "Service Unavailable" : "Bad Request",
        created.error,
        status == 503 ? "video_service_unavailable" : "invalid_video_request");
  }
  HttpResponse response = SnapshotResponse(202, "Accepted", *created.job);
  response.headers.emplace_back("Location", "/v1/videos/" + created.job->id);
  return response;
}

HttpResponse LookupVideo(std::string_view id, VideoJobService& service) {
  const VideoJobLookupResult lookup = service.Get(id);
  if (lookup.result != VideoJobResult::kOk || !lookup.job.has_value()) {
    return Error(404, "Not Found", "video job not found", "video_not_found");
  }
  return SnapshotResponse(200, "OK", *lookup.job);
}

struct ByteRange {
  std::uint64_t first{0};
  std::uint64_t last{0};
};

bool ParseUnsigned(std::string_view text, std::uint64_t* result) {
  if (text.empty() || result == nullptr) {
    return false;
  }
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, *result);
  return parsed.ec == std::errc() && parsed.ptr == end;
}

std::optional<ByteRange> ParseRange(std::string_view header,
                                    std::uint64_t bytes) {
  if (!header.starts_with("bytes=") || bytes == 0) {
    return std::nullopt;
  }
  const std::string_view value = header.substr(6);
  if (value.find(',') != std::string_view::npos) {
    return std::nullopt;
  }
  const std::size_t dash = value.find('-');
  if (dash == std::string_view::npos) {
    return std::nullopt;
  }
  const std::string_view first_text = value.substr(0, dash);
  const std::string_view last_text = value.substr(dash + 1);
  if (first_text.empty()) {
    std::uint64_t suffix = 0;
    if (!ParseUnsigned(last_text, &suffix) || suffix == 0) {
      return std::nullopt;
    }
    suffix = std::min(suffix, bytes);
    return ByteRange{.first = bytes - suffix, .last = bytes - 1};
  }
  std::uint64_t first = 0;
  if (!ParseUnsigned(first_text, &first) || first >= bytes) {
    return std::nullopt;
  }
  std::uint64_t last = bytes - 1;
  if (!last_text.empty() &&
      (!ParseUnsigned(last_text, &last) || last < first)) {
    return std::nullopt;
  }
  return ByteRange{.first = first, .last = std::min(last, bytes - 1)};
}

HttpResponse VideoContent(std::string_view id, const HttpRequest& request,
                          VideoJobService& service) {
  const VideoJobContentResult result = service.Content(id);
  if (result.result == VideoJobResult::kNotFound) {
    return Error(404, "Not Found", "video job not found", "video_not_found");
  }
  if (result.result == VideoJobResult::kNotReady) {
    return Error(409, "Conflict", "video content is not ready",
                 "video_not_ready");
  }
  if (result.result != VideoJobResult::kOk || !result.content.has_value()) {
    return Error(503, "Service Unavailable", "video content is unavailable",
                 "video_content_unavailable");
  }

  const VideoJobContent& content = *result.content;
  const std::string range_header = request.header("range");
  ByteRange range{
      .first = 0,
      .last = content.bytes == 0 ? 0 : content.bytes - 1,
  };
  bool partial = false;
  if (!range_header.empty()) {
    const auto parsed = ParseRange(range_header, content.bytes);
    if (!parsed.has_value()) {
      HttpResponse response = Error(416, "Range Not Satisfiable",
                                    "invalid byte range", "invalid_range");
      response.headers.emplace_back("Content-Range",
                                    "bytes */" + std::to_string(content.bytes));
      response.headers.emplace_back("Accept-Ranges", "bytes");
      return response;
    }
    range = *parsed;
    partial = true;
  }

  const std::uint64_t count =
      content.bytes == 0 ? 0 : range.last - range.first + 1;
  if (count >
      static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    return Error(503, "Service Unavailable", "video content is too large",
                 "video_content_unavailable");
  }
  std::ifstream input(content.path, std::ios::binary);
  input.seekg(static_cast<std::streamoff>(range.first));
  std::string body(static_cast<std::size_t>(count), '\0');
  input.read(body.data(), static_cast<std::streamsize>(count));
  if (!input && input.gcount() != static_cast<std::streamsize>(count)) {
    return Error(503, "Service Unavailable", "video content is unavailable",
                 "video_content_unavailable");
  }

  HttpResponse response{
      .status = partial ? 206 : 200,
      .reason = partial ? "Partial Content" : "OK",
      .body = std::move(body),
      .headers =
          {
              {"Content-Type", content.media_type},
              {"Accept-Ranges", "bytes"},
              {"Content-Disposition",
               "attachment; filename=\"" + std::string(id) +
                   (content.media_type == "video/mp4" ? ".mp4\"" : ".ppm\"")},
          },
      .streaming_body = {},
  };
  if (partial) {
    response.headers.emplace_back("Content-Range",
                                  "bytes " + std::to_string(range.first) + "-" +
                                      std::to_string(range.last) + "/" +
                                      std::to_string(content.bytes));
  }
  return response;
}

HttpResponse DeleteVideo(std::string_view id, VideoJobService& service) {
  const VideoJobLookupResult deleted = service.Delete(id);
  if (deleted.result != VideoJobResult::kOk || !deleted.job.has_value()) {
    return Error(404, "Not Found", "video job not found", "video_not_found");
  }
  json::Value root = json::Value::object();
  root["id"] = deleted.job->id;
  root["object"] = "video.deleted";
  root["deleted"] = true;
  return {
      .status = 200,
      .reason = "OK",
      .body = root.dump(),
      .headers = {},
      .streaming_body = {},
  };
}

std::optional<std::pair<std::string_view, bool>> ParseVideoPath(
    std::string_view path) {
  constexpr std::string_view prefix = "/v1/videos/";
  if (!path.starts_with(prefix)) {
    return std::nullopt;
  }
  std::string_view remainder = path.substr(prefix.size());
  bool content = false;
  constexpr std::string_view kContentSuffix = "/content";
  if (remainder.ends_with(kContentSuffix)) {
    remainder.remove_suffix(kContentSuffix.size());
    content = true;
  }
  if (remainder.empty() || remainder.find('/') != std::string_view::npos) {
    return std::nullopt;
  }
  return std::pair{remainder, content};
}

}  // namespace

bool IsVideoApiPath(std::string_view path) noexcept {
  return path == kVideosPath || path.starts_with("/v1/videos/");
}

HttpResponse HandleVideoApiRequest(const HttpRequest& request,
                                   VideoJobService& service) {
  if (request.path == kVideosPath) {
    if (request.method == "POST") {
      return CreateVideo(request, service);
    }
    return Error(405, "Method Not Allowed", "method is not allowed",
                 "method_not_allowed");
  }

  const auto parsed = ParseVideoPath(request.path);
  if (!parsed.has_value()) {
    return Error(404, "Not Found", "video job not found", "video_not_found");
  }
  const auto [id, content] = *parsed;
  if (content) {
    if (request.method == "GET") {
      return VideoContent(id, request, service);
    }
    return Error(405, "Method Not Allowed", "method is not allowed",
                 "method_not_allowed");
  }
  if (request.method == "GET") {
    return LookupVideo(id, service);
  }
  if (request.method == "DELETE") {
    return DeleteVideo(id, service);
  }
  return Error(405, "Method Not Allowed", "method is not allowed",
               "method_not_allowed");
}

}  // namespace gufo::server
