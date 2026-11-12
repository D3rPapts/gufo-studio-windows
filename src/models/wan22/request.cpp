// Written with AI assistance (Claude Code).
// Request parsing ported from the 1.x fork's src/cli/serve/video_api.cpp
// (CreateWan22Video, CreateWan21Video, CreateWan21VaceVideo,
// CreateTalkVideo) and the validators from video_jobs.cpp. The HTTP formats
// are unchanged; what used to be fields of the fork's VideoJobRequest now
// travels as `options_json` plus attachments of the 0.2 video API.
#include "src/models/wan22/request.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <exception>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <gufo/ext/image.hpp>

namespace gufo::models::wan22 {
namespace {

constexpr double kMaximumExactJsonInteger = 9007199254740991.0;

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

bool Fail(ext::VideoRequestError* error, std::string code,
          std::string message) {
  if (error != nullptr) {
    error->status = 400;
    error->code = std::move(code);
    error->message = std::move(message);
  }
  return false;
}

bool HasOnlyMembers(const json::Value& value,
                    std::initializer_list<std::string_view> allowed) {
  return std::ranges::all_of(value.members(), [&](const auto& member) {
    return std::ranges::find(allowed, member.first) != allowed.end();
  });
}

std::optional<std::uint64_t> JsonUnsigned(const json::Value* value) {
  if (value == nullptr || !value->is_number())
    return std::nullopt;
  const double number = value->as_double(-1.0);
  if (!std::isfinite(number) || number < 0.0 ||
      number > kMaximumExactJsonInteger || std::floor(number) != number)
    return std::nullopt;
  return static_cast<std::uint64_t>(number);
}

std::optional<int> JsonFrame(const json::Value* value) {
  const auto parsed = JsonUnsigned(value);
  if (!parsed.has_value() ||
      *parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    return std::nullopt;
  return static_cast<int>(*parsed);
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

// A data URL ("data:...;base64,...") or raw base64 string; false when the
// text is not base64 or decodes to fewer than `minimum` bytes.
bool DecodeMediaString(std::string_view text, std::size_t minimum,
                       std::vector<std::uint8_t>* out, bool* bad_data_url) {
  *bad_data_url = false;
  if (text.starts_with("data:")) {
    const auto comma = text.find(',');
    if (comma == std::string_view::npos ||
        text.substr(0, comma).find(";base64") == std::string_view::npos) {
      *bad_data_url = true;
      return false;
    }
    text.remove_prefix(comma + 1);
  }
  auto decoded = DecodeBase64(text);
  if (!decoded.has_value() || decoded->size() < minimum)
    return false;
  *out = std::move(*decoded);
  return true;
}

std::string SecondsText(const json::Value* value, std::string fallback) {
  if (value != nullptr && value->is_string())
    return value->str();
  if (value != nullptr && value->is_number())
    return value->dump();
  return fallback;
}

std::optional<double> ParseSeconds(std::string_view seconds) {
  double value = 0;
  const auto parsed =
      std::from_chars(seconds.data(), seconds.data() + seconds.size(), value);
  if (parsed.ec != std::errc() ||
      parsed.ptr != seconds.data() + seconds.size() || !std::isfinite(value))
    return std::nullopt;
  return value;
}

core::Image Decode(std::span<const std::uint8_t> bytes) {
  return core::DecodeImage(bytes);
}

// An image: an uploaded file, a base64 data URL, raw base64, a public HTTP(S)
// URL, or an object {"image_url": <string or {"url": string}>}. Returns the
// encoded bytes (what the runner decodes again) and checks that they decode.
bool ReadImage(const json::Value* value, const ext::VideoAttachment* file,
               core::ImageReadBudget& budget, std::vector<std::uint8_t>* bytes,
               std::string* error) {
  try {
    if (file != nullptr) {
      if (file->bytes.size() > budget.remaining_bytes)
        throw std::invalid_argument("images exceed the size limit");
      budget.remaining_bytes -= file->bytes.size();
      *bytes = file->bytes;
    } else {
      const json::Value* source = value;
      if (source != nullptr && source->is_object()) {
        source = source->find("image_url");
        if (source != nullptr && source->is_object())
          source = source->find("url");
      }
      if (source == nullptr || !source->is_string() || source->str().empty())
        throw std::invalid_argument(
            "must be an image file, a data URL, base64, an HTTPS URL or "
            "{\"image_url\": ...}");
      const std::string& text = source->str();
      *bytes = text.starts_with("data:") || text.starts_with("https://") ||
                       text.starts_with("http://")
                   ? core::ReadImageUrl(text, budget)
                   : core::ReadImageUrl("data:image/png;base64," + text,
                                        budget);
    }
    (void)Decode(*bytes);
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

// Speech: an uploaded file or a base64 (data URL) string of WAV, FLAC, Ogg
// or MP3; decoded here to check it and to measure its length.
bool ReadAudio(const json::Value* value, const ext::VideoAttachment* file,
               std::vector<std::uint8_t>* bytes, AudioWaveform* waveform,
               std::string* error) {
  if (file != nullptr) {
    *bytes = file->bytes;
  } else {
    if (value == nullptr || !value->is_string() || value->str().empty()) {
      SetError(error, "must be an audio file or a base64 (data URL) string");
      return false;
    }
    bool bad_data_url = false;
    if (!DecodeMediaString(value->str(), 1, bytes, &bad_data_url) ||
        bytes->size() > kMaximumSpeechBytes) {
      SetError(error, bad_data_url ? "data URL must be base64 encoded"
                                   : "is not valid base64 or exceeds 32 MiB");
      return false;
    }
  }
  return DecodeSpeech(*bytes, waveform, error);
}

// The request as the provider sees it: JSON members without the multipart
// file parts (the core lists those as empty strings too), plus the files.
struct Body {
  json::Value json = json::Value::object();
  std::map<std::string, const ext::VideoAttachment*, std::less<>> files;

  const ext::VideoAttachment* file(std::string_view name) const {
    const auto found = files.find(name);
    return found == files.end() ? nullptr : found->second;
  }
  bool has(std::string_view name) const {
    return json.contains(std::string(name)) || files.contains(name);
  }
};

bool ParseBody(const ext::VideoRequestInput& input, Body* body,
               ext::VideoRequestError* error) {
  json::Value parsed;
  try {
    parsed = json::parse(std::string(input.body_json));
  } catch (const std::exception&) {
    return Fail(error, "parse_error", "request body must be valid JSON");
  }
  if (!parsed.is_object())
    return Fail(error, "invalid_body", "request body must be a JSON object");
  for (const auto& file : input.files)
    body->files.emplace(file.name, &file);
  for (const auto& [key, value] : parsed.members())
    if (!body->files.contains(key))
      body->json[key] = value;
  return true;
}

// "gufo.preview" (checked here; the host renders the job's live previews).
// "gufo.interpolate" and "gufo.upscale" never reach the provider (see below).
bool CheckPostSteps(const json::Value& extension,
                    ext::VideoRequestError* error) {
  if (const json::Value* preview = extension.find("preview")) {
    if (!preview->is_object() ||
        !HasOnlyMembers(*preview, {"always", "every", "max_side"}))
      return Fail(error, "invalid_preview",
                  "'gufo.preview' may contain only 'always', 'every' and "
                  "'max_side' (the preview id is the job id)");
    const json::Value* always = preview->find("always");
    const json::Value* every = preview->find("every");
    const json::Value* side = preview->find("max_side");
    const auto every_value = JsonFrame(every);
    const auto side_value = JsonFrame(side);
    if ((always != nullptr && !always->is_bool()) ||
        (every != nullptr && (!every_value || *every_value < 1)) ||
        (side != nullptr && (!side_value || *side_value < 16)))
      return Fail(error, "invalid_preview",
                  "'gufo.preview' needs a boolean 'always', 'every' >= 1 and "
                  "'max_side' >= 16");
  }
  // "interpolate" / "upscale" are post-steps of the server
  // (src/cli/serve/media/video_postprocess): it takes them out of "gufo"
  // before this provider sees the request and runs them on the finished MP4.
  return true;
}

// The parts every family shares: "model" (checked by the server), "prompt",
// "size", "seconds".
struct Common {
  std::string prompt;
  std::string size;
};

bool ParseCommon(const Body& body, std::string_view default_size,
                 bool prompt_required, Common* common,
                 ext::VideoRequestError* error) {
  const json::Value* prompt_value = body.json.find("prompt");
  const json::Value* size_value = body.json.find("size");
  if (prompt_value == nullptr || !prompt_value->is_string() ||
      (size_value != nullptr && !size_value->is_string()))
    return Fail(error, "invalid_parameter_type",
                "'model', 'prompt', and optional 'size' must be strings");
  common->prompt = prompt_value->str();
  if (common->prompt.find('\0') != std::string::npos)
    return Fail(error, "invalid_prompt", "'prompt' must not contain NUL");
  if (prompt_required && common->prompt.empty())
    return Fail(error, "missing_required_parameter", "'prompt' is required");
  common->size = body.json.member_str("size", std::string(default_size));
  return true;
}

bool InitParameters(Family family, const Common& common, int steps,
                    ext::VideoParameters* parameters,
                    ext::VideoRequestError* error) {
  parameters->preset = std::string(Preset(family));
  parameters->evaluations = steps;
  parameters->mux = true;
  parameters->decode_audio = false;
  if (!ParseVideoSize(common.size, &parameters->output_width,
                      &parameters->output_height))
    return Fail(error, "invalid_size", "'size' must be WIDTHxHEIGHT");
  parameters->internal_width = parameters->output_width;
  parameters->internal_height = parameters->output_height;
  return true;
}

bool ParseSeedSteps(const json::Value& extension, const char* steps_range,
                    std::uint64_t* seed, ext::VideoParameters* parameters,
                    ext::VideoRequestError* error) {
  if (extension.contains("seed")) {
    const auto parsed = JsonUnsigned(extension.find("seed"));
    if (!parsed.has_value())
      return Fail(error, "invalid_seed",
                  "'gufo.seed' must be a non-negative exact integer");
    *seed = *parsed;
  }
  if (extension.contains("steps")) {
    const auto parsed = JsonFrame(extension.find("steps"));
    if (!parsed.has_value())
      return Fail(error, "invalid_steps",
                  std::string("'gufo.steps' must be ") + steps_range);
    parameters->evaluations = *parsed;
  }
  return true;
}

bool CheckFrames(const json::Value& extension, int duration_frames,
                 ext::VideoRequestError* error) {
  if (extension.contains("frames")) {
    const auto parsed = JsonFrame(extension.find("frames"));
    if (!parsed.has_value() || *parsed != duration_frames)
      return Fail(error, "invalid_frames",
                  "'gufo.frames' must match the 'seconds' duration");
  }
  return true;
}

bool ReadNegativePrompt(const json::Value& extension, json::Value* options,
                        ext::VideoRequestError* error) {
  if (const json::Value* value = extension.find("negative_prompt")) {
    if (!value->is_string() || value->str().find('\0') != std::string::npos)
      return Fail(error, "invalid_negative_prompt",
                  "'gufo.negative_prompt' must be a string");
    (*options)["negative_prompt"] = value->str();
  }
  return true;
}

// Uploads a family accepts (multipart file fields).
bool CheckUploads(const Body& body, Family family,
                  ext::VideoRequestError* error) {
  for (const auto& [name, file] : body.files) {
    (void)file;
    bool allowed = false;
    switch (family) {
      case Family::kWan22:
        allowed = name == "input_reference" || name == "first_frame";
        break;
      case Family::kWan21Uni3C:
        allowed = name == "input_reference" || name == "first_frame" ||
                  name == "render_video";
        break;
      case Family::kWan21Vace:
        allowed = name == "control_video" || name == "mask" ||
                  name.starts_with("reference_image");
        break;
      case Family::kWan21Talk:
        allowed = name == "input_reference" || name == "first_frame" ||
                  name == "audio_1" || name == "audio_2" ||
                  name == "mask_1" || name == "mask_2" ||
                  name == "talk_video";
        break;
    }
    if (!allowed)
      return Fail(error,
                  name == "last_frame" ? "unsupported_input_reference"
                                       : "unsupported_field",
                  "unsupported upload '" + name + "'");
  }
  return true;
}

// "input_reference" or "first_frame" (JSON or upload) as the attachment
// "first_frame". `required`: exactly one must be present.
bool ReadFirstFrame(const Body& body, bool required,
                    ext::VideoPreparedRequest* prepared,
                    ext::VideoRequestError* error) {
  const bool reference = body.has("input_reference");
  const bool first = body.has("first_frame");
  if (required && reference == first)
    return Fail(error, "invalid_input_reference",
                "exactly one of 'first_frame' and 'input_reference' is "
                "required");
  if (reference && first)
    return Fail(error, "invalid_input_reference",
                "'input_reference' and 'first_frame' both set the first frame");
  for (const char* name : {"input_reference", "first_frame"}) {
    if (!body.has(name))
      continue;
    core::ImageReadBudget budget;
    std::vector<std::uint8_t> bytes;
    std::string image_error;
    if (!ReadImage(body.json.find(name), body.file(name), budget, &bytes,
                   &image_error))
      return Fail(error, "invalid_input_reference",
                  "'" + std::string(name) + "' " + image_error);
    prepared->attachments.push_back(
        {.name = "first_frame", .bytes = std::move(bytes)});
  }
  return true;
}

// Wan 2.2 TI2V-5B: {"model", "prompt", "size" (default "1280x704"),
// "seconds" (0.2..5 at 24 fps), "input_reference" or "first_frame" (image-
// to-video; the image is scaled to cover the size and centre-cropped),
// "gufo": {"seed", "steps" (default 50), "frames" (4k + 1, must match
// seconds), "negative_prompt" (default ""), "guidance" (default 5; <= 1
// disables classifier-free guidance), "preview"}}. MP4 only.
bool PrepareWan22(const Body& body, ext::VideoPreparedRequest* prepared,
                  ext::VideoRequestError* error) {
  if (body.has("last_frame") || body.json.contains("references"))
    return Fail(error, "unsupported_input_reference",
                "Wan 2.2 supports only a first frame ('input_reference' or "
                "'first_frame')");
  if (!HasOnlyMembers(body.json, {"model", "prompt", "size", "seconds", "gufo",
                                  "input_reference", "first_frame"}))
    return Fail(error, "unsupported_field",
                "request contains an unsupported video field");
  if (!CheckUploads(body, Family::kWan22, error) ||
      !ReadFirstFrame(body, false, prepared, error))
    return false;
  Common common;
  if (!ParseCommon(body, "1280x704", true, &common, error))
    return false;
  const std::string seconds = SecondsText(body.json.find("seconds"), "5");
  std::string duration_error;
  const auto frames =
      ResolveDurationFrames(Family::kWan22, seconds, &duration_error);
  if (!frames)
    return Fail(error, "invalid_seconds", duration_error);
  ext::VideoParameters parameters;
  if (!InitParameters(Family::kWan22, common, 50, &parameters, error))
    return false;
  parameters.frames = *frames;
  std::uint64_t seed = 42;
  json::Value options = json::Value::object();
  options["guidance"] = 5.0;
  if (const json::Value* extension = body.json.find("gufo")) {
    if (!extension->is_object() ||
        !HasOnlyMembers(*extension,
                        {"seed", "steps", "frames", "negative_prompt",
                         "guidance", "interpolate", "upscale", "preview"}))
      return Fail(error, "invalid_strix_options",
                  "'gufo' may contain only 'seed', 'steps', 'frames', "
                  "'negative_prompt', 'guidance', 'interpolate', "
                  "'upscale' and 'preview'");
    if (!CheckPostSteps(*extension, error) ||
        !ReadNegativePrompt(*extension, &options, error))
      return false;
    if (const json::Value* value = extension->find("guidance")) {
      const double guidance = value->is_number() ? value->as_double(-1.0) : -1;
      if (!std::isfinite(guidance) || guidance < 0.0 || guidance > 20.0)
        return Fail(error, "invalid_guidance",
                    "'gufo.guidance' must be a number in 0..20");
      options["guidance"] = guidance;
    }
    if (!ParseSeedSteps(*extension, "1..100", &seed, &parameters, error) ||
        !CheckFrames(*extension, *frames, error))
      return false;
  }
  std::string parameter_error;
  if (!ValidateParameters(Family::kWan22, parameters, &parameter_error))
    return Fail(error, "invalid_parameters", parameter_error);
  prepared->prompt = std::move(common.prompt);
  prepared->size = std::move(common.size);
  prepared->seconds = seconds;
  prepared->seed = seed;
  prepared->parameters = std::move(parameters);
  prepared->options_json = options.dump();
  return true;
}

// Wan 2.1 I2V 14B + Uni3C ("wan2.1-uni3c"): {"model", "prompt", "size"
// (default "832x480", multiples of 16, at most 896x512 pixels), "seconds"
// (0.25..5 at 16 fps, default 5), "first_frame" or "input_reference"
// (required), "gufo": {"seed", "steps" (default 4), "frames", "preview",
// "camera": {"type", "amount", "speed", "ease"} | {"poses": [...]} |
// {"render_video": MP4 data URL or base64}}}; a multipart upload may carry
// the guide as a "render_video" file instead. Without a camera it is plain
// image-to-video.
bool PrepareWan21(const Body& body, ext::VideoPreparedRequest* prepared,
                  ext::VideoRequestError* error) {
  if (body.has("last_frame") || body.json.contains("references"))
    return Fail(error, "unsupported_input_reference",
                "Wan 2.1 supports only a first frame ('first_frame' or "
                "'input_reference')");
  if (!HasOnlyMembers(body.json,
                      {"model", "prompt", "size", "seconds", "gufo",
                       "input_reference", "first_frame", "render_video"}))
    return Fail(error, "unsupported_field",
                "request contains an unsupported video field");
  if (!CheckUploads(body, Family::kWan21Uni3C, error) ||
      !ReadFirstFrame(body, true, prepared, error))
    return false;
  Common common;
  if (!ParseCommon(body, "832x480", false, &common, error))
    return false;
  const std::string seconds = SecondsText(body.json.find("seconds"), "5");
  std::string duration_error;
  const auto frames =
      ResolveDurationFrames(Family::kWan21Uni3C, seconds, &duration_error);
  if (!frames)
    return Fail(error, "invalid_seconds", duration_error);
  ext::VideoParameters parameters;
  if (!InitParameters(Family::kWan21Uni3C, common, 4, &parameters, error))
    return false;
  parameters.frames = *frames;
  std::uint64_t seed = 42;
  json::Value options = json::Value::object();
  std::vector<std::uint8_t> guide_video;
  if (const auto* upload = body.file("render_video"))
    guide_video = upload->bytes;
  if (const json::Value* extension = body.json.find("gufo")) {
    if (!extension->is_object() ||
        !HasOnlyMembers(*extension, {"seed", "steps", "frames", "interpolate",
                                     "upscale", "camera", "preview"}))
      return Fail(error, "invalid_strix_options",
                  "'gufo' may contain only 'seed', 'steps', 'frames', "
                  "'interpolate', 'upscale', 'camera' and 'preview'");
    if (!CheckPostSteps(*extension, error) ||
        !ParseSeedSteps(*extension, "1..20", &seed, &parameters, error) ||
        !CheckFrames(*extension, *frames, error))
      return false;
    if (const json::Value* camera = extension->find("camera")) {
      if (!guide_video.empty())
        return Fail(error, "invalid_camera",
                    "'gufo.camera' and an uploaded 'render_video' exclude "
                    "each other");
      if (camera->is_object() && camera->contains("render_video")) {
        const json::Value* video = camera->find("render_video");
        if (camera->size() != 1 || !video->is_string())
          return Fail(error, "invalid_camera",
                      "'gufo.camera.render_video' must be the only camera "
                      "field and an MP4 data URL or base64 string");
        bool bad_data_url = false;
        if (!DecodeMediaString(video->str(), 64, &guide_video, &bad_data_url))
          return Fail(error, "invalid_camera",
                      bad_data_url
                          ? "'gufo.camera.render_video' data URL must be "
                            "base64"
                          : "'gufo.camera.render_video' is not valid base64 "
                            "video");
      } else {
        CameraSpec spec;
        std::string camera_error;
        if (!ParseCameraSpec(*camera, &spec, &camera_error))
          return Fail(error, "invalid_camera", camera_error);
        options["camera"] = *camera;
      }
    }
  }
  std::string parameter_error;
  if (!ValidateParameters(Family::kWan21Uni3C, parameters, &parameter_error))
    return Fail(error, "invalid_parameters", parameter_error);
  if (!guide_video.empty())
    prepared->attachments.push_back(
        {.name = "guide_video", .bytes = std::move(guide_video)});
  prepared->prompt = std::move(common.prompt);
  prepared->size = std::move(common.size);
  prepared->seconds = seconds;
  prepared->seed = seed;
  prepared->parameters = std::move(parameters);
  prepared->options_json = options.dump();
  return true;
}

// Wan 2.1 VACE 14B ("wan2.1-vace-14b"): {"model", "prompt", "size" (default
// "832x480", multiples of 16, at most 1280x720 pixels), "seconds" (0.25..5 at
// 16 fps, default 5), "gufo": {"seed", "steps" (default 4), "frames",
// "negative_prompt", "guidance" (1..20, default 1 = no CFG), "preview",
// "vace": {"control_video" (MP4 data URL or base64), "mask" (MP4, or a
// PNG/JPEG/WebP still for every frame; white = generate), "mask_channel"
// ("red" | "green" | "blue" | "luma"), "invert_mask", "reference_images" (up
// to 4 images: data URL, base64 or HTTPS), "strength" (0..1000, default 1),
// "sampler" ("lcm" | "euler"), "shift" (1..20), "lora" (false: without the
// bundle's step-distill LoRA)}}}. A multipart upload may carry
// "control_video", "mask" and "reference_image" / "reference_image_<n>"
// files instead. Without a control video the model sees grey
// (reference-to-video / text-to-video).
bool PrepareVace(const Body& body, ext::VideoPreparedRequest* prepared,
                 ext::VideoRequestError* error) {
  if (!HasOnlyMembers(body.json, {"model", "prompt", "size", "seconds", "gufo"}))
    return Fail(error, "unsupported_field",
                "request contains an unsupported video field (VACE inputs "
                "go into 'gufo.vace')");
  if (!CheckUploads(body, Family::kWan21Vace, error))
    return false;
  Common common;
  if (!ParseCommon(body, "832x480", false, &common, error))
    return false;
  const std::string seconds = SecondsText(body.json.find("seconds"), "5");
  std::string duration_error;
  const auto frames =
      ResolveDurationFrames(Family::kWan21Vace, seconds, &duration_error);
  if (!frames)
    return Fail(error, "invalid_seconds", duration_error);
  ext::VideoParameters parameters;
  if (!InitParameters(Family::kWan21Vace, common, 4, &parameters, error))
    return false;
  parameters.frames = *frames;
  std::uint64_t seed = 42;
  json::Value options = json::Value::object();
  options["guidance"] = 1.0;
  json::Value vace_options = json::Value::object();
  std::vector<std::uint8_t> control_video;
  std::vector<std::uint8_t> mask_file;
  core::ImageReadBudget image_budget;
  image_budget.remaining_bytes = 4 * core::kMaxEncodedImageBytes;

  // A video or still: an uploaded file, a data URL or raw base64.
  const auto media = [&](const json::Value* value, const char* name,
                         std::vector<std::uint8_t>* out,
                         std::string* message) {
    if (const auto* upload = body.file(name)) {
      if (value != nullptr) {
        *message = std::string("'gufo.vace.") + name + "' and an uploaded '" +
                   name + "' exclude each other";
        return false;
      }
      *out = upload->bytes;
      return true;
    }
    if (value == nullptr)
      return true;
    if (!value->is_string()) {
      *message = std::string("'gufo.vace.") + name +
                 "' must be a data URL or base64 string";
      return false;
    }
    bool bad_data_url = false;
    if (!DecodeMediaString(value->str(), 64, out, &bad_data_url)) {
      *message = std::string("'gufo.vace.") + name +
                 (bad_data_url ? "' data URL must be base64"
                               : "' is not valid base64");
      return false;
    }
    return true;
  };
  const json::Value* vace = nullptr;
  if (const json::Value* extension = body.json.find("gufo")) {
    if (!extension->is_object() ||
        !HasOnlyMembers(*extension,
                        {"seed", "steps", "frames", "negative_prompt",
                         "guidance", "interpolate", "upscale", "preview",
                         "vace"}))
      return Fail(error, "invalid_strix_options",
                  "'gufo' may contain only 'seed', 'steps', 'frames', "
                  "'negative_prompt', 'guidance', 'interpolate', 'upscale', "
                  "'preview' and 'vace'");
    if (!CheckPostSteps(*extension, error) ||
        !ReadNegativePrompt(*extension, &options, error))
      return false;
    if (const json::Value* value = extension->find("guidance")) {
      const double guidance = value->is_number() ? value->as_double(-1.0) : -1;
      if (!std::isfinite(guidance) || guidance < 1.0 || guidance > 20.0)
        return Fail(error, "invalid_guidance",
                    "'gufo.guidance' must be a number in 1..20 (1 = no "
                    "classifier-free guidance)");
      options["guidance"] = guidance;
    }
    if (!ParseSeedSteps(*extension, "1..50", &seed, &parameters, error) ||
        !CheckFrames(*extension, *frames, error))
      return false;
    vace = extension->find("vace");
  }
  if (vace != nullptr &&
      (!vace->is_object() ||
       !HasOnlyMembers(*vace, {"control_video", "mask", "mask_channel",
                               "invert_mask", "reference_images", "strength",
                               "sampler", "shift", "lora"})))
    return Fail(error, "invalid_vace",
                "'gufo.vace' may contain only 'control_video', 'mask', "
                "'mask_channel', 'invert_mask', 'reference_images', "
                "'strength', 'sampler', 'shift' and 'lora'");
  const auto member = [&](const char* name) -> const json::Value* {
    return vace == nullptr ? nullptr : vace->find(name);
  };
  std::string media_error;
  if (!media(member("control_video"), "control_video", &control_video,
             &media_error) ||
      !media(member("mask"), "mask", &mask_file, &media_error))
    return Fail(error, "invalid_vace", media_error);
  std::string mask_format;
  if (!mask_file.empty()) {
    const auto starts = [&](std::string_view magic) {
      if (mask_file.size() < magic.size())
        return false;
      for (std::size_t i = 0; i < magic.size(); ++i)
        if (static_cast<std::uint8_t>(magic[i]) != mask_file[i])
          return false;
      return true;
    };
    const bool still =
        starts("\x89PNG") || starts("\xFF\xD8\xFF") ||
        (starts("RIFF") && mask_file.size() > 12 && mask_file[8] == 'W' &&
         mask_file[9] == 'E' && mask_file[10] == 'B' && mask_file[11] == 'P');
    if (still) {
      // Decoded here (validated), handed on as a binary PPM.
      try {
        const core::Image image = Decode(mask_file);
        const std::string header = "P6\n" + std::to_string(image.width) + " " +
                                   std::to_string(image.height) + "\n255\n";
        mask_file.assign(header.begin(), header.end());
        mask_file.insert(mask_file.end(), image.pixels.begin(),
                         image.pixels.end());
        mask_format = "ppm";
      } catch (const std::exception& exception) {
        return Fail(error, "invalid_vace",
                    std::string("'gufo.vace.mask' ") + exception.what());
      }
    } else {
      mask_format = "mp4";
    }
  }
  if (const json::Value* channel = member("mask_channel")) {
    if (!channel->is_string() ||
        (channel->str() != "red" && channel->str() != "green" &&
         channel->str() != "blue" && channel->str() != "luma"))
      return Fail(error, "invalid_vace",
                  "'gufo.vace.mask_channel' must be red, green, blue or luma");
    vace_options["mask_channel"] = channel->str();
  }
  if (const json::Value* invert = member("invert_mask")) {
    if (!invert->is_bool())
      return Fail(error, "invalid_vace",
                  "'gufo.vace.invert_mask' must be a boolean");
    vace_options["invert_mask"] = invert->as_bool();
  }
  if (const json::Value* lora = member("lora")) {
    if (!lora->is_bool())
      return Fail(error, "invalid_vace", "'gufo.vace.lora' must be a boolean");
    vace_options["lora"] = lora->as_bool();
  }
  if (const json::Value* strength = member("strength")) {
    const double value = strength->is_number() ? strength->as_double(-1) : -1;
    if (!std::isfinite(value) || value < 0.0 || value > 1000.0)
      return Fail(error, "invalid_vace",
                  "'gufo.vace.strength' must be a number in 0..1000");
    vace_options["strength"] = value;
  }
  if (const json::Value* sampler = member("sampler")) {
    if (!sampler->is_string() ||
        (sampler->str() != "lcm" && sampler->str() != "euler"))
      return Fail(error, "invalid_vace",
                  "'gufo.vace.sampler' must be lcm or euler");
    vace_options["sampler"] = sampler->str();
  }
  if (const json::Value* shift = member("shift")) {
    const double value = shift->is_number() ? shift->as_double(-1) : -1;
    if (!std::isfinite(value) || value < 1.0 || value > 20.0)
      return Fail(error, "invalid_vace",
                  "'gufo.vace.shift' must be a number in 1..20");
    vace_options["shift"] = value;
  }
  // Reference images: JSON entries first, then uploads in name order.
  std::vector<std::vector<std::uint8_t>> references;
  if (const json::Value* list = member("reference_images")) {
    if (!list->is_array() || list->size() > 4)
      return Fail(error, "invalid_vace",
                  "'gufo.vace.reference_images' must be an array of at most "
                  "4 images");
    for (const json::Value& item : list->items()) {
      std::string image_error;
      std::vector<std::uint8_t> bytes;
      if (!ReadImage(&item, nullptr, image_budget, &bytes, &image_error))
        return Fail(error, "invalid_vace",
                    "'gufo.vace.reference_images' " + image_error);
      references.push_back(std::move(bytes));
    }
  }
  for (const auto& [name, file] : body.files) {
    if (!name.starts_with("reference_image"))
      continue;
    std::string image_error;
    std::vector<std::uint8_t> bytes;
    if (!ReadImage(nullptr, file, image_budget, &bytes, &image_error))
      return Fail(error, "invalid_vace", "'" + name + "' " + image_error);
    references.push_back(std::move(bytes));
  }
  if (references.size() > 4)
    return Fail(error, "invalid_vace", "at most 4 VACE reference images");
  std::string parameter_error;
  if (!ValidateParameters(Family::kWan21Vace, parameters, &parameter_error))
    return Fail(error, "invalid_parameters", parameter_error);
  options["vace"] = std::move(vace_options);
  if (!control_video.empty())
    prepared->attachments.push_back(
        {.name = "guide_video", .bytes = std::move(control_video)});
  if (!mask_file.empty()) {
    options["mask_format"] = mask_format;
    prepared->attachments.push_back(
        {.name = "mask", .bytes = std::move(mask_file)});
  }
  for (std::size_t i = 0; i < references.size(); ++i)
    prepared->attachments.push_back(
        {.name = "reference_image_" + std::to_string(i),
         .bytes = std::move(references[i])});
  prepared->prompt = std::move(common.prompt);
  prepared->size = std::move(common.size);
  prepared->seconds = seconds;
  prepared->seed = seed;
  prepared->parameters = std::move(parameters);
  prepared->options_json = options.dump();
  return true;
}

// Wan 2.1 I2V 14B + InfiniteTalk ("wan2.1-infinitetalk-14b"): {"model",
// "prompt", "size" (default "832x480", multiples of 16, at most 896x512
// pixels), "seconds" (optional, 0.04..60 at 25 fps; default: the audio
// length), "first_frame" or "input_reference" (required), "gufo": {"seed",
// "steps" (default 4), "preview", "talk": {"audio": [one or two audio files
// as data URLs / base64], "speaker_masks": [two images, bright = the
// speaker; required with two speakers], "audio_scale", "audio_mode" ("add":
// speaker 2 after speaker 1, "para": together, "auto": "para" when both
// tracks have the same length, else "add"), "motion_frames", "denoise"
// (0 < d <= 1, default 1), "video" (MP4 data URL or base64; with denoise < 1:
// video to video, the sampler starts from this video's frames so its motion
// stays and the audio drives the mouth)}}}. Multipart uploads may carry
// "audio_1", "audio_2", "mask_1", "mask_2" and "talk_video" files instead.
bool PrepareTalk(const Body& body, ext::VideoPreparedRequest* prepared,
                 ext::VideoRequestError* error) {
  if (body.has("last_frame") || body.json.contains("references"))
    return Fail(error, "unsupported_input_reference",
                "InfiniteTalk takes only a start image ('first_frame' or "
                "'input_reference')");
  if (!HasOnlyMembers(body.json, {"model", "prompt", "size", "seconds", "gufo",
                                  "input_reference", "first_frame"}))
    return Fail(error, "unsupported_field",
                "request contains an unsupported video field");
  if (!CheckUploads(body, Family::kWan21Talk, error) ||
      !ReadFirstFrame(body, true, prepared, error))
    return false;
  Common common;
  if (!ParseCommon(body, "832x480", false, &common, error))
    return false;
  ext::VideoParameters parameters;
  if (!InitParameters(Family::kWan21Talk, common, 4, &parameters, error))
    return false;
  std::uint64_t seed = 42;
  std::string talk_json = "{}";
  const json::Value* talk = nullptr;
  if (const json::Value* extension = body.json.find("gufo")) {
    if (!extension->is_object() ||
        !HasOnlyMembers(*extension, {"seed", "steps", "interpolate",
                                     "upscale", "preview", "talk"}))
      return Fail(error, "invalid_strix_options",
                  "'gufo' may contain only 'seed', 'steps', 'interpolate', "
                  "'upscale', 'preview' and 'talk'");
    if (!CheckPostSteps(*extension, error) ||
        !ParseSeedSteps(*extension, "1..20", &seed, &parameters, error))
      return false;
    talk = extension->find("talk");
  }
  if (talk != nullptr && !talk->is_object())
    return Fail(error, "invalid_talk", "'gufo.talk' must be an object");
  const json::Value* audio = talk != nullptr ? talk->find("audio") : nullptr;
  const json::Value* mask_values =
      talk != nullptr ? talk->find("speaker_masks") : nullptr;
  if (talk != nullptr) {
    json::Value talk_options = json::Value::object();
    for (const auto& [key, value] : talk->members())
      if (key != "audio" && key != "speaker_masks" && key != "video")
        talk_options[key] = value;
    std::string talk_error;
    if (!ParseTalkOptions(talk_options, &talk_json, &talk_error))
      return Fail(error, "invalid_talk", talk_error);
  }
  // Video to video: the input video ("gufo.talk.video" or a "talk_video"
  // upload), decoded by the worker; it needs denoise < 1.
  std::vector<std::uint8_t> talk_video;
  {
    const json::Value* video = talk != nullptr ? talk->find("video") : nullptr;
    const auto* upload = body.file("talk_video");
    if (video != nullptr && upload != nullptr)
      return Fail(error, "invalid_talk_video", "the talk video is given twice");
    if (upload != nullptr) {
      talk_video = upload->bytes;
    } else if (video != nullptr) {
      bool bad_data_url = false;
      if (!video->is_string() ||
          !DecodeMediaString(video->str(), 64, &talk_video, &bad_data_url))
        return Fail(error, "invalid_talk_video",
                    bad_data_url
                        ? "'gufo.talk.video' data URL must be base64"
                        : "'gufo.talk.video' must be an MP4 data URL or "
                          "base64");
    }
    const bool partial =
        json::parse(talk_json).member_double("denoise", 1.0) < 1.0;
    if (partial && talk_video.empty())
      return Fail(error, "invalid_talk_video",
                  "'gufo.talk.denoise' below 1 needs 'gufo.talk.video'");
    if (!partial && !talk_video.empty())
      return Fail(error, "invalid_talk_video",
                  "'gufo.talk.video' needs 'gufo.talk.denoise' below 1 (at 1 "
                  "the sampler starts from pure noise)");
  }
  if (audio != nullptr &&
      (!audio->is_array() || audio->empty() || audio->size() > 2))
    return Fail(error, "invalid_audio_reference",
                "'gufo.talk.audio' must be an array of one or two audio "
                "files");
  std::vector<AudioWaveform> speakers;
  std::vector<std::vector<std::uint8_t>> speaker_bytes;
  for (int index = 0; index < 2; ++index) {
    const std::string name = "audio_" + std::to_string(index + 1);
    const auto* file = body.file(name);
    const json::Value* value =
        audio != nullptr && static_cast<std::size_t>(index) < audio->size()
            ? &audio->items()[index]
            : nullptr;
    if (value == nullptr && file == nullptr)
      continue;
    if (value != nullptr && file != nullptr)
      return Fail(error, "invalid_audio_reference",
                  "speaker " + std::to_string(index + 1) +
                      " audio is given twice");
    std::string audio_error;
    std::vector<std::uint8_t> bytes;
    AudioWaveform waveform;
    if (!ReadAudio(value, file, &bytes, &waveform, &audio_error))
      return Fail(error, "invalid_audio_reference",
                  "speaker " + std::to_string(index + 1) + " audio " +
                      audio_error);
    if (static_cast<int>(speakers.size()) != index)
      return Fail(error, "invalid_audio_reference",
                  "speaker audio 2 needs speaker 1");
    speakers.push_back(std::move(waveform));
    speaker_bytes.push_back(std::move(bytes));
  }
  if (speakers.empty())
    return Fail(error, "missing_audio_reference",
                "InfiniteTalk needs the speech: 'gufo.talk.audio' or an "
                "'audio_1' upload");
  if (mask_values != nullptr &&
      (!mask_values->is_array() || mask_values->size() != 2))
    return Fail(error, "invalid_speaker_masks",
                "'gufo.talk.speaker_masks' must be an array of two images");
  std::vector<std::vector<std::uint8_t>> masks;
  for (int index = 0; index < 2; ++index) {
    const std::string name = "mask_" + std::to_string(index + 1);
    const auto* file = body.file(name);
    const json::Value* value =
        mask_values != nullptr ? &mask_values->items()[index] : nullptr;
    if (value == nullptr && file == nullptr)
      continue;
    core::ImageReadBudget budget;
    std::string image_error;
    std::vector<std::uint8_t> bytes;
    if (!ReadImage(value, file, budget, &bytes, &image_error))
      return Fail(error, "invalid_speaker_masks",
                  "speaker mask " + std::to_string(index + 1) + " " +
                      image_error);
    masks.push_back(std::move(bytes));
  }
  if (masks.size() != (speakers.size() == 2 ? 2U : 0U))
    return Fail(error, "invalid_speaker_masks",
                "two speakers need exactly two speaker masks (and one "
                "speaker none)");
  // "auto" (ComfyUI adapter): two tracks of the same length already share
  // one time line -> "para"; otherwise "add", like the patched ComfyUI node.
  if (const auto at = talk_json.find("\"auto\""); at != std::string::npos) {
    const bool same = speakers.size() == 2 &&
                      TalkFramesForAudio(speakers[0]) ==
                          TalkFramesForAudio(speakers[1]);
    talk_json.replace(at, 6, same ? "\"para\"" : "\"add\"");
  }
  // Duration: "seconds", or the audio ("add": the speakers in turn).
  int audio_frames = 0;
  const bool para = talk_json.find("\"para\"") != std::string::npos;
  for (const auto& speaker : speakers) {
    const int frames = TalkFramesForAudio(speaker);
    audio_frames =
        para ? std::max(audio_frames, frames) : audio_frames + frames;
  }
  std::string seconds;
  if (const json::Value* value = body.json.find("seconds");
      value != nullptr && (value->is_string() || value->is_number())) {
    seconds = SecondsText(value, "");
  } else {
    std::ostringstream text;
    text << std::fixed << std::setprecision(2)
         << std::max(audio_frames, 1) / 25.0;
    seconds = text.str();
  }
  std::string duration_error;
  const auto frames =
      ResolveDurationFrames(Family::kWan21Talk, seconds, &duration_error);
  if (!frames)
    return Fail(error, "invalid_seconds", duration_error);
  parameters.frames = *frames;
  std::string parameter_error;
  if (!ValidateParameters(Family::kWan21Talk, parameters, &parameter_error))
    return Fail(error, "invalid_parameters", parameter_error);
  json::Value options = json::Value::object();
  options["talk"] = json::parse(talk_json);
  for (std::size_t i = 0; i < speaker_bytes.size(); ++i)
    prepared->attachments.push_back(
        {.name = "audio_" + std::to_string(i + 1),
         .bytes = std::move(speaker_bytes[i])});
  for (std::size_t i = 0; i < masks.size(); ++i)
    prepared->attachments.push_back(
        {.name = "speaker_mask_" + std::to_string(i + 1),
         .bytes = std::move(masks[i])});
  if (!talk_video.empty())
    prepared->attachments.push_back(
        {.name = "guide_video", .bytes = std::move(talk_video)});
  prepared->prompt = std::move(common.prompt);
  prepared->size = std::move(common.size);
  prepared->seconds = std::move(seconds);
  prepared->seed = seed;
  prepared->parameters = std::move(parameters);
  prepared->options_json = options.dump();
  return true;
}

bool ReadClassName(const std::filesystem::path& file, std::string_view name) {
  std::ifstream input(file, std::ios::binary);
  if (!input)
    return false;
  const std::string text((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  try {
    return json::parse(text).member_str("_class_name") == name;
  } catch (...) {
    return false;
  }
}

bool WriteFile(const std::filesystem::path& path,
               const std::vector<std::uint8_t>& bytes, std::string* error) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output) {
    SetError(error, "cannot write " + path.filename().string());
    return false;
  }
  return true;
}

}  // namespace

std::string_view ModelId(Family family) noexcept {
  switch (family) {
    case Family::kWan22:
      return kWan22ModelId;
    case Family::kWan21Uni3C:
      return kWan21ModelId;
    case Family::kWan21Vace:
      return kWan21VaceModelId;
    case Family::kWan21Talk:
      return kWan21TalkModelId;
  }
  return kWan22ModelId;
}

std::string_view Preset(Family family) noexcept {
  switch (family) {
    case Family::kWan22:
      return kWan22Preset;
    case Family::kWan21Uni3C:
      return kWan21Preset;
    case Family::kWan21Vace:
      return kWan21VacePreset;
    case Family::kWan21Talk:
      return kWan21TalkPreset;
  }
  return kWan22Preset;
}

int Fps(Family family) noexcept {
  switch (family) {
    case Family::kWan22:
      return 24;
    case Family::kWan21Talk:
      return 25;
    default:
      return 16;
  }
}

bool IsWan22ModelDirectory(const std::filesystem::path& root) {
  std::ifstream input(root / "model_index.json", std::ios::binary);
  if (!input)
    return false;
  const std::string text((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  try {
    const json::Value index = json::parse(text);
    const json::Value* transformer_2 = index.find("transformer_2");
    const bool single_transformer =
        transformer_2 == nullptr || transformer_2->is_null() ||
        (transformer_2->is_array() && !transformer_2->empty() &&
         transformer_2->items()[0].is_null());
    return index.member_str("_class_name") == "WanPipeline" &&
           single_transformer &&
           std::filesystem::exists(root / "vae" / "config.json") &&
           std::filesystem::exists(root / "transformer" / "config.json");
  } catch (...) {
    return false;
  }
}

bool IsWan21CameraDirectory(const std::filesystem::path& root) {
  return ReadClassName(root / "gufo-camera.json", "GufoWan21Uni3CCamera");
}

bool IsWan21VaceDirectory(const std::filesystem::path& root) {
  return ReadClassName(root / "gufo-vace.json", "GufoWan21Vace");
}

bool IsWan21TalkDirectory(const std::filesystem::path& root) {
  return ReadClassName(root / "gufo-talk.json", "GufoWan21InfiniteTalk");
}

bool ParseVideoSize(std::string_view size, int* width, int* height) {
  const auto x = size.find('x');
  if (x == std::string_view::npos || x == 0 || x + 1 >= size.size())
    return false;
  int w = 0, h = 0;
  const auto first = std::from_chars(size.data(), size.data() + x, w);
  const auto second =
      std::from_chars(size.data() + x + 1, size.data() + size.size(), h);
  if (first.ec != std::errc() || first.ptr != size.data() + x ||
      second.ec != std::errc() || second.ptr != size.data() + size.size())
    return false;
  *width = w;
  *height = h;
  return true;
}

bool ValidateParameters(Family family, const ext::VideoParameters& p,
                        std::string* error) {
  const auto fail = [error](const char* message) {
    SetError(error, message);
    return false;
  };
  if (p.preset != Preset(family))
    return fail("the preset does not match the video model");
  const int w = p.output_width, h = p.output_height;
  switch (family) {
    case Family::kWan22:
      if (w < 256 || h < 256 || w > 1280 || h > 1280 || w % 32 != 0 ||
          h % 32 != 0 || w * h > 1280 * 704)
        return fail(
            "size must be WxH with multiples of 32, 256..1280 per side and "
            "at most 1280x704 pixels");
      if (p.frames < 5 || p.frames > 121 || (p.frames - 1) % 4 != 0)
        return fail("frames must be 4k + 1 between 5 and 121");
      if (p.evaluations < 1 || p.evaluations > 100)
        return fail("steps must be 1..100");
      return true;
    case Family::kWan21Uni3C:
    case Family::kWan21Talk:
      if (w < 256 || h < 256 || w > 1024 || h > 1024 || w % 16 != 0 ||
          h % 16 != 0 || w * h > 896 * 512)
        return fail(
            "size must be WxH with multiples of 16, 256..1024 per side and "
            "at most 896x512 pixels");
      if (family == Family::kWan21Talk) {
        if (p.frames < 1 || p.frames > 1500)
          return fail("frames must be 1..1500 (60 s at 25 fps)");
      } else if (p.frames < 5 || p.frames > 81 || (p.frames - 1) % 4 != 0) {
        return fail("frames must be 4k + 1 between 5 and 81");
      }
      if (p.evaluations < 1 || p.evaluations > 20)
        return fail("steps must be 1..20");
      return true;
    case Family::kWan21Vace:
      if (w < 256 || h < 256 || w > 1280 || h > 1280 || w % 16 != 0 ||
          h % 16 != 0 || w * h > 1280 * 720)
        return fail(
            "size must be WxH with multiples of 16, 256..1280 per side and "
            "at most 1280x720 pixels");
      if (p.frames < 5 || p.frames > 81 || (p.frames - 1) % 4 != 0)
        return fail("frames must be 4k + 1 between 5 and 81");
      if (p.evaluations < 1 || p.evaluations > 50)
        return fail("steps must be 1..50");
      return true;
  }
  return fail("unknown Wan family");
}

std::optional<int> ResolveDurationFrames(Family family,
                                         std::string_view seconds,
                                         std::string* error) {
  const auto value = ParseSeconds(seconds);
  switch (family) {
    case Family::kWan22: {
      if (!value || *value < 0.2 || *value > 5.05) {
        SetError(error, "'seconds' must be a number between 0.2 and 5");
        return std::nullopt;
      }
      const int groups = std::max(1, static_cast<int>(std::lround(*value * 6)));
      return std::min(4 * groups + 1, 121);
    }
    case Family::kWan21Uni3C:
    case Family::kWan21Vace: {
      if (!value || *value < 0.25 || *value > 5.05) {
        SetError(error, "'seconds' must be a number between 0.25 and 5");
        return std::nullopt;
      }
      const int groups = std::max(1, static_cast<int>(std::lround(*value * 4)));
      return std::min(4 * groups + 1, 81);
    }
    case Family::kWan21Talk:
      if (!value || *value < 0.04 || *value > 60.0) {
        SetError(error, "'seconds' must be a number between 0.04 and 60");
        return std::nullopt;
      }
      return std::clamp(static_cast<int>(std::lround(*value * 25.0)), 1, 1500);
  }
  return std::nullopt;
}

bool ParseCameraSpec(const json::Value& camera, CameraSpec* spec,
                     std::string* error) {
  const auto number = [](const json::Value* value, double low, double high,
                         double* out) {
    if (value == nullptr)
      return true;
    if (!value->is_number())
      return false;
    const double v = value->as_double(NAN);
    if (!std::isfinite(v) || v < low || v > high)
      return false;
    *out = v;
    return true;
  };
  if (!camera.is_object()) {
    SetError(error, "'gufo.camera' must be an object");
    return false;
  }
  *spec = {};
  if (const json::Value* poses = camera.find("poses")) {
    if (camera.contains("type") || !poses->is_array() || poses->empty() ||
        poses->size() > 81) {
      SetError(error,
               "'gufo.camera.poses' must be a non-empty array (at most 81) "
               "without 'type'");
      return false;
    }
    spec->type = "poses";
    for (const json::Value& pose : poses->items()) {
      if (!pose.is_object()) {
        SetError(error, "each camera pose must be an object");
        return false;
      }
      CameraKeyframe k;
      double frame = 0;
      if (!number(pose.find("frame"), 0, 1000, &frame) ||
          std::floor(frame) != frame) {
        SetError(error, "camera pose 'frame' must be an integer 0..1000");
        return false;
      }
      k.frame = static_cast<int>(frame);
      if (const json::Value* position = pose.find("position")) {
        if (!position->is_array() || position->size() != 3) {
          SetError(error, "camera pose 'position' must be [x, y, z]");
          return false;
        }
        for (int j = 0; j < 3; ++j)
          if (!number(&position->items()[j], -10, 10, &k.position[j])) {
            SetError(error,
                     "camera pose positions are scene depths in -10..10");
            return false;
          }
      }
      if (!number(pose.find("yaw"), -180, 180, &k.yaw) ||
          !number(pose.find("pitch"), -90, 90, &k.pitch) ||
          !number(pose.find("roll"), -180, 180, &k.roll) ||
          !number(pose.find("zoom"), 0.25, 4, &k.zoom)) {
        SetError(error, "camera pose yaw/pitch/roll are degrees, zoom 0.25..4");
        return false;
      }
      spec->keyframes.push_back(k);
    }
    std::sort(spec->keyframes.begin(), spec->keyframes.end(),
              [](const auto& a, const auto& b) { return a.frame < b.frame; });
    return true;
  }
  const json::Value* type = camera.find("type");
  if (type == nullptr || !type->is_string()) {
    SetError(error, "'gufo.camera' needs 'type', 'poses' or 'render_video'");
    return false;
  }
  spec->type = type->str();
  if (!number(camera.find("amount"), 0, 360, &spec->amount) ||
      !number(camera.find("speed"), 0.05, 4, &spec->speed)) {
    SetError(error, "'gufo.camera.amount' must be 0..360, 'speed' 0.05..4");
    return false;
  }
  if (const json::Value* ease = camera.find("ease")) {
    if (!ease->is_string()) {
      SetError(error, "'gufo.camera.ease' must be a string");
      return false;
    }
    spec->ease = ease->str();
  }
  // Validate the type and ease with a dry run.
  std::vector<CameraPose> poses;
  return BuildCameraPath(*spec, 5, 1.0, &poses, error);
}

bool PrepareRequest(Family family, const ext::VideoRequestInput& input,
                    ext::VideoPreparedRequest* prepared,
                    ext::VideoRequestError* error) {
  Body body;
  if (!ParseBody(input, &body, error))
    return false;
  switch (family) {
    case Family::kWan22:
      return PrepareWan22(body, prepared, error);
    case Family::kWan21Uni3C:
      return PrepareWan21(body, prepared, error);
    case Family::kWan21Vace:
      return PrepareVace(body, prepared, error);
    case Family::kWan21Talk:
      return PrepareTalk(body, prepared, error);
  }
  return Fail(error, "invalid_model", "unknown Wan family");
}

bool BuildJobRequest(Family family, const ext::VideoRequest& request,
                     JobRequest* job, std::string* error) {
  json::Value options = json::Value::object();
  try {
    if (!request.options_json.empty())
      options = json::parse(request.options_json);
  } catch (const std::exception& exception) {
    SetError(error, std::string("invalid Wan options: ") + exception.what());
    return false;
  }
  const auto attachment =
      [&](std::string_view name) -> const ext::VideoAttachment* {
    for (const auto& item : request.attachments)
      if (item.name == name)
        return &item;
    return nullptr;
  };
  const auto image = [&](const ext::VideoAttachment& item, Image* out) {
    try {
      core::Image decoded = Decode(item.bytes);
      out->width = static_cast<int>(decoded.width);
      out->height = static_cast<int>(decoded.height);
      out->rgb = std::move(decoded.pixels);
      return true;
    } catch (const std::exception& exception) {
      SetError(error, item.name + ": " + exception.what());
      return false;
    }
  };
  *job = {};
  job->family = family;
  job->model_root = request.model_root;
  job->output_path = request.output_path;
  job->prompt = request.prompt;
  job->seed = request.seed;
  job->width = request.parameters.output_width;
  job->height = request.parameters.output_height;
  job->frames = request.parameters.frames;
  job->steps = request.parameters.evaluations;
  job->negative_prompt = options.member_str("negative_prompt");
  job->guidance = options.member_double(
      "guidance", family == Family::kWan21Vace ? 1.0 : 5.0);
  if (const auto* first = attachment("first_frame");
      first != nullptr && !image(*first, &job->first_frame))
    return false;
  const std::filesystem::path directory = request.output_path.parent_path();
  if (const auto* guide = attachment("guide_video")) {
    job->guide_video_path = directory / "guide.mp4";
    if (!WriteFile(job->guide_video_path, guide->bytes, error))
      return false;
  }
  if (const json::Value* camera = options.find("camera"))
    job->camera_json = camera->dump();
  if (family == Family::kWan21Vace) {
    const json::Value* vace = options.find("vace");
    job->vace_json = vace != nullptr ? vace->dump() : "{}";
    if (const auto* mask = attachment("mask")) {
      job->mask_path =
          directory /
          (options.member_str("mask_format") == "ppm" ? "mask.ppm" : "mask.mp4");
      if (!WriteFile(job->mask_path, mask->bytes, error))
        return false;
    }
    for (int i = 0;; ++i) {
      const auto* reference = attachment("reference_image_" + std::to_string(i));
      if (reference == nullptr)
        break;
      Image decoded;
      if (!image(*reference, &decoded))
        return false;
      job->references.push_back(std::move(decoded));
    }
  }
  if (family == Family::kWan21Talk) {
    const json::Value* talk = options.find("talk");
    job->talk_json = talk != nullptr ? talk->dump() : "{}";
    for (int i = 1; i <= 2; ++i) {
      if (const auto* audio = attachment("audio_" + std::to_string(i))) {
        AudioWaveform waveform;
        std::string audio_error;
        if (!DecodeSpeech(audio->bytes, &waveform, &audio_error)) {
          SetError(error, "speaker " + std::to_string(i) + " audio " +
                              audio_error);
          return false;
        }
        job->speakers.push_back(std::move(waveform));
      }
      if (const auto* mask = attachment("speaker_mask_" + std::to_string(i))) {
        Image decoded;
        if (!image(*mask, &decoded))
          return false;
        job->speaker_masks.push_back(std::move(decoded));
      }
    }
  }
  return true;
}

std::string ParametersJson(const ext::VideoParameters& parameters,
                           std::uint64_t seed, std::string_view prompt_sha256) {
  json::Value root = json::Value::object();
  root["preset"] = parameters.preset;
  root["width"] = parameters.output_width;
  root["height"] = parameters.output_height;
  root["frames"] = parameters.frames;
  root["steps"] = parameters.evaluations;
  root["seed"] = std::to_string(seed);
  root["prompt_sha256"] = std::string(prompt_sha256);
  return root.dump();
}

}  // namespace gufo::models::wan22
