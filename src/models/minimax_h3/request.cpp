// MiniMax H3 /v1/videos requests on gufo 2.0 (gufo-studio-windows): the
// 1.4.9 video_api.cpp rules for H3, moved into the provider.
// Written with AI assistance (Claude Code).
#include "src/models/minimax_h3/request.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <system_error>

#include "src/core/image.hpp"
#include "src/core/json.hpp"
#include "src/models/minimax_h3/audio_input.hpp"
#include "src/models/minimax_h3/sampling.hpp"
#include "src/models/rife/video_io.hpp"

namespace gufo::minimax_h3 {
namespace {

constexpr double kMaximumExactJsonInteger = 9007199254740991.0;

std::mutex g_options_mutex;
ProviderOptions g_options;

bool Fail(ext::VideoRequestError* error, std::string code,
          std::string message, int status = 400) {
  if (error != nullptr) {
    error->status = status;
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

// A signed exact integer (guide frames count from the end when negative).
std::optional<int> JsonSignedFrame(const json::Value* value) {
  if (value == nullptr || !value->is_number())
    return std::nullopt;
  const double number = value->as_double(0.5);
  if (!std::isfinite(number) || std::floor(number) != number ||
      number < -100000.0 || number > 100000.0)
    return std::nullopt;
  return static_cast<int>(number);
}

std::optional<int> JsonFrame(const json::Value* value) {
  const auto parsed = JsonUnsigned(value);
  if (!parsed.has_value() ||
      *parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    return std::nullopt;
  return static_cast<int>(*parsed);
}

std::optional<std::vector<std::uint8_t>> DecodeBase64(std::string_view text) {
  std::vector<std::uint8_t> out;
  out.reserve(text.size() * 3 / 4);
  std::uint32_t buffer = 0;
  int bits = 0;
  for (const char c : text) {
    int v = 0;
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

// The request as the provider sees it: JSON members without the multipart
// file parts, plus the files by name.
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

// An image: an uploaded file, a base64 data URL, raw base64, a public HTTP(S)
// URL, or {"image_url": <string or {"url": string}>}. Returns the encoded
// bytes (decoded again by the runner) after checking that they decode.
bool ReadImage(const json::Value* value, const ext::VideoAttachment* file,
               core::ImageReadBudget& budget, std::vector<std::uint8_t>* bytes,
               std::string* error) {
  try {
    if (file != nullptr) {
      if (file->bytes.size() > budget.remaining_bytes)
        throw std::invalid_argument("keyframe images exceed the size limit");
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
    (void)core::DecodeImage(*bytes);
    return true;
  } catch (const std::exception& exception) {
    *error = exception.what();
    return false;
  }
}

// An audio guide: a base64 data URL or raw base64 of WAV/FLAC/Ogg/MP3, or
// the uploaded file. Decoded here once to reject a broken file early.
bool ReadAudio(const json::Value* value, const ext::VideoAttachment* file,
               std::vector<std::uint8_t>* bytes, std::string* error) {
  if (file != nullptr) {
    *bytes = file->bytes;
  } else {
    if (value == nullptr || !value->is_string() || value->str().empty()) {
      *error = "must be an audio file or a base64 (data URL) string";
      return false;
    }
    std::string_view text = value->str();
    if (text.starts_with("data:")) {
      const std::size_t marker = text.find(";base64,");
      if (marker == std::string_view::npos) {
        *error = "data URL must be base64 encoded";
        return false;
      }
      text.remove_prefix(marker + 8);
    }
    auto decoded = DecodeBase64(text);
    if (!decoded.has_value() || decoded->size() > kH3MaximumGuideAudioBytes) {
      *error = "is not valid base64 or exceeds 32 MiB";
      return false;
    }
    *bytes = std::move(*decoded);
  }
  AudioWaveform waveform;
  return DecodeGuideAudio(*bytes, &waveform, error);
}

json::Value ParseJson(std::string_view text) {
  return json::parse(std::string(text));
}

}  // namespace

bool ParseProviderOptions(
    const std::vector<std::pair<std::string, std::string>>& options,
    ProviderOptions* parsed, std::string* error) {
  for (const auto& [name, value] : options) {
    if (name == "vision-projector") {
      std::error_code ignored;
      if (value.empty() ||
          !std::filesystem::is_regular_file(value, ignored)) {
        *error = "--vision-projector file not found: " + value;
        return false;
      }
      parsed->vision_projector = value;
      continue;
    }
    if (name != "lora") {
      *error = "minimax-h3 does not take --" + name;
      return false;
    }
    // --lora NAME=PATH[,steps=N][,strength=S] (as gufo 1.4.9).
    const std::string_view text = value;
    const std::size_t equals = text.find('=');
    if (equals == 0 || equals == std::string_view::npos) {
      *error = "--lora expects NAME=PATH[,steps=N][,strength=S]";
      return false;
    }
    const std::string lora_name(text.substr(0, equals));
    std::string_view rest = text.substr(equals + 1);
    RegisteredLora lora;
    std::size_t comma = rest.find(',');
    lora.path = std::filesystem::path(std::string(rest.substr(0, comma)));
    while (comma != std::string_view::npos) {
      rest.remove_prefix(comma + 1);
      comma = rest.find(',');
      const std::string_view item = rest.substr(0, comma);
      try {
        if (item.starts_with("steps="))
          lora.steps = std::stoi(std::string(item.substr(6)));
        else if (item.starts_with("strength="))
          lora.strength = std::stof(std::string(item.substr(9)));
        else
          throw std::invalid_argument("unknown key");
      } catch (const std::exception&) {
        *error = "--lora option '" + std::string(item) +
                 "' is not steps=N or strength=S";
        return false;
      }
    }
    if (lora.path.empty() || lora.steps < 0 || lora.steps > 1000 ||
        parsed->loras.contains(lora_name)) {
      *error = "--lora '" + lora_name +
               "' needs a path, steps in 0..1000 and a unique name";
      return false;
    }
    std::error_code filesystem_error;
    if (!std::filesystem::is_regular_file(lora.path, filesystem_error)) {
      *error = "--lora file not found: " + lora.path.string();
      return false;
    }
    parsed->loras.emplace(lora_name, std::move(lora));
  }
  return true;
}

void SetActiveProviderOptions(ProviderOptions options) {
  const std::lock_guard<std::mutex> lock(g_options_mutex);
  g_options = std::move(options);
}

ProviderOptions ActiveProviderOptions() {
  const std::lock_guard<std::mutex> lock(g_options_mutex);
  return g_options;
}

bool PrepareRequest(const ext::VideoRequestInput& input,
                    ext::VideoPreparedRequest* prepared,
                    ext::VideoRequestError* error) {
  Body body;
  try {
    json::Value parsed = ParseJson(input.body_json);
    if (!parsed.is_object())
      return Fail(error, "invalid_body", "request body must be a JSON object");
    for (const auto& file : input.files)
      body.files.emplace(file.name, &file);
    for (const auto& [key, value] : parsed.members())
      if (!body.files.contains(key))
        body.json[key] = value;
  } catch (const std::exception&) {
    return Fail(error, "parse_error", "request body must be valid JSON");
  }
  for (const auto& [name, file] : body.files) {
    (void)file;
    if (name == "control_video" || name == "mask" ||
        name.starts_with("reference_image"))
      return Fail(error, "unsupported_field",
                  "request contains an unsupported video field");
  }
  if (body.json.contains("references"))
    return Fail(error, "unsupported_input_reference",
                "image and ordered-reference conditioning are not supported");
  // Unknown top-level members are ignored, as in the other OpenAI-compatible
  // servers of gufo 2.0 (1.4.9 refused them); "gufo" stays strict.

  const json::Value* prompt_value = body.json.find("prompt");
  const json::Value* size_value = body.json.find("size");
  if (prompt_value == nullptr || !prompt_value->is_string() ||
      (size_value != nullptr && !size_value->is_string()))
    return Fail(error, "invalid_parameter_type",
                "'model', 'prompt', and optional 'size' must be strings");
  const std::string prompt = prompt_value->str();
  const std::string size = body.json.member_str("size", "512x512");
  const json::Value* seconds_value = body.json.find("seconds");
  std::string seconds;
  if (seconds_value != nullptr && seconds_value->is_string())
    seconds = seconds_value->str();
  else if (seconds_value != nullptr && seconds_value->is_number())
    seconds = seconds_value->dump();
  if (prompt.empty() || seconds.empty())
    return Fail(error, "missing_required_parameter",
                "'model', 'prompt', and 'seconds' are required");
  std::string duration_error;
  const auto duration_frames = ResolveDurationFrames(seconds, &duration_error);
  if (!duration_frames)
    return Fail(error, "invalid_seconds", duration_error);
  std::string preset = size == "1344x768" ? "exact-1344x768" : "exact";
  std::string model_preset;
  const std::string_view model = input.model;
  if (model == "minimax-h3-exact")
    model_preset = "exact";
  else if (model == "minimax-h3-fast")
    model_preset = "fast";
  else if (model == "minimax-h3-aggressive")
    model_preset = "aggressive";
  else if (model == "minimax-h3-dev")
    model_preset = "dev";
  else if (model == "minimax-h3-fullres")
    model_preset = "exact-1344x768";
  else if (model != "minimax-h3")
    return Fail(error, "invalid_model", "unsupported video model");
  if (!model_preset.empty())
    preset = model_preset;
  bool preset_explicit = false;

  std::string output_format = "mp4";
  std::uint64_t seed = 42;
  std::optional<int> frames;
  std::optional<int> selected_frame;
  std::optional<int> steps;
  std::string lora;
  std::optional<double> lora_strength;
  int audio_reference_frame = 0;
  const json::Value* guide_keyframes = nullptr;
  const json::Value* guide_video_value = nullptr;
  if (const json::Value* extension = body.json.find("gufo")) {
    // "interpolate" and "upscale" never reach the provider (the host runs
    // them); "preview" is the host's too.
    if (!extension->is_object() ||
        !HasOnlyMembers(*extension,
                        {"preset", "seed", "frames", "output_format",
                         "selected_frame", "steps", "lora", "lora_strength",
                         "interpolate", "upscale", "keyframes",
                         "audio_reference_frame", "guide_video", "preview"}))
      return Fail(error, "invalid_strix_options",
                  "'gufo' must contain only supported video options");
    if (extension->contains("preset")) {
      if (!extension->find("preset")->is_string())
        return Fail(error, "invalid_preset", "'gufo.preset' must be a string");
      preset = extension->member_str("preset");
      preset_explicit = true;
    }
    if (extension->contains("output_format") &&
        !extension->find("output_format")->is_string())
      return Fail(error, "invalid_output_format",
                  "'gufo.output_format' must be a string");
    output_format = extension->member_str("output_format", output_format);
    if (extension->contains("seed")) {
      const auto parsed = JsonUnsigned(extension->find("seed"));
      if (!parsed.has_value())
        return Fail(error, "invalid_seed",
                    "'gufo.seed' must be a non-negative exact integer");
      seed = *parsed;
    }
    if (extension->contains("frames")) {
      frames = JsonFrame(extension->find("frames"));
      if (!frames.has_value())
        return Fail(error, "invalid_frames",
                    "'gufo.frames' must be a positive legal frame count");
    }
    if (extension->contains("steps")) {
      steps = JsonFrame(extension->find("steps"));
      if (!steps.has_value() || *steps < 2 || *steps > kH3MaximumEvaluations)
        return Fail(error, "invalid_steps",
                    "'gufo.steps' must be an integer in 2..1000");
    }
    if (extension->contains("lora")) {
      const json::Value* value = extension->find("lora");
      if (!value->is_string() || value->str().empty())
        return Fail(error, "invalid_lora",
                    "'gufo.lora' must name a LoRA the server registered");
      lora = value->str();
    }
    if (extension->contains("lora_strength")) {
      const json::Value* value = extension->find("lora_strength");
      if (!value->is_number() || !std::isfinite(value->as_double(0.0)) ||
          std::fabs(value->as_double(0.0)) > 4.0)
        return Fail(error, "invalid_lora",
                    "'gufo.lora_strength' must be a number in -4..4");
      lora_strength = value->as_double(1.0);
    }
    if (extension->contains("keyframes")) {
      guide_keyframes = extension->find("keyframes");
      if (!guide_keyframes->is_array())
        return Fail(error, "invalid_keyframes",
                    "'gufo.keyframes' must be an array of {\"frame\", "
                    "\"image\"} objects");
    }
    if (extension->contains("guide_video")) {
      guide_video_value = extension->find("guide_video");
      if (!guide_video_value->is_object() && !guide_video_value->is_array())
        return Fail(error, "invalid_guide_video",
                    "'gufo.guide_video' must be {\"frame\", \"video\"} or an "
                    "array of them");
    }
    if (extension->contains("audio_reference_frame")) {
      const auto parsed =
          JsonSignedFrame(extension->find("audio_reference_frame"));
      if (!parsed.has_value())
        return Fail(error, "invalid_audio_reference",
                    "'gufo.audio_reference_frame' must be an integer");
      audio_reference_frame = *parsed;
    }
    if (extension->contains("selected_frame")) {
      selected_frame = JsonFrame(extension->find("selected_frame"));
      if (!selected_frame.has_value())
        return Fail(error, "invalid_selected_frame",
                    "'gufo.selected_frame' must be a non-negative integer");
    }
  }

  if (preset_explicit && !model_preset.empty() && preset != model_preset)
    return Fail(error, "preset_model_mismatch",
                "model alias and 'gufo.preset' must select the same preset");
  // "exact" at any other accepted canvas (e.g. FilmStudio's 832x512) is the
  // "exact-WxH" preset; the fixed presets keep their own sizes.
  if (preset == "exact" && size != "512x512" && size != "256x256") {
    if (!ParseCanvasSize(size).has_value())
      return Fail(error, "invalid_size",
                  "'size' must be WxH with multiples of 32, 256..1344 per side "
                  "and at most 768*1344 pixels");
    preset = "exact-" + size;
  }
  std::string preset_error;
  auto parameters = ResolveGenerationPreset(preset, &preset_error);
  if (!parameters.has_value())
    return Fail(error, "invalid_preset", preset_error);
  parameters->frames = frames.value_or(*duration_frames);
  if (steps.has_value())
    parameters->evaluations = *steps;
  if (!lora.empty()) {
    const ProviderOptions active = ActiveProviderOptions();
    const auto found = active.loras.find(lora);
    if (found == active.loras.end())
      return Fail(error, "invalid_video_request",
                  "unknown LoRA '" + lora +
                      "'; the server registers LoRAs with --lora NAME=PATH");
    if (!steps.has_value() && found->second.steps != 0)
      parameters->evaluations = found->second.steps;
  }

  json::Value options = json::Value::object();
  json::Value keyframe_list = json::Value::array();
  std::vector<std::pair<int, KeyframeAnchor>> anchors;  // for checks
  std::size_t keyframe_count = 0;
  core::ImageReadBudget budget;
  const auto add_keyframe = [&](const json::Value* value,
                                const ext::VideoAttachment* file,
                                KeyframeAnchor anchor, std::optional<int> frame,
                                bool vision, const std::string& label,
                                const char* code) -> bool {
    std::vector<std::uint8_t> bytes;
    std::string image_error;
    if (!ReadImage(value, file, budget, &bytes, &image_error))
      return Fail(error, code, "'" + label + "' " + image_error);
    const std::string name = "keyframe_image_" + std::to_string(keyframe_count++);
    json::Value entry = json::Value::object();
    entry["attachment"] = name;
    entry["anchor"] = anchor == KeyframeAnchor::kLast ? "last" : "first";
    if (frame.has_value())
      entry["frame"] = static_cast<double>(*frame);
    entry["vision"] = vision;
    keyframe_list.push_back(std::move(entry));
    prepared->attachments.push_back({.name = name, .bytes = std::move(bytes)});
    int resolved = frame.has_value() ? *frame
                                     : KeyframeAnchorFrame(anchor,
                                                           parameters->frames);
    anchors.emplace_back(resolved, anchor);
    return true;
  };
  // FL2VA keyframes: `input_reference` (OpenAI) and `first_frame` are the
  // same first-frame anchor; `last_frame` anchors the end.
  if (body.has("input_reference") && body.has("first_frame"))
    return Fail(error, "invalid_input_reference",
                "'input_reference' and 'first_frame' both set the first frame");
  for (const auto& [name, anchor] :
       {std::pair{"input_reference", KeyframeAnchor::kFirst},
        std::pair{"first_frame", KeyframeAnchor::kFirst},
        std::pair{"last_frame", KeyframeAnchor::kLast}}) {
    if (!body.has(name))
      continue;
    if (!add_keyframe(body.json.find(name), body.file(name), anchor,
                      std::nullopt, true, name, "invalid_input_reference"))
      return false;
  }
  // Guide frames (MiniMaxH3AddGuide image): "gufo.keyframes" entries
  // {"frame": N, "image": <image>} and multipart files "keyframe_<N>".
  if (guide_keyframes != nullptr) {
    std::size_t index = 0;
    for (const json::Value& entry : guide_keyframes->items()) {
      const std::string label = "gufo.keyframes[" + std::to_string(index++) + "]";
      const auto frame =
          entry.is_object() ? JsonSignedFrame(entry.find("frame")) : std::nullopt;
      if (!frame.has_value() || !HasOnlyMembers(entry, {"frame", "image"}))
        return Fail(error, "invalid_keyframes",
                    "'" + label +
                        "' must be {\"frame\": integer, \"image\": image}");
      if (!add_keyframe(entry.find("image"), nullptr, KeyframeAnchor::kFirst,
                        *frame, false, label, "invalid_keyframes"))
        return false;
    }
  }
  for (const auto& [name, file] : body.files) {
    if (!name.starts_with("keyframe_"))
      continue;
    int frame = 0;
    const std::string_view digits = std::string_view(name).substr(9);
    const auto parsed =
        std::from_chars(digits.data(), digits.data() + digits.size(), frame);
    if (digits.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != digits.data() + digits.size())
      return Fail(error, "invalid_keyframes",
                  "multipart guide frames are named keyframe_<frame>");
    if (!add_keyframe(nullptr, file, KeyframeAnchor::kFirst, frame, false, name,
                      "invalid_keyframes"))
      return false;
  }
  // One image per pixel frame, inside the video.
  {
    std::vector<int> used;
    for (const auto& [raw, anchor] : anchors) {
      (void)anchor;
      int frame = raw < 0 ? raw + parameters->frames : raw;
      if (frame < 0 || frame >= parameters->frames)
        return Fail(error, "invalid_keyframes",
                    "guide frame " + std::to_string(raw) +
                        " is outside the video's " +
                        std::to_string(parameters->frames) + " frames");
      if (std::ranges::find(used, frame) != used.end())
        return Fail(error, "invalid_keyframes",
                    "two images anchor frame " + std::to_string(frame));
      used.push_back(frame);
    }
    if (anchors.size() > kH3MaximumKeyframes)
      return Fail(error, "invalid_keyframes",
                  "at most " + std::to_string(kH3MaximumKeyframes) +
                      " keyframes and guide frames");
  }
  options["keyframes"] = std::move(keyframe_list);

  // Guide clips: "gufo.guide_video" {"frame": N, "video": MP4 data URL or
  // base64} (or an array of them) and multipart files "guide_video_<N>".
  json::Value clip_list = json::Value::array();
  std::size_t clips = 0;
  const auto check_frame = [&](int frame, const std::string& label) -> bool {
    const int resolved = frame < 0 ? frame + parameters->frames : frame;
    if (resolved < 0 || resolved >= parameters->frames)
      return Fail(error, "invalid_guide_video",
                  "'" + label + "' frame " + std::to_string(frame) +
                      " is outside the video's " +
                      std::to_string(parameters->frames) + " frames");
    return true;
  };
  const auto add_clip = [&](int frame, std::vector<std::uint8_t> mp4) {
    const std::string name = "guide_clip_" + std::to_string(clips++);
    json::Value entry = json::Value::object();
    entry["attachment"] = name;
    entry["frame"] = static_cast<double>(frame);
    clip_list.push_back(std::move(entry));
    prepared->attachments.push_back({.name = name, .bytes = std::move(mp4)});
  };
  if (guide_video_value != nullptr) {
    std::vector<const json::Value*> entries;
    if (guide_video_value->is_array()) {
      for (const json::Value& entry : guide_video_value->items())
        entries.push_back(&entry);
    } else {
      entries.push_back(guide_video_value);
    }
    std::size_t index = 0;
    for (const json::Value* entry : entries) {
      const std::string label =
          "gufo.guide_video[" + std::to_string(index++) + "]";
      const auto frame = entry->is_object()
                             ? JsonSignedFrame(entry->find("frame"))
                             : std::nullopt;
      const json::Value* video =
          entry->is_object() ? entry->find("video") : nullptr;
      if (!frame.has_value() || video == nullptr || !video->is_string() ||
          !HasOnlyMembers(*entry, {"frame", "video"}))
        return Fail(error, "invalid_guide_video",
                    "'" + label +
                        "' must be {\"frame\": integer, \"video\": MP4 data "
                        "URL or base64}");
      if (!check_frame(*frame, label))
        return false;
      std::string_view text = video->str();
      if (text.starts_with("data:")) {
        const auto comma = text.find(',');
        if (comma == std::string_view::npos ||
            text.substr(0, comma).find(";base64") == std::string_view::npos)
          return Fail(error, "invalid_guide_video",
                      "'" + label + ".video' data URL must be base64");
        text.remove_prefix(comma + 1);
      }
      auto decoded = DecodeBase64(text);
      if (!decoded.has_value() || decoded->size() < 64)
        return Fail(error, "invalid_guide_video",
                    "'" + label + ".video' is not valid base64 video");
      add_clip(*frame, std::move(*decoded));
    }
  }
  for (const auto& [name, file] : body.files) {
    if (!name.starts_with("guide_video_"))
      continue;
    int frame = 0;
    const std::string_view digits = std::string_view(name).substr(12);
    const auto parsed =
        std::from_chars(digits.data(), digits.data() + digits.size(), frame);
    if (digits.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != digits.data() + digits.size())
      return Fail(error, "invalid_guide_video",
                  "multipart guide clips are named guide_video_<frame>");
    if (!check_frame(frame, name))
      return false;
    if (file->bytes.size() < 64)
      return Fail(error, "invalid_guide_video",
                  "'" + name + "' is not a video file");
    add_clip(frame, file->bytes);
  }
  if (anchors.size() + clips > kH3MaximumKeyframes)
    return Fail(error, "invalid_guide_video",
                "at most " + std::to_string(kH3MaximumKeyframes) +
                    " keyframes, guide frames and guide clips");
  options["guide_videos"] = std::move(clip_list);

  // Audio anchor (MiniMaxH3AddGuide audio): "audio_reference" at
  // "gufo.audio_reference_frame".
  json::Value audio_list = json::Value::array();
  if (body.has("audio_reference")) {
    std::vector<std::uint8_t> bytes;
    std::string audio_error;
    if (!ReadAudio(body.json.find("audio_reference"),
                   body.file("audio_reference"), &bytes, &audio_error))
      return Fail(error, "invalid_audio_reference",
                  "'audio_reference' " + audio_error);
    int frame = audio_reference_frame;
    if (frame < 0)
      frame += parameters->frames;
    if (frame < 0 || frame >= parameters->frames)
      return Fail(error, "invalid_audio_reference",
                  "'gufo.audio_reference_frame' is outside the video");
    json::Value entry = json::Value::object();
    entry["attachment"] = "audio_reference";
    entry["frame"] = static_cast<double>(frame);
    audio_list.push_back(std::move(entry));
    prepared->attachments.push_back(
        {.name = "audio_reference", .bytes = std::move(bytes)});
  }
  options["audio_guides"] = std::move(audio_list);

  if (output_format == "ppm") {
    if (parameters->preset != "development-256" || size != "256x256")
      return Fail(error, "invalid_output_format",
                  "PPM output requires preset 'dev' and size '256x256'");
    parameters->selected_frames = {
        selected_frame.value_or(parameters->frames / 2)};
  } else if (output_format == "mp4") {
    if (!parameters->mux || selected_frame.has_value())
      return Fail(error, "invalid_output_format",
                  "MP4 output requires a supported preset, duration, and "
                  "resolution combination");
  } else {
    return Fail(error, "invalid_output_format",
                "'gufo.output_format' must be 'mp4' or 'ppm'");
  }
  std::string parameter_error;
  if (!ValidateGenerationParameters(*parameters, &parameter_error))
    return Fail(error, "invalid_frames", parameter_error);
  if (parameters->frames != *duration_frames)
    return Fail(error, "invalid_frames",
                "'gufo.frames' must match the aligned 'seconds' duration");

  if (!lora.empty()) {
    options["lora"] = lora;
    if (lora_strength.has_value())
      options["lora_strength"] = *lora_strength;
  }
  prepared->prompt = prompt;
  prepared->size = size;
  prepared->seconds = seconds;
  prepared->output_format = output_format;
  prepared->seed = seed;
  prepared->parameters = {
      .preset = parameters->preset,
      .internal_width = parameters->internal_width,
      .internal_height = parameters->internal_height,
      .output_width = parameters->output_width,
      .output_height = parameters->output_height,
      .frames = parameters->frames,
      .evaluations = parameters->evaluations,
      .active_blocks = parameters->active_blocks,
      .reuse_interval = parameters->reuse_interval,
      .row_parallel_attention = parameters->row_parallel_attention,
      .decode_audio = parameters->decode_audio,
      .mux = parameters->mux,
      .selected_frames = parameters->selected_frames,
  };
  prepared->options_json = options.dump();
  return true;
}

bool DecodePreparedRequest(const ext::VideoRequest& request,
                           const ProviderOptions& provider,
                           const std::filesystem::path& work_directory,
                           GenerationRequest* generation, std::string* error) {
  generation->vision_projector = provider.vision_projector;
  if (request.options_json.empty())
    return true;  // plain text-to-video (no prepared conditioning)
  json::Value options;
  try {
    options = ParseJson(request.options_json);
  } catch (const std::exception&) {
    *error = "invalid MiniMax H3 job options";
    return false;
  }
  const auto attachment =
      [&](std::string_view name) -> const std::vector<std::uint8_t>* {
    for (const auto& item : request.attachments)
      if (item.name == name)
        return &item.bytes;
    return nullptr;
  };
  try {
    if (const json::Value* keyframes = options.find("keyframes")) {
      for (const json::Value& entry : keyframes->items()) {
        const auto* bytes = attachment(entry.member_str("attachment"));
        if (bytes == nullptr) {
          *error = "missing MiniMax H3 keyframe attachment";
          return false;
        }
        core::Image image = core::DecodeImage(*bytes);
        GenerationKeyframe keyframe{
            .anchor = entry.member_str("anchor") == "last"
                          ? KeyframeAnchor::kLast
                          : KeyframeAnchor::kFirst,
            .width = static_cast<int>(image.width),
            .height = static_cast<int>(image.height),
            .rgb = std::move(image.pixels),
        };
        if (const json::Value* frame = entry.find("frame"))
          keyframe.frame = static_cast<int>(frame->as_double(0.0));
        if (const json::Value* vision = entry.find("vision"))
          keyframe.vision = vision->is_bool() ? vision->as_bool() : true;
        generation->keyframes.push_back(std::move(keyframe));
      }
    }
    if (const json::Value* audio = options.find("audio_guides")) {
      for (const json::Value& entry : audio->items()) {
        const auto* bytes = attachment(entry.member_str("attachment"));
        if (bytes == nullptr) {
          *error = "missing MiniMax H3 audio attachment";
          return false;
        }
        GenerationAudioGuide guide;
        guide.frame = static_cast<int>(entry.find("frame")->as_double(0.0));
        if (!DecodeGuideAudio(*bytes, &guide.waveform, error))
          return false;
        generation->audio_guides.push_back(std::move(guide));
      }
    }
    if (const json::Value* clips = options.find("guide_videos")) {
      std::size_t index = 0;
      for (const json::Value& entry : clips->items()) {
        const auto* bytes = attachment(entry.member_str("attachment"));
        if (bytes == nullptr) {
          *error = "missing MiniMax H3 guide clip attachment";
          return false;
        }
        // Decoded with FFmpeg (as 1.4.9's DecodeH3GuideVideos).
        std::error_code ignored;
        std::filesystem::create_directories(work_directory, ignored);
        const std::filesystem::path path =
            work_directory / ("guide_clip_" + std::to_string(index) + ".mp4");
        {
          std::ofstream file(path, std::ios::binary);
          file.write(reinterpret_cast<const char*>(bytes->data()),
                     static_cast<std::streamsize>(bytes->size()));
          if (!file) {
            *error = "cannot store MiniMax H3 guide clip " +
                     std::to_string(index);
            return false;
          }
        }
        models::rife::VideoInfo info;
        if (!models::rife::ProbeVideo(path, &info, error))
          return false;
        if (info.width < 16 || info.height < 16 || info.width > 4096 ||
            info.height > 4096) {
          *error = "MiniMax H3 guide clip " + std::to_string(index) +
                   " has an unsupported size";
          return false;
        }
        models::rife::FrameReader reader;
        if (!reader.Open(path, info.width, info.height, error))
          return false;
        GenerationVideoGuide guide;
        guide.frame = static_cast<int>(entry.find("frame")->as_double(0.0));
        guide.width = info.width;
        guide.height = info.height;
        const std::size_t frame_bytes = static_cast<std::size_t>(info.width) *
                                        static_cast<std::size_t>(info.height) *
                                        3U;
        std::vector<std::uint8_t> frame(frame_bytes);
        while (guide.frames < request.parameters.frames) {
          bool end = false;
          if (!reader.Read(frame.data(), &end, error)) {
            if (!end) {
              reader.Close();
              return false;
            }
            error->clear();
            break;
          }
          guide.rgb.insert(guide.rgb.end(), frame.begin(), frame.end());
          ++guide.frames;
        }
        reader.Close();
        std::filesystem::remove(path, ignored);
        if (guide.frames == 0) {
          *error = "MiniMax H3 guide clip " + std::to_string(index) +
                   " has no frames";
          return false;
        }
        generation->video_guides.push_back(std::move(guide));
        ++index;
      }
    }
    if (const json::Value* lora = options.find("lora")) {
      const auto found = provider.loras.find(lora->str());
      if (found == provider.loras.end()) {
        *error = "unknown LoRA '" + lora->str() + "'";
        return false;
      }
      generation->lora_path = found->second.path;
      generation->lora_strength = found->second.strength;
      if (const json::Value* strength = options.find("lora_strength"))
        generation->lora_strength = static_cast<float>(strength->as_double(1.0));
    }
  } catch (const std::exception& exception) {
    *error = exception.what();
    return false;
  }
  return true;
}

}  // namespace gufo::minimax_h3
