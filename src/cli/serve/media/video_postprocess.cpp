// RIFE / Real-ESRGAN post-steps of video jobs (see video_postprocess.hpp).
// Request parsing and the step order are those of gufo-studio-windows 1.4
// (video_api.cpp JsonInterpolate / JsonUpscale / CreateProcessVideo and the
// post_process lambda of video_jobs.cpp).
// Written with AI assistance (Claude Code).
#include "src/cli/serve/media/video_postprocess.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <system_error>
#include <utility>

#include "src/cli/serve/audio_stream.hpp"
#include "src/cli/serve/logging.hpp"

namespace gufo::server {
namespace {

// Largest accepted source video (the HTTP body limit applies first).
constexpr std::size_t kMaxSourceVideoBytes = std::size_t{2} << 30U;

bool HasOnlyMembers(const json::Value& value,
                    std::initializer_list<std::string_view> allowed) {
  return std::ranges::all_of(value.members(), [&](const auto& member) {
    return std::ranges::find(allowed, member.first) != allowed.end();
  });
}

std::optional<int> JsonCount(const json::Value* value) {
  if (value == nullptr || !value->is_number())
    return std::nullopt;
  const double number = value->as_double(-1.0);
  if (!std::isfinite(number) || number < 0.0 || std::floor(number) != number ||
      number > static_cast<double>(std::numeric_limits<int>::max()))
    return std::nullopt;
  return static_cast<int>(number);
}

bool ParseSize(std::string_view size, int* width, int* height) {
  const auto x = size.find('x');
  if (x == std::string_view::npos || x == 0 || x + 1 >= size.size())
    return false;
  const auto first = std::from_chars(size.data(), size.data() + x, *width);
  const auto second =
      std::from_chars(size.data() + x + 1, size.data() + size.size(), *height);
  return first.ec == std::errc() && first.ptr == size.data() + x &&
         second.ec == std::errc() && second.ptr == size.data() + size.size() &&
         *width > 0 && *height > 0;
}

// "gufo.interpolate": a whole-number multiplier (1 = off, 2..8; 4 turns
// 24 fps into 96 fps at the same speed) or {"fps": N} / {"fps": "N/D"}, a
// target frame rate at the same speed (60 from H3's 24 or Wan's 16 fps;
// 1..240).
std::optional<VideoInterpolation> JsonInterpolate(const json::Value* value) {
  if (value != nullptr && value->is_object()) {
    const json::Value* fps = value->find("fps");
    if (!HasOnlyMembers(*value, {"fps"}) || fps == nullptr)
      return std::nullopt;
    VideoInterpolation spec;
    if (fps->is_string()) {
      const std::string_view whole(fps->str());
      const std::size_t slash = whole.find('/');
      const auto parse = [](std::string_view part, int* out) {
        const auto result =
            std::from_chars(part.data(), part.data() + part.size(), *out);
        return result.ec == std::errc{} &&
               result.ptr == part.data() + part.size();
      };
      if (!parse(whole.substr(0, slash), &spec.fps_num) ||
          (slash != std::string_view::npos &&
           !parse(whole.substr(slash + 1), &spec.fps_den)))
        return std::nullopt;
    } else {
      const auto parsed = JsonCount(fps);
      if (!parsed.has_value())
        return std::nullopt;
      spec.fps_num = *parsed;
    }
    if (spec.fps_num < 1 || spec.fps_den < 1 || spec.fps_den > 1001 ||
        spec.fps_num > 240 * spec.fps_den)
      return std::nullopt;
    return spec;
  }
  const auto parsed = JsonCount(value);
  if (!parsed.has_value() || *parsed < 1 || *parsed > 8)
    return std::nullopt;
  return VideoInterpolation{.multiplier = *parsed};
}

// "gufo.upscale": "WxH" or {"width", "height", "fit" ("pad" default, "crop",
// "stretch"), "model" (a name registered with --upscale-model)}. The values
// are checked against the service in ValidateVideoPostProcess.
std::optional<VideoUpscale> JsonUpscale(const json::Value* value) {
  VideoUpscale spec;
  if (value == nullptr)
    return std::nullopt;
  if (value->is_string()) {
    if (!ParseSize(value->str(), &spec.width, &spec.height))
      return std::nullopt;
    return spec;
  }
  if (!value->is_object() ||
      !HasOnlyMembers(*value, {"width", "height", "fit", "model"}))
    return std::nullopt;
  const auto width = JsonCount(value->find("width"));
  const auto height = JsonCount(value->find("height"));
  if (!width.has_value() || !height.has_value() || *width < 2 || *height < 2)
    return std::nullopt;
  spec.width = *width;
  spec.height = *height;
  for (const char* key : {"fit", "model"}) {
    if (const json::Value* text = value->find(key)) {
      if (!text->is_string() || text->str().empty() ||
          text->str().size() > 128)
        return std::nullopt;
      (std::string_view(key) == "fit" ? spec.fit : spec.model) = text->str();
    }
  }
  return spec;
}

}  // namespace

bool ExtractVideoPostProcess(json::Value* body, VideoPostProcess* post,
                             VideoPostProcessError* error) {
  *post = {};
  const json::Value* extension = body->find("gufo");
  if (extension == nullptr || !extension->is_object())
    return true;
  if (const json::Value* interpolate = extension->find("interpolate")) {
    const auto parsed = JsonInterpolate(interpolate);
    if (!parsed.has_value()) {
      *error = {.code = "invalid_interpolate",
                .message =
                    "'gufo.interpolate' must be 1..8 or {\"fps\": 1..240}"};
      return false;
    }
    post->interpolation = *parsed;
  }
  if (const json::Value* upscale = extension->find("upscale")) {
    const auto parsed = JsonUpscale(upscale);
    if (!parsed.has_value()) {
      *error = {.code = "invalid_upscale",
                .message = "'gufo.upscale' must be \"WxH\" or {\"width\", "
                           "\"height\", \"fit\", \"model\"}"};
      return false;
    }
    post->upscale = *parsed;
  }
  if (!extension->contains("interpolate") && !extension->contains("upscale"))
    return true;
  // json::Value has no erase: rebuild "gufo" without the two post-steps.
  json::Value rest = json::Value::object();
  for (const auto& [key, value] : extension->members())
    if (key != "interpolate" && key != "upscale")
      rest.append_member(key, value);
  if (rest.members().empty()) {
    json::Value copy = json::Value::object();
    for (const auto& [key, value] : body->members())
      if (key != "gufo")
        copy.append_member(key, value);
    *body = std::move(copy);
  } else {
    (*body)["gufo"] = std::move(rest);
  }
  return true;
}

bool IsSourceVideoRequest(const json::Value& body,
                          const std::vector<ext::VideoAttachment>& files) {
  return body.contains("source_video") ||
         std::ranges::any_of(files, [](const ext::VideoAttachment& file) {
           return file.name == "source_video";
         });
}

bool ParseSourceVideoRequest(const json::Value& body,
                             const std::vector<ext::VideoAttachment>& files,
                             std::vector<std::uint8_t>* source,
                             VideoPostProcessError* error) {
  if (!HasOnlyMembers(body, {"model", "source_video", "gufo"})) {
    *error = {.code = "unsupported_field",
              .message = "a 'source_video' request takes only 'model', "
                         "'source_video' and 'gufo'"};
    return false;
  }
  if (const json::Value* extension = body.find("gufo");
      extension != nullptr && (!extension->is_object() ||
                               !extension->members().empty())) {
    // ExtractVideoPostProcess already took interpolate / upscale out.
    *error = {.code = "invalid_strix_options",
              .message = "'gufo' of a 'source_video' request may contain only "
                         "'interpolate' and 'upscale'"};
    return false;
  }
  source->clear();
  const auto upload =
      std::ranges::find_if(files, [](const ext::VideoAttachment& file) {
        return file.name == "source_video";
      });
  if (upload != files.end()) {
    *source = upload->bytes;
  } else if (const json::Value* video = body.find("source_video");
             video != nullptr && video->is_string()) {
    std::string_view text = video->str();
    if (text.starts_with("data:")) {
      const auto comma = text.find(',');
      if (comma == std::string_view::npos ||
          text.substr(0, comma).find(";base64") == std::string_view::npos) {
        *error = {.code = "invalid_source_video",
                  .message = "'source_video' data URL must be base64"};
        return false;
      }
      text.remove_prefix(comma + 1);
    }
    std::string decoded;
    if (DecodeAudioBase64(text, kMaxSourceVideoBytes, &decoded))
      source->assign(decoded.begin(), decoded.end());
  }
  if (source->size() < 64) {
    *error = {.code = "invalid_source_video",
              .message =
                  "'source_video' must be an MP4 file, data URL or base64"};
    return false;
  }
  return true;
}

bool ValidateVideoPostProcess(const VideoPostProcess& post,
                              const VideoPostProcessors& tools,
                              std::string_view output_format,
                              std::string* error) {
  const VideoInterpolation& interpolation = post.interpolation;
  const bool interpolation_valid =
      interpolation.multiplier >= 1 && interpolation.multiplier <= 8 &&
      interpolation.fps_num >= 0 && interpolation.fps_den >= 1 &&
      interpolation.fps_den <= 1001 &&
      interpolation.fps_num <= 240 * interpolation.fps_den &&
      !(interpolation.multiplier > 1 && interpolation.fps_num > 0);
  if (!interpolation_valid ||
      (interpolation.enabled() &&
       (output_format != "mp4" || !tools.interpolator))) {
    *error = interpolation_valid && output_format == "mp4"
                 ? "frame interpolation is not enabled on this server (gufo "
                   "serve video --rife-model)"
                 : "'interpolate' must be 1..8 or a target fps with MP4 "
                   "output";
    return false;
  }
  const VideoUpscale& upscale = post.upscale;
  if (!upscale.enabled())
    return true;
  const bool shape = upscale.width >= 2 && upscale.height >= 2 &&
                     upscale.width <= 7680 && upscale.height <= 4320 &&
                     upscale.width % 2 == 0 && upscale.height % 2 == 0 &&
                     (upscale.fit == "pad" || upscale.fit == "crop" ||
                      upscale.fit == "stretch");
  const auto& models = tools.upscale_models;
  if (!shape || output_format != "mp4") {
    *error = "'upscale' needs an even width and height up to 7680x4320, fit "
             "pad, crop or stretch, and MP4 output";
    return false;
  }
  if (!tools.upscaler || models.empty()) {
    *error = "upscaling is not enabled on this server (gufo serve video "
             "--upscale-model)";
    return false;
  }
  if (!upscale.model.empty() &&
      std::ranges::find(models, upscale.model) == models.end()) {
    *error = "unknown upscale model '" + upscale.model + "'";
    return false;
  }
  return true;
}

bool RunVideoPostProcess(const VideoPostProcess& post,
                         const VideoPostProcessors& tools,
                         const std::filesystem::path& directory,
                         const std::string& job_id,
                         const ext::CancellationCheck& cancelled,
                         ext::VideoProgress progress, void* progress_opaque) {
  // output.mp4 -> generated.mp4 -> processed output.mp4.
  const auto stage =
      [&](const char* event, const std::string& detail,
          const std::function<bool(const std::filesystem::path&,
                                   const std::filesystem::path&,
                                   std::string*)>& run) {
        const auto generated = directory / "generated.mp4";
        const auto start = std::chrono::steady_clock::now();
        std::error_code rename_error;
        std::filesystem::rename(directory / "output.mp4", generated,
                                rename_error);
        std::string stage_error =
            rename_error ? "cannot stage the generated MP4" : "";
        bool ok = false;
        if (!rename_error) {
          try {
            ok = run(generated, directory / "output.mp4", &stage_error);
          } catch (const std::exception& exception) {
            stage_error = exception.what();
          }
        }
        std::filesystem::remove(generated, rename_error);
        if (!ok && !(cancelled && cancelled())) {
          Logger::Error("video", "job=" + job_id + " event=" + event +
                                     "_failed detail=" + stage_error);
        } else if (ok) {
          Logger::Info(
              "video",
              "job=" + job_id + " event=" + event + "d " + detail +
                  " elapsed_ms=" +
                  std::to_string(
                      std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count()));
        }
        return ok;
      };
  const VideoInterpolation& interpolation = post.interpolation;
  if (interpolation.enabled() &&
      !stage("interpolate",
             interpolation.fps_num > 0
                 ? "fps=" + std::to_string(interpolation.fps_num) + "/" +
                       std::to_string(interpolation.fps_den)
                 : "multiplier=" + std::to_string(interpolation.multiplier),
             [&](const std::filesystem::path& in,
                 const std::filesystem::path& out, std::string* error) {
               return tools.interpolator(in, out, interpolation, cancelled,
                                         progress, progress_opaque, error);
             }))
    return false;
  const VideoUpscale& upscale = post.upscale;
  if (upscale.enabled() &&
      !stage("upscale",
             "size=" + std::to_string(upscale.width) + "x" +
                 std::to_string(upscale.height) + " fit=" + upscale.fit +
                 " model=" +
                 (upscale.model.empty() && !tools.upscale_models.empty()
                      ? tools.upscale_models.front()
                      : upscale.model),
             [&](const std::filesystem::path& in,
                 const std::filesystem::path& out, std::string* error) {
               return tools.upscaler(in, out, upscale, cancelled, progress,
                                     progress_opaque, error);
             }))
    return false;
  return true;
}

std::optional<int> VideoPostProcessPercent(std::string_view phase,
                                           int completed, int total) {
  if (phase != "interpolate" && phase != "upscale")
    return std::nullopt;
  if (total <= 0)
    return 98;
  return 98 + std::clamp(completed, 0, total) / total;
}

}  // namespace gufo::server
