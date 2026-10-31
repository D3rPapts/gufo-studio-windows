#include "src/cli/serve/video_api.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/audio_stream.hpp"
#include "src/cli/serve/preview.hpp"
#include "src/core/json.hpp"
#include "src/core/platform/os_info.hpp"

namespace {

namespace h3 = gufo::minimax_h3;
using gufo::server::HandleVideoApiRequest;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;
using gufo::server::VideoJobService;
using gufo::server::VideoJobServiceOptions;
using gufo::server::VideoModelFamily;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL video_api_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

// A 0.25 s mono 16 kHz PCM16 WAV (a 440 Hz tone).
std::string TinyWav() {
  constexpr std::uint32_t kRate = 16000;
  constexpr std::uint32_t kSamples = 4000;
  std::string wav;
  const auto u32 = [&](std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
      wav.push_back(static_cast<char>((value >> shift) & 255U));
    }
  };
  const auto u16 = [&](std::uint16_t value) {
    wav.push_back(static_cast<char>(value & 255U));
    wav.push_back(static_cast<char>(value >> 8U));
  };
  wav += "RIFF";
  u32(36 + kSamples * 2);
  wav += "WAVEfmt ";
  u32(16);
  u16(1);
  u16(1);
  u32(kRate);
  u32(kRate * 2);
  u16(2);
  u16(16);
  wav += "data";
  u32(kSamples * 2);
  for (std::uint32_t index = 0; index < kSamples; ++index) {
    const double phase = 2.0 * 3.14159265358979 * 440.0 * index / kRate;
    u16(static_cast<std::uint16_t>(
        static_cast<std::int16_t>(std::sin(phase) * 8000.0)));
  }
  return wav;
}

std::string NextId() {
  static std::atomic_uint64_t counter{0};
  std::ostringstream output;
  output << "video_" << std::setw(24) << std::setfill('0') << std::hex
         << counter.fetch_add(1, std::memory_order_relaxed) + 1;
  return output.str();
}

class FakeRunner {
public:
  explicit FakeRunner(bool blocked = false) : blocked_(blocked) {}

  bool Run(const h3::GenerationRequest& request,
           const h3::CancellationToken* cancellation,
           h3::GenerationProgress progress, void* progress_opaque,
           h3::GenerationTelemetry*, std::string* error) {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      ++calls_;
      latest_ = request;
    }
    condition_.notify_all();
    progress("denoiser", 1, 2, progress_opaque);
    {
      std::unique_lock<std::mutex> lock(mutex_);
      while (blocked_ && !released_ &&
             (cancellation == nullptr || !cancellation->IsCancelled())) {
        condition_.wait_for(lock, std::chrono::milliseconds(5));
      }
    }
    if (cancellation != nullptr && cancellation->IsCancelled()) {
      if (error != nullptr) {
        *error = "cancelled";
      }
      return false;
    }

    std::error_code filesystem_error;
    if (request.parameters.mux) {
      std::filesystem::create_directories(request.output_path.parent_path(),
                                          filesystem_error);
      std::ofstream output(request.output_path, std::ios::binary);
      output << "fake-video-content";
    } else {
      std::filesystem::create_directories(request.frames_directory,
                                          filesystem_error);
      std::ostringstream filename;
      filename << "frame-" << std::setw(4) << std::setfill('0')
               << request.parameters.selected_frames.front() << ".ppm";
      std::ofstream output(request.frames_directory / filename.str(),
                           std::ios::binary);
      output << "P6\n1 1\n255\nabc";
    }
    return !filesystem_error;
  }

  void WaitForCalls(int expected) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!condition_.wait_for(lock, std::chrono::seconds(5),
                             [&] { return calls_ >= expected; })) {
      Fail("timed out waiting for fake runner");
    }
  }

  void Release() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      released_ = true;
    }
    condition_.notify_all();
  }

  h3::GenerationRequest latest() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return latest_;
  }

private:
  bool blocked_{false};
  bool released_{false};
  int calls_{0};
  h3::GenerationRequest latest_;
  std::mutex mutex_;
  std::condition_variable condition_;
};

VideoJobServiceOptions Options(const std::filesystem::path& root,
                               FakeRunner* runner) {
  return {
      .model_root = "/private/models/MiniMax-H3",
      .source_manifest = "/private/manifest.json",
      .storage_root = root,
      .queue_capacity = 1,
      .artifact_ttl = std::chrono::seconds(60),
      .validate_model_inventory = false,
      .id_factory = NextId,
      .now = [] { return 1000; },
      .runner =
          [runner](const h3::GenerationRequest& request,
                   const h3::CancellationToken* cancellation,
                   h3::GenerationProgress progress, void* progress_opaque,
                   h3::GenerationTelemetry* telemetry, std::string* error) {
            return runner->Run(request, cancellation, progress, progress_opaque,
                               telemetry, error);
          },
      .loras = {{"turbo",
                 {.path = "/private/loras/turbo.safetensors",
                  .steps = 8,
                  .strength = 1.0F}}},
  };
}

HttpResponse Send(VideoJobService& service, std::string method,
                  std::string path, std::string body = {},
                  std::string range = {}, std::string content_type = {}) {
  HttpRequest request{
      .method = std::move(method),
      .path = std::move(path),
      .query = {},
      .body = std::move(body),
      .headers = {},
      .is_cancelled = {},
  };
  if (!range.empty()) {
    request.headers.emplace_back("RaNgE", std::move(range));
  }
  if (!content_type.empty()) {
    request.headers.emplace_back("CoNtEnT-TyPe", std::move(content_type));
  }
  return HandleVideoApiRequest(request, service);
}

std::string Multipart(
    std::string_view boundary,
    const std::vector<std::pair<std::string, std::string>>& fields) {
  std::string body;
  for (const auto& [name, value] : fields) {
    body += "--";
    body += boundary;
    body += "\r\nContent-Disposition: form-data; name=\"";
    body += name;
    body += "\"\r\n\r\n";
    body += value;
    body += "\r\n";
  }
  body += "--";
  body += boundary;
  body += "--\r\n";
  return body;
}

std::string CreatedId(const HttpResponse& response) {
  const auto body = gufo::json::parse(response.body);
  return body.member_str("id");
}

void WaitCompleted(VideoJobService& service, std::string_view id) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    const HttpResponse response =
        Send(service, "GET", "/v1/videos/" + std::string(id));
    if (response.body.find("\"status\":\"completed\"") != std::string::npos) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  Fail("video API job did not complete");
}

void TestLifecycleAndRange(const std::filesystem::path& root) {
  FakeRunner runner;
  VideoJobService service(Options(root, &runner));
  Check(service.ready(), service.initialization_error());

  const std::string request =
      R"({"model":"minimax-h3-dev","prompt":"private fox prompt","size":"256x256","seconds":"1","gufo":{"seed":7,"frames":22,"output_format":"ppm","selected_frame":4}})";
  const HttpResponse created = Send(service, "POST", "/v1/videos", request, {},
                                    "application/json; charset=utf-8");
  Check(created.status == 202 &&
            created.body.find("private fox prompt") == std::string::npos &&
            created.body.find("\"status\":\"queued\"") != std::string::npos,
        "create returns a private, queued video object");
  const std::string id = CreatedId(created);
  Check(!id.empty(), "create response includes an ID");
  Check(created.log_details.find("job=" + id) != std::string::npos &&
            created.log_details.find("private fox prompt") == std::string::npos,
        "request diagnostics link the worker job without logging the prompt");
  runner.WaitForCalls(1);
  WaitCompleted(service, id);

  const h3::GenerationRequest generation = runner.latest();
  Check(generation.seed == 7 &&
            generation.parameters.preset == "development-256" &&
            generation.parameters.frames == 22 &&
            generation.parameters.selected_frames == std::vector<int>({4}) &&
            !generation.parameters.mux,
        "HTTP adapter preserves direct generation parameters");

  const HttpResponse content =
      Send(service, "GET", "/v1/videos/" + id + "/content");
  Check(content.status == 200 && content.body == "P6\n1 1\n255\nabc",
        "completed diagnostic frame is returned");

  const HttpResponse partial =
      Send(service, "GET", "/v1/videos/" + id + "/content", {}, "bytes=0-4");
  Check(partial.status == 206 && partial.body == "P6\n1 ",
        "single HTTP byte range is returned");
  Check(std::ranges::any_of(partial.headers,
                            [](const auto& header) {
                              return header.first == "Content-Range" &&
                                     header.second == "bytes 0-4/14";
                            }),
        "partial content includes a stable Content-Range");

  const HttpResponse invalid_range =
      Send(service, "GET", "/v1/videos/" + id + "/content", {}, "bytes=99-100");
  Check(invalid_range.status == 416, "unsatisfiable range returns HTTP 416");

  const HttpResponse deleted = Send(service, "DELETE", "/v1/videos/" + id);
  Check(deleted.status == 200 &&
            deleted.body.find("\"deleted\":true") != std::string::npos,
        "DELETE returns a deletion object");
  Check(Send(service, "GET", "/v1/videos/" + id).status == 404,
        "deleted jobs are no longer visible");
}

void TestValidation(const std::filesystem::path& root) {
  FakeRunner runner;
  VideoJobService service(Options(root, &runner));
  const HttpResponse fullres = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","prompt":"A goalkeeper saves a shot","size":"1344x768","seconds":5})");
  Check(fullres.status == 202, "released full-resolution API request accepted");
  runner.WaitForCalls(1);
  const h3::GenerationRequest fullres_generation = runner.latest();
  Check(fullres_generation.parameters.preset == "exact-1344x768" &&
            fullres_generation.parameters.internal_width == 1344 &&
            fullres_generation.parameters.internal_height == 768 &&
            fullres_generation.parameters.frames == 124 &&
            fullres_generation.parameters.evaluations == 49,
        "API maps 1344x768 five-second request to the exact preset");
  WaitCompleted(service, CreatedId(fullres));

  const HttpResponse reference = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","prompt":"x","seconds":"1","input_reference":"secret.png"})");
  Check(reference.status == 400 &&
            reference.body.find("invalid_input_reference") != std::string::npos,
        "keyframes that are not images fail closed");

  const HttpResponse references = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","prompt":"x","seconds":"1","references":[]})");
  Check(references.status == 400 &&
            references.body.find("unsupported_input_reference") !=
                std::string::npos,
        "ordered Ref2VA references stay unsupported");

  const HttpResponse keyframed = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","prompt":"x","size":"512x512","seconds":"5","first_frame":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGM4IccFRAwQCgAdRgPBqgMWjQAAAABJRU5ErkJggg==","last_frame":{"image_url":"iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGM4IccFRAwQCgAdRgPBqgMWjQAAAABJRU5ErkJggg=="},"gufo":{"lora":"turbo"}})");
  Check(keyframed.status == 202, "first and last frame request accepted");
  runner.WaitForCalls(2);
  const h3::GenerationRequest keyframed_generation = runner.latest();
  Check(keyframed_generation.keyframes.size() == 2 &&
            keyframed_generation.keyframes[0].anchor ==
                h3::KeyframeAnchor::kFirst &&
            keyframed_generation.keyframes[1].anchor ==
                h3::KeyframeAnchor::kLast &&
            keyframed_generation.keyframes[0].width == 2 &&
            keyframed_generation.keyframes[0].rgb.size() == 12 &&
            keyframed_generation.keyframes[0].rgb[0] == 200,
        "API decodes first and last keyframes");
  Check(keyframed_generation.lora_path ==
                std::filesystem::path("/private/loras/turbo.safetensors") &&
            keyframed_generation.parameters.evaluations == 8,
        "registered LoRA selects its file and default steps");
  WaitCompleted(service, CreatedId(keyframed));

  {
    // A FilmStudio shot: 832x512, first frame, an intermediate and a last
    // guide frame, and the voice as audio anchor at frame 0.
    const std::string audio = gufo::server::EncodeAudioBase64(TinyWav());
    const HttpResponse guided = Send(
        service, "POST", "/v1/videos",
        R"({"model":"minimax-h3","prompt":"x","size":"832x512","seconds":"5","first_frame":"iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGM4IccFRAwQCgAdRgPBqgMWjQAAAABJRU5ErkJggg==","audio_reference":"data:audio/wav;base64,)" +
            audio +
            R"(","gufo":{"lora":"turbo","keyframes":[{"frame":60,"image":"iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGM4IccFRAwQCgAdRgPBqgMWjQAAAABJRU5ErkJggg=="},{"frame":-1,"image":{"image_url":"iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGM4IccFRAwQCgAdRgPBqgMWjQAAAABJRU5ErkJggg=="}}]}})");
    Check(guided.status == 202,
          "832x512 guided request accepted: " + guided.body);
    runner.WaitForCalls(3);
    const h3::GenerationRequest guided_generation = runner.latest();
    Check(guided_generation.parameters.preset == "exact-832x512" &&
              guided_generation.parameters.internal_width == 832 &&
              guided_generation.parameters.internal_height == 512 &&
              guided_generation.parameters.output_width == 832 &&
              guided_generation.parameters.output_height == 512,
          "832x512 maps to the exact-832x512 preset");
    Check(guided_generation.keyframes.size() == 3 &&
              guided_generation.keyframes[0].vision &&
              guided_generation.keyframes[1].frame == 60 &&
              !guided_generation.keyframes[1].vision &&
              guided_generation.keyframes[2].frame == -1 &&
              guided_generation.keyframes[1].rgb.size() == 12,
          "guide frames keep their frame index and skip the vision prompt");
    Check(guided_generation.audio_guides.size() == 1 &&
              guided_generation.audio_guides[0].frame == 0 &&
              guided_generation.audio_guides[0].waveform.sample_rate == 32000 &&
              guided_generation.audio_guides[0].waveform.channels == 2 &&
              guided_generation.audio_guides[0].waveform.samples == 8000 &&
              guided_generation.audio_guides[0].waveform.pcm[100] ==
                  guided_generation.audio_guides[0].waveform.pcm[8100],
          "audio_reference decodes to 32 kHz stereo at frame 0");
    WaitCompleted(service, CreatedId(guided));

    const HttpResponse same_frame = Send(
        service, "POST", "/v1/videos",
        R"({"model":"minimax-h3","prompt":"x","seconds":"5","first_frame":"iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGM4IccFRAwQCgAdRgPBqgMWjQAAAABJRU5ErkJggg==","gufo":{"keyframes":[{"frame":0,"image":"iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGM4IccFRAwQCgAdRgPBqgMWjQAAAABJRU5ErkJggg=="}]}})");
    Check(same_frame.status == 400 &&
              same_frame.body.find("invalid_keyframes") != std::string::npos,
          "two images cannot anchor the same frame");
    const HttpResponse outside = Send(
        service, "POST", "/v1/videos",
        R"({"model":"minimax-h3","prompt":"x","seconds":"5","gufo":{"keyframes":[{"frame":500,"image":"iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGM4IccFRAwQCgAdRgPBqgMWjQAAAABJRU5ErkJggg=="}]}})");
    Check(outside.status == 400, "guide frames must lie inside the video");
    const HttpResponse bad_audio = Send(
        service, "POST", "/v1/videos",
        R"({"model":"minimax-h3","prompt":"x","seconds":"5","audio_reference":"bm90IGF1ZGlv"})");
    Check(
        bad_audio.status == 400 &&
            bad_audio.body.find("invalid_audio_reference") != std::string::npos,
        "audio_reference must decode as audio");
    const HttpResponse bad_size = Send(
        service, "POST", "/v1/videos",
        R"({"model":"minimax-h3","prompt":"x","size":"800x600","seconds":"5"})");
    Check(bad_size.status == 400 &&
              bad_size.body.find("invalid_size") != std::string::npos,
          "sizes off the 32-pixel grid are rejected");
  }

  const HttpResponse unknown_lora = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","prompt":"x","seconds":"5","gufo":{"lora":"/etc/passwd"}})");
  Check(unknown_lora.status == 400, "only registered LoRA names are accepted");

  const HttpResponse duplicate_first = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","prompt":"x","seconds":"5","input_reference":"iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGM4IccFRAwQCgAdRgPBqgMWjQAAAABJRU5ErkJggg==","first_frame":"iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGM4IccFRAwQCgAdRgPBqgMWjQAAAABJRU5ErkJggg=="})");
  Check(duplicate_first.status == 400,
        "input_reference and first_frame cannot both anchor the start");

  const HttpResponse bad_output = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","prompt":"x","size":"512x512","seconds":"1","gufo":{"preset":"dev","output_format":"mp4"}})");
  Check(bad_output.status == 400,
        "development preset cannot silently produce MP4");

  const HttpResponse mismatch = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3-fast","prompt":"x","size":"512x512","seconds":"1","gufo":{"preset":"exact"}})");
  Check(mismatch.status == 400 &&
            mismatch.body.find("preset_model_mismatch") != std::string::npos,
        "model aliases cannot drift from the frozen preset");

  const HttpResponse too_short = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3-dev","prompt":"x","size":"256x256","seconds":"1","gufo":{"frames":5,"output_format":"ppm","selected_frame":4}})");
  Check(too_short.status == 400 &&
            too_short.body.find("invalid_frames") != std::string::npos &&
            too_short.body.find("at least 22 aligned frames") !=
                std::string::npos,
        "sub-chunk requests fail before the released VisualVAE is invoked");

  const HttpResponse unknown = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","prompt":"x","seconds":"1","silent_typo":true})");
  Check(unknown.status == 400 &&
            unknown.body.find("unsupported_field") != std::string::npos,
        "unknown top-level fields fail closed");

  const HttpResponse duplicate_json = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","model":"minimax-h3-fast","prompt":"x","seconds":"1"})");
  Check(duplicate_json.status == 400 &&
            duplicate_json.body.find("parse_error") != std::string::npos,
        "duplicate JSON fields fail closed");

  const HttpResponse trailing_json =
      Send(service, "POST", "/v1/videos",
           R"({"model":"minimax-h3","prompt":"x","seconds":"1"} trailing)");
  Check(trailing_json.status == 400 &&
            trailing_json.body.find("parse_error") != std::string::npos,
        "trailing JSON input fails closed");

  const HttpResponse malformed_number = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","prompt":"x","seconds":"1","gufo":{"seed":1e}})");
  Check(malformed_number.status == 400 &&
            malformed_number.body.find("parse_error") != std::string::npos,
        "malformed JSON numbers fail closed");

  const HttpResponse wrong_size_type =
      Send(service, "POST", "/v1/videos",
           R"({"model":"minimax-h3","prompt":"x","size":512,"seconds":"1"})");
  Check(wrong_size_type.status == 400 &&
            wrong_size_type.body.find("invalid_parameter_type") !=
                std::string::npos,
        "wrongly typed optional fields do not fall back to defaults");

  Check(Send(service, "GET", "/v1/videos/../../private").status == 404,
        "path traversal-like IDs fail closed");

  constexpr std::string_view kBoundary = "gufo-openai-video-boundary";
  const HttpResponse multipart =
      Send(service, "POST", "/v1/videos",
           Multipart(kBoundary,
                     {
                         {"model", "minimax-h3-fast"},
                         {"prompt", "A red fox walking through snow"},
                         {"size", "512x512"},
                         {"seconds", "1"},
                     }),
           {}, "multipart/form-data; boundary=\"gufo-openai-video-boundary\"");
  Check(multipart.status == 202,
        "OpenAI-style multipart text-to-video request is admitted");
  runner.WaitForCalls(4);
  Check(runner.latest().parameters.preset == "fast-384",
        "multipart request preserves the selected H3 mode");

  const HttpResponse duplicate =
      Send(service, "POST", "/v1/videos",
           Multipart(kBoundary,
                     {
                         {"model", "minimax-h3"},
                         {"prompt", "first"},
                         {"prompt", "second"},
                         {"seconds", "1"},
                     }),
           {}, "multipart/form-data; boundary=gufo-openai-video-boundary");
  Check(duplicate.status == 400 &&
            duplicate.body.find("invalid_multipart_form") != std::string::npos,
        "multipart duplicate fields fail closed");

  const std::string reference_body =
      "--gufo-openai-video-boundary\r\n"
      "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
      "minimax-h3\r\n"
      "--gufo-openai-video-boundary\r\n"
      "Content-Disposition: form-data; name=\"prompt\"\r\n\r\n"
      "x\r\n"
      "--gufo-openai-video-boundary\r\n"
      "Content-Disposition: form-data; name=\"seconds\"\r\n\r\n"
      "1\r\n"
      "--gufo-openai-video-boundary\r\n"
      "Content-Disposition: form-data; name=\"input_reference\"; "
      "filename=\"frame.png\"\r\n"
      "Content-Type: image/png\r\n\r\n"
      "not-an-image\r\n"
      "--gufo-openai-video-boundary--\r\n";
  const HttpResponse multipart_reference =
      Send(service, "POST", "/v1/videos", reference_body, {},
           "multipart/form-data; boundary=gufo-openai-video-boundary");
  Check(multipart_reference.status == 400 &&
            multipart_reference.body.find("invalid_input_reference") !=
                std::string::npos,
        "multipart keyframe files must decode as images");

  const HttpResponse unsupported_media =
      Send(service, "POST", "/v1/videos", "model=minimax-h3", {},
           "application/x-www-form-urlencoded");
  Check(unsupported_media.status == 415 &&
            unsupported_media.body.find("unsupported_media_type") !=
                std::string::npos,
        "unsupported create content types fail explicitly");
}

void TestDurations(const std::filesystem::path& root) {
  FakeRunner runner;
  VideoJobService service(Options(root, &runner));
  int calls = 0;
  for (const auto size : {"512x512", "1344x768"}) {
    for (const auto& [seconds, frames] :
         std::vector<std::pair<std::string, int>>{
             {"5", 124}, {"8", 192}, {"10", 243}, {"14.375", 345}}) {
      const auto response =
          Send(service, "POST", "/v1/videos",
               "{\"model\":\"minimax-h3\",\"prompt\":\"x\",\"size\":\"" +
                   std::string(size) + "\",\"seconds\":" + seconds + "}");
      Check(response.status == 202,
            "numeric aligned duration accepted: " + response.body);
      runner.WaitForCalls(++calls);
      WaitCompleted(service, CreatedId(response));
      Check(runner.latest().parameters.frames == frames,
            "duration reaches generation with official aligned frames");
    }
  }
  const std::string boundary = "h3-duration";
  const auto multipart = Send(service, "POST", "/v1/videos",
                              Multipart(boundary, {{"model", "minimax-h3"},
                                                   {"prompt", "x"},
                                                   {"size", "1344x768"},
                                                   {"seconds", "14.375"}}),
                              {}, "multipart/form-data; boundary=" + boundary);
  Check(multipart.status == 202, "multipart accepts fractional duration");
  runner.WaitForCalls(++calls);
  WaitCompleted(service, CreatedId(multipart));
  Check(runner.latest().parameters.frames == 345,
        "multipart resolves the maximum aligned duration");

  for (const auto seconds : {"0", "4", "14.376", "15", "16"}) {
    const auto response =
        Send(service, "POST", "/v1/videos",
             "{\"model\":\"minimax-h3\",\"prompt\":\"x\",\"seconds\":\"" +
                 std::string(seconds) + "\"}");
    Check(response.status == 400 &&
              response.body.find("invalid_seconds") != std::string::npos,
          "unsupported duration fails before job admission");
  }
  const auto mismatch = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","prompt":"x","seconds":10,"gufo":{"frames":124}})");
  Check(mismatch.status == 400 &&
            mismatch.body.find("invalid_frames") != std::string::npos,
        "explicit frames cannot disagree with duration");
}

void TestLongPrompts(const std::filesystem::path& root) {
  FakeRunner runner;
  VideoJobService service(Options(root, &runner));
  std::string unicode;
  for (int i = 0; i < 2048; ++i)
    unicode += "e\xCC\x81";
  int calls = 0;
  for (const auto& prompt : {std::string(8192, 'A'), unicode}) {
    for (const bool multipart : {false, true}) {
      std::string body;
      std::string content_type;
      if (multipart) {
        body = Multipart(
            "long-prompt",
            {{"model", "minimax-h3"}, {"prompt", prompt}, {"seconds", "1"}});
        content_type = "multipart/form-data; boundary=long-prompt";
      } else {
        auto json = gufo::json::Value::object();
        json["model"] = "minimax-h3";
        json["prompt"] = prompt;
        json["seconds"] = "1";
        body = json.dump();
        content_type = "application/json";
      }
      const auto response =
          Send(service, "POST", "/v1/videos", body, {}, content_type);
      Check(response.status == 202,
            "byte length does not replace the encoder's token limit");
      runner.WaitForCalls(++calls);
      Check(runner.latest().prompt == prompt,
            "HTTP and job admission preserve the entire UTF-8 prompt");
      WaitCompleted(service, CreatedId(response));
    }
  }
}

void TestQueueSaturation(const std::filesystem::path& root) {
  FakeRunner runner(true);
  VideoJobService service(Options(root, &runner));
  const std::string body =
      R"({"model":"minimax-h3","prompt":"x","size":"512x512","seconds":"1","gufo":{"preset":"fast"}})";
  const HttpResponse first = Send(service, "POST", "/v1/videos", body);
  runner.WaitForCalls(1);
  const HttpResponse second = Send(service, "POST", "/v1/videos", body);
  const HttpResponse third = Send(service, "POST", "/v1/videos", body);
  Check(first.status == 202 && second.status == 202 && third.status == 429,
        "one active and one queued request enforce bounded admission");
  Check(std::ranges::any_of(
            third.headers,
            [](const auto& header) { return header.first == "Retry-After"; }),
        "queue saturation supplies Retry-After");
  Check(Send(service, "DELETE", "/v1/videos/" + CreatedId(first)).status == 200,
        "active request can be cancelled through DELETE");
  runner.WaitForCalls(2);
  runner.Release();
}

// Wan 2.2: first frame (JSON base64 and multipart upload), negative prompt
// and guidance reach the runner; last frames and unknown options do not.
void TestWan22Options(const std::filesystem::path& root) {
  FakeRunner runner;
  VideoJobServiceOptions options = Options(root, &runner);
  options.family = VideoModelFamily::kWan22;
  options.model_root = "/private/models/Wan2.2-TI2V-5B";
  VideoJobService service(std::move(options));
  Check(service.ready(), service.initialization_error());
  // A 1x1 RGBA PNG.
  const std::string png =
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
      "hK"
      "mMIQAAAABJRU5ErkJggg==";
  const std::string body =
      R"({"model":"wan2.2-ti2v-5b","prompt":"fox","size":"480x480","seconds":1,"input_reference":")" +
      png +
      R"(","gufo":{"seed":3,"steps":4,"negative_prompt":"blurry","guidance":3.5}})";
  const HttpResponse created = Send(service, "POST", "/v1/videos", body);
  Check(created.status == 202,
        "Wan image-to-video request is accepted: " + created.body);
  runner.WaitForCalls(1);
  WaitCompleted(service, CreatedId(created));
  h3::GenerationRequest generation = runner.latest();
  Check(generation.first_frame_width == 1 &&
            generation.first_frame_height == 1 &&
            generation.first_frame.size() == 3 &&
            generation.negative_prompt == "blurry" &&
            generation.guidance == 3.5F && generation.seed == 3 &&
            generation.parameters.evaluations == 4 &&
            generation.parameters.output_width == 480,
        "Wan first frame, negative prompt and guidance reach the runner");

  const HttpResponse plain = Send(
      service, "POST", "/v1/videos",
      R"({"model":"wan2.2-ti2v-5b","prompt":"fox","size":"480x480","seconds":1})");
  Check(plain.status == 202, "Wan text-to-video request is accepted");
  runner.WaitForCalls(2);
  WaitCompleted(service, CreatedId(plain));
  generation = runner.latest();
  Check(generation.first_frame.empty() && generation.negative_prompt.empty() &&
            generation.guidance == 5.0F,
        "Wan defaults: text-to-video, empty negative prompt, guidance 5");

  // Decoded PNG bytes as a multipart file upload.
  std::string file;
  {
    static constexpr std::string_view kAlphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::uint32_t buffer = 0;
    int bits = 0;
    for (const char c : png) {
      const auto index = kAlphabet.find(c);
      if (index == std::string_view::npos)
        continue;
      buffer = (buffer << 6U) | static_cast<std::uint32_t>(index);
      bits += 6;
      if (bits >= 8) {
        bits -= 8;
        file += static_cast<char>((buffer >> bits) & 0xFFU);
      }
    }
  }
  const std::string boundary = "wan-boundary";
  std::string multipart = Multipart(boundary, {{"model", "wan2.2-ti2v-5b"},
                                               {"prompt", "fox"},
                                               {"size", "480x480"},
                                               {"seconds", "1"}});
  multipart.erase(multipart.size() - (boundary.size() + 6));
  multipart += "--" + boundary +
               "\r\nContent-Disposition: form-data; name=\"first_frame\"; "
               "filename=\"f.png\"\r\nContent-Type: image/png\r\n\r\n" +
               file + "\r\n--" + boundary + "--\r\n";
  const HttpResponse uploaded =
      Send(service, "POST", "/v1/videos", multipart, {},
           "multipart/form-data; boundary=" + boundary);
  Check(uploaded.status == 202,
        "Wan multipart first frame is accepted: " + uploaded.body);
  runner.WaitForCalls(3);
  WaitCompleted(service, CreatedId(uploaded));
  Check(runner.latest().first_frame.size() == 3,
        "Wan multipart first frame reaches the runner");

  for (
      const std::string bad :
      {R"({"model":"wan2.2-ti2v-5b","prompt":"fox","size":"480x480","seconds":1,"last_frame":"x"})",
       R"({"model":"wan2.2-ti2v-5b","prompt":"fox","size":"480x480","seconds":1,"input_reference":"!!"})",
       R"({"model":"wan2.2-ti2v-5b","prompt":"fox","size":"480x480","seconds":1,"gufo":{"guidance":"5"}})",
       R"({"model":"wan2.2-ti2v-5b","prompt":"fox","size":"480x480","seconds":1,"gufo":{"guidance":50}})",
       R"({"model":"wan2.2-ti2v-5b","prompt":"fox","size":"480x480","seconds":1,"gufo":{"negative_prompt":1}})",
       R"({"model":"wan2.2-ti2v-5b","prompt":"fox","size":"480x480","seconds":1,"gufo":{"lora":"x"}})"}) {
    Check(Send(service, "POST", "/v1/videos", bad).status == 400,
          "invalid Wan request is rejected: " + bad);
  }
}

// Wan 2.1 Uni3C camera moves: "gufo.preview" is accepted, the runner gets a
// hook that keeps the dense default (every step but the last of the 4-step
// LCM schedule), and a 16-channel latent becomes a preview loop under the
// job id. A bad preview object is refused.
void TestWan21Preview(const std::filesystem::path& root) {
  FakeRunner runner;
  VideoJobServiceOptions options = Options(root, &runner);
  options.family = VideoModelFamily::kWan21Uni3C;
  options.model_root = "/private/models/wan21-camera";
  std::mutex mutex;
  std::vector<int> wanted;
  options.runner = [&](const h3::GenerationRequest& request,
                       const h3::CancellationToken*, h3::GenerationProgress,
                       void*, h3::GenerationTelemetry*, std::string*) {
    // 2 latent frames of 2x3 pixels, [16][2][2][3]; the frames differ (an
    // animated WebP drops identical frames).
    std::vector<float> latent(16 * 2 * 2 * 3, 0.0F);
    for (std::size_t i = 0; i < latent.size(); ++i)
      latent[i] = (i / 6) % 2 == 0 ? -1.0F : 1.0F;
    for (int step = 1; step <= 4; ++step) {
      if (!request.preview.Wants(step, 4))
        continue;
      {
        const std::lock_guard<std::mutex> lock(mutex);
        wanted.push_back(step);
      }
      const auto encoded = gufo::server::PreviewHub::Global().stats().encoded;
      request.preview.publish({.step = step, .steps = 4, .frames = 2,
                               .height = 2, .width = 3, .channels = 16,
                               .channel_major = true, .data = latent});
      // Real steps are minutes apart: let the hub encode this preview
      // before the next step asks (a queued preview defers the next one).
      const auto until =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (gufo::server::PreviewHub::Global().stats().encoded == encoded &&
             std::chrono::steady_clock::now() < until)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::error_code ignored;
    std::filesystem::create_directories(request.output_path.parent_path(),
                                        ignored);
    std::ofstream(request.output_path, std::ios::binary) << "fake-video";
    return true;
  };
  VideoJobService service(std::move(options));
  Check(service.ready(), service.initialization_error());
  const std::string png =
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
      "hKmMIQAAAABJRU5ErkJggg==";
  const std::string prefix =
      R"({"model":"wan2.1-uni3c","prompt":"orbit","size":"832x480","seconds":"1","first_frame":")" +
      png + R"(","gufo":{"camera":{"type":"orbit_left"},)";
  const HttpResponse created =
      Send(service, "POST", "/v1/videos",
           prefix + R"("preview":{"always":true}}})");
  Check(created.status == 202,
        "Wan 2.1 request with gufo.preview is accepted: " + created.body);
  const std::string id = CreatedId(created);
  WaitCompleted(service, id);
  {
    const std::lock_guard<std::mutex> lock(mutex);
    Check(wanted == std::vector<int>({1, 2, 3}),
          "Wan 2.1 previews steps 1..3 of 4 (dense, never the last)");
  }
  HttpRequest poll;
  poll.method = "GET";
  poll.path = "/v1/previews/" + id;
  HttpResponse preview;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  do {
    preview = gufo::server::HandlePreviewRequest(
        poll, gufo::server::PreviewHub::Global());
    if (preview.status == 200)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  } while (std::chrono::steady_clock::now() < deadline);
  Check(preview.status == 200 && preview.body.starts_with("RIFF") &&
            preview.body.find("ANIM") != std::string::npos,
        "Wan 2.1 preview is an animated WebP under the job id");

  for (const char* bad : {R"("preview":{"id":"x"}}})",
                          R"("preview":{"every":0}}})",
                          R"("preview":7}})"}) {
    const HttpResponse rejected =
        Send(service, "POST", "/v1/videos", prefix + bad);
    Check(rejected.status == 400 &&
              rejected.body.find("invalid_preview") != std::string::npos,
          std::string("invalid Wan 2.1 preview rejected: ") + bad);
  }
}

}  // namespace
void TestDelivery(const std::filesystem::path& root) {
  FakeRunner runner;
  auto options = Options(root, &runner);
  std::mutex mutex;
  std::vector<gufo::server::VideoInterpolation> interpolations;
  std::vector<gufo::server::VideoUpscale> upscales;
  const auto copy = [](const std::filesystem::path& in,
                       const std::filesystem::path& out) {
    std::error_code ignored;
    std::filesystem::copy_file(in, out, ignored);
    return !ignored;
  };
  options.interpolator = [&](const std::filesystem::path& in,
                             const std::filesystem::path& out,
                             const gufo::server::VideoInterpolation& spec,
                             const h3::CancellationToken*,
                             h3::GenerationProgress, void*, std::string*) {
    const std::lock_guard<std::mutex> lock(mutex);
    interpolations.push_back(spec);
    return copy(in, out);
  };
  options.upscaler = [&](const std::filesystem::path& in,
                         const std::filesystem::path& out,
                         const gufo::server::VideoUpscale& spec,
                         const h3::CancellationToken*, h3::GenerationProgress,
                         void*, std::string*) {
    const std::lock_guard<std::mutex> lock(mutex);
    upscales.push_back(spec);
    return copy(in, out);
  };
  options.upscale_models = {"anime", "general"};
  VideoJobService service(std::move(options));

  const HttpResponse both = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","prompt":"x","seconds":"1","gufo":{"interpolate":{"fps":60},"upscale":{"width":1920,"height":1200,"model":"general"}}})");
  Check(both.status == 202, "target fps and upscale request accepted");
  WaitCompleted(service, CreatedId(both));
  {
    const std::lock_guard<std::mutex> lock(mutex);
    Check(interpolations.size() == 1 && interpolations[0].fps_num == 60 &&
              interpolations[0].fps_den == 1 &&
              interpolations[0].multiplier == 1,
          "API passes the target frame rate");
    Check(upscales.size() == 1 && upscales[0].width == 1920 &&
              upscales[0].height == 1200 && upscales[0].fit == "pad" &&
              upscales[0].model == "general",
          "API passes the upscale size, default fit and model");
  }

  const HttpResponse ntsc = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","prompt":"x","seconds":"1","gufo":{"interpolate":{"fps":"60000/1001"},"upscale":"1920x1080"}})");
  Check(ntsc.status == 202, "fractional target fps and WxH upscale accepted");
  WaitCompleted(service, CreatedId(ntsc));
  {
    const std::lock_guard<std::mutex> lock(mutex);
    Check(interpolations.size() == 2 && interpolations[1].fps_num == 60000 &&
              interpolations[1].fps_den == 1001,
          "API parses N/D target frame rates");
    Check(upscales.size() == 2 && upscales[1].width == 1920 &&
              upscales[1].height == 1080 && upscales[1].model.empty(),
          "API parses WxH upscale strings");
  }

  for (const char* bad :
       {R"({"interpolate":{"fps":0}})", R"({"interpolate":{"fps":241}})",
        R"({"interpolate":{"fps":"60/0"}})",
        R"({"interpolate":{"fps":60,"multiplier":2}})",
        R"({"upscale":{"width":1920}})", R"({"upscale":"big"})",
        R"({"upscale":{"width":1920,"height":1200,"fit":7}})"}) {
    const HttpResponse rejected = Send(
        service, "POST", "/v1/videos",
        std::string(R"({"model":"minimax-h3","prompt":"x","seconds":"1","gufo":)") +
            bad + "}");
    Check(rejected.status == 400 &&
              (rejected.body.find("invalid_interpolate") != std::string::npos ||
               rejected.body.find("invalid_upscale") != std::string::npos),
          std::string("malformed delivery option rejected: ") + bad);
  }
  for (const char* bad : {R"({"upscale":"1921x1200"})",
                          R"({"upscale":{"width":1920,"height":1200,"fit":"zoom"}})",
                          R"({"upscale":{"width":1920,"height":1200,"model":"x"}})"}) {
    const HttpResponse rejected = Send(
        service, "POST", "/v1/videos",
        std::string(R"({"model":"minimax-h3","prompt":"x","seconds":"1","gufo":)") +
            bad + "}");
    Check(rejected.status == 400, std::string("invalid upscale rejected: ") + bad);
  }
}

int main() {
  const std::filesystem::path base =
      std::filesystem::temp_directory_path() /
      ("gufo-video-api-test-" + std::to_string(gufo::platform::ProcessId()));
  std::error_code ignored;
  std::filesystem::remove_all(base, ignored);
  std::filesystem::create_directories(base);
  TestLifecycleAndRange(base / "lifecycle");
  TestValidation(base / "validation");
  TestDurations(base / "durations");
  TestLongPrompts(base / "long-prompts");
  TestQueueSaturation(base / "queue");
  TestWan22Options(base / "wan22");
  TestWan21Preview(base / "wan21");
  TestDelivery(base / "delivery");
  std::filesystem::remove_all(base, ignored);
  std::cout << "Video API contract tests passed.\n";
  return 0;
}
