// Written with AI assistance (Claude Code).
// Wan video providers on the 0.2 video capability: /v1/videos requests in the
// 1.x formats go through the core video API (provider-owned parsing), reach
// the runner as options + attachments, and decode into the job the HIP
// runners take. Ports the Wan cases of the fork's tests/cli/video_api_test
// (Wan 2.2 options, VACE, Uni3C camera) and adds InfiniteTalk (one and two
// speakers, add / para / auto, video to video, 896x512) and provider probing.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/video.hpp>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/video_api.hpp"
#include "src/core/json.hpp"
#include "src/core/platform/os_info.hpp"
#include "src/models/wan22/request.hpp"

namespace {

namespace wan22 = gufo::models::wan22;
using gufo::ext::VideoProgress;
using gufo::ext::VideoRequest;
using gufo::ext::VideoTelemetry;
using gufo::server::HandleVideoApiRequest;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;
using gufo::server::VideoJobService;
using gufo::server::VideoJobServiceOptions;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL wan22_video_request_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition)
    Fail(message);
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
  bool Run(const VideoRequest& request, VideoProgress progress,
           void* progress_opaque) {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      ++calls_;
      latest_ = request;
    }
    condition_.notify_all();
    progress("denoise", 1, 2, progress_opaque);
    std::error_code ignored;
    std::filesystem::create_directories(request.output_path.parent_path(),
                                        ignored);
    std::ofstream(request.output_path, std::ios::binary) << "fake-video";
    return true;
  }

  void WaitForCalls(int expected) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!condition_.wait_for(lock, std::chrono::seconds(5),
                             [&] { return calls_ >= expected; }))
      Fail("timed out waiting for fake runner");
  }

  VideoRequest latest() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return latest_;
  }

private:
  int calls_{0};
  VideoRequest latest_;
  std::mutex mutex_;
  std::condition_variable condition_;
};

VideoJobServiceOptions Options(const std::filesystem::path& root,
                               std::string provider, FakeRunner* runner) {
  return {
      .provider_id = std::move(provider),
      .model_root = "/private/models/wan",
      .source_manifest = "/private/manifest.json",
      .storage_root = root,
      .queue_capacity = 4,
      .artifact_ttl = std::chrono::seconds(60),
      .model_aliases = {},
      .validate_model_inventory = false,
      .id_factory = NextId,
      .now = [] { return 1000; },
      .runner =
          [runner](const VideoRequest& request,
                   const gufo::ext::CancellationCheck&, VideoProgress progress,
                   void* progress_opaque, VideoTelemetry*, std::string*) {
            return runner->Run(request, progress, progress_opaque);
          },
  };
}

HttpResponse Send(VideoJobService& service, std::string body,
                  std::string content_type = {}) {
  HttpRequest request{
      .method = "POST",
      .path = "/v1/videos",
      .query = {},
      .body = std::move(body),
      .headers = {},
      .is_cancelled = {},
  };
  if (!content_type.empty())
    request.headers.emplace_back("Content-Type", std::move(content_type));
  return HandleVideoApiRequest(request, service);
}

HttpResponse Get(VideoJobService& service, const std::string& id) {
  HttpRequest request{
      .method = "GET",
      .path = "/v1/videos/" + id,
      .query = {},
      .body = {},
      .headers = {},
      .is_cancelled = {},
  };
  return HandleVideoApiRequest(request, service);
}

std::string CreatedId(const HttpResponse& response) {
  return gufo::json::parse(response.body).member_str("id");
}

void WaitCompleted(VideoJobService& service, const std::string& id) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    if (Get(service, id).body.find("\"status\":\"completed\"") !=
        std::string::npos)
      return;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  Fail("video job did not complete: " + Get(service, id).body);
}

// Accepts `body`, waits for the job and decodes what the runner got.
wan22::JobRequest Accept(VideoJobService& service, FakeRunner& runner,
                         int* calls, wan22::Family family,
                         const std::string& body,
                         const std::string& content_type = {},
                         VideoRequest* raw = nullptr) {
  const HttpResponse created = Send(service, body, content_type);
  Check(created.status == 202, "request is accepted: " + created.body);
  runner.WaitForCalls(++*calls);
  WaitCompleted(service, CreatedId(created));
  const VideoRequest request = runner.latest();
  if (raw != nullptr)
    *raw = request;
  wan22::JobRequest job;
  std::string error;
  Check(wan22::BuildJobRequest(family, request, &job, &error),
        "runner request decodes: " + error);
  return job;
}

void Rejected(VideoJobService& service, const std::string& body,
              std::string_view code, const std::string& what) {
  const HttpResponse response = Send(service, body);
  Check(response.status == 400 &&
            (code.empty() || response.body.find("\"" + std::string(code) +
                                                "\"") != std::string::npos),
        what + " is rejected" +
            (code.empty() ? "" : " with " + std::string(code)) + ": " +
            std::to_string(response.status) + " " + response.body);
}

std::string Base64(std::string_view bytes) {
  static constexpr std::string_view kAlphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  std::size_t i = 0;
  for (; i + 2 < bytes.size(); i += 3) {
    const std::uint32_t v = (static_cast<std::uint8_t>(bytes[i]) << 16U) |
                            (static_cast<std::uint8_t>(bytes[i + 1]) << 8U) |
                            static_cast<std::uint8_t>(bytes[i + 2]);
    for (int s = 18; s >= 0; s -= 6)
      out += kAlphabet[(v >> s) & 63U];
  }
  if (i < bytes.size()) {
    std::uint32_t v = static_cast<std::uint8_t>(bytes[i]) << 16U;
    if (i + 1 < bytes.size())
      v |= static_cast<std::uint8_t>(bytes[i + 1]) << 8U;
    out += kAlphabet[(v >> 18) & 63U];
    out += kAlphabet[(v >> 12) & 63U];
    out += i + 1 < bytes.size() ? kAlphabet[(v >> 6) & 63U] : '=';
    out += '=';
  }
  return out;
}

std::string Unbase64(std::string_view text) {
  static constexpr std::string_view kAlphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  std::uint32_t buffer = 0;
  int bits = 0;
  for (const char c : text) {
    const auto index = kAlphabet.find(c);
    if (index == std::string_view::npos)
      continue;
    buffer = (buffer << 6U) | static_cast<std::uint32_t>(index);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out += static_cast<char>((buffer >> bits) & 0xFFU);
    }
  }
  return out;
}

// A 1x1 RGBA PNG.
constexpr std::string_view kPng =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
    "hKmMIQAAAABJRU5ErkJggg==";

// 16-bit mono PCM WAV of a 440 Hz tone.
std::string Wav(double seconds, int rate) {
  const int samples = static_cast<int>(std::lround(seconds * rate));
  std::string data;
  for (int i = 0; i < samples; ++i) {
    const auto s = static_cast<std::int16_t>(
        std::lround(8000.0 * std::sin(2.0 * 3.14159265358979 * 440.0 * i /
                                      static_cast<double>(rate))));
    data.append(reinterpret_cast<const char*>(&s), 2);
  }
  const auto u32 = [](std::uint32_t v) {
    return std::string(reinterpret_cast<const char*>(&v), 4);
  };
  const auto u16 = [](std::uint16_t v) {
    return std::string(reinterpret_cast<const char*>(&v), 2);
  };
  return "RIFF" + u32(static_cast<std::uint32_t>(36 + data.size())) +
         "WAVEfmt " + u32(16) + u16(1) + u16(1) +
         u32(static_cast<std::uint32_t>(rate)) +
         u32(static_cast<std::uint32_t>(rate * 2)) + u16(2) + u16(16) +
         "data" + u32(static_cast<std::uint32_t>(data.size())) + data;
}

std::string WavUrl(double seconds) {
  return "data:audio/wav;base64," + Base64(Wav(seconds, 16000));
}

struct Part {
  std::string name;
  std::string value;
  bool file{false};
};

std::string Multipart(std::string_view boundary,
                      const std::vector<Part>& parts) {
  std::string body;
  for (const auto& part : parts) {
    body += "--" + std::string(boundary) +
            "\r\nContent-Disposition: form-data; name=\"" + part.name + "\"";
    if (part.file)
      body += "; filename=\"" + part.name + ".bin\"";
    body += "\r\n\r\n" + part.value + "\r\n";
  }
  body += "--" + std::string(boundary) + "--\r\n";
  return body;
}

// Wan 2.2: first frame (JSON base64 and multipart upload), negative prompt
// and guidance reach the runner; last frames and unknown options do not.
void TestWan22(const std::filesystem::path& root) {
  FakeRunner runner;
  VideoJobService service(Options(root, "wan2.2-ti2v-5b", &runner));
  Check(service.ready(), service.initialization_error());
  Check(service.model_id() == "wan2.2-ti2v-5b", "Wan 2.2 model id");
  int calls = 0;
  wan22::JobRequest job = Accept(
      service, runner, &calls, wan22::Family::kWan22,
      R"({"model":"wan2.2-ti2v-5b","prompt":"fox","size":"480x480","seconds":1,"input_reference":")" +
          std::string(kPng) +
          R"(","gufo":{"seed":3,"steps":4,"negative_prompt":"blurry","guidance":3.5}})");
  Check(job.first_frame.width == 1 && job.first_frame.height == 1 &&
            job.first_frame.rgb.size() == 3 &&
            job.negative_prompt == "blurry" && job.guidance == 3.5 &&
            job.seed == 3 && job.steps == 4 && job.width == 480 &&
            job.frames == 25 && job.prompt == "fox",
        "Wan first frame, negative prompt and guidance reach the runner");

  job = Accept(
      service, runner, &calls, wan22::Family::kWan22,
      R"({"model":"wan2.2-ti2v-5b","prompt":"fox","size":"480x480","seconds":1})");
  Check(job.first_frame.rgb.empty() && job.negative_prompt.empty() &&
            job.guidance == 5.0 && job.steps == 50,
        "Wan defaults: text-to-video, empty negative prompt, guidance 5");

  const std::string boundary = "wan-boundary";
  job = Accept(service, runner, &calls, wan22::Family::kWan22,
               Multipart(boundary, {{"model", "wan2.2-ti2v-5b"},
                                    {"prompt", "fox"},
                                    {"size", "480x480"},
                                    {"seconds", "1"},
                                    {"first_frame", Unbase64(kPng), true}}),
               "multipart/form-data; boundary=" + boundary);
  Check(job.first_frame.rgb.size() == 3,
        "Wan multipart first frame reaches the runner");

  const std::string head =
      R"({"model":"wan2.2-ti2v-5b","prompt":"fox","size":"480x480","seconds":1)";
  Rejected(service, head + R"(,"last_frame":"x"})",
           "unsupported_input_reference", "a Wan 2.2 last frame");
  Rejected(service, head + R"(,"input_reference":"!!"})",
           "invalid_input_reference", "an undecodable first frame");
  Rejected(service, head + R"(,"gufo":{"guidance":"5"}})", "invalid_guidance",
           "a string guidance");
  Rejected(service, head + R"(,"gufo":{"guidance":50}})", "invalid_guidance",
           "guidance 50");
  Rejected(service, head + R"(,"gufo":{"negative_prompt":1}})",
           "invalid_negative_prompt", "a numeric negative prompt");
  Rejected(service, head + R"(,"gufo":{"lora":"x"}})", "invalid_strix_options",
           "an unknown Wan 2.2 option");
  // Post-steps are the server's (src/cli/serve/media); this service has
  // no --rife-model / --upscale-model.
  Rejected(service, head + R"(,"gufo":{"interpolate":2}})",
           "invalid_video_request", "RIFE without an interpolator");
  Rejected(service, head + R"(,"gufo":{"upscale":"960x960"}})",
           "invalid_video_request", "Real-ESRGAN without an upscaler");
  Rejected(service, head + R"(,"gufo":{"interpolate":9}})",
           "invalid_interpolate", "a RIFE multiplier above 8");
  Rejected(service,
           R"({"model":"wan2.2-ti2v-5b","prompt":"fox","size":"500x480","seconds":1})",
           "invalid_parameters", "a size off the 32-pixel grid");
  Rejected(service,
           R"({"model":"minimax-h3","prompt":"fox","size":"480x480","seconds":1})",
           "invalid_model", "another provider's model");
}

// Wan 2.1 VACE: gufo.vace options reach the runner, the control video as the
// guide file, a still mask as a binary PPM, reference images decoded;
// negative prompt, guidance and steps pass through.
void TestVace(const std::filesystem::path& root) {
  FakeRunner runner;
  VideoJobService service(Options(root, "wan2.1-vace-14b", &runner));
  Check(service.ready(), service.initialization_error());
  Check(service.model_id() == "wan2.1-vace-14b", "VACE model id");
  const std::string mp4(96, 'A');
  const std::string prefix =
      R"({"model":"wan2.1-vace-14b","prompt":"walk","size":"832x480","seconds":"1","gufo":{"negative_prompt":"blurry","guidance":5,"steps":20,"vace":{)";
  int calls = 0;
  const wan22::JobRequest job = Accept(
      service, runner, &calls, wan22::Family::kWan21Vace,
      prefix + R"("control_video":"data:video/mp4;base64,)" + mp4 +
          R"(","mask":"data:image/png;base64,)" + std::string(kPng) +
          R"(","mask_channel":"green","invert_mask":true,"reference_images":[")" +
          std::string(kPng) +
          R"("],"strength":0.5,"sampler":"euler","shift":8,"lora":false}}})");
  Check(job.frames == 17 && job.steps == 20 && job.width == 832 &&
            job.negative_prompt == "blurry" && job.guidance == 5.0,
        "VACE size, frames, steps, negative prompt and guidance reach the "
        "runner");
  Check(!job.guide_video_path.empty() &&
            std::filesystem::file_size(job.guide_video_path) == 72 &&
            job.mask_path.extension() == ".ppm" &&
            job.references.size() == 1 && job.references[0].width == 1,
        "VACE control video, still mask and reference image reach the runner");
  {
    std::ifstream mask(job.mask_path, std::ios::binary);
    std::string header(11, '\0');
    mask.read(header.data(), 11);
    Check(header == "P6\n1 1\n255\n", "VACE still mask is a binary PPM");
  }
  const gufo::json::Value vace = gufo::json::parse(job.vace_json);
  Check(vace.member_double("strength") == 0.5 &&
            vace.member_str("mask_channel") == "green" &&
            vace.find("invert_mask")->as_bool() &&
            vace.member_str("sampler") == "euler" &&
            vace.member_double("shift") == 8 && !vace.find("lora")->as_bool(),
        "VACE options reach the runner: " + job.vace_json);

  const wan22::JobRequest plain = Accept(
      service, runner, &calls, wan22::Family::kWan21Vace,
      R"({"model":"wan2.1-vace-14b","prompt":"walk","seconds":"1"})");
  Check(plain.guidance == 1.0 && plain.steps == 4 &&
            plain.guide_video_path.empty() && plain.mask_path.empty() &&
            plain.vace_json == "{}",
        "VACE defaults: no CFG, 4 steps, no control video");

  for (const std::string& bad :
       {prefix + R"("strength":-1}}})", prefix + R"("sampler":"uni_pc"}}})",
        prefix + R"("mask_channel":"alpha"}}})",
        prefix + R"("control_video":"!!"}}})", prefix + R"("unknown":1}}})",
        prefix + R"("reference_images":["a","b","c","d","e"]}}})",
        std::string(
            R"({"model":"wan2.1-vace-14b","prompt":"x","size":"1920x1080","seconds":"1"})"),
        std::string(
            R"({"model":"wan2.1-vace-14b","prompt":"x","seconds":"1","first_frame":"x"})"),
        std::string(
            R"({"model":"wan2.1-vace-14b","prompt":"x","seconds":"1","gufo":{"guidance":0.5}})")})
    Rejected(service, bad, "", "invalid VACE request " + bad);
}

// Wan 2.1 Uni3C: a camera move or a render video reaches the runner; the
// first frame is required; "gufo.preview" is accepted (no preview service
// yet) and a bad one refused.
void TestUni3C(const std::filesystem::path& root) {
  FakeRunner runner;
  VideoJobService service(Options(root, "wan2.1-uni3c", &runner));
  Check(service.ready(), service.initialization_error());
  const std::string prefix =
      R"({"model":"wan2.1-uni3c","prompt":"orbit","size":"832x480","seconds":"1","first_frame":")" +
      std::string(kPng) + R"(","gufo":{)";
  int calls = 0;
  wan22::JobRequest job =
      Accept(service, runner, &calls, wan22::Family::kWan21Uni3C,
             prefix +
                 R"("camera":{"type":"orbit_left","amount":30},"preview":{"always":true}}})");
  Check(job.frames == 17 && job.steps == 4 &&
            gufo::json::parse(job.camera_json).member_str("type") ==
                "orbit_left" &&
            job.guide_video_path.empty() && job.first_frame.rgb.size() == 3,
        "Uni3C camera move reaches the runner: " + job.camera_json);

  const std::string render(96, 'B');
  job = Accept(service, runner, &calls, wan22::Family::kWan21Uni3C,
               prefix + R"("camera":{"render_video":")" + render + R"("}}})");
  Check(job.camera_json.empty() && !job.guide_video_path.empty() &&
            std::filesystem::file_size(job.guide_video_path) == 72,
        "Uni3C render video reaches the runner as the guide file");

  job = Accept(service, runner, &calls, wan22::Family::kWan21Uni3C,
               prefix + R"("steps":6}})");
  Check(job.camera_json.empty() && job.guide_video_path.empty() &&
            job.steps == 6,
        "Uni3C without a camera is plain image-to-video");

  Rejected(service,
           R"({"model":"wan2.1-uni3c","prompt":"x","seconds":"1"})",
           "invalid_input_reference", "Uni3C without a first frame");
  Rejected(service, prefix + R"("camera":{"type":"barrel_roll_9000"}}})",
           "invalid_camera", "an unknown camera move");
  Rejected(service,
           prefix + R"("camera":{"poses":[{"frame":0,"yaw":500}]}}})",
           "invalid_camera", "a camera pose out of range");
  for (const char* bad : {R"("preview":{"id":"x"}}})",
                          R"("preview":{"every":0}}})", R"("preview":7}})"})
    Rejected(service, prefix + bad, "invalid_preview",
             std::string("preview ") + bad);
  Rejected(service,
           R"({"model":"wan2.1-uni3c","prompt":"x","size":"960x544","seconds":"1","first_frame":")" +
               std::string(kPng) + R"("})",
           "invalid_parameters", "a Uni3C size above 896x512 pixels");
}

// InfiniteTalk: one or two speakers (JSON or multipart), add / para / auto,
// the duration from the audio, speaker masks, video to video with denoise,
// the 896x512 limit.
void TestTalk(const std::filesystem::path& root) {
  FakeRunner runner;
  VideoJobServiceOptions options =
      Options(root, "wan2.1-infinitetalk-14b", &runner);
  options.model_aliases = {"talk"};
  VideoJobService service(std::move(options));
  Check(service.ready(), service.initialization_error());
  const std::string head =
      R"({"model":"wan2.1-infinitetalk-14b","prompt":"a woman speaks","first_frame":")" +
      std::string(kPng) + R"(")";
  // One second of 16 kHz speech: wav2vec2 gives 49 frames at 50 Hz, 24 video
  // frames at 25 fps.
  int calls = 0;
  VideoRequest raw;
  wan22::JobRequest job = Accept(
      service, runner, &calls, wan22::Family::kWan21Talk,
      head + R"(,"gufo":{"seed":7,"talk":{"audio":[")" + WavUrl(1.0) +
          R"("],"audio_scale":1.5,"motion_frames":9}}})",
      {}, &raw);
  Check(job.frames == 24 && job.steps == 4 && job.seed == 7 &&
            job.width == 832 && job.height == 480,
        "one speaker: duration from the audio (24 frames at 25 fps), got " +
            std::to_string(job.frames));
  Check(job.speakers.size() == 1 && job.speakers[0].sample_rate == 32000 &&
            job.speakers[0].channels == 2 &&
            std::abs(job.speakers[0].samples - 32000) <= 2 &&
            job.speaker_masks.empty() && job.guide_video_path.empty(),
        "one speaker decodes to 32 kHz stereo without masks");
  Check(wan22::TalkFramesForAudio(job.speakers[0]) == 24 &&
            wan22::TalkMono16k(job.speakers[0]).size() == 16000,
        "the speech drives 24 frames and resamples to 16 kHz mono");
  const gufo::json::Value talk = gufo::json::parse(job.talk_json);
  Check(talk.member_double("audio_scale") == 1.5 &&
            talk.member_double("motion_frames") == 9 &&
            !talk.contains("audio"),
        "talk options reach the runner without the media: " + job.talk_json);
  Check(raw.attachments.size() == 2 && raw.parameters.preset == "wan21-talk",
        "the runner gets the start image and the speech as attachments");

  // Two speakers as multipart uploads, "auto" with equal lengths -> "para"
  // (the longer track sets the length), at 896x512.
  const std::string boundary = "talk-boundary";
  const std::string png = Unbase64(kPng);
  const auto two = [&](double second_seconds, const std::string& mode,
                       const std::string& size) {
    return Multipart(
        boundary,
        {{"model", "talk"},
         {"prompt", "two people talk"},
         {"size", size},
         {"gufo", R"({"talk":{"audio_mode":")" + mode + R"("}})"},
         {"first_frame", png, true},
         {"audio_1", Wav(1.0, 16000), true},
         {"audio_2", Wav(second_seconds, 16000), true},
         {"mask_1", png, true},
         {"mask_2", png, true}});
  };
  job = Accept(service, runner, &calls, wan22::Family::kWan21Talk,
               two(1.0, "auto", "896x512"),
               "multipart/form-data; boundary=" + boundary);
  Check(job.speakers.size() == 2 && job.speaker_masks.size() == 2 &&
            job.speaker_masks[1].rgb.size() == 3 && job.width == 896 &&
            job.height == 512 && job.frames == 24 &&
            gufo::json::parse(job.talk_json).member_str("audio_mode") ==
                "para",
        "two equal speakers: auto -> para, 24 frames, 896x512: " +
            job.talk_json + " frames " + std::to_string(job.frames));
  // Different lengths: auto -> add, the speakers in turn.
  job = Accept(service, runner, &calls, wan22::Family::kWan21Talk,
               two(0.5, "auto", "832x480"),
               "multipart/form-data; boundary=" + boundary);
  const int second = wan22::TalkFramesForAudio(job.speakers[1]);
  Check(gufo::json::parse(job.talk_json).member_str("audio_mode") == "add" &&
            job.frames == 24 + second,
        "two speakers of different length: auto -> add, frames in turn (" +
            std::to_string(job.frames) + ")");
  job = Accept(service, runner, &calls, wan22::Family::kWan21Talk,
               two(0.5, "para", "832x480"),
               "multipart/form-data; boundary=" + boundary);
  Check(job.frames == 24, "para: the longer speaker sets the duration");

  // Explicit seconds win over the audio.
  job = Accept(service, runner, &calls, wan22::Family::kWan21Talk,
               head + R"(,"seconds":2,"gufo":{"talk":{"audio":[")" +
                   WavUrl(1.0) + R"("]}}})");
  Check(job.frames == 50, "explicit seconds set the duration");

  // Video to video: denoise below 1 with an input video.
  job = Accept(service, runner, &calls, wan22::Family::kWan21Talk,
               head + R"(,"gufo":{"talk":{"denoise":0.6,"video":"data:video/mp4;base64,)" +
                   std::string(96, 'C') + R"(","audio":[")" + WavUrl(1.0) +
                   R"("]}}})");
  Check(!job.guide_video_path.empty() &&
            gufo::json::parse(job.talk_json).member_double("denoise") == 0.6,
        "video to video: the input video and denoise reach the runner");

  const std::string speech = R"("audio":[")" + WavUrl(1.0) + R"("])";
  Rejected(service, head + R"(,"gufo":{"talk":{"denoise":0.6,)" + speech + "}}}",
           "invalid_talk_video", "denoise below 1 without a video");
  Rejected(service,
           head + R"(,"gufo":{"talk":{"video":")" + std::string(96, 'C') +
               R"(",)" + speech + "}}}",
           "invalid_talk_video", "a talk video without denoise");
  Rejected(service, head + "}", "missing_audio_reference",
           "InfiniteTalk without speech");
  Rejected(service,
           head + R"(,"gufo":{"talk":{"audio":[")" + WavUrl(1.0) + R"(",")" +
               WavUrl(1.0) + R"("]}}})",
           "invalid_speaker_masks", "two speakers without masks");
  Rejected(service,
           head + R"(,"gufo":{"talk":{"audio_mode":"sing",)" + speech + "}}}",
           "invalid_talk", "an unknown audio mode");
  Rejected(service,
           head + R"(,"gufo":{"talk":{"motion_frames":8,)" + speech + "}}}",
           "invalid_talk", "motion frames off 4k + 1");
  Rejected(service,
           head + R"(,"size":"1024x576","gufo":{"talk":{)" + speech + "}}}",
           "invalid_parameters", "a talk size above 896x512 pixels");
  Rejected(service,
           head + R"(,"gufo":{"interpolate":4,"talk":{)" + speech + "}}}",
           "invalid_video_request", "RIFE on a talk video without RIFE");
  Rejected(service,
           R"({"model":"wan2.1-infinitetalk-14b","prompt":"x","gufo":{"talk":{)" +
               speech + "}}}",
           "invalid_input_reference", "InfiniteTalk without a start image");
  Rejected(service,
           head + R"(,"gufo":{"talk":{"audio":["data:audio/wav;base64,AAAA"]}}})",
           "invalid_audio_reference", "undecodable speech");
}

// Probing: a Wan 2.2 Diffusers directory or a talk bundle selects the Wan
// provider over a generic provider that accepts every directory.
// gufo.interpolate / gufo.upscale on a Wan job: the server takes them out of
// the request (the provider never sees them) and runs them on the finished
// MP4 in order; "source_video" jobs only post-process an upload.
void TestPostSteps(const std::filesystem::path& root) {
  FakeRunner runner;
  auto options = Options(root, "wan2.2-ti2v-5b", &runner);
  std::mutex seen_mutex;
  std::vector<std::string> order;
  gufo::server::VideoInterpolation seen_interpolation;
  gufo::server::VideoUpscale seen_upscale;
  const auto read_all = [](const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
  };
  options.postprocessors.interpolator =
      [&](const std::filesystem::path& input,
          const std::filesystem::path& output,
          const gufo::server::VideoInterpolation& spec,
          const gufo::ext::CancellationCheck&, VideoProgress progress,
          void* opaque, std::string*) {
        progress("interpolate", 1, 2, opaque);
        const std::lock_guard<std::mutex> lock(seen_mutex);
        order.push_back("interpolate:" + read_all(input));
        seen_interpolation = spec;
        std::ofstream(output, std::ios::binary) << "interpolated";
        return true;
      };
  options.postprocessors.upscaler =
      [&](const std::filesystem::path& input,
          const std::filesystem::path& output,
          const gufo::server::VideoUpscale& spec,
          const gufo::ext::CancellationCheck&, VideoProgress, void*,
          std::string*) {
        const std::lock_guard<std::mutex> lock(seen_mutex);
        order.push_back("upscale:" + read_all(input));
        seen_upscale = spec;
        std::ofstream(output, std::ios::binary) << "upscaled";
        return true;
      };
  options.postprocessors.upscale_models = {"anime", "general"};
  VideoJobService service(std::move(options));
  Check(service.ready(), service.initialization_error());
  int calls = 0;
  VideoRequest raw;
  (void)Accept(
      service, runner, &calls, wan22::Family::kWan22,
      R"({"model":"wan2.2-ti2v-5b","prompt":"fox","size":"480x480","seconds":1,"gufo":{"seed":3,"interpolate":{"fps":60},"upscale":{"width":1920,"height":1080,"fit":"crop","model":"general"}}})",
      {}, &raw);
  {
    const std::lock_guard<std::mutex> lock(seen_mutex);
    Check(raw.options_json.find("interpolate") == std::string::npos &&
              raw.options_json.find("upscale") == std::string::npos,
          "the provider never sees the post-steps: " + raw.options_json);
    Check(order.size() == 2 && order[0] == "interpolate:fake-video" &&
              order[1] == "upscale:interpolated",
          "RIFE runs on the generated MP4, Real-ESRGAN on its output");
    Check(seen_interpolation.fps_num == 60 &&
              seen_interpolation.multiplier == 1 &&
              seen_upscale.width == 1920 && seen_upscale.height == 1080 &&
              seen_upscale.fit == "crop" && seen_upscale.model == "general",
          "post-steps get the requested rate, size, fit and model");
    order.clear();
  }

  // A "gufo" object with only post-steps is removed before the provider.
  (void)Accept(
      service, runner, &calls, wan22::Family::kWan22,
      R"({"model":"wan2.2-ti2v-5b","prompt":"fox","size":"480x480","seconds":1,"gufo":{"interpolate":2}})");
  {
    const std::lock_guard<std::mutex> lock(seen_mutex);
    Check(order.size() == 1 && seen_interpolation.multiplier == 2,
          "a multiplier alone runs RIFE only");
    order.clear();
  }

  const std::string head =
      R"({"model":"wan2.2-ti2v-5b","prompt":"fox","size":"480x480","seconds":1)";
  Rejected(service, head + R"(,"gufo":{"upscale":"1921x1080"}})",
           "invalid_video_request", "an odd upscale width");
  Rejected(service,
           head +
               R"(,"gufo":{"upscale":{"width":1920,"height":1080,"model":"x"}}})",
           "invalid_video_request", "an unknown upscale model");
  Rejected(service,
           head + R"(,"gufo":{"interpolate":{"fps":60,"multiplier":2}}})",
           "invalid_interpolate", "fps and a multiplier together");

  // Post-processing only: the upload stands in for the generated video.
  const std::string upload(100, 'v');
  const HttpResponse created =
      Send(service,
           R"({"model":"wan2.2-ti2v-5b","source_video":"data:video/mp4;base64,)" +
               Base64(upload) + R"(","gufo":{"interpolate":4}})");
  Check(created.status == 202, "source_video job accepted: " + created.body);
  WaitCompleted(service, CreatedId(created));
  {
    const std::lock_guard<std::mutex> lock(seen_mutex);
    Check(order.size() == 1 && order[0] == "interpolate:" + upload &&
              seen_interpolation.multiplier == 4,
          "source_video is interpolated without running the generator");
  }
  Check(gufo::json::parse(Get(service, CreatedId(created)).body)
                .member_str("model") == "wan2.2-ti2v-5b",
        "source_video jobs are recorded under the served model");
  const std::string boundary = "post-boundary";
  const HttpResponse multipart =
      Send(service,
           Multipart(boundary, {{"model", "wan2.2-ti2v-5b"},
                                {"gufo", R"({"upscale":"1280x720"})"},
                                {"source_video", upload, true}}),
           "multipart/form-data; boundary=" + boundary);
  Check(multipart.status == 202,
        "multipart source_video accepted: " + multipart.body);
  WaitCompleted(service, CreatedId(multipart));
  {
    const std::lock_guard<std::mutex> lock(seen_mutex);
    Check(order.size() == 2 && order[1] == "upscale:" + upload,
          "a multipart source_video is upscaled");
  }
  Rejected(service,
           R"({"model":"wan2.2-ti2v-5b","source_video":"data:video/mp4;base64,)" +
               Base64(upload) + R"("})",
           "invalid_video_request", "source_video without a post-step");
  Rejected(service,
           R"({"model":"wan2.2-ti2v-5b","prompt":"x","source_video":"data:video/mp4;base64,)" +
               Base64(upload) + R"(","gufo":{"interpolate":2}})",
           "unsupported_field", "source_video with a prompt");
  Rejected(service,
           R"({"model":"wan2.2-ti2v-5b","source_video":"AAAA","gufo":{"interpolate":2}})",
           "invalid_source_video", "a too short source_video");
}

void TestProbe(const std::filesystem::path& root) {
  gufo::ext::RegisterVideoProvider({
      .id = "generic-test",
      .probe = [](std::string_view) { return true; },
  });
  const auto write = [](const std::filesystem::path& path,
                        const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary) << text;
  };
  const std::filesystem::path wan = root / "Wan2.2-TI2V-5B";
  write(wan / "model_index.json",
        R"({"_class_name":"WanPipeline","transformer_2":[null,null]})");
  write(wan / "vae" / "config.json", "{}");
  write(wan / "transformer" / "config.json", "{}");
  const std::filesystem::path talk = root / "talk-bundle";
  write(talk / "gufo-talk.json", R"({"_class_name":"GufoWan21InfiniteTalk"})");
  const std::filesystem::path vace = root / "vace-bundle";
  write(vace / "gufo-vace.json", R"({"_class_name":"GufoWan21Vace"})");
  const std::filesystem::path camera = root / "camera-bundle";
  write(camera / "gufo-camera.json", R"({"_class_name":"GufoWan21Uni3CCamera"})");
  const auto probed = [](const std::filesystem::path& path) {
    const auto* provider = gufo::ext::ProbeVideoProvider(path.string());
    return provider == nullptr ? std::string() : provider->id;
  };
  Check(probed(wan) == "wan2.2-ti2v-5b", "Wan 2.2 directory probe");
  Check(probed(talk) == "wan2.1-infinitetalk-14b", "talk bundle probe");
  Check(probed(vace) == "wan2.1-vace-14b", "VACE bundle probe");
  Check(probed(camera) == "wan2.1-uni3c", "camera bundle probe");
  Check(probed(root) == "generic-test", "unknown directory: generic provider");

  FakeRunner runner;
  VideoJobServiceOptions options = Options(root / "jobs", {}, &runner);
  options.model_root = talk;
  VideoJobService service(std::move(options));
  Check(service.ready() && service.model_id() == "wan2.1-infinitetalk-14b",
        "a video service on a talk bundle serves InfiniteTalk: " +
            service.initialization_error());
}

}  // namespace

void wan22_register(gufo::ext::Registry&);

int main() {
  gufo::ext::Registry registry;
  wan22_register(registry);
  const std::filesystem::path base =
      std::filesystem::temp_directory_path() /
      ("gufo-wan22-request-test-" +
       std::to_string(gufo::platform::ProcessId()));
  std::error_code ignored;
  std::filesystem::remove_all(base, ignored);
  std::filesystem::create_directories(base);
  TestWan22(base / "wan22");
  TestVace(base / "vace");
  TestUni3C(base / "uni3c");
  TestTalk(base / "talk");
  TestPostSteps(base / "post");
  TestProbe(base / "probe");
  std::filesystem::remove_all(base, ignored);
  std::cout << "Wan video request tests passed.\n";
  return 0;
}
