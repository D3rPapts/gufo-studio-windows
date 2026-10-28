#include "src/models/wan22/pipeline.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <system_error>

#include "src/models/wan22/scheduler.hpp"
#include "src/models/wan22/text_encoder.hpp"
#include "src/models/wan22/tokenizer.hpp"
#include "src/models/wan22/transformer.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/wait.h>

#include <cstdio>
#endif

namespace gufo::models::wan22 {
namespace {

using Clock = std::chrono::steady_clock;

double Milliseconds(Clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin)
      .count();
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

float Bf16(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, 4);
  if ((bits & 0x7FFFFFFFU) > 0x7F800000U)
    return value;
  bits = (bits + 0x7FFFU + ((bits >> 16U) & 1U)) & 0xFFFF0000U;
  float result;
  std::memcpy(&result, &bits, 4);
  return result;
}

void TrackDevice(std::size_t* peak) {
  std::size_t free = 0, total = 0;
  if (hipMemGetInfo(&free, &total) == hipSuccess)
    *peak = std::max(*peak, total - std::min(free, total));
}

// Encodes `text` and returns the 512 text rows the transformer receives
// (states of the real tokens, zero rows after them).
bool EncodeText(const Tokenizer& tokenizer, TextEncoder& encoder,
                const std::string& text, std::vector<float>* rows, int* tokens,
                std::string* error) {
  const auto ids = tokenizer.Encode(CleanPrompt(text));
  std::vector<float> states;
  if (!encoder.Encode(ids, &states, error))
    return false;
  *tokens = static_cast<int>(ids.size());
  rows->assign(TextEncoder::kMaxTokens * TextEncoder::kModelWidth, 0.0F);
  std::copy(states.begin(), states.end(), rows->begin());
  return true;
}

}  // namespace

bool ValidateVideoShape(int width, int height, int frames, std::string* error) {
  if (width < 32 || height < 32 || width % 32 != 0 || height % 32 != 0 ||
      width > 4096 || height > 4096) {
    SetError(error, "width and height must be multiples of 32 (32..4096)");
    return false;
  }
  if (frames < 1 || frames > 1025 || (frames - 1) % 4 != 0) {
    SetError(error, "frames must be 4k + 1 (1..1025)");
    return false;
  }
  return true;
}

std::vector<float> GaussianNoise(std::size_t count, std::uint64_t seed) {
  std::uint64_t state = seed;
  const auto next = [&state] {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
  };
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; i += 2) {
    // Uniforms in (0, 1] and [0, 1) from 53 random bits each.
    const double u1 =
        (static_cast<double>(next() >> 11U) + 1.0) * (1.0 / 9007199254740992.0);
    const double u2 =
        static_cast<double>(next() >> 11U) * (1.0 / 9007199254740992.0);
    const double radius = std::sqrt(-2.0 * std::log(u1));
    const double angle = 6.283185307179586 * u2;
    values[i] = static_cast<float>(radius * std::cos(angle));
    if (i + 1 < count)
      values[i + 1] = static_cast<float>(radius * std::sin(angle));
  }
  return values;
}

bool GenerateVideo(const VideoRequest& request, const VideoCallbacks& callbacks,
                   std::vector<float>* final_latents, DecodedVideo* video,
                   bool float_video, VideoTelemetry* telemetry,
                   std::string* error) {
  VideoTelemetry local;
  VideoTelemetry& t = telemetry != nullptr ? *telemetry : local;
  t = VideoTelemetry{};
  const auto start = Clock::now();
  if (!ValidateVideoShape(request.width, request.height, request.frames, error))
    return false;
  const int latent_frames = (request.frames - 1) / 4 + 1;
  const int latent_height = request.height / 16;
  const int latent_width = request.width / 16;
  const std::size_t latent_count = static_cast<std::size_t>(48) *
                                   latent_frames * latent_height * latent_width;
  if (!request.noise.empty() && request.noise.size() != latent_count) {
    SetError(error, "initial noise does not match the latent shape");
    return false;
  }
  const auto cancelled = [&] {
    return callbacks.keep_going && !callbacks.keep_going();
  };
  const auto progress = [&](std::string_view phase, int done, int total) {
    if (callbacks.progress)
      callbacks.progress(phase, done, total);
  };
  const auto scheduler_config = ReadUniPcConfig(
      request.model_root / "scheduler" / "scheduler_config.json", error);
  if (!scheduler_config)
    return false;

  // 1. Prompt and negative prompt (the pipeline's default negative is "").
  std::vector<float> text, negative;
  {
    auto clock = Clock::now();
    progress("prompt", 0, 2);
    const auto tokenizer = Tokenizer::Load(
        request.model_root / "tokenizer" / "tokenizer.json", error);
    if (!tokenizer)
      return false;
    auto encoder =
        TextEncoder::Load(request.model_root / "text_encoder", error);
    if (!encoder)
      return false;
    TrackDevice(&t.peak_device_bytes);
    if (!EncodeText(*tokenizer, *encoder, request.prompt, &text,
                    &t.prompt_tokens, error))
      return false;
    progress("prompt", 1, 2);
    if (request.guidance > 1.0F &&
        !EncodeText(*tokenizer, *encoder, request.negative_prompt, &negative,
                    &t.negative_tokens, error))
      return false;
    progress("prompt", 2, 2);
    t.text_ms = Milliseconds(clock);
  }
  if (cancelled()) {
    SetError(error, "cancelled");
    return false;
  }

  // 2. Denoising loop.
  std::vector<float> latents =
      request.noise.empty()
          ? GaussianNoise(latent_count, request.seed)
          : std::vector<float>(request.noise.begin(), request.noise.end());
  {
    auto clock = Clock::now();
    auto transformer =
        Transformer::Load(request.model_root / "transformer", error);
    if (!transformer)
      return false;
    t.transformer_load_ms = Milliseconds(clock);
    UniPcScheduler scheduler(*scheduler_config, request.steps);
    const bool guided = request.guidance > 1.0F;
    std::vector<float> cond, uncond, input(latent_count);
    clock = Clock::now();
    progress("denoiser", 0, request.steps);
    for (int step = 0; step < request.steps; ++step) {
      if (cancelled()) {
        SetError(error, "cancelled");
        return false;
      }
      const auto step_clock = Clock::now();
      // The transformer runs in BF16: latents.to(bfloat16).
      for (std::size_t i = 0; i < latent_count; ++i)
        input[i] = Bf16(latents[i]);
      const float timestep = static_cast<float>(
          scheduler.timesteps()[static_cast<std::size_t>(step)]);
      TransformerInput in;
      in.latent = input;
      in.frames = latent_frames;
      in.height = latent_height;
      in.width = latent_width;
      in.timesteps = std::span<const float>(&timestep, 1);
      in.text = text;
      in.text_rows = static_cast<int>(TextEncoder::kMaxTokens);
      if (!transformer->Forward(in, &cond, nullptr, error))
        return false;
      if (guided) {
        in.text = negative;
        if (!transformer->Forward(in, &uncond, nullptr, error))
          return false;
        // noise_uncond + g * (noise_pred - noise_uncond), each op in BF16.
        for (std::size_t i = 0; i < latent_count; ++i)
          cond[i] = Bf16(uncond[i] +
                         Bf16(request.guidance * Bf16(cond[i] - uncond[i])));
      }
      if (step == 0)
        TrackDevice(&t.peak_device_bytes);
      scheduler.Step(cond, latents);
      t.step_ms.push_back(Milliseconds(step_clock));
      if (callbacks.on_step)
        callbacks.on_step(step, latents);
      progress("denoiser", step + 1, request.steps);
    }
    t.denoise_ms = Milliseconds(clock);
  }
  if (final_latents != nullptr)
    *final_latents = latents;

  // 3. VAE decode.
  if (video != nullptr) {
    auto clock = Clock::now();
    auto vae = VaeDecoder::Load(request.model_root / "vae", error);
    if (!vae)
      return false;
    t.vae_load_ms = Milliseconds(clock);
    clock = Clock::now();
    progress("video_vae", 0, latent_frames);
    if (!vae->Decode(
            latents, latent_frames, latent_height, latent_width, float_video,
            true, video,
            [&](int done, int total) {
              if (done == 1)
                TrackDevice(&t.peak_device_bytes);
              progress("video_vae", done, total);
            },
            callbacks.keep_going, error))
      return false;
    t.vae_ms = Milliseconds(clock);
  }
  t.total_ms = Milliseconds(start);
  return true;
}

namespace {

#if defined(_WIN32)
std::wstring Widen(const std::string& text) {
  if (text.empty())
    return {};
  const int length = MultiByteToWideChar(
      CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
  std::wstring wide(static_cast<std::size_t>(std::max(length, 0)), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
                      wide.data(), length);
  return wide;
}

std::wstring Quote(const std::wstring& argument) {
  if (!argument.empty() &&
      argument.find_first_of(L" \t\"") == std::wstring::npos)
    return argument;
  std::wstring quoted = L"\"";
  for (const wchar_t c : argument) {
    if (c == L'"')
      quoted += L'\\';
    quoted += c;
  }
  return quoted + L"\"";
}

// Spawns FFmpeg (searched beside the executable first, as CreateProcess
// does) with its stdin on a pipe and stderr in `log`, streams the frames.
bool RunFfmpeg(const std::vector<std::wstring>& arguments,
               std::span<const std::uint8_t> data,
               const std::filesystem::path& log, std::string* error) {
  const char* override = std::getenv("GUFO_FFMPEG");
  std::wstring command = Quote(
      override != nullptr && *override != '\0' ? Widen(override) : L"ffmpeg");
  for (const auto& argument : arguments)
    command += L" " + Quote(argument);
  SECURITY_ATTRIBUTES inherit{static_cast<DWORD>(sizeof(SECURITY_ATTRIBUTES)),
                              nullptr, TRUE};
  HANDLE read_end = nullptr, write_end = nullptr;
  if (!CreatePipe(&read_end, &write_end, &inherit, 4U << 20U)) {
    SetError(error, "cannot create the FFmpeg pipe");
    return false;
  }
  SetHandleInformation(write_end, HANDLE_FLAG_INHERIT, 0);
  HANDLE log_file =
      CreateFileW(log.wstring().c_str(), GENERIC_WRITE, FILE_SHARE_READ,
                  &inherit, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdInput = read_end;
  startup.hStdOutput = log_file;
  startup.hStdError = log_file;
  PROCESS_INFORMATION process{};
  const BOOL spawned =
      CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE,
                     CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);
  CloseHandle(read_end);
  if (log_file != INVALID_HANDLE_VALUE)
    CloseHandle(log_file);
  if (!spawned) {
    CloseHandle(write_end);
    SetError(error,
             "cannot start FFmpeg (set GUFO_FFMPEG or place "
             "ffmpeg.exe beside the executable)");
    return false;
  }
  CloseHandle(process.hThread);
  bool written = true;
  for (std::size_t offset = 0; offset < data.size() && written;) {
    const DWORD chunk = static_cast<DWORD>(
        std::min<std::size_t>(data.size() - offset, 4U << 20U));
    DWORD done = 0;
    written = WriteFile(write_end, data.data() + offset, chunk, &done,
                        nullptr) != 0 &&
              done > 0;
    offset += done;
  }
  CloseHandle(write_end);
  WaitForSingleObject(process.hProcess, INFINITE);
  DWORD code = 1;
  GetExitCodeProcess(process.hProcess, &code);
  CloseHandle(process.hProcess);
  if (!written || code != 0) {
    SetError(error, "FFmpeg failed (exit " + std::to_string(code) + "), see " +
                        log.string());
    return false;
  }
  return true;
}
#endif

}  // namespace

bool WriteMp4(const std::filesystem::path& path, const DecodedVideo& video,
              int fps, std::string* error) {
  const std::size_t frame_bytes =
      static_cast<std::size_t>(video.width) * video.height * 3;
  if (video.frames < 1 || video.rgb8.size() != frame_bytes * video.frames ||
      fps < 1) {
    SetError(error, "no 8-bit frames to encode");
    return false;
  }
  std::error_code ignored;
  if (path.has_parent_path())
    std::filesystem::create_directories(path.parent_path(), ignored);
  // The ".gufo-partial-" names are swept by the video job recovery.
  const std::filesystem::path partial = path.string() + ".gufo-partial-wan.mp4";
  const std::filesystem::path log = path.string() + ".gufo-partial-wan.log";
  const std::string size =
      std::to_string(video.width) + "x" + std::to_string(video.height);
#if defined(_WIN32)
  const std::vector<std::wstring> arguments = {L"-y",
                                               L"-loglevel",
                                               L"error",
                                               L"-f",
                                               L"rawvideo",
                                               L"-pixel_format",
                                               L"rgb24",
                                               L"-video_size",
                                               Widen(size),
                                               L"-framerate",
                                               std::to_wstring(fps),
                                               L"-i",
                                               L"-",
                                               L"-c:v",
                                               L"libx264",
                                               L"-preset",
                                               L"fast",
                                               L"-crf",
                                               L"18",
                                               L"-pix_fmt",
                                               L"yuv420p",
                                               L"-movflags",
                                               L"+faststart",
                                               partial.wstring()};
  if (!RunFfmpeg(arguments, video.rgb8, log, error))
    return false;
#else
  const char* override = std::getenv("GUFO_FFMPEG");
  const std::string command =
      std::string("'") +
      (override != nullptr && *override != '\0' ? override : "ffmpeg") +
      "' -y -loglevel error -f rawvideo -pixel_format rgb24 -video_size " +
      size + " -framerate " + std::to_string(fps) +
      " -i - -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p -movflags "
      "+faststart '" +
      partial.string() + "' 2> '" + log.string() + "'";
  FILE* pipe = popen(command.c_str(), "w");
  if (pipe == nullptr) {
    SetError(error, "cannot start FFmpeg");
    return false;
  }
  const bool written = std::fwrite(video.rgb8.data(), 1, video.rgb8.size(),
                                   pipe) == video.rgb8.size();
  const int status = pclose(pipe);
  if (!written || status != 0) {
    SetError(error, "FFmpeg failed, see " + log.string());
    return false;
  }
#endif
  std::filesystem::rename(partial, path, ignored);
  if (ignored) {
    SetError(error,
             "cannot publish " + path.string() + ": " + ignored.message());
    return false;
  }
  std::filesystem::remove(log, ignored);
  return true;
}

}  // namespace gufo::models::wan22
