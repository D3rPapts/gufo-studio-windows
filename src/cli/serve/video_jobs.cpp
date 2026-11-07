#include "src/cli/serve/video_jobs.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <ranges>
#include <sstream>
#include <stop_token>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/logging.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"
#include "src/core/platform/os_info.hpp"
#include "src/ext/video.hpp"

namespace gufo::server {
namespace {

std::optional<int> ResolveDurationFrames(const ext::VideoProvider* provider,
                                         std::string_view seconds,
                                         std::string* error = nullptr) {
  if (provider && provider->resolve_duration_frames) {
    return provider->resolve_duration_frames(seconds, error);
  }
  return std::nullopt;
}

bool ValidateGenerationParameters(const ext::VideoProvider* provider,
                                  const ext::VideoParameters& parameters,
                                  std::string* error = nullptr) {
  if (provider && provider->validate_parameters) {
    return provider->validate_parameters(parameters, error);
  }
  return false;
}

std::string GenerationParametersJson(const ext::VideoProvider* provider,
                                     const ext::VideoParameters& parameters,
                                     std::uint64_t seed,
                                     std::string_view prompt_sha256) {
  if (provider && provider->parameters_json) {
    return provider->parameters_json(parameters, seed, prompt_sha256);
  }
  return "{}";
}

std::string GenerationTelemetryJson(const ext::VideoProvider* provider,
                                    const ext::VideoTelemetry& telemetry) {
  if (provider && provider->telemetry_json) {
    return provider->telemetry_json(telemetry);
  }
  return "{}";
}

bool IsH3Model(std::string_view model) {
  return model == "minimax-h3" || model == "minimax-h3-exact" ||
         model == "minimax-h3-fast" || model == "minimax-h3-aggressive" ||
         model == "minimax-h3-dev" || model == "minimax-h3-fullres";
}

bool ModelMatchesPreset(std::string_view model, std::string_view preset) {
  if (model == "minimax-h3") {
    return true;
  }
  return (model == "minimax-h3-exact" && preset == "exact-512") ||
         (model == "minimax-h3-fast" && preset == "fast-384") ||
         (model == "minimax-h3-aggressive" && preset == "aggressive-320") ||
         (model == "minimax-h3-dev" && preset == "development-256") ||
         (model == "minimax-h3-fullres" && preset == "exact-1344x768");
}

// Providers that parse their own requests: the provider id is the model,
// output is always MP4 and the provider checks everything else.
bool IsPreparedProvider(const ext::VideoProvider* provider) {
  return provider != nullptr && provider->prepare_request != nullptr;
}

bool IsServedModel(const ext::VideoProvider* provider, std::string_view model) {
  return IsPreparedProvider(provider) ? model == provider->id
                                      : IsH3Model(model);
}

bool ParseSize(std::string_view size, int* width, int* height) {
  const auto x = size.find('x');
  if (x == std::string_view::npos || x == 0 || x + 1 >= size.size()) {
    return false;
  }
  const auto first = std::from_chars(size.data(), size.data() + x, *width);
  const auto second =
      std::from_chars(size.data() + x + 1, size.data() + size.size(), *height);
  return first.ec == std::errc() && first.ptr == size.data() + x &&
         second.ec == std::errc() && second.ptr == size.data() + size.size() &&
         *width > 0 && *height > 0;
}

bool MatchesOutputContract(const ext::VideoProvider* provider,
                           const VideoJobRequest& request) {
  const auto& parameters = request.parameters;
  if (IsPreparedProvider(provider)) {
    int width = 0;
    int height = 0;
    return request.output_format == "mp4" &&
           ParseSize(request.size, &width, &height) &&
           ResolveDurationFrames(provider, request.seconds).has_value();
  }
  const auto frames = ResolveDurationFrames(provider, request.seconds);
  if (!frames || parameters.frames != *frames ||
      !ModelMatchesPreset(request.model, parameters.preset)) {
    return false;
  }
  if (request.output_format == "mp4") {
    const bool square =
        request.size == "512x512" && parameters.output_width == 512 &&
        parameters.output_height == 512 &&
        (parameters.preset == "exact-512" || parameters.preset == "fast-384" ||
         parameters.preset == "aggressive-320");
    const bool released_full_resolution =
        request.size == "1344x768" && parameters.preset == "exact-1344x768" &&
        parameters.internal_width == 1344 &&
        parameters.internal_height == 768 && parameters.output_width == 1344 &&
        parameters.output_height == 768;
    return (square || released_full_resolution) && parameters.mux &&
           parameters.decode_audio && parameters.selected_frames.empty();
  }
  return request.output_format == "ppm" && request.size == "256x256" &&
         parameters.preset == "development-256" &&
         parameters.internal_width == 256 &&
         parameters.internal_height == 256 && parameters.output_width == 256 &&
         parameters.output_height == 256 && !parameters.mux &&
         !parameters.decode_audio && parameters.selected_frames.size() == 1;
}

void WipeString(std::string* value) noexcept {
  if (value == nullptr) {
    return;
  }
  volatile char* const data = value->data();
  for (std::size_t index = 0; index < value->size(); ++index) {
    data[index] = '\0';
  }
  value->clear();
}

std::string RandomVideoId() {
  std::array<unsigned char, 12> bytes{};
  std::random_device random;
  for (unsigned char& byte : bytes) {
    byte = static_cast<unsigned char>(random());
  }
  static constexpr char kHex[] = "0123456789abcdef";
  std::string id = "video_";
  id.reserve(6 + bytes.size() * 2U);
  for (const unsigned char byte : bytes) {
    id.push_back(kHex[byte >> 4U]);
    id.push_back(kHex[byte & 0xFU]);
  }
  return id;
}

std::int64_t UnixNow() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

bool IsVideoId(std::string_view id) {
  if (!id.starts_with("video_") || id.size() != 30) {
    return false;
  }
  return std::ranges::all_of(id.substr(6), [](char value) {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
  });
}

bool IsStrictDescendant(const std::filesystem::path& root,
                        const std::filesystem::path& candidate) {
  auto root_iterator = root.begin();
  auto candidate_iterator = candidate.begin();
  for (; root_iterator != root.end() && candidate_iterator != candidate.end();
       ++root_iterator, ++candidate_iterator) {
    if (*root_iterator != *candidate_iterator) {
      return false;
    }
  }
  return root_iterator == root.end() && candidate_iterator != candidate.end();
}

bool IsSafeRegularContent(const std::filesystem::path& storage_root,
                          const std::filesystem::path& candidate,
                          std::uint64_t* bytes = nullptr) {
  std::error_code filesystem_error;
  const std::filesystem::file_status content_status =
      std::filesystem::symlink_status(candidate, filesystem_error);
  if (filesystem_error || std::filesystem::is_symlink(content_status) ||
      !std::filesystem::is_regular_file(content_status)) {
    return false;
  }
  const std::filesystem::path canonical_content =
      std::filesystem::canonical(candidate, filesystem_error);
  if (filesystem_error ||
      !IsStrictDescendant(storage_root, canonical_content)) {
    return false;
  }
  if (bytes != nullptr) {
    *bytes = std::filesystem::file_size(candidate, filesystem_error);
    if (filesystem_error) {
      return false;
    }
  }
  return true;
}

std::uint64_t NextFileNonce() {
  static std::atomic_uint64_t nonce{0};
  return nonce.fetch_add(1, std::memory_order_relaxed);
}

bool WriteAtomic(const std::filesystem::path& path, std::string_view contents,
                 std::string* error) {
  std::error_code filesystem_error;
  std::filesystem::create_directories(path.parent_path(), filesystem_error);
  if (filesystem_error) {
    if (error != nullptr) {
      *error =
          "cannot create video job directory: " + filesystem_error.message();
    }
    return false;
  }
  const std::filesystem::path partial = path.string() + ".gufo-partial-" +
                                        std::to_string(platform::ProcessId()) +
                                        "-" + std::to_string(NextFileNonce());
  {
    std::ofstream output(partial, std::ios::binary | std::ios::trunc);
    output.write(contents.data(),
                 static_cast<std::streamsize>(contents.size()));
    if (!output) {
      std::filesystem::remove(partial, filesystem_error);
      if (error != nullptr) {
        *error = "cannot write video job metadata";
      }
      return false;
    }
  }
  std::filesystem::rename(partial, path, filesystem_error);
  if (filesystem_error) {
    std::filesystem::remove(partial, filesystem_error);
    if (error != nullptr) {
      *error =
          "cannot publish video job metadata: " + filesystem_error.message();
    }
    return false;
  }
  return true;
}

std::string SnapshotJson(const VideoJobSnapshot& snapshot) {
  json::Value root = json::Value::object();
  root["schema"] = "gufo.video-job.v1";
  root["id"] = snapshot.id;
  root["status"] = std::string(ToString(snapshot.status));
  root["progress"] = snapshot.progress;
  root["created_at"] = static_cast<long long>(snapshot.created_at);
  root["completed_at"] = static_cast<long long>(snapshot.completed_at);
  root["expires_at"] = static_cast<long long>(snapshot.expires_at);
  root["model"] = snapshot.model;
  root["size"] = snapshot.size;
  root["seconds"] = snapshot.seconds;
  root["output_format"] = snapshot.output_format;
  if (!snapshot.error_code.empty()) {
    json::Value error = json::Value::object();
    error["code"] = snapshot.error_code;
    error["message"] = snapshot.error_message;
    root["error"] = std::move(error);
  } else {
    root["error"] = json::Value();
  }
  return root.dump() + "\n";
}

int ProgressPercent(std::string_view phase, int completed, int total) {
  const auto scaled = [completed, total](int begin, int end) {
    if (total <= 0) {
      return begin;
    }
    const int bounded = std::clamp(completed, 0, total);
    return begin + ((end - begin) * bounded) / total;
  };
  if (phase == "inventory") {
    return scaled(1, 3);
  }
  if (phase == "prompt") {
    return scaled(3, 10);
  }
  if (phase == "denoiser") {
    return scaled(10, 80);
  }
  if (phase == "video_vae") {
    return scaled(80, 90);
  }
  if (phase == "audio_vae") {
    return scaled(90, 97);
  }
  if (phase == "media") {
    return scaled(97, 99);
  }
  if (const auto post = VideoPostProcessPercent(phase, completed, total)) {
    return *post;
  }
  return 1;
}

}  // namespace

std::string_view ToString(VideoJobStatus status) noexcept {
  switch (status) {
    case VideoJobStatus::kQueued:
      return "queued";
    case VideoJobStatus::kInProgress:
      return "in_progress";
    case VideoJobStatus::kCompleted:
      return "completed";
    case VideoJobStatus::kFailed:
      return "failed";
    case VideoJobStatus::kCancelled:
      return "cancelled";
  }
  return "failed";
}

struct VideoJobService::Impl {
  struct Job {
    VideoJobSnapshot snapshot;
    VideoJobRequest request;
    std::filesystem::path directory;
    std::filesystem::path content_path;
    std::string media_type;
    std::atomic<bool> cancelled{false};
    bool deleted{false};
    std::string logged_phase;
    int logged_bucket{-1};
  };

  explicit Impl(VideoJobServiceOptions supplied)
      : options(std::move(supplied)) {
    if (!options.id_factory) {
      options.id_factory = RandomVideoId;
    }
    if (!options.now) {
      options.now = UnixNow;
    }
    // The provider: named, else the one that accepts the model directory,
    // else MiniMax H3 (the only provider before provider-owned requests).
    provider = options.provider_id.empty()
                   ? ext::ProbeVideoProvider(options.model_root.string())
                   : ext::FindVideoProvider(options.provider_id);
    if (provider == nullptr && options.provider_id.empty()) {
      provider = ext::FindVideoProvider("minimax-h3");
    }
    if (provider == nullptr) {
      initialization_error =
          "video provider '" +
          (options.provider_id.empty() ? std::string("minimax-h3")
                                       : options.provider_id) +
          "' is not available";
      return;
    }
    if (!options.runner) {
      if (provider->create) {
        std::string create_err;
        auto video_runner =
            provider->create(options.model_root.string(), &create_err);
        if (video_runner) {
          std::shared_ptr<ext::VideoRunner> shared_runner =
              std::move(video_runner);
          options.runner = [shared_runner](const ext::VideoRequest& req,
                                           const ext::CancellationCheck& check,
                                           ext::VideoProgress progress,
                                           void* progress_opaque,
                                           ext::VideoTelemetry* telem,
                                           std::string* err) {
            return shared_runner->Generate(req, check, progress,
                                           progress_opaque, telem, err);
          };
        }
      }
    }
    if (options.model_root.empty() || options.source_manifest.empty() ||
        options.storage_root.empty() || options.queue_capacity == 0 ||
        options.artifact_ttl <= std::chrono::seconds::zero()) {
      initialization_error = "invalid video job configuration";
      return;
    }
#if !defined(ENGINE_ENABLE_HIP)
    if (options.validate_model_inventory) {
      initialization_error =
          "video serving (" + provider->id +
          ") requires a ROCm HIP-enabled build; no CPU inference fallback is "
          "provided";
      return;
    }
#endif
    if (options.validate_model_inventory) {
      std::string resolve_error;
      if ((provider->resolve &&
           !provider->resolve(options.model_root.string(),
                              options.source_manifest.string(),
                              &resolve_error))) {
        initialization_error = resolve_error.empty()
                                   ? "video model not resolved"
                                   : std::move(resolve_error);
        return;
      }
    }
    std::error_code filesystem_error;
    std::filesystem::create_directories(options.storage_root, filesystem_error);
    if (filesystem_error) {
      initialization_error =
          "cannot create video job storage: " + filesystem_error.message();
      return;
    }
    canonical_storage_root =
        std::filesystem::canonical(options.storage_root, filesystem_error);
    if (filesystem_error) {
      initialization_error =
          "cannot resolve video job storage: " + filesystem_error.message();
      return;
    }
    options.storage_root = canonical_storage_root;
    RecoverJobs();
    if (!initialization_error.empty()) {
      return;
    }
    worker = std::jthread([this](const std::stop_token& stop) { Run(stop); });
    ready = true;
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  ~Impl() {
    if (worker.joinable()) {
      worker.request_stop();
      {
        const std::lock_guard<std::mutex> lock(mutex);
        if (active != nullptr) {
          active->cancelled.store(true, std::memory_order_relaxed);
        }
        for (const auto& job : queue) {
          job->cancelled.store(true, std::memory_order_relaxed);
          WipeString(&job->request.prompt);
        }
      }
      condition.notify_all();
      worker.join();
    }
  }

  VideoJobSnapshot Snapshot(const Job& job) const { return job.snapshot; }

  bool Persist(const VideoJobSnapshot& snapshot, std::string* error = nullptr) {
    return WriteAtomic(options.storage_root / snapshot.id / "job.json",
                       SnapshotJson(snapshot), error);
  }

  void RecoverJobs() {
    std::error_code filesystem_error;
    const std::int64_t now_value = options.now();
    for (const auto& entry : std::filesystem::directory_iterator(
             options.storage_root, filesystem_error)) {
      if (filesystem_error) {
        initialization_error =
            "cannot inspect video job storage: " + filesystem_error.message();
        return;
      }
      const std::filesystem::file_status entry_status =
          entry.symlink_status(filesystem_error);
      if (filesystem_error) {
        initialization_error =
            "cannot inspect video job entry: " + filesystem_error.message();
        return;
      }
      const std::string directory_id = entry.path().filename().string();
      if (std::filesystem::is_symlink(entry_status) ||
          !std::filesystem::is_directory(entry_status) ||
          !IsVideoId(directory_id)) {
        continue;
      }
      for (const auto& child : std::filesystem::directory_iterator(
               entry.path(), filesystem_error)) {
        if (filesystem_error) {
          break;
        }
        if (child.path().filename().string().find(".gufo-partial-") !=
            std::string::npos) {
          std::filesystem::remove_all(child.path(), filesystem_error);
          filesystem_error.clear();
        }
      }
      const std::filesystem::path metadata = entry.path() / "job.json";
      std::ifstream input(metadata, std::ios::binary);
      const std::string text((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());
      if (text.empty()) {
        std::filesystem::remove_all(entry.path(), filesystem_error);
        filesystem_error.clear();
        continue;
      }
      try {
        const json::Value value = json::parse(text);
        auto job = std::make_shared<Job>();
        job->snapshot.id = value.member_str("id");
        job->snapshot.model = value.member_str("model");
        job->snapshot.size = value.member_str("size");
        job->snapshot.seconds = value.member_str("seconds");
        job->snapshot.output_format = value.member_str("output_format");
        const double progress = value.member_double("progress", -1);
        const double created_at = value.member_double("created_at", -1);
        const double completed_at = value.member_double("completed_at", -1);
        const double expires_at = value.member_double("expires_at", -1);
        job->directory = entry.path();
        int recovered_width = 0;
        int recovered_height = 0;
        const bool valid_output =
            IsPreparedProvider(provider)
                ? job->snapshot.output_format == "mp4" &&
                      ParseSize(job->snapshot.size, &recovered_width,
                                &recovered_height)
                : (job->snapshot.output_format == "mp4" &&
                                   (job->snapshot.size == "512x512" ||
                                    job->snapshot.size == "1344x768")) ||
                                  (job->snapshot.output_format == "ppm" &&
                                   job->snapshot.size == "256x256");
        const bool valid_numbers =
            std::isfinite(progress) && std::floor(progress) == progress &&
            progress == 100.0 && std::isfinite(created_at) &&
            std::floor(created_at) == created_at && created_at > 0.0 &&
            created_at <=
                static_cast<double>(std::numeric_limits<std::int64_t>::max()) &&
            std::isfinite(completed_at) &&
            std::floor(completed_at) == completed_at &&
            completed_at >= created_at &&
            completed_at <=
                static_cast<double>(std::numeric_limits<std::int64_t>::max()) &&
            std::isfinite(expires_at) && std::floor(expires_at) == expires_at &&
            expires_at > completed_at &&
            expires_at <=
                static_cast<double>(std::numeric_limits<std::int64_t>::max());
        if (value.member_str("schema") != "gufo.video-job.v1" ||
            !IsVideoId(job->snapshot.id) || job->snapshot.id != directory_id ||
            !IsServedModel(provider, job->snapshot.model) ||
            !ResolveDurationFrames(provider, job->snapshot.seconds) ||
            !valid_output ||
            !valid_numbers || expires_at <= static_cast<double>(now_value) ||
            value.member_str("status") != "completed") {
          std::filesystem::remove_all(entry.path(), filesystem_error);
          filesystem_error.clear();
          continue;
        }
        job->snapshot.progress = static_cast<int>(progress);
        job->snapshot.created_at = static_cast<std::int64_t>(created_at);
        job->snapshot.completed_at = static_cast<std::int64_t>(completed_at);
        job->snapshot.expires_at = static_cast<std::int64_t>(expires_at);
        if (job->snapshot.output_format == "mp4") {
          job->content_path = entry.path() / "output.mp4";
          job->media_type = "video/mp4";
        } else {
          job->content_path = entry.path() / "output.ppm";
          job->media_type = "image/x-portable-pixmap";
        }
        if (!IsSafeRegularContent(canonical_storage_root, job->content_path)) {
          std::filesystem::remove_all(entry.path(), filesystem_error);
          filesystem_error.clear();
          continue;
        }
        job->snapshot.status = VideoJobStatus::kCompleted;
        jobs.emplace(job->snapshot.id, std::move(job));
      } catch (...) {
        std::filesystem::remove_all(entry.path(), filesystem_error);
        filesystem_error.clear();
      }
    }
  }

  void SweepExpiredLocked() {
    const std::int64_t now_value = options.now();
    for (auto iterator = jobs.begin(); iterator != jobs.end();) {
      const auto& job = iterator->second;
      if (job->snapshot.expires_at > 0 &&
          job->snapshot.expires_at <= now_value &&
          job->snapshot.status != VideoJobStatus::kInProgress &&
          job->snapshot.status != VideoJobStatus::kQueued) {
        std::error_code ignored;
        std::filesystem::remove_all(job->directory, ignored);
        iterator = jobs.erase(iterator);
      } else {
        ++iterator;
      }
    }
  }

  void UpdateProgress(const std::shared_ptr<Job>& job, std::string_view phase,
                      int completed, int total) {
    const std::lock_guard<std::mutex> lock(mutex);
    if (job->deleted || job->snapshot.status != VideoJobStatus::kInProgress) {
      return;
    }
    job->snapshot.progress =
        std::max(job->snapshot.progress,
                 std::clamp(ProgressPercent(phase, completed, total), 1, 99));
    const int bucket =
        total > 0 ? static_cast<int>(10LL * completed / total) : 0;
    if (job->logged_phase != phase || bucket > job->logged_bucket) {
      job->logged_phase = phase;
      job->logged_bucket = bucket;
      Logger::Info("video", "job=" + job->snapshot.id +
                                " event=progress phase=" + std::string(phase) +
                                " completed=" + std::to_string(completed) +
                                " total=" + std::to_string(total) + " " +
                                Logger::MemoryStatus());
    }
  }

  struct ProgressContext {
    Impl* service{nullptr};
    std::shared_ptr<Job> job;
  };

  static void Progress(std::string_view phase, int completed, int total,
                       void* opaque) {
    auto* context = static_cast<ProgressContext*>(opaque);
    if (context != nullptr && context->service != nullptr) {
      context->service->UpdateProgress(context->job, phase, completed, total);
    }
  }

  void Run(const std::stop_token& stop) {
    while (!stop.stop_requested()) {
      std::shared_ptr<Job> job;
      bool started = true;
      {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock,
                       [&] { return stop.stop_requested() || !queue.empty(); });
        if (stop.stop_requested()) {
          break;
        }
        job = queue.front();
        queue.pop_front();
        if (job->deleted) {
          continue;
        }
        active = job;
        job->snapshot.status = VideoJobStatus::kInProgress;
        job->snapshot.progress = 1;
        std::string persist_error;
        if (!Persist(job->snapshot, &persist_error)) {
          active.reset();
          WipeString(&job->request.prompt);
          job->snapshot.status = VideoJobStatus::kFailed;
          job->snapshot.completed_at = options.now();
          job->snapshot.expires_at =
              job->snapshot.completed_at + options.artifact_ttl.count();
          job->snapshot.error_code = "storage_error";
          job->snapshot.error_message = "cannot persist video job transition";
          (void)Persist(job->snapshot);
          started = false;
        }
      }
      if (!started) {
        condition.notify_all();
        continue;
      }

      const auto start_time = std::chrono::steady_clock::now();
      Logger::Info(
          "video",
          "job=" + job->snapshot.id + " event=started preset=" +
              job->request.parameters.preset + " size=" + job->request.size +
              " frames=" + std::to_string(job->request.parameters.frames) +
              " steps=" + std::to_string(job->request.parameters.evaluations) +
              " seed=" + std::to_string(job->request.seed) + " " +
              Logger::MemoryStatus());
      ext::VideoRequest generation{
          .model_root = options.model_root,
          .source_manifest = options.source_manifest,
          .output_path = job->directory / "output.mp4",
          .frames_directory = job->request.output_format == "ppm"
                                  ? job->directory / "frames"
                                  : std::filesystem::path(),
          .latents_directory = {},
          .prompt = {},
          .seed = job->request.seed,
          .parameters = job->request.parameters,
          .options_json = job->request.options_json,
          .attachments = {},
      };
      std::vector<std::uint8_t> source_video;
      {
        // Delete can clear a queued/active prompt concurrently with admission.
        const std::lock_guard<std::mutex> lock(mutex);
        generation.prompt = std::move(job->request.prompt);
        WipeString(&job->request.prompt);
        // Inputs are needed once; the job keeps only its metadata.
        generation.attachments = std::move(job->request.attachments);
        job->request.attachments.clear();
        source_video = std::move(job->request.source_video);
        job->request.source_video.clear();
      }
      const std::string prompt_hash =
          crypto::Sha256Hex(std::span<const std::uint8_t>(
              reinterpret_cast<const std::uint8_t*>(generation.prompt.data()),
              generation.prompt.size()));
      std::string metadata_error;
      if (!WriteAtomic(job->directory / "parameters.json",
                       GenerationParametersJson(provider,
                                                job->request.parameters,
                                                job->request.seed, prompt_hash),
                       &metadata_error)) {
        WipeString(&generation.prompt);
        const std::lock_guard<std::mutex> lock(mutex);
        active.reset();
        job->snapshot.status = VideoJobStatus::kFailed;
        job->snapshot.completed_at = options.now();
        job->snapshot.expires_at =
            job->snapshot.completed_at + options.artifact_ttl.count();
        job->snapshot.error_code = "storage_error";
        job->snapshot.error_message = "cannot persist video job parameters";
        (void)Persist(job->snapshot);
        condition.notify_all();
        continue;
      }
      ProgressContext progress{this, job};
      ext::VideoTelemetry telemetry;
      std::string error;
      bool success = false;
      const ext::CancellationCheck is_cancelled = [&] {
        return job->cancelled.load(std::memory_order_relaxed);
      };
      try {
        if (!source_video.empty()) {
          // Post-processing only: the upload stands in for the generated MP4.
          std::ofstream source(generation.output_path, std::ios::binary);
          source.write(reinterpret_cast<const char*>(source_video.data()),
                       static_cast<std::streamsize>(source_video.size()));
          success = static_cast<bool>(source.flush());
          if (!success)
            error = "cannot store the uploaded source video";
          std::vector<std::uint8_t>().swap(source_video);
        } else {
          success = options.runner(generation, is_cancelled, Progress,
                                   &progress, &telemetry, &error);
        }
      } catch (const std::exception& exception) {
        error = exception.what();
      } catch (...) {
        error = "video generation backend failed unexpectedly";
      }
      WipeString(&generation.prompt);
      if (!success && !job->cancelled.load(std::memory_order_relaxed)) {
        Logger::Error("video", "job=" + job->snapshot.id +
                                   " event=backend_failed detail=" + error);
      }

      std::filesystem::path content_path;
      std::string media_type;
      bool output_ready =
          success && !job->cancelled.load(std::memory_order_relaxed);
      if (output_ready && job->request.output_format == "mp4" &&
          job->request.post.enabled()) {
        // gufo-studio-windows post-steps (src/cli/serve/media).
        output_ready = RunVideoPostProcess(
            job->request.post, options.postprocessors, job->directory,
            job->snapshot.id, is_cancelled, Progress, &progress);
      }
      if (output_ready && job->request.output_format == "mp4") {
        content_path = job->directory / "output.mp4";
        media_type = "video/mp4";
      } else if (output_ready) {
        std::ostringstream filename;
        filename << "frame-" << std::setw(4) << std::setfill('0')
                 << job->request.parameters.selected_frames.front() << ".ppm";
        const std::filesystem::path generated =
            job->directory / "frames" / filename.str();
        content_path = job->directory / "output.ppm";
        std::error_code rename_error;
        std::filesystem::rename(generated, content_path, rename_error);
        if (!rename_error) {
          std::filesystem::remove_all(job->directory / "frames", rename_error);
        } else {
          output_ready = false;
        }
        media_type = "image/x-portable-pixmap";
      }
      if (output_ready) {
        output_ready =
            IsSafeRegularContent(canonical_storage_root, content_path);
      }
      bool storage_failure = false;
      if (output_ready) {
        std::string telemetry_error;
        if (!WriteAtomic(job->directory / "telemetry.json",
                         GenerationTelemetryJson(provider, telemetry),
                         &telemetry_error)) {
          output_ready = false;
          storage_failure = true;
        }
      }

      VideoJobSnapshot snapshot;
      bool deleted = false;
      {
        const std::lock_guard<std::mutex> lock(mutex);
        active.reset();
        deleted = job->deleted;
        if (!deleted) {
          const std::int64_t completed_at = options.now();
          snapshot = job->snapshot;
          snapshot.completed_at = completed_at;
          snapshot.expires_at = completed_at + options.artifact_ttl.count();
          if (output_ready) {
            snapshot.status = VideoJobStatus::kCompleted;
            snapshot.progress = 100;
          } else if (job->cancelled.load(std::memory_order_relaxed)) {
            snapshot.status = VideoJobStatus::kCancelled;
            snapshot.error_code = "cancelled";
            snapshot.error_message = "video generation was cancelled";
          } else if (storage_failure) {
            snapshot.status = VideoJobStatus::kFailed;
            snapshot.error_code = "storage_error";
            snapshot.error_message = "cannot persist video job telemetry";
          } else {
            snapshot.status = VideoJobStatus::kFailed;
            snapshot.error_code = "generation_failed";
            snapshot.error_message = "video generation failed";
          }
          std::string persist_error;
          if (!Persist(snapshot, &persist_error)) {
            snapshot.status = VideoJobStatus::kFailed;
            snapshot.progress = std::min(snapshot.progress, 99);
            snapshot.error_code = "storage_error";
            snapshot.error_message = "cannot persist video job completion";
            (void)Persist(snapshot);
          }
          job->snapshot = snapshot;
          if (snapshot.status == VideoJobStatus::kCompleted) {
            job->content_path = std::move(content_path);
            job->media_type = std::move(media_type);
          }
        }
      }
      Logger::Log(snapshot.status == VideoJobStatus::kFailed ? LogLevel::kError
                                                             : LogLevel::kInfo,
                  "video",
                  "job=" + job->snapshot.id + " event=finished state=" +
                      (deleted ? std::string("deleted")
                               : std::string(ToString(snapshot.status))) +
                      " elapsed_ms=" +
                      std::to_string(
                          std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - start_time)
                              .count()) +
                      " error_code=" + snapshot.error_code + " " +
                      Logger::MemoryStatus());
      if (deleted) {
        std::error_code ignored;
        std::filesystem::remove_all(job->directory, ignored);
      } else {
        if (snapshot.status != VideoJobStatus::kCompleted) {
          std::error_code ignored;
          std::filesystem::remove(job->directory / "output.mp4", ignored);
          std::filesystem::remove(job->directory / "output.ppm", ignored);
          std::filesystem::remove_all(job->directory / "frames", ignored);
          std::filesystem::remove(job->directory / "telemetry.json", ignored);
        }
      }
      condition.notify_all();
    }
  }

  VideoJobServiceOptions options;
  const ext::VideoProvider* provider{nullptr};
  bool ready{false};
  std::string initialization_error;
  std::filesystem::path canonical_storage_root;
  std::mutex mutex;
  std::condition_variable condition;
  std::map<std::string, std::shared_ptr<Job>, std::less<>> jobs;
  std::deque<std::shared_ptr<Job>> queue;
  std::shared_ptr<Job> active;
  std::jthread worker;
};

VideoJobService::VideoJobService(VideoJobServiceOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

VideoJobService::~VideoJobService() = default;

bool VideoJobService::ready() const noexcept {
  return impl_ != nullptr && impl_->ready;
}

std::string VideoJobService::initialization_error() const {
  return impl_ == nullptr ? "video job service is unavailable"
                          : impl_->initialization_error;
}

const ext::VideoProvider* VideoJobService::provider() const noexcept {
  return impl_ == nullptr ? nullptr : impl_->provider;
}

std::string VideoJobService::model_id() const {
  const auto* video = provider();
  return IsPreparedProvider(video) ? video->id : std::string("minimax-h3");
}

bool VideoJobService::serves_model(std::string_view name) const {
  return IsServedModel(impl_->provider, name) ||
         std::find(impl_->options.model_aliases.begin(),
                   impl_->options.model_aliases.end(),
                   std::string(name)) != impl_->options.model_aliases.end();
}

std::string VideoJobService::normalize_model(std::string_view name) const {
  if (IsServedModel(impl_->provider, name))
    return std::string(name);
  return serves_model(name) ? model_id() : std::string(name);
}

const std::vector<std::string>& VideoJobService::model_aliases() const {
  return impl_->options.model_aliases;
}

VideoJobCreateResult VideoJobService::Create(const VideoJobRequest& request) {
  if (!ready()) {
    return {.result = VideoJobResult::kIoError,
            .job = std::nullopt,
            .error = initialization_error()};
  }
  const ext::VideoProvider* video = impl_->provider;
  const bool process_only = !request.source_video.empty();
  std::string post_error;
  if (process_only) {
    // Post-processing only: the generator's request shape does not apply.
    if (request.output_format != "mp4" || !request.post.enabled()) {
      return {.result = VideoJobResult::kInvalid,
              .job = std::nullopt,
              .error = "'source_video' needs MP4 output and 'interpolate' or "
                       "'upscale'"};
    }
  } else if (!IsServedModel(video, request.model) ||
             // Prepared providers decide whether a prompt is required (an
             // empty prompt is legal for image- or audio-driven video).
             (request.prompt.empty() && !IsPreparedProvider(video)) ||
             request.prompt.find('\0') != std::string::npos ||
             (request.output_format != "mp4" &&
              request.output_format != "ppm") ||
             (!IsPreparedProvider(video) &&
              !ValidateGenerationParameters(video, request.parameters,
                                            nullptr)) ||
             (IsPreparedProvider(video) &&
              video->validate_parameters != nullptr &&
              !ValidateGenerationParameters(video, request.parameters,
                                            nullptr)) ||
             !MatchesOutputContract(video, request)) {
    return {.result = VideoJobResult::kInvalid,
            .job = std::nullopt,
            .error = "invalid video job request for " + video->id};
  }
  // gufo-studio-windows post-steps (src/cli/serve/media).
  if (!ValidateVideoPostProcess(request.post, impl_->options.postprocessors,
                                request.output_format, &post_error)) {
    return {.result = VideoJobResult::kInvalid,
            .job = std::nullopt,
            .error = std::move(post_error)};
  }

  auto job = std::make_shared<Impl::Job>();
  job->request = request;
  job->snapshot.id = impl_->options.id_factory();
  job->snapshot.status = VideoJobStatus::kQueued;
  job->snapshot.created_at = impl_->options.now();
  // A post-processing-only job is recorded under the served model with a
  // size and duration every provider recovers after a restart.
  job->snapshot.model = process_only ? model_id() : request.model;
  job->snapshot.size = process_only ? std::string("512x512") : request.size;
  job->snapshot.seconds = process_only ? std::string("1") : request.seconds;
  job->snapshot.output_format = request.output_format;
  if (!IsVideoId(job->snapshot.id)) {
    WipeString(&job->request.prompt);
    return {.result = VideoJobResult::kInvalid,
            .job = std::nullopt,
            .error = "video job ID factory returned an invalid ID"};
  }
  job->directory = impl_->options.storage_root / job->snapshot.id;

  VideoJobSnapshot admitted;
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->SweepExpiredLocked();
    if (impl_->queue.size() >= impl_->options.queue_capacity) {
      WipeString(&job->request.prompt);
      return {.result = VideoJobResult::kQueueFull,
              .job = std::nullopt,
              .error = "video queue is full"};
    }
    if (impl_->jobs.contains(job->snapshot.id)) {
      WipeString(&job->request.prompt);
      return {.result = VideoJobResult::kInvalid,
              .job = std::nullopt,
              .error = "duplicate video job ID"};
    }
    std::error_code filesystem_error;
    const bool directory_exists =
        std::filesystem::exists(job->directory, filesystem_error);
    if (filesystem_error) {
      WipeString(&job->request.prompt);
      return {.result = VideoJobResult::kIoError,
              .job = std::nullopt,
              .error = "cannot inspect video job destination"};
    }
    if (directory_exists) {
      WipeString(&job->request.prompt);
      return {.result = VideoJobResult::kInvalid,
              .job = std::nullopt,
              .error = "video job destination already exists"};
    }
    impl_->jobs.emplace(job->snapshot.id, job);
    impl_->queue.push_back(job);
    std::string persist_error;
    if (!impl_->Persist(job->snapshot, &persist_error)) {
      impl_->queue.pop_back();
      impl_->jobs.erase(job->snapshot.id);
      std::error_code ignored;
      std::filesystem::remove_all(job->directory, ignored);
      WipeString(&job->request.prompt);
      return {.result = VideoJobResult::kIoError,
              .job = std::nullopt,
              .error = std::move(persist_error)};
    }
    admitted = impl_->Snapshot(*job);
  }
  Logger::Info("video", "job=" + admitted.id + " event=queued preset=" +
                            request.parameters.preset);
  impl_->condition.notify_one();
  return {
      .result = VideoJobResult::kOk, .job = std::move(admitted), .error = {}};
}

VideoJobLookupResult VideoJobService::Get(std::string_view id) {
  if (!IsVideoId(id)) {
    return {.result = VideoJobResult::kNotFound,
            .job = std::nullopt,
            .error = "video job not found"};
  }
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->SweepExpiredLocked();
  const auto iterator = impl_->jobs.find(id);
  if (iterator == impl_->jobs.end()) {
    return {.result = VideoJobResult::kNotFound,
            .job = std::nullopt,
            .error = "video job not found"};
  }
  return {.result = VideoJobResult::kOk,
          .job = impl_->Snapshot(*iterator->second),
          .error = {}};
}

VideoJobContentResult VideoJobService::Content(std::string_view id) {
  std::shared_ptr<Impl::Job> job;
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->SweepExpiredLocked();
    const auto iterator = impl_->jobs.find(id);
    if (iterator == impl_->jobs.end()) {
      return {.result = VideoJobResult::kNotFound,
              .content = std::nullopt,
              .error = "video job not found"};
    }
    job = iterator->second;
    if (job->snapshot.status != VideoJobStatus::kCompleted) {
      return {.result = VideoJobResult::kNotReady,
              .content = std::nullopt,
              .error = "video content is not ready"};
    }
  }
  std::uint64_t bytes = 0;
  if (!IsSafeRegularContent(impl_->canonical_storage_root, job->content_path,
                            &bytes)) {
    return {.result = VideoJobResult::kIoError,
            .content = std::nullopt,
            .error = "video content is unavailable"};
  }
  return {.result = VideoJobResult::kOk,
          .content =
              VideoJobContent{
                  .path = job->content_path,
                  .media_type = job->media_type,
                  .bytes = bytes,
              },
          .error = {}};
}

VideoJobLookupResult VideoJobService::Delete(std::string_view id) {
  std::shared_ptr<Impl::Job> job;
  VideoJobSnapshot snapshot;
  bool was_active = false;
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->SweepExpiredLocked();
    const auto iterator = impl_->jobs.find(id);
    if (iterator == impl_->jobs.end()) {
      return {.result = VideoJobResult::kNotFound,
              .job = std::nullopt,
              .error = "video job not found"};
    }
    job = iterator->second;
    snapshot = job->snapshot;
    job->deleted = true;
    job->cancelled.store(true, std::memory_order_relaxed);
    WipeString(&job->request.prompt);
    was_active = impl_->active == job;
    std::erase(impl_->queue, job);
    impl_->jobs.erase(iterator);
  }
  if (!was_active) {
    std::error_code ignored;
    std::filesystem::remove_all(job->directory, ignored);
  }
  impl_->condition.notify_all();
  return {
      .result = VideoJobResult::kOk, .job = std::move(snapshot), .error = {}};
}

}  // namespace gufo::server
