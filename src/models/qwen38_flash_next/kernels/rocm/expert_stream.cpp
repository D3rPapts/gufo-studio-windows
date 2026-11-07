#include "src/models/qwen38_flash_next/kernels/rocm/expert_stream.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <system_error>

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

void SetError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr)
    *error_msg = std::move(message);
}

}  // namespace

std::unique_ptr<ExpertStream> ExpertStream::Create(
    std::vector<Layer> layers, std::uint32_t num_experts, std::uint32_t slots,
    std::uint32_t max_misses, std::uint32_t threads, std::string* error_msg) {
  if (layers.empty() || num_experts == 0 || slots == 0 || slots > num_experts ||
      max_misses == 0 || threads == 0) {
    SetError(error_msg, "invalid expert stream configuration");
    return nullptr;
  }
  std::unique_ptr<ExpertStream> s(new ExpertStream());
  s->num_experts_ = num_experts;
  s->slots_ = slots;
  s->max_misses_ = max_misses;
  std::uint64_t set_bytes = 0;
  for (const auto& layer : layers) {
    std::uint64_t bytes = 0;
    for (const auto& tensor : layer.tensors) {
      if (!platform::FileValid(tensor.file) || tensor.pool == nullptr ||
          tensor.expert_bytes == 0) {
        SetError(error_msg, "incomplete expert stream layer");
        return nullptr;
      }
      bytes += tensor.expert_bytes;
    }
    set_bytes = std::max(set_bytes, bytes);
  }
  s->staging_bytes_ = static_cast<std::size_t>(set_bytes) * max_misses;
  if (hipHostMalloc(reinterpret_cast<void**>(&s->staging_), s->staging_bytes_,
                    hipHostMallocDefault) != hipSuccess ||
      hipEventCreateWithFlags(&s->staging_done_, hipEventDisableTiming) !=
          hipSuccess) {
    (void)hipGetLastError();
    SetError(error_msg, "cannot allocate expert staging");
    return nullptr;
  }
  s->state_.resize(layers.size());
  for (auto& state : s->state_) {
    state.slot_of.assign(num_experts, -1);
    state.expert_in.assign(slots, -1);
    state.used.assign(slots, 0);
  }
  s->layers_ = std::move(layers);
  for (std::uint32_t i = 0; i < threads; ++i)
    s->workers_.emplace_back([raw = s.get()] { raw->Worker(); });
  return s;
}

ExpertStream::~ExpertStream() {
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
  }
  work_.notify_all();
  for (auto& worker : workers_)
    worker.join();
  if (staging_done_ != nullptr) {
    (void)hipEventSynchronize(staging_done_);
    (void)hipEventDestroy(staging_done_);
  }
  if (staging_ != nullptr)
    (void)hipHostFree(staging_);
}

void ExpertStream::Worker() {
  for (;;) {
    Job job;
    {
      std::unique_lock lock(mutex_);
      work_.wait(lock, [this] { return stop_ || !queue_.empty(); });
      if (stop_)
        return;
      job = queue_.front();
      queue_.pop_front();
    }
    bool ok = true;
    std::uint64_t got = 0;
    while (got < job.bytes) {
      std::error_code ec;
      const std::size_t n =
          platform::ReadAt(job.file, job.offset + got, job.destination + got,
                           static_cast<std::size_t>(job.bytes - got), ec);
      if (n == 0 && ec == std::errc::interrupted)
        continue;
      if (n == 0) {
        ok = false;
        break;
      }
      got += n;
    }
    {
      std::lock_guard lock(mutex_);
      failed_ = failed_ || !ok;
      if (--pending_ == 0)
        done_.notify_all();
    }
  }
}

bool ExpertStream::ReadAll(std::vector<Job>& jobs, std::string* error_msg) {
  {
    std::lock_guard lock(mutex_);
    failed_ = false;
    pending_ = jobs.size();
    for (const auto& job : jobs)
      queue_.push_back(job);
  }
  work_.notify_all();
  std::unique_lock lock(mutex_);
  done_.wait(lock, [this] { return pending_ == 0; });
  if (failed_) {
    SetError(error_msg, "expert read failed");
    return false;
  }
  return true;
}

bool ExpertStream::Map(std::uint32_t layer, std::span<std::int32_t> ids,
                       hipStream_t stream, std::string* error_msg) {
  if (layer >= layers_.size()) {
    SetError(error_msg, "expert stream layer out of range");
    return false;
  }
  auto& state = state_[layer];
  const auto& tensors = layers_[layer].tensors;
  const std::uint64_t now = ++tick_;
  stats_.routed += ids.size();

  // Mark resident experts first, so loading never evicts one of this call's.
  std::vector<std::int32_t> missing;
  for (const std::int32_t expert : ids) {
    if (expert < 0 || static_cast<std::uint32_t>(expert) >= num_experts_) {
      SetError(error_msg, "routed expert id out of range");
      return false;
    }
    const std::int32_t slot = state.slot_of[expert];
    if (slot >= 0)
      state.used[slot] = now;
    else if (std::find(missing.begin(), missing.end(), expert) == missing.end())
      missing.push_back(expert);
  }
  if (missing.size() > max_misses_ || missing.size() > slots_) {
    SetError(error_msg, "expert stream batch exceeds its staging");
    return false;
  }

  if (!missing.empty()) {
    // The previous call's uploads must have left the staging buffer.
    if (hipEventSynchronize(staging_done_) != hipSuccess) {
      SetError(error_msg, "expert staging wait failed");
      return false;
    }
    std::vector<Job> jobs;
    std::vector<std::int32_t> targets;
    std::uint8_t* cursor = staging_;
    for (const std::int32_t expert : missing) {
      // Least recently used slot not touched by this call.
      std::int32_t victim = 0;
      for (std::int32_t slot = 1; slot < static_cast<std::int32_t>(slots_);
           ++slot)
        if (state.used[slot] < state.used[victim])
          victim = slot;
      if (state.used[victim] == now) {
        SetError(error_msg, "expert stream has no free slot");
        return false;
      }
      if (state.expert_in[victim] >= 0)
        state.slot_of[state.expert_in[victim]] = -1;
      state.expert_in[victim] = expert;
      state.slot_of[expert] = victim;
      state.used[victim] = now;
      targets.push_back(victim);
      for (const auto& tensor : tensors) {
        jobs.push_back(
            {tensor.file,
             tensor.file_offset +
                 static_cast<std::uint64_t>(expert) * tensor.expert_bytes,
             tensor.expert_bytes, cursor});
        cursor += tensor.expert_bytes;
      }
    }
    const auto start = std::chrono::steady_clock::now();
    if (!ReadAll(jobs, error_msg))
      return false;
    stats_.read_ms += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    std::size_t job = 0;
    for (const std::int32_t slot : targets) {
      for (const auto& tensor : tensors) {
        auto* destination =
            static_cast<std::uint8_t*>(tensor.pool) +
            static_cast<std::uint64_t>(slot) * tensor.expert_bytes;
        if (hipMemcpyAsync(destination, jobs[job].destination,
                           tensor.expert_bytes, hipMemcpyHostToDevice,
                           stream) != hipSuccess) {
          SetError(error_msg, "expert upload failed");
          return false;
        }
        stats_.bytes_read += tensor.expert_bytes;
        ++job;
      }
    }
    if (hipEventRecord(staging_done_, stream) != hipSuccess) {
      SetError(error_msg, "expert staging record failed");
      return false;
    }
    stats_.misses += missing.size();
  }
  for (std::int32_t& id : ids)
    id = state.slot_of[id];
  Report();
  return true;
}

void ExpertStream::Report() {
  const auto now = std::chrono::steady_clock::now();
  if (report_at_ == std::chrono::steady_clock::time_point{}) {
    report_at_ = now;
    return;
  }
  const double seconds =
      std::chrono::duration<double>(now - report_at_).count();
  if (seconds < 20)
    return;
  const std::uint64_t routed = stats_.routed - reported_.routed;
  const std::uint64_t misses = stats_.misses - reported_.misses;
  const double gb =
      static_cast<double>(stats_.bytes_read - reported_.bytes_read) / 1e9;
  const double read_s = (stats_.read_ms - reported_.read_ms) / 1000;
  // One line in the server's stderr log, beside its own event lines.
  std::fprintf(
      stderr,
      "[INFO] [experts] event=stream_stats window_s=%.0f routed=%llu "
      "misses=%llu hit_rate=%.1f%% read_gb=%.2f read_s=%.1f "
      "read_gbps=%.2f total_misses=%llu total_read_gb=%.1f\n",
      seconds, static_cast<unsigned long long>(routed),
      static_cast<unsigned long long>(misses),
      routed == 0 ? 0.0 : 100.0 * (1.0 - double(misses) / double(routed)), gb,
      read_s, read_s > 0 ? gb / read_s : 0.0,
      static_cast<unsigned long long>(stats_.misses),
      static_cast<double>(stats_.bytes_read) / 1e9);
  std::fflush(stderr);
  reported_ = stats_;
  report_at_ = now;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
