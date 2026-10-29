#ifndef GUFO_CLI_SERVE_ROUTER_RESIDENCY_HPP_
#define GUFO_CLI_SERVE_ROUTER_RESIDENCY_HPP_

// Glue between the router state and the memory module: samples worker and
// system memory, keeps learned peaks, turns registry residents into planner
// input, records decisions for GET /router/status.

#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/router/memory.hpp"
#include "src/cli/serve/router/preset.hpp"
#include "src/cli/serve/router/registry.hpp"
#include "src/core/json.hpp"

namespace gufo::router {

struct ResidencyOptions {
  bool enabled{false};
  MemoryPolicy policy;
  std::filesystem::path memory_file;  // learned peaks; empty = not persisted
  std::chrono::seconds idle_default{0};
};

/// Thread-safe; never calls back into the router, so it may be used with
/// the router mutex held or free.
class Residency {
public:
  using Clock = std::chrono::steady_clock;
  using WorkerPids = std::vector<std::pair<std::string, std::uint32_t>>;

  Residency(ResidencyOptions options, std::unique_ptr<MemoryProbe> probe,
            const std::vector<PresetModel>& presets);

  [[nodiscard]] bool enabled() const { return options_.enabled; }
  [[nodiscard]] const MemoryPolicy& policy() const { return options_.policy; }
  /// Reads the memory file and applies the preset estimates on top.
  bool Init(std::string* error);

  /// Fresh sample, then the admission plan for `id`.
  AdmissionPlan PlanFor(std::string_view id,
                        const std::vector<Registry::ResidentEntry>& residents,
                        const WorkerPids& pids);

  /// Free GPU+host before an eviction and what the evicted workers held.
  struct ReleaseTicket {
    std::int64_t gpu_free_before{0};
    std::int64_t host_free_before{0};
    std::int64_t gpu_expected{0};
    std::int64_t host_expected{0};
  };
  [[nodiscard]] ReleaseTicket BeginRelease(const std::vector<std::string>& ids);
  /// Polls until most of the evicted memory shows up as free again (the
  /// driver returns it shortly after process exit) or `timeout` passes.
  /// Call without the router mutex.
  void AwaitRelease(const ReleaseTicket& ticket,
                    std::chrono::milliseconds timeout);

  /// Samples system + worker memory and updates learned peaks (saved to the
  /// memory file at most every 30 s while growing). Call without the router
  /// mutex.
  void Sample(const WorkerPids& pids);
  /// Watchdog on the last sample.
  WatchdogAction Watchdog(
      const std::vector<Registry::ResidentEntry>& residents);

  void OnSpawn(std::string_view id);
  /// Worker gone: saves learned peaks, drops the cached usage.
  void OnUnloaded(std::string_view id);
  /// The watchdog killed this load: remember it so the waiting request fails
  /// instead of respawning, and raise the learned peak.
  void OnAborted(std::string_view id);
  /// True when `id` was aborted after `since`.
  [[nodiscard]] bool AbortedSince(std::string_view id,
                                  Clock::time_point since) const;

  void Event(std::string text);

  [[nodiscard]] json::Value Status(
      const std::vector<Registry::ResidentEntry>& residents,
      const std::set<std::string>& stopping,
      Clock::time_point registry_now) const;

private:
  [[nodiscard]] std::vector<ResidentView> ViewsLocked(
      const std::vector<Registry::ResidentEntry>& residents) const;
  void SampleLocked(const WorkerPids& pids);
  void SaveLocked(bool force);

  ResidencyOptions options_;
  std::unique_ptr<MemoryProbe> probe_;
  MemoryBook book_;
  std::vector<std::pair<std::string, PresetHints>> hints_;
  mutable std::mutex mutex_;
  SystemMemory system_;
  std::map<std::string, ProcessMemory, std::less<>> current_;
  std::map<std::string, Clock::time_point, std::less<>> spawned_;
  std::map<std::string, Clock::time_point, std::less<>> aborted_;
  std::deque<std::pair<std::string, std::string>> events_;  // time, text
  Clock::time_point last_save_{};
  bool learned_pending_{false};
};

}  // namespace gufo::router

#endif  // GUFO_CLI_SERVE_ROUTER_RESIDENCY_HPP_
