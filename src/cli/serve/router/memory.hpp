#ifndef GUFO_CLI_SERVE_ROUTER_MEMORY_HPP_
#define GUFO_CLI_SERVE_ROUTER_MEMORY_HPP_

// Memory-aware residency for router workers: live memory probes, learned
// per-model peaks, and the pure admission planner that decides whether a new
// worker may start beside the resident ones, which idle ones to evict first,
// or whether the request has to wait.

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::router {

constexpr std::int64_t kGiB = 1024LL * 1024 * 1024;

/// Live system-wide memory. GPU numbers are for the adapter the workers use
/// (dedicated = VRAM carve-out, shared = system RAM the GPU may map); shared
/// GPU allocations are backed by host RAM, so they also lower host_available.
struct SystemMemory {
  bool gpu_valid{false};
  bool host_valid{false};
  std::int64_t gpu_dedicated_total{0};
  std::int64_t gpu_dedicated_used{0};
  std::int64_t gpu_shared_total{0};
  std::int64_t gpu_shared_used{0};
  std::int64_t host_total{0};
  std::int64_t host_available{0};

  [[nodiscard]] std::int64_t DedicatedFree() const;
  [[nodiscard]] std::int64_t SharedFree() const;
};

/// One worker process. host_* is the working set (on this APU, VRAM
/// allocations are charged to private bytes, so commit is not host RAM).
struct ProcessMemory {
  bool valid{false};
  std::int64_t gpu_dedicated{0};
  std::int64_t gpu_shared{0};
  std::int64_t host_working_set{0};
  std::int64_t host_peak_working_set{0};

  [[nodiscard]] std::int64_t Gpu() const { return gpu_dedicated + gpu_shared; }
};

class MemoryProbe {
public:
  virtual ~MemoryProbe() = default;
  virtual SystemMemory System() = 0;
  /// Usage per pid; pids that cannot be read are missing from the map.
  virtual std::map<std::uint32_t, ProcessMemory> Processes(
      const std::vector<std::uint32_t>& pids) = 0;
};

/// DXGI + PDH + psapi on Windows; host-only (/proc/meminfo) elsewhere.
std::unique_ptr<MemoryProbe> MakeSystemMemoryProbe();

/// Peak a model needs: total GPU (dedicated first, the driver spills the rest
/// into shared) and host working set.
struct MemoryNeed {
  std::int64_t gpu{0};
  std::int64_t host{0};
};

struct MemoryPolicy {
  double factor{1.15};                    // safety factor on estimates/peaks
  std::int64_t reserve_gpu{3 * kGiB};     // kept free on the GPU (desktop, dwm)
  std::int64_t reserve_host{4 * kGiB};    // kept free in host RAM
  std::int64_t abort_host{3 * kGiB / 2};  // watchdog: host below this
  std::int64_t abort_gpu{kGiB / 2};       // watchdog: GPU (ded+shared) below
};

/// What the planner knows about one resident (loading or loaded) worker.
struct ResidentView {
  std::string id;
  bool loading{false};
  bool busy{false};
  bool exclusive{false};
  MemoryNeed need;        // already multiplied by the safety factor
  ProcessMemory current;  // last sample; invalid = nothing measured yet
  std::chrono::steady_clock::time_point last_active{};
};

struct AdmissionPlan {
  enum class Action { kSpawn, kEvict, kWait };
  Action action{Action::kWait};
  std::vector<std::string> evict;  // kEvict: stop these, then plan again
  bool alone{false};  // spawn although the estimate does not fit: no other
                      // worker is left that could be evicted
  std::string reason;
};

/// Pure admission decision for starting `id` (need already factored). Rules:
///  - memory still owed by residents (need minus current usage) is counted
///    as taken, so concurrent loads cannot double-book the same free bytes;
///  - fits: dedicated+shared free minus reserve_gpu >= need.gpu, and host
///    free minus reserve_host minus the part that spills into shared >=
///    need.host;
///  - otherwise idle, fully loaded residents are evicted least recently used
///    first until it fits; busy or loading residents are never evicted (wait);
///  - an exclusive candidate evicts every idle resident and waits for the
///    rest; an exclusive resident blocks every other model the same way.
AdmissionPlan PlanAdmission(const SystemMemory& system,
                            const std::vector<ResidentView>& residents,
                            std::string_view id, const MemoryNeed& need,
                            bool exclusive, const MemoryPolicy& policy);

/// Watchdog decision for one sample.
struct WatchdogAction {
  std::string abort_load;  // kill this loading worker (newest load)
  std::string evict;       // or stop this idle resident
  std::string reason;      // non-empty when memory is critical
};

/// `loading_order` lists loading ids oldest first.
WatchdogAction PlanWatchdog(const SystemMemory& system,
                            const std::vector<ResidentView>& residents,
                            const std::vector<std::string>& loading_order,
                            const MemoryPolicy& policy);

/// Estimates (preset keys or the JSON file) and learned peaks per model,
/// persisted as JSON. Thread-safe.
class MemoryBook {
public:
  struct Record {
    bool has_estimate{false};
    MemoryNeed estimate;
    bool exclusive{false};
    bool has_learned{false};
    MemoryNeed learned;
    std::string learned_at;
  };

  /// Empty path: nothing is read or written.
  explicit MemoryBook(std::filesystem::path file);

  /// Reads the file if it exists; returns false + error on a malformed file.
  bool Load(std::string* error);
  /// Writes when something changed since the last save (temp file + rename).
  bool Save(std::string* error);

  void SetEstimate(std::string_view id, const MemoryNeed& estimate);
  void SetExclusive(std::string_view id, bool exclusive);
  /// Raises the learned peak; true when it grew by more than 256 MiB.
  bool Observe(std::string_view id, const ProcessMemory& usage);
  /// Raises the learned peak to at least `need` (watchdog abort).
  void RaiseLearned(std::string_view id, const MemoryNeed& need);

  /// max(estimate, learned) x factor. A model with neither is exclusive.
  [[nodiscard]] MemoryNeed Need(std::string_view id, double factor,
                                bool* exclusive) const;
  [[nodiscard]] Record Get(std::string_view id) const;
  [[nodiscard]] const std::filesystem::path& file() const { return file_; }

private:
  std::filesystem::path file_;
  mutable std::mutex mutex_;
  std::map<std::string, Record, std::less<>> records_;
  bool dirty_{false};
};

[[nodiscard]] double ToGb(std::int64_t bytes);
[[nodiscard]] std::int64_t FromGb(double gb);

}  // namespace gufo::router

#endif  // GUFO_CLI_SERVE_ROUTER_MEMORY_HPP_
