#ifndef GUFO_CLI_SERVE_ROUTER_REGISTRY_HPP_
#define GUFO_CLI_SERVE_ROUTER_REGISTRY_HPP_

#include <chrono>
#include <cstddef>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/router/preset.hpp"

namespace gufo::router {

struct RegistryOptions {
  std::size_t max_active{2};               // 0 = unlimited
  std::chrono::seconds idle_timeout{900};  // 0 disables idle unload
  std::chrono::seconds load_timeout{600};  // 0 = unlimited wait
};

enum class LoadDecision { kSpawn, kAwaitLoad, kQueue, kLoadTimeout };

/// Admission control, LRU eviction and idle accounting for router workers.
/// Pure state: no processes, sockets or sleeps. The clock is injectable for
/// tests (nullptr = real steady_clock).
class Registry {
public:
  using Clock = std::chrono::steady_clock;
  explicit Registry(std::vector<PresetModel> presets, RegistryOptions options,
                    std::function<Clock::time_point()> now = nullptr);

  [[nodiscard]] const PresetModel* Find(std::string_view model_id) const;
  /// Section order, loaded (ready) flag per model.
  [[nodiscard]] std::vector<std::pair<std::string, bool>> Listing() const;
  /// Section order with the llama.cpp router's status names:
  /// "unloaded", "loading" or "loaded".
  struct ModelStatus {
    const PresetModel* preset;
    std::string_view status;
  };
  [[nodiscard]] std::vector<ModelStatus> Statuses() const;

  /// Called by the front for every request needing the model. Tracks the
  /// caller's deadline on first call: spawn + queue + readiness together may
  /// not exceed load_timeout, else returns kLoadTimeout (cleared by
  /// MarkUnloaded; a fresh request re-arms a fresh deadline).
  LoadDecision RequestLoad(std::string_view model_id);

  // Supervision-thread queries.
  /// LRU unloadable id among ready+loading (never Busy), "" if none.
  [[nodiscard]] std::string EvictionCandidate();
  [[nodiscard]] bool HasQueued() const;
  [[nodiscard]] std::vector<std::string> DueIdleUnloads();
  /// Loading or queued ids past the load deadline, section order.
  [[nodiscard]] std::vector<std::string> DueLoadTimeouts();

  // State transitions driven by the supervisor / proxy.
  void MarkLoading(std::string_view model_id);
  void MarkReady(std::string_view model_id);
  void MarkUnloaded(std::string_view model_id);

  // Activity accounting (all nestable per connection).
  void RequestStarted(std::string_view model_id);
  void RequestFinished(std::string_view model_id);
  void WebSocketOpened(std::string_view model_id);
  void WebSocketClosed(std::string_view model_id);
  void TrackVideoJob(std::string_view job_id, std::string_view model_id);
  void CompleteVideoJob(std::string_view job_id);
  /// Owner model id; "" when the job is unknown.
  [[nodiscard]] std::string JobOwner(std::string_view job_id) const;

  /// True while any request, socket or video job is open for the model.
  [[nodiscard]] bool Busy(std::string_view model_id) const;
  /// Number of loading or ready models.
  [[nodiscard]] std::size_t ActiveCount() const;

  /// Loading or ready models, section order (memory residency, status).
  struct ResidentEntry {
    std::string id;
    bool loading{false};
    bool busy{false};
    Clock::time_point last_active{};
    /// Idle unload after this long; 0 = never.
    std::chrono::seconds idle_timeout{0};
  };
  [[nodiscard]] std::vector<ResidentEntry> Residents() const;
  [[nodiscard]] Clock::time_point NowForStatus() const { return Now(); }

private:
  enum class State { kUnloaded, kLoading, kReady };

  struct Entry {
    PresetModel preset;
    State state{State::kUnloaded};
    std::size_t open_requests{0};
    std::size_t open_sockets{0};
    std::size_t open_jobs{0};
    Clock::time_point last_active{};
    bool deadline_armed{false};
    Clock::time_point deadline{};
    bool queued{false};
  };

  [[nodiscard]] Clock::time_point Now() const;
  [[nodiscard]] Entry* Lookup(std::string_view model_id);
  [[nodiscard]] const Entry* Lookup(std::string_view model_id) const;
  [[nodiscard]] bool DeadlineExpired(const Entry& entry) const;
  [[nodiscard]] static bool BusyEntry(const Entry& entry);
  void Touch(Entry& entry);
  /// Router default unless the preset sets router-idle-seconds.
  [[nodiscard]] std::chrono::seconds IdleTimeout(const Entry& entry) const;

  std::vector<Entry> entries_;
  std::map<std::string, std::string> job_owners_;
  // Finished jobs keep their owner so /content and status reads still route
  // there (the worker reloads and answers from its persisted job state);
  // oldest first, bounded.
  std::map<std::string, std::string> finished_owners_;
  std::vector<std::string> finished_order_;
  RegistryOptions options_;
  std::function<Clock::time_point()> now_;
};

}  // namespace gufo::router

#endif  // GUFO_CLI_SERVE_ROUTER_REGISTRY_HPP_
