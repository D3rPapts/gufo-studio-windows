#include "src/cli/serve/router/residency.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <utility>

#include "src/cli/serve/logging.hpp"

namespace gufo::router {
namespace {

constexpr std::size_t kEventsKept = 20;
constexpr std::chrono::seconds kSaveInterval{30};
constexpr std::chrono::seconds kReleaseWait{30};

double Round1(std::int64_t bytes) {
  return std::round(ToGb(bytes) * 10.0) / 10.0;
}

std::string LocalTime() {
  const std::time_t now = std::time(nullptr);
  std::tm local{};
#ifdef _WIN32
  localtime_s(&local, &now);
#else
  localtime_r(&now, &local);
#endif
  char text[32];
  std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%S", &local);
  return text;
}

}  // namespace

Residency::Residency(ResidencyOptions options,
                     std::unique_ptr<MemoryProbe> probe,
                     const std::vector<PresetModel>& presets)
    : options_(std::move(options)),
      probe_(std::move(probe)),
      book_(options_.memory_file) {
  for (const auto& preset : presets) {
    hints_.emplace_back(preset.id, preset.hints);
  }
}

bool Residency::Init(std::string* error) {
  if (!book_.Load(error)) {
    return false;
  }
  // Preset keys win over the file's estimates.
  for (const auto& [id, hints] : hints_) {
    if (hints.gpu_gb >= 0.0) {
      book_.SetEstimate(id, {FromGb(hints.gpu_gb),
                             hints.host_gb >= 0.0 ? FromGb(hints.host_gb) : 0});
    }
    if (hints.exclusive >= 0) {
      book_.SetExclusive(id, hints.exclusive == 1);
    }
  }
  return true;
}

std::vector<ResidentView> Residency::ViewsLocked(
    const std::vector<Registry::ResidentEntry>& residents) const {
  std::vector<ResidentView> views;
  views.reserve(residents.size());
  for (const auto& entry : residents) {
    ResidentView view;
    view.id = entry.id;
    view.loading = entry.loading;
    view.busy = entry.busy;
    view.last_active = entry.last_active;
    view.need = book_.Need(entry.id, options_.policy.factor,
                           options_.policy.learned_factor, &view.exclusive);
    if (const auto it = current_.find(entry.id); it != current_.end()) {
      view.current = it->second;
    }
    views.push_back(std::move(view));
  }
  return views;
}

void Residency::SampleLocked(const WorkerPids& pids) {
  system_ = probe_->System();
  // Workers that are gone (exit, stop) leave the cache; their peaks stay in
  // the book.
  for (auto it = current_.begin(); it != current_.end();) {
    const bool live = std::any_of(
        pids.begin(), pids.end(),
        [&](const auto& worker) { return worker.first == it->first; });
    if (live) {
      ++it;
    } else {
      spawned_.erase(it->first);
      it = current_.erase(it);
    }
  }
  std::vector<std::uint32_t> list;
  for (const auto& [id, pid] : pids) {
    if (pid != 0) {
      list.push_back(pid);
    }
  }
  const auto usage = probe_->Processes(list);
  for (const auto& [id, pid] : pids) {
    const auto it = usage.find(pid);
    if (it == usage.end()) {
      continue;
    }
    current_[id] = it->second;
    if (book_.Observe(id, it->second)) {
      learned_pending_ = true;
    }
  }
}

void Residency::SaveLocked(bool force) {
  if (!learned_pending_ && !force) {
    return;
  }
  const auto now = Clock::now();
  if (!force && now - last_save_ < kSaveInterval) {
    return;
  }
  std::string error;
  if (!book_.Save(&error)) {
    server::Logger::Warn("router", "event=memory_file_error " + error);
  }
  last_save_ = now;
  learned_pending_ = false;
}

AdmissionPlan Residency::PlanFor(
    std::string_view id, const std::vector<Registry::ResidentEntry>& residents,
    const WorkerPids& pids) {
  const std::lock_guard<std::mutex> lock(mutex_);
  SampleLocked(pids);
  // A stopped worker returns its GPU memory with a delay (seconds after a
  // kill mid-load): plan only on settled numbers, or a second eviction would
  // follow the first for nothing.
  if (const std::string waiting = PendingReleaseLocked(); !waiting.empty()) {
    AdmissionPlan plan;
    plan.action = AdmissionPlan::Action::kWait;
    plan.reason = "waiting until " + waiting + " has returned its memory";
    return plan;
  }
  bool exclusive = false;
  const MemoryNeed need = book_.Need(
      id, options_.policy.factor, options_.policy.learned_factor, &exclusive);
  return PlanAdmission(system_, ViewsLocked(residents), id, need, exclusive,
                       options_.policy);
}

void Residency::NoteStopping(std::string_view id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = current_.find(id);
  if (it == current_.end() || !it->second.valid) {
    return;
  }
  Release release;
  release.id = std::string(id);
  release.gpu_free_before = system_.DedicatedFree() + system_.SharedFree();
  release.host_free_before = system_.host_available;
  release.gpu_expected = it->second.Gpu();
  release.host_expected = it->second.host_working_set;
  release.until = Clock::now() + kReleaseWait;
  releases_.push_back(std::move(release));
}

std::string Residency::PendingReleaseLocked() {
  const auto now = Clock::now();
  std::string waiting;
  for (auto it = releases_.begin(); it != releases_.end();) {
    // Most of the stopped worker's memory is free again, or we gave up.
    const bool gpu_back = !system_.gpu_valid ||
                          system_.DedicatedFree() + system_.SharedFree() >=
                              it->gpu_free_before + it->gpu_expected * 7 / 10;
    const bool host_back =
        !system_.host_valid ||
        system_.host_available >= it->host_free_before + it->host_expected / 2;
    if ((gpu_back && host_back) || now >= it->until) {
      server::Logger::Info(
          "router",
          "event=memory_released model=" + it->id + " complete=" +
              (gpu_back && host_back ? "yes" : "timeout") + " gpu_free_gb=" +
              std::to_string(
                  ToGb(system_.DedicatedFree() + system_.SharedFree())) +
              " host_free_gb=" + std::to_string(ToGb(system_.host_available)));
      it = releases_.erase(it);
    } else {
      waiting += (waiting.empty() ? "" : ", ") + it->id;
      ++it;
    }
  }
  return waiting;
}

void Residency::Sample(const WorkerPids& pids) {
  const std::lock_guard<std::mutex> lock(mutex_);
  SampleLocked(pids);
  SaveLocked(false);
}

WatchdogAction Residency::Watchdog(
    const std::vector<Registry::ResidentEntry>& residents) {
  const std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::pair<Clock::time_point, std::string>> loading;
  for (const auto& entry : residents) {
    if (entry.loading) {
      const auto it = spawned_.find(entry.id);
      loading.emplace_back(
          it == spawned_.end() ? Clock::time_point{} : it->second, entry.id);
    }
  }
  std::sort(loading.begin(), loading.end());
  std::vector<std::string> order;
  for (auto& [time, id] : loading) {
    order.push_back(std::move(id));
  }
  return PlanWatchdog(system_, ViewsLocked(residents), order, options_.policy);
}

void Residency::OnSpawn(std::string_view id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  spawned_[std::string(id)] = Clock::now();
  current_.erase(std::string(id));
}

void Residency::OnUnloaded(std::string_view id, bool complete) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (complete) {
    book_.MarkComplete(id);
  }
  current_.erase(std::string(id));
  spawned_.erase(std::string(id));
  SaveLocked(true);
}

void Residency::OnAborted(std::string_view id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  aborted_[std::string(id)] = Clock::now();
  // The load was still growing when memory ran out: count what it held plus
  // the missing headroom, so the next admission makes that much room.
  if (const auto it = current_.find(id); it != current_.end()) {
    MemoryNeed raised;
    raised.gpu = it->second.Gpu() +
                 std::max<std::int64_t>(0, options_.policy.abort_gpu -
                                               system_.DedicatedFree() -
                                               system_.SharedFree());
    raised.host = std::max(it->second.host_working_set,
                           it->second.host_peak_working_set) +
                  std::max<std::int64_t>(
                      0, options_.policy.abort_host - system_.host_available);
    book_.RaiseLearned(id, raised);
    learned_pending_ = true;
  }
}

bool Residency::AbortedSince(std::string_view id,
                             Clock::time_point since) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = aborted_.find(id);
  return it != aborted_.end() && it->second >= since;
}

void Residency::Event(std::string text) {
  server::Logger::Info("router", "event=residency " + text);
  const std::lock_guard<std::mutex> lock(mutex_);
  events_.emplace_back(LocalTime(), std::move(text));
  while (events_.size() > kEventsKept) {
    events_.pop_front();
  }
}

json::Value Residency::Status(
    const std::vector<Registry::ResidentEntry>& residents,
    const std::set<std::string>& stopping,
    Clock::time_point registry_now) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  json::Value root = json::Value::object();
  json::Value memory = json::Value::object();
  memory["gpu_dedicated_free_gb"] = Round1(system_.DedicatedFree());
  memory["gpu_dedicated_total_gb"] = Round1(system_.gpu_dedicated_total);
  memory["gpu_shared_free_gb"] = Round1(system_.SharedFree());
  memory["gpu_shared_total_gb"] = Round1(system_.gpu_shared_total);
  memory["host_free_gb"] = Round1(system_.host_available);
  memory["host_total_gb"] = Round1(system_.host_total);
  memory["gpu_valid"] = system_.gpu_valid;
  root["memory"] = std::move(memory);
  root["memory_guard"] = options_.enabled;
  root["idle_unload_seconds"] =
      static_cast<long long>(options_.idle_default.count());
  json::Value policy = json::Value::object();
  policy["factor"] = options_.policy.factor;
  policy["learned_factor"] = options_.policy.learned_factor;
  policy["reserve_gpu_gb"] = Round1(options_.policy.reserve_gpu);
  policy["reserve_host_gb"] = Round1(options_.policy.reserve_host);
  policy["abort_host_gb"] = Round1(options_.policy.abort_host);
  policy["abort_gpu_gb"] = Round1(options_.policy.abort_gpu);
  root["policy"] = std::move(policy);

  json::Value models = json::Value::array();
  const auto add = [&](const std::string& id, std::string_view state, bool busy,
                       const Registry::ResidentEntry* entry) {
    json::Value model = json::Value::object();
    model["id"] = id;
    model["state"] = std::string(state);
    model["busy"] = busy;
    bool exclusive = false;
    const MemoryNeed need = book_.Need(
        id, options_.policy.factor, options_.policy.learned_factor, &exclusive);
    const MemoryBook::Record record = book_.Get(id);
    model["exclusive"] = exclusive;
    ProcessMemory usage;
    if (const auto it = current_.find(id); it != current_.end()) {
      usage = it->second;
    }
    model["gpu_gb"] = Round1(usage.Gpu());
    model["gpu_dedicated_gb"] = Round1(usage.gpu_dedicated);
    model["gpu_shared_gb"] = Round1(usage.gpu_shared);
    model["host_gb"] = Round1(usage.host_working_set);
    model["estimate_gb"] = record.has_estimate
                               ? json::Value(Round1(record.estimate.gpu))
                               : json::Value();
    model["learned_peak_gb"] = record.has_learned
                                   ? json::Value(Round1(record.learned.gpu))
                                   : json::Value();
    model["learned_host_peak_gb"] =
        record.has_learned ? json::Value(Round1(record.learned.host))
                           : json::Value();
    model["need_gb"] = Round1(need.gpu);
    if (entry != nullptr) {
      const long long idle = std::chrono::duration_cast<std::chrono::seconds>(
                                 registry_now - entry->last_active)
                                 .count();
      model["idle_seconds"] =
          busy || entry->loading ? 0LL : std::max(0LL, idle);
      if (entry->idle_timeout.count() > 0 && !busy && !entry->loading) {
        model["unload_in_seconds"] = std::max(
            0LL, static_cast<long long>(entry->idle_timeout.count()) - idle);
      } else {
        model["unload_in_seconds"] = json::Value();
      }
    } else {
      model["idle_seconds"] = json::Value();
      model["unload_in_seconds"] = json::Value();
    }
    models.push_back(std::move(model));
  };
  for (const auto& entry : residents) {
    add(entry.id, entry.loading ? "loading" : (entry.busy ? "busy" : "loaded"),
        entry.busy, &entry);
  }
  for (const auto& id : stopping) {
    add(id, "unloading", false, nullptr);
  }
  root["models"] = std::move(models);

  json::Value events = json::Value::array();
  for (const auto& [time, text] : events_) {
    json::Value event = json::Value::object();
    event["time"] = time;
    event["text"] = text;
    events.push_back(std::move(event));
  }
  root["events"] = std::move(events);
  return root;
}

}  // namespace gufo::router
