#include "src/cli/serve/router/memory.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iterator>
#include <sstream>
#include <system_error>
#include <utility>

#include "src/core/json.hpp"

namespace gufo::router {
namespace {

namespace json = gufo::json;

constexpr std::int64_t kUnbounded = std::int64_t{1} << 60;
constexpr std::int64_t kLearnStep = 256LL * 1024 * 1024;

/// Free bytes still available to a new worker.
struct Room {
  std::int64_t dedicated{0};
  std::int64_t shared{0};
  std::int64_t host{0};
};

/// Live free memory, plus what evicted residents give back, minus what the
/// kept residents still owe (need not yet reached by their measured usage).
Room Project(const SystemMemory& system,
             const std::vector<const ResidentView*>& kept,
             const std::vector<const ResidentView*>& evicted) {
  Room room;
  room.dedicated = system.gpu_valid ? system.DedicatedFree() : kUnbounded;
  room.shared = system.gpu_valid ? system.SharedFree() : kUnbounded;
  room.host = system.host_valid ? system.host_available : kUnbounded;
  for (const ResidentView* view : evicted) {
    if (!view->current.valid) {
      continue;
    }
    room.dedicated += view->current.gpu_dedicated;
    room.shared += view->current.gpu_shared;
    // A worker's own shared buffers (pinned host memory) are part of its
    // working set: measured on the 27B, host free dropped by the working set
    // alone. Count them once.
    room.host += view->current.host_working_set;
  }
  for (const ResidentView* view : kept) {
    const bool measured = view->current.valid;
    const std::int64_t need_dedicated = view->need.gpu - view->need.shared;
    const std::int64_t owed_dedicated = std::max<std::int64_t>(
        0, need_dedicated - (measured ? view->current.gpu_dedicated : 0));
    const std::int64_t owed_shared = std::max<std::int64_t>(
        0, view->need.shared - (measured ? view->current.gpu_shared : 0));
    const std::int64_t owed_host = std::max<std::int64_t>(
        0, view->need.host - (measured ? view->current.host_working_set : 0));
    const std::int64_t from_dedicated =
        std::min(owed_dedicated, std::max<std::int64_t>(0, room.dedicated));
    const std::int64_t spill = owed_dedicated - from_dedicated;
    room.dedicated -= from_dedicated;
    room.shared -= spill + owed_shared;
    room.host -= spill + owed_host;
  }
  return room;
}

bool Fits(const Room& room, const MemoryNeed& need,
          const MemoryPolicy& policy) {
  // The GPU reserve guards dedicated + shared together: a full dedicated
  // segment is no failure (the driver places the rest in shared), but every
  // byte that spills into shared comes out of host RAM.
  const std::int64_t dedicated_part = std::max<std::int64_t>(0, room.dedicated);
  if (dedicated_part + room.shared - policy.reserve_gpu < need.gpu) {
    return false;
  }
  // Only what the model wants in dedicated but does not find there spills;
  // its own shared buffers are already in need.host.
  const std::int64_t spill =
      std::max<std::int64_t>(0, need.gpu - need.shared - dedicated_part);
  return room.host - policy.reserve_host - spill >= need.host;
}

/// True when even an empty machine could not give `need` (estimate larger
/// than the hardware): waiting would never end, so such a model may start
/// alone and the watchdog guards it.
bool NeverFits(const SystemMemory& system, const MemoryNeed& need,
               const MemoryPolicy& policy) {
  if (system.gpu_valid && system.gpu_dedicated_total + system.gpu_shared_total -
                                  policy.reserve_gpu <
                              need.gpu) {
    return true;
  }
  return system.host_valid &&
         system.host_total - policy.reserve_host < need.host;
}

std::string Gb(std::int64_t bytes) {
  std::ostringstream out;
  out.setf(std::ios::fixed);
  out.precision(1);
  out << ToGb(bytes);
  return out.str();
}

std::string RoomText(const Room& room) {
  const auto part = [](std::int64_t value) {
    return value >= kUnbounded / 2 ? std::string("n/a") : Gb(value);
  };
  return "free gpu " + part(room.dedicated) + "+" + part(room.shared) +
         " GB, host " + part(room.host) + " GB";
}

std::string NowText() {
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

std::int64_t SystemMemory::DedicatedFree() const {
  return std::max<std::int64_t>(0, gpu_dedicated_total - gpu_dedicated_used);
}

std::int64_t SystemMemory::SharedFree() const {
  return std::max<std::int64_t>(0, gpu_shared_total - gpu_shared_used);
}

double ToGb(std::int64_t bytes) {
  return static_cast<double>(bytes) / static_cast<double>(kGiB);
}

std::int64_t FromGb(double gb) {
  return static_cast<std::int64_t>(
      std::llround(gb * static_cast<double>(kGiB)));
}

AdmissionPlan PlanAdmission(const SystemMemory& system,
                            const std::vector<ResidentView>& residents,
                            std::string_view id, const MemoryNeed& need,
                            bool exclusive, const MemoryPolicy& policy) {
  AdmissionPlan plan;
  std::vector<const ResidentView*> others;
  for (const auto& view : residents) {
    if (view.id != id) {
      others.push_back(&view);
    }
  }
  const auto evictable = [](const ResidentView* view) {
    return !view->loading && !view->busy;
  };
  std::vector<const ResidentView*> blockers;
  std::vector<const ResidentView*> evict;
  if (exclusive) {
    for (const ResidentView* view : others) {
      (evictable(view) ? evict : blockers).push_back(view);
    }
    if (!blockers.empty()) {
      plan.action = AdmissionPlan::Action::kWait;
      plan.reason = "exclusive, waiting for " + blockers.front()->id +
                    (blockers.front()->loading ? " (loading)" : " (busy)");
      return plan;
    }
    // Free memory once every other worker is gone.
    const Room room = Project(system, {}, evict);
    const bool fits = Fits(room, need, policy);
    if (!fits && !NeverFits(system, need, policy)) {
      plan.action = AdmissionPlan::Action::kWait;
      plan.reason =
          "exclusive, not enough free memory yet (held outside "
          "the router or still being released); " +
          RoomText(room);
      return plan;
    }
    if (!evict.empty()) {
      plan.action = AdmissionPlan::Action::kEvict;
      for (const ResidentView* view : evict) {
        plan.evict.push_back(view->id);
      }
      plan.reason = "exclusive; after eviction " + RoomText(room);
      return plan;
    }
    plan.action = AdmissionPlan::Action::kSpawn;
    plan.alone = !fits;
    plan.reason = std::string(fits ? "exclusive, fits; "
                                   : "exclusive, alone (estimate exceeds the "
                                     "machine); ") +
                  RoomText(room);
    return plan;
  }

  std::vector<const ResidentView*> kept;
  for (const ResidentView* view : others) {
    if (view->exclusive) {
      if (!evictable(view)) {
        plan.action = AdmissionPlan::Action::kWait;
        plan.reason = "waiting for exclusive " + view->id;
        return plan;
      }
      evict.push_back(view);
    } else {
      kept.push_back(view);
    }
  }
  Room room = Project(system, kept, evict);
  std::vector<const ResidentView*> candidates;
  for (const ResidentView* view : kept) {
    if (evictable(view)) {
      candidates.push_back(view);
    }
  }
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const ResidentView* a, const ResidentView* b) {
                     return a->last_active < b->last_active;
                   });
  std::size_t next = 0;
  while (!Fits(room, need, policy) && next < candidates.size()) {
    const ResidentView* victim = candidates[next++];
    evict.push_back(victim);
    kept.erase(std::find(kept.begin(), kept.end(), victim));
    room = Project(system, kept, evict);
  }
  const bool fits = Fits(room, need, policy);
  if (!fits && !kept.empty()) {
    // Only busy or loading residents are left: evicting the idle ones would
    // not make room yet, so keep them and wait for the busy ones.
    plan.action = AdmissionPlan::Action::kWait;
    plan.reason = "waiting for " + kept.front()->id + "; need " + Gb(need.gpu) +
                  " GB gpu, " + RoomText(room);
    return plan;
  }
  if (!fits && !NeverFits(system, need, policy)) {
    // Even with every worker of ours gone it would not fit: the memory is
    // held by other processes or a stopped worker has not returned it yet.
    // Evict nothing (that would only cost the residents) and wait; the
    // request's load timeout bounds this.
    plan.action = AdmissionPlan::Action::kWait;
    plan.reason =
        "not enough free memory yet (held outside the router or "
        "still being released); need " +
        Gb(need.gpu) + " GB gpu + " + Gb(need.host) + " GB host; " +
        RoomText(room);
    return plan;
  }
  if (!evict.empty()) {
    plan.action = AdmissionPlan::Action::kEvict;
    for (const ResidentView* view : evict) {
      plan.evict.push_back(view->id);
    }
    plan.reason = "need " + Gb(need.gpu) + " GB gpu + " + Gb(need.host) +
                  " GB host; after eviction " + RoomText(room);
    return plan;
  }
  plan.action = AdmissionPlan::Action::kSpawn;
  plan.alone = !fits;
  plan.reason = std::string(fits ? "fits"
                                 : "alone (estimate exceeds the "
                                   "machine)") +
                "; need " + Gb(need.gpu) + " GB gpu + " + Gb(need.host) +
                " GB host; " + RoomText(room);
  return plan;
}

WatchdogAction PlanWatchdog(const SystemMemory& system,
                            const std::vector<ResidentView>& residents,
                            const std::vector<std::string>& loading_order,
                            const MemoryPolicy& policy) {
  WatchdogAction action;
  const bool host_low =
      system.host_valid && system.host_available < policy.abort_host;
  const bool gpu_low =
      system.gpu_valid &&
      system.DedicatedFree() + system.SharedFree() < policy.abort_gpu;
  if (!host_low && !gpu_low) {
    return action;
  }
  action.reason = std::string(host_low ? "host" : "gpu") +
                  " memory critical: host free " + Gb(system.host_available) +
                  " GB, gpu free " + Gb(system.DedicatedFree()) + "+" +
                  Gb(system.SharedFree()) + " GB";
  if (!loading_order.empty()) {
    action.abort_load = loading_order.back();
    return action;
  }
  const ResidentView* oldest = nullptr;
  for (const auto& view : residents) {
    if (view.loading || view.busy) {
      continue;
    }
    if (oldest == nullptr || view.last_active < oldest->last_active) {
      oldest = &view;
    }
  }
  if (oldest != nullptr) {
    action.evict = oldest->id;
  }
  return action;
}

MemoryBook::MemoryBook(std::filesystem::path file) : file_(std::move(file)) {}

bool MemoryBook::Load(std::string* error) {
  const std::lock_guard<std::mutex> lock(mutex_);
  std::error_code exists_error;
  if (file_.empty() || !std::filesystem::exists(file_, exists_error)) {
    return true;
  }
  std::ifstream in(file_, std::ios::binary);
  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  json::Value root;
  try {
    root = json::parse(text);
  } catch (const std::exception& e) {
    *error = "memory file '" + file_.string() + "': " + e.what();
    return false;
  }
  const json::Value* models = root.find("models");
  if (models == nullptr || !models->is_object()) {
    return true;
  }
  for (const auto& [id, entry] : models->members()) {
    Record& record = records_[id];
    if (const auto* gpu = entry.find("estimate_gpu_gb");
        gpu != nullptr && gpu->is_number() && !record.has_estimate) {
      record.has_estimate = true;
      record.estimate.gpu = FromGb(gpu->as_double());
      record.estimate.host = FromGb(entry.member_double("estimate_host_gb"));
    }
    if (const auto* flag = entry.find("exclusive");
        flag != nullptr && flag->is_bool()) {
      record.exclusive = flag->as_bool();
    }
    if (const auto* gpu = entry.find("learned_gpu_gb");
        gpu != nullptr && gpu->is_number()) {
      record.has_learned = true;
      record.learned.gpu = FromGb(gpu->as_double());
      record.learned.host = FromGb(entry.member_double("learned_host_gb"));
      record.learned.shared = FromGb(entry.member_double("learned_shared_gb"));
      record.learned_at = entry.member_str("learned_at");
      if (const auto* complete = entry.find("learned_complete");
          complete != nullptr && complete->is_bool()) {
        record.learned_complete = complete->as_bool();
      }
    }
  }
  return true;
}

bool MemoryBook::Save(std::string* error) {
  json::Value root = json::Value::object();
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (file_.empty() || !dirty_) {
      return true;
    }
    root["version"] = 1;
    json::Value models = json::Value::object();
    const auto round = [](std::int64_t bytes) {
      return std::round(ToGb(bytes) * 100.0) / 100.0;
    };
    for (const auto& [id, record] : records_) {
      json::Value entry = json::Value::object();
      if (record.has_estimate) {
        entry["estimate_gpu_gb"] = round(record.estimate.gpu);
        entry["estimate_host_gb"] = round(record.estimate.host);
      }
      if (record.exclusive) {
        entry["exclusive"] = true;
      }
      if (record.has_learned) {
        entry["learned_gpu_gb"] = round(record.learned.gpu);
        entry["learned_host_gb"] = round(record.learned.host);
        entry["learned_shared_gb"] = round(record.learned.shared);
        entry["learned_at"] = record.learned_at;
        entry["learned_complete"] = record.learned_complete;
      }
      models[id] = std::move(entry);
    }
    root["models"] = std::move(models);
    dirty_ = false;
  }
  std::filesystem::path temp = file_;
  temp += ".tmp";
  {
    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
    out << root.dump() << "\n";
    if (!out) {
      *error = "cannot write '" + temp.string() + "'";
      return false;
    }
  }
  std::error_code rename_error;
  std::filesystem::rename(temp, file_, rename_error);
  if (rename_error) {
    *error =
        "cannot replace '" + file_.string() + "': " + rename_error.message();
    return false;
  }
  return true;
}

void MemoryBook::SetEstimate(std::string_view id, const MemoryNeed& estimate) {
  const std::lock_guard<std::mutex> lock(mutex_);
  Record& record = records_[std::string(id)];
  record.has_estimate = true;
  record.estimate = estimate;
}

void MemoryBook::SetExclusive(std::string_view id, bool exclusive) {
  const std::lock_guard<std::mutex> lock(mutex_);
  records_[std::string(id)].exclusive = exclusive;
}

bool MemoryBook::Observe(std::string_view id, const ProcessMemory& usage) {
  if (!usage.valid) {
    return false;
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  Record& record = records_[std::string(id)];
  const std::int64_t host =
      std::max(usage.host_working_set, usage.host_peak_working_set);
  const std::int64_t gpu = usage.Gpu();
  if (!record.has_learned) {
    record.has_learned = true;
    record.learned = {gpu, host, usage.gpu_shared};
    record.learned_at = NowText();
    dirty_ = true;
    return true;
  }
  bool grew = false;
  if (gpu > record.learned.gpu) {
    grew = grew || gpu - record.learned.gpu > kLearnStep;
    record.learned.gpu = gpu;
  }
  record.learned.shared = std::max(record.learned.shared, usage.gpu_shared);
  if (host > record.learned.host) {
    grew = grew || host - record.learned.host > kLearnStep;
    record.learned.host = host;
  }
  if (grew) {
    record.learned_at = NowText();
    dirty_ = true;
  }
  return grew;
}

void MemoryBook::RaiseLearned(std::string_view id, const MemoryNeed& need) {
  const std::lock_guard<std::mutex> lock(mutex_);
  Record& record = records_[std::string(id)];
  record.has_learned = true;
  record.learned.gpu = std::max(record.learned.gpu, need.gpu);
  record.learned.host = std::max(record.learned.host, need.host);
  record.learned_at = NowText();
  dirty_ = true;
}

void MemoryBook::MarkComplete(std::string_view id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = records_.find(id);
  if (it == records_.end() || !it->second.has_learned ||
      it->second.learned_complete) {
    return;
  }
  it->second.learned_complete = true;
  dirty_ = true;
}

MemoryNeed MemoryBook::Need(std::string_view id, double factor,
                            double learned_factor, bool* exclusive) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = records_.find(id);
  if (it == records_.end() ||
      (!it->second.has_estimate && !it->second.has_learned)) {
    *exclusive = true;
    return {};
  }
  const Record& record = it->second;
  *exclusive = record.exclusive;
  const auto scale = [](std::int64_t bytes, double by) {
    return static_cast<std::int64_t>(static_cast<double>(bytes) * by);
  };
  MemoryNeed need;
  if (record.has_estimate && !(record.has_learned && record.learned_complete)) {
    need.gpu = scale(record.estimate.gpu, factor);
    need.host = scale(record.estimate.host, factor);
  }
  if (record.has_learned) {
    need.shared = std::min(scale(record.learned.shared, learned_factor),
                           scale(record.learned.gpu, learned_factor));
    need.gpu = std::max(need.gpu, scale(record.learned.gpu, learned_factor));
    need.host = std::max(need.host, scale(record.learned.host, learned_factor));
  }
  return need;
}

MemoryBook::Record MemoryBook::Get(std::string_view id) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = records_.find(id);
  return it == records_.end() ? Record{} : it->second;
}

}  // namespace gufo::router
