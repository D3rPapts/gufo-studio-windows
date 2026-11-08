#include "src/cli/serve/router/registry.hpp"

#include <chrono>
#include <utility>

namespace gufo::router {

Registry::Registry(std::vector<PresetModel> presets, RegistryOptions options,
                   std::function<Clock::time_point()> now)
    : options_(options), now_(std::move(now)) {
  entries_.reserve(presets.size());
  for (auto& preset : presets) {
    Entry entry;
    entry.preset = std::move(preset);
    entry.placement = entry.preset.placement;
    entries_.push_back(std::move(entry));
  }
}

std::chrono::steady_clock::time_point Registry::Now() const {
  if (now_) {
    return now_();
  }
  return Clock::now();
}

Registry::Entry* Registry::Lookup(std::string_view model_id) {
  for (auto& entry : entries_) {
    if (entry.preset.id == model_id) {
      return &entry;
    }
  }
  return nullptr;
}

const Registry::Entry* Registry::Lookup(std::string_view model_id) const {
  for (const auto& entry : entries_) {
    if (entry.preset.id == model_id) {
      return &entry;
    }
  }
  return nullptr;
}

bool Registry::BusyEntry(const Entry& entry) {
  return entry.open_requests > 0 || entry.open_sockets > 0 ||
         entry.open_jobs > 0;
}

void Registry::Touch(Entry& entry) {
  entry.last_active = Now();
}

bool Registry::DeadlineExpired(const Entry& entry) const {
  if (options_.load_timeout == std::chrono::seconds(0) ||
      !entry.deadline_armed) {
    return false;
  }
  return Now() > entry.deadline;
}

const PresetModel* Registry::Find(std::string_view model_id) const {
  const Entry* entry = Lookup(model_id);
  return entry == nullptr ? nullptr : &entry->preset;
}

std::vector<std::pair<std::string, bool>> Registry::Listing() const {
  std::vector<std::pair<std::string, bool>> listing;
  listing.reserve(entries_.size());
  for (const auto& entry : entries_) {
    listing.emplace_back(entry.preset.id, entry.state == State::kReady);
  }
  return listing;
}

std::vector<Registry::ModelStatus> Registry::Statuses() const {
  std::vector<ModelStatus> statuses;
  statuses.reserve(entries_.size());
  for (const auto& entry : entries_) {
    statuses.push_back(
        {&entry.preset, entry.state == State::kReady     ? "loaded"
                        : entry.state == State::kLoading ? "loading"
                                                         : "unloaded"});
  }
  return statuses;
}

LoadDecision Registry::RequestLoad(std::string_view model_id) {
  Entry* entry = Lookup(model_id);
  if (entry == nullptr) {
    return LoadDecision::kQueue;
  }
  switch (entry->state) {
    case State::kReady:
      return LoadDecision::kAwaitLoad;
    case State::kLoading:
      return DeadlineExpired(*entry) ? LoadDecision::kLoadTimeout
                                     : LoadDecision::kAwaitLoad;
    case State::kUnloaded:
      break;
  }
  if (entry->deadline_armed && DeadlineExpired(*entry)) {
    return LoadDecision::kLoadTimeout;
  }
  if (!entry->deadline_armed &&
      options_.load_timeout != std::chrono::seconds(0)) {
    entry->deadline_armed = true;
    entry->deadline = Now() + options_.load_timeout;
  }
  if (options_.max_active == 0 || ActiveCount() < options_.max_active) {
    return LoadDecision::kSpawn;
  }
  entry->queued = true;
  return LoadDecision::kQueue;
}

std::string Registry::EvictionCandidate() {
  Entry* best = nullptr;
  for (auto& entry : entries_) {
    if (entry.state == State::kUnloaded || BusyEntry(entry)) {
      continue;
    }
    if (best == nullptr || entry.last_active < best->last_active) {
      best = &entry;
    }
  }
  return best == nullptr ? std::string() : best->preset.id;
}

bool Registry::HasQueued() const {
  for (const auto& entry : entries_) {
    if (entry.queued) {
      return true;
    }
  }
  return false;
}

std::chrono::seconds Registry::IdleTimeout(const Entry& entry) const {
  if (entry.preset.hints.idle_seconds >= 0) {
    return std::chrono::seconds(entry.preset.hints.idle_seconds);
  }
  return options_.idle_timeout;
}

std::vector<Registry::ResidentEntry> Registry::Residents() const {
  std::vector<ResidentEntry> residents;
  for (const auto& entry : entries_) {
    if (entry.state == State::kUnloaded) {
      continue;
    }
    residents.push_back({entry.preset.id, entry.state == State::kLoading,
                         BusyEntry(entry), entry.last_active,
                         IdleTimeout(entry)});
  }
  return residents;
}

std::vector<std::string> Registry::DueIdleUnloads() {
  std::vector<std::string> due;
  const Clock::time_point now = Now();
  for (const auto& entry : entries_) {
    if (entry.state != State::kReady || BusyEntry(entry)) {
      continue;
    }
    const std::chrono::seconds timeout = IdleTimeout(entry);
    if (timeout == std::chrono::seconds(0)) {
      continue;
    }
    if (now - entry.last_active >= timeout) {
      due.push_back(entry.preset.id);
    }
  }
  return due;
}

std::vector<std::string> Registry::DueLoadTimeouts() {
  std::vector<std::string> due;
  if (options_.load_timeout == std::chrono::seconds(0)) {
    return due;
  }
  for (const auto& entry : entries_) {
    if (entry.state != State::kReady && DeadlineExpired(entry)) {
      due.push_back(entry.preset.id);
    }
  }
  return due;
}

void Registry::MarkLoading(std::string_view model_id) {
  Entry* entry = Lookup(model_id);
  if (entry == nullptr || entry->state == State::kReady) {
    return;
  }
  entry->state = State::kLoading;
  entry->queued = false;
  Touch(*entry);
}

void Registry::MarkReady(std::string_view model_id) {
  Entry* entry = Lookup(model_id);
  if (entry == nullptr) {
    return;
  }
  entry->state = State::kReady;
  entry->deadline_armed = false;
  entry->queued = false;
  Touch(*entry);
}

void Registry::MarkUnloaded(std::string_view model_id) {
  Entry* entry = Lookup(model_id);
  if (entry == nullptr) {
    return;
  }
  entry->state = State::kUnloaded;
  entry->deadline_armed = false;
  entry->queued = false;
  Touch(*entry);
}

Placement Registry::PlacementOf(std::string_view model_id) const {
  const Entry* entry = Lookup(model_id);
  return entry == nullptr ? Placement::kLocal : entry->placement;
}

bool Registry::RoutesRemote(std::string_view model_id) const {
  const Entry* entry = Lookup(model_id);
  return entry != nullptr && entry->preset.remote.has_value() &&
         entry->placement == Placement::kRemote;
}

bool Registry::SetPlacement(std::string_view model_id, Placement placement) {
  Entry* entry = Lookup(model_id);
  if (entry == nullptr || !entry->preset.switchable) {
    return false;
  }
  entry->placement = placement;
  return true;
}

void Registry::RequestStarted(std::string_view model_id, bool remote) {
  Entry* entry = Lookup(model_id);
  if (entry == nullptr) {
    return;
  }
  if (remote) {
    // A relay leaves the local idle timer alone.
    ++entry->open_remote_requests;
    return;
  }
  ++entry->open_requests;
  Touch(*entry);
}

void Registry::RequestFinished(std::string_view model_id, bool remote) {
  Entry* entry = Lookup(model_id);
  if (entry == nullptr) {
    return;
  }
  if (remote) {
    if (entry->open_remote_requests > 0) {
      --entry->open_remote_requests;
    }
    entry->remote_finished = Now();
    return;
  }
  if (entry->open_requests > 0) {
    --entry->open_requests;
  }
  Touch(*entry);
}

void Registry::WebSocketOpened(std::string_view model_id) {
  Entry* entry = Lookup(model_id);
  if (entry == nullptr) {
    return;
  }
  ++entry->open_sockets;
  Touch(*entry);
}

void Registry::WebSocketClosed(std::string_view model_id) {
  Entry* entry = Lookup(model_id);
  if (entry == nullptr) {
    return;
  }
  if (entry->open_sockets > 0) {
    --entry->open_sockets;
  }
  Touch(*entry);
}

void Registry::TrackVideoJob(std::string_view job_id,
                             std::string_view model_id, bool remote) {
  Entry* entry = Lookup(model_id);
  if (entry == nullptr) {
    return;
  }
  const std::string key(job_id);
  const auto previous = job_owners_.find(key);
  if (previous != job_owners_.end()) {
    if (Entry* old_owner = Lookup(previous->second); old_owner != nullptr) {
      std::size_t& count = remote_jobs_.count(key) > 0
                               ? old_owner->open_remote_jobs
                               : old_owner->open_jobs;
      if (count > 0) {
        --count;
      }
    }
    job_owners_.erase(previous);
  }
  job_owners_.emplace(key, std::string(model_id));
  if (remote) {
    remote_jobs_.insert(key);
    ++entry->open_remote_jobs;
    return;
  }
  remote_jobs_.erase(key);
  ++entry->open_jobs;
  Touch(*entry);
}

void Registry::CompleteVideoJob(std::string_view job_id) {
  const auto it = job_owners_.find(std::string(job_id));
  if (it == job_owners_.end()) {
    return;
  }
  Entry* entry = Lookup(it->second);
  // insert_or_assign: async music ids are numbered per worker (music_1 ...),
  // so a finished id can come back for another model or a restarted worker;
  // the newest finished owner answers.
  if (finished_owners_.insert_or_assign(it->first, it->second).second) {
    finished_order_.push_back(it->first);
    constexpr std::size_t kFinishedJobs = 1024;
    if (finished_order_.size() > kFinishedJobs) {
      if (!job_owners_.contains(finished_order_.front())) {
        remote_jobs_.erase(finished_order_.front());
      }
      finished_owners_.erase(finished_order_.front());
      finished_order_.erase(finished_order_.begin());
    }
  }
  const bool remote = remote_jobs_.count(it->first) > 0;
  job_owners_.erase(it);
  if (entry == nullptr) {
    return;
  }
  std::size_t& count = remote ? entry->open_remote_jobs : entry->open_jobs;
  if (count > 0) {
    --count;
  }
  if (!remote) {
    Touch(*entry);
  }
}

std::vector<std::pair<std::string, std::string>> Registry::OpenJobs() const {
  std::vector<std::pair<std::string, std::string>> jobs;
  for (const auto& [job, owner] : job_owners_) {
    if (remote_jobs_.count(job) == 0) {
      jobs.emplace_back(job, owner);
    }
  }
  return jobs;
}

bool Registry::JobRemote(std::string_view job_id) const {
  return remote_jobs_.count(std::string(job_id)) > 0;
}

std::string Registry::JobOwner(std::string_view job_id) const {
  const std::string key(job_id);
  if (const auto it = job_owners_.find(key); it != job_owners_.end()) {
    return it->second;
  }
  const auto it = finished_owners_.find(key);
  return it == finished_owners_.end() ? std::string() : it->second;
}

bool Registry::JobOpen(std::string_view job_id) const {
  return job_owners_.find(std::string(job_id)) != job_owners_.end();
}

bool Registry::Busy(std::string_view model_id) const {
  const Entry* entry = Lookup(model_id);
  return entry != nullptr && BusyEntry(*entry);
}

bool Registry::RemoteBusy(std::string_view model_id) const {
  const Entry* entry = Lookup(model_id);
  return entry != nullptr &&
         (entry->open_remote_requests > 0 || entry->open_remote_jobs > 0);
}

std::size_t Registry::LocalRequests(std::string_view model_id) const {
  const Entry* entry = Lookup(model_id);
  return entry == nullptr ? 0 : entry->open_requests;
}

std::size_t Registry::RemoteRequests(std::string_view model_id) const {
  const Entry* entry = Lookup(model_id);
  return entry == nullptr
             ? 0
             : entry->open_remote_requests + entry->open_remote_jobs;
}

Registry::Clock::time_point Registry::LastRemoteFinished(
    std::string_view model_id) const {
  const Entry* entry = Lookup(model_id);
  return entry == nullptr ? Clock::time_point{} : entry->remote_finished;
}

bool Registry::AnyBusy() const {
  for (const auto& entry : entries_) {
    if (BusyEntry(entry)) {
      return true;
    }
  }
  return false;
}

std::size_t Registry::ActiveCount() const {
  std::size_t count = 0;
  for (const auto& entry : entries_) {
    if (entry.state != State::kUnloaded) {
      ++count;
    }
  }
  return count;
}

}  // namespace gufo::router
