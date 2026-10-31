// Admission, eviction and watchdog rules of the memory-aware router with a
// fake memory picture; MemoryBook persistence; router-* preset keys; the
// per-model idle override.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/cli/serve/router/memory.hpp"
#include "src/cli/serve/router/preset.hpp"
#include "src/cli/serve/router/registry.hpp"

namespace {

using gufo::router::AdmissionPlan;
using gufo::router::kGiB;
using gufo::router::MemoryBook;
using gufo::router::MemoryNeed;
using gufo::router::MemoryPolicy;
using gufo::router::PlanAdmission;
using gufo::router::PlanWatchdog;
using gufo::router::ProcessMemory;
using gufo::router::ResidentView;
using gufo::router::SystemMemory;
using Action = AdmissionPlan::Action;
using Clock = std::chrono::steady_clock;

int failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond \
                << "\n";                                                   \
      ++failures;                                                          \
    }                                                                      \
  } while (0)

std::int64_t Gb(double gb) {
  return static_cast<std::int64_t>(gb * static_cast<double>(kGiB));
}

/// 32 GB dedicated, 15.5 GB shared, 31.6 GB host; `ded_used` etc. in GB.
SystemMemory Machine(double ded_used, double shared_used, double host_free) {
  SystemMemory memory;
  memory.gpu_valid = true;
  memory.host_valid = true;
  memory.gpu_dedicated_total = Gb(32);
  memory.gpu_dedicated_used = Gb(ded_used);
  memory.gpu_shared_total = Gb(15.5);
  memory.gpu_shared_used = Gb(shared_used);
  memory.host_total = Gb(31.6);
  memory.host_available = Gb(host_free);
  return memory;
}

ResidentView Resident(const std::string& id, double need_gpu, double cur_gpu,
                      int last_active_s, bool busy = false,
                      bool loading = false) {
  ResidentView view;
  view.id = id;
  view.busy = busy;
  view.loading = loading;
  view.need = {Gb(need_gpu), Gb(2)};
  view.current.valid = true;
  view.current.gpu_dedicated = Gb(cur_gpu);
  view.current.host_working_set = Gb(2);
  view.last_active = Clock::time_point{} + std::chrono::seconds(last_active_s);
  return view;
}

MemoryNeed Need(double gpu, double host = 1) {
  return {Gb(gpu), Gb(host)};
}

void TestEmptyFits() {
  const auto plan = PlanAdmission(Machine(2, 1, 25), {}, "a", Need(20), false,
                                  MemoryPolicy{});
  CHECK(plan.action == Action::kSpawn);
  CHECK(!plan.alone);
}

void TestCoResidency() {
  // 27B holds 20 GB (desktop 2 GB): a 5 GB TTS still fits beside it.
  const std::vector<ResidentView> residents = {Resident("27b", 20, 20, 10)};
  const auto plan = PlanAdmission(Machine(22, 1, 20), residents, "tts", Need(5),
                                  false, MemoryPolicy{});
  CHECK(plan.action == Action::kSpawn);
  CHECK(plan.evict.empty());
}

void TestEvictsForLargeModel() {
  const std::vector<ResidentView> residents = {Resident("27b", 20, 20, 10)};
  const auto plan = PlanAdmission(Machine(22, 1, 20), residents, "image",
                                  Need(34.5, 2), false, MemoryPolicy{});
  CHECK(plan.action == Action::kEvict);
  CHECK(plan.evict.size() == 1 && plan.evict[0] == "27b");
}

void TestLruOrder() {
  // Two idle residents; one eviction is enough, the older one goes.
  const std::vector<ResidentView> residents = {Resident("new", 8, 8, 50),
                                               Resident("old", 8, 8, 5)};
  const auto plan = PlanAdmission(Machine(18, 15, 20), residents, "c", Need(12),
                                  false, MemoryPolicy{});
  CHECK(plan.action == Action::kEvict);
  CHECK(plan.evict.size() == 1 && plan.evict[0] == "old");
}

void TestBusyNeverEvicted() {
  const std::vector<ResidentView> residents = {
      Resident("27b", 20, 20, 10, /*busy=*/true)};
  const auto plan = PlanAdmission(Machine(22, 1, 20), residents, "image",
                                  Need(34.5), false, MemoryPolicy{});
  CHECK(plan.action == Action::kWait);
  CHECK(plan.evict.empty());
}

void TestIdleKeptWhileBusyBlocks() {
  // Even after evicting the idle one it does not fit because of the busy
  // one: keep the idle one and wait.
  const std::vector<ResidentView> residents = {
      Resident("busy", 20, 20, 10, true), Resident("idle", 5, 5, 1)};
  const auto plan = PlanAdmission(Machine(27, 10, 20), residents, "big",
                                  Need(20), false, MemoryPolicy{});
  CHECK(plan.action == Action::kWait);
  CHECK(plan.evict.empty());
}

void TestLoadingOwesMemory() {
  // A loading worker has 2 of its 20 GB so far: the 18 GB it still needs
  // are not free for a second load.
  const std::vector<ResidentView> residents = {
      Resident("loading", 20, 2, 10, false, /*loading=*/true)};
  const auto plan = PlanAdmission(Machine(4, 15, 25), residents, "tts", Need(8),
                                  false, MemoryPolicy{});
  CHECK(plan.action == Action::kWait);
  const auto small = PlanAdmission(Machine(4, 15, 25), residents, "tiny",
                                   Need(1), false, MemoryPolicy{});
  CHECK(small.action == Action::kSpawn);
}

void TestUnmeasuredResidentCountsFullNeed() {
  ResidentView unknown = Resident("x", 20, 0, 10);
  unknown.current.valid = false;
  // Live numbers already show the 20 GB, but without a sample the planner
  // counts the full need again: conservative.
  const auto plan = PlanAdmission(Machine(22, 1, 20), {unknown}, "tts", Need(5),
                                  false, MemoryPolicy{});
  CHECK(plan.action == Action::kEvict);
}

void TestExclusiveCandidate() {
  const MemoryPolicy policy;
  const std::vector<ResidentView> idle = {Resident("a", 5, 5, 1),
                                          Resident("b", 5, 5, 2)};
  auto plan =
      PlanAdmission(Machine(12, 1, 20), idle, "h3", Need(45), true, policy);
  CHECK(plan.action == Action::kEvict);
  CHECK(plan.evict.size() == 2);
  const std::vector<ResidentView> one_busy = {Resident("a", 5, 5, 1),
                                              Resident("b", 5, 5, 2, true)};
  plan =
      PlanAdmission(Machine(12, 1, 20), one_busy, "h3", Need(45), true, policy);
  CHECK(plan.action == Action::kWait);
  plan = PlanAdmission(Machine(2, 1, 25), {}, "h3", Need(45), true, policy);
  CHECK(plan.action == Action::kSpawn);
  CHECK(plan.alone);
}

void TestExclusiveResident() {
  ResidentView h3 = Resident("h3", 45, 30, 1);
  h3.exclusive = true;
  auto plan = PlanAdmission(Machine(32, 13, 10), {h3}, "tts", Need(1), false,
                            MemoryPolicy{});
  CHECK(plan.action == Action::kEvict);
  CHECK(plan.evict.size() == 1 && plan.evict[0] == "h3");
  h3.busy = true;
  plan = PlanAdmission(Machine(32, 13, 10), {h3}, "tts", Need(1), false,
                       MemoryPolicy{});
  CHECK(plan.action == Action::kWait);
}

void TestWaitWhenMemoryHeldOutside() {
  // Another process (or a stopped worker that has not returned its memory
  // yet) holds a lot and nothing of ours is resident: wait, do not
  // over-commit.
  auto plan = PlanAdmission(Machine(20, 10, 10), {}, "image", Need(30), false,
                            MemoryPolicy{});
  CHECK(plan.action == Action::kWait);
  plan = PlanAdmission(Machine(20, 10, 10), {}, "h3", Need(30), true,
                       MemoryPolicy{});
  CHECK(plan.action == Action::kWait);
  // A need larger than the whole machine would wait forever: start alone.
  plan = PlanAdmission(Machine(20, 10, 10), {}, "huge", Need(50), false,
                       MemoryPolicy{});
  CHECK(plan.action == Action::kSpawn);
  CHECK(plan.alone);
}

void TestHostLimit() {
  // GPU has room but host RAM does not: evict the idle resident.
  ResidentView r = Resident("r", 4, 4, 1);
  r.current.host_working_set = Gb(6);
  r.need.host = Gb(6);
  const auto plan = PlanAdmission(Machine(6, 1, 8), {r}, "c", Need(4, 5), false,
                                  MemoryPolicy{});
  CHECK(plan.action == Action::kEvict);
}

void TestSpillCountsAgainstHost() {
  // Dedicated is nearly full; the 10 GB that spill into shared also come out
  // of host RAM, which then is too small.
  const auto plan = PlanAdmission(Machine(29, 0, 11), {}, "big", Need(10, 1),
                                  false, MemoryPolicy{});
  CHECK(plan.action == Action::kWait);
  const auto fits = PlanAdmission(Machine(29, 0, 20), {}, "big", Need(10, 1),
                                  false, MemoryPolicy{});
  CHECK(fits.action == Action::kSpawn);
  CHECK(!fits.alone);
}

void TestOwnSharedIsNoSpill() {
  // Measured 27B: 31.6 GB dedicated + 3.5 GB shared (pinned prompt cache,
  // part of its 6.5 GB working set). Only the dedicated part that does not
  // find room spills into host RAM.
  MemoryNeed need = Need(38.6, 7.1);
  need.shared = Gb(3.85);
  auto plan = PlanAdmission(Machine(1, 0.5, 18), {}, "27b", need, false,
                            MemoryPolicy{});
  CHECK(plan.action == Action::kSpawn && !plan.alone);
  // Without the split the whole excess would count as spill and not fit.
  need.shared = 0;
  plan = PlanAdmission(Machine(1, 0.5, 18), {}, "27b", need, false,
                       MemoryPolicy{});
  CHECK(plan.action == Action::kWait);
}

void TestReload27bAfterIdleUnload() {
  // 30.10.: the 27B (complete learned peak 35.12 GB gpu, 6.22 GB of it shared,
  // 8.29 GB host) was idle-unloaded; the reload saw 30.0 GB dedicated + 15.1
  // GB shared and 14.6 GB host free and waited forever, because the factored
  // dedicated part (31.8 GB) "spilled" 1.8 GB on top of host x 1.1. A real
  // load at that free memory left 6.5 GB host free.
  MemoryBook book({});
  ProcessMemory usage;
  usage.valid = true;
  usage.gpu_dedicated = Gb(35.12 - 6.22);
  usage.gpu_shared = Gb(6.22);
  usage.host_working_set = Gb(8.29);
  book.Observe("qwen3.8-27b", usage);
  bool exclusive = false;
  const MemoryPolicy policy;
  // Before a complete run the factored split stays (no measured dedicated).
  CHECK(book.Need("qwen3.8-27b", policy.factor, policy.learned_factor,
                  &exclusive)
            .dedicated == 0);
  book.MarkComplete("qwen3.8-27b");
  const MemoryNeed need = book.Need("qwen3.8-27b", policy.factor,
                                    policy.learned_factor, &exclusive);
  CHECK(need.dedicated == Gb(35.12 - 6.22));
  const SystemMemory reload = Machine(2.0, 0.4, 14.6);
  auto plan = PlanAdmission(reload, {}, "qwen3.8-27b", need, false, policy);
  CHECK(plan.action == Action::kSpawn && !plan.alone);
  // The old rule (factored dedicated part spills) waits here.
  MemoryNeed old = need;
  old.dedicated = 0;
  plan = PlanAdmission(reload, {}, "qwen3.8-27b", old, false, policy);
  CHECK(plan.action == Action::kWait);
  // Still safe: too little host RAM (12 GB - 4 reserve < 9.1) waits ...
  plan = PlanAdmission(Machine(2.0, 0.4, 12), {}, "qwen3.8-27b", need, false,
                       policy);
  CHECK(plan.action == Action::kWait);
  // ... and when dedicated really is short (28 GB free), the measured part
  // that does not fit (0.9 GB) still counts as host spill: 13.5 GB host
  // would do without it, with it the load waits.
  plan = PlanAdmission(Machine(4.0, 0.4, 13.5), {}, "qwen3.8-27b", need, false,
                       policy);
  CHECK(plan.action == Action::kWait);
  plan = PlanAdmission(Machine(4.0, 0.4, 14.6), {}, "qwen3.8-27b", need, false,
                       policy);
  CHECK(plan.action == Action::kSpawn && !plan.alone);
}

void TestWatchdog() {
  const MemoryPolicy policy;
  const std::vector<ResidentView> residents = {
      Resident("idle", 5, 5, 1), Resident("l1", 5, 5, 2, false, true),
      Resident("l2", 5, 5, 3, false, true)};
  auto action =
      PlanWatchdog(Machine(10, 1, 1.0), residents, {"l1", "l2"}, policy);
  CHECK(action.abort_load == "l2");
  action = PlanWatchdog(Machine(10, 1, 1.0), {Resident("idle", 5, 5, 1)}, {},
                        policy);
  CHECK(action.abort_load.empty() && action.evict == "idle");
  action = PlanWatchdog(Machine(10, 1, 1.0), {Resident("busy", 5, 5, 1, true)},
                        {}, policy);
  CHECK(action.evict.empty() && action.abort_load.empty() &&
        !action.reason.empty());
  action = PlanWatchdog(Machine(31.8, 15.3, 8), residents, {"l1"}, policy);
  CHECK(action.abort_load == "l1");
  action = PlanWatchdog(Machine(10, 1, 8), residents, {"l1"}, policy);
  CHECK(action.reason.empty() && action.abort_load.empty());
}

/// Machine() with a Windows commit reading: `commit_free` of a 75.6 GB limit.
SystemMemory WithCommit(SystemMemory memory, double commit_free) {
  memory.commit_valid = true;
  memory.commit_limit = Gb(75.6);
  memory.commit_available = Gb(commit_free);
  return memory;
}

void TestCommitLimit() {
  const MemoryPolicy policy;
  // RAM and GPU fit, but GPU + host (20 + 4 GB) exceed free commit minus
  // the 3 GB reserve: wait instead of a bad_alloc.
  auto plan = PlanAdmission(WithCommit(Machine(2, 0.2, 20), 25), {}, "m",
                            Need(20, 4), false, policy);
  CHECK(plan.action == Action::kWait);
  CHECK(plan.reason.find("commit") != std::string::npos);
  plan = PlanAdmission(WithCommit(Machine(2, 0.2, 20), 28), {}, "m",
                       Need(20, 4), false, policy);
  CHECK(plan.action == Action::kSpawn && !plan.alone);
  // The model's own shared buffers are in its GPU and host need: counted once.
  MemoryNeed with_shared = Need(20, 4);
  with_shared.shared = Gb(3);
  CHECK(gufo::router::CommitNeed(with_shared) == Gb(21));
  plan = PlanAdmission(WithCommit(Machine(2, 0.2, 20), 24.5), {}, "m",
                       with_shared, false, policy);
  CHECK(plan.action == Action::kSpawn);
  // An idle resident's private bytes come back when it is evicted.
  ResidentView idle = Resident("idle", 10, 10, 1);
  idle.current.commit = Gb(12);
  plan = PlanAdmission(WithCommit(Machine(12, 0.2, 18), 20), {idle}, "m",
                       Need(20, 4), false, policy);
  CHECK(plan.action == Action::kEvict && plan.evict.size() == 1 &&
        plan.evict[0] == "idle");
  // A loading resident still owes commit up to its need (10 + 2 GB, holds 5).
  ResidentView loading = Resident("l", 10, 3, 1, false, true);
  loading.current.commit = Gb(5);
  plan = PlanAdmission(WithCommit(Machine(3, 0.2, 25), 19.5), {loading}, "m",
                       Need(8, 2), false, policy);
  CHECK(plan.action == Action::kWait);  // 19.5 - 7 owed - 3 < 10
  plan = PlanAdmission(WithCommit(Machine(3, 0.2, 25), 21), {loading}, "m",
                       Need(8, 2), false, policy);
  CHECK(plan.action == Action::kSpawn);
  // A custom reserve.
  MemoryPolicy loose = policy;
  loose.reserve_commit = Gb(0.5);
  plan = PlanAdmission(WithCommit(Machine(2, 0.2, 20), 25), {}, "m",
                       Need(20, 4), false, loose);
  CHECK(plan.action == Action::kSpawn);
  // More than the whole commit limit: starts alone, the watchdog guards it.
  plan = PlanAdmission(WithCommit(Machine(2, 0.2, 30), 70), {}, "m",
                       Need(47, 30), false, policy);
  CHECK(plan.action == Action::kSpawn && plan.alone);
  // Without a commit reading (POSIX) nothing changes.
  plan = PlanAdmission(Machine(2, 0.2, 20), {}, "m", Need(20, 4), false,
                       policy);
  CHECK(plan.action == Action::kSpawn);
}

void TestCommitWatchdog() {
  const MemoryPolicy policy;
  const std::vector<ResidentView> residents = {
      Resident("idle", 5, 5, 1), Resident("l1", 5, 5, 2, false, true)};
  auto action = PlanWatchdog(WithCommit(Machine(10, 1, 8), 0.8), residents,
                             {"l1"}, policy);
  CHECK(action.abort_load == "l1");
  CHECK(action.reason.find("commit memory critical") != std::string::npos);
  action = PlanWatchdog(WithCommit(Machine(10, 1, 8), 0.8),
                        {Resident("idle", 5, 5, 1)}, {}, policy);
  CHECK(action.evict == "idle");
  action = PlanWatchdog(WithCommit(Machine(10, 1, 8), 1.5), residents,
                        {"l1"}, policy);
  CHECK(action.reason.empty() && action.abort_load.empty());
}

void TestFactors() {
  // Estimates get the large factor, measured peaks the small one.
  MemoryBook book({});
  book.SetEstimate("a", {Gb(20), Gb(2)});
  ProcessMemory usage;
  usage.valid = true;
  usage.gpu_dedicated = Gb(30);
  usage.host_working_set = Gb(1);
  book.Observe("a", usage);
  bool exclusive = false;
  const MemoryNeed need = book.Need("a", 1.15, 1.05, &exclusive);
  CHECK(need.gpu ==
        static_cast<std::int64_t>(static_cast<double>(Gb(30)) * 1.05));
  CHECK(need.host ==
        static_cast<std::int64_t>(static_cast<double>(Gb(2)) * 1.15));
  // After a complete run the measured peak replaces the estimate.
  book.MarkComplete("a");
  const MemoryNeed learned = book.Need("a", 1.15, 1.05, &exclusive);
  CHECK(learned.host ==
        static_cast<std::int64_t>(static_cast<double>(Gb(1)) * 1.05));
}

void TestMemoryBook() {
  const auto file =
      std::filesystem::temp_directory_path() /
      ("router_memory_test_" + std::to_string(std::rand()) + ".json");
  {
    MemoryBook book(file);
    std::string error;
    CHECK(book.Load(&error));
    bool exclusive = false;
    book.Need("unknown", 1.15, 1.05, &exclusive);
    CHECK(exclusive);
    book.SetEstimate("a", {Gb(10), Gb(2)});
    MemoryNeed need = book.Need("a", 1.0, 1.0, &exclusive);
    CHECK(!exclusive && need.gpu == Gb(10));
    ProcessMemory usage;
    usage.valid = true;
    usage.gpu_dedicated = Gb(12);
    usage.gpu_shared = Gb(1);
    usage.host_working_set = Gb(1);
    usage.host_peak_working_set = Gb(3);
    CHECK(book.Observe("a", usage));
    need = book.Need("a", 1.0, 1.0, &exclusive);
    CHECK(need.gpu == Gb(13) && need.host == Gb(3));
    usage.gpu_dedicated = Gb(12.1);
    CHECK(!book.Observe("a", usage));  // below the 256 MiB step
    book.SetExclusive("h3", true);
    book.SetEstimate("h3", {Gb(45), Gb(13)});
    CHECK(book.Save(&error));
  }
  {
    MemoryBook book(file);
    std::string error;
    CHECK(book.Load(&error));
    bool exclusive = false;
    const MemoryNeed need = book.Need("a", 1.0, 1.0, &exclusive);
    CHECK(need.gpu > Gb(13.05) && need.gpu < Gb(13.15));
    book.Need("h3", 1.0, 1.0, &exclusive);
    CHECK(exclusive);
  }
  std::error_code ignored;
  std::filesystem::remove(file, ignored);
}

void TestPresetHintsAndIdleOverride() {
  const auto dir = std::filesystem::temp_directory_path();
  const auto file = dir / "router_memory_test_preset.ini";
  const auto exe = dir / "router_memory_test_fake.exe";
  {
    std::ofstream(exe) << "x";
  }
  {
    std::ofstream out(file);
    out << "[external/a]\ncommand = " << exe.string()
        << "\nmodel = m\nrouter-gpu-gb = 19.5\nrouter-host-gb = 3\n"
           "router-exclusive = true\nrouter-idle-seconds = 60\n";
  }
  std::vector<gufo::router::PresetModel> models;
  std::string error;
  CHECK(gufo::router::LoadPreset(file, {}, &models, &error));
  CHECK(models.size() == 1);
  if (models.size() == 1) {
    const auto& model = models[0];
    CHECK(model.hints.gpu_gb == 19.5 && model.hints.host_gb == 3.0);
    CHECK(model.hints.exclusive == 1 && model.hints.idle_seconds == 60);
    for (const auto& arg : model.argv) {
      CHECK(arg.find("router") == std::string::npos);
    }
    // Idle override: 60 s for this model although the router default is 300.
    auto now = std::make_shared<Clock::time_point>();
    gufo::router::RegistryOptions options;
    options.max_active = 0;
    options.idle_timeout = std::chrono::seconds(300);
    gufo::router::Registry registry(models, options, [now] { return *now; });
    registry.MarkReady("a");
    *now += std::chrono::seconds(59);
    CHECK(registry.DueIdleUnloads().empty());
    *now += std::chrono::seconds(2);
    CHECK(registry.DueIdleUnloads().size() == 1);
    const auto residents = registry.Residents();
    CHECK(residents.size() == 1 &&
          residents[0].idle_timeout == std::chrono::seconds(60));
  }
  {
    std::ofstream out(file);
    out << "[external/a]\ncommand = " << exe.string() << "\nrouter-bogus = 1\n";
  }
  CHECK(!gufo::router::LoadPreset(file, {}, &models, &error));
  CHECK(error.find("router-bogus") != std::string::npos);
  std::error_code ignored;
  std::filesystem::remove(file, ignored);
  std::filesystem::remove(exe, ignored);
}

}  // namespace

int main() {
  TestEmptyFits();
  TestCoResidency();
  TestEvictsForLargeModel();
  TestLruOrder();
  TestBusyNeverEvicted();
  TestIdleKeptWhileBusyBlocks();
  TestLoadingOwesMemory();
  TestUnmeasuredResidentCountsFullNeed();
  TestExclusiveCandidate();
  TestExclusiveResident();
  TestWaitWhenMemoryHeldOutside();
  TestHostLimit();
  TestSpillCountsAgainstHost();
  TestOwnSharedIsNoSpill();
  TestReload27bAfterIdleUnload();
  TestWatchdog();
  TestCommitLimit();
  TestCommitWatchdog();
  TestFactors();
  TestMemoryBook();
  TestPresetHintsAndIdleOverride();
  if (failures != 0) {
    std::cerr << failures << " router memory checks failed.\n";
    return 1;
  }
  std::cout << "All router memory tests passed.\n";
  return 0;
}
