// placement = pool: the per-request decision (pool.hpp), the partner's
// "activity" read by the probe (ParseRemoteModels, RemoteHub::View), the
// GPU slot idle check, the GPU-work model test and the preset/registry
// pieces the router feeds into the decision. Pure state, no sockets.

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "src/cli/serve/router/pool.hpp"
#include "src/cli/serve/router/preset.hpp"
#include "src/cli/serve/router/registry.hpp"
#include "src/cli/serve/router/remote.hpp"

namespace {

using gufo::router::ChoosePool;
using gufo::router::PoolChoice;
using gufo::router::PoolInput;
using gufo::router::PoolSide;

int failures = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond \
                << "\n";                                                  \
      ++failures;                                                         \
    }                                                                     \
  } while (0)

/// An available, idle, empty machine without the model.
PoolSide Cold() {
  PoolSide side;
  side.available = true;
  side.free = true;
  side.idle = true;
  side.empty = true;
  return side;
}

PoolSide Warm(bool free) {
  PoolSide side = Cold();
  side.warm = true;
  side.sticky = true;
  side.free = free;
  side.idle = free;
  return side;
}

PoolSide Unreachable() { return PoolSide{}; }

PoolChoice Choose(const PoolSide& local, const PoolSide& remote,
                  std::chrono::milliseconds waited = {},
                  bool forwarded = false) {
  PoolInput in;
  in.local = local;
  in.remote = remote;
  in.waited = waited;
  in.forwarded = forwarded;
  return ChoosePool(in).choice;
}

void TestWarmAndFree() {
  // Both warm and free: this machine.
  CHECK(Choose(Warm(true), Warm(true)) == PoolChoice::kLocal);
  // Local busy, remote warm and free: the remote.
  CHECK(Choose(Warm(false), Warm(true)) == PoolChoice::kRemote);
  // Local warm and idle while the remote is busy: here.
  CHECK(Choose(Warm(true), Warm(false)) == PoolChoice::kLocal);
  // Both warm and busy: wait.
  CHECK(Choose(Warm(false), Warm(false)) == PoolChoice::kWait);
  CHECK(Choose(Warm(false), Warm(false), std::chrono::seconds(60)) ==
        PoolChoice::kWait);
  // Warm beats cold: a warm free remote wins over a cold idle local.
  CHECK(Choose(Cold(), Warm(true)) == PoolChoice::kRemote);
}

void TestColdLoads() {
  // Nothing loaded anywhere: load here.
  CHECK(Choose(Cold(), Cold()) == PoolChoice::kLocal);
  // An empty machine before one that must evict idle models.
  PoolSide crowded = Cold();
  crowded.empty = false;
  CHECK(Choose(crowded, Cold()) == PoolChoice::kRemote);
  CHECK(Choose(Cold(), crowded) == PoolChoice::kLocal);
  // Only machines with idle loaded models: evict here first.
  CHECK(Choose(crowded, crowded) == PoolChoice::kLocal);
  // A machine with running work never evicts for the pool.
  PoolSide working = crowded;
  working.idle = false;
  CHECK(Choose(working, crowded) == PoolChoice::kRemote);
  CHECK(Choose(working, working) == PoolChoice::kWait);
  // The machine that held the model recently loads it again first.
  PoolSide recent = Cold();
  recent.sticky = true;
  CHECK(Choose(Cold(), recent) == PoolChoice::kRemote);
}

void TestHysteresis() {
  // Warm and busy on the remote: no second load here before
  // kPoolSpillAfter, then the backlog spills over.
  CHECK(Choose(Cold(), Warm(false)) == PoolChoice::kWait);
  CHECK(Choose(Cold(), Warm(false), gufo::router::kPoolSpillAfter) ==
        PoolChoice::kLocal);
  // Same the other way round.
  CHECK(Choose(Warm(false), Cold()) == PoolChoice::kWait);
  CHECK(Choose(Warm(false), Cold(), std::chrono::seconds(11)) ==
        PoolChoice::kRemote);
  // A machine that just held it (unloaded meanwhile) counts the same.
  PoolSide recent = Cold();
  recent.sticky = true;
  recent.idle = false;
  recent.free = false;
  CHECK(Choose(recent, Cold()) == PoolChoice::kWait);
  // Gone machines do not hold anything back.
  PoolSide gone = Warm(false);
  gone.available = false;
  CHECK(Choose(gone, Cold()) == PoolChoice::kRemote);
}

void TestRemoteUnreachable() {
  CHECK(Choose(Cold(), Unreachable()) == PoolChoice::kLocal);
  CHECK(Choose(Warm(true), Unreachable()) == PoolChoice::kLocal);
  // Local busy and nowhere else to go: wait for the local worker.
  CHECK(Choose(Warm(false), Unreachable()) == PoolChoice::kWait);
  CHECK(Choose(Warm(false), Unreachable(), std::chrono::seconds(60)) ==
        PoolChoice::kWait);
  // A failed local load (available = false) with a reachable remote.
  PoolSide failed = Cold();
  failed.available = false;
  CHECK(Choose(failed, Cold()) == PoolChoice::kRemote);
  CHECK(Choose(failed, Unreachable()) == PoolChoice::kWait);
}

void TestGpuBusyRule() {
  // ComfyUI work here: the remote, even when the local worker is warm.
  PoolSide local = Warm(true);
  local.gpu_busy = true;
  CHECK(Choose(local, Warm(true)) == PoolChoice::kRemote);
  CHECK(Choose(local, Cold()) == PoolChoice::kRemote);
  // GPU work on both: wait, never load next to it.
  PoolSide remote = Cold();
  remote.gpu_busy = true;
  CHECK(Choose(local, remote) == PoolChoice::kWait);
  PoolSide cold_local = Cold();
  cold_local.gpu_busy = true;
  CHECK(Choose(cold_local, remote, std::chrono::seconds(60)) ==
        PoolChoice::kWait);
  // GPU work on the remote only: here.
  CHECK(Choose(Cold(), remote) == PoolChoice::kLocal);
  // GPU work here and the partner gone (paused, restarting): serve here
  // anyway, never stand still (09.11.: 27B on the .56 beside a Comfy job).
  CHECK(Choose(local, Unreachable()) == PoolChoice::kLocal);
  CHECK(Choose(cold_local, Unreachable()) == PoolChoice::kLocal);
  PoolSide warm_idle_busy_gpu = Warm(true);
  warm_idle_busy_gpu.gpu_busy = true;
  warm_idle_busy_gpu.empty = false;
  CHECK(Choose(warm_idle_busy_gpu, Unreachable()) == PoolChoice::kLocal);
}

void TestForwardedStaysLocal() {
  // A relayed request is served here whatever the remote looks like.
  CHECK(Choose(Warm(false), Warm(true), {}, true) == PoolChoice::kLocal);
  CHECK(Choose(Unreachable(), Warm(true), {}, true) == PoolChoice::kLocal);
  // It only waits while this machine has GPU work.
  PoolSide local = Warm(true);
  local.gpu_busy = true;
  CHECK(Choose(local, Warm(true), {}, true) == PoolChoice::kWait);
}

void TestGpuWorkModel() {
  gufo::router::PresetModel model;
  model.modality = "external";
  model.argv = {"serve", "image", "--model", "x"};
  CHECK(gufo::router::IsGpuWorkModel(model));
  model.argv = {"serve", "video"};
  CHECK(gufo::router::IsGpuWorkModel(model));
  model.argv = {"serve", "music"};
  CHECK(gufo::router::IsGpuWorkModel(model));
  model.argv = {"serve", "objekt"};
  CHECK(gufo::router::IsGpuWorkModel(model));
  model.argv = {"serve", "llm", "--model", "x"};
  CHECK(!gufo::router::IsGpuWorkModel(model));
  model.argv = {"--model", "image"};
  CHECK(!gufo::router::IsGpuWorkModel(model));
  model.modality = "image";
  CHECK(gufo::router::IsGpuWorkModel(model));
  model.modality = "llm";
  CHECK(!gufo::router::IsGpuWorkModel(model));
}

void TestLoraNames() {
  using gufo::router::LoraNamesOf;
  gufo::router::PresetModel model;
  model.argv = {"serve", "image", "--lora", "anime=/l/a.safetensors"};
  CHECK(LoraNamesOf(model) == std::vector<std::string>{"anime"});
  // gufo 2.0 takes --lora once: several adapters in one ';' value
  // (12.11.: "winkel" was missing from /router/capabilities).
  model.argv = {"--lora", "Qwen2.1_Anime_consistency=D:/l/a.safetensors;"
                          "winkel=D:/l/w.safetensors"};
  CHECK(LoraNamesOf(model) ==
        (std::vector<std::string>{"Qwen2.1_Anime_consistency", "winkel"}));
  // Repeated lines (1.4.x) and empty entries; ',' keeps an entry's own
  // attributes (video LoRA steps/strength), the name stays before '='.
  model.argv = {"--lora", "a=p1;;b=p2,steps=4,strength=0.8;", "--lora",
                "c=p3", "--lora", "a=p4"};
  CHECK(LoraNamesOf(model) == (std::vector<std::string>{"a", "b", "c"}));
  model.argv = {"--lora"};  // no value: nothing
  CHECK(LoraNamesOf(model).empty());
  model.argv = {"--lora", "=p;x=y"};  // a nameless entry is skipped
  CHECK(LoraNamesOf(model) == std::vector<std::string>{"x"});
}

gufo::router::PresetModel PoolTwin() {
  gufo::router::PresetModel model;
  model.modality = "llm";
  model.id = "big";
  model.argv = {"--model", "/m/big.gguf"};
  gufo::router::RemoteSpec spec;
  std::string error;
  if (!gufo::router::ParseRemoteUrl("http://10.0.0.2:8081", false, &spec.url,
                                    &error)) {
    ++failures;
  }
  spec.gpu_slot = "s56";
  model.remote = spec;
  model.switchable = true;
  model.placement = gufo::router::Placement::kPool;
  return model;
}

void TestProbeActivity() {
  gufo::router::RemoteProbe probe;
  probe.reachable = true;
  gufo::router::ParseRemoteModels(
      R"({"object":"list","data":[)"
      R"({"id":"big","status":{"value":"loaded"},)"
      R"("activity":{"busy":true,"gpu_busy":false,"others_busy":false}},)"
      R"({"id":"img","status":{"value":"unloaded"},)"
      R"("activity":{"busy":false,"gpu_busy":true,"others_busy":true}},)"
      R"({"id":"old","loaded":true},)"
      R"({"id":"relayed","remote":{"url":"http://x"}}]})",
      &probe);
  CHECK(probe.models["big"] == "loaded");
  CHECK(probe.models["old"] == "loaded");
  CHECK(probe.models["relayed"] == "remote");
  CHECK(probe.activity.count("big") == 1 && probe.activity["big"].busy);
  CHECK(probe.activity["img"].gpu_busy && probe.activity["img"].others_busy);
  CHECK(probe.activity.count("old") == 0);  // an older router sends none

  const std::vector<gufo::router::PresetModel> presets = {PoolTwin()};
  gufo::router::RemoteHub hub(presets);
  const auto& url = presets[0].remote->url;
  // Never probed: nothing is known, the pool does not use it yet.
  CHECK(!hub.View(url, "big").served);
  (void)hub.Record(url, probe);
  auto view = hub.View(url, "big");
  CHECK(view.served && view.status == "loaded");
  CHECK(view.has_activity && view.activity.busy && !view.activity.gpu_busy);
  CHECK(view.others_loaded);  // "old" is loaded there
  CHECK(view.checked_at != gufo::router::RemoteHub::Clock::time_point{});
  CHECK(!hub.View(url, "relayed").served);
  hub.MarkUnreachable(url, "connect refused");
  CHECK(!hub.View(url, "big").served);
}

void TestGpuSlotIdle() {
  gufo::router::GpuSlots slots;
  CHECK(slots.Idle(""));
  CHECK(slots.Idle("s56"));
  auto hold = slots.Acquire("s56", "test", std::chrono::milliseconds(10), {});
  CHECK(hold != nullptr && !slots.Idle("s56"));
  hold->Release();
  CHECK(slots.Idle("s56"));
}

void TestRegistryCounts() {
  using gufo::router::Placement;
  gufo::router::Registry registry({PoolTwin()},
                                  gufo::router::RegistryOptions{});
  CHECK(registry.PlacementOf("big") == Placement::kPool);
  // Pool never routes everything remote: the remote block stays off
  // /v1/models (the partner's probe sees the model as served here).
  CHECK(!registry.RoutesRemote("big"));
  CHECK(!registry.AnyBusy());
  CHECK(registry.LastRemoteFinished("big") ==
        gufo::router::Registry::Clock::time_point{});
  registry.RequestStarted("big");
  registry.RequestStarted("big", /*remote=*/true);
  CHECK(registry.LocalRequests("big") == 1);
  CHECK(registry.RemoteRequests("big") == 1);
  CHECK(registry.AnyBusy());
  registry.RequestFinished("big", /*remote=*/true);
  CHECK(registry.RemoteRequests("big") == 0);
  CHECK(registry.LastRemoteFinished("big") !=
        gufo::router::Registry::Clock::time_point{});
  registry.RequestFinished("big");
  CHECK(registry.LocalRequests("big") == 0 && !registry.AnyBusy());
  CHECK(registry.SetPlacement("big", Placement::kLocal));
  CHECK(registry.SetPlacement("big", Placement::kPool));
  CHECK(std::string(gufo::router::PlacementName(Placement::kPool)) == "pool");
}

}  // namespace

int main() {
  TestWarmAndFree();
  TestColdLoads();
  TestHysteresis();
  TestRemoteUnreachable();
  TestGpuBusyRule();
  TestForwardedStaysLocal();
  TestGpuWorkModel();
  TestLoraNames();
  TestProbeActivity();
  TestGpuSlotIdle();
  TestRegistryCounts();
  if (failures != 0) {
    std::cerr << failures << " router pool checks failed\n";
    return 1;
  }
  std::cout << "All router pool tests passed.\n";
  return 0;
}
