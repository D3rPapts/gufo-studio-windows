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

void TestRemoteAnswerClassification() {
  using gufo::router::ClassifyPoolRemoteAnswer;
  using gufo::router::PoolRemoteAnswer;
  CHECK(ClassifyPoolRemoteAnswer(200, true, "") == PoolRemoteAnswer::kPass);
  CHECK(ClassifyPoolRemoteAnswer(400, true, "") == PoolRemoteAnswer::kPass);
  CHECK(ClassifyPoolRemoteAnswer(500, true, "") == PoolRemoteAnswer::kPass);
  CHECK(ClassifyPoolRemoteAnswer(507, true, "") ==
        PoolRemoteAnswer::kRetryMemory);
  CHECK(ClassifyPoolRemoteAnswer(503, false, "remote_unavailable detail=x") ==
        PoolRemoteAnswer::kRetryUnreachable);
  // 12.11. 21:00: the partner's 502 "worker failed to start" reached the
  // client although this machine was warm and free.
  CHECK(ClassifyPoolRemoteAnswer(502, true, "") ==
        PoolRemoteAnswer::kRetryFailed);
  CHECK(ClassifyPoolRemoteAnswer(503, true, "") ==
        PoolRemoteAnswer::kRetryFailed);
  CHECK(ClassifyPoolRemoteAnswer(504, true, "") ==
        PoolRemoteAnswer::kRetryFailed);
  CHECK(gufo::router::kPoolRemoteFailHold == std::chrono::seconds(60));
}

gufo::router::PresetModel PoolTwin();

/// One router of the pair as its partner's probe sees it: a GET
/// /v1/models answer with "big" loaded and the given activity object.
gufo::router::RemoteHub::ModelView ProbedView(
    gufo::router::RemoteHub& hub, const gufo::router::RemoteEndpoint& url,
    const std::string& activity) {
  gufo::router::RemoteProbe probe;
  probe.reachable = true;
  gufo::router::ParseRemoteModels(
      R"({"object":"list","data":[{"id":"big","status":{"value":"loaded"},)"
      R"("activity":)" +
          activity + "}]}",
      &probe);
  (void)hub.Record(url, probe);
  return hub.View(url, "big");
}

void TestWaitingIsNotBusy() {
  using gufo::router::PoolActivityBusy;
  using gufo::router::RemoteObservation;
  using gufo::router::RemotePoolSide;
  // 12.11. 17:34-18:05: router A holds a request pinned to B (X-Gufo-Pool:
  // remote after a redirect), B holds one pinned to A; both models warm,
  // nothing computes anywhere. Each reports its waiting request.
  const std::vector<gufo::router::PresetModel> presets = {PoolTwin()};
  gufo::router::RemoteHub hub(presets);
  const auto& url = presets[0].remote->url;
  const bool busy = PoolActivityBusy(/*in_flight=*/false, /*local_claims=*/0);
  CHECK(!busy);
  CHECK(PoolActivityBusy(true, 0) && PoolActivityBusy(false, 1));
  const std::string reported = std::string(R"({"busy":)") +
                               (busy ? "true" : "false") +
                               R"(,"waiting":1,"gpu_busy":false,)"
                               R"("others_busy":false})";
  RemoteObservation seen;
  seen.view = ProbedView(hub, url, reported);
  seen.reachable = true;
  CHECK(seen.view.activity.waiting == 1 && !seen.view.activity.busy);
  PoolSide pinned_away = Warm(true);
  pinned_away.available = false;  // forced remote: this side is out
  PoolInput in;
  in.local = pinned_away;
  in.remote = RemotePoolSide(seen);
  CHECK(in.remote.available && in.remote.warm && in.remote.free);
  // Each side's request goes to the other at once (before: wait 900 s).
  CHECK(ChoosePool(in).choice == PoolChoice::kRemote);
  // The old report (busy = waiting > 0) reproduces the deadlock.
  seen.view = ProbedView(
      hub, url,
      R"({"busy":true,"gpu_busy":false,"others_busy":false})");
  in.remote = RemotePoolSide(seen);
  CHECK(!in.remote.free);
  CHECK(ChoosePool(in).choice == PoolChoice::kWait);
  // Real work there still counts: busy, a relay of ours, a stale probe.
  seen.view = ProbedView(hub, url, reported);
  seen.ours = 1;
  CHECK(!RemotePoolSide(seen).free);
  seen.ours = 0;
  seen.slot_idle = false;
  CHECK(!RemotePoolSide(seen).free);
  seen.slot_idle = true;
  seen.reachable = false;
  CHECK(!RemotePoolSide(seen).available);
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

// B5 "never stuck" (Fabian 13.11.): a request this router can run waits on
// the partner at most pool-wait-seconds, then the partner drops out.
void TestFallback() {
  using gufo::router::PoolFallbackInput;
  using gufo::router::PoolFallbackReason;
  using gufo::router::PoolRemoteAnswer;
  using gufo::router::PoolRemoteHold;
  using gufo::router::WithoutPartner;
  using std::chrono::seconds;
  PoolFallbackInput in;
  in.limit = seconds(60);
  in.waited = seconds(59);
  CHECK(PoolFallbackReason(in) == nullptr);
  in.waited = seconds(60);
  CHECK(PoolFallbackReason(in) != nullptr);
  in.pinned_remote = true;
  CHECK(PoolFallbackReason(in) != nullptr);
  // Without the LoRAs here the pin holds; a relayed request is the
  // partner's to decide.
  in.local_capable = false;
  CHECK(PoolFallbackReason(in) == nullptr);
  in.local_capable = true;
  in.forwarded = true;
  CHECK(PoolFallbackReason(in) == nullptr);
  in.forwarded = false;
  in.limit = seconds(0);
  in.waited = std::chrono::milliseconds(0);
  CHECK(PoolFallbackReason(in) != nullptr);

  // GPU work here and the partner busy: "both busy" until the fallback,
  // then this machine despite its GPU work.
  PoolInput both;
  both.local = Warm(true);
  both.local.gpu_busy = true;
  both.remote = Warm(false);
  both.waited = seconds(61);
  CHECK(ChoosePool(both).choice == PoolChoice::kWait);
  CHECK(ChoosePool(WithoutPartner(both)).choice == PoolChoice::kLocal);
  // Pinned to a busy partner (local.available = false by the pin): after the
  // fallback the pin is lifted and a cold load here evicts idle models.
  PoolInput pinned;
  pinned.local = Cold();
  pinned.local.empty = false;
  pinned.remote = Warm(false);
  pinned.waited = seconds(61);
  PoolInput pinned_now = pinned;
  pinned_now.local.available = false;
  CHECK(ChoosePool(pinned_now).choice == PoolChoice::kWait);
  const auto fell = ChoosePool(WithoutPartner(pinned));
  CHECK(fell.choice == PoolChoice::kLocal);
  CHECK(std::string(fell.reason) == "load after evicting idle models");
  // Busy here too: it still waits for this machine (FIFO), never fails.
  PoolInput busy = WithoutPartner(pinned);
  busy.local = Warm(false);
  CHECK(ChoosePool(busy).choice == PoolChoice::kWait);

  // An unreachable partner is held off like a failed one.
  CHECK(PoolRemoteHold(PoolRemoteAnswer::kRetryUnreachable));
  CHECK(PoolRemoteHold(PoolRemoteAnswer::kRetryFailed));
  CHECK(!PoolRemoteHold(PoolRemoteAnswer::kRetryMemory));
  CHECK(!PoolRemoteHold(PoolRemoteAnswer::kPass));
}

void TestPriority() {
  using gufo::router::kPoolYieldMax;
  using gufo::router::ParsePriority;
  using gufo::router::PoolShouldYield;
  using gufo::router::PoolYieldInput;
  using gufo::router::PriorityOfText;
  using gufo::router::PriorityOfView;
  using gufo::router::RunsBefore;
  using gufo::json::Value;
  using std::chrono::milliseconds;
  // Text: one digit 0..9, spaces around it allowed.
  CHECK(PriorityOfText("7") == 7);
  CHECK(PriorityOfText(" 0 ") == 0);
  CHECK(PriorityOfText("9") == 9);
  CHECK(!PriorityOfText("10"));
  CHECK(!PriorityOfText("-1"));
  CHECK(!PriorityOfText(""));
  CHECK(!PriorityOfText("a"));
  CHECK(!PriorityOfText("3.5"));
  // Header wins over the JSON member; absent = 0; bad values refused.
  int priority = -1;
  std::string error;
  const Value five = gufo::json::parse("5");
  CHECK(ParsePriority("", nullptr, &priority, &error) && priority == 0);
  CHECK(ParsePriority("", &five, &priority, &error) && priority == 5);
  CHECK(ParsePriority("8", &five, &priority, &error) && priority == 8);
  CHECK(ParsePriority(" 2 ", nullptr, &priority, &error) && priority == 2);
  const Value text = gufo::json::parse("\"6\"");
  CHECK(ParsePriority("", &text, &priority, &error) && priority == 6);
  const Value none = gufo::json::parse("null");
  CHECK(ParsePriority("", &none, &priority, &error) && priority == 0);
  const Value ten = gufo::json::parse("10");
  CHECK(!ParsePriority("", &ten, &priority, &error));
  CHECK(error.find("priority") != std::string::npos);
  const Value half = gufo::json::parse("2.5");
  CHECK(!ParsePriority("", &half, &priority, &error));
  const Value negative = gufo::json::parse("-1");
  CHECK(!ParsePriority("", &negative, &priority, &error));
  const Value word = gufo::json::parse("\"hoch\"");
  CHECK(!ParsePriority("", &word, &priority, &error));
  const Value flag = gufo::json::parse("true");
  CHECK(!ParsePriority("", &flag, &priority, &error));
  CHECK(!ParsePriority("x", &five, &priority, &error));
  CHECK(error.find("X-Gufo-Priority") != std::string::npos);
  CHECK(!ParsePriority("10", nullptr, &priority, &error));
  // A view without the member (partner older than 2.0.3) is normal.
  CHECK(PriorityOfView(gufo::json::parse("{}")) == 0);
  CHECK(PriorityOfView(gufo::json::parse("{\"priority\":4}")) == 4);
  CHECK(PriorityOfView(gufo::json::parse("{\"priority\":40}")) == 9);
  CHECK(PriorityOfView(gufo::json::parse("{\"priority\":-3}")) == 0);
  // Order: the higher priority first, then the older; equal = neither.
  CHECK(RunsBefore(5, 2000, 0, 1000));
  CHECK(!RunsBefore(0, 1000, 5, 2000));
  CHECK(RunsBefore(0, 1000, 0, 1001));
  CHECK(!RunsBefore(0, 1001, 0, 1000));
  CHECK(!RunsBefore(3, 1000, 3, 1000));
  // Yield: only to a strictly higher waiting priority, never when
  // forwarded, and at most kPoolYieldMax.
  PoolYieldInput in;
  in.priority = 0;
  in.highest_waiting = 5;
  CHECK(PoolShouldYield(in));
  in.highest_waiting = 0;
  CHECK(!PoolShouldYield(in));
  in.highest_waiting = -1;
  CHECK(!PoolShouldYield(in));
  in.priority = 5;
  in.highest_waiting = 5;
  CHECK(!PoolShouldYield(in));
  in.highest_waiting = 9;
  CHECK(PoolShouldYield(in));
  in.yielded = milliseconds(kPoolYieldMax) - milliseconds(1);
  CHECK(PoolShouldYield(in));
  in.yielded = milliseconds(kPoolYieldMax);
  CHECK(!PoolShouldYield(in));
  in.yielded = milliseconds(0);
  in.forwarded = true;
  CHECK(!PoolShouldYield(in));
}

}  // namespace

void TestTopLevelMember() {
  using gufo::router::TopLevelMemberText;
  const auto text = [](std::string_view body, std::string_view key) {
    const auto found = TopLevelMemberText(body, key);
    return found ? std::string(*found) : std::string("<none>");
  };
  CHECK(text(R"({"model":"m","priority":7})", "priority") == "7");
  CHECK(text(R"({ "priority" : "3" , "x": 1})", "priority") == R"("3")");
  CHECK(text(R"({"priority":null})", "priority") == "null");
  CHECK(text(R"({"a":{"priority":9},"b":[{"priority":8}],"priority":2})",
             "priority") == "2");
  // Inside a string or nested: not a member of the body.
  CHECK(text(R"({"messages":[{"content":"the \"priority\" is high"}]})",
             "priority") == "<none>");
  CHECK(text(R"({"a":{"priority":9}})", "priority") == "<none>");
  CHECK(text(R"({"model":"a \"priority\" b"})", "priority") == "<none>");
  CHECK(text(R"({"priority":{"n":1},"z":0})", "priority") == R"({"n":1})");
  CHECK(text(R"({"priority":[1,2]})", "priority") == "[1,2]");
  CHECK(text(R"([{"priority":1}])", "priority") == "<none>");
  CHECK(text("not json", "priority") == "<none>");
  CHECK(text(R"({"priority":)", "priority") == "<none>");
  CHECK(text("", "priority") == "<none>");
}

void TestYieldClock() {
  using gufo::router::PoolYieldClock;
  using gufo::router::PoolShouldYield;
  using gufo::router::PoolYieldInput;
  using Clock = std::chrono::steady_clock;
  const Clock::time_point t0 = Clock::now();
  const auto at = [t0](int ms) { return t0 + std::chrono::milliseconds(ms); };
  PoolYieldClock clock;
  CHECK(!clock.yielding());
  CHECK(clock.YieldedFor(5, at(0)) == std::chrono::milliseconds(0));
  CHECK(clock.Begin(5, at(0)));
  CHECK(!clock.Begin(5, at(100)));  // goes on, not a new stretch
  CHECK(clock.YieldedFor(5, at(1500)) == std::chrono::milliseconds(1500));
  CHECK(clock.Total(at(1500)) == std::chrono::milliseconds(1500));
  // Both machines busy (kWait for a machine): the clock pauses, the time
  // waited for a machine never counts, but the budget toward the same
  // waiter keeps adding up over several frees.
  clock.Pause(at(1500));
  CHECK(!clock.yielding());
  CHECK(clock.YieldedFor(5, at(60000)) == std::chrono::milliseconds(1500));
  CHECK(clock.Begin(5, at(60000)));  // a new stretch toward the same waiter
  PoolYieldInput in;
  in.priority = 0;
  in.highest_waiting = 5;
  in.yielded = clock.YieldedFor(5, at(60400));
  in.total_yielded = clock.Total(at(60400));
  CHECK(PoolShouldYield(in));  // 1900 ms toward 5 so far
  in.yielded = clock.YieldedFor(5, at(60600));
  in.total_yielded = clock.Total(at(60600));
  CHECK(!PoolShouldYield(in));  // 2100 ms: the same waiter got its 2 s
  clock.Pause(at(60600));
  CHECK(clock.total == std::chrono::milliseconds(2100));
  // Freed again, the same waiter still there: no further yielding.
  in.yielded = clock.YieldedFor(5, at(90000));
  in.total_yielded = clock.Total(at(90000));
  CHECK(!PoolShouldYield(in));
  // A different, higher priority shows up: its own budget.
  CHECK(clock.YieldedFor(9, at(90000)) == std::chrono::milliseconds(0));
  in.highest_waiting = 9;
  in.yielded = clock.YieldedFor(9, at(90000));
  CHECK(PoolShouldYield(in));
  CHECK(clock.Begin(9, at(90000)));
  CHECK(clock.target == 9);
  CHECK(clock.YieldedFor(9, at(91000)) == std::chrono::milliseconds(1000));
  // Back to 5 while 9 is open: 5 keeps its spent budget, 9 keeps its own.
  CHECK(clock.Begin(5, at(91000)));  // books 1000 ms toward 9
  CHECK(clock.YieldedFor(9, at(91000)) == std::chrono::milliseconds(1000));
  CHECK(clock.YieldedFor(5, at(91500)) == std::chrono::milliseconds(2600));
  in.highest_waiting = 5;
  in.yielded = clock.YieldedFor(5, at(91500));
  in.total_yielded = clock.Total(at(91500));
  CHECK(!PoolShouldYield(in));  // alternating 5/9 never reopens 5
  clock.Pause(at(91500));
  CHECK(clock.total == std::chrono::milliseconds(3600));
  // The overall cap: with 3 x kPoolYieldMax booked, no priority gets more.
  CHECK(clock.Begin(9, at(92000)));
  clock.Pause(at(93000));  // 9: 2000 in all; total 4600
  CHECK(clock.YieldedFor(9, at(93000)) == std::chrono::milliseconds(2000));
  CHECK(clock.Begin(7, at(93000)));
  clock.Pause(at(94400));  // 7: 1400; total 6000
  CHECK(clock.total == std::chrono::milliseconds(6000));
  in.highest_waiting = 8;
  in.yielded = clock.YieldedFor(8, at(94400));
  in.total_yielded = clock.Total(at(94400));
  CHECK(in.yielded == std::chrono::milliseconds(0));
  CHECK(!PoolShouldYield(in));  // a fresh priority, but the request is done yielding
  clock.Pause(at(99000));  // idempotent
  CHECK(clock.total == std::chrono::milliseconds(6000));
}

int main() {
  TestTopLevelMember();
  TestYieldClock();
  TestWarmAndFree();
  TestColdLoads();
  TestHysteresis();
  TestRemoteUnreachable();
  TestGpuBusyRule();
  TestForwardedStaysLocal();
  TestGpuWorkModel();
  TestLoraNames();
  TestRemoteAnswerClassification();
  TestWaitingIsNotBusy();
  TestProbeActivity();
  TestGpuSlotIdle();
  TestRegistryCounts();
  TestFallback();
  TestPriority();
  if (failures != 0) {
    std::cerr << failures << " router pool checks failed\n";
    return 1;
  }
  std::cout << "All router pool tests passed.\n";
  return 0;
}
