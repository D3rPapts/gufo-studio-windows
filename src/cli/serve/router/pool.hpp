#ifndef GUFO_CLI_SERVE_ROUTER_POOL_HPP_
#define GUFO_CLI_SERVE_ROUTER_POOL_HPP_

// `placement = pool` for switchable models (a local section with a
// `[remote/<id>]` twin): every new request picks this machine or the remote
// router, so both machines serve the model at the same time and either one
// may be switched off. Pure decision code, no locks, sockets or clocks: the
// router fills a PoolInput under its state mutex and acts on the answer.
//
// Rule, in this order (owner's rule, 08.11.2026):
//  1. Warm and free: a machine that has the model loaded (or loading) and
//     no request of it in flight -> there; this machine first.
//  2. Cold, idle and empty: a machine with no work in flight and no other
//     model loaded loads it without evicting anything.
//  3. Cold and idle: a machine with no work in flight whose loaded models
//     are all idle; its own memory guard / models-max evicts them.
//  4. Otherwise wait and decide again when something frees.
// Never on a machine busy with GPU work (its ComfyUI adapter queue is not
// empty, or an image/video/music model there has work in flight): an LLM
// and a ComfyUI graph never run on one machine at the same time.
// Hysteresis: in steps 2 and 3 the machine that held the model within the
// sticky window goes first, and loading it on the other machine while the
// first still holds (or just held) it waits kPoolSpillAfter first, so a
// short busy spell never loads the model twice. A machine that is gone or
// busy with GPU work holds nothing back.
// Not here (follow-up): the partner's free memory; a cold load there relies
// on its own memory guard, and "idle" means only that nothing is in flight.
// A request another router relayed here (X-Gufo-Forwarded) is always served
// here (loop guard); it only waits while this machine has GPU work.

#include <algorithm>
#include <chrono>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/router/preset.hpp"

namespace gufo::router {

/// Set by the ComfyUI adapter on the requests its graphs make: they are
/// part of the GPU work that keeps pool requests off this machine, so the
/// busy rule does not hold them back (never honoured on a relayed request).
inline constexpr const char* kOriginHeader = "X-Gufo-Origin";

/// One relay at a time per pool model to the remote router (its GPU runs
/// one request of it at a time).
inline constexpr int kPoolRemoteCapacity = 1;
/// After a machine held a pool model, the other machine loads it only once
/// a request has waited this long (hysteresis against load ping-pong).
inline constexpr std::chrono::seconds kPoolSpillAfter{10};
/// How long a machine counts as "had the model" after it was last warm.
inline constexpr std::chrono::seconds kPoolSticky{300};
/// A local load failure keeps pool requests off the local worker this long.
inline constexpr std::chrono::seconds kPoolLocalFailHold{60};

/// What one machine looks like for one pool model.
struct PoolSide {
  /// Remote: reachable and lists the model as served there (not relayed on,
  /// not failed for this request). Local: no recent load failure.
  bool available{false};
  /// The model is loaded or loading there.
  bool warm{false};
  /// No request of this model in flight there (ours or anyone's).
  bool free{false};
  /// GPU work runs there (ComfyUI queue, image/video/music model busy):
  /// never send a pool request there.
  bool gpu_busy{false};
  /// Nothing at all in flight there (every loaded model is idle, so a load
  /// may evict them).
  bool idle{false};
  /// No other model loaded or loading there (a load evicts nothing).
  bool empty{false};
  /// The model was warm there within kPoolSticky.
  bool sticky{false};
};

struct PoolInput {
  PoolSide local;
  PoolSide remote;
  /// The request carries X-Gufo-Forwarded: serve it here.
  bool forwarded{false};
  /// How long this request has been waiting for a decision.
  std::chrono::milliseconds waited{0};
};

enum class PoolChoice { kLocal, kRemote, kWait };

struct PoolDecision {
  PoolChoice choice{PoolChoice::kWait};
  /// Short reason for the log and /router/status ("warm", "load", ...).
  const char* reason{""};
};

[[nodiscard]] inline const char* PoolChoiceName(PoolChoice choice) {
  return choice == PoolChoice::kLocal    ? "local"
         : choice == PoolChoice::kRemote ? "remote"
                                         : "wait";
}

[[nodiscard]] inline PoolDecision ChoosePool(const PoolInput& in) {
  if (in.forwarded) {
    if (in.local.gpu_busy) {
      return {PoolChoice::kWait, "forwarded, local gpu busy"};
    }
    return {PoolChoice::kLocal, "forwarded"};
  }
  // GPU work on a machine keeps pool requests off it only while the other
  // machine can take them: with the partner gone (pause, restart, cable),
  // this machine serves despite its GPU work instead of waiting forever
  // (09.11.: the 27B on the .56 stood still beside a ComfyUI job while the
  // .198 was paused).
  const auto usable = [&in](const PoolSide& side) {
    const PoolSide& other = &side == &in.local ? in.remote : in.local;
    return side.available && (!side.gpu_busy || !other.available);
  };
  // 1. Warm and free, this machine first.
  if (usable(in.local) && in.local.warm && in.local.free) {
    return {PoolChoice::kLocal, "warm"};
  }
  if (usable(in.remote) && in.remote.warm && in.remote.free) {
    return {PoolChoice::kRemote, "warm"};
  }
  // 2./3. A cold load on an idle machine; the machine that had the model
  // recently goes first.
  const bool remote_first = in.remote.sticky && !in.local.sticky;
  const PoolSide* order[2] = {remote_first ? &in.remote : &in.local,
                              remote_first ? &in.local : &in.remote};
  for (const bool need_empty : {true, false}) {
    for (const PoolSide* side : order) {
      const PoolSide& other = side == &in.local ? in.remote : in.local;
      if (!usable(*side) || side->warm || !side->idle ||
          (need_empty && !side->empty)) {
        continue;
      }
      // Hysteresis: the other machine holds (or just held) it and could
      // take it again soon; a machine that is gone or busy with GPU work
      // holds nothing back.
      if (usable(other) && (other.warm || other.sticky) && !side->sticky &&
          in.waited < kPoolSpillAfter) {
        continue;
      }
      return {side == &in.local ? PoolChoice::kLocal : PoolChoice::kRemote,
              need_empty ? "load" : "load after evicting idle models"};
    }
  }
  return {PoolChoice::kWait, "both busy"};
}

/// The LoRA names a preset registers: every `lora = ...` value (argv
/// "--lora", value) split at ';' into entries `name=path[,steps=N,...]`,
/// the name before '=' of each. One line may register several adapters
/// (`lora = a=p1;b=p2`, the form gufo 2.0's image worker requires since
/// it rejects a repeated --lora); ',' only separates an entry's own
/// attributes (video LoRAs), never two adapters. What GET
/// /router/capabilities lists and the pool checks a request's "loras"
/// against.
[[nodiscard]] inline std::vector<std::string> LoraNamesOf(
    const PresetModel& preset) {
  std::vector<std::string> names;
  for (std::size_t i = 0; i + 1 < preset.argv.size(); ++i) {
    if (preset.argv[i] != "--lora")
      continue;
    const std::string_view value = preset.argv[++i];
    std::size_t start = 0;
    while (start <= value.size()) {
      const std::size_t end = value.find(';', start);
      const std::string_view entry = value.substr(
          start, end == std::string_view::npos ? std::string_view::npos
                                               : end - start);
      const std::string_view name = entry.substr(0, entry.find('='));
      if (!name.empty() &&
          std::find(names.begin(), names.end(), name) == names.end())
        names.emplace_back(name);
      if (end == std::string_view::npos)
        break;
      start = end + 1;
    }
  }
  return names;
}

/// "local" | "remote" from an X-Gufo-Pool header value; "" for anything
/// else (any side).
[[nodiscard]] inline std::string PoolSideOf(std::string_view header) {
  return header == "local" || header == "remote" ? std::string(header)
                                                  : std::string();
}

/// Whether a local model counts as GPU work for the pool's busy rule:
/// modality image or video, or an external
/// `args = serve image|video|music|objekt`.
[[nodiscard]] inline bool IsGpuWorkModel(const PresetModel& preset) {
  if (preset.modality == "image" || preset.modality == "video") {
    return true;
  }
  return preset.argv.size() >= 2 && preset.argv[0] == "serve" &&
         (preset.argv[1] == "image" || preset.argv[1] == "video" ||
          preset.argv[1] == "music" || preset.argv[1] == "objekt");
}

}  // namespace gufo::router

#endif  // GUFO_CLI_SERVE_ROUTER_POOL_HPP_
