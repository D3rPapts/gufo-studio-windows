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

#include <chrono>
#include <string>

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
  const auto usable = [](const PoolSide& side) {
    return side.available && !side.gpu_busy;
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

/// Whether a local model counts as GPU work for the pool's busy rule:
/// modality image or video, or an external `args = serve image|video|music`.
[[nodiscard]] inline bool IsGpuWorkModel(const PresetModel& preset) {
  if (preset.modality == "image" || preset.modality == "video") {
    return true;
  }
  return preset.argv.size() >= 2 && preset.argv[0] == "serve" &&
         (preset.argv[1] == "image" || preset.argv[1] == "video" ||
          preset.argv[1] == "music");
}

}  // namespace gufo::router

#endif  // GUFO_CLI_SERVE_ROUTER_POOL_HPP_
