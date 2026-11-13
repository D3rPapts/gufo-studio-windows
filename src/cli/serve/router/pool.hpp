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
#include <cctype>
#include <chrono>
#include <cmath>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/router/preset.hpp"
#include "src/cli/serve/router/remote.hpp"
#include "src/core/json.hpp"

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
/// A partner that answered a pool request 502/503/504 (its worker did not
/// start, its load timed out) gets no pool request of that model this long.
inline constexpr std::chrono::seconds kPoolRemoteFailHold{60};

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

/// The "busy" a router reports to its partner for a model (GET /v1/models
/// "activity"): a request, socket or job of it is in flight here, or a
/// pool request picked this machine and is loading it (local claim). A
/// pool request that only waits for a machine is not busy; it is reported
/// apart as "waiting". 12.11. 17:34-18:05: two requests pinned to the
/// other side each saw the other's waiting request as busy and both
/// waited the full 900 s.
[[nodiscard]] inline bool PoolActivityBusy(bool in_flight, int local_claims) {
  return in_flight || local_claims > 0;
}

/// What the pool knows about the partner for one model.
struct RemoteObservation {
  /// The last probe's view of the model.
  RemoteHub::ModelView view;
  /// The partner counts as reachable.
  bool reachable{false};
  /// Relays of this model from here that are open now.
  std::size_t ours{0};
  /// The partner's GPU slot (if any) is idle.
  bool slot_idle{true};
  /// The probe ran before our last relay ended (may still count it).
  bool stale{false};
};

/// The partner's PoolSide (without `sticky`, which the router keeps).
/// Only the partner's "busy" counts as occupied: its "waiting" requests
/// wait for a machine and hold nothing.
[[nodiscard]] inline PoolSide RemotePoolSide(const RemoteObservation& in) {
  PoolSide side;
  const RemoteHub::ModelView& view = in.view;
  side.available = view.served && in.reachable;
  side.warm = view.status == "loaded" || view.status == "loading";
  const bool theirs = view.has_activity && view.activity.busy && !in.stale;
  side.free = in.ours < static_cast<std::size_t>(kPoolRemoteCapacity) &&
              in.slot_idle && !theirs;
  side.gpu_busy = view.has_activity && view.activity.gpu_busy;
  side.idle = side.free && in.ours == 0 &&
              !(view.has_activity && view.activity.others_busy);
  side.empty = !view.others_loaded;
  return side;
}

/// How the pool treats the partner's answer to a relayed pool request
/// that has not reached the client yet (a relayed answer streams lazily,
/// so even a streaming error body can still be dropped).
enum class PoolRemoteAnswer {
  /// Pass it to the client (success, client errors, 500 and the like).
  kPass,
  /// 507: the partner could not fit the model; this machine tries once.
  kRetryMemory,
  /// No answer at all (503 remote_unavailable, made here): try here once.
  kRetryUnreachable,
  /// 502 (worker failed to start), 503 (upstream_unavailable, pool_busy,
  /// ...) or 504 (load timed out) from the partner: try here once and keep
  /// pool requests of this model off the partner for kPoolRemoteFailHold.
  kRetryFailed,
};

[[nodiscard]] inline PoolRemoteAnswer ClassifyPoolRemoteAnswer(
    int status, bool streaming_body, std::string_view log_details) {
  if (status == 507)
    return PoolRemoteAnswer::kRetryMemory;
  if (status == 503 && !streaming_body &&
      log_details.starts_with("remote_unavailable"))
    return PoolRemoteAnswer::kRetryUnreachable;
  if (status == 502 || status == 503 || status == 504)
    return PoolRemoteAnswer::kRetryFailed;
  return PoolRemoteAnswer::kPass;
}

/// Whether the partner should be held off after this answer: it failed the
/// model (502/503/504) or could not be reached at all (connect refused,
/// timeout, name lookup). Both keep pool requests of the model off it for
/// kPoolRemoteFailHold (B5: an unreachable partner is not retried on every
/// request until the next probe notices).
[[nodiscard]] inline bool PoolRemoteHold(PoolRemoteAnswer answer) {
  return answer == PoolRemoteAnswer::kRetryFailed ||
         answer == PoolRemoteAnswer::kRetryUnreachable;
}

/// pool-wait-seconds default: how long a pool request waits on the partner
/// before it falls back to this machine.
inline constexpr int kPoolWaitSecondsDefault = 60;

/// B5 "never stuck" (Fabian 13.11.): a pool request that could run here
/// never waits on the partner longer than pool-wait-seconds.
struct PoolFallbackInput {
  /// How long the request has waited for a decision.
  std::chrono::milliseconds waited{0};
  /// pool-wait-seconds of the model's remote section.
  std::chrono::seconds limit{kPoolWaitSecondsDefault};
  /// Relayed here by the partner (X-Gufo-Forwarded): the partner decides.
  bool forwarded{false};
  /// X-Gufo-Pool: remote (pulled by the partner, redirected, or a client).
  bool pinned_remote{false};
  /// This router serves the model with every LoRA the request names.
  bool local_capable{true};
};

/// Why the request stops counting on the partner now; nullptr while it
/// still may wait for it. With a reason, the router drops the partner from
/// the decision (WithoutPartner) and a remote pin no longer holds: the
/// request runs here as soon as this machine can load or run the model
/// (evicting idle models), and queue-timeout-seconds stays the last resort.
[[nodiscard]] inline const char* PoolFallbackReason(
    const PoolFallbackInput& in) {
  if (in.forwarded || !in.local_capable || in.waited < in.limit)
    return nullptr;
  return in.pinned_remote ? "pinned to the partner too long"
                          : "waited too long for the partner";
}

/// The decision input with the partner out of the picture.
[[nodiscard]] inline PoolInput WithoutPartner(PoolInput in) {
  in.remote = PoolSide{};
  return in;
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

// ------------------------------------------------------------- priority

/// Priority of a request or job in the two-router pool (2.0.3): 0 (normal,
/// the default) to kPriorityMax; a higher value goes first, among equal
/// values the older request or job. Given as the header X-Gufo-Priority
/// (any request; it wins) or the JSON member "priority" of the body (a job
/// submit, a direct request). The job store forwards a job's priority as
/// the header on every request of the job, so the partner knows it.
inline constexpr const char* kPriorityHeader = "X-Gufo-Priority";
inline constexpr int kPriorityMax = 9;

/// The raw JSON text of the top-level member `key` of an object body
/// ("7", "\"7\"", "null", "{...}"), or nullopt when the body is not an
/// object or has no such member at depth 0. One pass over the text, no
/// allocation: the front reads a request's "priority" without parsing
/// a large body, and a key inside a string or a nested object (a chat
/// text mentioning "priority") never counts.
[[nodiscard]] inline std::optional<std::string_view> TopLevelMemberText(
    std::string_view body, std::string_view key) {
  std::size_t i = 0;
  const auto skip_ws = [&] {
    while (i < body.size() && std::isspace(static_cast<unsigned char>(body[i])))
      ++i;
  };
  // A JSON string from body[i] == '"': the index past its closing quote.
  const auto skip_string = [&]() -> bool {
    if (i >= body.size() || body[i] != '"')
      return false;
    for (++i; i < body.size(); ++i) {
      if (body[i] == '\\') {
        ++i;
      } else if (body[i] == '"') {
        ++i;
        return true;
      }
    }
    return false;
  };
  // Any value from body[i]: the index past it.
  const auto skip_value = [&]() -> bool {
    skip_ws();
    if (i >= body.size())
      return false;
    if (body[i] == '"')
      return skip_string();
    if (body[i] == '{' || body[i] == '[') {
      int depth = 0;
      for (; i < body.size(); ++i) {
        if (body[i] == '"') {
          if (!skip_string())
            return false;
          --i;
        } else if (body[i] == '{' || body[i] == '[') {
          ++depth;
        } else if (body[i] == '}' || body[i] == ']') {
          if (--depth == 0) {
            ++i;
            return true;
          }
        }
      }
      return false;
    }
    while (i < body.size() && body[i] != ',' && body[i] != '}' &&
           !std::isspace(static_cast<unsigned char>(body[i])))
      ++i;
    return true;
  };
  skip_ws();
  if (i >= body.size() || body[i] != '{')
    return std::nullopt;
  ++i;
  while (true) {
    skip_ws();
    if (i >= body.size() || body[i] == '}')
      return std::nullopt;
    const std::size_t key_begin = i;
    if (!skip_string())
      return std::nullopt;
    const std::string_view found = body.substr(key_begin + 1, i - key_begin - 2);
    skip_ws();
    if (i >= body.size() || body[i] != ':')
      return std::nullopt;
    ++i;
    skip_ws();
    const std::size_t value_begin = i;
    if (!skip_value())
      return std::nullopt;
    if (found == key)
      return body.substr(value_begin, i - value_begin);
    skip_ws();
    if (i < body.size() && body[i] == ',')
      ++i;
  }
}

/// " 7 " -> 7; nullopt for anything but an integer 0..kPriorityMax.
[[nodiscard]] inline std::optional<int> PriorityOfText(std::string_view text) {
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())))
    text.remove_prefix(1);
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
    text.remove_suffix(1);
  if (text.size() != 1 || !std::isdigit(static_cast<unsigned char>(text[0])))
    return std::nullopt;
  const int value = text[0] - '0';
  return value <= kPriorityMax ? std::optional<int>(value) : std::nullopt;
}

/// The priority a request names: the header when present (wins), else the
/// JSON member `member` (nullptr or null = not given), else 0. False with
/// *error when a value is given but is not an integer 0..kPriorityMax (as
/// a number or a digit string).
inline bool ParsePriority(std::string_view header, const json::Value* member,
                          int* out, std::string* error) {
  *out = 0;
  if (!header.empty()) {
    const auto value = PriorityOfText(header);
    if (!value) {
      *error = std::string(kPriorityHeader) + " must be an integer 0.." +
               std::to_string(kPriorityMax) + ", got '" + std::string(header) +
               "'";
      return false;
    }
    *out = *value;
    return true;
  }
  if (member == nullptr || member->is_null())
    return true;
  std::optional<int> value;
  if (member->is_number()) {
    const double number = member->as_double();
    if (number == std::floor(number) && number >= 0 && number <= kPriorityMax)
      value = static_cast<int>(number);
  } else if (member->is_string()) {
    value = PriorityOfText(member->str());
  }
  if (!value) {
    *error = "'priority' must be an integer 0.." + std::to_string(kPriorityMax);
    return false;
  }
  *out = *value;
  return true;
}

/// The priority a job view or record carries ("priority", missing = 0),
/// clamped to 0..kPriorityMax (a partner older than 2.0.3 sends none).
[[nodiscard]] inline int PriorityOfView(const json::Value& view) {
  const double value = view.member_double("priority", 0);
  if (!(value >= 0))
    return 0;
  return static_cast<int>(std::min<double>(value, kPriorityMax));
}

/// Order of the shared queue: whether a job or request of (priority_a,
/// created_a) runs before one of (priority_b, created_b): the higher
/// priority first, then the older one. Equal pairs: false (neither first).
[[nodiscard]] inline bool RunsBefore(int priority_a, long long created_a,
                                     int priority_b, long long created_b) {
  if (priority_a != priority_b)
    return priority_a > priority_b;
  return created_a < created_b;
}

/// Synchronous pool requests (no job) only poll the pool every 200 ms and
/// take the first free machine they see, so there is no order among several
/// waiting ones. The precedence rule: a request that sees a free machine
/// lets a waiting request of the same model with a higher priority go
/// first, but at most kPoolYieldMax in a row (the higher one may be pinned
/// to the other side, or have left), so nothing waits forever (B5).
inline constexpr std::chrono::seconds kPoolYieldMax{2};

/// Over its whole wait a request yields at most this much in all.
inline constexpr std::chrono::seconds kPoolYieldTotalMax{3 * kPoolYieldMax};

struct PoolYieldInput {
  /// This request's priority.
  int priority{0};
  /// The highest priority among the model's waiting requests (-1 = none).
  int highest_waiting{-1};
  /// How long this request has already yielded to that priority.
  std::chrono::milliseconds yielded{0};
  /// How long it has yielded in all (every priority).
  std::chrono::milliseconds total_yielded{0};
  /// Relayed here by the partner: it decided already, never yields.
  bool forwarded{false};
};

[[nodiscard]] inline bool PoolShouldYield(const PoolYieldInput& in) {
  return !in.forwarded && in.highest_waiting > in.priority &&
         in.yielded < kPoolYieldMax && in.total_yielded < kPoolYieldTotalMax;
}

/// How long a request has actually yielded: only the stretches in which
/// it stood back count (never the time both machines were busy), booked
/// per waiting priority (0..kPriorityMax) and in all. Each higher
/// priority gets at most kPoolYieldMax of our time however often the
/// machine frees up or the waiters alternate, and the request yields at
/// most kPoolYieldTotalMax over its whole wait.
struct PoolYieldClock {
  using Clock = std::chrono::steady_clock;
  /// Booked yield time per waiting priority.
  std::chrono::milliseconds booked[kPriorityMax + 1]{};
  /// Booked yield time in all.
  std::chrono::milliseconds total{0};
  Clock::time_point began{};
  /// The waiting priority of the open stretch (-1 = none).
  int target{-1};

  [[nodiscard]] bool yielding() const { return began != Clock::time_point{}; }
  [[nodiscard]] std::chrono::milliseconds Open(Clock::time_point now) const {
    return yielding() ? std::chrono::duration_cast<std::chrono::milliseconds>(
                            now - began)
                      : std::chrono::milliseconds(0);
  }
  /// Yielded toward `highest` so far, the open stretch included.
  [[nodiscard]] std::chrono::milliseconds YieldedFor(int highest,
                                                     Clock::time_point now) const {
    if (highest < 0 || highest > kPriorityMax)
      return std::chrono::milliseconds(0);
    return booked[highest] +
           (target == highest ? Open(now) : std::chrono::milliseconds(0));
  }
  /// Yielded in all, the open stretch included.
  [[nodiscard]] std::chrono::milliseconds Total(Clock::time_point now) const {
    return total + Open(now);
  }
  /// A yield toward `highest` starts or goes on. True when a stretch just
  /// started (for the log).
  bool Begin(int highest, Clock::time_point now) {
    if (yielding() && target == highest)
      return false;
    Pause(now);
    target = highest;
    began = now;
    return true;
  }
  /// The request no longer yields (it waits for a machine, or runs): the
  /// open stretch is booked, the clock stops.
  void Pause(Clock::time_point now) {
    if (!yielding())
      return;
    const auto open = Open(now);
    if (target >= 0 && target <= kPriorityMax)
      booked[target] += open;
    total += open;
    began = {};
  }
};

}  // namespace gufo::router

#endif  // GUFO_CLI_SERVE_ROUTER_POOL_HPP_
