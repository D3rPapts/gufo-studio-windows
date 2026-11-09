#pragma once

// GUFO_H3_STAGE_PROFILE=<step> times the stages of one denoiser forward
// (0-based step) with HIP events and prints the per-stage sums to stderr.
// Without the variable every call is a branch on a cached flag.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <utility>
#include <vector>

namespace gufo::minimax_h3::stage_profile {

struct State {
  int step{-1};
  bool active{false};
  std::vector<hipEvent_t> pool;
  std::vector<std::pair<const char*, std::size_t>> marks;
};

inline State& Get() {
  static State state = [] {
    State result;
    if (const char* value = std::getenv("GUFO_H3_STAGE_PROFILE")) {
      result.step = std::atoi(value);
    }
    return result;
  }();
  return state;
}

// Starts recording when `step` is the profiled step.
inline void Begin(int step, hipStream_t stream);

// Records the end of stage `label` (string literal) on `stream`.
inline void Mark(const char* label, hipStream_t stream) {
  State& state = Get();
  if (!state.active) {
    return;
  }
  const std::size_t index = state.marks.size();
  if (index == state.pool.size()) {
    hipEvent_t event = nullptr;
    if (hipEventCreate(&event) != hipSuccess) {
      state.active = false;
      return;
    }
    state.pool.push_back(event);
  }
  if (hipEventRecord(state.pool[index], stream) != hipSuccess) {
    state.active = false;
    return;
  }
  state.marks.emplace_back(label, index);
}

inline void Begin(int step, hipStream_t stream) {
  State& state = Get();
  if (state.step < 0 || step != state.step) {
    return;
  }
  state.active = true;
  state.marks.clear();
  Mark("begin", stream);
}

// After the stream was synchronized: sums the time between consecutive marks
// by the label of the later mark and prints the table.
inline void End() {
  State& state = Get();
  if (!state.active) {
    return;
  }
  state.active = false;
  std::vector<std::pair<std::string_view, double>> sums;
  double total = 0.0;
  for (std::size_t index = 1; index < state.marks.size(); ++index) {
    float elapsed = 0.0F;
    if (hipEventElapsedTime(&elapsed, state.pool[state.marks[index - 1].second],
                            state.pool[state.marks[index].second]) !=
        hipSuccess) {
      continue;
    }
    const std::string_view label = state.marks[index].first;
    auto it = sums.begin();
    while (it != sums.end() && it->first != label) {
      ++it;
    }
    if (it == sums.end()) {
      sums.emplace_back(label, 0.0);
      it = sums.end() - 1;
    }
    it->second += elapsed;
    total += elapsed;
  }
  std::fprintf(stderr, "h3 stage profile: step %d, %.1f ms GPU\n", state.step,
               total);
  for (const auto& [label, ms] : sums) {
    std::fprintf(stderr, "h3 stage profile: %-24.*s %9.1f ms %5.1f%%\n",
                 static_cast<int>(label.size()), label.data(), ms,
                 total > 0.0 ? 100.0 * ms / total : 0.0);
  }
}

}  // namespace gufo::minimax_h3::stage_profile
