#ifndef GUFO_CORE_TTFT_TRACE_HPP_
#define GUFO_CORE_TTFT_TRACE_HPP_

// Prompt-start diagnostics, off unless GUFO_TTFT_TRACE is set to a value
// other than "0". Prints one stderr line per traced step ("ttft-trace ...").
// Traced steps may synchronize the device to attribute GPU time, so timings
// with the switch on are for attribution only.

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace gufo::ttft {

[[nodiscard]] inline bool TraceEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("GUFO_TTFT_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

using Clock = std::chrono::steady_clock;

[[nodiscard]] inline double MsBetween(Clock::time_point from,
                                      Clock::time_point to) {
  return std::chrono::duration<double, std::milli>(to - from).count();
}

}  // namespace gufo::ttft

#endif  // GUFO_CORE_TTFT_TRACE_HPP_
