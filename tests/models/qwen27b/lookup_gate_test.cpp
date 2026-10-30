#include "src/models/qwen/hip/dflash/prompt_lookup_27b.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace qwen27 = gufo::hip::qwen27;
namespace {
void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

void SetEnv(const char* name, const char* value) {
#if defined(_WIN32)
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

void CheckPhase() {
  constexpr std::uint32_t kStart = 151667;
  constexpr std::uint32_t kEnd = 151668;
  const auto reasoning = [&](std::vector<std::uint32_t> tokens) {
    return qwen27::InReasoning<std::uint32_t>(tokens, kStart, kEnd);
  };
  Require(!reasoning({}), "empty context is an answer");
  Require(!reasoning({1, 2, 3}), "no thinking tokens is an answer");
  Require(reasoning({1, kStart, 5, 6}), "open <think> is reasoning");
  Require(!reasoning({1, kStart, 5, kEnd, 7}), "after </think> is the answer");
  Require(!reasoning({kStart, 10, kEnd}), "disabled thinking is an answer");
  // A new turn reopens reasoning after an earlier closed block.
  Require(reasoning({kStart, 1, kEnd, 2, kStart, 3}), "second turn reasons");
}

void CheckGate() {
  // The gate is read once per process: set every knob before the first call.
  SetEnv("GUFO_QWEN27_LOOKUP_MIN", "12");
  SetEnv("GUFO_QWEN27_LOOKUP_MIN_ANSWER", "6");
  SetEnv("GUFO_QWEN27_LOOKUP_GENERATED", "1");
  SetEnv("GUFO_QWEN27_LOOKUP", "7");
  const auto& gate = qwen27::LookupStrictGate(24);
  Require(gate.reasoning == 12, "reasoning gate falls back to _MIN");
  Require(gate.answer == 6, "answer gate reads _MIN_ANSWER");
  Require(gate.PhaseAware(), "different gates are phase aware");
  Require(qwen27::LookupIndexesGenerated(), "_GENERATED=1 indexes output");
  Require(qwen27::LookupDFlashCap() == 7, "GUFO_QWEN27_LOOKUP caps DFlash2");
}
}  // namespace

int main() {
  try {
    CheckPhase();
    CheckGate();
  } catch (const std::exception& error) {
    std::cerr << "qwen27b_lookup_gate_test: " << error.what() << "\n";
    return 1;
  }
  std::cout << "qwen27b lookup gate: phase detection and knobs pass\n";
  return 0;
}
