// Non-Windows memory probe: host memory from /proc/meminfo and the worker's
// resident set from /proc/<pid>/status. No GPU numbers (gpu_valid stays
// false), so admission there only guards host RAM.

#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "src/cli/serve/router/memory.hpp"

namespace gufo::router {
namespace {

std::int64_t ReadKiB(const std::string& file, const std::string& key) {
  std::ifstream in(file);
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind(key, 0) == 0) {
      std::istringstream fields(line.substr(key.size()));
      std::int64_t kib = -1;
      fields >> kib;
      return kib < 0 ? -1 : kib * 1024;
    }
  }
  return -1;
}

class ProcMemoryProbe final : public MemoryProbe {
public:
  SystemMemory System() override {
    SystemMemory memory;
    memory.host_total = ReadKiB("/proc/meminfo", "MemTotal:");
    memory.host_available = ReadKiB("/proc/meminfo", "MemAvailable:");
    memory.host_valid = memory.host_total > 0 && memory.host_available >= 0;
    return memory;
  }

  std::map<std::uint32_t, ProcessMemory> Processes(
      const std::vector<std::uint32_t>& pids) override {
    std::map<std::uint32_t, ProcessMemory> result;
    for (const std::uint32_t pid : pids) {
      const std::string file = "/proc/" + std::to_string(pid) + "/status";
      ProcessMemory usage;
      usage.host_working_set = ReadKiB(file, "VmRSS:");
      usage.host_peak_working_set = ReadKiB(file, "VmHWM:");
      usage.valid = usage.host_working_set >= 0;
      if (usage.valid) {
        result.emplace(pid, usage);
      }
    }
    return result;
  }
};

}  // namespace

std::unique_ptr<MemoryProbe> MakeSystemMemoryProbe() {
  return std::make_unique<ProcMemoryProbe>();
}

}  // namespace gufo::router
