// Windows memory probe for router residency.
//
// GPU: the adapter with the largest dedicated segment (DXGI) gives the
// dedicated and shared totals; current use system-wide and per process come
// from the PDH "GPU Adapter Memory" and "GPU Process Memory" counters for that
// adapter's LUID (the same numbers Task Manager shows). Host: available
// physical memory (GlobalMemoryStatusEx) and each worker's working set.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <dxgi1_4.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <psapi.h>
#include <windows.h>

#include <cstdio>
#include <cwchar>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "src/cli/serve/router/memory.hpp"

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "pdh.lib")

namespace gufo::router {
namespace {

/// One wildcard counter's instances: name -> value.
std::vector<std::pair<std::wstring, std::int64_t>> ReadArray(
    PDH_HCOUNTER counter) {
  std::vector<std::pair<std::wstring, std::int64_t>> values;
  DWORD bytes = 0;
  DWORD count = 0;
  PDH_STATUS status = PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE,
                                                   &bytes, &count, nullptr);
  if (status != static_cast<PDH_STATUS>(PDH_MORE_DATA) || bytes == 0) {
    return values;
  }
  std::vector<unsigned char> buffer(bytes);
  auto* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buffer.data());
  status = PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &bytes, &count,
                                        items);
  if (status != ERROR_SUCCESS) {
    return values;
  }
  values.reserve(count);
  for (DWORD i = 0; i < count; ++i) {
    if (items[i].FmtValue.CStatus != ERROR_SUCCESS &&
        items[i].FmtValue.CStatus != PDH_CSTATUS_NEW_DATA) {
      continue;
    }
    values.emplace_back(items[i].szName, items[i].FmtValue.largeValue);
  }
  return values;
}

class WindowsMemoryProbe final : public MemoryProbe {
public:
  WindowsMemoryProbe() {
    FindAdapter();
    if (PdhOpenQueryW(nullptr, 0, &query_) != ERROR_SUCCESS) {
      query_ = nullptr;
      return;
    }
    const bool ok =
        PdhAddEnglishCounterW(query_,
                              L"\\GPU Adapter Memory(*)\\Dedicated Usage", 0,
                              &adapter_dedicated_) == ERROR_SUCCESS &&
        PdhAddEnglishCounterW(query_, L"\\GPU Adapter Memory(*)\\Shared Usage",
                              0, &adapter_shared_) == ERROR_SUCCESS &&
        PdhAddEnglishCounterW(query_,
                              L"\\GPU Process Memory(*)\\Dedicated Usage", 0,
                              &process_dedicated_) == ERROR_SUCCESS &&
        PdhAddEnglishCounterW(query_, L"\\GPU Process Memory(*)\\Shared Usage",
                              0, &process_shared_) == ERROR_SUCCESS;
    if (!ok) {
      PdhCloseQuery(query_);
      query_ = nullptr;
    }
  }

  ~WindowsMemoryProbe() override {
    if (query_ != nullptr) {
      PdhCloseQuery(query_);
    }
  }

  WindowsMemoryProbe(const WindowsMemoryProbe&) = delete;
  WindowsMemoryProbe& operator=(const WindowsMemoryProbe&) = delete;

  SystemMemory System() override {
    const std::lock_guard<std::mutex> lock(mutex_);
    SystemMemory memory;
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0) {
      memory.host_valid = true;
      memory.host_total = static_cast<std::int64_t>(status.ullTotalPhys);
      memory.host_available = static_cast<std::int64_t>(status.ullAvailPhys);
    }
    if (query_ == nullptr || dedicated_total_ <= 0 ||
        PdhCollectQueryData(query_) != ERROR_SUCCESS) {
      return memory;
    }
    bool found = false;
    for (const auto& [name, value] : ReadArray(adapter_dedicated_)) {
      if (MatchesAdapter(name)) {
        memory.gpu_dedicated_used += value;
        found = true;
      }
    }
    for (const auto& [name, value] : ReadArray(adapter_shared_)) {
      if (MatchesAdapter(name)) {
        memory.gpu_shared_used += value;
      }
    }
    memory.gpu_valid = found;
    memory.gpu_dedicated_total = dedicated_total_;
    memory.gpu_shared_total = shared_total_;
    return memory;
  }

  std::map<std::uint32_t, ProcessMemory> Processes(
      const std::vector<std::uint32_t>& pids) override {
    const std::lock_guard<std::mutex> lock(mutex_);
    std::map<std::uint32_t, ProcessMemory> result;
    if (pids.empty()) {
      return result;
    }
    const bool gpu = query_ != nullptr && dedicated_total_ > 0 &&
                     PdhCollectQueryData(query_) == ERROR_SUCCESS;
    for (const std::uint32_t pid : pids) {
      ProcessMemory usage;
      const HANDLE process =
          OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
      if (process == nullptr) {
        continue;
      }
      PROCESS_MEMORY_COUNTERS counters{};
      if (GetProcessMemoryInfo(process, &counters, sizeof(counters)) != 0) {
        usage.valid = true;
        usage.host_working_set =
            static_cast<std::int64_t>(counters.WorkingSetSize);
        usage.host_peak_working_set =
            static_cast<std::int64_t>(counters.PeakWorkingSetSize);
      }
      CloseHandle(process);
      if (usage.valid) {
        result.emplace(pid, usage);
      }
    }
    if (!gpu) {
      return result;
    }
    const auto add = [&](PDH_HCOUNTER counter, bool dedicated) {
      for (const auto& [name, value] : ReadArray(counter)) {
        unsigned long pid = 0;
        if (std::swscanf(name.c_str(), L"pid_%lu_", &pid) != 1 ||
            !MatchesAdapter(name)) {
          continue;
        }
        const auto it = result.find(static_cast<std::uint32_t>(pid));
        if (it == result.end()) {
          continue;
        }
        (dedicated ? it->second.gpu_dedicated : it->second.gpu_shared) += value;
      }
    };
    add(process_dedicated_, true);
    add(process_shared_, false);
    return result;
  }

private:
  void FindAdapter() {
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),
                                  reinterpret_cast<void**>(&factory)))) {
      return;
    }
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0;
         factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
      DXGI_ADAPTER_DESC1 desc{};
      if (SUCCEEDED(adapter->GetDesc1(&desc)) &&
          (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
          static_cast<std::int64_t>(desc.DedicatedVideoMemory) >
              dedicated_total_) {
        dedicated_total_ = static_cast<std::int64_t>(desc.DedicatedVideoMemory);
        shared_total_ = static_cast<std::int64_t>(desc.SharedSystemMemory);
        wchar_t luid[64];
        std::swprintf(luid, 64, L"luid_0x%08lx_0x%08lx_",
                      static_cast<unsigned long>(desc.AdapterLuid.HighPart),
                      static_cast<unsigned long>(desc.AdapterLuid.LowPart));
        luid_ = luid;
      }
      adapter->Release();
    }
    factory->Release();
  }

  [[nodiscard]] bool MatchesAdapter(const std::wstring& instance) const {
    return !luid_.empty() && instance.find(luid_) != std::wstring::npos;
  }

  std::mutex mutex_;
  PDH_HQUERY query_{nullptr};
  PDH_HCOUNTER adapter_dedicated_{nullptr};
  PDH_HCOUNTER adapter_shared_{nullptr};
  PDH_HCOUNTER process_dedicated_{nullptr};
  PDH_HCOUNTER process_shared_{nullptr};
  std::int64_t dedicated_total_{0};
  std::int64_t shared_total_{0};
  std::wstring luid_;
};

}  // namespace

std::unique_ptr<MemoryProbe> MakeSystemMemoryProbe() {
  return std::make_unique<WindowsMemoryProbe>();
}

}  // namespace gufo::router
