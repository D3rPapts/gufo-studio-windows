#include "src/models/qwen_image_21/hip/memory.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <sstream>

#include "src/core/platform/os_info.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <dxgi1_4.h>
#include <windows.h>
#endif

namespace gufo::models::qwen_image_21::hip {
namespace {

#if defined(_WIN32)
// The first hardware AMD adapter; HIP on Windows allocates through it.
bool QuerySegments(DeviceMemory* memory) {
  IDXGIFactory4* factory = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4),
                                reinterpret_cast<void**>(&factory))))
    return false;
  bool found = false;
  IDXGIAdapter1* adapter = nullptr;
  for (UINT i = 0;
       !found && factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND;
       ++i) {
    DXGI_ADAPTER_DESC1 desc{};
    IDXGIAdapter3* adapter3 = nullptr;
    if (SUCCEEDED(adapter->GetDesc1(&desc)) && desc.VendorId == 0x1002 &&
        !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
        SUCCEEDED(adapter->QueryInterface(
            __uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter3)))) {
      DXGI_QUERY_VIDEO_MEMORY_INFO local{}, nonlocal{};
      if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(
              0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local)) &&
          SUCCEEDED(adapter3->QueryVideoMemoryInfo(
              0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &nonlocal))) {
        memory->local_usage = local.CurrentUsage;
        memory->local_budget = local.Budget;
        memory->nonlocal_usage = nonlocal.CurrentUsage;
        memory->nonlocal_budget = nonlocal.Budget;
        found = true;
      }
      adapter3->Release();
    }
    adapter->Release();
  }
  factory->Release();
  return found;
}
#endif

std::uint64_t Headroom(std::uint64_t budget, std::uint64_t usage) {
  return budget > usage ? budget - usage : 0;
}

constexpr std::uint64_t kMiB = 1ULL << 20;

}  // namespace

std::uint64_t DeviceMemory::Available(std::uint64_t host_reserve) const {
  const std::uint64_t host =
      host_available > host_reserve ? host_available - host_reserve : 0;
  if (!segments)
    return hip_free;
  return Headroom(local_budget, local_usage) +
         std::min(Headroom(nonlocal_budget, nonlocal_usage), host);
}

std::string DeviceMemory::Describe() const {
  std::ostringstream out;
  out << "hip_free_mib=" << hip_free / kMiB
      << " hip_total_mib=" << hip_total / kMiB;
  if (segments)
    out << " local_usage_mib=" << local_usage / kMiB
        << " local_budget_mib=" << local_budget / kMiB
        << " shared_usage_mib=" << nonlocal_usage / kMiB
        << " shared_budget_mib=" << nonlocal_budget / kMiB;
  out << " host_available_mib=" << host_available / kMiB;
  return out.str();
}

DeviceMemory QueryDeviceMemory() {
  DeviceMemory memory;
  std::size_t free = 0, total = 0;
  if (hipMemGetInfo(&free, &total) == hipSuccess) {
    memory.hip_free = free;
    memory.hip_total = total;
  } else {
    (void)hipGetLastError();
  }
#if defined(_WIN32)
  memory.segments = QuerySegments(&memory);
#endif
  memory.host_available = platform::AvailableMemoryBytes();
  return memory;
}

}  // namespace gufo::models::qwen_image_21::hip
