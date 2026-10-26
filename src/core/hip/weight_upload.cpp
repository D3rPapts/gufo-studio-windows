#include "src/core/hip/weight_upload.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/hip/host_allocations.hpp"
#include "src/core/platform/fs.hpp"

namespace gufo::hip {
namespace {

constexpr std::size_t kAlignment = 4096;
constexpr std::size_t kChunkBytes = 16ULL << 20;
constexpr std::size_t kReaders = 16;

// Test seams: force the degraded staging paths on machines where pinned
// allocations and non-blocking streams succeed, so the fallbacks stay
// exercised. Read per Create(), not cached, so tests can flip them mid-run.
bool ForceUnpinnedStaging() {
  return std::getenv("GUFO_TEST_UNPINNED_STAGING") != nullptr;
}

bool ForceSyncTransfers() {
  return std::getenv("GUFO_TEST_SYNC_TRANSFERS") != nullptr;
}

void* AlignUp(void* data) {
  auto address = reinterpret_cast<std::uintptr_t>(data);
  address = (address + kAlignment - 1) & ~(kAlignment - 1);
  return reinterpret_cast<void*>(address);
}

bool ReadFully(platform::PlatformFile fd, void* buffer, std::uint64_t offset,
               std::size_t length, std::size_t required, std::error_code& ec) {
  std::size_t got = 0;
  while (got < required) {
    const auto n = platform::ReadAt(
        fd, offset + got, static_cast<char*>(buffer) + got, length - got, ec);
    if (n == 0 && ec == std::errc::interrupted) {
      continue;
    }
    if (n == 0) {
      if (!ec) {
        // End of file before the required bytes.
        ec = std::make_error_code(std::errc::io_error);
      }
      return false;
    }
    got += n;
  }
  return true;
}

}  // namespace

struct WeightUpload::State {
  struct Shard {
    platform::PlatformFile fd{platform::kInvalidFile};
    std::uint64_t size{0};
  };
  struct Slot {
    void* buffer{nullptr};
    // Owns the backing when the pinned allocation was rejected and the slot
    // fell back to pageable memory.
    std::unique_ptr<std::byte[]> unpinned;
    hipStream_t stream{nullptr};
    // One independent description per reader, reopened when the reader moves
    // to another shard: concurrent readers never queue on one synchronous
    // handle. Windows reopens cached overlapped because unbuffered reads of
    // a file that is also mapped serialize (NTFS); Linux attempts direct
    // reads, which need alignment.
    platform::PlatformFile reopened{platform::kInvalidFile};
    std::uint32_t reopened_shard{0xffffffffu};
    bool direct{false};
  };
  struct Task {
    std::uint32_t shard;
    std::uint64_t offset;
    std::size_t size;
    void* destination;
  };

  std::vector<Shard> shards;
  HostAllocations host_allocations;
  std::array<Slot, kReaders> slots{};
  std::vector<std::jthread> readers;
  std::vector<std::string> notices;
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<Task> queue;
  std::size_t pending{0};
  bool stopping{false};
  std::string failure;

  ~State() {
    {
      std::lock_guard lock(mutex);
      stopping = true;
    }
    changed.notify_all();
    // Workers finish queued and in-flight copies before their buffers or the
    // caller's destinations can be freed, even on a failed upload.
    readers.clear();
    for (auto& slot : slots) {
      platform::CloseFile(slot.reopened);
      if (slot.stream != nullptr) {
        (void)hipStreamDestroy(slot.stream);
      }
    }
    for (const auto& shard : shards) {
      platform::CloseFile(shard.fd);
    }
  }

  bool Status(std::string* error) const {
    if (!failure.empty() && error != nullptr) {
      *error = failure;
    }
    return failure.empty();
  }

  void Fail(std::string message) {
    // Caller holds mutex (or workers have not started yet).
    if (failure.empty()) {
      failure = std::move(message);
    }
    pending -= queue.size();
    queue.clear();
    changed.notify_all();
  }

  std::string Transfer(const Task& task, Slot& slot) {
    const auto& shard = shards[task.shard];
    const void* payload = slot.buffer;
    std::error_code ec;
    bool read = false;
    // Each reader keeps an independent description of the shard it reads, so
    // concurrent readers never queue on one synchronous handle: Windows
    // serializes synchronous I/O per file object, not per process. Windows
    // reopens cached overlapped; Linux attempts direct I/O. When neither is
    // available, reads fall through to the shared retained handle.
    if (slot.reopened_shard != task.shard) {
      platform::CloseFile(slot.reopened);
      slot.reopened = platform::kInvalidFile;
      slot.reopened_shard = task.shard;
    }
    if (!platform::FileValid(slot.reopened)) {
      slot.reopened = platform::ReopenRandomReadFile(shard.fd, slot.direct);
    }
    if (platform::FileValid(slot.reopened) && !slot.direct) {
      read = ReadFully(slot.reopened, slot.buffer, task.offset, task.size,
                       task.size, ec);
      payload = slot.buffer;
      if (!read) {
        return "shard read failed: " + ec.message();
      }
    } else if (platform::FileValid(slot.reopened)) {
      const auto begin = task.offset & ~(kAlignment - 1);
      const auto skip = static_cast<std::size_t>(task.offset - begin);
      const auto length =
          (skip + task.size + kAlignment - 1) & ~(kAlignment - 1);
      read = ReadFully(slot.reopened, slot.buffer, begin, length,
                       skip + task.size, ec);
      payload = static_cast<const char*>(slot.buffer) + skip;
      if (!read && ec != std::errc::invalid_argument &&
          ec != std::errc::operation_not_supported) {
        return "shard read failed: " + ec.message();
      }
    }
    if (!read) {
      // Some filesystems accept unbuffered opens but reject aligned reads.
      ec = {};
      if (!ReadFully(shard.fd, slot.buffer, task.offset, task.size, task.size,
                     ec)) {
        return "shard read failed: " + ec.message();
      }
      payload = slot.buffer;
      platform::Advise(shard.fd, platform::AdviseHint::kDontNeed, task.offset,
                       task.size);
    }
    // Without a stream (stream creation was rejected) the copy runs on the
    // blocking default stream; hipMemcpy is its synchronous spelling.
    auto status = slot.stream != nullptr
                      ? hipMemcpyAsync(task.destination, payload, task.size,
                                       hipMemcpyHostToDevice, slot.stream)
                      : hipMemcpy(task.destination, payload, task.size,
                                  hipMemcpyHostToDevice);
    if (status == hipSuccess && slot.stream != nullptr) {
      status = hipStreamSynchronize(slot.stream);
    }
    if (status != hipSuccess) {
      return "weight upload failed: " + std::string(hipGetErrorString(status));
    }
    return {};
  }

  void Run(Slot& slot) {
    for (;;) {
      Task task{};
      {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return stopping || !queue.empty(); });
        if (queue.empty()) {
          return;
        }
        task = queue.front();
        queue.pop_front();
        changed.notify_all();
      }
      auto error = Transfer(task, slot);
      {
        std::lock_guard lock(mutex);
        --pending;
        if (!error.empty()) {
          Fail(std::move(error));
        }
      }
      changed.notify_all();
    }
  }
};

WeightUpload::WeightUpload() : state_(std::make_unique<State>()) {}
WeightUpload::~WeightUpload() = default;

std::unique_ptr<WeightUpload> WeightUpload::Create(
    std::span<const core::GgufMappedRegion> regions, std::string* error) {
  std::unique_ptr<WeightUpload> uploader(new WeightUpload());
  auto& s = *uploader->state_;
  for (const auto& region : regions) {
    auto& shard = s.shards.emplace_back();
    shard.fd = platform::DuplicateFile(region.file_descriptor);
    std::error_code ec;
    platform::FileInfo info{};
    if (!platform::FileValid(shard.fd) ||
        !platform::Stat(shard.fd, &info, ec) || info.size == 0 ||
        info.size != region.size) {
      s.Fail("cannot read the mapped GGUF shard");
      s.Status(error);
      return nullptr;
    }
    shard.size = info.size;
  }
  // Machines can reject pinned allocations or extra streams outright (driver
  // lockable-memory policy, hardware queue caps); degrade per slot instead of
  // failing the whole load, and record a notice the caller can log.
  std::size_t unpinned_slots = 0;
  std::size_t sync_slots = 0;
  std::string pinned_error;
  std::string stream_error;
  for (std::size_t index = 0; index < s.slots.size(); ++index) {
    auto& slot = s.slots[index];
    const auto pinned = ForceUnpinnedStaging()
                            ? hipErrorOutOfMemory
                            : s.host_allocations.Allocate(
                                  &slot.buffer, kChunkBytes + kAlignment);
    if (pinned == hipSuccess) {
      continue;
    }
    // Aligned reads touch up to kChunkBytes + kAlignment bytes from the
    // aligned base, so the raw allocation carries a second kAlignment for the
    // AlignUp() shift.
    slot.unpinned.reset(new (std::nothrow)
                            std::byte[kChunkBytes + 2 * kAlignment]);
    if (slot.unpinned == nullptr) {
      s.Fail("weight upload staging buffer allocation failed (slot " +
             std::to_string(index) + "): pinned reported " +
             hipGetErrorString(pinned) +
             "; the unpinned fallback exhausted host memory");
      s.Status(error);
      return nullptr;
    }
    slot.buffer = AlignUp(slot.unpinned.get());
    if (pinned_error.empty()) {
      pinned_error = hipGetErrorString(pinned);
    }
    ++unpinned_slots;
  }
  for (auto& slot : s.slots) {
    if (slot.buffer == nullptr) {
      continue;
    }
    const auto stream =
        ForceSyncTransfers()
            ? hipErrorOutOfMemory
            : hipStreamCreateWithFlags(&slot.stream, hipStreamNonBlocking);
    if (stream != hipSuccess) {
      slot.stream = nullptr;
      if (stream_error.empty()) {
        stream_error = hipGetErrorString(stream);
      }
      ++sync_slots;
    }
  }
  if (unpinned_slots != 0) {
    s.notices.push_back(
        "event=staging_unpinned count=" + std::to_string(unpinned_slots) +
        " of " + std::to_string(s.slots.size()) + " hip_error=" + pinned_error);
  }
  if (sync_slots != 0) {
    s.notices.push_back(
        "event=staging_sync count=" + std::to_string(sync_slots) + " of " +
        std::to_string(s.slots.size()) + " hip_error=" + stream_error);
  }
  try {
    for (auto& slot : s.slots) {
      s.readers.emplace_back([&s, &slot] { s.Run(slot); });
    }
  } catch (const std::system_error& e) {
    std::lock_guard lock(s.mutex);
    s.Fail("weight upload worker creation failed: " + std::string(e.what()));
    s.Status(error);
    return nullptr;
  }
  return uploader;
}

bool WeightUpload::Copy(std::uint32_t shard, std::uint64_t offset,
                        std::size_t size, void* device, std::string* error) {
  auto& s = *state_;
  std::unique_lock lock(s.mutex);
  if (shard >= s.shards.size() || offset > s.shards[shard].size ||
      size > s.shards[shard].size - offset || (size != 0 && !device)) {
    s.Fail("weight upload range is outside its shard");
  }
  std::size_t done = 0;
  while (s.Status(error) && done < size) {
    s.changed.wait(lock, [&] {
      return !s.failure.empty() || s.queue.size() < 2 * kReaders;
    });
    if (!s.Status(error)) {
      break;
    }
    const auto count = std::min(size - done, kChunkBytes);
    s.queue.push_back(
        {shard, offset + done, count, static_cast<char*>(device) + done});
    ++s.pending;
    done += count;
    s.changed.notify_all();
  }
  return s.Status(error);
}

bool WeightUpload::Finish(std::string* error) {
  auto& s = *state_;
  std::unique_lock lock(s.mutex);
  s.changed.wait(lock, [&] { return s.pending == 0; });
  return s.Status(error);
}

std::vector<std::string> WeightUpload::TakeNotices() {
  auto& s = *state_;
  std::lock_guard lock(s.mutex);
  return std::exchange(s.notices, {});
}

}  // namespace gufo::hip
