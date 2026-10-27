#ifndef GUFO_CORE_HIP_WEIGHT_UPLOAD_HPP_
#define GUFO_CORE_HIP_WEIGHT_UPLOAD_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::hip {

/// How upload destinations are written. Managed placements use host writes:
/// HIP's copy engine writing into managed memory faults (and wedges the
/// context with a sticky error) on some Windows drivers, while the CPU can
/// write the same allocation directly.
enum class WeightUploadMode {
  /// Destinations are device-heap memory; writes go through HIP copies.
  kHipCopy,
  /// Destinations are host-coherent managed memory; every write is a plain
  /// host memory copy, and staging never calls into HIP at all.
  kHostWrite,
};

/// Bounded disk-to-device pipeline. Copy queues a tensor without waiting for
/// its upload; destinations must remain alive until Finish or destruction.
/// The mapped GGUF payloads are never touched. In kHostWrite mode the
/// destinations are host-coherent memory the CPU writes directly.
class WeightUpload {
public:
  static std::unique_ptr<WeightUpload> Create(
      std::span<const core::GgufMappedRegion> shards, std::string* error,
      WeightUploadMode mode = WeightUploadMode::kHipCopy);
  ~WeightUpload();
  WeightUpload(const WeightUpload&) = delete;
  WeightUpload& operator=(const WeightUpload&) = delete;

  bool Copy(std::uint32_t shard, std::uint64_t offset, std::size_t size,
            void* device, std::string* error);
  /// Waits for every queued copy, including on failure.
  bool Finish(std::string* error);
  /// Returns and clears the diagnostics Create() recorded when staging could
  /// not use pinned buffers or non-blocking streams and fell back instead.
  /// Empty for a fully pinned, fully streamed upload; always empty in
  /// kHostWrite mode, which never attempts either. Callers log the lines.
  std::vector<std::string> TakeNotices();

  /// True when destinations are host-coherent memory the CPU writes through
  /// WriteHost/CopyRegion/ZeroTail without any HIP call.
  bool HostWrite() const;

  /// Writes `bytes` from host memory into an upload destination. Synchronous;
  /// a host write in kHostWrite mode, a hipMemcpy otherwise.
  bool WriteHost(const void* source, void* destination, std::size_t bytes);

  /// Copies `bytes` between two upload destinations (both were filled by this
  /// upload). Synchronous; a host copy in kHostWrite mode, a device-to-device
  /// hipMemcpy otherwise.
  bool CopyBytes(void* destination, const void* source, std::size_t bytes);

  /// Copies a strided 2-D region between upload destinations: `rows` rows of
  /// `width` bytes, `source` walking `source_stride` per row and
  /// `destination` walking `destination_stride`.
  bool CopyRegion(void* destination, std::size_t destination_stride,
                  const void* source, std::size_t source_stride,
                  std::size_t width, std::size_t rows);

  /// Zeroes `bytes` at `destination + offset` (the decode-path tail margin).
  /// Ordered with the upload's other writes.
  void ZeroTail(void* destination, std::size_t offset, std::size_t bytes);

private:
  struct State;
  explicit WeightUpload(WeightUploadMode mode);
  std::unique_ptr<State> state_;
};

}  // namespace gufo::hip

#endif
