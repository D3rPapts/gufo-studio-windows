#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXPERT_STREAM_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXPERT_STREAM_HPP_

#include <hip/hip_runtime_api.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "src/core/platform/fs.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

/// Routed-expert cache for a host whose memory cannot hold every expert.
///
/// Each MoE layer keeps `slots` experts of its gate, up and down matrices in
/// device pools laid out exactly like the stacked GGUF tensors, so the routed
/// kernels address a slot as if it were an expert. Before a layer's experts
/// run, Map() replaces the routed expert ids by slot ids, reading every
/// missing expert from the GGUF shard on worker threads into pinned staging
/// and copying it into its slot on the layer's stream. Slots are evicted
/// least-recently-used; an expert routed in the current call is never
/// evicted by the same call.
class ExpertStream {
public:
  /// One stacked expert tensor in a GGUF shard and its device pool.
  struct Tensor {
    platform::PlatformFile file{platform::kInvalidFile};
    std::uint64_t file_offset{0};   ///< Offset of expert 0.
    std::uint64_t expert_bytes{0};  ///< Bytes of one expert matrix.
    void* pool{nullptr};            ///< slots * expert_bytes (+ tail).
  };
  struct Layer {
    std::array<Tensor, 3> tensors;  ///< gate, up, down.
  };
  struct Stats {
    std::uint64_t routed{0};
    std::uint64_t misses{0};
    std::uint64_t bytes_read{0};
    double read_ms{0};
  };

  /// `files` own nothing: the caller keeps them open for the stream's life.
  /// `max_misses` bounds the experts one Map() call may load (pinned
  /// staging is sized for it).
  [[nodiscard]] static std::unique_ptr<ExpertStream> Create(
      std::vector<Layer> layers, std::uint32_t num_experts, std::uint32_t slots,
      std::uint32_t max_misses, std::uint32_t threads,
      std::string* error_msg = nullptr);
  ~ExpertStream();
  ExpertStream(const ExpertStream&) = delete;
  ExpertStream& operator=(const ExpertStream&) = delete;

  /// Rewrites `ids` (expert ids of one layer, any count) into slot ids,
  /// loading missing experts and queueing their uploads on `stream`.
  [[nodiscard]] bool Map(std::uint32_t layer, std::span<std::int32_t> ids,
                         hipStream_t stream, std::string* error_msg = nullptr);

  [[nodiscard]] std::uint32_t slots() const noexcept { return slots_; }
  [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
  struct LayerState {
    std::vector<std::int32_t> slot_of;    ///< expert -> slot or -1.
    std::vector<std::int32_t> expert_in;  ///< slot -> expert or -1.
    std::vector<std::uint64_t> used;      ///< slot -> last use tick.
  };
  struct Job {
    platform::PlatformFile file;
    std::uint64_t offset;
    std::uint64_t bytes;
    std::uint8_t* destination;
  };

  ExpertStream() = default;
  void Worker();
  void Report();
  bool ReadAll(std::vector<Job>& jobs, std::string* error_msg);

  std::vector<Layer> layers_;
  std::vector<LayerState> state_;
  std::uint32_t num_experts_{0};
  std::uint32_t slots_{0};
  std::uint32_t max_misses_{0};
  std::uint64_t tick_{0};
  Stats stats_;
  /// Totals at the last stderr report (one line per 20 s of activity).
  Stats reported_;
  std::chrono::steady_clock::time_point report_at_{};

  std::uint8_t* staging_{nullptr};  ///< Pinned, max_misses * expert set.
  std::size_t staging_bytes_{0};
  hipEvent_t staging_done_{nullptr};

  std::mutex mutex_;
  std::condition_variable work_, done_;
  std::deque<Job> queue_;
  std::size_t pending_{0};
  bool failed_{false};
  bool stop_{false};
  std::vector<std::thread> workers_;
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXPERT_STREAM_HPP_
