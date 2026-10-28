#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXPERT_STREAM_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXPERT_STREAM_HPP_

#include <hip/hip_runtime_api.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
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
    std::uint64_t prefetched{0};     ///< experts loaded ahead of Map()
    std::uint64_t prefetch_hits{0};  ///< of those, routed by Map()
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

  /// Routing counts that survive restarts. LoadProfile ignores a file of
  /// another layer/expert shape; SaveProfile replaces the file atomically.
  /// With a profile path set, the counts are also saved with every stats
  /// report and on destruction.
  bool LoadProfile(const std::filesystem::path& path);
  bool SaveProfile(const std::filesystem::path& path) const;
  void SetProfilePath(std::filesystem::path path) {
    profile_path_ = std::move(path);
  }
  /// Loads each layer's most routed experts into its slots without counting
  /// them as routing; returns the number of experts loaded.
  [[nodiscard]] bool Warm(hipStream_t stream, std::uint64_t* warmed,
                          std::string* error_msg = nullptr);

  /// Starts background reads of `ids` (predicted experts of `layer`) into
  /// free slots; Map(layer) uploads them before mapping. One prediction is
  /// in flight at a time; returns false when none was started.
  bool Prefetch(std::uint32_t layer, std::span<const std::int32_t> ids);
  /// Adds per-expert routing counts of a prefill batch to the eviction
  /// score, so the experts a prompt uses most stay for its continuation.
  void AddUses(std::uint32_t layer, std::span<const std::uint32_t> rows) {
    auto& uses = state_[layer].uses;
    for (std::size_t e = 0; e < rows.size() && e < uses.size(); ++e)
      uses[e] += rows[e];
  }

  [[nodiscard]] std::uint32_t slots() const noexcept { return slots_; }
  [[nodiscard]] std::uint32_t max_misses() const noexcept {
    return max_misses_;
  }
  [[nodiscard]] bool Resident(std::uint32_t layer,
                              std::int32_t expert) const noexcept {
    return state_[layer].slot_of[expert] >= 0;
  }
  [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
  struct LayerState {
    std::vector<std::int32_t> slot_of;    ///< expert -> slot or -1.
    std::vector<std::int32_t> expert_in;  ///< slot -> expert or -1.
    std::vector<std::uint64_t> used;      ///< slot -> last use tick.
    std::vector<std::uint32_t> uses;      ///< expert -> aged routing count.
    std::uint32_t maps{0};                ///< Map() calls since the last aging.
  };
  struct Job {
    platform::PlatformFile file;
    std::uint64_t offset;
    std::uint64_t bytes;
    std::uint8_t* destination;
    bool prefetch{false};  ///< counted in prefetch_pending_
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
  std::filesystem::path profile_path_;
  bool count_routing_{true};
  std::uint64_t use_ticks_{0};  // 0 = pure LRU, measured fastest on .198
  std::uint32_t age_maps_{256};
  std::chrono::steady_clock::time_point report_at_{};

  std::uint8_t* staging_{nullptr};  ///< Pinned, max_misses * expert set.
  std::size_t staging_bytes_{0};
  hipEvent_t staging_done_{nullptr};

  std::mutex mutex_;
  std::condition_variable work_, done_;
  std::deque<Job> queue_;
  std::size_t pending_{0};
  bool failed_{false};
  std::size_t prefetch_pending_{0};
  bool prefetch_failed_{false};
  struct PendingPrefetch {
    bool active{false};
    std::uint32_t layer{0};
    std::vector<std::int32_t> experts, slots;
  };
  PendingPrefetch prefetch_;
  std::uint8_t* prefetch_staging_{nullptr};  ///< Pinned, like staging_.
  hipEvent_t prefetch_done_{nullptr};
  bool FinishPrefetch(hipStream_t stream, std::string* error_msg);
  bool stop_{false};
  std::vector<std::thread> workers_;
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXPERT_STREAM_HPP_
