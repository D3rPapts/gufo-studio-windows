#ifndef GUFO_MODELS_MINIMAX_H3_STAGE_CACHE_HPP_
#define GUFO_MODELS_MINIMAX_H3_STAGE_CACHE_HPP_

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"

namespace gufo::minimax_h3 {

/// Disk cache for per-job intermediates that depend only on their inputs
/// (the layer-50 prompt embedding and the AdaLN modulations). Entries are
/// written by the GPU path that would otherwise compute them, so a hit
/// returns the same bytes; they survive the worker recycle between jobs.
/// Opt-in: GUFO_H3_CACHE_DIR names the directory, GUFO_H3_CACHE_MAX_MB
/// bounds it (least recently used entries are removed first).
inline constexpr std::string_view kStageCacheDirectoryVariable =
    "GUFO_H3_CACHE_DIR";
inline constexpr std::string_view kStageCacheLimitVariable =
    "GUFO_H3_CACHE_MAX_MB";
inline constexpr std::uint64_t kDefaultStageCacheBytes = 8ULL << 30U;

struct StageCacheOptions {
  std::filesystem::path directory;
  std::uint64_t max_bytes{kDefaultStageCacheBytes};
};

/// Options from the two values (either may be null). An empty or missing
/// directory disables the cache; an invalid or zero limit keeps the default.
[[nodiscard]] std::optional<StageCacheOptions> ParseStageCacheOptions(
    const char* directory, const char* max_megabytes);
[[nodiscard]] std::optional<StageCacheOptions> StageCacheFromEnvironment();

/// Builds a cache key as the SHA-256 of length-prefixed, labelled fields,
/// so no two field sequences share an encoding.
class StageCacheKey {
public:
  explicit StageCacheKey(std::string_view schema);

  void Add(std::string_view label, std::span<const unsigned char> bytes);
  void AddText(std::string_view label, std::string_view text);
  void AddInteger(std::string_view label, std::uint64_t value);
  template <typename T>
    requires std::is_trivially_copyable_v<T>
  void AddValues(std::string_view label, std::span<const T> values) {
    Add(label, std::span(reinterpret_cast<const unsigned char*>(values.data()),
                         values.size_bytes()));
  }
  /// Path, size and last write time; a missing file adds a marker.
  void AddFileIdentity(std::string_view label,
                       const std::filesystem::path& path);
  /// Name, shard, offset and size of every tensor of `phase`, and the file
  /// identity of each shard that holds one.
  void AddInventoryPhase(const ModelInventory& inventory, Phase phase);

  [[nodiscard]] std::string Hex() const;

private:
  std::vector<unsigned char> bytes_;
};

/// 64-bit payload checksum (word-wise multiply-xor; integrity only).
class StageChecksum {
public:
  void Update(std::span<const unsigned char> bytes);
  [[nodiscard]] std::uint64_t Digest() const;

private:
  void Mix(std::uint64_t word);
  std::uint64_t state_{0x9E3779B97F4A7C15ULL};
  std::uint64_t total_{0};
  unsigned char tail_[8]{};
  std::size_t tail_size_{0};
};

/// True when no BF16 value is an infinity or NaN. Entries are only stored
/// when finite, so a faulted computation is never replayed.
[[nodiscard]] bool AllFiniteBf16(std::span<const std::uint16_t> values);

[[nodiscard]] std::filesystem::path StageCachePath(
    const StageCacheOptions& options, std::string_view kind,
    std::string_view key);

/// Writes the arrays under (kind, key) through a temporary file and a
/// rename, then trims the directory to `max_bytes`. An existing entry is
/// kept (equal keys mean equal content).
[[nodiscard]] bool StoreStageArrays(
    const StageCacheOptions& options, std::string_view kind,
    std::string_view key, std::span<const std::vector<std::uint16_t>> arrays,
    std::string* error = nullptr);

/// The arrays stored under (kind, key), or nullopt on a miss. A damaged
/// entry is removed, reported through `error` and treated as a miss. A hit
/// refreshes the entry's write time for the LRU trim.
[[nodiscard]] std::optional<std::vector<std::vector<std::uint16_t>>>
LoadStageArrays(const StageCacheOptions& options, std::string_view kind,
                std::string_view key, std::string* error = nullptr);

/// Removes the least recently written entries until the cache fits
/// `max_bytes`, and temporary files older than an hour.
void TrimStageCache(const StageCacheOptions& options);

/// HIP runtime, rocBLAS and GPU architecture of the current device, for
/// cache keys of GPU-computed values (defined in runtime_hip.cpp; HIP
/// builds only).
[[nodiscard]] std::string ComputeBackendIdentity();

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_STAGE_CACHE_HPP_
