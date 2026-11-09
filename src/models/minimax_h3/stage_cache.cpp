#include "src/models/minimax_h3/stage_cache.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <set>
#include <thread>
#include <utility>

#include "src/models/minimax_h3/sha256.hpp"

namespace gufo::minimax_h3 {
namespace {

constexpr std::array<char, 8> kMagic = {'G', 'U', 'F', 'O', 'H', '3', 'C', '1'};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint16_t kByteOrderMark = 0xFEFF;
constexpr std::string_view kFilePrefix = "h3-";
constexpr std::string_view kFileSuffix = ".bin";
constexpr std::string_view kTemporaryMarker = ".tmp-";
// Upper bound on arrays per entry (the AdaLN entry has 51).
constexpr std::uint64_t kMaximumArrays = 4096;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool ValidName(std::string_view value) {
  return !value.empty() && value.size() <= 128 &&
         std::ranges::all_of(value, [](char character) {
           return (character >= 'a' && character <= 'z') ||
                  (character >= '0' && character <= '9') || character == '-';
         });
}

template <typename T>
void Put(std::ofstream* stream, StageChecksum* checksum, const T& value) {
  const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
  checksum->Update(std::span(bytes, sizeof(T)));
  stream->write(reinterpret_cast<const char*>(bytes), sizeof(T));
}

void PutBytes(std::ofstream* stream, StageChecksum* checksum,
              std::span<const unsigned char> bytes) {
  checksum->Update(bytes);
  stream->write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
}

template <typename T>
bool Get(std::ifstream* stream, StageChecksum* checksum, T* value) {
  auto* bytes = reinterpret_cast<unsigned char*>(value);
  if (!stream->read(reinterpret_cast<char*>(bytes), sizeof(T))) {
    return false;
  }
  checksum->Update(std::span<const unsigned char>(bytes, sizeof(T)));
  return true;
}

bool GetBytes(std::ifstream* stream, StageChecksum* checksum,
              std::span<unsigned char> bytes) {
  if (!stream->read(reinterpret_cast<char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()))) {
    return false;
  }
  checksum->Update(bytes);
  return true;
}

std::span<const unsigned char> TextBytes(std::string_view text) {
  return {reinterpret_cast<const unsigned char*>(text.data()), text.size()};
}

std::span<unsigned char> MutableBytes(std::string* text) {
  return {reinterpret_cast<unsigned char*>(text->data()), text->size()};
}

std::string Utf8(const std::filesystem::path& path) {
  const std::u8string text = path.generic_u8string();
  return {reinterpret_cast<const char*>(text.data()), text.size()};
}

std::string TemporarySuffix() {
  static std::atomic<std::uint64_t> sequence{0};
  const std::uint64_t thread =
      std::hash<std::thread::id>{}(std::this_thread::get_id());
  const auto now =
      std::chrono::steady_clock::now().time_since_epoch().count();
  return std::string(kTemporaryMarker) + std::to_string(thread) + "-" +
         std::to_string(now) + "-" + std::to_string(sequence.fetch_add(1));
}

}  // namespace

std::optional<StageCacheOptions> ParseStageCacheOptions(
    const char* directory, const char* max_megabytes) {
  if (directory == nullptr || directory[0] == '\0') {
    return std::nullopt;
  }
  StageCacheOptions options;
  options.directory = std::filesystem::path(directory);
  if (max_megabytes != nullptr && max_megabytes[0] != '\0') {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(max_megabytes, &end, 10);
    if (end != nullptr && *end == '\0' && value > 0 &&
        value <= (std::numeric_limits<std::uint64_t>::max() >> 20U)) {
      options.max_bytes = static_cast<std::uint64_t>(value) << 20U;
    }
  }
  return options;
}

std::optional<StageCacheOptions> StageCacheFromEnvironment() {
  return ParseStageCacheOptions(
      std::getenv(std::string(kStageCacheDirectoryVariable).c_str()),
      std::getenv(std::string(kStageCacheLimitVariable).c_str()));
}

StageCacheKey::StageCacheKey(std::string_view schema) {
  AddText("schema", schema);
}

void StageCacheKey::Add(std::string_view label,
                        std::span<const unsigned char> bytes) {
  const auto append_size = [this](std::uint64_t size) {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
      bytes_.push_back(static_cast<unsigned char>((size >> shift) & 0xFFU));
    }
  };
  append_size(label.size());
  bytes_.insert(bytes_.end(), label.begin(), label.end());
  append_size(bytes.size());
  bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
}

void StageCacheKey::AddText(std::string_view label, std::string_view text) {
  Add(label, TextBytes(text));
}

void StageCacheKey::AddInteger(std::string_view label, std::uint64_t value) {
  std::array<unsigned char, 8> bytes{};
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    bytes[index] = static_cast<unsigned char>((value >> (index * 8U)) & 0xFFU);
  }
  Add(label, bytes);
}

void StageCacheKey::AddFileIdentity(std::string_view label,
                                    const std::filesystem::path& path) {
  AddText(label, Utf8(path));
  std::error_code size_error;
  const std::uintmax_t size = std::filesystem::file_size(path, size_error);
  std::error_code time_error;
  const auto time = std::filesystem::last_write_time(path, time_error);
  if (size_error || time_error) {
    AddText("file", "missing");
    return;
  }
  AddInteger("size", static_cast<std::uint64_t>(size));
  AddInteger("mtime",
             static_cast<std::uint64_t>(time.time_since_epoch().count()));
}

void StageCacheKey::AddInventoryPhase(const ModelInventory& inventory,
                                      Phase phase) {
  AddText("phase", ToString(phase));
  std::set<std::filesystem::path> shards;
  for (const TensorDescriptor& tensor : inventory.tensors()) {
    if (tensor.phase != phase) {
      continue;
    }
    AddText("tensor", tensor.name);
    AddText("dtype", ToString(tensor.dtype));
    AddValues("shape", std::span<const std::uint64_t>(tensor.shape));
    AddText("shard", Utf8(tensor.shard));
    AddInteger("offset", tensor.file_offset);
    AddInteger("bytes", tensor.payload_bytes);
    shards.insert(tensor.shard);
  }
  for (const std::filesystem::path& shard : shards) {
    AddFileIdentity("shard_file", inventory.model_root() / shard);
  }
}

std::string StageCacheKey::Hex() const { return Sha256(bytes_); }

void StageChecksum::Mix(std::uint64_t word) {
  state_ ^= word;
  state_ *= 0x100000001B3ULL;
  state_ ^= state_ >> 29U;
}

void StageChecksum::Update(std::span<const unsigned char> bytes) {
  total_ += bytes.size();
  std::size_t offset = 0;
  while (tail_size_ != 0 && offset < bytes.size()) {
    tail_[tail_size_++] = bytes[offset++];
    if (tail_size_ == sizeof(tail_)) {
      std::uint64_t word = 0;
      std::memcpy(&word, tail_, sizeof(word));
      Mix(word);
      tail_size_ = 0;
    }
  }
  for (; offset + 8U <= bytes.size(); offset += 8U) {
    std::uint64_t word = 0;
    std::memcpy(&word, bytes.data() + offset, sizeof(word));
    Mix(word);
  }
  for (; offset < bytes.size(); ++offset) {
    tail_[tail_size_++] = bytes[offset];
  }
}

std::uint64_t StageChecksum::Digest() const {
  StageChecksum copy = *this;
  std::uint64_t word = 0;
  std::memcpy(&word, copy.tail_, copy.tail_size_);
  copy.Mix(word ^ (static_cast<std::uint64_t>(copy.tail_size_) << 56U));
  copy.Mix(copy.total_);
  return copy.state_;
}

bool AllFiniteBf16(std::span<const std::uint16_t> values) {
  return std::ranges::none_of(values, [](std::uint16_t value) {
    return (value & 0x7F80U) == 0x7F80U;
  });
}

std::filesystem::path StageCachePath(const StageCacheOptions& options,
                                     std::string_view kind,
                                     std::string_view key) {
  return options.directory / (std::string(kFilePrefix) + std::string(kind) +
                              "-" + std::string(key) +
                              std::string(kFileSuffix));
}

bool StoreStageArrays(const StageCacheOptions& options, std::string_view kind,
                      std::string_view key,
                      std::span<const std::vector<std::uint16_t>> arrays,
                      std::string* error) {
  if (!ValidName(kind) || !ValidName(key) || arrays.empty() ||
      arrays.size() > kMaximumArrays) {
    SetError(error, "invalid MiniMax H3 stage cache entry");
    return false;
  }
  std::uint64_t payload_bytes = 0;
  for (const auto& array : arrays) {
    payload_bytes += array.size() * sizeof(std::uint16_t);
  }
  if (payload_bytes > options.max_bytes) {
    SetError(error, "MiniMax H3 stage cache entry exceeds the cache limit");
    return false;
  }
  std::error_code filesystem_error;
  std::filesystem::create_directories(options.directory, filesystem_error);
  if (filesystem_error) {
    SetError(error, "cannot create MiniMax H3 stage cache " +
                        options.directory.string() + ": " +
                        filesystem_error.message());
    return false;
  }
  const std::filesystem::path path = StageCachePath(options, kind, key);
  if (std::filesystem::exists(path, filesystem_error)) {
    return true;
  }
  std::filesystem::path temporary = path;
  temporary += TemporarySuffix();
  bool written = false;
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (stream) {
      StageChecksum checksum;
      stream.write(kMagic.data(), kMagic.size());
      Put(&stream, &checksum, kFormatVersion);
      Put(&stream, &checksum, kByteOrderMark);
      Put(&stream, &checksum, static_cast<std::uint32_t>(kind.size()));
      PutBytes(&stream, &checksum, TextBytes(kind));
      Put(&stream, &checksum, static_cast<std::uint32_t>(key.size()));
      PutBytes(&stream, &checksum, TextBytes(key));
      Put(&stream, &checksum, static_cast<std::uint64_t>(arrays.size()));
      for (const auto& array : arrays) {
        Put(&stream, &checksum, static_cast<std::uint64_t>(array.size()));
      }
      for (const auto& array : arrays) {
        PutBytes(&stream, &checksum,
                 std::span(reinterpret_cast<const unsigned char*>(array.data()),
                           array.size() * sizeof(std::uint16_t)));
      }
      const std::uint64_t digest = checksum.Digest();
      stream.write(reinterpret_cast<const char*>(&digest), sizeof(digest));
      stream.flush();
      written = static_cast<bool>(stream);
    }
  }
  if (!written) {
    std::filesystem::remove(temporary, filesystem_error);
    SetError(error, "cannot write MiniMax H3 stage cache entry " +
                        temporary.string());
    return false;
  }
  std::filesystem::rename(temporary, path, filesystem_error);
  if (filesystem_error) {
    // A concurrent writer may have published the same key first.
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    if (!std::filesystem::exists(path, ignored)) {
      SetError(error, "cannot publish MiniMax H3 stage cache entry " +
                          path.string() + ": " + filesystem_error.message());
      return false;
    }
  }
  TrimStageCache(options);
  return true;
}

std::optional<std::vector<std::vector<std::uint16_t>>> LoadStageArrays(
    const StageCacheOptions& options, std::string_view kind,
    std::string_view key, std::string* error) {
  if (!ValidName(kind) || !ValidName(key)) {
    SetError(error, "invalid MiniMax H3 stage cache key");
    return std::nullopt;
  }
  const std::filesystem::path path = StageCachePath(options, kind, key);
  std::error_code filesystem_error;
  const std::uintmax_t file_bytes =
      std::filesystem::file_size(path, filesystem_error);
  if (filesystem_error) {
    return std::nullopt;
  }
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return std::nullopt;
  }
  // Windows refuses to remove a file that is still open.
  const auto reject = [&](std::string_view reason)
      -> std::optional<std::vector<std::vector<std::uint16_t>>> {
    stream.close();
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    SetError(error, "discarded MiniMax H3 stage cache entry " + path.string() +
                        ": " + std::string(reason));
    return std::nullopt;
  };
  std::vector<std::vector<std::uint16_t>> arrays;
  bool valid = false;
  {
    StageChecksum checksum;
    std::array<char, 8> magic{};
    std::uint32_t version = 0;
    std::uint16_t byte_order = 0;
    std::uint32_t kind_size = 0;
    std::uint32_t key_size = 0;
    std::uint64_t count = 0;
    if (!stream.read(magic.data(), magic.size()) || magic != kMagic ||
        !Get(&stream, &checksum, &version) || version != kFormatVersion ||
        !Get(&stream, &checksum, &byte_order) ||
        byte_order != kByteOrderMark ||
        !Get(&stream, &checksum, &kind_size) || kind_size != kind.size()) {
      return reject("bad header");
    }
    std::string stored_kind(kind_size, '\0');
    if (!GetBytes(&stream, &checksum, MutableBytes(&stored_kind)) ||
        stored_kind != kind || !Get(&stream, &checksum, &key_size) ||
        key_size != key.size()) {
      return reject("bad kind");
    }
    std::string stored_key(key_size, '\0');
    if (!GetBytes(&stream, &checksum, MutableBytes(&stored_key)) ||
        stored_key != key || !Get(&stream, &checksum, &count) || count == 0 ||
        count > kMaximumArrays) {
      return reject("bad key");
    }
    std::vector<std::uint64_t> sizes(static_cast<std::size_t>(count));
    std::uint64_t payload_bytes = 0;
    for (std::uint64_t& size : sizes) {
      if (!Get(&stream, &checksum, &size) ||
          size > file_bytes / sizeof(std::uint16_t)) {
        return reject("bad array size");
      }
      payload_bytes += size * sizeof(std::uint16_t);
    }
    const std::uint64_t header_bytes =
        kMagic.size() + sizeof(version) + sizeof(byte_order) +
        sizeof(kind_size) + kind_size + sizeof(key_size) + key_size +
        sizeof(count) + count * sizeof(std::uint64_t);
    if (header_bytes + payload_bytes + sizeof(std::uint64_t) != file_bytes) {
      return reject("bad file size");
    }
    arrays.resize(sizes.size());
    for (std::size_t index = 0; index < sizes.size(); ++index) {
      arrays[index].resize(static_cast<std::size_t>(sizes[index]));
      if (!GetBytes(&stream, &checksum,
                    std::span(reinterpret_cast<unsigned char*>(
                                  arrays[index].data()),
                              arrays[index].size() * sizeof(std::uint16_t)))) {
        return reject("short payload");
      }
    }
    std::uint64_t digest = 0;
    valid = static_cast<bool>(stream.read(reinterpret_cast<char*>(&digest),
                                          sizeof(digest))) &&
            digest == checksum.Digest();
  }
  stream.close();
  if (!valid) {
    return reject("checksum mismatch");
  }
  std::filesystem::last_write_time(
      path, std::filesystem::file_time_type::clock::now(), filesystem_error);
  return arrays;
}

namespace {

void TrimStageCacheUnchecked(const StageCacheOptions& options) {
  struct Entry {
    std::filesystem::path path;
    std::filesystem::file_time_type time;
    std::uint64_t bytes{0};
  };
  std::vector<Entry> entries;
  std::uint64_t total = 0;
  std::error_code filesystem_error;
  const auto now = std::filesystem::file_time_type::clock::now();
  for (const auto& item : std::filesystem::directory_iterator(
           options.directory, filesystem_error)) {
    const std::string name = Utf8(item.path().filename());
    if (!name.starts_with(kFilePrefix)) {
      continue;
    }
    std::error_code item_error;
    const auto time = item.last_write_time(item_error);
    if (item_error) {
      continue;
    }
    if (name.find(kTemporaryMarker) != std::string::npos) {
      if (now - time > std::chrono::hours(1)) {
        std::filesystem::remove(item.path(), item_error);
      }
      continue;
    }
    if (!name.ends_with(kFileSuffix)) {
      continue;
    }
    const std::uintmax_t bytes = item.file_size(item_error);
    if (item_error) {
      continue;
    }
    entries.push_back({item.path(), time, static_cast<std::uint64_t>(bytes)});
    total += static_cast<std::uint64_t>(bytes);
  }
  if (total <= options.max_bytes) {
    return;
  }
  std::ranges::sort(entries, [](const Entry& left, const Entry& right) {
    return left.time < right.time;
  });
  for (const Entry& entry : entries) {
    if (total <= options.max_bytes) {
      break;
    }
    std::error_code remove_error;
    if (std::filesystem::remove(entry.path, remove_error)) {
      total -= entry.bytes;
    }
  }
}

}  // namespace

void TrimStageCache(const StageCacheOptions& options) {
  // Best effort: a directory that cannot be listed is left alone.
  try {
    TrimStageCacheUnchecked(options);
  } catch (const std::exception&) {
  }
}

}  // namespace gufo::minimax_h3
