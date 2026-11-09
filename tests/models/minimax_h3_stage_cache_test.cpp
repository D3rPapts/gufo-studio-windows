#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "src/models/minimax_h3/stage_cache.hpp"

namespace {

using gufo::minimax_h3::AllFiniteBf16;
using gufo::minimax_h3::LoadStageArrays;
using gufo::minimax_h3::ParseStageCacheOptions;
using gufo::minimax_h3::StageCacheKey;
using gufo::minimax_h3::StageCacheOptions;
using gufo::minimax_h3::StageCachePath;
using gufo::minimax_h3::StageChecksum;
using gufo::minimax_h3::StoreStageArrays;
using gufo::minimax_h3::TrimStageCache;

int failures = 0;

#define CHECK(condition)                                          \
  do {                                                            \
    if (!(condition)) {                                           \
      std::cerr << "FAIL " << __FILE__ << ':' << __LINE__ << ": " \
                << #condition << '\n';                            \
      ++failures;                                                 \
    }                                                             \
  } while (false)

std::filesystem::path FreshDirectory(std::string_view name) {
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("gufo-h3-stage-cache-test-" + std::string(name) + "-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::remove_all(directory);
  return directory;
}

std::vector<std::uint16_t> Pattern(std::size_t size, std::uint16_t seed) {
  std::vector<std::uint16_t> values(size);
  std::uint32_t state = seed;
  for (std::uint16_t& value : values) {
    state = state * 1664525U + 1013904223U;
    value = static_cast<std::uint16_t>(state >> 16U);
  }
  return values;
}

void TestOptions() {
  CHECK(!ParseStageCacheOptions(nullptr, nullptr).has_value());
  CHECK(!ParseStageCacheOptions("", "100").has_value());
  const auto defaults = ParseStageCacheOptions("cache", nullptr);
  CHECK(defaults.has_value() &&
        defaults->max_bytes == gufo::minimax_h3::kDefaultStageCacheBytes);
  const auto limited = ParseStageCacheOptions("cache", "100");
  CHECK(limited.has_value() && limited->max_bytes == (100ULL << 20U));
  for (const char* invalid : {"0", "-5", "12x", "abc"}) {
    const auto parsed = ParseStageCacheOptions("cache", invalid);
    CHECK(parsed.has_value() &&
          parsed->max_bytes == gufo::minimax_h3::kDefaultStageCacheBytes);
  }
}

void TestKeys() {
  const auto key = [](std::string_view schema, std::string_view label,
                      std::string_view text) {
    StageCacheKey built(schema);
    built.AddText(label, text);
    return built.Hex();
  };
  CHECK(key("a", "x", "1") == key("a", "x", "1"));
  CHECK(key("a", "x", "1") != key("b", "x", "1"));
  CHECK(key("a", "x", "1") != key("a", "y", "1"));
  CHECK(key("a", "x", "1") != key("a", "x", "2"));
  CHECK(key("a", "x", "1").size() == 64);
  // Length prefixes keep field boundaries apart.
  StageCacheKey joined("s");
  joined.AddText("ab", "c");
  StageCacheKey split("s");
  split.AddText("a", "bc");
  CHECK(joined.Hex() != split.Hex());
  // Token ids and float bits enter as raw values.
  const std::vector<std::uint32_t> tokens = {1, 2, 3};
  const std::vector<std::uint32_t> other = {1, 2, 4};
  StageCacheKey first("s");
  first.AddValues("tokens", std::span<const std::uint32_t>(tokens));
  StageCacheKey second("s");
  second.AddValues("tokens", std::span<const std::uint32_t>(other));
  CHECK(first.Hex() != second.Hex());
  const std::vector<float> times = {0.0F, -0.0F};
  const std::vector<float> zeros = {0.0F, 0.0F};
  StageCacheKey signed_zero("s");
  signed_zero.AddValues("times", std::span<const float>(times));
  StageCacheKey plain_zero("s");
  plain_zero.AddValues("times", std::span<const float>(zeros));
  CHECK(signed_zero.Hex() != plain_zero.Hex());
  StageCacheKey missing("s");
  missing.AddFileIdentity("file", "does-not-exist.gufo");
  StageCacheKey missing_again("s");
  missing_again.AddFileIdentity("file", "does-not-exist.gufo");
  CHECK(missing.Hex() == missing_again.Hex());
}

void TestChecksum() {
  const std::vector<std::uint16_t> values = Pattern(1001, 7);
  const std::span<const unsigned char> bytes(
      reinterpret_cast<const unsigned char*>(values.data()),
      values.size() * sizeof(std::uint16_t));
  StageChecksum whole;
  whole.Update(bytes);
  // Any split of the same bytes gives the same digest.
  for (const std::size_t cut : {std::size_t{1}, std::size_t{3},
                                std::size_t{8}, std::size_t{1001}}) {
    StageChecksum pieces;
    pieces.Update(bytes.subspan(0, cut));
    pieces.Update(bytes.subspan(cut));
    CHECK(pieces.Digest() == whole.Digest());
  }
  std::vector<std::uint16_t> changed = values;
  changed[500] ^= 1U;
  StageChecksum flipped;
  flipped.Update(std::span(reinterpret_cast<const unsigned char*>(
                               changed.data()),
                           changed.size() * sizeof(std::uint16_t)));
  CHECK(flipped.Digest() != whole.Digest());
  StageChecksum shorter;
  shorter.Update(bytes.subspan(0, bytes.size() - 2));
  CHECK(shorter.Digest() != whole.Digest());
}

void TestFinite() {
  // 1.0, -2.0, largest finite, smallest subnormal, zero.
  const std::vector<std::uint16_t> finite = {0x3F80, 0xC000, 0x7F7F, 0x0001,
                                             0x0000};
  CHECK(AllFiniteBf16(finite));
  CHECK(AllFiniteBf16(std::span<const std::uint16_t>()));
  // +inf, -inf, quiet NaN, negative NaN with payload.
  const std::vector<std::uint16_t> non_finite = {0x7F80, 0xFF80, 0x7FC0,
                                                 0xFFC1};
  for (const std::uint16_t bad : non_finite) {
    std::vector<std::uint16_t> values = finite;
    values[2] = bad;
    CHECK(!AllFiniteBf16(values));
  }
}

void TestRoundTrip() {
  StageCacheOptions options;
  options.directory = FreshDirectory("round-trip");
  const std::string key = StageCacheKey("round-trip").Hex();
  CHECK(!LoadStageArrays(options, "adaln", key).has_value());
  const std::vector<std::vector<std::uint16_t>> arrays = {
      Pattern(5, 1), Pattern(1, 2), Pattern(4099, 3)};
  std::string error;
  CHECK(StoreStageArrays(options, "adaln", key, arrays, &error));
  CHECK(error.empty());
  const auto loaded = LoadStageArrays(options, "adaln", key, &error);
  CHECK(loaded.has_value() && *loaded == arrays);
  // The kind is part of the entry.
  CHECK(!LoadStageArrays(options, "prompt", key).has_value());
  // A second store of the same key keeps the first entry.
  CHECK(StoreStageArrays(options, "adaln", key,
                         std::vector<std::vector<std::uint16_t>>{Pattern(2, 9)},
                         &error));
  const auto kept = LoadStageArrays(options, "adaln", key);
  CHECK(kept.has_value() && *kept == arrays);
  // Invalid names and empty entries are refused.
  CHECK(!StoreStageArrays(options, "../x", key, arrays));
  CHECK(!StoreStageArrays(options, "adaln", "KEY", arrays));
  CHECK(!StoreStageArrays(options, "adaln", key,
                          std::span<const std::vector<std::uint16_t>>()));
  std::filesystem::remove_all(options.directory);
}

void TestDamagedEntries() {
  StageCacheOptions options;
  options.directory = FreshDirectory("damaged");
  const std::string key = StageCacheKey("damaged").Hex();
  const std::vector<std::vector<std::uint16_t>> arrays = {Pattern(64, 4)};
  CHECK(StoreStageArrays(options, "prompt", key, arrays));
  const std::filesystem::path path = StageCachePath(options, "prompt", key);
  {
    // Flip one payload byte.
    std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
    stream.seekp(-20, std::ios::end);
    char byte = 0;
    stream.seekg(-20, std::ios::end);
    stream.read(&byte, 1);
    byte = static_cast<char>(byte ^ 0x40);
    stream.seekp(-20, std::ios::end);
    stream.write(&byte, 1);
  }
  std::string error;
  CHECK(!LoadStageArrays(options, "prompt", key, &error).has_value());
  CHECK(!error.empty());
  CHECK(!std::filesystem::exists(path));
  // Truncated entries are discarded too.
  CHECK(StoreStageArrays(options, "prompt", key, arrays));
  std::filesystem::resize_file(path, std::filesystem::file_size(path) - 3);
  error.clear();
  CHECK(!LoadStageArrays(options, "prompt", key, &error).has_value());
  CHECK(!error.empty());
  CHECK(!std::filesystem::exists(path));
  std::filesystem::remove_all(options.directory);
}

void TestTrim() {
  StageCacheOptions options;
  options.directory = FreshDirectory("trim");
  // Each entry holds 4 KiB of payload; the limit fits two of them.
  options.max_bytes = 2U * 4096U + 1024U;
  const std::vector<std::vector<std::uint16_t>> arrays = {Pattern(2048, 5)};
  std::vector<std::string> keys;
  for (int index = 0; index < 3; ++index) {
    keys.push_back(StageCacheKey("trim-" + std::to_string(index)).Hex());
  }
  const auto age = [&options](const std::string& key, int seconds) {
    std::filesystem::last_write_time(
        StageCachePath(options, "adaln", key),
        std::filesystem::file_time_type::clock::now() -
            std::chrono::seconds(seconds));
  };
  CHECK(StoreStageArrays(options, "adaln", keys[0], arrays));
  age(keys[0], 300);
  CHECK(StoreStageArrays(options, "adaln", keys[1], arrays));
  age(keys[1], 200);
  // A hit refreshes the entry, so the never-read keys[1] is now the oldest.
  CHECK(LoadStageArrays(options, "adaln", keys[0]).has_value());
  CHECK(StoreStageArrays(options, "adaln", keys[2], arrays));
  CHECK(std::filesystem::exists(StageCachePath(options, "adaln", keys[0])));
  CHECK(!std::filesystem::exists(StageCachePath(options, "adaln", keys[1])));
  CHECK(std::filesystem::exists(StageCachePath(options, "adaln", keys[2])));
  // Stale temporary files are removed, other files are left alone.
  const std::filesystem::path stale =
      options.directory / "h3-adaln-x.bin.tmp-1-2-3";
  const std::filesystem::path foreign = options.directory / "notes.txt";
  std::ofstream(stale) << "partial";
  std::ofstream(foreign) << "keep";
  std::filesystem::last_write_time(
      stale,
      std::filesystem::file_time_type::clock::now() - std::chrono::hours(2));
  TrimStageCache(options);
  CHECK(!std::filesystem::exists(stale));
  CHECK(std::filesystem::exists(foreign));
  // An entry larger than the whole cache is refused.
  options.max_bytes = 1024;
  std::string error;
  CHECK(!StoreStageArrays(options, "adaln", StageCacheKey("big").Hex(),
                          arrays, &error));
  CHECK(!error.empty());
  std::filesystem::remove_all(options.directory);
}

}  // namespace

int main() {
  TestOptions();
  TestKeys();
  TestChecksum();
  TestFinite();
  TestRoundTrip();
  TestDamagedEntries();
  TestTrim();
  if (failures != 0) {
    std::cerr << failures << " MiniMax H3 stage cache checks failed\n";
    return 1;
  }
  std::cout << "MiniMax H3 stage cache checks passed\n";
  return 0;
}
