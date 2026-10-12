#ifndef GUFO_CLI_BENCH_HPP_
#define GUFO_CLI_BENCH_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen38_flash_next/kv_cache_mode.hpp"

namespace gufo::cli {

struct BenchOptions {
  std::string model_path;
  std::vector<std::size_t> n_prompts{2048};
  std::vector<std::size_t> n_gens{128};
  std::vector<std::size_t> n_depths{0};
  std::vector<std::size_t> concurrency{1};
  std::size_t repetitions{1};
  std::size_t validate_prefill_tokens{0};
  /// Teacher-forced logit dump (`--logit-eval`): feed a text file through
  /// the decode arithmetic and write per-position target/top-64 log-probs
  /// for cross-build and cross-schedule bit comparisons.
  std::string logit_eval_path;
  std::string logit_out;
  /// Colon-separated schedules of comma-separated widths, `p`-prefixed for
  /// prompt arithmetic (`--logit-schedules 1:2,4:p8`).
  std::string logit_schedules{"1"};
  /// Continuation-cache boundary check (`--checkpoint-eval`): compare the
  /// next-token row after TEXT between a fresh prefill, a prefix+suffix
  /// split and a restored prefix checkpoint.
  std::string checkpoint_eval_path;
  /// Prefix depths for --checkpoint-eval.
  std::string checkpoint_boundaries{"600"};
  std::string speculative_backend{""};
  std::string mtp_model_path;
  std::string dflash_model_path;
  std::string draft_policy;
  std::string dspark_model_path;
  std::uint32_t draft_tokens{7};
  std::uint32_t min_draft_tokens{1};
  /// Flash-Next attention KV cache storage mode (`--kv-cache`).
  models::qwen38_flash_next::KvCacheMode kv_cache_mode{
      models::qwen38_flash_next::KvCacheMode::kF16};
  sampling::SamplingConfig sampling{.seed = 0};
  bool verbose{false};
};

void PrintBenchHelp(std::string_view program_name);
std::optional<BenchOptions> ParseBenchOptions(std::span<const char* const> args,
                                              std::string* error_msg = nullptr);
int RunBench(std::span<const char* const> args);

}  // namespace gufo::cli

#endif  // GUFO_CLI_BENCH_HPP_
