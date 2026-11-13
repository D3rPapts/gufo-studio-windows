#ifndef GUFO_CLI_SERVE_ROUTER_DOWNLOAD_HPP_
#define GUFO_CLI_SERVE_ROUTER_DOWNLOAD_HPP_

// File downloads for the setup assistant (gufo-studio-windows 2.1, plan
// EINRICHTUNG-ROUTER-PLAN-2026-11-13 section 4): libcurl, which gufo.exe
// already links, streaming into `<target>.part`, resumed with a Range
// request, SHA-256 computed while writing (crypto::Sha256Hasher), a HEAD
// size check against the catalogue, five retries with growing pauses,
// `parallel` worker threads over one queue. The Hugging Face token lives
// only in this object's memory and is never logged.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::router {

/// One file of a download job.
struct DownloadFile {
  enum class Status {
    kWartet,
    kLaeuft,
    kPause,
    kFertig,
    kFehler,
    kVorhanden,
    kPruefung,
    kAbgebrochen
  };

  // --- fixed by the plan
  std::string pfad;    // catalogue path, relative to the target folder
  std::string modell;  // model id the file belongs to
  std::filesystem::path ziel;  // the final file; `.part` is appended while loading
  std::string url;             // "" = no download source
  std::string hf_repo;
  std::string revision;
  long long bytes{0};   // expected size; 0 = unknown
  std::string sha256;   // expected, 64 lowercase hex, or ""

  // --- live
  Status status{Status::kWartet};
  long long geladen{0};
  /// "offen" | "ok" | "falsch" | "unbekannt"
  std::string sha256_status{"offen"};
  /// Where the compared digest came from: "katalog", "hf" (X-Linked-ETag
  /// when the catalogue has none) or "".
  std::string sha256_quelle;
  /// The digest of the loaded file (set after the check).
  std::string sha256_ist;
  std::string fehler;
  /// "" | "no_source" | "gated" | "not_found" | "size_mismatch" | "http" |
  /// "network" | "sha256" | "disk" | "write"
  std::string fehler_code;
  int versuche{0};
};

[[nodiscard]] const char* ToString(DownloadFile::Status status);

struct DownloaderOptions {
  int parallel{2};
  /// Pauses before retry 1..n (plan: 5/10/20/40/60 s); a test shortens them.
  std::vector<int> retry_wait_ms{5000, 10000, 20000, 40000, 60000};
  /// Only https unless a test points at a local http server.
  bool allow_http{false};
  std::string user_agent{"gufo-studio-windows-router/2.1 (einrichtung)"};
  long connect_timeout_s{30};
  /// A transfer slower than 1 byte/s for this long counts as a network
  /// error (and is retried with resume).
  long stall_timeout_s{120};
  /// Below this much free space on the target drive the job pauses itself.
  long long min_free_bytes{1LL << 30};
  /// Called (outside the lock, from the workers) after a status change;
  /// the service writes the state file from the given job's Status().
  std::function<void(const class Downloader&)> on_change;
};

/// A download job: the files, the worker threads and the live status.
/// Thread-safe; the destructor cancels and joins.
class Downloader {
public:
  Downloader(DownloaderOptions options, std::vector<DownloadFile> files);
  ~Downloader();
  Downloader(const Downloader&) = delete;
  Downloader& operator=(const Downloader&) = delete;

  /// Starts the workers over every file in `wartet`; `token` (optional)
  /// becomes the Bearer header for Hugging Face. Files already running
  /// keep running. Returns false when nothing is waiting.
  bool Start(const std::string& token);
  /// Stops the transfers after the current chunk; the .part files stay.
  void Pause();
  /// `pause`, `fehler` and `abgebrochen` back to `wartet`, workers started
  /// again (a new token replaces the old one when given).
  bool Resume(const std::string& token);
  /// Stops everything; waiting files become `abgebrochen`; the token is
  /// forgotten.
  void Cancel();

  [[nodiscard]] bool running() const;
  [[nodiscard]] bool paused() const;
  /// Snapshot of the files (copies).
  [[nodiscard]] std::vector<DownloadFile> Files() const;
  /// The GET /router/setup/downloads body (laeuft, pause, parallel,
  /// rate_bytes_s, gesamt_bytes, geladen_bytes, hinweis, dateien[]).
  [[nodiscard]] json::Value Status() const;
  /// Bytes still to load: for every waiting file its size minus the .part
  /// already on disk.
  [[nodiscard]] long long BytesOpen() const;

private:
  struct Transfer;
  void Worker();
  void JoinFinished();
  bool LoadOne(std::size_t index);
  /// One HEAD + GET attempt; false with *retry set when worth another try.
  bool Attempt(std::size_t index, Transfer* transfer, bool* retry);
  bool WaitRetry(int attempt);
  void Set(std::size_t index, DownloadFile::Status status,
           const std::string& code, const std::string& text);
  void Changed();
  void AddBytes(std::size_t index, long long n);
  [[nodiscard]] long long RateLocked() const;
  [[nodiscard]] bool TargetFull(const std::filesystem::path& file) const;

  DownloaderOptions options_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<DownloadFile> files_;
  std::vector<std::thread> threads_;
  int active_{0};
  bool pause_{false};
  bool cancel_{false};
  std::string token_;
  std::string hinweis_;
  /// Bytes per wall-clock second, the last 8 seconds (ring).
  struct Bucket {
    long long second{-1};
    long long bytes{0};
  };
  Bucket rate_[8];
  std::chrono::steady_clock::time_point epoch_{std::chrono::steady_clock::now()};
};

}  // namespace gufo::router

#endif  // GUFO_CLI_SERVE_ROUTER_DOWNLOAD_HPP_
