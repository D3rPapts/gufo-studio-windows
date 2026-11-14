#ifndef GUFO_CLI_SERVE_ROUTER_SELBSTTEST_HPP_
#define GUFO_CLI_SERVE_ROUTER_SELBSTTEST_HPP_

// The setup assistant's self-test (gufo-studio-windows 2.1, plan
// EINRICHTUNG-ROUTER-PLAN-2026-11-13 sections 2.5 and 6): one small request
// per chosen building block against a router that runs the new preset,
// one after the other (one model in memory), the requests of
// D:\claude\gufo-arbeit\modell-pruefung\modell_pruefung.py. The setup
// router (`--einrichtung`, no models) starts the production router from
// the preset's "# start:" line as a child on a free loopback port, asks it
// over HTTP and stops it at the end; a router that already runs a preset
// is asked directly. Results (text, PNG, WAV, MP4, GLB) land in
// <daten>\jobs\selbsttest\, the report in einrichtung.json next to the
// preset.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::router {
class Worker;
}

namespace gufo::router::einrichtung {

/// One check of the self-test.
struct SelfTestCase {
  std::string baustein;
  std::string modell;
  /// chat | bild | stimme | video | musik | 3d | upscale | partner |
  /// keiner (no request known for this model: ASR, RIFE, LoRA, ...).
  std::string art;
  /// wartet | laeuft | ok | fehler | uebersprungen | abgebrochen
  std::string status{"wartet"};
  double dauer_s{0};
  /// The filtered cause when red; a note otherwise.
  std::string hinweis;
  /// The last error line the target gave (HTTP body, trimmed).
  std::string log;
  /// The result file written (relative to the results folder).
  std::string datei;
};

/// The request a model of a block gets; "keiner" when the self-test has
/// none for it.
[[nodiscard]] std::string TestKind(std::string_view baustein,
                                   std::string_view modell);

/// The cases for the chosen blocks: per block the first required model
/// of the catalogue that is in `modelle` and has a request (bilder prefers
/// the turbo model), every testable chosen model with `alle_modelle`.
/// "zweiter-rechner" becomes one partner case when `partner_url` is set.
[[nodiscard]] std::vector<SelfTestCase> PlanSelfTest(
    const json::Value& katalog, const std::vector<std::string>& bausteine,
    const std::vector<std::string>& modelle, bool alle_modelle,
    std::string_view partner_url);

/// The router's command line from the preset header.
struct StartLine {
  std::string program;
  /// Without `--host X` and `--port N`; the self-test adds its own.
  std::vector<std::string> args;
  std::vector<std::pair<std::string, std::string>> env;
};

/// Splits a command line: double quotes group, no escapes.
[[nodiscard]] std::vector<std::string> SplitArgs(std::string_view line);

/// Reads "# start:", "# start-opt:" and "# env:" of a preset text (the
/// lines einrichten.py / BuildPlan write from bausteine\router.ini). False
/// without a "# start:" line.
[[nodiscard]] bool ParseStartLine(std::string_view preset_text,
                                  StartLine* out);

/// A 64x64 RGBA test picture (orange disc on transparent ground), PNG
/// bytes: the input of the 3D and upscale checks.
[[nodiscard]] std::string TestImagePng();

struct SelfTestOptions {
  std::filesystem::path preset;
  /// Results go to <daten>\jobs\selbsttest.
  std::filesystem::path daten;
  /// The router program when the preset has no "# start:" line.
  std::filesystem::path exe;
  /// The target; "" = start the preset's router as a child.
  std::string base_url;
  std::string partner_url;
  std::vector<SelfTestCase> tests;
  int start_timeout_s{180};
  /// Per case, loading included (plan: 15 min).
  int test_timeout_s{900};
  int poll_ms{2000};
  /// POST /models/unload after every case, so the next one has room.
  bool unload_after{true};
};

/// Runs the cases in its own thread. The destructor cancels, stops the
/// child router and joins.
class SelfTest {
public:
  explicit SelfTest(SelfTestOptions options);
  ~SelfTest();
  SelfTest(const SelfTest&) = delete;
  SelfTest& operator=(const SelfTest&) = delete;

  /// False when nothing is to test or it already runs.
  bool Start();
  void Cancel();
  [[nodiscard]] bool running() const;
  /// GET /router/setup/selbsttest: laeuft, abgebrochen, fertig, ziel,
  /// ordner, kind {port, pid}, ok, fehler, tests[].
  [[nodiscard]] json::Value Status() const;
  /// The results folder (<daten>\jobs\selbsttest).
  [[nodiscard]] std::filesystem::path ordner() const;

private:
  struct Reply {
    bool ok{false};        // transport ok
    long status{0};
    std::string body;
    std::string content_type;
    std::string error;     // transport error text
  };
  void Run();
  bool StartChild(std::string* error);
  void StopChild();
  bool WaitReady(std::string* error);
  void RunCase(std::size_t index);
  bool Chat(const SelfTestCase& c, std::string* hint, std::string* log,
            std::string* out, std::string* ext);
  bool Image(const SelfTestCase& c, std::string* hint, std::string* log,
             std::string* out, std::string* ext);
  bool Speech(const SelfTestCase& c, std::string* hint, std::string* log,
              std::string* out, std::string* ext);
  bool Music(const SelfTestCase& c, std::string* hint, std::string* log,
             std::string* out, std::string* ext);
  bool Job(const SelfTestCase& c, const std::string& path,
           const json::Value& body, std::string* hint, std::string* log,
           std::string* out, std::string* ext);
  bool Upscale(const SelfTestCase& c, std::string* hint, std::string* log,
               std::string* out, std::string* ext);
  bool Partner(std::string* hint, std::string* log);
  void Unload(const std::string& model);
  Reply Request(const std::string& method, const std::string& url,
                const std::string& body, long timeout_s);
  Reply Get(const std::string& path, long timeout_s);
  Reply Post(const std::string& path, const json::Value& body,
             long timeout_s);
  /// "HTTP <code>: <message>" with the memory special case, and the log.
  static void Explain(const Reply& r, std::string* hint, std::string* log);
  void Set(std::size_t index, const std::string& status,
           const std::string& hint, const std::string& log,
           const std::string& datei, double dauer_s);
  void WriteReport();
  bool Cancelled() const { return cancel_.load(); }

  SelfTestOptions options_;
  mutable std::mutex mutex_;
  std::vector<SelfTestCase> tests_;
  std::thread thread_;
  std::atomic<bool> cancel_{false};
  bool running_{false};
  bool finished_{false};
  std::string base_url_;
  std::string hinweis_;
  std::string zeit_start_;
  std::string zeit_ende_;
  int child_port_{0};
  std::uint32_t child_pid_{0};
  std::unique_ptr<Worker> child_;
  std::chrono::steady_clock::time_point started_;
};

}  // namespace gufo::router::einrichtung

#endif  // GUFO_CLI_SERVE_ROUTER_SELBSTTEST_HPP_
