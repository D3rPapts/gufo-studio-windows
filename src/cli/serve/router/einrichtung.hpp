#ifndef GUFO_CLI_SERVE_ROUTER_EINRICHTUNG_HPP_
#define GUFO_CLI_SERVE_ROUTER_EINRICHTUNG_HPP_

// The setup assistant's router side (gufo-studio-windows 2.1, plan
// EINRICHTUNG-ROUTER-PLAN-2026-11-13): the endpoints under /router/setup/
// behind the gate (loopback or an admin session, else 404), the machine
// check, the building-block catalogue, the preset written from the blocks
// (a port of einrichtung/einrichten.py, which stays the reference and the
// unattended tool), the first admin account and the downloads from Hugging
// Face (download.hpp) and the self-test (selbsttest.hpp: one request per
// block against a router started from the new preset). `gufo router
// --einrichtung` runs without a preset: pages and this API only, nothing
// loads.

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/auth.hpp"
#include "src/cli/serve/router/download.hpp"
#include "src/cli/serve/router/memory.hpp"
#include "src/cli/serve/router/selbsttest.hpp"
#include "src/core/json.hpp"

namespace gufo::router::einrichtung {

/// Written into the preset header and einrichtung.json.
inline constexpr const char* kVersion = "0.2 (2026-11-14, gufo 2.1 router)";
/// The building blocks in preset order (REIHENFOLGE in einrichten.py).
inline constexpr const char* kBausteine[] = {
    "chat", "bilder", "stimme", "video", "musik", "3d", "werkzeuge",
    "zweiter-rechner"};
/// Graphics driver the .198 runs (32.0.32015.7029, Adrenalin 2026-10-18);
/// an older one is a hint, never a stop.
inline constexpr const char* kMinDriver = "32.0.32015.7029";

/// The folder with katalog.json and bausteine\: `override` when given,
/// else the first of <exe dir>\..\examples\einrichtung,
/// <exe dir>\einrichtung, <exe dir>\..\einrichtung and
/// <exe dir>\..\..\einrichtung (a build tree) that has a katalog.json;
/// "" when none.
[[nodiscard]] std::filesystem::path FindDir(
    const std::filesystem::path& exe, const std::filesystem::path& override);

/// An absolute local path the router may write under: `X:\...` on Windows
/// (no UNC, no forward-slash root), `/...` elsewhere, no `..` segment.
[[nodiscard]] bool IsSafePath(std::string_view path);

/// Parent folder of a Windows or POSIX path by text (no filesystem call,
/// so a test on Linux sees `C:\a\b` -> `C:\a` like einrichten.py).
[[nodiscard]] std::string ParentOf(std::string_view path);

/// Section -> key -> value of a preset text, comments and blanks skipped
/// (vergleich_preset.py's `lies`).
[[nodiscard]] std::map<std::string, std::map<std::string, std::string>>
PresetSections(std::string_view text);

/// What BuildPlan makes of the answers.
struct Plan {
  /// The answers with defaults applied (router.port, idle_seconds, ...).
  json::Value antworten;
  std::string preset_text;
  std::string start_cmd;
  std::string bericht;
  /// Chosen model ids in catalogue order, the partner twins, the notes
  /// ("x braucht y: mit aufgenommen").
  std::vector<std::string> modelle;
  std::vector<std::string> partner_modelle;
  std::vector<std::string> hinweise;
  /// Licences the answers have not confirmed (model ids).
  std::vector<std::string> lizenzen_offen;
  /// Files missing or of another size under pfade.modelle (catalogue
  /// paths) and their bytes; data files missing under pfade.daten.
  std::vector<std::string> dateien_fehlend;
  long long bytes_fehlend{0};
  long long bytes_da{0};
  std::vector<std::string> daten_fehlend;
  /// pfade.* keys whose gufo.exe is missing -> the models needing it.
  std::map<std::string, std::vector<std::string>> worker_fehlend;
  /// The filled placeholders (preset, pages, app, daten, zeit, ...).
  std::map<std::string, std::string> werte;
};

/// Checks the answers (the messages of einrichten.py's fehler()), picks
/// the models (required + RAM default + optional + dependencies), checks
/// the files, fills the templates of `bausteine_dir` and builds the
/// preset text, start-router.cmd and the report. Nothing is written.
[[nodiscard]] bool BuildPlan(const json::Value& antworten,
                             const json::Value& katalog,
                             const std::filesystem::path& bausteine_dir,
                             Plan* out, std::string* error);

/// Lines of an existing preset that belong to no "# baustein:" block
/// (einrichten.py's fremde_bloecke): kept at the end of a rewritten one.
[[nodiscard]] std::vector<std::string> ForeignBlocks(std::string_view old);

/// The download address of one catalogue file:
/// `<base>/<hf_repo>/resolve/<revision or main>/<path in the repo>`, the
/// path percent-encoded per segment. `hf_pfad` is the file's own entry
/// when the catalogue has one, else the catalogue path without the
/// source's folder prefix (backslashes become slashes).
[[nodiscard]] std::string DownloadUrl(std::string_view base,
                                      std::string_view hf_repo,
                                      std::string_view revision,
                                      std::string_view hf_pfad);

/// The files of the given models (catalogue order, each path once) with
/// their source and the state on disk under `ziel`: `vorhanden` when the
/// file is there with the catalogue size, `fehler`/no_source without a
/// Hugging Face source, else `wartet`. Unknown model ids are skipped and
/// named in *hinweise.
[[nodiscard]] std::vector<DownloadFile> PlanDownloads(
    const json::Value& katalog, const std::vector<std::string>& modelle,
    std::string_view ziel, std::string_view hf_base_url,
    std::vector<std::string>* hinweise);

/// One result line of the machine check.
struct CheckOptions {
  std::filesystem::path exe;  // gufo.exe of this router
  int own_port{0};            // the port this router listens on
  int wunsch_port{8081};      // the port the preset will use
  SystemMemory memory;
};

/// GET /router/setup/pruefen: RAM size, GPU, drives, worker folders and
/// the check list (ok | warn | err with a German text and what to do).
[[nodiscard]] json::Value MachineCheck(const CheckOptions& options);

/// What the service needs from the router.
struct ServiceOptions {
  std::filesystem::path dir;  // FindDir()
  std::filesystem::path exe;
  int port{0};
  /// --einrichtung without --models-preset: the users file is unknown
  /// until step 3 names the data folder (UserStore::Relocate).
  bool users_unbound{false};
  auth::Gate* gate{nullptr};
  /// Live memory; MakeSystemMemoryProbe() when empty.
  std::function<SystemMemory()> memory;
  /// Parses a preset file like the router at start (router::LoadPreset);
  /// "" = valid, else the message. Set by the router.
  std::function<std::string(const std::filesystem::path&)> check_preset;
  /// Where the files come from; a test points this at a local server.
  std::string hf_base_url{"https://huggingface.co"};
  /// Download tuning (tests shorten the retry pauses and allow http).
  std::vector<int> download_retry_wait_ms{5000, 10000, 20000, 40000, 60000};
  bool download_allow_http{false};
  int download_parallel{2};
  /// Space the start keeps free on top of the open bytes (plan: 2 GB).
  long long download_reserve_bytes{2LL << 30};
  /// Self-test target: "" = start the preset's router as a child (only
  /// with --einrichtung); a router running a preset sets its own loopback
  /// (review finding 1: a second router from the same preset would fight
  /// the first for worker ports and memory); a test sets a fake router.
  std::string selbsttest_base_url;
  /// "Name: value" headers of the calls to that target (the router's
  /// internal key for its own loopback).
  std::vector<std::string> selbsttest_headers;
  int selbsttest_start_timeout_s{180};
  int selbsttest_test_timeout_s{900};
  int selbsttest_poll_ms{2000};
};

/// /router/setup/{katalog,pruefen,preset,admin,downloads,selbsttest}; the
/// gate has already refused everyone who is neither loopback nor admin.
class SetupService {
public:
  explicit SetupService(ServiceOptions options);
  ~SetupService();
  SetupService(const SetupService&) = delete;
  SetupService& operator=(const SetupService&) = delete;

  [[nodiscard]] std::optional<server::HttpResponse> Handle(
      const server::HttpRequest& req);

  /// The folder with katalog.json and bausteine\ ("" = none found).
  [[nodiscard]] const std::filesystem::path& dir() const {
    return options_.dir;
  }

private:
  [[nodiscard]] bool LoadKatalog(json::Value* out, std::string* error) const;
  [[nodiscard]] server::HttpResponse Katalog() const;
  [[nodiscard]] server::HttpResponse Pruefen(const server::HttpRequest& req);
  [[nodiscard]] server::HttpResponse PruefenPfade(
      const server::HttpRequest& req);
  [[nodiscard]] server::HttpResponse Preset(const server::HttpRequest& req);
  [[nodiscard]] server::HttpResponse Admin(const server::HttpRequest& req);
  /// GET: the live state of the download job (empty when none).
  [[nodiscard]] server::HttpResponse Downloads();
  /// POST {aktion: planen|start|pause|weiter|abbrechen, modelle, ziel,
  /// hf_token?}.
  [[nodiscard]] server::HttpResponse DownloadsPost(
      const server::HttpRequest& req);
  /// `<ziel>\.einrichtung-downloads.json`: the status snapshot, so a
  /// restarted router (and the next `planen`) finds the digests again.
  void WriteDownloadState(const Downloader& job);
  /// GET: the state of the self-test (409 no_preset before any preset).
  [[nodiscard]] server::HttpResponse Selbsttest();
  /// POST {aktion: start|abbrechen, bausteine?, alle_modelle?, preset?,
  /// tests?: [{baustein, modell, art?}]}.
  [[nodiscard]] server::HttpResponse SelbsttestPost(
      const server::HttpRequest& req);
  /// Binds the gate's users file to <preset dir>\nutzer\nutzer.json once
  /// the paths are known (users_unbound only).
  void BindUsers(const std::filesystem::path& preset);

  ServiceOptions options_;
  std::mutex mutex_;
  std::unique_ptr<MemoryProbe> probe_;
  /// The preset path step 3 implied (pfade.preset or <daten>\gufo-router.ini).
  std::filesystem::path preset_path_;
  /// Download job: the state-file lock and target are declared before the
  /// downloader so its last on_change still finds them.
  std::mutex download_state_mutex_;
  std::filesystem::path download_ziel_;
  std::unique_ptr<Downloader> downloader_;
  /// Self-test job (one at a time); stopped before the downloader.
  std::unique_ptr<SelfTest> selbsttest_;
};

}  // namespace gufo::router::einrichtung

#endif  // GUFO_CLI_SERVE_ROUTER_EINRICHTUNG_HPP_
