#ifndef GUFO_CLI_SERVE_ROUTER_EINRICHTUNG_HPP_
#define GUFO_CLI_SERVE_ROUTER_EINRICHTUNG_HPP_

// The setup assistant's router side (gufo-studio-windows 2.1, plan
// EINRICHTUNG-ROUTER-PLAN-2026-11-13): the endpoints under /router/setup/
// behind the gate (loopback or an admin session, else 404), the machine
// check, the building-block catalogue, the preset written from the blocks
// (a port of einrichtung/einrichten.py, which stays the reference and the
// unattended tool) and the first admin account. The downloads (C3) and the
// self-test (C5) come later. `gufo router --einrichtung` runs without a
// preset: pages and this API only, nothing loads.

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
#include "src/cli/serve/router/memory.hpp"
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
};

/// /router/setup/{katalog,pruefen,preset,admin}; the gate has already
/// refused everyone who is neither loopback nor admin.
class SetupService {
public:
  explicit SetupService(ServiceOptions options);

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
  /// Binds the gate's users file to <preset dir>\nutzer\nutzer.json once
  /// the paths are known (users_unbound only).
  void BindUsers(const std::filesystem::path& preset);

  ServiceOptions options_;
  std::mutex mutex_;
  std::unique_ptr<MemoryProbe> probe_;
  /// The preset path step 3 implied (pfade.preset or <daten>\gufo-router.ini).
  std::filesystem::path preset_path_;
};

}  // namespace gufo::router::einrichtung

#endif  // GUFO_CLI_SERVE_ROUTER_EINRICHTUNG_HPP_
