// The setup assistant's router side (src/cli/serve/router/einrichtung.*):
// the gate rules for /router/setup/* and /einrichtung/, the machine check,
// the catalogue, the preset built from the building blocks (against the
// real einrichtung/ folder and, when python3 is known, against
// einrichten.py --plan section by section like vergleich_preset.py), the
// foreign-block keeper, the service's endpoints with a temp folder and the
// self-test against a fake router.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/auth.hpp"
#include "src/cli/serve/router/einrichtung.hpp"
#include "src/cli/serve/router/preset.hpp"
#include "src/core/base64.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"

namespace {

namespace auth = gufo::router::auth;
namespace einrichtung = gufo::router::einrichtung;
namespace json = gufo::json;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;

int failures = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond \
                << "\n";                                                  \
      ++failures;                                                         \
    }                                                                     \
  } while (0)

#define CHECK_EQ(a, b)                                                  \
  do {                                                                  \
    const auto va = (a);                                                \
    const auto vb = (b);                                                \
    if (!(va == vb)) {                                                  \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #a \
                << " == " #b << " (" << va << " vs " << vb << ")\n";    \
      ++failures;                                                       \
    }                                                                   \
  } while (0)

constexpr long long kTestRounds = 1000;

#ifndef GUFO_EINRICHTUNG_DIR
#define GUFO_EINRICHTUNG_DIR "einrichtung"
#endif

const std::filesystem::path kDir = GUFO_EINRICHTUNG_DIR;

std::filesystem::path TempDir() {
  const auto base = std::filesystem::temp_directory_path() /
                    ("gufo-einrichtung-test-" + auth::RandomHex(4));
  std::filesystem::create_directories(base);
  return base;
}

std::string ReadFile(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  std::string text((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  if (text.starts_with("\xEF\xBB\xBF"))
    text.erase(0, 3);
  return text;
}

void WriteFile(const std::filesystem::path& file, const std::string& text) {
  std::filesystem::create_directories(file.parent_path());
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  out << text;
}

json::Value LoadJson(const std::filesystem::path& file) {
  return json::parse(ReadFile(file));
}

HttpRequest Req(std::string method, std::string path, std::string client,
                std::string body = {},
                std::vector<std::pair<std::string, std::string>> headers = {}) {
  HttpRequest req;
  req.method = std::move(method);
  req.path = std::move(path);
  if (const auto q = req.path.find('?'); q != std::string::npos) {
    req.query = req.path.substr(q + 1);
    req.path.resize(q);
  }
  req.client_id = std::move(client);
  req.body = std::move(body);
  req.headers = std::move(headers);
  if (!req.body.empty() && req.header("Content-Type").empty())
    req.headers.emplace_back("Content-Type", "application/json");
  return req;
}

std::string ErrorCode(const HttpResponse& r) {
  try {
    const json::Value body = json::parse(r.body);
    if (const json::Value* e = body.find("error"))
      return e->member_str("code");
  } catch (const std::exception&) {
  }
  return "";
}

json::Value Body(const HttpResponse& r) {
  try {
    return json::parse(r.body);
  } catch (const std::exception&) {
    return json::Value();
  }
}

std::string Header(const HttpResponse& r, std::string_view name) {
  for (const auto& [k, v] : r.headers)
    if (k == name)
      return v;
  return "";
}

/// A gate with a users file in a temp folder and a `next` that records.
struct Fixture {
  std::filesystem::path dir = TempDir();
  std::shared_ptr<auth::UserStore> store =
      std::make_shared<auth::UserStore>(dir / "nutzer" / "nutzer.json",
                                        kTestRounds);
  std::unique_ptr<auth::Gate> gate;
  std::vector<std::string> seen;
  std::function<HttpResponse(const HttpRequest&)> next;

  explicit Fixture(auth::Mode mode = auth::Mode::kLan) {
    auth::Settings settings;
    settings.mode = mode;
    settings.users_file = store->path();
    std::string error;
    CHECK(store->Load(&error));
    gate = std::make_unique<auth::Gate>(settings, store, auth::Gate::Hooks{});
    next = [this](const HttpRequest& req) {
      seen.push_back(req.path);
      return HttpResponse{
          .status = 200, .reason = "OK", .body = "{}", .headers = {}};
    };
  }
  ~Fixture() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
  HttpResponse Serve(const HttpRequest& req) { return gate->Serve(req, next); }
  void AddUser(const std::string& name, const std::string& rolle,
               const std::string& password) {
    auth::User user;
    user.name = name;
    user.rolle = rolle;
    std::string error;
    CHECK(store->Put(user, &error));
    CHECK(store->SetPassword(name, password, &error));
  }
  std::string Login(const std::string& name, const std::string& password,
                    const std::string& client = "192.168.1.5") {
    const HttpResponse r = Serve(Req(
        "POST", "/login", client,
        "{\"name\":\"" + name + "\",\"passwort\":\"" + password + "\"}"));
    CHECK_EQ(r.status, 200);
    const std::string set = Header(r, "Set-Cookie");
    return set.substr(0, set.find(';'));
  }
};

/// LoadPreset checks that every `command` exists: the test's answers point
/// the app and worker folders at `tmp` with an empty gufo.exe each.
void PointWorkersAt(json::Value* antworten, const std::filesystem::path& tmp) {
  json::Value& pfade = (*antworten)["pfade"];
  for (const char* key : {"app", "app_35b", "app_3d", "app_moss", "app_d",
                          "app_trellis2"}) {
    const std::filesystem::path dir = tmp / key;
    WriteFile(dir / "gufo.exe", "");
    pfade[key] = dir.string();
  }
#if !GUFO_ROUTER_FORK_SERVE
  // A build without `gufo serve omni` refuses a local [omni/...] section.
  json::Value weg = json::Value::array();
  if (const json::Value* old = antworten->find("modelle_weglassen"))
    for (const json::Value& id : old->items())
      weg.push_back(id);
  weg.push_back("qwen3-omni-captioner");
  (*antworten)["modelle_weglassen"] = std::move(weg);
#endif
}

std::string CheckStatus(const json::Value& check_list, std::string_view id) {
  for (const json::Value& c : check_list.items())
    if (c.member_str("id") == id)
      return c.member_str("status");
  return "(missing)";
}

}  // namespace

// ---------------------------------------------------------------------------

void TestClassifyAndGate() {
  CHECK(auth::Classify(Req("GET", "/router/setup/katalog", "x")).kind ==
        auth::Route::Kind::kSetup);
  CHECK(auth::Classify(Req("POST", "/router/setup/admin", "x")).kind ==
        auth::Route::Kind::kSetup);
  CHECK(auth::Classify(Req("GET", "/einrichtung/", "x")).kind ==
        auth::Route::Kind::kAdminPage);
  CHECK(auth::Classify(Req("GET", "/einrichtung", "x")).kind ==
        auth::Route::Kind::kAdminPage);
  CHECK(auth::Classify(Req("GET", "/router/setupx", "x")).kind !=
        auth::Route::Kind::kSetup);

  // auth = lan: loopback is admin -> through; the LAN sees 404, never 401.
  Fixture f(auth::Mode::kLan);
  HttpResponse r = f.Serve(Req("GET", "/router/setup/katalog", "127.0.0.1"));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(f.seen.size(), std::size_t{1});
  r = f.Serve(Req("GET", "/router/setup/katalog", "192.168.1.5"));
  CHECK_EQ(r.status, 404);
  r = f.Serve(Req("GET", "/einrichtung/", "192.168.1.5"));
  CHECK_EQ(r.status, 404);
  r = f.Serve(Req("GET", "/einrichtung/", "::1"));
  CHECK_EQ(r.status, 200);
  // /router/me says the API exists, also before the first admin.
  r = f.Serve(Req("GET", "/router/me", "192.168.1.5"));
  CHECK_EQ(r.status, 401);
  CHECK(Body(r).find("setup_api") != nullptr &&
        Body(r).find("setup_api")->as_bool());
  CHECK(Body(r).find("setup")->as_bool());
  r = f.Serve(Req("GET", "/router/me", "127.0.0.1"));
  CHECK_EQ(r.status, 200);
  CHECK(Body(r).find("setup_api")->as_bool());
  // A browser page from another site must not write through loopback.
  r = f.Serve(Req("POST", "/router/setup/preset", "127.0.0.1", "{}",
                  {{"Content-Type", "application/json"},
                   {"Origin", "http://evil.example"},
                   {"Host", "127.0.0.1:8081"}}));
  CHECK_EQ(r.status, 403);
  CHECK_EQ(ErrorCode(r), "csrf_rejected");
  r = f.Serve(Req("POST", "/router/setup/preset", "127.0.0.1", "{}",
                  {{"Content-Type", "application/json"},
                   {"Origin", "http://127.0.0.1:8081"},
                   {"Host", "127.0.0.1:8081"}}));
  CHECK_EQ(r.status, 200);
  // JSON only.
  r = f.Serve(Req("POST", "/router/setup/preset", "127.0.0.1", "x=1",
                  {{"Content-Type", "text/plain"}}));
  CHECK_EQ(r.status, 415);
  CHECK_EQ(ErrorCode(r), "json_required");
  // An admin session from the LAN passes, a user's session sees 404.
  f.AddUser("chef", "admin", "geheim");
  f.AddUser("gast", "nutzer", "geheim");
  const std::string admin = f.Login("chef", "geheim");
  const std::string gast = f.Login("gast", "geheim");
  r = f.Serve(Req("GET", "/router/setup/katalog", "192.168.1.5", "",
                  {{"Cookie", admin}}));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("GET", "/router/setup/katalog", "192.168.1.5", "",
                  {{"Cookie", gast}}));
  CHECK_EQ(r.status, 404);
  r = f.Serve(Req("GET", "/einrichtung/", "192.168.1.5", "",
                  {{"Cookie", gast}}));
  CHECK_EQ(r.status, 404);

  // auth = off (the --einrichtung default): loopback through, the LAN
  // 404, and a cross-site POST through loopback refused even here.
  Fixture off(auth::Mode::kOff);
  r = off.Serve(Req("GET", "/router/setup/pruefen", "127.0.0.1"));
  CHECK_EQ(r.status, 200);
  r = off.Serve(Req("GET", "/router/setup/pruefen", "192.168.1.5"));
  CHECK_EQ(r.status, 404);
  r = off.Serve(Req("POST", "/router/setup/admin", "127.0.0.1", "{}",
                    {{"Content-Type", "application/json"},
                     {"Sec-Fetch-Site", "cross-site"}}));
  CHECK_EQ(r.status, 403);
  CHECK_EQ(ErrorCode(r), "csrf_rejected");
  r = off.Serve(Req("POST", "/router/setup/admin", "127.0.0.1", "{}",
                    {{"Content-Type", "application/json"},
                     {"Sec-Fetch-Site", "same-origin"}}));
  CHECK_EQ(r.status, 200);
}

void TestHelpers() {
#ifdef _WIN32
  CHECK(einrichtung::IsSafePath("C:\\gufo\\modelle"));
  CHECK(einrichtung::IsSafePath("d:/gufo"));
  CHECK(!einrichtung::IsSafePath("\\\\server\\share"));
  CHECK(!einrichtung::IsSafePath("gufo\\modelle"));
  CHECK(!einrichtung::IsSafePath("C:\\gufo\\..\\windows"));
  CHECK(!einrichtung::IsSafePath(""));
#else
  CHECK(einrichtung::IsSafePath("/srv/gufo"));
  CHECK(!einrichtung::IsSafePath("srv/gufo"));
  CHECK(!einrichtung::IsSafePath("/srv/../etc"));
#endif
  CHECK_EQ(einrichtung::ParentOf("C:\\gufo\\2.0.3\\router"), "C:\\gufo\\2.0.3");
  CHECK_EQ(einrichtung::ParentOf("C:\\gufo\\"), "C:\\");
  CHECK_EQ(einrichtung::ParentOf("C:\\gufo"), "C:\\");
  CHECK_EQ(einrichtung::ParentOf("/srv/gufo/x"), "/srv/gufo");

  const auto sections = einrichtung::PresetSections(
      "# c\n[a/b]\nkey = v1\n\n[router]\nauth = lan\n# x = y\n");
  CHECK_EQ(sections.size(), std::size_t{2});
  CHECK_EQ(sections.at("a/b").at("key"), "v1");
  CHECK_EQ(sections.at("router").at("auth"), "lan");

  const std::vector<std::string> fremd = einrichtung::ForeignBlocks(
      "# head\n"
      "# baustein: router\n# start: x\n\n"
      "# baustein: chat\n# modell: a\n[external/a]\nmodel = x\n\n"
      "[external/mine]\ncommand = y\n\n"
      "# --- kept from the previous preset (not managed by einrichten.py) ---\n"
      "[external/old]\nmodel = z\n");
  const std::string joined = [&] {
    std::string out;
    for (const std::string& line : fremd)
      out += line + "\n";
    return out;
  }();
  CHECK(joined.find("[external/mine]") != std::string::npos);
  CHECK(joined.find("[external/old]") != std::string::npos);
  CHECK(joined.find("[external/a]") == std::string::npos);
  CHECK(joined.find("kept from") == std::string::npos);
  CHECK(joined.starts_with("# head\n"));
}

void TestMachineCheck() {
  einrichtung::CheckOptions o;
  o.exe = std::filesystem::path("C:\\gufo\\2.1\\bin\\gufo.exe");
  o.own_port = 8098;
  o.wunsch_port = 8098;
  o.memory.host_valid = true;
  o.memory.host_total = static_cast<std::int64_t>(15.6 * 1024 * 1024 * 1024);
  o.memory.host_available = o.memory.host_total / 2;
  o.memory.gpu_valid = true;
  o.memory.gpu_dedicated_total = 48LL * 1024 * 1024 * 1024;
  o.memory.commit_valid = true;
  o.memory.commit_limit = 64LL * 1024 * 1024 * 1024;
  json::Value out = einrichtung::MachineCheck(o);
  CHECK_EQ(out.member_double("ram_gb"), 64.0);
  CHECK(out.find("worker") != nullptr &&
        out.find("worker")->find("app_3d") != nullptr);
  CHECK(out.find("worker")->find("app_35b") != nullptr);
  CHECK(out.find("gpu") != nullptr && out.find("laufwerke") != nullptr);
  const json::Value& checks = *out.find("pruefungen");
  for (const char* id : {"gpu", "treiber", "ram", "zuteilung", "platte",
                         "windows", "ffmpeg", "tdr", "port", "auslagerung"}) {
    const std::string status = CheckStatus(checks, id);
    CHECK(status == "ok" || status == "warn" || status == "err");
    if (!(status == "ok" || status == "warn" || status == "err"))
      std::cerr << "  check " << id << ": " << status << "\n";
  }
  CHECK_EQ(CheckStatus(checks, "ram"), "ok");
  CHECK_EQ(CheckStatus(checks, "zuteilung"), "ok");
  CHECK_EQ(CheckStatus(checks, "port"), "ok");  // the own port
  CHECK_EQ(CheckStatus(checks, "auslagerung"), "ok");

  o.memory.gpu_dedicated_total = 96LL * 1024 * 1024 * 1024;
  o.memory.host_total = 32LL * 1024 * 1024 * 1024;
  o.memory.commit_limit = 40LL * 1024 * 1024 * 1024;
  out = einrichtung::MachineCheck(o);
  CHECK_EQ(out.member_double("ram_gb"), 128.0);
  CHECK_EQ(CheckStatus(*out.find("pruefungen"), "zuteilung"), "ok");
  CHECK_EQ(CheckStatus(*out.find("pruefungen"), "auslagerung"), "warn");
  o.memory.gpu_dedicated_total = 32LL * 1024 * 1024 * 1024;
  out = einrichtung::MachineCheck(o);
  CHECK_EQ(out.member_double("ram_gb"), 64.0);
  CHECK_EQ(CheckStatus(*out.find("pruefungen"), "zuteilung"), "warn");
}

void TestPlan() {
  const json::Value katalog = LoadJson(kDir / "katalog.json");
  json::Value beispiel = LoadJson(kDir / "antworten-beispiel.json");
  CHECK(katalog.find("bausteine") != nullptr);
  const std::filesystem::path tmp = TempDir();
  PointWorkersAt(&beispiel, tmp);
  // The same answers for einrichten.py below.
  WriteFile(tmp / "antworten.json", beispiel.tojson());
  einrichtung::Plan plan;
  std::string error;
  CHECK(einrichtung::BuildPlan(beispiel, katalog, kDir / "bausteine", &plan,
                               &error));
  if (!error.empty())
    std::cerr << "  BuildPlan: " << error << "\n";
  CHECK(!plan.modelle.empty());
  CHECK(plan.preset_text.find("[external/qwen3.8-27b]") != std::string::npos);
  CHECK(plan.preset_text.find("[external/qwen3.8-flash-next]") !=
        std::string::npos);  // vorgabe_ram 128
  CHECK(plan.preset_text.find("[external/qwen3.8-flash-next-q3]") ==
        std::string::npos);
  CHECK(plan.preset_text.find("[remote/qwen3.8-27b]") != std::string::npos);
  CHECK(plan.preset_text.find("url = http://192.168.100.198:8081") !=
        std::string::npos);
  CHECK(plan.preset_text.find("--remote-allow 192.168.100.198") !=
        std::string::npos);
  CHECK(plan.preset_text.find("pool-wait-seconds =") == std::string::npos);
  CHECK(plan.preset_text.find("--host 0.0.0.0") != std::string::npos);
  const auto sections = einrichtung::PresetSections(plan.preset_text);
  CHECK(sections.contains("router"));
  if (sections.contains("router")) {
    CHECK_EQ(sections.at("router").at("auth"), "lan");
    CHECK_EQ(sections.at("router").at("auth-loopback"), "admin");
  }
  CHECK(sections.contains("external/qwen-image-2.1"));
  if (sections.contains("external/qwen-image-2.1"))
    CHECK(sections.at("external/qwen-image-2.1").at("lora").find(';') !=
          std::string::npos);  // two LoRAs on one line
  CHECK(!plan.lizenzen_offen.empty());  // nothing confirmed in the example
  CHECK(!plan.partner_modelle.empty());
  CHECK_EQ(plan.werte.at("preset"), "C:\\gufo\\2.0.3\\gufo-router.ini");
  CHECK_EQ(plan.werte.at("pages"), "C:\\gufo\\2.0.3\\pages");
  CHECK(plan.start_cmd.find("\"" + (tmp / "app").string() +
                            "\\gufo.exe\" router --models-preset") !=
        std::string::npos);
  CHECK(plan.start_cmd.find("cd /d \"" + (tmp / "app").string() + "\"") !=
        std::string::npos);
  CHECK(plan.start_cmd.find("set GUFO_HOST_RESERVE_MIB=2048") !=
        std::string::npos);
  CHECK(plan.bericht.find("## Bausteine und Modelle") != std::string::npos);
  CHECK(plan.bericht.find("[OFFEN]") != std::string::npos);

  // The router itself accepts the text (the point of the parse probe).
  WriteFile(tmp / "plan.ini", plan.preset_text);
  std::vector<gufo::router::PresetModel> models;
  gufo::router::RouterSection section;
  CHECK(gufo::router::LoadPreset(tmp / "plan.ini", {}, &models, &error,
                                 &section));
  if (!error.empty())
    std::cerr << "  LoadPreset: " << error << "\n";
  CHECK_EQ(section.auth, "lan");
  CHECK(models.size() >= plan.modelle.size() - 2);  // LoRAs have no section

  // The same answers as einrichten.py sees them: section by section equal
  // (keys and values; comments and the time stamp do not count).
#ifdef GUFO_PYTHON3
  {
    const std::filesystem::path py_out = tmp / "py-plan.ini";
    const std::filesystem::path py_log = tmp / "py-plan.log";
    std::string cmd = "\"" GUFO_PYTHON3 "\" -X utf8 \"" +
                      (kDir / "einrichten.py").string() + "\" --antworten \"" +
                      (tmp / "antworten.json").string() +
                      "\" --katalog \"" + (kDir / "katalog.json").string() +
                      "\" --plan --preset-ausgabe \"" + py_out.string() +
                      "\" > \"" + py_log.string() + "\" 2>&1";
#ifdef _WIN32
    cmd = "\"" + cmd + "\"";
#endif
    const int rc = std::system(cmd.c_str());
    if (rc != 0 || !std::filesystem::is_regular_file(py_out)) {
      std::cerr << "  einrichten.py --plan did not run (rc " << rc
                << "); python comparison skipped\n";
    } else {
      const auto py = einrichtung::PresetSections(ReadFile(py_out));
      CHECK_EQ(py.size(), sections.size());
      std::size_t same = 0;
      for (const auto& [name, keys] : py) {
        const auto it = sections.find(name);
        if (it == sections.end()) {
          std::cerr << "  only in python: [" << name << "]\n";
          ++failures;
          continue;
        }
        if (it->second == keys) {
          ++same;
          continue;
        }
        ++failures;
        std::cerr << "  [" << name << "] differs\n";
        for (const auto& [k, v] : keys)
          if (!it->second.contains(k) || it->second.at(k) != v)
            std::cerr << "    py  " << k << " = " << v << "\n    c++ "
                      << (it->second.contains(k) ? it->second.at(k)
                                                 : "(missing)")
                      << "\n";
        for (const auto& [k, v] : it->second)
          if (!keys.contains(k))
            std::cerr << "    c++ only " << k << " = " << v << "\n";
      }
      for (const auto& [name, keys] : sections)
        if (!py.contains(name)) {
          std::cerr << "  only in c++: [" << name << "]\n";
          ++failures;
        }
      std::cerr << "  python comparison: " << same << " of " << py.size()
                << " sections identical\n";
      CHECK_EQ(same, py.size());
    }
  }
#else
  std::cerr << "  (no python3 known to the build: einrichten.py comparison "
               "skipped)\n";
#endif

  // Switches: nur-dieser-rechner -> auth = off; 64 GB -> Flash-Next Q3;
  // LoRAs left out -> no lora line; the optional line disappears.
  json::Value a = beispiel;
  a["ram_gb"] = 64;
  a["router"]["erreichbarkeit"] = "nur-dieser-rechner";
  a["modelle_weglassen"] = json::parse(
      "[\"lora-anime-consistency\",\"lora-multiple-angles\",\"minimax-h3\"]");
  CHECK(einrichtung::BuildPlan(a, katalog, kDir / "bausteine", &plan, &error));
  const auto s2 = einrichtung::PresetSections(plan.preset_text);
  CHECK_EQ(s2.at("router").at("auth"), "off");
  CHECK(!s2.at("router").contains("auth-loopback"));
  CHECK(s2.contains("external/qwen3.8-flash-next-q3"));
  CHECK(!s2.contains("external/qwen3.8-flash-next"));
  CHECK(!s2.at("external/qwen-image-2.1").contains("lora"));
  CHECK(plan.preset_text.find("--host 127.0.0.1") != std::string::npos);

  // Errors with einrichten.py's words.
  json::Value bad = beispiel;
  bad["ram_gb"] = 96;
  CHECK(!einrichtung::BuildPlan(bad, katalog, kDir / "bausteine", &plan, &error));
  CHECK_EQ(error, "ram_gb muss 64 oder 128 sein");
  bad = beispiel;
  bad["bausteine"] = json::parse("[\"chat\",\"kaffee\"]");
  CHECK(!einrichtung::BuildPlan(bad, katalog, kDir / "bausteine", &plan, &error));
  CHECK(error.starts_with("unbekannter Baustein 'kaffee'"));
  bad = beispiel;
  bad["modelle_optional"] = json::parse("[\"gibt-es-nicht\"]");
  CHECK(!einrichtung::BuildPlan(bad, katalog, kDir / "bausteine", &plan, &error));
  CHECK(error.starts_with("unbekanntes Modell 'gibt-es-nicht'"));
  bad = beispiel;
  bad["bausteine"] = json::parse("[\"chat\"]");
  bad["modelle_optional"] = json::parse("[\"moss-image\"]");
  CHECK(!einrichtung::BuildPlan(bad, katalog, kDir / "bausteine", &plan, &error));
  CHECK(error.find("gehoert zum Baustein 'bilder'") != std::string::npos);
  bad = json::parse("{\"rechner\":\"x\",\"ram_gb\":64,\"bausteine\":[\"chat\"],"
                    "\"pfade\":{\"modelle\":\"C:\\\\m\",\"daten\":\"C:\\\\d\"}}");
  CHECK(!einrichtung::BuildPlan(bad, katalog, kDir / "bausteine", &plan, &error));
  CHECK_EQ(error, "pfade.app fehlt");
  std::error_code ec;
  std::filesystem::remove_all(tmp, ec);
}

void TestService() {
  const std::filesystem::path tmp = TempDir();
  Fixture f(auth::Mode::kOff);
  einrichtung::ServiceOptions options;
  options.dir = kDir;
  options.exe = tmp / "bin" / "gufo.exe";
  options.port = 8098;
  options.users_unbound = true;
  options.gate = f.gate.get();
  options.check_preset = [](const std::filesystem::path& file) {
    std::vector<gufo::router::PresetModel> models;
    std::string error;
    return gufo::router::LoadPreset(file, {}, &models, &error) ? std::string()
                                                                : error;
  };
  einrichtung::SetupService service(options);

  // Catalogue and machine check.
  std::optional<HttpResponse> r =
      service.Handle(Req("GET", "/router/setup/katalog", "127.0.0.1"));
  CHECK(r.has_value() && r->status == 200);
  CHECK(Body(*r).find("bausteine") != nullptr);
  CHECK(!Body(*r).member_str("katalog_datei").empty());
  r = service.Handle(Req("GET", "/router/setup/pruefen?port=8098", "127.0.0.1"));
  CHECK(r.has_value() && r->status == 200);
  CHECK(Body(*r).find("pruefungen") != nullptr);
  CHECK(Body(*r).member_double("ram_gb") == 64.0 ||
        Body(*r).member_double("ram_gb") == 128.0);
  CHECK(!service.Handle(Req("GET", "/router/status", "127.0.0.1")).has_value());
  r = service.Handle(Req("GET", "/router/setup/nix", "127.0.0.1"));
  CHECK(r.has_value() && r->status == 404);
  r = service.Handle(Req("GET", "/router/setup/downloads", "127.0.0.1"));
  CHECK(r.has_value() && r->status == 200);
  CHECK(!Body(*r).find("laeuft")->as_bool());
  r = service.Handle(Req("GET", "/router/setup/selbsttest", "127.0.0.1"));
  CHECK(r.has_value() && r->status == 409);
  CHECK_EQ(ErrorCode(*r), "no_preset");

  // The admin needs the data folder first (users file unknown).
  r = service.Handle(Req("POST", "/router/setup/admin", "127.0.0.1",
                         "{\"name\":\"chef\",\"passwort\":\"geheim\"}"));
  CHECK(r.has_value() && r->status == 409);
  CHECK_EQ(ErrorCode(*r), "no_data_dir");

  // Step 3: folders checked and created, the users file bound.
  const std::string modelle = (tmp / "modelle").string();
  const std::string daten = (tmp / "daten").string();
  const auto esc = [](std::string text) {
    std::string out;
    for (const char c : text)
      out += c == '\\' ? std::string("\\\\") : std::string(1, c);
    return out;
  };
  r = service.Handle(Req("POST", "/router/setup/pruefen", "127.0.0.1",
                         "{\"pfade\":{\"modelle\":\"" + esc(modelle) +
                             "\",\"daten\":\"" + esc(daten) + "\"}}"));
  CHECK(r.has_value() && r->status == 200);
  {
    const json::Value body = Body(*r);
    const json::Value* ordner = body.find("ordner");
    CHECK(ordner != nullptr && ordner->size() == 2);
    if (ordner != nullptr && ordner->size() == 2) {
      CHECK(ordner->items()[0].find("schreibbar")->as_bool());
      CHECK(ordner->items()[0].find("angelegt")->as_bool());
      CHECK(ordner->items()[1].find("schreibbar")->as_bool());
    }
  }
  CHECK(std::filesystem::is_directory(tmp / "daten"));
  CHECK_EQ(f.store->path().string(),
           (tmp / "daten" / "nutzer" / "nutzer.json").string());
  r = service.Handle(Req("POST", "/router/setup/pruefen", "127.0.0.1",
                         "{\"pfade\":{\"modelle\":\"modelle\",\"daten\":\"" +
                             esc(daten) + "\"}}"));
  CHECK(r.has_value() && r->status == 400);
  CHECK_EQ(ErrorCode(*r), "invalid_path");

  // The first admin: loopback only, once.
  r = service.Handle(Req("POST", "/router/setup/admin", "192.168.1.5",
                         "{\"name\":\"chef\",\"passwort\":\"geheim\"}"));
  CHECK(r.has_value() && r->status == 404);
  r = service.Handle(Req("POST", "/router/setup/admin", "127.0.0.1",
                         "{\"name\":\"Chef!\",\"passwort\":\"geheim\"}"));
  CHECK(r.has_value() && r->status == 400);
  CHECK_EQ(ErrorCode(*r), "invalid_name");
  r = service.Handle(Req("POST", "/router/setup/admin", "127.0.0.1",
                         "{\"name\":\"chef\",\"passwort\":\"abc\"}"));
  CHECK(r.has_value() && r->status == 400);
  CHECK_EQ(ErrorCode(*r), "invalid_password");
  r = service.Handle(Req("POST", "/router/setup/admin", "127.0.0.1",
                         "{\"name\":\"lokal\",\"passwort\":\"geheim\"}"));
  CHECK(r.has_value() && r->status == 400);
  CHECK_EQ(ErrorCode(*r), "name_reserved");
  // Review finding 7: a failed save stores nothing, so the retry is no
  // 409 for a passwordless admin (the users file is a folder here).
  {
    const auto blocker = tmp / "daten" / "nutzer" / "nutzer.json";
    std::filesystem::create_directories(blocker / "x");
    r = service.Handle(Req("POST", "/router/setup/admin", "127.0.0.1",
                           "{\"name\":\"Chef\",\"passwort\":\"geheim\"}"));
    CHECK(r.has_value() && r->status == 500);
    CHECK_EQ(ErrorCode(*r), "users_file");
    CHECK(!f.store->Find("chef").has_value());
    CHECK(!f.store->HasAdmin());
    std::error_code ec;
    std::filesystem::remove_all(blocker, ec);
  }
  r = service.Handle(Req("POST", "/router/setup/admin", "127.0.0.1",
                         "{\"name\":\"Chef\",\"passwort\":\"geheim\"}"));
  CHECK(r.has_value() && r->status == 201);
  CHECK_EQ(Body(*r).member_str("name"), "chef");
  CHECK_EQ(Body(*r).member_str("rolle"), "admin");
  CHECK(std::filesystem::is_regular_file(tmp / "daten" / "nutzer" /
                                         "nutzer.json"));
  CHECK(f.store->HasAdmin());
  CHECK(f.store->Verify("chef", "geheim").has_value());
  r = service.Handle(Req("POST", "/router/setup/admin", "127.0.0.1",
                         "{\"name\":\"zwei\",\"passwort\":\"geheim\"}"));
  CHECK(r.has_value() && r->status == 409);
  CHECK_EQ(ErrorCode(*r), "admin_exists");

  // Preset: preview, licences, write, backup and kept sections.
  json::Value antworten = LoadJson(kDir / "antworten-beispiel.json");
  PointWorkersAt(&antworten, tmp);
  json::Value& pfade = antworten["pfade"];
  pfade["modelle"] = modelle;
  pfade["daten"] = daten;
  pfade["preset"] = (tmp / "daten" / "gufo-router.ini").string();
  pfade["pages"] = (tmp / "pages").string();
  const auto request = [&](bool schreiben) {
    json::Value body = json::Value::object();
    body["antworten"] = antworten;
    body["schreiben"] = schreiben;
    return Req("POST", "/router/setup/preset", "127.0.0.1", body.dump());
  };
#ifdef _WIN32
  {
    // Review finding 10: a TEMP that is no folder (here: below a file)
    // neither throws nor fails; the parse probe goes next to the preset.
    WriteFile(tmp / "keintemp", "x");
    const std::string bogus = (tmp / "keintemp" / "unter").string();
    std::string old_tmp;
    std::string old_temp;
    if (const char* v = std::getenv("TMP"))
      old_tmp = v;
    if (const char* v = std::getenv("TEMP"))
      old_temp = v;
    _putenv_s("TMP", bogus.c_str());
    _putenv_s("TEMP", bogus.c_str());
    r = service.Handle(request(false));
    _putenv_s("TMP", old_tmp.c_str());
    _putenv_s("TEMP", old_temp.c_str());
    CHECK(r.has_value() && r->status == 200);
    if (r.has_value() && r->status != 200)
      std::cerr << "  " << r->body << "\n";
  }
#endif
  r = service.Handle(request(false));
  CHECK(r.has_value() && r->status == 200);
  if (r.has_value() && r->status != 200)
    std::cerr << "  " << r->body << "\n";
  {
    const json::Value body = Body(*r);
    CHECK(body.member_str("preset_text").find("[external/qwen3.8-27b]") !=
          std::string::npos);
    CHECK(!body.member_str("start_cmd").empty());
    CHECK(!body.member_str("bericht").empty());
    CHECK(body.find("lizenzen_offen") != nullptr &&
          !body.find("lizenzen_offen")->empty());
    CHECK(body.find("dateien_fehlend") != nullptr &&
          !body.find("dateien_fehlend")->empty());  // nothing under tmp
    CHECK(body.find("geschrieben") != nullptr &&
          body.find("geschrieben")->empty());
  }
  CHECK(!std::filesystem::exists(tmp / "daten" / "gufo-router.ini"));
  r = service.Handle(request(true));
  CHECK(r.has_value() && r->status == 409);
  CHECK_EQ(ErrorCode(*r), "licenses_open");
  CHECK(Body(*r).find("lizenzen_offen") != nullptr);
  antworten["lizenzen_bestaetigt"] = json::parse("[\"alle\"]");
  r = service.Handle(request(true));
  CHECK(r.has_value() && r->status == 200);
  if (r.has_value() && r->status != 200) {
    std::cerr << "  " << r->body << "\n";
    return;
  }
  const std::filesystem::path preset = tmp / "daten" / "gufo-router.ini";
  CHECK(std::filesystem::is_regular_file(preset));
  CHECK(std::filesystem::is_regular_file(tmp / "daten" / "start-router.cmd"));
  CHECK(std::filesystem::is_regular_file(tmp / "daten" / "einrichtung.json"));
  CHECK(std::filesystem::is_regular_file(tmp / "daten" /
                                         "einrichtung-bericht.md"));
  CHECK(std::filesystem::is_directory(tmp / "daten" / "jobs" / "objekt"));
  CHECK(std::filesystem::is_directory(tmp / "daten" / "voices"));
  {
    const json::Value body = Body(*r);
    CHECK(body.find("geschrieben") != nullptr &&
          body.find("geschrieben")->size() >= 4);
    CHECK(body.member_str("bericht").find("- Preset: ") != std::string::npos);
    const json::Value stand = LoadJson(tmp / "daten" / "einrichtung.json");
    CHECK(stand.find("antworten") != nullptr);
    CHECK_EQ(stand.find("antworten")->member_str("rechner"), "evo-56");
    CHECK(!stand.find("modelle")->empty());
  }
  {
    std::vector<gufo::router::PresetModel> models;
    std::string error;
    gufo::router::RouterSection section;
    CHECK(gufo::router::LoadPreset(preset, {}, &models, &error, &section));
    CHECK_EQ(section.auth, "lan");
  }
  // A section of the user's own survives the next write, after the blocks.
  {
    std::string text = ReadFile(preset);
    // LoadPreset wants the command to exist: the dummy gufo.exe of `app`.
    text += "\n[external/meins]\ncommand = " +
            (tmp / "app" / "gufo.exe").string() + "\nmodel = C:\\x\\m.gguf\n";
    WriteFile(preset, text);
  }
  r = service.Handle(request(true));
  CHECK(r.has_value() && r->status == 200);
  if (r.has_value() && r->status != 200)
    std::cerr << "  " << r->body << "\n";
  {
    const std::string text = ReadFile(preset);
    CHECK(text.find("[external/meins]") != std::string::npos);
    CHECK(text.find("kept from the previous preset") != std::string::npos);
    CHECK(text.find("[external/meins]") > text.find("[remote/"));
    int backups = 0;
    for (const auto& entry : std::filesystem::directory_iterator(tmp / "daten"))
      if (entry.path().filename().string().find(".bak-") != std::string::npos)
        ++backups;
    CHECK_EQ(backups, 1);
    std::vector<gufo::router::PresetModel> models;
    std::string error;
    CHECK(gufo::router::LoadPreset(preset, {}, &models, &error));
    bool meins = false;
    for (const auto& m : models)
      meins = meins || m.id == "meins";
    CHECK(meins);
  }
  // Paths outside the rules are refused before anything is written.
  {
    json::Value relative = antworten;
    relative["pfade"]["daten"] = "daten";
    relative["pfade"]["preset"] = "daten\\gufo-router.ini";
    json::Value body = json::Value::object();
    body["antworten"] = relative;
    body["schreiben"] = true;
    r = service.Handle(Req("POST", "/router/setup/preset", "127.0.0.1",
                           body.dump()));
    CHECK(r.has_value() && r->status == 400);
    CHECK_EQ(ErrorCode(*r), "invalid_path");
  }
  // The parse probe stops a preset the router would refuse.
  {
    einrichtung::ServiceOptions strict = options;
    strict.check_preset = [](const std::filesystem::path&) {
      return std::string("[external/x] Unknown option: --bogus");
    };
    einrichtung::SetupService refusing(strict);
    r = refusing.Handle(request(true));
    CHECK(r.has_value() && r->status == 422);
    CHECK_EQ(ErrorCode(*r), "preset_invalid");
  }
  // Bad input.
  r = service.Handle(Req("POST", "/router/setup/preset", "127.0.0.1", "{"));
  CHECK(r.has_value() && r->status == 400);
  r = service.Handle(Req("POST", "/router/setup/preset", "127.0.0.1",
                         "{\"antworten\":{\"rechner\":\"x\"}}"));
  CHECK(r.has_value() && r->status == 400);
  CHECK_EQ(ErrorCode(*r), "invalid_request");
  std::error_code ec;
  std::filesystem::remove_all(tmp, ec);
}

// ---------------------------------------------------------------------------
// Downloads: a local stand-in for Hugging Face (HEAD, GET, Range, 401 for a
// gated repo, 503 for a flaky one, a server that ignores Range, a slow
// file for pause/cancel) and the service's /router/setup/downloads.

namespace {

std::string PseudoRandom(std::size_t n, unsigned seed) {
  std::string out(n, '\0');
  unsigned x = seed * 2654435761u + 12345u;
  for (std::size_t i = 0; i < n; ++i) {
    x = x * 1103515245u + 12345u;
    out[i] = static_cast<char>((x >> 16) & 0xff);
  }
  return out;
}

std::string Sha(const std::string& text) {
  return gufo::crypto::Sha256Hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
}

struct FakeHf {
  struct Entry {
    std::string body;
    bool gated{false};
    int fail_first{0};     // 503 for the first n requests
    bool ignore_range{false};
    bool refuse_range{false};  // 416 for every Range request
    int chunk_ms{0};       // >0: streamed in 64 KB chunks with this pause
    bool linked_etag{false};  // X-Linked-ETag with the body's SHA-256
  };
  std::map<std::string, Entry> files;
  std::mutex mutex;
  std::vector<std::pair<std::string, std::string>> ranges;  // path, Range
  std::map<std::string, int> fails;
  int requests{0};
  std::unique_ptr<gufo::server::HttpServer> server;
  std::thread thread;
  int port{0};

  void Start() {
    gufo::server::HttpServerOptions options;
    options.dispatcher = [this](const HttpRequest& req) { return Serve(req); };
    server = std::make_unique<gufo::server::HttpServer>(
        "127.0.0.1", 0, nullptr, nullptr, nullptr, nullptr,
        std::move(options));
    std::string error;
    CHECK(server->start(&error));
    port = server->port();
    thread = std::thread([this] { server->run(false); });
  }
  ~FakeHf() {
    if (server != nullptr)
      server->stop();
    if (thread.joinable())
      thread.join();
  }
  std::string Base() const { return "http://127.0.0.1:" + std::to_string(port); }

  HttpResponse Serve(const HttpRequest& req) {
    const auto it = files.find(req.path);
    {
      const std::lock_guard<std::mutex> lock(mutex);
      ++requests;
    }
    if (it == files.end())
      return HttpResponse{.status = 404, .reason = "Not Found", .body = "{}"};
    const Entry& e = it->second;
    if (e.gated && req.header("Authorization") != "Bearer hf_test")
      return HttpResponse{.status = 401, .reason = "Unauthorized", .body = "{}"};
    {
      const std::lock_guard<std::mutex> lock(mutex);
      if (fails[req.path] < e.fail_first) {
        ++fails[req.path];
        return HttpResponse{.status = 503, .reason = "Service Unavailable",
                            .body = "{}"};
      }
    }
    HttpResponse r;
    r.headers.emplace_back("Content-Type", "application/octet-stream");
    r.headers.emplace_back("Accept-Ranges", "bytes");
    if (e.linked_etag) {
      r.headers.emplace_back("X-Linked-Size", std::to_string(e.body.size()));
      r.headers.emplace_back("X-Linked-ETag", "\"" + Sha(e.body) + "\"");
    }
    if (req.method == "HEAD") {
      // The body is sent too (the server sets Content-Length from it);
      // curl's HEAD reads the headers only.
      r.body = e.body;
      return r;
    }
    std::size_t from = 0;
    const std::string range = req.header("Range");
    if (!range.empty()) {
      const std::lock_guard<std::mutex> lock(mutex);
      ranges.emplace_back(req.path, range);
    }
    if (!range.empty() && e.refuse_range)
      return HttpResponse{.status = 416, .reason = "Range Not Satisfiable",
                          .body = "{}"};
    if (!range.empty() && !e.ignore_range && range.starts_with("bytes=")) {
      from = static_cast<std::size_t>(
          std::strtoull(range.c_str() + 6, nullptr, 10));
      if (from >= e.body.size())
        return HttpResponse{.status = 416, .reason = "Range Not Satisfiable",
                            .body = "{}"};
      r.status = 206;
      r.reason = "Partial Content";
      r.headers.emplace_back(
          "Content-Range", "bytes " + std::to_string(from) + "-" +
                               std::to_string(e.body.size() - 1) + "/" +
                               std::to_string(e.body.size()));
    }
    const std::string part = e.body.substr(from);
    if (e.chunk_ms <= 0) {
      r.body = part;
      return r;
    }
    r.headers.emplace_back("Content-Length", std::to_string(part.size()));
    const int chunk_ms = e.chunk_ms;
    r.streaming_body = [part, chunk_ms](const HttpResponse::BodyWriter& write) {
      for (std::size_t at = 0; at < part.size(); at += 65536) {
        if (!write(std::string_view(part).substr(at, 65536)))
          return;
        std::this_thread::sleep_for(std::chrono::milliseconds(chunk_ms));
      }
    };
    return r;
  }
};

const json::Value* FindFile(const json::Value& body, std::string_view pfad) {
  const json::Value* list = body.find("dateien");
  if (list == nullptr)
    return nullptr;
  for (const json::Value& d : list->items())
    if (d.member_str("pfad") == pfad)
      return &d;
  return nullptr;
}

std::string FileStatus(const json::Value& body, std::string_view pfad) {
  const json::Value* d = FindFile(body, pfad);
  return d == nullptr ? std::string("(missing)") : d->member_str("status");
}

std::string FileField(const json::Value& body, std::string_view pfad,
                      const std::string& key) {
  const json::Value* d = FindFile(body, pfad);
  return d == nullptr ? std::string("(missing)") : d->member_str(key);
}

/// Polls GET until the job neither runs nor pauses (or the deadline).
json::Value WaitDone(einrichtung::SetupService& service,
                     std::chrono::milliseconds limit =
                         std::chrono::milliseconds(20000)) {
  const auto deadline = std::chrono::steady_clock::now() + limit;
  for (;;) {
    const auto r =
        service.Handle(Req("GET", "/router/setup/downloads", "127.0.0.1"));
    const json::Value body = Body(*r);
    if (!body.find("laeuft")->as_bool() && !body.find("pause")->as_bool())
      return body;
    if (std::chrono::steady_clock::now() > deadline) {
      std::cerr << "WaitDone: timeout\n";
      ++failures;
      return body;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

json::Value WaitStatus(einrichtung::SetupService& service,
                       std::string_view pfad, std::string_view status,
                       std::chrono::milliseconds limit =
                           std::chrono::milliseconds(10000)) {
  const auto deadline = std::chrono::steady_clock::now() + limit;
  for (;;) {
    const auto r =
        service.Handle(Req("GET", "/router/setup/downloads", "127.0.0.1"));
    const json::Value body = Body(*r);
    if (FileStatus(body, pfad) == status)
      return body;
    if (std::chrono::steady_clock::now() > deadline) {
      std::cerr << "WaitStatus: timeout waiting for " << pfad << " = "
                << status << " (is " << FileStatus(body, pfad) << ")\n";
      ++failures;
      return body;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

json::Value FileEntry(const std::string& pfad, std::size_t bytes,
                      const std::string& sha) {
  json::Value d = json::Value::object();
  d["pfad"] = pfad;
  d["bytes"] = static_cast<double>(bytes);
  d["sha256"] = sha;
  d["sha256_status"] = sha.empty() ? "fehlt" : "test";
  return d;
}

json::Value ModelEntry(const std::string& id, const std::string& ordner,
                       const std::string& repo, const std::string& rev,
                       json::Value dateien) {
  json::Value m = json::Value::object();
  m["id"] = id;
  m["name"] = id;
  m["pflicht"] = true;
  m["worker"] = "gufo.exe";
  json::Value q = json::Value::object();
  q["ordner"] = ordner;
  q["hf_repo"] = repo;
  q["hf_revision"] = rev;
  json::Value quellen = json::Value::array();
  quellen.push_back(std::move(q));
  m["quellen"] = std::move(quellen);
  m["dateien"] = std::move(dateien);
  return m;
}

std::string JsonList(const std::vector<std::string>& items) {
  std::string out = "[";
  for (std::size_t i = 0; i < items.size(); ++i)
    out += (i ? ",\"" : "\"") + items[i] + "\"";
  return out + "]";
}

std::string Esc(std::string text) {
  std::string out;
  for (const char c : text)
    out += c == '\\' ? std::string("\\\\") : std::string(1, c);
  return out;
}

}  // namespace

void TestDownloads() {
  CHECK_EQ(einrichtung::DownloadUrl("https://huggingface.co/", "a/b", "",
                                    "x y\\z.bin"),
           std::string("https://huggingface.co/a/b/resolve/main/x%20y/z.bin"));
  CHECK_EQ(einrichtung::DownloadUrl("http://127.0.0.1:1", "a/b", "abc",
                                    "m.gguf"),
           std::string("http://127.0.0.1:1/a/b/resolve/abc/m.gguf"));

  const std::filesystem::path tmp = TempDir();
  FakeHf hf;
  const std::string good = PseudoRandom(1 << 20, 1);
  const std::string nosha = PseudoRandom(300000, 2);
  const std::string bad = PseudoRandom(50000, 3);
  const std::string have = PseudoRandom(1234, 4);
  const std::string wrongsize = PseudoRandom(20, 5);
  const std::string gated = PseudoRandom(4000, 6);
  const std::string flaky = PseudoRandom(70000, 7);
  const std::string resume = PseudoRandom(200000, 8);
  const std::string noresume = PseudoRandom(150000, 9);
  const std::string slow = PseudoRandom(3 << 20, 10);
  const std::string refuse = PseudoRandom(90000, 11);
  const std::string badhex = PseudoRandom(5000, 12);
  hf.files["/acme/repo-a/resolve/abc/good.bin"] = {.body = good};
  hf.files["/acme/repo-a/resolve/abc/nosha.bin"] = {.body = nosha,
                                                     .linked_etag = true};
  hf.files["/acme/repo-a/resolve/abc/bad.bin"] = {.body = bad};
  hf.files["/acme/repo-a/resolve/abc/sub/have.bin"] = {.body = have};
  hf.files["/acme/repo-a/resolve/abc/wrongsize.bin"] = {.body = wrongsize};
  hf.files["/acme/gated/resolve/main/g.bin"] = {.body = gated, .gated = true};
  hf.files["/acme/flaky/resolve/main/f.bin"] = {.body = flaky, .fail_first = 2};
  hf.files["/acme/repo-a/resolve/abc/resume.bin"] = {.body = resume};
  hf.files["/acme/repo-a/resolve/abc/noresume.bin"] = {.body = noresume,
                                                        .ignore_range = true};
  hf.files["/acme/repo-a/resolve/abc/refuse.bin"] = {.body = refuse,
                                                      .refuse_range = true};
  hf.files["/acme/repo-a/resolve/abc/badhex.bin"] = {.body = badhex};
  hf.files["/acme/slow/resolve/main/s.bin"] = {.body = slow, .chunk_ms = 15};
  hf.Start();

  // A catalogue of its own (dir with katalog.json only).
  json::Value katalog = json::Value::object();
  katalog["katalog_version"] = "test";
  json::Value modelle = json::Value::array();
  {
    json::Value d = json::Value::array();
    d.push_back(FileEntry("acme\\repo-a\\good.bin", good.size(), Sha(good)));
    d.push_back(FileEntry("acme\\repo-a\\nosha.bin", nosha.size(), ""));
    d.push_back(FileEntry("acme\\repo-a\\bad.bin", bad.size(), Sha(good)));
    d.push_back(FileEntry("acme\\repo-a\\sub\\have.bin", have.size(), ""));
    d.push_back(FileEntry("acme\\repo-a\\wrongsize.bin", 10, ""));
    modelle.push_back(ModelEntry("m1", "acme\\repo-a", "acme/repo-a", "abc",
                                 std::move(d)));
  }
  {
    json::Value d = json::Value::array();
    d.push_back(FileEntry("acme\\gated\\g.bin", gated.size(), Sha(gated)));
    modelle.push_back(ModelEntry("m2", "acme\\gated", "acme/gated", "",
                                 std::move(d)));
  }
  {
    json::Value d = json::Value::array();
    json::Value none = FileEntry("acme\\none\\n.bin", 10, "");
    none["hf_repo"] = json::Value();
    none["hf_pfad"] = json::Value();
    d.push_back(std::move(none));
    modelle.push_back(ModelEntry("m3", "acme\\none", "acme/none", "",
                                 std::move(d)));
  }
  {
    json::Value d = json::Value::array();
    d.push_back(FileEntry("acme\\flaky\\f.bin", flaky.size(), Sha(flaky)));
    modelle.push_back(ModelEntry("m4", "acme\\flaky", "acme/flaky", "",
                                 std::move(d)));
  }
  {
    json::Value d = json::Value::array();
    d.push_back(FileEntry("acme\\repo-a\\resume.bin", resume.size(),
                          Sha(resume)));
    d.push_back(FileEntry("acme\\repo-a\\noresume.bin", noresume.size(),
                          Sha(noresume)));
    d.push_back(FileEntry("acme\\repo-a\\refuse.bin", refuse.size(),
                          Sha(refuse)));
    // Finding 13: 64 characters but no hex digest = unknown, no mismatch.
    d.push_back(FileEntry("acme\\repo-a\\badhex.bin", badhex.size(),
                          std::string(64, 'z')));
    modelle.push_back(ModelEntry("m5", "acme\\repo-a", "acme/repo-a", "abc",
                                 std::move(d)));
  }
  {
    json::Value d = json::Value::array();
    d.push_back(FileEntry("acme\\slow\\s.bin", slow.size(), Sha(slow)));
    modelle.push_back(ModelEntry("m6", "acme\\slow", "acme/slow", "",
                                 std::move(d)));
  }
  {
    json::Value d = json::Value::array();
    d.push_back(FileEntry("acme\\huge\\h.bin", 1ull << 56, ""));
    modelle.push_back(ModelEntry("m7", "acme\\huge", "acme/huge", "",
                                 std::move(d)));
  }
  json::Value chat = json::Value::object();
  chat["titel"] = "Chat";
  chat["modelle"] = std::move(modelle);
  json::Value bausteine = json::Value::object();
  bausteine["chat"] = std::move(chat);
  katalog["bausteine"] = std::move(bausteine);
  WriteFile(tmp / "einr" / "katalog.json", katalog.dump());

  Fixture f(auth::Mode::kOff);
  einrichtung::ServiceOptions options;
  options.dir = tmp / "einr";
  options.exe = tmp / "bin" / "gufo.exe";
  options.port = 8098;
  options.gate = f.gate.get();
  options.hf_base_url = hf.Base();
  options.download_allow_http = true;
  options.download_retry_wait_ms = {30, 30, 30, 30, 30};
  options.download_reserve_bytes = 0;
  einrichtung::SetupService service(options);

  const std::filesystem::path ziel = tmp / "modelle";
  WriteFile(ziel / "acme" / "repo-a" / "sub" / "have.bin", have);
  WriteFile(ziel / "acme" / "repo-a" / "resume.bin.part", resume.substr(0, 1000));
  WriteFile(ziel / "acme" / "repo-a" / "noresume.bin.part",
            noresume.substr(0, 777));
  WriteFile(ziel / "acme" / "repo-a" / "refuse.bin.part",
            refuse.substr(0, 500));
  const std::string ziel_json = "\"ziel\":\"" + Esc(ziel.string()) + "\"";
  const auto post = [&](const std::string& body) {
    return service.Handle(
        Req("POST", "/router/setup/downloads", "127.0.0.1", body));
  };

  // Nothing yet.
  std::optional<HttpResponse> r =
      service.Handle(Req("GET", "/router/setup/downloads", "127.0.0.1"));
  CHECK(r.has_value() && r->status == 200);
  CHECK_EQ(Body(*r).find("dateien")->size(), std::size_t(0));
  r = post("{\"aktion\":\"pause\"}");
  CHECK(r.has_value() && r->status == 409);
  CHECK_EQ(ErrorCode(*r), "no_download");
  r = post("{\"aktion\":\"planen\",\"modelle\":[\"m1\"],\"ziel\":\"modelle\"}");
  CHECK(r.has_value() && r->status == 400);
  CHECK_EQ(ErrorCode(*r), "invalid_path");

  // Plan: the state on disk, sources, no source.
  const std::vector<std::string> first = {"m1", "m2", "m3", "m4", "m5"};
  r = post("{\"aktion\":\"planen\",\"modelle\":" + JsonList(first) + "," +
           ziel_json + "}");
  CHECK(r.has_value() && r->status == 200);
  {
    const json::Value b = Body(*r);
    CHECK(!b.find("laeuft")->as_bool());
    CHECK_EQ(b.find("dateien")->size(), std::size_t(12));
    CHECK_EQ(FileStatus(b, "acme\\repo-a\\good.bin"), "wartet");
    CHECK_EQ(FileStatus(b, "acme\\repo-a\\sub\\have.bin"), "vorhanden");
    CHECK_EQ(FileStatus(b, "acme\\none\\n.bin"), "fehler");
    CHECK_EQ(FileField(b, "acme\\none\\n.bin", "fehler_code"), "no_source");
    const json::Value* good_entry = FindFile(b, "acme\\repo-a\\good.bin");
    CHECK(good_entry != nullptr && good_entry->find("quelle") != nullptr);
    if (good_entry != nullptr && good_entry->find("quelle") != nullptr) {
      CHECK_EQ(good_entry->find("quelle")->member_str("hf_repo"), "acme/repo-a");
      CHECK_EQ(good_entry->find("quelle")->member_str("revision"), "abc");
    }
    CHECK_EQ(b.member_str("ziel"), ziel.string());
  }
  CHECK(!std::filesystem::exists(ziel / "acme" / "repo-a" / "good.bin"));

  // Start without a token: everything but the gated file settles.
  r = post("{\"aktion\":\"start\",\"modelle\":" + JsonList(first) + "," +
           ziel_json + "}");
  CHECK(r.has_value() && r->status == 202);
  {
    const json::Value b = WaitDone(service);
    CHECK_EQ(FileStatus(b, "acme\\repo-a\\good.bin"), "fertig");
    CHECK_EQ(FileField(b, "acme\\repo-a\\good.bin", "sha256_status"), "ok");
    CHECK_EQ(FileField(b, "acme\\repo-a\\good.bin", "sha256_quelle"), "katalog");
    CHECK_EQ(ReadFile(ziel / "acme" / "repo-a" / "good.bin"), good);
    CHECK(!std::filesystem::exists(ziel / "acme" / "repo-a" / "good.bin.part"));
    CHECK_EQ(FileStatus(b, "acme\\repo-a\\nosha.bin"), "fertig");
    CHECK_EQ(FileField(b, "acme\\repo-a\\nosha.bin", "sha256_status"), "ok");
    CHECK_EQ(FileField(b, "acme\\repo-a\\nosha.bin", "sha256_quelle"), "hf");
    CHECK_EQ(FileField(b, "acme\\repo-a\\nosha.bin", "sha256_ist"), Sha(nosha));
    CHECK_EQ(FileStatus(b, "acme\\repo-a\\bad.bin"), "fehler");
    CHECK_EQ(FileField(b, "acme\\repo-a\\bad.bin", "fehler_code"), "sha256");
    CHECK_EQ(FileField(b, "acme\\repo-a\\bad.bin", "sha256_status"), "falsch");
    CHECK(std::filesystem::exists(ziel / "acme" / "repo-a" / "bad.bin.falsch"));
    CHECK(!std::filesystem::exists(ziel / "acme" / "repo-a" / "bad.bin"));
    CHECK_EQ(FileStatus(b, "acme\\repo-a\\sub\\have.bin"), "vorhanden");
    CHECK_EQ(FileStatus(b, "acme\\repo-a\\wrongsize.bin"), "fehler");
    CHECK_EQ(FileField(b, "acme\\repo-a\\wrongsize.bin", "fehler_code"),
             "size_mismatch");
    CHECK_EQ(FileStatus(b, "acme\\gated\\g.bin"), "fehler");
    CHECK_EQ(FileField(b, "acme\\gated\\g.bin", "fehler_code"), "gated");
    CHECK_EQ(FileStatus(b, "acme\\none\\n.bin"), "fehler");
    CHECK_EQ(FileStatus(b, "acme\\flaky\\f.bin"), "fertig");
    CHECK_EQ(FindFile(b, "acme\\flaky\\f.bin")->member_double("versuche"), 2.0);
    CHECK_EQ(FileStatus(b, "acme\\repo-a\\resume.bin"), "fertig");
    CHECK_EQ(FileField(b, "acme\\repo-a\\resume.bin", "sha256_status"), "ok");
    CHECK_EQ(ReadFile(ziel / "acme" / "repo-a" / "resume.bin"), resume);
    CHECK_EQ(FileStatus(b, "acme\\repo-a\\noresume.bin"), "fertig");
    CHECK_EQ(FileField(b, "acme\\repo-a\\noresume.bin", "sha256_status"), "ok");
    CHECK_EQ(ReadFile(ziel / "acme" / "repo-a" / "noresume.bin"), noresume);
    // Review finding 9: a 416 for a short .part starts it over at once
    // instead of retrying the same range until the retries run out.
    CHECK_EQ(FileStatus(b, "acme\\repo-a\\refuse.bin"), "fertig");
    CHECK_EQ(FileField(b, "acme\\repo-a\\refuse.bin", "sha256_status"), "ok");
    CHECK_EQ(ReadFile(ziel / "acme" / "repo-a" / "refuse.bin"), refuse);
    {
      const std::lock_guard<std::mutex> lock(hf.mutex);
      int refused = 0;
      for (const auto& [path, range] : hf.ranges)
        if (path == "/acme/repo-a/resolve/abc/refuse.bin")
          ++refused;
      CHECK_EQ(refused, 1);
    }
    CHECK_EQ(FileStatus(b, "acme\\repo-a\\badhex.bin"), "fertig");
    CHECK_EQ(FileField(b, "acme\\repo-a\\badhex.bin", "sha256_status"),
             "unbekannt");
    bool resumed = false;
    bool noresumed = false;
    {
      const std::lock_guard<std::mutex> lock(hf.mutex);
      for (const auto& [path, range] : hf.ranges) {
        if (path == "/acme/repo-a/resolve/abc/resume.bin" &&
            range == "bytes=1000-")
          resumed = true;
        if (path == "/acme/repo-a/resolve/abc/noresume.bin" &&
            range == "bytes=777-")
          noresumed = true;
      }
    }
    CHECK(resumed);
    CHECK(noresumed);
    CHECK(b.member_double("gesamt_bytes") > 0);
  }
  // The state file exists and never carries the token.
  CHECK(std::filesystem::exists(ziel / ".einrichtung-downloads.json"));

  // Resume with the token: the gated file loads; the others stay.
  r = post("{\"aktion\":\"weiter\",\"hf_token\":\"hf_test\"}");
  CHECK(r.has_value() && r->status == 200);
  {
    const json::Value b = WaitDone(service);
    CHECK_EQ(FileStatus(b, "acme\\gated\\g.bin"), "fertig");
    CHECK_EQ(ReadFile(ziel / "acme" / "gated" / "g.bin"), gated);
    CHECK_EQ(FileStatus(b, "acme\\repo-a\\good.bin"), "fertig");
    CHECK_EQ(FileStatus(b, "acme\\repo-a\\bad.bin"), "fehler");
    CHECK_EQ(FileStatus(b, "acme\\repo-a\\wrongsize.bin"), "fehler");
  }
  CHECK(ReadFile(ziel / ".einrichtung-downloads.json").find("hf_test") ==
        std::string::npos);

  // Pause mid-transfer, resume with a Range, cancel, start again.
  r = post("{\"aktion\":\"start\",\"modelle\":[\"m6\"]," + ziel_json + "}");
  CHECK(r.has_value() && r->status == 202);
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  r = post("{\"aktion\":\"start\",\"modelle\":[\"m6\"]," + ziel_json + "}");
  CHECK(r.has_value() && r->status == 409);
  CHECK_EQ(ErrorCode(*r), "download_running");
  r = post("{\"aktion\":\"pause\"}");
  CHECK(r.has_value() && r->status == 200);
  CHECK(Body(*r).find("pause")->as_bool());
  {
    const json::Value b = WaitStatus(service, "acme\\slow\\s.bin", "pause");
    const std::filesystem::path part = ziel / "acme" / "slow" / "s.bin.part";
    CHECK(std::filesystem::exists(part));
    std::error_code ec;
    const auto size = std::filesystem::file_size(part, ec);
    CHECK(size > 0 && size < slow.size());
    CHECK_EQ(static_cast<double>(size),
             FindFile(b, "acme\\slow\\s.bin")->member_double("geladen"));
    std::size_t before = 0;
    {
      const std::lock_guard<std::mutex> lock(hf.mutex);
      before = hf.ranges.size();
    }
    r = post("{\"aktion\":\"weiter\"}");
    CHECK(r.has_value() && r->status == 200);
    const json::Value done = WaitDone(service);
    CHECK_EQ(FileStatus(done, "acme\\slow\\s.bin"), "fertig");
    CHECK_EQ(FileField(done, "acme\\slow\\s.bin", "sha256_status"), "ok");
    CHECK_EQ(ReadFile(ziel / "acme" / "slow" / "s.bin"), slow);
    bool ranged = false;
    {
      const std::lock_guard<std::mutex> lock(hf.mutex);
      for (std::size_t i = before; i < hf.ranges.size(); ++i)
        if (hf.ranges[i].first == "/acme/slow/resolve/main/s.bin" &&
            hf.ranges[i].second == "bytes=" + std::to_string(size) + "-")
          ranged = true;
    }
    CHECK(ranged);
  }
  std::filesystem::remove(ziel / "acme" / "slow" / "s.bin");
  r = post("{\"aktion\":\"start\",\"modelle\":[\"m6\"]," + ziel_json + "}");
  CHECK(r.has_value() && r->status == 202);
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  r = post("{\"aktion\":\"abbrechen\"}");
  CHECK(r.has_value() && r->status == 200);
  {
    (void)WaitStatus(service, "acme\\slow\\s.bin", "abgebrochen");
    // The worker marks the file before it leaves: wait until it has.
    const json::Value b = WaitDone(service);
    CHECK_EQ(FileStatus(b, "acme\\slow\\s.bin"), "abgebrochen");
    CHECK(!b.find("laeuft")->as_bool());
    CHECK(!b.find("pause")->as_bool());
    CHECK(std::filesystem::exists(ziel / "acme" / "slow" / "s.bin.part"));
    CHECK(!std::filesystem::exists(ziel / "acme" / "slow" / "s.bin"));
  }
  r = post("{\"aktion\":\"start\",\"modelle\":[\"m6\"]," + ziel_json + "}");
  CHECK(r.has_value() && r->status == 202);
  {
    const json::Value b = WaitDone(service);
    CHECK_EQ(FileStatus(b, "acme\\slow\\s.bin"), "fertig");
    CHECK_EQ(ReadFile(ziel / "acme" / "slow" / "s.bin"), slow);
  }

  // Not enough space: 507 with the numbers, nothing started.
  r = post("{\"aktion\":\"start\",\"modelle\":[\"m7\"]," + ziel_json + "}");
  CHECK(r.has_value() && r->status == 507);
  CHECK_EQ(ErrorCode(*r), "disk_full");
  CHECK(Body(*r).member_double("fehlt_bytes") > 0);
  CHECK(Body(*r).member_double("frei_bytes") > 0);
  CHECK(!std::filesystem::exists(ziel / "acme" / "huge" / "h.bin.part"));

  // A new plan finds the digests of the finished files in the state file.
  r = post("{\"aktion\":\"planen\",\"modelle\":[\"m1\"]," + ziel_json + "}");
  CHECK(r.has_value() && r->status == 200);
  {
    const json::Value b = Body(*r);
    CHECK_EQ(FileStatus(b, "acme\\repo-a\\good.bin"), "vorhanden");
    CHECK_EQ(FileField(b, "acme\\repo-a\\good.bin", "sha256_status"), "ok");
    CHECK_EQ(FileStatus(b, "acme\\repo-a\\bad.bin"), "wartet");
  }
  std::error_code ec;
  std::filesystem::remove_all(tmp, ec);
}

/// One small real file from our own Hugging Face repo
/// (D3rPapst/rife-ifnet-gufo, rife47.safetensors, 21 MB; SHA-256 from
/// hf-upload 13.11.2026, equal to the X-Linked-ETag Hugging Face reports).
/// Skipped without network or with GUFO_TEST_OFFLINE set; a transfer that
/// does not finish within 40 s is cancelled and counts as skipped.
void TestRealDownload() {
  if (std::getenv("GUFO_TEST_OFFLINE") != nullptr) {
    std::cout << "real download: skipped (GUFO_TEST_OFFLINE)\n";
    return;
  }
  const std::filesystem::path tmp = TempDir();
  json::Value katalog = json::Value::object();
  json::Value d = json::Value::array();
  d.push_back(FileEntry(
      "rife\\weights\\rife47.safetensors", 21312128,
      "eb3bc69d7539ce6df78b10f00bf0ff450d7617a2f9448d3e380a2259339a94e9"));
  json::Value modelle = json::Value::array();
  modelle.push_back(ModelEntry("rife47", "rife\\weights",
                               "D3rPapst/rife-ifnet-gufo", "", std::move(d)));
  json::Value chat = json::Value::object();
  chat["modelle"] = std::move(modelle);
  json::Value bausteine = json::Value::object();
  bausteine["chat"] = std::move(chat);
  katalog["bausteine"] = std::move(bausteine);
  WriteFile(tmp / "einr" / "katalog.json", katalog.dump());
  Fixture f(auth::Mode::kOff);
  einrichtung::ServiceOptions options;
  options.dir = tmp / "einr";
  options.exe = tmp / "bin" / "gufo.exe";
  options.gate = f.gate.get();
  options.download_retry_wait_ms = {500};
  options.download_reserve_bytes = 0;
  const std::filesystem::path ziel = tmp / "modelle";
  {
    einrichtung::SetupService service(options);
    const auto r = service.Handle(Req(
        "POST", "/router/setup/downloads", "127.0.0.1",
        "{\"aktion\":\"start\",\"modelle\":[\"rife47\"],\"ziel\":\"" +
            Esc(ziel.string()) + "\"}"));
    CHECK(r.has_value() && r->status == 202);
    const int before = failures;
    const auto start = std::chrono::steady_clock::now();
    json::Value b = WaitDone(service, std::chrono::milliseconds(40000));
    const auto seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    if (failures != before) {
      failures = before;  // the timeout is not a failure of the code
      (void)service.Handle(Req("POST", "/router/setup/downloads", "127.0.0.1",
                               "{\"aktion\":\"abbrechen\"}"));
      std::cout << "real download: skipped (not finished in 40 s)\n";
    } else if (FileStatus(b, "rife\\weights\\rife47.safetensors") == "fertig") {
      CHECK_EQ(FileField(b, "rife\\weights\\rife47.safetensors", "sha256_status"),
               "ok");
      std::cout << "real download: rife47.safetensors 21312128 bytes in "
                << seconds << " s, sha256 "
                << FileField(b, "rife\\weights\\rife47.safetensors",
                             "sha256_status")
                << "\n";
    } else {
      std::cout << "real download: skipped ("
                << FileField(b, "rife\\weights\\rife47.safetensors", "fehler")
                << ")\n";
    }
  }
  std::error_code ec;
  std::filesystem::remove_all(tmp, ec);
}

// ---------------------------------------------------------------------------
// The self-test: planning, the preset's start line, the service against a
// fake router (every endpoint the cases use), the report.

struct FakeRouter {
  std::mutex mutex;
  std::vector<std::string> unloaded;
  std::vector<std::string> paths;
  /// "<path> <X-Gufo-Internal>" of every request.
  std::vector<std::string> keys;
  int video_polls{0};
  int objekt_polls{0};
  int chat_delay_ms{0};
  std::unique_ptr<gufo::server::HttpServer> server;
  std::thread thread;
  int port{0};

  void Start() {
    gufo::server::HttpServerOptions options;
    options.dispatcher = [this](const HttpRequest& req) { return Serve(req); };
    server = std::make_unique<gufo::server::HttpServer>(
        "127.0.0.1", 0, nullptr, nullptr, nullptr, nullptr,
        std::move(options));
    std::string error;
    CHECK(server->start(&error));
    port = server->port();
    thread = std::thread([this] { server->run(false); });
  }
  ~FakeRouter() {
    if (server != nullptr)
      server->stop();
    if (thread.joinable())
      thread.join();
  }
  std::string Base() const { return "http://127.0.0.1:" + std::to_string(port); }

  static HttpResponse JsonOk(const json::Value& v, int status = 200) {
    HttpResponse r;
    r.status = status;
    r.reason = status == 202 ? "Accepted" : "OK";
    r.headers.emplace_back("Content-Type", "application/json");
    r.body = v.dump();
    return r;
  }
  static HttpResponse Wav() {
    HttpResponse r;
    r.headers.emplace_back("Content-Type", "audio/wav");
    r.body = "RIFF" + std::string(60, 'x');
    return r;
  }
  static HttpResponse Err(int status, const std::string& message) {
    json::Value e = json::Value::object();
    e["message"] = message;
    e["code"] = status;
    json::Value v = json::Value::object();
    v["error"] = std::move(e);
    HttpResponse r = JsonOk(v);
    r.status = status;
    r.reason = "Error";
    return r;
  }

  HttpResponse Serve(const HttpRequest& req) {
    json::Value body;
    if (!req.body.empty()) {
      try {
        body = json::parse(req.body);
      } catch (const std::exception&) {
      }
    }
    const std::string model = body.member_str("model");
    {
      const std::lock_guard<std::mutex> lock(mutex);
      paths.push_back(req.method + " " + req.path +
                      (req.query.empty() ? "" : "?" + req.query));
      keys.push_back(req.path + " " + req.header("X-Gufo-Internal"));
    }
    if (req.path == "/ready" || req.path == "/health") {
      json::Value v = json::Value::object();
      v["status"] = "ok";
      return JsonOk(v);
    }
    if (req.path == "/v1/models") {
      json::Value data = json::Value::array();
      for (const char* id :
           {"qwen3.8-27b", "qwen-image-2.1-turbo", "qwen3-tts",
            "wan2.2-ti2v-5b", "ace-step-1.5-xl-turbo", "triposg",
            "gufo-upscale", "m-fail"}) {
        json::Value m = json::Value::object();
        m["id"] = id;
        data.push_back(std::move(m));
      }
      json::Value v = json::Value::object();
      v["data"] = std::move(data);
      return JsonOk(v);
    }
    if (req.path == "/models/unload") {
      const std::lock_guard<std::mutex> lock(mutex);
      unloaded.push_back(model);
      return JsonOk(json::Value::object());
    }
    if (req.path == "/v1/chat/completions") {
      if (model == "m-fail")
        return Err(507, "ResourceExhausted: GPU memory");
      if (chat_delay_ms > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(chat_delay_ms));
      json::Value msg = json::Value::object();
      msg["role"] = "assistant";
      msg["content"] = "Die Sonne steigt langsam aus dem Meer.";
      json::Value choice = json::Value::object();
      choice["message"] = std::move(msg);
      json::Value choices = json::Value::array();
      choices.push_back(std::move(choice));
      json::Value usage = json::Value::object();
      usage["completion_tokens"] = 9;
      json::Value v = json::Value::object();
      v["choices"] = std::move(choices);
      v["usage"] = std::move(usage);
      return JsonOk(v);
    }
    if (req.path == "/v1/images/generations" ||
        req.path == "/v1/images/upscale") {
      if (body.member_str("size", "512x512") != "512x512" &&
          req.path == "/v1/images/generations")
        return Err(400, "size");
      json::Value d = json::Value::object();
      d["b64_json"] = gufo::core::EncodeBase64(einrichtung::TestImagePng());
      json::Value data = json::Value::array();
      data.push_back(std::move(d));
      json::Value v = json::Value::object();
      v["data"] = std::move(data);
      v["width"] = 128;
      v["height"] = 128;
      return JsonOk(v);
    }
    if (req.path == "/v1/audio/voices") {
      json::Value voices = json::Value::array();
      voices.push_back(json::Value("erzaehler"));
      json::Value v = json::Value::object();
      v["voices"] = std::move(voices);
      return JsonOk(v);
    }
    if (req.path == "/v1/audio/speech") {
      if (body.member_str("voice") != "erzaehler")
        return Err(400, "voice missing");
      return Wav();
    }
    if (req.path == "/v1/audio/music")
      return Wav();
    if (req.path == "/v1/videos" || req.path == "/v1/3d/generations") {
      json::Value v = json::Value::object();
      v["id"] = req.path == "/v1/videos" ? "vid_1" : "obj_1";
      v["status"] = "queued";
      return JsonOk(v, 202);
    }
    if (req.path == "/v1/videos/vid_1" || req.path == "/v1/3d/generations/obj_1") {
      int& polls = req.path == "/v1/videos/vid_1" ? video_polls : objekt_polls;
      int n = 0;
      {
        const std::lock_guard<std::mutex> lock(mutex);
        n = ++polls;
      }
      json::Value v = json::Value::object();
      v["id"] = req.path == "/v1/videos/vid_1" ? "vid_1" : "obj_1";
      v["status"] = n < 2 ? "running" : "completed";
      return JsonOk(v);
    }
    if (req.path == "/v1/videos/vid_1/content") {
      HttpResponse r;
      r.headers.emplace_back("Content-Type", "video/mp4");
      r.body = "MP4DATA";
      return r;
    }
    if (req.path == "/v1/3d/generations/obj_1/content") {
      HttpResponse r;
      r.headers.emplace_back("Content-Type", "model/gltf-binary");
      r.body = "GLBDATA";
      return r;
    }
    return HttpResponse{.status = 404, .reason = "Not Found", .body = "{}"};
  }
};

json::Value WaitSelfTest(einrichtung::SetupService& service,
                         std::chrono::milliseconds limit =
                             std::chrono::milliseconds(20000)) {
  const auto deadline = std::chrono::steady_clock::now() + limit;
  for (;;) {
    const auto r =
        service.Handle(Req("GET", "/router/setup/selbsttest", "127.0.0.1"));
    const json::Value body = Body(*r);
    if (!body.find("laeuft")->as_bool())
      return body;
    if (std::chrono::steady_clock::now() > deadline) {
      std::cerr << "WaitSelfTest: timeout\n";
      ++failures;
      return body;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

const json::Value* FindTest(const json::Value& body, std::string_view modell) {
  const json::Value* list = body.find("tests");
  if (list == nullptr)
    return nullptr;
  for (const json::Value& t : list->items())
    if (t.member_str("modell") == modell)
      return &t;
  return nullptr;
}

std::string TestStatus(const json::Value& body, std::string_view modell) {
  const json::Value* t = FindTest(body, modell);
  return t == nullptr ? std::string("(missing)") : t->member_str("status");
}

void TestSelfTestPlanning() {
  // The request per model.
  CHECK_EQ(einrichtung::TestKind("chat", "qwen3.8-27b"), std::string("chat"));
  CHECK_EQ(einrichtung::TestKind("bilder", "lora-anime-consistency"),
           std::string("keiner"));
  CHECK_EQ(einrichtung::TestKind("stimme", "qwen3-asr-1.7b"),
           std::string("keiner"));
  CHECK_EQ(einrichtung::TestKind("stimme", "qwen3-tts-clone"),
           std::string("keiner"));
  CHECK_EQ(einrichtung::TestKind("stimme", "fish-drama-3"),
           std::string("stimme"));
  CHECK_EQ(einrichtung::TestKind("video", "wan2.1-uni3c"), std::string("keiner"));
  CHECK_EQ(einrichtung::TestKind("video", "minimax-h3"), std::string("video"));
  CHECK_EQ(einrichtung::TestKind("werkzeuge", "gufo-rife"),
           std::string("keiner"));
  CHECK_EQ(einrichtung::TestKind("werkzeuge", "gufo-upscale"),
           std::string("upscale"));
  CHECK_EQ(einrichtung::TestKind("zweiter-rechner", "x"),
           std::string("partner"));

  // One case per block from the real catalogue: the required model, the
  // turbo one for pictures; unchosen models are left out; the partner only
  // with an address.
  const json::Value katalog = LoadJson(kDir / "katalog.json");
  const std::vector<std::string> all = {"chat", "bilder", "stimme", "video",
                                        "musik", "3d", "werkzeuge",
                                        "zweiter-rechner"};
  const std::vector<std::string> modelle = {
      "qwen3.8-27b", "qwen3.8-flash-next-q3", "qwen-image-2.1",
      "qwen-image-2.1-turbo", "qwen3-tts", "qwen3-tts-voicedesign",
      "qwen3-tts-clone", "qwen3-asr-1.7b", "wan2.2-ti2v-5b",
      "ace-step-1.5-xl-turbo", "triposg", "gufo-upscale", "gufo-rife"};
  std::vector<einrichtung::SelfTestCase> plan =
      einrichtung::PlanSelfTest(katalog, all, modelle, false, "");
  CHECK_EQ(plan.size(), std::size_t(7));
  if (plan.size() == 7) {
    CHECK_EQ(plan[0].modell, std::string("qwen3.8-27b"));
    CHECK_EQ(plan[1].modell, std::string("qwen-image-2.1-turbo"));
    CHECK_EQ(plan[2].modell, std::string("qwen3-tts"));
    CHECK_EQ(plan[3].modell, std::string("wan2.2-ti2v-5b"));
    CHECK_EQ(plan[4].modell, std::string("ace-step-1.5-xl-turbo"));
    CHECK_EQ(plan[5].modell, std::string("triposg"));
    CHECK_EQ(plan[6].modell, std::string("gufo-upscale"));
    CHECK_EQ(plan[6].art, std::string("upscale"));
  }
  plan = einrichtung::PlanSelfTest(katalog, all, modelle, true,
                                   "http://192.168.100.56:8081");
  // Every chosen model with a request: 27b, flash-next-q3, both pictures,
  // qwen3-tts, wan, ace, triposg, upscale, partner.
  CHECK_EQ(plan.size(), std::size_t(10));
  if (plan.size() == 10) {
    CHECK_EQ(plan[1].modell, std::string("qwen3.8-flash-next-q3"));
    CHECK_EQ(plan.back().art, std::string("partner"));
    CHECK_EQ(plan.back().modell, std::string("http://192.168.100.56:8081"));
  }
  plan = einrichtung::PlanSelfTest(katalog, {"chat", "nix"}, {"hemmingway-1.5"},
                                   false, "");
  CHECK_EQ(plan.size(), std::size_t(1));
  if (!plan.empty())
    CHECK_EQ(plan[0].modell, std::string("hemmingway-1.5"));

  // The start line of a written preset.
  const std::vector<std::string> args =
      einrichtung::SplitArgs("a \"b c\" d \"\" e");
  CHECK_EQ(args.size(), std::size_t(5));
  if (args.size() == 5) {
    CHECK_EQ(args[1], std::string("b c"));
    CHECK_EQ(args[3], std::string(""));
  }
  einrichtung::StartLine start;
  CHECK(!einrichtung::ParseStartLine("[router]\nauth = off\n", &start));
  const std::string header =
      "# baustein: router\n"
      "# start: \"C:\\gufo\\2.1\\bin\\gufo.exe\" router --models-preset "
      "\"C:\\gufo\\gufo-router.ini\" --models-max 0 --host 0.0.0.0 --port "
      "8081 --pages-dir \"C:\\gufo\\2.1\\pages\"\n"
      "# start-opt: --remote-allow 192.168.100.198\n"
      "# env: GUFO_HOST_RESERVE_MIB=2048\n"
      "# env: GUFO_QWEN27_LOOKUP=7\n"
      "# a comment\n"
      "[router]\n";
  CHECK(einrichtung::ParseStartLine(header, &start));
  CHECK_EQ(start.program, std::string("C:\\gufo\\2.1\\bin\\gufo.exe"));
  // router --models-preset <ini> --models-max 0 --pages-dir <dir>
  // --remote-allow <ip>: nine, --host/--port gone.
  CHECK_EQ(start.args.size(), std::size_t(9));
  if (start.args.size() == 9) {
    CHECK_EQ(start.args[0], std::string("router"));
    CHECK_EQ(start.args[2], std::string("C:\\gufo\\gufo-router.ini"));
    CHECK_EQ(start.args[3], std::string("--models-max"));
    CHECK_EQ(start.args[6], std::string("C:\\gufo\\2.1\\pages"));
    CHECK_EQ(start.args[7], std::string("--remote-allow"));
    CHECK_EQ(start.args[8], std::string("192.168.100.198"));
  }
  CHECK_EQ(start.env.size(), std::size_t(2));
  if (start.env.size() == 2) {
    CHECK_EQ(start.env[0].first, std::string("GUFO_HOST_RESERVE_MIB"));
    CHECK_EQ(start.env[1].second, std::string("7"));
  }

  // The built-in picture is a PNG with alpha.
  const std::string png = einrichtung::TestImagePng();
  CHECK(png.starts_with("\x89PNG"));
  CHECK(png.size() > 100);
}

void TestSelfTestService() {
  const std::filesystem::path tmp = TempDir();
  FakeRouter router;
  router.Start();
  Fixture f(auth::Mode::kOff);
  einrichtung::ServiceOptions options;
  options.dir = kDir;
  options.exe = tmp / "bin" / "gufo.exe";
  options.port = 8098;
  options.users_unbound = true;
  options.gate = f.gate.get();
  options.selbsttest_base_url = router.Base();
  // Review finding 1: the router's own loopback is called with its
  // internal key; the partner never gets it.
  options.selbsttest_headers = {"X-Gufo-Internal: interner-schluessel"};
  options.selbsttest_start_timeout_s = 5;
  options.selbsttest_test_timeout_s = 10;
  options.selbsttest_poll_ms = 20;
  einrichtung::SetupService service(options);

  // Before a preset: 409 no_preset on GET and POST.
  std::optional<HttpResponse> r =
      service.Handle(Req("GET", "/router/setup/selbsttest", "127.0.0.1"));
  CHECK(r.has_value() && r->status == 409);
  CHECK_EQ(ErrorCode(*r), "no_preset");
  r = service.Handle(Req("POST", "/router/setup/selbsttest", "127.0.0.1",
                         "{\"aktion\":\"start\",\"bausteine\":[\"chat\"]}"));
  CHECK(r.has_value() && r->status == 409);
  CHECK_EQ(ErrorCode(*r), "no_preset");
  r = service.Handle(Req("POST", "/router/setup/selbsttest", "127.0.0.1",
                         "{\"aktion\":\"abbrechen\"}"));
  CHECK(r.has_value() && r->status == 409);
  CHECK_EQ(ErrorCode(*r), "no_selbsttest");
  r = service.Handle(Req("POST", "/router/setup/selbsttest", "127.0.0.1",
                         "{\"aktion\":\"start\",\"preset\":\"relativ.ini\"}"));
  CHECK(r.has_value() && r->status == 400);
  CHECK_EQ(ErrorCode(*r), "invalid_path");

  // A preset with einrichtung.json next to it (as step 5 writes them).
  const std::filesystem::path daten = tmp / "daten";
  const std::filesystem::path preset = daten / "gufo-router.ini";
  WriteFile(preset, "# baustein: router\n[router]\nauth = off\n");
  {
    json::Value ein = json::Value::object();
    json::Value antworten = json::Value::object();
    json::Value bausteine = json::Value::array();
    for (const char* b : {"chat", "bilder", "stimme", "video", "musik", "3d",
                          "werkzeuge", "zweiter-rechner"})
      bausteine.push_back(json::Value(b));
    antworten["bausteine"] = std::move(bausteine);
    json::Value pfade = json::Value::object();
    pfade["daten"] = daten.string();
    antworten["pfade"] = std::move(pfade);
    json::Value partner = json::Value::object();
    partner["url"] = router.Base();
    antworten["partner"] = std::move(partner);
    ein["antworten"] = std::move(antworten);
    json::Value modelle = json::Value::array();
    for (const char* m : {"qwen3.8-27b", "qwen-image-2.1",
                          "qwen-image-2.1-turbo", "qwen3-tts",
                          "qwen3-asr-1.7b", "wan2.2-ti2v-5b",
                          "ace-step-1.5-xl-turbo", "triposg", "gufo-upscale"})
      modelle.push_back(json::Value(m));
    ein["modelle"] = std::move(modelle);
    WriteFile(daten / "einrichtung.json", ein.tojson() + "\n");
  }

  // Start with the preset named (the page's way after a restart); the
  // blocks come from einrichtung.json.
  router.chat_delay_ms = 300;
  r = service.Handle(Req("POST", "/router/setup/selbsttest", "127.0.0.1",
                         "{\"aktion\":\"start\",\"preset\":\"" +
                             Esc(preset.string()) + "\"}"));
  CHECK(r.has_value() && r->status == 202);
  {
    const json::Value body = Body(*r);
    CHECK(body.find("laeuft")->as_bool());
    CHECK_EQ(body.find("tests")->size(), std::size_t(8));
    CHECK_EQ(TestStatus(body, "qwen3.8-27b"), std::string("wartet"));
  }
  // A second start while it runs: 409.
  r = service.Handle(Req("POST", "/router/setup/selbsttest", "127.0.0.1",
                         "{\"aktion\":\"start\"}"));
  CHECK(r.has_value() && r->status == 409);
  CHECK_EQ(ErrorCode(*r), "selbsttest_running");
  json::Value done = WaitSelfTest(service);
  CHECK(done.find("fertig")->as_bool());
  CHECK_EQ(done.member_double("ok"), 8.0);
  CHECK_EQ(done.member_double("fehler"), 0.0);
  for (const char* m : {"qwen3.8-27b", "qwen-image-2.1-turbo", "qwen3-tts",
                        "wan2.2-ti2v-5b", "ace-step-1.5-xl-turbo", "triposg",
                        "gufo-upscale"})
    CHECK_EQ(TestStatus(done, m), std::string("ok"));
  CHECK_EQ(TestStatus(done, router.Base()), std::string("ok"));
  {
    const std::lock_guard<std::mutex> lock(router.mutex);
    std::size_t own = 0;
    std::size_t partner = 0;
    for (const std::string& k : router.keys) {
      if (k.starts_with("/health ")) {
        ++partner;
        CHECK_EQ(k, std::string("/health "));
      } else {
        ++own;
        CHECK(k.ends_with(" interner-schluessel"));
      }
    }
    CHECK(own > 8);
    CHECK(partner >= 1);
  }
  {
    const json::Value* chat = FindTest(done, "qwen3.8-27b");
    CHECK(chat != nullptr && chat->member_double("dauer_s") >= 0.3);
    CHECK(chat != nullptr &&
          chat->member_str("hinweis").find("Sonne") != std::string::npos);
    CHECK(chat != nullptr &&
          chat->member_str("datei") == "chat-qwen3.8-27b.txt");
  }
  const std::filesystem::path ordner = daten / "jobs" / "selbsttest";
  CHECK_EQ(done.member_str("ordner"), ordner.string());
  CHECK(std::filesystem::is_regular_file(ordner / "chat-qwen3.8-27b.txt"));
  CHECK(std::filesystem::is_regular_file(ordner / "bilder-qwen-image-2.1-turbo.png"));
  CHECK(std::filesystem::is_regular_file(ordner / "stimme-qwen3-tts.wav"));
  CHECK(std::filesystem::is_regular_file(ordner / "video-wan2.2-ti2v-5b.mp4"));
  CHECK(std::filesystem::is_regular_file(ordner / "musik-ace-step-1.5-xl-turbo.wav"));
  CHECK(std::filesystem::is_regular_file(ordner / "3d-triposg.glb"));
  CHECK(std::filesystem::is_regular_file(ordner / "werkzeuge-gufo-upscale.png"));
  CHECK(std::filesystem::is_regular_file(ordner / "selbsttest.json"));
  CHECK_EQ(ReadFile(ordner / "video-wan2.2-ti2v-5b.mp4"), std::string("MP4DATA"));
  CHECK_EQ(ReadFile(ordner / "3d-triposg.glb"), std::string("GLBDATA"));
  CHECK_EQ(ReadFile(ordner / "chat-qwen3.8-27b.txt"),
           std::string("Die Sonne steigt langsam aus dem Meer."));
  // Every model was unloaded after its case; the partner is no model.
  {
    const std::lock_guard<std::mutex> lock(router.mutex);
    CHECK_EQ(router.unloaded.size(), std::size_t(7));
    CHECK(std::find(router.paths.begin(), router.paths.end(),
                    "GET /v1/audio/voices?model=qwen3-tts") !=
          router.paths.end());
    CHECK(std::find(router.paths.begin(), router.paths.end(),
                    "GET /v1/videos/vid_1/content") != router.paths.end());
  }
  // The report in einrichtung.json.
  {
    const json::Value ein = LoadJson(daten / "einrichtung.json");
    const json::Value* st = ein.find("selbsttest");
    CHECK(st != nullptr && st->member_double("ok") == 8.0);
    CHECK(st != nullptr && !st->member_str("zeit_ende").empty());
    CHECK(ein.find("antworten") != nullptr);
  }
  // GET after the run keeps the result.
  r = service.Handle(Req("GET", "/router/setup/selbsttest", "127.0.0.1"));
  CHECK(r.has_value() && r->status == 200);
  CHECK(!Body(*r).find("laeuft")->as_bool());

  // Explicit cases: a memory failure is red with the filtered cause, a
  // model the target does not know is skipped, a model without a request
  // is skipped.
  router.chat_delay_ms = 0;
  r = service.Handle(Req(
      "POST", "/router/setup/selbsttest", "127.0.0.1",
      "{\"aktion\":\"start\",\"tests\":[{\"baustein\":\"chat\",\"modell\":"
      "\"m-fail\"},{\"baustein\":\"chat\",\"modell\":\"m-unknown\"},"
      "{\"baustein\":\"stimme\",\"modell\":\"qwen3-asr-1.7b\"},"
      "{\"baustein\":\"chat\",\"modell\":\"qwen3.8-27b\"}]}"));
  CHECK(r.has_value() && r->status == 202);
  done = WaitSelfTest(service);
  CHECK_EQ(TestStatus(done, "m-fail"), std::string("fehler"));
  CHECK_EQ(TestStatus(done, "m-unknown"), std::string("uebersprungen"));
  CHECK_EQ(TestStatus(done, "qwen3-asr-1.7b"), std::string("uebersprungen"));
  CHECK_EQ(TestStatus(done, "qwen3.8-27b"), std::string("ok"));
  {
    const json::Value* fail = FindTest(done, "m-fail");
    CHECK(fail != nullptr &&
          fail->member_str("hinweis").starts_with("zu wenig Speicher"));
    CHECK(fail != nullptr &&
          fail->member_str("log").find("ResourceExhausted") !=
              std::string::npos);
  }
  CHECK_EQ(done.member_double("ok"), 1.0);
  CHECK_EQ(done.member_double("fehler"), 1.0);

  // Cancel: the running case ends, the waiting ones are abgebrochen.
  router.chat_delay_ms = 2000;
  r = service.Handle(Req(
      "POST", "/router/setup/selbsttest", "127.0.0.1",
      "{\"aktion\":\"start\",\"tests\":[{\"baustein\":\"chat\",\"modell\":"
      "\"qwen3.8-27b\"},{\"baustein\":\"chat\",\"modell\":\"qwen3.8-27b\"}]}"));
  CHECK(r.has_value() && r->status == 202);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  r = service.Handle(Req("POST", "/router/setup/selbsttest", "127.0.0.1",
                         "{\"aktion\":\"abbrechen\"}"));
  CHECK(r.has_value() && r->status == 200);
  CHECK(Body(*r).find("abgebrochen")->as_bool());
  done = WaitSelfTest(service);
  CHECK(done.find("abgebrochen")->as_bool());
  CHECK_EQ(done.member_double("ok"), 0.0);
  {
    const json::Value* list = done.find("tests");
    CHECK(list != nullptr && list->size() == 2);
    if (list != nullptr && list->size() == 2) {
      CHECK_EQ(list->items()[0].member_str("status"), std::string("abgebrochen"));
      CHECK_EQ(list->items()[1].member_str("status"), std::string("abgebrochen"));
    }
  }

  // An unreachable target: every case red with the reason, nothing hangs.
  {
    einrichtung::ServiceOptions dead = options;
    dead.selbsttest_base_url = "http://127.0.0.1:9";
    dead.selbsttest_start_timeout_s = 1;
    einrichtung::SetupService service2(dead);
    r = service2.Handle(Req("POST", "/router/setup/selbsttest", "127.0.0.1",
                            "{\"aktion\":\"start\",\"preset\":\"" +
                                Esc(preset.string()) +
                                "\",\"bausteine\":[\"chat\"]}"));
    CHECK(r.has_value() && r->status == 202);
    done = WaitSelfTest(service2);
    CHECK_EQ(TestStatus(done, "qwen3.8-27b"), std::string("fehler"));
    CHECK(done.member_str("hinweis").find("antwortet nicht") !=
          std::string::npos);
  }
  std::error_code ec;
  std::filesystem::remove_all(tmp, ec);
}

#ifdef _WIN32
void TestSelfTestChildEnv() {
  // Review finding 2: the preset's `# env:` lines reach the child router
  // only, never this process's environment. The "router" is cmd.exe,
  // which writes the variable it sees into a file and exits.
  const char* comspec = std::getenv("ComSpec");
  if (comspec == nullptr || *comspec == '\0')
    return;
  const std::filesystem::path tmp = TempDir();
  Fixture f(auth::Mode::kOff);
  einrichtung::ServiceOptions options;
  options.dir = kDir;
  options.exe = tmp / "bin" / "gufo.exe";
  options.port = 8098;
  options.users_unbound = true;
  options.gate = f.gate.get();
  options.selbsttest_start_timeout_s = 5;
  options.selbsttest_poll_ms = 20;
  einrichtung::SetupService service(options);
  const std::filesystem::path probe =
      std::filesystem::current_path() / "selbsttest-env-probe.txt";
  std::error_code ec;
  std::filesystem::remove(probe, ec);
  const std::filesystem::path preset = tmp / "daten" / "gufo-router.ini";
  WriteFile(preset, std::string("# start: ") + comspec +
                        " /d /c echo %GUFO_SELBSTTEST_PROBE%>" +
                        probe.filename().string() +
                        "\n# env: GUFO_SELBSTTEST_PROBE=nur-im-kind\n"
                        "# baustein: router\n[router]\nauth = off\n");
  std::optional<HttpResponse> r = service.Handle(Req(
      "POST", "/router/setup/selbsttest", "127.0.0.1",
      "{\"aktion\":\"start\",\"preset\":\"" + Esc(preset.string()) +
          "\",\"tests\":[{\"baustein\":\"chat\",\"modell\":\"x\"}]}"));
  CHECK(r.has_value() && r->status == 202);
  const json::Value done = WaitSelfTest(service);
  CHECK(done.find("fertig")->as_bool());
  CHECK(std::getenv("GUFO_SELBSTTEST_PROBE") == nullptr);
  std::string seen;
  {
    std::ifstream in(probe, std::ios::binary);
    seen.assign(std::istreambuf_iterator<char>(in),
                std::istreambuf_iterator<char>());
  }
  CHECK(seen.starts_with("nur-im-kind"));
  if (!seen.starts_with("nur-im-kind"))
    std::cerr << "  probe: '" << seen << "'\n";
  std::filesystem::remove(probe, ec);
  std::filesystem::remove_all(tmp, ec);
}
#endif

int main() {
  TestClassifyAndGate();
  TestHelpers();
  TestMachineCheck();
  TestPlan();
  TestService();
  TestDownloads();
  TestRealDownload();
  TestSelfTestPlanning();
  TestSelfTestService();
#ifdef _WIN32
  TestSelfTestChildEnv();
#endif
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "router_einrichtung_test: ok\n";
  return 0;
}
