// The setup assistant's router side (src/cli/serve/router/einrichtung.*):
// the gate rules for /router/setup/* and /einrichtung/, the machine check,
// the catalogue, the preset built from the building blocks (against the
// real einrichtung/ folder and, when python3 is known, against
// einrichten.py --plan section by section like vergleich_preset.py), the
// foreign-block keeper and the service's endpoints with a temp folder.

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/auth.hpp"
#include "src/cli/serve/router/einrichtung.hpp"
#include "src/cli/serve/router/preset.hpp"
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
  CHECK(r.has_value() && r->status == 501);

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

int main() {
  TestClassifyAndGate();
  TestHelpers();
  TestMachineCheck();
  TestPlan();
  TestService();
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "router_einrichtung_test: ok\n";
  return 0;
}
