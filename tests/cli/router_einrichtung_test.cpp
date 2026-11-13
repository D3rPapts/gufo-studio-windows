// The setup assistant's router side (src/cli/serve/router/einrichtung.*):
// the gate rules for /router/setup/* and /einrichtung/, the machine check,
// the catalogue and the service's endpoints with a temp folder.

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

void TestService() {
  const std::filesystem::path tmp = TempDir();
  Fixture f(auth::Mode::kOff);
  einrichtung::ServiceOptions options;
  options.dir = kDir;
  options.exe = tmp / "bin" / "gufo.exe";
  options.port = 8098;
  options.users_unbound = true;
  options.gate = f.gate.get();
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

  // Step 3: folders checked and created.
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
  r = service.Handle(Req("POST", "/router/setup/pruefen", "127.0.0.1",
                         "{\"pfade\":{\"modelle\":\"modelle\",\"daten\":\"" +
                             esc(daten) + "\"}}"));
  CHECK(r.has_value() && r->status == 400);
  CHECK_EQ(ErrorCode(*r), "invalid_path");

  std::error_code ec;
  std::filesystem::remove_all(tmp, ec);
}

int main() {
  TestClassifyAndGate();
  TestHelpers();
  TestMachineCheck();
  TestService();
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "router_einrichtung_test: ok\n";
  return 0;
}
