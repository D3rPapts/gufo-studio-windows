// Users and access control (src/cli/serve/router/auth.*): the users file,
// PBKDF2 and token hashes, sessions, the gate's order (loopback -> bearer
// -> cookie -> 401/302), per-area rights, own-jobs-only, the pool secret,
// the login block after failed attempts, the word filter, the daily limit,
// the `[router]` preset section and `auth = off` passing everything through.
// Pure state, no sockets: the gate sees HttpRequest objects directly.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/auth.hpp"
#include "src/cli/serve/router/preset.hpp"
#include "src/core/json.hpp"

namespace {

namespace auth = gufo::router::auth;
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

constexpr long long kTestRounds = 1000;  // fast hashes for the tests

std::filesystem::path TempDir() {
  const auto base = std::filesystem::temp_directory_path() /
                    ("gufo-auth-test-" + auth::RandomHex(4));
  std::filesystem::create_directories(base);
  return base;
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

std::string Header(const HttpResponse& r, std::string_view name) {
  for (const auto& [k, v] : r.headers)
    if (k == name)
      return v;
  return "";
}

std::string CookieOf(const HttpResponse& r) {
  const std::string set = Header(r, "Set-Cookie");
  const auto semi = set.find(';');
  return set.substr(0, semi);
}

struct Fixture {
  std::filesystem::path dir = TempDir();
  std::shared_ptr<auth::UserStore> store =
      std::make_shared<auth::UserStore>(dir / "nutzer" / "nutzer.json",
                                        kTestRounds);
  std::unique_ptr<auth::Gate> gate;
  std::vector<std::string> seen;  // paths `next` received
  HttpRequest last;               // the request `next` received
  std::function<HttpResponse(const HttpRequest&)> next;

  explicit Fixture(auth::Mode mode = auth::Mode::kLan,
                   std::string pool_secret = {},
                   auth::LoopbackPolicy loopback = auth::LoopbackPolicy::kAdmin) {
    auth::Settings settings;
    settings.mode = mode;
    settings.loopback = loopback;
    settings.users_file = store->path();
    settings.pool_secret = std::move(pool_secret);
    auth::Gate::Hooks hooks;
    hooks.model_of = [](const HttpRequest& req) {
      try {
        return json::parse(req.body).member_str("model");
      } catch (const std::exception&) {
        return std::string();
      }
    };
    hooks.job_owner = [](const std::string& id) -> std::optional<std::string> {
      if (id == "j-tim")
        return std::string("tim");
      if (id == "j-anna")
        return std::string("anna");
      return std::nullopt;
    };
    gate = std::make_unique<auth::Gate>(settings, store, std::move(hooks));
    next = [this](const HttpRequest& req) {
      seen.push_back(req.path);
      last = req;
      json::Value body = json::Value::object();
      body["ok"] = true;
      return HttpResponse{.status = 200, .reason = "OK", .body = body.dump()};
    };
  }
  ~Fixture() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }

  void AddUser(const std::string& name, const std::string& role,
               const std::string& password, auth::Rights rights = {}) {
    auth::User user;
    user.name = name;
    user.rolle = role;
    user.rechte = std::move(rights);
    std::string error;
    CHECK(store->Put(user, &error));
    if (!password.empty())
      CHECK(store->SetPassword(name, password, &error));
  }

  HttpResponse Serve(const HttpRequest& req) { return gate->Serve(req, next); }

  /// Logs in and returns the cookie header value.
  std::string LoginCookie(const std::string& name, const std::string& pw,
                          bool remember = false, std::string client = "192.168.178.50") {
    json::Value body = json::Value::object();
    body["name"] = name;
    body["passwort"] = pw;
    body["bleiben"] = remember;
    const HttpResponse r = Serve(Req("POST", "/login", client, body.dump()));
    CHECK_EQ(r.status, 200);
    return CookieOf(r);
  }
};

// ---------------------------------------------------------------------------

void TestCrypto() {
  // RFC 7914 section 11 test vectors for PBKDF2-HMAC-SHA256.
  CHECK_EQ(auth::Pbkdf2Hex("passwd", "salt", 1).substr(0, 32),
           std::string("55ac046e56e3089fec1691c22544b605"));
  CHECK_EQ(auth::Pbkdf2Hex("Password", "NaCl", 80000).substr(0, 32),
           std::string("4ddcd8f60b98be21830cee5ef22701f9"));
  CHECK_EQ(auth::Sha256Hex("abc"),
           std::string("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff6"
                       "1f20015ad"));
  CHECK_EQ(auth::RandomHex(16).size(), std::size_t(32));
  CHECK(auth::RandomHex(16) != auth::RandomHex(16));
  CHECK(auth::ConstantTimeEquals("abc", "abc"));
  CHECK(!auth::ConstantTimeEquals("abc", "abd"));
  CHECK(!auth::ConstantTimeEquals("abc", "ab"));
  CHECK_EQ(auth::HexOf(auth::BytesOfHex("00ff10")), std::string("00ff10"));
  CHECK(auth::IsValidName("fabian"));
  CHECK(auth::IsValidName("film-56.a_b"));
  CHECK(!auth::IsValidName("Fabian"));
  CHECK(!auth::IsValidName(""));
  CHECK(!auth::IsValidName("a b"));
  CHECK(auth::IsLoopback("127.0.0.1"));
  CHECK(auth::IsLoopback("::1"));
  CHECK(auth::IsLoopback("::ffff:127.0.0.1"));
  CHECK(!auth::IsLoopback("192.168.178.50"));
  CHECK(!auth::IsLoopback(""));
}

void TestRights() {
  auth::Rights r;
  r.bereiche = {"text", "bild"};
  r.modelle = {"qwen-image-2.1"};
  r.filter = false;
  r.tageslimit = 7;
  r.pool = false;
  const auth::Rights back = auth::Rights::Decode(r.Encode());
  CHECK(back.bereiche == r.bereiche);
  CHECK(back.modelle == r.modelle);
  CHECK(back.filter.has_value() && !*back.filter);
  CHECK_EQ(back.tageslimit, 7LL);
  CHECK(!back.pool);
  CHECK(r.AllowsArea("text"));
  CHECK(!r.AllowsArea("video"));
  CHECK(r.AllowsArea("bild|video"));
  CHECK(r.AllowsArea(""));
  CHECK(r.AllowsModel("qwen-image-2.1"));
  CHECK(!r.AllowsModel("minimax-h3"));
  CHECK(r.AllowsModel(""));
  auth::Rights empty;
  CHECK(empty.AllowsArea("3d"));
  CHECK(empty.AllowsModel("x"));
  CHECK(empty.FilterOn("nutzer"));
  CHECK(!empty.FilterOn("admin"));
  CHECK(empty.pool);
  const auth::Rights from_json = auth::Rights::FromJson(r.ToJson());
  CHECK(from_json.bereiche == r.bereiche);
  CHECK(from_json.filter == r.filter);
  CHECK_EQ(from_json.tageslimit, 7LL);
  // Unknown areas in the header are dropped, not an error.
  CHECK(auth::Rights::Decode("bereiche=text,quatsch;pool=aus").bereiche ==
        std::vector<std::string>{"text"});
}

void TestStoreRoundTrip() {
  Fixture f;
  CHECK(!f.store->exists());
  CHECK(f.store->empty());
  CHECK(!f.store->HasAdmin());
  auth::Rights tim_rights;
  tim_rights.bereiche = {"text", "bild"};
  tim_rights.tageslimit = 20;
  f.AddUser("fabian", "admin", "geheim123");
  f.AddUser("tim", "nutzer", "tim-pw", tim_rights);
  f.AddUser("film-56", "dienst", "");
  CHECK(f.store->exists());
  CHECK(f.store->HasAdmin());
  CHECK(f.store->Verify("fabian", "geheim123").has_value());
  CHECK(!f.store->Verify("fabian", "falsch").has_value());
  CHECK(!f.store->Verify("niemand", "x").has_value());
  CHECK(!f.store->Verify("film-56", "").has_value());  // no password
  // Tokens.
  std::string error;
  const std::string token =
      f.store->CreateToken("film-56", "pipeline", 0, &error);
  CHECK(token.starts_with("gufo_"));
  CHECK_EQ(token.size(), std::size_t(5 + 64));
  CHECK(f.store->CreateToken("film-56", "pipeline", 0, &error).empty());
  CHECK(!error.empty());
  const auto by_token = f.store->FindByToken(token);
  CHECK(by_token.has_value() && by_token->name == "film-56");
  CHECK(!f.store->FindByToken("gufo_" + std::string(64, '0')).has_value());
  CHECK(!f.store->FindByToken("nicht-gufo").has_value());
  const std::string expired = f.store->CreateToken(
      "film-56", "alt", auth::NowUnix() - 10, &error);
  CHECK(!f.store->FindByToken(expired).has_value());
  // Only the hash is stored.
  {
    std::ifstream in(f.store->path(), std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    CHECK(text.find(token) == std::string::npos);
    CHECK(text.find(auth::Sha256Hex(token)) != std::string::npos);
    CHECK(text.find("geheim123") == std::string::npos);
    CHECK(text.find("pbkdf2-sha256") != std::string::npos);
  }
  // Remembered sessions.
  CHECK(f.store->RememberSession("tim", "s-abc", auth::NowUnix() + 100, &error));
  CHECK(f.store->FindBySession("s-abc").has_value());
  CHECK(!f.store->FindBySession("s-xyz").has_value());
  // A fresh store reads everything back.
  auth::UserStore again(f.store->path(), kTestRounds);
  CHECK(again.Load(&error));
  CHECK_EQ(again.List().size(), std::size_t(3));
  const auto tim = again.Find("tim");
  CHECK(tim.has_value());
  CHECK(tim->rechte.bereiche == tim_rights.bereiche);
  CHECK_EQ(tim->rechte.tageslimit, 20LL);
  CHECK(again.Verify("tim", "tim-pw").has_value());
  CHECK(again.FindByToken(token).has_value());
  CHECK(again.FindBySession("s-abc").has_value());
  again.ForgetSession("s-abc");
  CHECK(!again.FindBySession("s-abc").has_value());
  CHECK(again.RevokeToken("film-56", "pipeline", &error));
  CHECK(!again.FindByToken(token).has_value());
  CHECK(!again.RevokeToken("film-56", "pipeline", &error));
  // A new password ends remembered sessions and blocks the old one.
  CHECK(again.RememberSession("tim", "s-neu", auth::NowUnix() + 100, &error));
  CHECK(again.SetPassword("tim", "neu-pw", &error));
  CHECK(!again.FindBySession("s-neu").has_value());
  CHECK(!again.Verify("tim", "tim-pw").has_value());
  CHECK(again.Verify("tim", "neu-pw").has_value());
  // Blocked users verify nothing.
  auto blocked = *again.Find("tim");
  blocked.gesperrt = true;
  CHECK(again.Put(blocked, &error));
  CHECK(!again.Verify("tim", "neu-pw").has_value());
  CHECK(again.Remove("tim", &error));
  CHECK(!again.Find("tim").has_value());
  CHECK(!again.Remove("tim", &error));
  CHECK(!again.SetPassword("fabian", "abc", &error));  // too short
}

void TestOldFileFormat() {
  // A file from an older build: users without rechte, token, sitzungen,
  // gesperrt or timestamps still load with defaults.
  const auto dir = TempDir();
  const auto file = dir / "nutzer.json";
  {
    std::ofstream out(file, std::ios::binary);
    out << "{\"nutzer\": [{\"name\": \"Alt\", \"rolle\": \"admin\", "
           "\"passwort\": {\"salz\": \"00\", \"hash\": \"ff\", \"runden\": 1}},"
           " {\"name\": \"ohne\"}]}";
  }
  auth::UserStore store(file, kTestRounds);
  std::string error;
  CHECK(store.Load(&error));
  CHECK_EQ(store.List().size(), std::size_t(2));
  const auto alt = store.Find("alt");  // names are lower-cased
  CHECK(alt.has_value());
  CHECK(alt->admin());
  CHECK(alt->rechte.bereiche.empty());
  CHECK(alt->token.empty());
  CHECK(!alt->gesperrt);
  const auto ohne = store.Find("ohne");
  CHECK(ohne.has_value());
  CHECK_EQ(ohne->rolle, std::string("nutzer"));
  CHECK(ohne->hash_hex.empty());
  CHECK(!store.Verify("ohne", "x").has_value());
  // Unknown document shape: an error, not a crash.
  {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out << "[1, 2]";
  }
  CHECK(!store.Load(&error));
  {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out << "{not json";
  }
  CHECK(!store.Load(&error));
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

void TestClassify() {
  using Kind = auth::Route::Kind;
  CHECK(auth::Classify(Req("GET", "/health", "x")).kind == Kind::kFree);
  CHECK(auth::Classify(Req("GET", "/login/", "x")).kind == Kind::kFree);
  CHECK(auth::Classify(Req("GET", "/login/index.html", "x")).kind == Kind::kFree);
  CHECK(auth::Classify(Req("GET", "/gemeinsam/sprache.js", "x")).kind ==
        Kind::kFree);
  CHECK(auth::Classify(Req("POST", "/login", "x")).kind == Kind::kLogin);
  CHECK(auth::Classify(Req("GET", "/router/me", "x")).kind == Kind::kMe);
  CHECK(auth::Classify(Req("GET", "/router/users", "x")).kind == Kind::kUsers);
  CHECK(auth::Classify(Req("POST", "/router/users/tim/tokens", "x")).kind ==
        Kind::kUsers);
  CHECK(auth::Classify(Req("GET", "/verwaltung/", "x")).kind == Kind::kAdminPage);
  CHECK(auth::Classify(Req("GET", "/v1/jobs", "x")).kind == Kind::kJobs);
  CHECK(auth::Classify(Req("GET", "/v1/jobs/abc/files/a.png", "x")).kind ==
        Kind::kJobs);
  CHECK(auth::Classify(Req("POST", "/models/unload", "x")).kind ==
        Kind::kAdminControl);
  CHECK(auth::Classify(Req("POST", "/router/placement", "x")).kind ==
        Kind::kAdminControl);
  CHECK(auth::Classify(Req("GET", "/router/placement", "x")).kind ==
        Kind::kAnyUser);
  CHECK(auth::Classify(Req("GET", "/v1/models", "x")).kind == Kind::kAnyUser);
  CHECK(auth::Classify(Req("GET", "/router/status", "x")).kind == Kind::kAnyUser);
  const auto chat = auth::Classify(Req("POST", "/v1/chat/completions", "x"));
  CHECK(chat.kind == Kind::kWork);
  CHECK_EQ(chat.area, std::string("text"));
  CHECK_EQ(chat.counts, 1);
  CHECK(!chat.filtered);
  const auto image = auth::Classify(Req("POST", "/v1/images/generations", "x"));
  CHECK_EQ(image.area, std::string("bild"));
  CHECK(image.filtered);
  const auto video = auth::Classify(Req("POST", "/v1/videos", "x"));
  CHECK_EQ(video.area, std::string("video"));
  CHECK_EQ(video.counts, 2);
  CHECK(video.filtered);
  const auto video_get = auth::Classify(Req("GET", "/v1/videos/v1", "x"));
  CHECK_EQ(video_get.counts, 0);
  CHECK(!video_get.filtered);
  CHECK_EQ(auth::Classify(Req("POST", "/v1/videos/concat", "x")).area,
           std::string("video"));
  CHECK_EQ(auth::Classify(Req("POST", "/v1/audio/speech", "x")).area,
           std::string("stimme"));
  CHECK_EQ(auth::Classify(Req("POST", "/v1/audio/transcriptions", "x")).area,
           std::string("stimme"));
  CHECK_EQ(auth::Classify(Req("POST", "/v1/audio/music", "x")).area,
           std::string("musik"));
  CHECK_EQ(auth::Classify(Req("POST", "/v1/3d/generations", "x")).area,
           std::string("3d"));
  CHECK_EQ(auth::Classify(Req("GET", "/v1/previews/p1", "x")).area,
           std::string("bild|video"));
  CHECK_EQ(auth::Classify(Req("POST", "/prompt", "x")).area, std::string("bild"));
  CHECK_EQ(auth::Classify(Req("POST", "/v1/figuren", "x")).area,
           std::string("text"));
  const auto page = auth::Classify(Req("GET", "/bild/", "x"));
  CHECK(page.kind == Kind::kPage);
  CHECK_EQ(page.area, std::string("bild"));
  CHECK_EQ(auth::Classify(Req("GET", "/lipsync/", "x")).area, std::string("video"));
  CHECK_EQ(auth::Classify(Req("GET", "/pool/", "x")).area, std::string("status"));
  CHECK_EQ(auth::Classify(Req("GET", "/start/", "x")).area, std::string(""));
  CHECK(auth::Classify(Req("GET", "/", "x")).kind == Kind::kPage);
  CHECK(auth::Classify(Req("POST", "/v1/irgendwas", "x")).kind == Kind::kOther);
}

void TestGateOrder() {
  Fixture f;
  f.AddUser("fabian", "admin", "geheim123");
  auth::Rights tim_rights;
  tim_rights.bereiche = {"text", "ablage"};
  f.AddUser("tim", "nutzer", "tim-pw", tim_rights);
  f.AddUser("film-56", "dienst", "");
  std::string error;
  const std::string token = f.store->CreateToken("film-56", "p", 0, &error);

  // Loopback is admin without a login.
  const auto lokal = f.gate->Identify(Req("GET", "/v1/models", "127.0.0.1"));
  CHECK(lokal.source == auth::Identity::Source::kLoopback);
  CHECK(lokal.admin());
  CHECK_EQ(lokal.name, std::string("lokal"));
  // LAN without anything: 401 for the API, 302 for pages.
  HttpResponse r = f.Serve(Req("POST", "/v1/chat/completions", "192.168.178.50",
                               "{\"model\":\"m\"}"));
  CHECK_EQ(r.status, 401);
  CHECK_EQ(ErrorCode(r), std::string("not_authenticated"));
  r = f.Serve(Req("GET", "/bild/?x=1", "192.168.178.50"));
  CHECK_EQ(r.status, 302);
  CHECK_EQ(Header(r, "Location"), std::string("/login/?zurueck=/bild/%3Fx%3D1"));
  // Free paths pass without a login.
  r = f.Serve(Req("GET", "/health", "192.168.178.50"));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("GET", "/login/", "192.168.178.50"));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("GET", "/gemeinsam/sprache.js", "192.168.178.50"));
  CHECK_EQ(r.status, 200);
  // Bearer.
  const auto bearer = f.gate->Identify(
      Req("POST", "/v1/videos", "192.168.178.50", "",
          {{"Authorization", "Bearer " + token}}));
  CHECK(bearer.source == auth::Identity::Source::kBearer);
  CHECK_EQ(bearer.name, std::string("film-56"));
  CHECK_EQ(bearer.rolle, std::string("dienst"));
  r = f.Serve(Req("POST", "/v1/videos", "192.168.178.50", "{\"model\":\"h3\"}",
                  {{"Authorization", "Bearer gufo_" + std::string(64, 'a')}}));
  CHECK_EQ(r.status, 401);
  CHECK_EQ(ErrorCode(r), std::string("invalid_token"));
  // A bearer beats a cookie; a bearer wins over loopback too (a token on
  // the machine itself runs as its user).
  const auto token_local = f.gate->Identify(
      Req("GET", "/v1/models", "127.0.0.1", "",
          {{"Authorization", "Bearer " + token}}));
  CHECK_EQ(token_local.name, std::string("film-56"));
  // Service accounts cannot log in with a password.
  r = f.Serve(Req("POST", "/login", "192.168.178.50",
                  "{\"name\":\"film-56\",\"passwort\":\"x\"}"));
  CHECK_EQ(r.status, 401);  // no password set -> login_failed
  // Cookie login.
  const std::string cookie = f.LoginCookie("tim", "tim-pw");
  CHECK(cookie.starts_with("gufo_sitzung=s"));
  r = f.Serve(Req("GET", "/router/me", "192.168.178.50", "", {{"Cookie", cookie}}));
  CHECK_EQ(r.status, 200);
  {
    const json::Value me = json::parse(r.body);
    CHECK_EQ(me.member_str("name"), std::string("tim"));
    CHECK_EQ(me.member_str("rolle"), std::string("nutzer"));
    CHECK_EQ(me.member_str("auth"), std::string("lan"));
    CHECK(me.find("filter")->as_bool());
    CHECK_EQ(me.find("bereiche")->size(), std::size_t(2));
    CHECK(!me.find("verwaltung")->as_bool());
  }
  // Case-insensitive header, other cookies around it.
  r = f.Serve(Req("GET", "/router/me", "192.168.178.50", "",
                  {{"cookie", "a=b; " + cookie + "; c=d"}}));
  CHECK_EQ(r.status, 200);
  // The checked request reaching the router carries the user headers and
  // no client-supplied copies.
  f.seen.clear();
  r = f.Serve(Req("POST", "/v1/chat/completions", "192.168.178.50",
                  "{\"model\":\"m\"}",
                  {{"Cookie", cookie}, {"X-Gufo-User", "fabian"},
                   {"X-Gufo-Rights", "rolle=admin"}}));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(f.seen.size(), std::size_t(1));
  CHECK_EQ(f.last.header("X-Gufo-User"), std::string("tim"));
  CHECK(f.last.header("X-Gufo-Rights").starts_with("rolle=nutzer;bereiche=text,ablage"));
  int user_headers = 0;
  for (const auto& [k, v] : f.last.headers)
    if (k == "X-Gufo-User")
      ++user_headers;
  CHECK_EQ(user_headers, 1);
  // Logout ends the session.
  r = f.Serve(Req("POST", "/logout", "192.168.178.50", "", {{"Cookie", cookie}}));
  CHECK_EQ(r.status, 200);
  CHECK(Header(r, "Set-Cookie").find("Max-Age=0") != std::string::npos);
  r = f.Serve(Req("GET", "/router/me", "192.168.178.50", "", {{"Cookie", cookie}}));
  CHECK_EQ(r.status, 401);
  // "Stay signed in" survives a new gate (the hash is in the file).
  const std::string remembered = f.LoginCookie("tim", "tim-pw", true);
  CHECK(Header(f.Serve(Req("POST", "/login", "192.168.178.50",
                           "{\"name\":\"tim\",\"passwort\":\"tim-pw\",\"bleiben\":true}")),
               "Set-Cookie")
            .find("Max-Age=2592000") != std::string::npos);
  {
    auth::Settings settings = f.gate->settings();
    auth::Gate fresh(settings, f.store, {});
    const auto who = fresh.Identify(
        Req("GET", "/router/me", "192.168.178.50", "", {{"Cookie", remembered}}));
    CHECK(who.source == auth::Identity::Source::kCookie);
    CHECK_EQ(who.name, std::string("tim"));
    // A plain session does not.
    const std::string plain = f.LoginCookie("tim", "tim-pw", false);
    CHECK(!fresh.Identify(Req("GET", "/router/me", "192.168.178.50", "",
                              {{"Cookie", plain}}))
               .authenticated());
  }
}

void TestAreasAndJobs() {
  Fixture f;
  f.AddUser("fabian", "admin", "geheim123");
  auth::Rights tim_rights;
  tim_rights.bereiche = {"text", "ablage"};
  tim_rights.modelle = {"qwen3.5-4b"};
  f.AddUser("tim", "nutzer", "tim-pw", tim_rights);
  const std::string tim = f.LoginCookie("tim", "tim-pw");
  const std::string admin = f.LoginCookie("fabian", "geheim123");
  // 403 per area with the concept's message.
  HttpResponse r = f.Serve(Req("POST", "/v1/images/generations", "192.168.178.50",
                               "{\"model\":\"qwen-image-2.1\",\"prompt\":\"Haus\"}",
                               {{"Cookie", tim}}));
  CHECK_EQ(r.status, 403);
  CHECK_EQ(ErrorCode(r), std::string("forbidden"));
  CHECK(r.body.find("Bereich bild ist f") != std::string::npos);
  for (const char* path : {"/v1/videos", "/v1/audio/speech", "/v1/audio/music",
                           "/v1/3d/generations"}) {
    r = f.Serve(Req("POST", path, "192.168.178.50", "{\"model\":\"m\"}",
                    {{"Cookie", tim}}));
    CHECK_EQ(r.status, 403);
  }
  // Pages of a closed area: 403, open ones: through.
  r = f.Serve(Req("GET", "/bild/", "192.168.178.50", "", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 403);
  r = f.Serve(Req("GET", "/chat/", "192.168.178.50", "", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("GET", "/start/", "192.168.178.50", "", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 200);
  // Model list.
  r = f.Serve(Req("POST", "/v1/chat/completions", "192.168.178.50",
                  "{\"model\":\"qwen3.8-27b\"}", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 403);
  CHECK_EQ(ErrorCode(r), std::string("model_forbidden"));
  r = f.Serve(Req("POST", "/v1/chat/completions", "192.168.178.50",
                  "{\"model\":\"qwen3.5-4b\"}", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 200);
  // Admin control is admin only.
  r = f.Serve(Req("POST", "/models/unload", "192.168.178.50", "{}",
                  {{"Cookie", tim}}));
  CHECK_EQ(r.status, 403);
  r = f.Serve(Req("POST", "/models/unload", "192.168.178.50", "{}",
                  {{"Cookie", admin}}));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("POST", "/router/pool/pull", "192.168.178.50", "{}",
                  {{"Cookie", tim}}));
  CHECK_EQ(r.status, 403);
  // Admin sees everything, even a closed-looking area.
  r = f.Serve(Req("POST", "/v1/images/generations", "192.168.178.50",
                  "{\"model\":\"qwen-image-2.1\",\"prompt\":\"Haus\"}",
                  {{"Cookie", admin}}));
  CHECK_EQ(r.status, 200);
  // Jobs: own only.
  r = f.Serve(Req("GET", "/v1/jobs/j-tim", "192.168.178.50", "", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("GET", "/v1/jobs/j-anna", "192.168.178.50", "", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 404);
  r = f.Serve(Req("DELETE", "/v1/jobs/j-anna", "192.168.178.50", "", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 404);
  r = f.Serve(Req("GET", "/v1/jobs/j-anna/files/a.png", "192.168.178.50", "",
                  {{"Cookie", tim}}));
  CHECK_EQ(r.status, 404);
  r = f.Serve(Req("GET", "/v1/jobs/j-anna", "192.168.178.50", "", {{"Cookie", admin}}));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("DELETE", "/v1/jobs?older_than=3600", "192.168.178.50", "",
                  {{"Cookie", tim}}));
  CHECK_EQ(r.status, 403);
  // A submitted job is stamped with its owner and its inner request is
  // checked like a direct one.
  f.seen.clear();
  r = f.Serve(Req("POST", "/v1/jobs", "192.168.178.50",
                  "{\"request\":{\"path\":\"/v1/chat/completions\",\"body\":"
                  "{\"model\":\"qwen3.5-4b\"}},\"metadata\":{\"titel\":\"t\"}}",
                  {{"Cookie", tim}}));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(f.seen.size(), std::size_t(1));
  {
    const json::Value sent = json::parse(f.last.body);
    CHECK_EQ(sent.find("metadata")->member_str("nutzer"), std::string("tim"));
    CHECK_EQ(sent.find("metadata")->member_str("titel"), std::string("t"));
  }
  r = f.Serve(Req("POST", "/v1/jobs", "192.168.178.50",
                  "{\"request\":{\"path\":\"/v1/images/generations\",\"body\":"
                  "{\"model\":\"qwen-image-2.1\",\"prompt\":\"Haus\"}}}",
                  {{"Cookie", tim}}));
  CHECK_EQ(r.status, 403);
  // The job list of a user shows only its own jobs; a job read of another
  // user's job (unknown to the owner hook, e.g. the partner's) is 404
  // after the fact.
  f.next = [](const HttpRequest& req) {
    json::Value body = json::Value::object();
    if (req.path == "/v1/jobs") {
      json::Value data = json::Value::array();
      for (const char* owner : {"tim", "anna", ""}) {
        json::Value job = json::Value::object();
        job["id"] = std::string("j-") + owner;
        json::Value metadata = json::Value::object();
        if (*owner != '\0')
          metadata["nutzer"] = owner;
        job["metadata"] = std::move(metadata);
        data.push_back(std::move(job));
      }
      body["data"] = std::move(data);
    } else {
      body["id"] = "j-fremd";
      json::Value metadata = json::Value::object();
      metadata["nutzer"] = "anna";
      body["metadata"] = std::move(metadata);
    }
    return HttpResponse{.status = 200, .reason = "OK", .body = body.dump()};
  };
  r = f.Serve(Req("GET", "/v1/jobs?limit=10", "192.168.178.50", "", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 200);
  {
    const json::Value list = json::parse(r.body);
    CHECK_EQ(list.find("data")->size(), std::size_t(1));
    CHECK_EQ(list.find("data")->items()[0].member_str("id"), std::string("j-tim"));
  }
  r = f.Serve(Req("GET", "/v1/jobs", "192.168.178.50", "", {{"Cookie", admin}}));
  CHECK_EQ(json::parse(r.body).find("data")->size(), std::size_t(3));
  r = f.Serve(Req("GET", "/v1/jobs/j-fremd", "192.168.178.50", "", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 404);
  r = f.Serve(Req("GET", "/v1/jobs/j-fremd", "192.168.178.50", "", {{"Cookie", admin}}));
  CHECK_EQ(r.status, 200);
  // pool = aus pins the request to this machine.
  auth::Rights no_pool = tim_rights;
  no_pool.pool = false;
  auto user = *f.store->Find("tim");
  user.rechte = no_pool;
  std::string error;
  CHECK(f.store->Put(user, &error));
  f.next = [&f](const HttpRequest& req) {
    f.last = req;
    return HttpResponse{.status = 200, .reason = "OK", .body = "{}"};
  };
  r = f.Serve(Req("POST", "/v1/chat/completions", "192.168.178.50",
                  "{\"model\":\"qwen3.5-4b\"}",
                  {{"Cookie", tim}, {"X-Gufo-Pool", "remote"}}));
  CHECK_EQ(f.last.header("X-Gufo-Pool"), std::string("local"));
}

void TestPoolSecret() {
  Fixture f(auth::Mode::kLan, "sehr-geheim");
  f.AddUser("fabian", "admin", "geheim123");
  // Without the key a partner call from the LAN is just an anonymous
  // request: 401, so the partner marks this router unreachable.
  HttpResponse r = f.Serve(Req("GET", "/router/capabilities", "192.168.100.56", "",
                               {{"X-Gufo-Forwarded", "r-56"}}));
  CHECK_EQ(r.status, 401);
  r = f.Serve(Req("GET", "/v1/models", "192.168.100.56", "",
                  {{"X-Gufo-Pool-Key", "falsch"}}));
  CHECK_EQ(r.status, 401);
  CHECK_EQ(ErrorCode(r), std::string("pool_key_invalid"));
  // A wrong key never falls back to loopback.
  r = f.Serve(Req("GET", "/v1/models", "127.0.0.1", "",
                  {{"X-Gufo-Pool-Key", "falsch"}}));
  CHECK_EQ(r.status, 401);
  // With the key: trusted, under the relayed user, rights taken from the
  // header, nothing checked again, metadata stamped with that user.
  f.seen.clear();
  r = f.Serve(Req("POST", "/v1/jobs", "192.168.100.56",
                  "{\"request\":{\"path\":\"/v1/images/generations\",\"body\":{}}}",
                  {{"X-Gufo-Pool-Key", "sehr-geheim"},
                   {"X-Gufo-User", "tim"},
                   {"X-Gufo-Rights", "rolle=nutzer;bereiche=text"}}));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(f.seen.size(), std::size_t(1));
  CHECK_EQ(json::parse(f.last.body).find("metadata")->member_str("nutzer"),
           std::string("tim"));
  const auto who = f.gate->Identify(Req("GET", "/v1/models", "192.168.100.56", "",
                                        {{"X-Gufo-Pool-Key", "sehr-geheim"}}));
  CHECK(who.source == auth::Identity::Source::kPool);
  CHECK_EQ(who.name, std::string("partner"));
  CHECK_EQ(who.rolle, std::string("dienst"));
  // Outgoing: RemoteTarget adds the key when the process secret is set.
  auth::SetPoolSecret("sehr-geheim");
  CHECK_EQ(auth::PoolSecret(), std::string("sehr-geheim"));
  auth::SetPoolSecret("");
  // No secret configured: any key is wrong.
  Fixture g;
  g.AddUser("fabian", "admin", "geheim123");
  r = g.Serve(Req("GET", "/v1/models", "192.168.100.56", "",
                  {{"X-Gufo-Pool-Key", "sehr-geheim"}}));
  CHECK_EQ(r.status, 401);
  // The router's own calls (internal key, loopback only) run for the
  // job's owner.
  const std::string key = g.gate->internal_key();
  const auto own = g.gate->Identify(Req("POST", "/v1/images/generations", "127.0.0.1",
                                        "{}", {{"X-Gufo-Internal", key},
                                               {"X-Gufo-Job", "j-tim"}}));
  CHECK(own.source == auth::Identity::Source::kInternal);
  CHECK_EQ(own.name, std::string("tim"));
  CHECK(own.admin());
  CHECK(!g.gate->Identify(Req("POST", "/v1/images/generations", "192.168.178.50",
                              "{}", {{"X-Gufo-Internal", key}}))
             .authenticated());
}

void TestFailedLogins() {
  Fixture f;
  f.AddUser("fabian", "admin", "geheim123");
  long long now = 1'000'000;
  f.gate->SetClock([&now] { return now; });
  const auto attempt = [&f](const std::string& pw) {
    return f.Serve(Req("POST", "/login", "192.168.178.77",
                       "{\"name\":\"fabian\",\"passwort\":\"" + pw + "\"}"));
  };
  for (int i = 0; i < 5; ++i) {
    const HttpResponse r = attempt("falsch");
    CHECK_EQ(r.status, 401);
    CHECK_EQ(ErrorCode(r), std::string("login_failed"));
    now += 5;
  }
  HttpResponse r = attempt("geheim123");
  CHECK_EQ(r.status, 429);
  CHECK_EQ(ErrorCode(r), std::string("too_many_attempts"));
  CHECK(!Header(r, "Retry-After").empty());
  // Another address is not blocked.
  r = f.Serve(Req("POST", "/login", "192.168.178.78",
                  "{\"name\":\"fabian\",\"passwort\":\"geheim123\"}"));
  CHECK_EQ(r.status, 200);
  // After 60 s the block ends.
  now += 61;
  r = attempt("geheim123");
  CHECK_EQ(r.status, 200);
  // Four failures spread over more than a minute never block.
  for (int i = 0; i < 4; ++i) {
    attempt("falsch");
    now += 20;
  }
  r = attempt("geheim123");
  CHECK_EQ(r.status, 200);
  // The log has the failures, no passwords.
  const auto lines = f.gate->LogTail(100);
  CHECK(lines.size() >= 10);
  bool found = false;
  for (const std::string& line : lines) {
    CHECK(line.find("falsch") == std::string::npos);
    CHECK(line.find("geheim123") == std::string::npos);
    found = found || line.find("login_fehlgeschlagen") != std::string::npos;
  }
  CHECK(found);
}

void TestFilterAndLimit() {
  Fixture f;
  f.AddUser("fabian", "admin", "geheim123");
  auth::Rights limited;
  limited.tageslimit = 3;
  f.AddUser("kind", "nutzer", "kind-pw", limited);
  auth::Rights free_rights;
  free_rights.filter = false;
  f.AddUser("tim", "nutzer", "tim-pw", free_rights);
  CHECK(std::filesystem::is_regular_file(f.dir / "nutzer" / "filter-woerter.txt"));
  CHECK(f.gate->filter().size() >= 20);
  const std::string kind = f.LoginCookie("kind", "kind-pw");
  const std::string tim = f.LoginCookie("tim", "tim-pw");
  const std::string admin = f.LoginCookie("fabian", "geheim123");
  const auto image = [&](const std::string& cookie, const std::string& body) {
    return f.Serve(Req("POST", "/v1/images/generations", "192.168.178.50", body,
                       {{"Cookie", cookie}}));
  };
  HttpResponse r = image(kind, "{\"model\":\"m\",\"prompt\":\"Ein Haus am See\"}");
  CHECK_EQ(r.status, 200);
  r = image(kind, "{\"model\":\"m\",\"prompt\":\"Eine NACKTE Frau\"}");
  CHECK_EQ(r.status, 400);
  CHECK_EQ(ErrorCode(r), std::string("content_filtered"));
  r = image(kind, "{\"model\":\"m\",\"prompt\":\"Haus\",\"negative_prompt\":\"blood\"}");
  CHECK_EQ(r.status, 400);
  // Whole words: "Sextett" is not "sex"? sex* is a stem entry, so it is
  // caught; "Rosexy" is not (no word start).
  r = image(kind, "{\"model\":\"m\",\"prompt\":\"Rosexy\"}");
  CHECK_EQ(r.status, 200);
  r = image(kind, "{\"model\":\"m\",\"prompt\":\"nsfw-bild\"}");
  CHECK_EQ(r.status, 400);
  // The filter is per user: off for tim and for admins.
  r = image(tim, "{\"model\":\"m\",\"prompt\":\"nackt\"}");
  CHECK_EQ(r.status, 200);
  r = image(admin, "{\"model\":\"m\",\"prompt\":\"nackt\"}");
  CHECK_EQ(r.status, 200);
  // Videos and the ComfyUI graph are filtered too; chats are not.
  r = f.Serve(Req("POST", "/v1/videos", "192.168.178.50",
                  "{\"model\":\"m\",\"prompt\":\"porno\"}", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 200);  // tim: filter off
  Fixture g;
  g.AddUser("kind", "nutzer", "kind-pw");
  const std::string kind2 = g.LoginCookie("kind", "kind-pw");
  r = g.Serve(Req("POST", "/v1/videos", "192.168.178.50",
                  "{\"model\":\"m\",\"prompt\":\"porno\"}", {{"Cookie", kind2}}));
  CHECK_EQ(r.status, 400);
  r = g.Serve(Req("POST", "/prompt", "192.168.178.50",
                  "{\"prompt\":{\"3\":{\"inputs\":{\"text\":\"a nude figure\"}}}}",
                  {{"Cookie", kind2}}));
  CHECK_EQ(r.status, 400);
  r = g.Serve(Req("POST", "/v1/chat/completions", "192.168.178.50",
                  "{\"model\":\"m\",\"messages\":[{\"content\":\"nackt\"}]}",
                  {{"Cookie", kind2}}));
  CHECK_EQ(r.status, 200);
  // Multipart edits: the prompt field, not the image bytes.
  const std::string multipart =
      "--xyz\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm\r\n"
      "--xyz\r\nContent-Disposition: form-data; name=\"image\"; "
      "filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\nnackt\r\n"
      "--xyz\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\n"
      "Ein Haus\r\n--xyz--\r\n";
  r = g.Serve(Req("POST", "/v1/images/edits", "192.168.178.50", multipart,
                  {{"Cookie", kind2},
                   {"Content-Type", "multipart/form-data; boundary=xyz"}}));
  CHECK_EQ(r.status, 200);
  const std::string bad_multipart =
      "--xyz\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\n"
      "ganz nackt\r\n--xyz--\r\n";
  r = g.Serve(Req("POST", "/v1/images/edits", "192.168.178.50", bad_multipart,
                  {{"Cookie", kind2},
                   {"Content-Type", "multipart/form-data; boundary=xyz"}}));
  CHECK_EQ(r.status, 400);
  // An edited word list is picked up (newer mtime).
  {
    std::ofstream out(g.dir / "nutzer" / "filter-woerter.txt",
                      std::ios::binary | std::ios::trunc);
    out << "# nur ein Wort\nbanane\n";
  }
  std::filesystem::last_write_time(
      g.dir / "nutzer" / "filter-woerter.txt",
      std::filesystem::file_time_type::clock::now() + std::chrono::seconds(2));
  CHECK_EQ(g.gate->filter().size(), std::size_t(1));
  r = g.Serve(Req("POST", "/v1/images/generations", "192.168.178.50",
                  "{\"model\":\"m\",\"prompt\":\"nackt\"}", {{"Cookie", kind2}}));
  CHECK_EQ(r.status, 200);
  r = g.Serve(Req("POST", "/v1/images/generations", "192.168.178.50",
                  "{\"model\":\"m\",\"prompt\":\"eine Banane\"}", {{"Cookie", kind2}}));
  CHECK_EQ(r.status, 400);
  // Daily limit: 3 per day, a video counts twice; refused requests (the
  // two filtered ones above) never counted.
  CHECK_EQ(f.gate->DailyCount("kind"), 2LL);  // "Haus am See", "Rosexy"
  f.AddUser("knapp", "nutzer", "knapp-pw", limited);
  const std::string knapp = f.LoginCookie("knapp", "knapp-pw");
  r = f.Serve(Req("POST", "/v1/chat/completions", "192.168.178.50",
                  "{\"model\":\"m\"}", {{"Cookie", knapp}}));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(f.gate->DailyCount("knapp"), 1LL);
  r = f.Serve(Req("POST", "/v1/videos", "192.168.178.50",
                  "{\"model\":\"m\",\"prompt\":\"See\"}", {{"Cookie", knapp}}));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(f.gate->DailyCount("knapp"), 3LL);
  r = f.Serve(Req("POST", "/v1/chat/completions", "192.168.178.50",
                  "{\"model\":\"m\"}", {{"Cookie", knapp}}));
  CHECK_EQ(r.status, 429);
  CHECK_EQ(ErrorCode(r), std::string("daily_limit"));
  // Reads never count.
  r = f.Serve(Req("GET", "/v1/videos/v1", "192.168.178.50", "", {{"Cookie", knapp}}));
  CHECK_EQ(r.status, 200);
  // A new day resets (the 12 h session is over by then: log in again).
  long long now = auth::NowUnix() + 86400;
  f.gate->SetClock([&now] { return now; });
  r = f.Serve(Req("GET", "/router/me", "192.168.178.50", "", {{"Cookie", knapp}}));
  CHECK_EQ(r.status, 401);  // session expired
  const std::string knapp2 = f.LoginCookie("knapp", "knapp-pw");
  r = f.Serve(Req("POST", "/v1/chat/completions", "192.168.178.50",
                  "{\"model\":\"m\"}", {{"Cookie", knapp2}}));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(f.gate->DailyCount("knapp"), 1LL);
}

void TestUsersApi() {
  Fixture f;
  // First setup: nothing but loopback gets in; the LAN sees the hint.
  HttpResponse r = f.Serve(Req("GET", "/router/me", "192.168.178.50"));
  CHECK_EQ(r.status, 401);
  CHECK_EQ(ErrorCode(r), std::string("no_admin"));
  CHECK(json::parse(r.body).find("setup")->as_bool());
  r = f.Serve(Req("POST", "/v1/chat/completions", "192.168.178.50", "{}"));
  CHECK_EQ(r.status, 401);
  CHECK_EQ(ErrorCode(r), std::string("no_admin"));
  r = f.Serve(Req("GET", "/router/users", "192.168.178.50"));
  CHECK_EQ(r.status, 404);
  r = f.Serve(Req("GET", "/verwaltung/", "192.168.178.50"));
  CHECK_EQ(r.status, 404);  // invisible from outside, even the login hint
  r = f.Serve(Req("GET", "/router/users", "127.0.0.1"));
  CHECK_EQ(r.status, 200);
  CHECK(json::parse(r.body).find("setup")->as_bool());
  // Create the admin from the machine itself.
  r = f.Serve(Req("POST", "/router/users", "127.0.0.1",
                  "{\"name\":\"Fabian\",\"rolle\":\"admin\",\"passwort\":\"geheim123\"}"));
  CHECK_EQ(r.status, 201);
  CHECK_EQ(json::parse(r.body).member_str("name"), std::string("fabian"));
  CHECK(f.store->HasAdmin());
  r = f.Serve(Req("POST", "/router/users", "127.0.0.1",
                  "{\"name\":\"fabian\",\"passwort\":\"x\"}"));
  CHECK_EQ(r.status, 409);
  r = f.Serve(Req("POST", "/router/users", "127.0.0.1",
                  "{\"name\":\"tim\",\"passwort\":\"abc\"}"));
  CHECK_EQ(r.status, 400);  // too short
  r = f.Serve(Req("POST", "/router/users", "127.0.0.1",
                  "{\"name\":\"tim\",\"rolle\":\"chef\",\"passwort\":\"abcd\"}"));
  CHECK_EQ(r.status, 400);
  r = f.Serve(Req("POST", "/router/users", "127.0.0.1",
                  "{\"name\":\"tim\",\"passwort\":\"tim-pw\",\"rechte\":{\"bereiche\":"
                  "[\"text\",\"bild\"],\"tageslimit\":5}}"));
  CHECK_EQ(r.status, 201);
  r = f.Serve(Req("POST", "/router/users", "127.0.0.1",
                  "{\"name\":\"film-56\",\"rolle\":\"dienst\"}"));
  CHECK_EQ(r.status, 201);
  // The admin session may manage from the LAN; a user may not (404).
  const std::string admin = f.LoginCookie("fabian", "geheim123");
  const std::string tim = f.LoginCookie("tim", "tim-pw");
  r = f.Serve(Req("GET", "/router/users", "192.168.178.50", "", {{"Cookie", admin}}));
  CHECK_EQ(r.status, 200);
  {
    const json::Value list = json::parse(r.body);
    CHECK_EQ(list.find("users")->size(), std::size_t(3));
    CHECK(r.body.find("hash") == std::string::npos);
    CHECK(r.body.find("salz") == std::string::npos);
  }
  r = f.Serve(Req("GET", "/router/users", "192.168.178.50", "", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 404);
  r = f.Serve(Req("GET", "/verwaltung/", "192.168.178.50", "", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 404);
  r = f.Serve(Req("GET", "/verwaltung/", "192.168.178.50", "", {{"Cookie", admin}}));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("GET", "/router/me", "192.168.178.50", "", {{"Cookie", admin}}));
  CHECK(json::parse(r.body).find("verwaltung")->as_bool());
  // Change role and rights, reset the password, block, token lifecycle.
  r = f.Serve(Req("PUT", "/router/users/tim", "192.168.178.50",
                  "{\"rechte\":{\"bereiche\":[\"text\"],\"filter\":false},"
                  "\"passwort\":\"neu-pw\"}",
                  {{"Cookie", admin}}));
  CHECK_EQ(r.status, 200);
  CHECK(f.store->Verify("tim", "neu-pw").has_value());
  CHECK(f.store->Find("tim")->rechte.bereiche == std::vector<std::string>{"text"});
  r = f.Serve(Req("POST", "/router/users/film-56/tokens", "192.168.178.50",
                  "{\"name\":\"pipeline\",\"tage\":30}", {{"Cookie", admin}}));
  CHECK_EQ(r.status, 201);
  const std::string token = json::parse(r.body).member_str("token");
  CHECK(token.starts_with("gufo_"));
  CHECK(json::parse(r.body).find("ablauf")->as_double() > 0);
  r = f.Serve(Req("GET", "/router/me", "192.168.178.50", "",
                  {{"Authorization", "Bearer " + token}}));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(json::parse(r.body).member_str("name"), std::string("film-56"));
  r = f.Serve(Req("GET", "/router/users/film-56", "127.0.0.1"));
  CHECK_EQ(json::parse(r.body).find("token")->size(), std::size_t(1));
  r = f.Serve(Req("DELETE", "/router/users/film-56/tokens/pipeline", "127.0.0.1"));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("GET", "/router/me", "192.168.178.50", "",
                  {{"Authorization", "Bearer " + token}}));
  CHECK_EQ(r.status, 401);
  r = f.Serve(Req("DELETE", "/router/users/tim", "127.0.0.1"));
  CHECK_EQ(r.status, 200);
  CHECK(f.store->Find("tim")->gesperrt);
  r = f.Serve(Req("GET", "/router/me", "192.168.178.50", "", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 401);  // blocked users lose their session
  r = f.Serve(Req("DELETE", "/router/users/tim?loeschen=1", "127.0.0.1"));
  CHECK_EQ(r.status, 200);
  CHECK(!f.store->Find("tim").has_value());
  r = f.Serve(Req("GET", "/router/users/tim", "127.0.0.1"));
  CHECK_EQ(r.status, 404);
  // The log endpoint.
  r = f.Serve(Req("GET", "/router/zugriff?n=5", "127.0.0.1"));
  CHECK_EQ(r.status, 200);
  CHECK(json::parse(r.body).find("zeilen")->size() <= 5);
  CHECK(json::parse(r.body).find("zeilen")->size() >= 1);
}

void TestModes() {
  // auth = off: everything passes untouched; only the auth endpoints
  // exist, the users API from loopback alone.
  {
    Fixture f(auth::Mode::kOff);
    HttpResponse r = f.Serve(Req("POST", "/v1/images/generations", "192.168.178.50",
                                 "{\"model\":\"m\",\"prompt\":\"nackt\"}"));
    CHECK_EQ(r.status, 200);
    CHECK_EQ(f.last.header("X-Gufo-User"), std::string(""));  // untouched
    r = f.Serve(Req("GET", "/verwaltung/", "192.168.178.50"));
    CHECK_EQ(r.status, 404);
    r = f.Serve(Req("GET", "/router/users", "192.168.178.50"));
    CHECK_EQ(r.status, 404);
    r = f.Serve(Req("GET", "/router/users", "127.0.0.1"));
    CHECK_EQ(r.status, 200);
    r = f.Serve(Req("GET", "/router/me", "192.168.178.50"));
    CHECK_EQ(r.status, 200);
    {
      const json::Value me = json::parse(r.body);
      CHECK_EQ(me.member_str("auth"), std::string("off"));
      CHECK_EQ(me.member_str("rolle"), std::string("admin"));
      CHECK_EQ(me.member_str("quelle"), std::string("off"));
      CHECK(!me.find("verwaltung")->as_bool());
    }
    r = f.Serve(Req("GET", "/bild/", "192.168.178.50"));
    CHECK_EQ(r.status, 200);
    // An SDK's stray default key is ignored with auth = off.
    r = f.Serve(Req("GET", "/router/me", "192.168.178.50", "",
                    {{"Authorization", "Bearer sk-no-key"}}));
    CHECK_EQ(r.status, 200);
    r = f.Serve(Req("POST", "/v1/chat/completions", "192.168.178.50",
                    "{\"model\":\"m\"}", {{"Authorization", "Bearer sk-no-key"}}));
    CHECK_EQ(r.status, 200);
    // A login still works (pages may show the name before stage 1).
    f.AddUser("tim", "nutzer", "tim-pw");
    const std::string tim = f.LoginCookie("tim", "tim-pw");
    r = f.Serve(Req("GET", "/router/me", "192.168.178.50", "", {{"Cookie", tim}}));
    CHECK_EQ(json::parse(r.body).member_str("name"), std::string("tim"));
    r = f.Serve(Req("POST", "/v1/videos", "192.168.178.50", "{\"model\":\"m\"}",
                    {{"Cookie", tim}}));
    CHECK_EQ(r.status, 200);  // no rights checks with auth = off
  }
  // auth = all: the machine itself logs in too.
  {
    Fixture f(auth::Mode::kAll);
    f.AddUser("fabian", "admin", "geheim123");
    HttpResponse r = f.Serve(Req("GET", "/v1/models", "127.0.0.1"));
    CHECK_EQ(r.status, 401);
    r = f.Serve(Req("GET", "/router/users", "127.0.0.1"));
    CHECK_EQ(r.status, 404);
    const std::string admin = f.LoginCookie("fabian", "geheim123", false, "127.0.0.1");
    r = f.Serve(Req("GET", "/router/users", "127.0.0.1", "", {{"Cookie", admin}}));
    CHECK_EQ(r.status, 200);
  }
  // auth = lan with auth-loopback = login: loopback logs in, but may
  // still manage users (the emergency path).
  {
    Fixture f(auth::Mode::kLan, "", auth::LoopbackPolicy::kLogin);
    f.AddUser("fabian", "admin", "geheim123");
    HttpResponse r = f.Serve(Req("GET", "/v1/models", "127.0.0.1"));
    CHECK_EQ(r.status, 401);
    r = f.Serve(Req("GET", "/router/users", "127.0.0.1"));
    CHECK_EQ(r.status, 200);
  }
}

void TestPresetSection() {
  const auto dir = TempDir();
  const auto file = dir / "preset.ini";
  std::vector<gufo::router::PresetModel> models;
  gufo::router::RouterSection section;
  std::string error;
  {
    std::ofstream out(file, std::ios::binary);
    out << "[remote/m]\nurl = http://192.168.1.2:8081\nmodality = llm\n\n"
           "[router]\nauth = lan\nauth-loopback = login\n"
           "users-file = konten\\nutzer.json\npool-secret-env = GUFO_POOL\n";
  }
  CHECK(gufo::router::LoadPreset(file, {}, &models, &error, &section));
  CHECK(error.empty());
  CHECK(section.present);
  CHECK_EQ(section.auth, std::string("lan"));
  CHECK_EQ(section.auth_loopback, std::string("login"));
  CHECK_EQ(section.users_file, std::string("konten\\nutzer.json"));
  CHECK_EQ(section.pool_secret_env, std::string("GUFO_POOL"));
  CHECK_EQ(models.size(), std::size_t(1));
  // Without the section: defaults, and the old call without the argument.
  {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out << "[remote/m]\nurl = http://192.168.1.2:8081\nmodality = llm\n";
  }
  section = {};
  CHECK(gufo::router::LoadPreset(file, {}, &models, &error, &section));
  CHECK(!section.present);
  CHECK_EQ(section.auth, std::string("off"));
  CHECK(gufo::router::LoadPreset(file, {}, &models, &error));
  // Bad values stop the router like any preset error.
  for (const char* body : {"[router]\nauth = ja\n", "[router]\nfoo = 1\n",
                           "[router]\nauth-loopback = x\n",
                           "[router]\nauth = lan\n[router]\nauth = lan\n"}) {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out << "[remote/m]\nurl = http://192.168.1.2:8081\nmodality = llm\n"
        << body;
    out.close();
    CHECK(!gufo::router::LoadPreset(file, {}, &models, &error, &section));
    CHECK(error.find("[router]") != std::string::npos);
  }
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

}  // namespace

int main() {
  TestCrypto();
  TestRights();
  TestStoreRoundTrip();
  TestOldFileFormat();
  TestClassify();
  TestGateOrder();
  TestAreasAndJobs();
  TestPoolSecret();
  TestFailedLogins();
  TestFilterAndLimit();
  TestUsersApi();
  TestModes();
  TestPresetSection();
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "router_auth_test: ok\n";
  return 0;
}
