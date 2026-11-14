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
#include "src/cli/serve/router/comfy_graph.hpp"
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
      const std::string type = req.header("Content-Type");
      if (type.starts_with("multipart/form-data")) {
        std::vector<gufo::router::comfy::MultipartPart> parts;
        std::string error;
        if (gufo::router::comfy::ParseMultipart(type, req.body, &parts, &error))
          for (const auto& part : parts)
            if (part.name == "model")
              return part.data;
        return std::string();
      }
      try {
        const std::string model = json::parse(req.body).member_str("model");
        if (!model.empty())
          return model;
      } catch (const std::exception&) {
      }
      return req.query_param("model");
    };
    // A fake graph plan: the body's "tasks" [{area, model, counts, texts}].
    hooks.comfy_plan = [](const HttpRequest&, const json::Value& body)
        -> std::optional<std::vector<auth::WorkItem>> {
      try {
        const json::Value* tasks = body.find("tasks");
        if (tasks == nullptr || !tasks->is_array())
          return std::nullopt;
        std::vector<auth::WorkItem> items;
        for (const json::Value& task : tasks->items()) {
          auth::WorkItem item;
          item.area = task.member_str("area");
          item.model = task.member_str("model");
          item.counts = static_cast<int>(task.member_double("counts", 1));
          if (const json::Value* texts = task.find("texts");
              texts != nullptr && texts->is_array())
            for (const json::Value& text : texts->items())
              item.texts.push_back(text.str());
          items.push_back(std::move(item));
        }
        return items;
      } catch (const std::exception&) {
        return std::nullopt;
      }
    };
    hooks.comfy_file_prompt =
        [](const std::string& filename, const std::string& subfolder,
           const std::string& type) -> std::optional<std::string> {
      if (!subfolder.empty() || (!type.empty() && type != "output"))
        return std::nullopt;
      if (filename == "tim_00001_.png")
        return std::string("p-tim");
      if (filename == "anna_00001_.png")
        return std::string("p-anna");
      return std::nullopt;
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
           " {\"name\": \"ohne\"}, {\"name\": \"partner\", \"rolle\": \"nutzer\"}]}";
  }
  auth::UserStore store(file, kTestRounds);
  std::string error;
  CHECK(store.Load(&error));
  CHECK_EQ(store.List().size(), std::size_t(3));
  // A name reserved since 2.0.3 ("partner", "lokal") still loads, as a
  // blocked account (no refusal to start).
  CHECK(store.Find("partner").has_value());
  CHECK(store.Find("partner")->gesperrt);
  CHECK(store.SetPassword("partner", "abcd", &error));
  CHECK(!store.Verify("partner", "abcd").has_value());
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
  // When such an account is the only working admin it stays (the file
  // must keep opening the door).
  {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out << "{\"nutzer\": [{\"name\": \"lokal\", \"rolle\": \"admin\", "
           "\"passwort\": {\"salz\": \"00\", \"hash\": \"ff\", \"runden\": 1}},"
           " {\"name\": \"ohne\"}]}";
  }
  CHECK(store.Load(&error));
  CHECK(store.Find("lokal").has_value());
  CHECK(!store.Find("lokal")->gesperrt);
  CHECK(store.HasAdmin());
  // Two reserved admins and no other: the first stays, the second is
  // blocked.
  {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out << "{\"nutzer\": [{\"name\": \"lokal\", \"rolle\": \"admin\", "
           "\"passwort\": {\"salz\": \"00\", \"hash\": \"ff\", \"runden\": 1}},"
           " {\"name\": \"partner\", \"rolle\": \"admin\", "
           "\"passwort\": {\"salz\": \"00\", \"hash\": \"ff\", \"runden\": 1}}]}";
  }
  CHECK(store.Load(&error));
  CHECK(!store.Find("lokal")->gesperrt);
  CHECK(store.Find("partner")->gesperrt);
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

void TestRelocateAndCreate() {
  // 2.1 (review finding 4): Relocate to a file that does not parse leaves
  // the store as it was (users and path), so nothing overwrites the
  // broken file later; a missing file is an empty store at the new path.
  // Finding 7: Create stores name and password in one save, or nothing.
  const auto dir = TempDir();
  auth::UserStore store(dir / "a" / "nutzer.json", kTestRounds);
  std::string error;
  CHECK(store.Load(&error));
  CHECK(store.Create(auth::User{.name = "fabian", .rolle = "admin"},
                     "geheim123", &error));
  CHECK(store.HasAdmin());
  CHECK(store.Verify("fabian", "geheim123").has_value());
  CHECK(!store.Create(auth::User{.name = "fabian", .rolle = "admin"},
                      "anders", &error));
  CHECK(!store.Create(auth::User{.name = "kurz", .rolle = "nutzer"}, "abc",
                      &error));
  CHECK(!store.Find("kurz").has_value());
  CHECK(!store.Create(auth::User{.name = "lokal", .rolle = "nutzer"},
                      "geheim123", &error));
  CHECK_EQ(store.List().size(), std::size_t(1));
  {
    std::filesystem::create_directories(dir / "b");
    std::ofstream out(dir / "b" / "nutzer.json", std::ios::binary);
    out << "{ kaputt";
  }
  CHECK(!store.Relocate(dir / "b" / "nutzer.json", &error));
  CHECK(error.find("does not parse") != std::string::npos);
  CHECK_EQ(store.path().string(), (dir / "a" / "nutzer.json").string());
  CHECK_EQ(store.List().size(), std::size_t(1));
  CHECK(store.HasAdmin());
  // An admin created now goes to the old file, the broken one is intact.
  CHECK(store.Create(auth::User{.name = "zwei", .rolle = "admin"},
                     "geheim123", &error));
  {
    std::ifstream in(dir / "b" / "nutzer.json", std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    CHECK_EQ(text, std::string("{ kaputt"));
  }
  CHECK(store.Relocate(dir / "c" / "nutzer.json", &error));
  CHECK_EQ(store.path().string(), (dir / "c" / "nutzer.json").string());
  CHECK(store.empty());
  CHECK(!store.HasAdmin());
  CHECK(store.Relocate(dir / "a" / "nutzer.json", &error));
  CHECK_EQ(store.List().size(), std::size_t(2));
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
  CHECK(auth::Classify(Req("GET", "/router/partner/status", "x")).kind ==
        Kind::kAnyUser);
  CHECK(auth::Classify(Req("GET", "/router/partner/capabilities", "x")).kind ==
        Kind::kAnyUser);
  CHECK(auth::Classify(Req("GET", "/router/partner/jobs", "x")).kind == Kind::kJobs);
  const auto chat = auth::Classify(Req("POST", "/v1/chat/completions", "x"));
  CHECK(chat.kind == Kind::kWork);
  CHECK_EQ(chat.area, std::string("text"));
  CHECK_EQ(chat.counts, 1);
  CHECK(chat.filtered);  // 2.0.5: chats are filtered on their user texts
  CHECK(!auth::Classify(Req("GET", "/v1/chat/completions", "x")).filtered);
  CHECK(!auth::Classify(Req("POST", "/v1/embeddings", "x")).filtered);
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
  CHECK(lokal.placeholder);
  // A placeholder names no user: a job it submits has no owner.
  const HttpResponse own = f.Serve(Req("POST", "/v1/jobs", "127.0.0.1",
                                       "{\"request\":{}}"));
  CHECK_EQ(own.status, 200);
  CHECK_EQ(f.last.header("X-Gufo-User"), std::string(""));
  CHECK_EQ(f.last.header("X-Gufo-Rights"), std::string(""));
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
    // The stored expiry counts after a restart, not a fresh 30 days.
    std::string error;
    const long long soon = auth::NowUnix() + 50;
    CHECK(f.store->RememberSession("tim", "s-kurz", soon, &error));
    CHECK(fresh.Identify(Req("GET", "/router/me", "192.168.178.50", "",
                             {{"Cookie", "gufo_sitzung=s-kurz"}}))
              .authenticated());
    auth::Gate later(settings, f.store, {});
    later.SetClock([soon] { return soon + 10; });
    CHECK(!later.Identify(Req("GET", "/router/me", "192.168.178.50", "",
                              {{"Cookie", "gufo_sitzung=s-kurz"}}))
               .authenticated());
    // Cached once, it expires at the stored time as well.
    auth::Gate cached(settings, f.store, {});
    long long clock = soon - 10;
    cached.SetClock([&clock] { return clock; });
    CHECK(cached.Identify(Req("GET", "/router/me", "192.168.178.50", "",
                              {{"Cookie", "gufo_sitzung=s-kurz"}}))
              .authenticated());
    clock = soon + 10;
    CHECK(!cached.Identify(Req("GET", "/router/me", "192.168.178.50", "",
                               {{"Cookie", "gufo_sitzung=s-kurz"}}))
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
  // GETs that load the model they name are work of that model's area.
  r = f.Serve(Req("GET", "/sdapi/v1/loras?model=qwen-image-2.1", "192.168.178.50",
                  "", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 403);
  CHECK_EQ(ErrorCode(r), std::string("forbidden"));
  r = f.Serve(Req("GET", "/props?model=qwen3.8-27b", "192.168.178.50", "",
                  {{"Cookie", tim}}));
  CHECK_EQ(r.status, 403);
  CHECK_EQ(ErrorCode(r), std::string("model_forbidden"));
  r = f.Serve(Req("GET", "/props?model=qwen3.5-4b", "192.168.178.50", "",
                  {{"Cookie", tim}}));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("GET", "/v1/slots?model=qwen3.8-27b", "192.168.178.50", "",
                  {{"Cookie", tim}}));
  CHECK_EQ(r.status, 403);
  // With or without a model: work of the text area (the router answers
  // 400 missing_model itself when none is named).
  {
    auth::Rights bild_only;
    bild_only.bereiche = {"bild"};
    f.AddUser("maler", "nutzer", "maler-pw", bild_only);
    const std::string maler = f.LoginCookie("maler", "maler-pw");
    for (const char* path : {"/props", "/v1/slots", "/slots"}) {
      r = f.Serve(Req("GET", path, "192.168.178.50", "", {{"Cookie", maler}}));
      CHECK_EQ(r.status, 403);
      r = f.Serve(Req("GET", path, "192.168.178.50", "", {{"Cookie", tim}}));
      CHECK_EQ(r.status, 200);
    }
    r = f.Serve(Req("GET", "/props?model=qwen3.5-4b", "192.168.178.50", "",
                    {{"Cookie", maler}}));
    CHECK_EQ(r.status, 403);
  }
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
    // The body passes untouched (no re-serialisation); the job API stamps
    // metadata.nutzer from the user header the gate sets.
    const json::Value sent = json::parse(f.last.body);
    CHECK_EQ(sent.find("metadata")->member_str("titel"), std::string("t"));
    CHECK(!sent.find("metadata")->contains("nutzer"));
    CHECK_EQ(f.last.header("X-Gufo-User"), std::string("tim"));
  }
  // A multipart job (body_base64, as the pages send edits and upscales)
  // is checked like a direct one: area, model, filter.
  const auto multipart_job = [](const std::string& model,
                                const std::string& prompt) {
    const std::string raw =
        "--xyz\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n" +
        model + "\r\n--xyz\r\nContent-Disposition: form-data; name=\"prompt\""
        "\r\n\r\n" + prompt + "\r\n--xyz--\r\n";
    json::Value body = json::Value::object();
    json::Value request = json::Value::object();
    request["path"] = "/v1/images/edits";
    request["content_type"] = "multipart/form-data; boundary=xyz";
    request["body_base64"] = gufo::router::comfy::Base64Encode(raw);
    body["request"] = std::move(request);
    return body.dump();
  };
  r = f.Serve(Req("POST", "/v1/jobs", "192.168.178.50",
                  multipart_job("qwen-image-2.1", "Haus"), {{"Cookie", tim}}));
  CHECK_EQ(r.status, 403);  // tim: no bild area
  CHECK_EQ(ErrorCode(r), std::string("forbidden"));
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
  // The partner's list read through the router (2.0.4): filtered the same
  // way; the status read is open to every signed-in user.
  f.next = [](const HttpRequest& req) {
    json::Value body = json::Value::object();
    if (req.path == "/router/partner/jobs") {
      json::Value data = json::Value::array();
      for (const char* owner : {"tim", "anna"}) {
        json::Value job = json::Value::object();
        job["id"] = std::string("p-") + owner;
        json::Value metadata = json::Value::object();
        metadata["nutzer"] = owner;
        job["metadata"] = std::move(metadata);
        data.push_back(std::move(job));
      }
      body["data"] = std::move(data);
    }
    return HttpResponse{.status = 200, .reason = "OK", .body = body.dump()};
  };
  r = f.Serve(Req("GET", "/router/partner/jobs?status=queued", "192.168.178.50", "",
                  {{"Cookie", tim}}));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(json::parse(r.body).find("data")->size(), std::size_t(1));
  r = f.Serve(Req("GET", "/router/partner/jobs", "192.168.178.50", "",
                  {{"Cookie", admin}}));
  CHECK_EQ(json::parse(r.body).find("data")->size(), std::size_t(2));
  r = f.Serve(Req("GET", "/router/partner/status", "192.168.178.50", "",
                  {{"Cookie", tim}}));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("GET", "/router/partner/status", "192.168.178.50"));
  CHECK_EQ(r.status, 401);
  {
    auth::Rights no_ablage;
    no_ablage.bereiche = {"status"};
    f.AddUser("gast", "nutzer", "gast-pw", no_ablage);
    const std::string gast = f.LoginCookie("gast", "gast-pw");
    r = f.Serve(Req("GET", "/router/partner/jobs", "192.168.178.50", "",
                    {{"Cookie", gast}}));
    CHECK_EQ(r.status, 403);
    r = f.Serve(Req("GET", "/router/partner/status", "192.168.178.50", "",
                    {{"Cookie", gast}}));
    CHECK_EQ(r.status, 200);
  }
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
  CHECK_EQ(f.last.header("X-Gufo-User"), std::string("tim"));
  const auto who = f.gate->Identify(Req("GET", "/v1/models", "192.168.100.56", "",
                                        {{"X-Gufo-Pool-Key", "sehr-geheim"}}));
  CHECK(who.source == auth::Identity::Source::kPool);
  CHECK_EQ(who.name, std::string("partner"));
  CHECK_EQ(who.rolle, std::string("partner"));  // the partner router itself
  CHECK(who.trusted());
  CHECK(who.partner_router());
  CHECK(!who.admin());
  // The pool secret opens no admin door: users API 404, model control
  // and bulk delete 403; work and job reads pass.
  r = f.Serve(Req("POST", "/router/users", "192.168.100.56",
                  "{\"name\":\"x\",\"passwort\":\"abcd\"}",
                  {{"X-Gufo-Pool-Key", "sehr-geheim"}}));
  CHECK_EQ(r.status, 404);
  r = f.Serve(Req("GET", "/router/zugriff", "192.168.100.56", "",
                  {{"X-Gufo-Pool-Key", "sehr-geheim"}}));
  CHECK_EQ(r.status, 404);
  r = f.Serve(Req("POST", "/models/unload", "192.168.100.56", "{}",
                  {{"X-Gufo-Pool-Key", "sehr-geheim"}}));
  CHECK_EQ(r.status, 403);
  r = f.Serve(Req("POST", "/router/placement", "192.168.100.56", "{}",
                  {{"X-Gufo-Pool-Key", "sehr-geheim"}}));
  CHECK_EQ(r.status, 403);
  r = f.Serve(Req("DELETE", "/v1/jobs?older_than=1", "192.168.100.56", "",
                  {{"X-Gufo-Pool-Key", "sehr-geheim"}}));
  CHECK_EQ(r.status, 403);
  r = f.Serve(Req("POST", "/v1/images/generations", "192.168.100.56",
                  "{\"model\":\"m\",\"prompt\":\"nackt\"}",
                  {{"X-Gufo-Pool-Key", "sehr-geheim"}}));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(f.last.header("X-Gufo-User"), std::string(""));  // it names no user
  // Job requests the partner relays for a user: that user's jobs only
  // (the entry router cannot know this router's jobs); an admin sees all.
  const std::vector<std::pair<std::string, std::string>> as_tim = {
      {"X-Gufo-Pool-Key", "sehr-geheim"}, {"X-Gufo-User", "tim"},
      {"X-Gufo-Rights", "rolle=nutzer;bereiche=ablage"}};
  const std::vector<std::pair<std::string, std::string>> as_admin = {
      {"X-Gufo-Pool-Key", "sehr-geheim"}, {"X-Gufo-User", "fabian"},
      {"X-Gufo-Rights", "rolle=admin"}};
  r = f.Serve(Req("GET", "/v1/jobs/j-tim", "192.168.100.56", "", as_tim));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("GET", "/v1/jobs/j-anna", "192.168.100.56", "", as_tim));
  CHECK_EQ(r.status, 404);
  r = f.Serve(Req("DELETE", "/v1/jobs/j-anna", "192.168.100.56", "", as_tim));
  CHECK_EQ(r.status, 404);
  r = f.Serve(Req("GET", "/v1/jobs/j-anna/files/a.png", "192.168.100.56", "",
                  as_tim));
  CHECK_EQ(r.status, 404);
  r = f.Serve(Req("DELETE", "/v1/jobs?older_than=1", "192.168.100.56", "", as_tim));
  CHECK_EQ(r.status, 403);
  r = f.Serve(Req("GET", "/v1/jobs/j-anna", "192.168.100.56", "", as_admin));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("DELETE", "/v1/jobs/j-anna", "192.168.100.56", "", as_admin));
  CHECK_EQ(r.status, 200);
  // A relayed list shows the relayed user only its own jobs.
  f.next = [](const HttpRequest&) {
    json::Value body = json::Value::object();
    json::Value data = json::Value::array();
    for (const char* owner : {"tim", "anna"}) {
      json::Value job = json::Value::object();
      job["id"] = std::string("j-") + owner;
      json::Value metadata = json::Value::object();
      metadata["nutzer"] = owner;
      job["metadata"] = std::move(metadata);
      data.push_back(std::move(job));
    }
    body["data"] = std::move(data);
    return HttpResponse{.status = 200, .reason = "OK", .body = body.dump()};
  };
  r = f.Serve(Req("GET", "/v1/jobs?scope=pool", "192.168.100.56", "", as_tim));
  CHECK_EQ(json::parse(r.body).find("data")->size(), std::size_t(1));
  r = f.Serve(Req("GET", "/v1/jobs", "192.168.100.56", "", as_admin));
  CHECK_EQ(json::parse(r.body).find("data")->size(), std::size_t(2));
  // The partner router's own calls (shared queue: the queued list it
  // orders by, a pull, a pulled job's state) see everything.
  r = f.Serve(Req("GET", "/v1/jobs?status=queued&limit=1000", "192.168.100.56",
                  "", {{"X-Gufo-Pool-Key", "sehr-geheim"}, {"X-Gufo-Relay", "1"}}));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(json::parse(r.body).find("data")->size(), std::size_t(2));
  r = f.Serve(Req("GET", "/v1/jobs/j-anna", "192.168.100.56", "",
                  {{"X-Gufo-Pool-Key", "sehr-geheim"}}));
  CHECK_EQ(r.status, 200);
  f.next = [&f](const HttpRequest& req) {
    f.seen.push_back(req.path);
    f.last = req;
    return HttpResponse{.status = 200, .reason = "OK", .body = "{}"};
  };
  f.seen.clear();
  r = f.Serve(Req("POST", "/router/pool/pull", "192.168.100.56",
                  "{\"from\":\"http://192.168.100.56:8081\"}",
                  {{"X-Gufo-Pool-Key", "sehr-geheim"}}));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(f.seen.size(), std::size_t(1));
  CHECK_EQ(f.last.header("X-Gufo-User"), std::string(""));
  // With auth = off a partner relay keeps its user (the job's owner).
  {
    Fixture off(auth::Mode::kOff, "sehr-geheim");
    HttpResponse o = off.Serve(Req("POST", "/v1/jobs", "192.168.100.56",
                                   "{\"request\":{}}",
                                   {{"X-Gufo-Pool-Key", "sehr-geheim"},
                                    {"X-Gufo-User", "tim"},
                                    {"X-Gufo-Rights", "rolle=nutzer"}}));
    CHECK_EQ(o.status, 200);
    CHECK_EQ(off.last.header("X-Gufo-User"), std::string("tim"));
    CHECK_EQ(off.last.header("X-Gufo-Pool-Key"), std::string("sehr-geheim"));
    o = off.Serve(Req("POST", "/v1/jobs", "192.168.178.50", "{\"request\":{}}",
                      {{"X-Gufo-User", "tim"}}));
    CHECK_EQ(off.last.header("X-Gufo-User"), std::string(""));
  }
  // Generation requests of a relayed user are not checked again here.
  r = f.Serve(Req("POST", "/v1/images/generations", "192.168.100.56",
                  "{\"model\":\"m\",\"prompt\":\"nackt\"}", as_tim));
  CHECK_EQ(r.status, 200);
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
  // self_fetch connects to --host, which may be the LAN address: the
  // 256-bit process key counts wherever it arrives from.
  CHECK_EQ(key.size(), std::size_t(64));
  const auto lan = g.gate->Identify(Req("POST", "/v1/images/generations",
                                        "192.168.178.198", "{}",
                                        {{"X-Gufo-Internal", key}}));
  CHECK(lan.source == auth::Identity::Source::kInternal);
  CHECK(!g.gate->Identify(Req("POST", "/v1/images/generations", "192.168.178.198",
                              "{}", {{"X-Gufo-Internal", std::string(64, 'f')}}))
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
  // Videos, the ComfyUI graph and (2.0.5) chats are filtered too.
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
  CHECK_EQ(r.status, 400);
  CHECK_EQ(ErrorCode(r), std::string("content_filtered"));
  // Umlauts: "Töten" and "TÖTEN" meet "töt*" (UTF-8 aware lower case).
  CHECK_EQ(auth::LowerUtf8("TÖTEN Ärger ÜBER Straße ×"),
           std::string("töten ärger über straße ×"));
  for (const char* word : {"Töten", "TÖTEN", "töten"}) {
    r = g.Serve(Req("POST", "/v1/images/generations", "192.168.178.50",
                    std::string("{\"model\":\"m\",\"prompt\":\"") + word +
                        " im Park\"}",
                    {{"Cookie", kind2}}));
    CHECK_EQ(r.status, 400);
  }
  r = g.Serve(Req("POST", "/v1/images/generations", "192.168.178.50",
                  "{\"model\":\"m\",\"prompt\":\"Tölpel\"}", {{"Cookie", kind2}}));
  CHECK_EQ(r.status, 200);
  // ComfyUI graphs: every task of the plan is checked (area, model,
  // filter), the daily limit counts them all (videos twice).
  const auto graph = [](const std::string& tasks) {
    return "{\"prompt\":{},\"tasks\":[" + tasks + "]}";
  };
  r = g.Serve(Req("POST", "/prompt", "192.168.178.50",
                  graph("{\"area\":\"bild\",\"model\":\"m\",\"texts\":[\"ein nacktes Bild\"]}"),
                  {{"Cookie", kind2}}));
  CHECK_EQ(r.status, 400);
  CHECK_EQ(ErrorCode(r), std::string("content_filtered"));
  r = g.Serve(Req("POST", "/prompt", "192.168.178.50",
                  graph("{\"area\":\"bild\",\"model\":\"m\",\"texts\":[\"Haus\"]},"
                        "{\"area\":\"stimme\",\"model\":\"tts\"}"),
                  {{"Cookie", kind2}}));
  CHECK_EQ(r.status, 200);
  // Speech, dialogue and music texts of the plan are filtered too, and
  // every string of the graph as the floor.
  r = g.Serve(Req("POST", "/prompt", "192.168.178.50",
                  graph("{\"area\":\"stimme\",\"model\":\"tts\",\"texts\":[\"sag nackt\"]}"),
                  {{"Cookie", kind2}}));
  CHECK_EQ(r.status, 400);
  CHECK_EQ(ErrorCode(r), std::string("content_filtered"));
  r = g.Serve(Req("POST", "/prompt", "192.168.178.50",
                  "{\"prompt\":{\"7\":{\"inputs\":{\"lyrics\":\"blood\"}}},"
                  "\"tasks\":[{\"area\":\"musik\",\"model\":\"ace\",\"texts\":[\"pop\"]}]}",
                  {{"Cookie", kind2}}));
  CHECK_EQ(r.status, 400);
  {
    Fixture c;
    auth::Rights only_text;
    only_text.bereiche = {"text", "bild"};
    only_text.modelle = {"qwen-image-2.1"};
    only_text.tageslimit = 3;
    c.AddUser("tim", "nutzer", "tim-pw", only_text);
    const std::string cookie = c.LoginCookie("tim", "tim-pw");
    r = c.Serve(Req("POST", "/prompt", "192.168.178.50",
                    graph("{\"area\":\"video\",\"model\":\"minimax-h3\",\"counts\":2}"),
                    {{"Cookie", cookie}}));
    CHECK_EQ(r.status, 403);  // video is closed
    CHECK(r.body.find("Bereich video") != std::string::npos);
    r = c.Serve(Req("POST", "/prompt", "192.168.178.50",
                    graph("{\"area\":\"bild\",\"model\":\"moss-image\"}"),
                    {{"Cookie", cookie}}));
    CHECK_EQ(r.status, 403);
    CHECK_EQ(ErrorCode(r), std::string("model_forbidden"));
    r = c.Serve(Req("POST", "/prompt", "192.168.178.50",
                    graph("{\"area\":\"bild\",\"model\":\"qwen-image-2.1\"},"
                          "{\"area\":\"bild\",\"model\":\"qwen-image-2.1\"}"),
                    {{"Cookie", cookie}}));
    CHECK_EQ(r.status, 200);
    CHECK_EQ(c.gate->DailyCount("tim"), 2LL);
    r = c.Serve(Req("POST", "/prompt", "192.168.178.50",
                    graph("{\"area\":\"bild\",\"model\":\"qwen-image-2.1\",\"counts\":2}"),
                    {{"Cookie", cookie}}));
    CHECK_EQ(r.status, 429);
    // A graph that does not build: the adapter answers, nothing counted.
    r = c.Serve(Req("POST", "/prompt", "192.168.178.50", "{\"prompt\":{}}",
                    {{"Cookie", cookie}}));
    CHECK_EQ(r.status, 200);
    CHECK_EQ(c.gate->DailyCount("tim"), 2LL);
    // Owners beyond the job store: a prompt's history entry and a direct
    // video job are remembered from the answers.
    c.AddUser("anna", "nutzer", "anna-pw");
    const std::string anna = c.LoginCookie("anna", "anna-pw");
    c.next = [](const HttpRequest& req) {
      std::string body = "{}";
      if (req.path == "/prompt")
        body = "{\"prompt_id\":\"p-tim\"}";
      else if (req.path == "/history")
        body = "{\"p-tim\":{\"n\":1},\"p-anna\":{\"n\":2},\"p-alt\":{\"n\":3}}";
      return HttpResponse{.status = 200, .reason = "OK", .body = body};
    };
    r = c.Serve(Req("POST", "/prompt", "192.168.178.50",
                    graph("{\"area\":\"bild\",\"model\":\"qwen-image-2.1\",\"counts\":0}"),
                    {{"Cookie", cookie}}));
    CHECK_EQ(r.status, 200);
    CHECK(c.gate->OwnerOf("prompt:p-tim") == std::optional<std::string>("tim"));
    CHECK(c.gate->RememberOwner("prompt:p-anna", "anna"));
    // The first owner keeps an id: neither a later answer nor a client's
    // chosen prompt_id takes it over, whoever asks (an admin too).
    CHECK(!c.gate->RememberOwner("prompt:p-anna", "tim"));
    CHECK(c.gate->OwnerOf("prompt:p-anna") == std::optional<std::string>("anna"));
    r = c.Serve(Req("POST", "/prompt", "192.168.178.50",
                    "{\"prompt_id\":\"p-anna\",\"prompt\":{},\"tasks\":[]}",
                    {{"Cookie", cookie}}));
    CHECK_EQ(r.status, 409);
    CHECK_EQ(ErrorCode(r), std::string("prompt_id_taken"));
    c.AddUser("chef", "admin", "chef-pw");
    const std::string chef = c.LoginCookie("chef", "chef-pw");
    r = c.Serve(Req("POST", "/prompt", "192.168.178.50",
                    "{\"prompt_id\":\"p-anna\",\"prompt\":{},\"tasks\":[]}",
                    {{"Cookie", chef}}));
    CHECK_EQ(r.status, 409);
    r = c.Serve(Req("POST", "/prompt", "127.0.0.1",
                    "{\"prompt_id\":\"p-anna\",\"prompt\":{},\"tasks\":[]}"));
    CHECK_EQ(r.status, 409);  // loopback as admin too
    r = c.Serve(Req("POST", "/prompt", "192.168.178.50",
                    "{\"prompt_id\":\"p-tim\",\"prompt\":{},\"tasks\":[]}",
                    {{"Cookie", cookie}}));
    CHECK_EQ(r.status, 200);
    CHECK(c.gate->OwnerOf("prompt:p-anna") == std::optional<std::string>("anna"));
    // A refused submit reserves nothing: tim (no video area) names id
    // p-x and is refused, anna may use p-x afterwards.
    r = c.Serve(Req("POST", "/prompt", "192.168.178.50",
                    "{\"prompt_id\":\"p-x\",\"prompt\":{},\"tasks\":"
                    "[{\"area\":\"video\",\"model\":\"minimax-h3\"}]}",
                    {{"Cookie", cookie}}));
    CHECK_EQ(r.status, 403);
    CHECK(!c.gate->OwnerOf("prompt:p-x").has_value());
    r = c.Serve(Req("POST", "/prompt", "192.168.178.50",
                    "{\"prompt_id\":\"p-x\",\"prompt\":{},\"tasks\":[]}",
                    {{"Cookie", anna}}));
    CHECK_EQ(r.status, 200);
    // The adapter answered "p-tim" (c.next above), so p-x went back.
    CHECK(!c.gate->OwnerOf("prompt:p-x").has_value());
    // A new id is taken for the submitter once every check passed (the
    // gate's sequence: ReservePrompt, the adapter, After, release). Two
    // requests of one user share it; it stays theirs while one is in
    // flight, and for good once an answer confirmed it.
    {
      const HttpRequest first_req = Req("POST", "/prompt", "192.168.178.50",
                                        "{\"prompt_id\":\"p-neu\",\"prompt\":{},\"tasks\":[]}",
                                        {{"Cookie", cookie}});
      const json::Value body = json::parse(first_req.body);
      const auto route = auth::Classify(first_req);
      const auto tim_id = c.gate->Identify(first_req);
      const auto anna_id = c.gate->Identify(
          Req("GET", "/router/me", "192.168.178.50", "", {{"Cookie", anna}}));
      std::string key_a;
      std::string key_b;
      std::string key_anna;
      unsigned long long gen_a = 0;
      unsigned long long gen_b = 0;
      unsigned long long gen_anna = 0;
      CHECK(!c.gate->ReservePrompt(tim_id, body, &key_a, &gen_a).has_value());
      CHECK_EQ(key_a, std::string("prompt:p-neu"));
      CHECK(gen_a != 0);
      CHECK(!c.gate->ReservePrompt(tim_id, body, &key_b, &gen_b).has_value());  // same user, twice
      CHECK_EQ(gen_b, gen_a);  // one holding, shared
      const auto taken = c.gate->ReservePrompt(anna_id, body, &key_anna, &gen_anna);
      CHECK(taken.has_value() && taken->status == 409);
      CHECK(key_anna.empty());
      CHECK_EQ(gen_anna, 0ULL);
      // The second request fails at the adapter and goes: the id stays
      // tim's while the first is still in flight.
      HttpResponse failed{.status = 400, .reason = "Bad Request", .body = "{}"};
      c.gate->After(first_req, route, tim_id, &failed);
      c.gate->ReleaseOwner(key_b, tim_id.name, gen_b);
      CHECK(c.gate->OwnerOf("prompt:p-neu") == std::optional<std::string>("tim"));
      CHECK(c.gate->ReservePrompt(anna_id, body, &key_anna, &gen_anna).has_value());
      // The first is confirmed by its answer: the id is tim's for good,
      // its release is a no-op.
      HttpResponse kept{.status = 200, .reason = "OK",
                        .body = "{\"prompt_id\":\"p-neu\"}"};
      c.gate->After(first_req, route, tim_id, &kept);
      c.gate->ReleaseOwner(key_a, tim_id.name, gen_a);
      CHECK(c.gate->OwnerOf("prompt:p-neu") == std::optional<std::string>("tim"));
      CHECK(c.gate->ReservePrompt(anna_id, body, &key_anna, &gen_anna).has_value());
      // Both in flight, both released without a confirming answer: free.
      const json::Value frei = json::parse("{\"prompt_id\":\"p-frei\"}");
      CHECK(!c.gate->ReservePrompt(tim_id, frei, &key_a, &gen_a).has_value());
      CHECK(!c.gate->ReservePrompt(tim_id, frei, &key_b, &gen_b).has_value());
      c.gate->ReleaseOwner(key_a, tim_id.name, gen_a);
      CHECK(c.gate->OwnerOf("prompt:p-frei") == std::optional<std::string>("tim"));
      c.gate->ReleaseOwner(key_b, tim_id.name, gen_b);
      CHECK(!c.gate->OwnerOf("prompt:p-frei").has_value());
      CHECK(!c.gate->ReservePrompt(anna_id, frei, &key_anna, &gen_anna).has_value());
      c.gate->ReleaseOwner(key_anna, anna_id.name, gen_anna);
      // An expired holding taken afresh by another user: the old
      // request's release names the old generation and frees nothing.
      {
        Fixture e;
        long long now = auth::NowUnix();
        e.gate->SetClock([&now] { return now; });
        std::string k_old;
        unsigned long long g_old = 0;
        CHECK(!e.gate->ReservePrompt(tim_id, frei, &k_old, &g_old).has_value());
        now += 86400 + 10;  // the holding expired, the request still runs
        std::string k_new;
        unsigned long long g_new = 0;
        CHECK(!e.gate->ReservePrompt(anna_id, frei, &k_new, &g_new).has_value());
        CHECK(g_new != g_old);
        e.gate->ReleaseOwner(k_old, tim_id.name, g_old);  // stale: no-op
        CHECK(e.gate->OwnerOf("prompt:p-frei") == std::optional<std::string>("anna"));
        // The same user after expiry: the count goes on, both releases needed.
        now += 86400 + 10;
        std::string k_again;
        unsigned long long g_again = 0;
        CHECK(!e.gate->ReservePrompt(anna_id, frei, &k_again, &g_again).has_value());
        CHECK_EQ(g_again, g_new);
        e.gate->ReleaseOwner(k_new, anna_id.name, g_new);
        CHECK(e.gate->OwnerOf("prompt:p-frei") == std::optional<std::string>("anna"));
        e.gate->ReleaseOwner(k_again, anna_id.name, g_again);
        CHECK(!e.gate->OwnerOf("prompt:p-frei").has_value());
      }
    }
    // Over the limit the oldest confirmed entries go, never a reservation
    // in flight.
    {
      Fixture p;
      p.gate->SetOwnerLimit(4);
      long long now = auth::NowUnix();
      p.gate->SetClock([&now] { return now; });
      std::string k1;
      std::string k2;
      unsigned long long g1 = 0;
      unsigned long long g2 = 0;
      auth::Identity tim_like;
      tim_like.source = auth::Identity::Source::kCookie;
      tim_like.name = "tim";
      tim_like.rolle = "nutzer";
      CHECK(!p.gate->ReservePrompt(tim_like, json::parse("{\"prompt_id\":\"r1\"}"), &k1, &g1).has_value());
      now += 1;
      CHECK(p.gate->RememberOwner("prompt:c1", "anna"));
      now += 1;
      CHECK(!p.gate->ReservePrompt(tim_like, json::parse("{\"prompt_id\":\"r2\"}"), &k2, &g2).has_value());
      now += 1;
      CHECK(p.gate->RememberOwner("prompt:c2", "anna"));
      now += 1;
      // The fifth: over the limit, but nothing is an hour old yet, so the
      // table grows past it (fresh owners never go).
      CHECK(p.gate->RememberOwner("prompt:c3", "anna"));
      CHECK(p.gate->OwnerOf("prompt:c1") == std::optional<std::string>("anna"));
      CHECK(p.gate->OwnerOf("prompt:c2") == std::optional<std::string>("anna"));
      // An hour later (and more than a minute after the last try): the
      // confirmed owners older than an hour go, the reservations stay.
      now += 3601;
      CHECK(p.gate->RememberOwner("prompt:c4", "anna"));
      CHECK(p.gate->OwnerOf("prompt:r1") == std::optional<std::string>("tim"));
      CHECK(p.gate->OwnerOf("prompt:r2") == std::optional<std::string>("tim"));
      CHECK(p.gate->OwnerOf("prompt:c4") == std::optional<std::string>("anna"));
      CHECK(!p.gate->OwnerOf("prompt:c1").has_value());
      CHECK(!p.gate->OwnerOf("prompt:c2").has_value());
      CHECK(!p.gate->OwnerOf("prompt:c3").has_value());
      // Within the next minute no further try, even when over the limit.
      now += 30;
      CHECK(p.gate->RememberOwner("prompt:c5", "anna"));
      now += 3700;
      CHECK(p.gate->RememberOwner("prompt:c6", "anna"));  // next try: c4, c5 go
      CHECK(!p.gate->OwnerOf("prompt:c4").has_value());
      CHECK(!p.gate->OwnerOf("prompt:c5").has_value());
      CHECK(p.gate->OwnerOf("prompt:c6") == std::optional<std::string>("anna"));
      CHECK(p.gate->OwnerOf("prompt:r1") == std::optional<std::string>("tim"));
      p.gate->ReleaseOwner(k1, "tim", g1);
      p.gate->ReleaseOwner(k2, "tim", g2);
    }
    // Placeholders (the router itself, loopback) own nothing, by source:
    // their answers are not remembered.
    {
      const HttpRequest own = Req("POST", "/v1/videos", "127.0.0.1",
                                  "{\"model\":\"h3\",\"prompt\":\"See\"}");
      const auto lokal = c.gate->Identify(own);
      CHECK(lokal.placeholder);
      HttpResponse made{.status = 200, .reason = "OK", .body = "{\"id\":\"video_lokal\"}"};
      c.gate->After(own, auth::Classify(own), lokal, &made);
      CHECK(!c.gate->OwnerOf("video:video_lokal").has_value());
    }
    r = c.Serve(Req("GET", "/history", "192.168.178.50", "", {{"Cookie", cookie}}));
    CHECK_EQ(r.status, 200);
    {
      const json::Value history = json::parse(r.body);
      CHECK_EQ(history.size(), std::size_t(1));
      CHECK(history.contains("p-tim"));
    }
    r = c.Serve(Req("POST", "/history", "192.168.178.50",
                    "{\"delete\":[\"p-anna\"]}", {{"Cookie", cookie}}));
    CHECK_EQ(r.status, 404);
    r = c.Serve(Req("POST", "/history", "192.168.178.50",
                    "{\"delete\":[\"p-tim\",\"p-unbekannt\"]}", {{"Cookie", cookie}}));
    CHECK_EQ(r.status, 200);
    // A direct video job: its id streams back from the worker.
    c.next = [](const HttpRequest& req) {
      HttpResponse out{.status = 200, .reason = "OK"};
      if (req.method == "POST" && req.path == "/v1/videos")
        out.streaming_body = [](const HttpResponse::BodyWriter& write) {
          write("{\"id\":\"vi");
          write("deo_7\"}");
        };
      else
        out.body = "{}";
      return out;
    };
    r = c.Serve(Req("POST", "/v1/videos", "192.168.178.50",
                    "{\"model\":\"h3\",\"prompt\":\"See\"}", {{"Cookie", anna}}));
    CHECK_EQ(r.status, 200);
    std::string streamed;
    r.streaming_body([&streamed](std::string_view chunk) {
      streamed.append(chunk);
      return true;
    });
    CHECK_EQ(streamed, std::string("{\"id\":\"video_7\"}"));
    CHECK(c.gate->OwnerOf("video:video_7") == std::optional<std::string>("anna"));
    r = c.Serve(Req("GET", "/v1/videos/video_7", "192.168.178.50", "", {{"Cookie", anna}}));
    CHECK_EQ(r.status, 200);
    r = c.Serve(Req("GET", "/v1/videos/video_7/content", "192.168.178.50", "",
                    {{"Cookie", anna}}));
    CHECK_EQ(r.status, 200);
    // tim has no video area at all; give anna's colleague one to see the
    // owner check alone.
    c.AddUser("ben", "nutzer", "ben-pw");
    const std::string ben = c.LoginCookie("ben", "ben-pw");
    r = c.Serve(Req("GET", "/v1/videos/video_7", "192.168.178.50", "", {{"Cookie", ben}}));
    CHECK_EQ(r.status, 404);
    r = c.Serve(Req("DELETE", "/v1/videos/video_7/content", "192.168.178.50", "",
                    {{"Cookie", ben}}));
    CHECK_EQ(r.status, 404);
    r = c.Serve(Req("GET", "/v1/videos/video_unknown", "192.168.178.50", "",
                    {{"Cookie", ben}}));
    CHECK_EQ(r.status, 200);  // unknown to the gate: open
    r = c.Serve(Req("GET", "/v1/videos/video_lokal/content", "192.168.178.50", "",
                    {{"Cookie", ben}}));
    CHECK_EQ(r.status, 200);  // the router's own: open
    // A video the router's own call makes (a job without an owner) is
    // not remembered either.
    c.next = [](const HttpRequest& req) {
      HttpResponse out{.status = 200, .reason = "OK"};
      out.body = req.path == "/v1/videos" ? "{\"id\":\"video_int\"}" : "{}";
      return out;
    };
    r = c.Serve(Req("POST", "/v1/videos", "127.0.0.1",
                    "{\"model\":\"h3\",\"prompt\":\"See\"}",
                    {{"X-Gufo-Internal", c.gate->internal_key()}}));
    CHECK_EQ(r.status, 200);
    CHECK(!c.gate->OwnerOf("video:video_int").has_value());
    r = c.Serve(Req("GET", "/v1/videos/video_int/content", "192.168.178.50", "",
                    {{"Cookie", ben}}));
    CHECK_EQ(r.status, 200);
    // Concat names job files: only own jobs (the job store's owner).
    r = c.Serve(Req("POST", "/v1/videos/concat", "192.168.178.50",
                    "{\"videos\":[\"/v1/jobs/j-anna/files/video.mp4\",\"/v1/jobs/j-tim/files/video.mp4\"]}",
                    {{"Cookie", anna}}));
    CHECK_EQ(r.status, 404);
    r = c.Serve(Req("POST", "/v1/videos/concat", "192.168.178.50",
                    "{\"videos\":[\"http://x/v1/jobs/j-anna/files/video.mp4\",\"/v1/jobs/j-anna/files/b.mp4\",\"/v1/jobs/j-neu/files/c.mp4\"]}",
                    {{"Cookie", anna}}));
    CHECK_EQ(r.status, 200);  // own and unknown jobs pass
  }
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
  for (const char* reserved : {"lokal", "partner", "Lokal"}) {
    r = f.Serve(Req("POST", "/router/users", "127.0.0.1",
                    std::string("{\"name\":\"") + reserved +
                        "\",\"passwort\":\"abcd\"}"));
    CHECK_EQ(r.status, 400);
    CHECK_EQ(ErrorCode(r), std::string("name_reserved"));
  }
  {
    auth::User reserved;
    reserved.name = "partner";
    std::string error;
    CHECK(!f.store->Put(reserved, &error));
  }
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
  // Writes: JSON only, never cross-site (a page open on the router machine
  // could otherwise reach the loopback admin path with a no-cors POST).
  r = f.Serve(Req("POST", "/router/users", "127.0.0.1",
                  "{\"name\":\"csrf\",\"passwort\":\"abcd\"}",
                  {{"Content-Type", "text/plain"}}));
  CHECK_EQ(r.status, 415);
  CHECK_EQ(ErrorCode(r), std::string("json_required"));
  r = f.Serve(Req("POST", "/router/users", "127.0.0.1",
                  "{\"name\":\"csrf\",\"passwort\":\"abcd\"}",
                  {{"Content-Type", "application/json"},
                   {"Origin", "http://boese.example"}, {"Host", "127.0.0.1:8081"}}));
  CHECK_EQ(r.status, 403);
  CHECK_EQ(ErrorCode(r), std::string("csrf_rejected"));
  r = f.Serve(Req("POST", "/router/users", "127.0.0.1",
                  "{\"name\":\"csrf\",\"passwort\":\"abcd\"}",
                  {{"Content-Type", "application/json"},
                   {"Sec-Fetch-Site", "cross-site"}}));
  CHECK_EQ(r.status, 403);
  r = f.Serve(Req("DELETE", "/router/users/tim", "127.0.0.1", "",
                  {{"Origin", "http://boese.example"}, {"Host", "127.0.0.1:8081"}}));
  CHECK_EQ(r.status, 403);
  CHECK(!f.store->Find("csrf").has_value());
  // The router's own pages: same origin, JSON.
  r = f.Serve(Req("POST", "/router/users", "192.168.178.50",
                  "{\"name\":\"csrf\",\"passwort\":\"abcd\"}",
                  {{"Cookie", admin}, {"Content-Type", "application/json"},
                   {"Origin", "http://192.168.178.198:8081"},
                   {"Host", "192.168.178.198:8081"}, {"Sec-Fetch-Site", "same-origin"}}));
  CHECK_EQ(r.status, 201);
  r = f.Serve(Req("DELETE", "/router/users/csrf?loeschen=1", "127.0.0.1"));
  CHECK_EQ(r.status, 200);
  // The guard covers every change a browser identity makes, not only the
  // users API: model control and generation from a foreign page are
  // refused for a session and for loopback-as-admin; a token is no
  // browser and passes.
  r = f.Serve(Req("POST", "/models/unload", "192.168.178.50", "{}",
                  {{"Cookie", admin}, {"Origin", "http://boese.example"},
                   {"Host", "192.168.178.198:8081"}}));
  CHECK_EQ(r.status, 403);
  CHECK_EQ(ErrorCode(r), std::string("csrf_rejected"));
  r = f.Serve(Req("POST", "/v1/images/generations", "192.168.178.50",
                  "{\"model\":\"m\",\"prompt\":\"Haus\"}",
                  {{"Cookie", admin}, {"Sec-Fetch-Site", "cross-site"}}));
  CHECK_EQ(r.status, 403);
  r = f.Serve(Req("POST", "/models/unload", "127.0.0.1", "{}",
                  {{"Origin", "http://boese.example"}, {"Host", "127.0.0.1:8081"}}));
  CHECK_EQ(r.status, 403);
  r = f.Serve(Req("POST", "/v1/jobs", "127.0.0.1", "{}",
                  {{"Sec-Fetch-Site", "cross-site"}}));
  CHECK_EQ(r.status, 403);
  r = f.Serve(Req("POST", "/router/users/film-56/tokens", "127.0.0.1",
                  "{\"name\":\"csrf\"}"));
  CHECK_EQ(r.status, 201);
  const std::string csrf_token = json::parse(r.body).member_str("token");
  r = f.Serve(Req("POST", "/v1/images/generations", "192.168.178.50",
                  "{\"model\":\"m\",\"prompt\":\"Haus\"}",
                  {{"Authorization", "Bearer " + csrf_token},
                   {"Origin", "http://boese.example"}, {"Host", "192.168.178.198:8081"},
                   {"Sec-Fetch-Site", "cross-site"}}));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("GET", "/v1/models", "192.168.178.50", "",
                  {{"Cookie", admin}, {"Sec-Fetch-Site", "cross-site"}}));
  CHECK_EQ(r.status, 200);  // reads are never refused
  // Exactly this router (host and port): another port of this machine
  // (the pages on 8096, the transcript server) is another site and needs
  // csrf-allow-origins; a browser's missing default port matches.
  r = f.Serve(Req("POST", "/models/unload", "192.168.178.50", "{}",
                  {{"Cookie", admin}, {"Origin", "http://192.168.178.198:8096"},
                   {"Host", "192.168.178.198:8081"}}));
  CHECK_EQ(r.status, 403);
  r = f.Serve(Req("POST", "/models/unload", "192.168.178.50", "{}",
                  {{"Cookie", admin}, {"Origin", "http://192.168.178.199:8081"},
                   {"Host", "192.168.178.198:8081"}}));
  CHECK_EQ(r.status, 403);
  r = f.Serve(Req("POST", "/models/unload", "127.0.0.1", "{}",
                  {{"Origin", "http://[::1]:8096"}, {"Host", "[::1]:8081"}}));
  CHECK_EQ(r.status, 403);
  r = f.Serve(Req("POST", "/models/unload", "127.0.0.1", "{}",
                  {{"Origin", "http://[::1]:8081"}, {"Host", "[::1]:8081"}}));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("POST", "/models/unload", "192.168.178.50", "{}",
                  {{"Cookie", admin}, {"Origin", "http://192.168.178.198:80"},
                   {"Host", "192.168.178.198"}}));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("POST", "/models/unload", "192.168.178.50", "{}",
                  {{"Cookie", admin}, {"Origin", "HTTP://192.168.178.198/"},
                   {"Host", "192.168.178.198:80"}}));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(auth::Gate::NormalizeOrigin("HTTPS://Gufo.Proxy.Local:443/"),
           std::string("https://gufo.proxy.local"));
  CHECK_EQ(auth::Gate::NormalizeOrigin("http://a.b:80"), std::string("http://a.b"));
  CHECK_EQ(auth::Gate::NormalizeOrigin("http://a.b:8080/"),
           std::string("http://a.b:8080"));
  CHECK_EQ(auth::Gate::NormalizeOrigin("https://a.b:80"), std::string("https://a.b:80"));
  CHECK_EQ(auth::Gate::NormalizeOrigin(" 192.168.178.198:8081 "),
           std::string("http://192.168.178.198:8081"));
  CHECK_EQ(auth::Gate::NormalizeOrigin(""), std::string(""));
  // A TLS proxy says so (X-Forwarded-Proto) and sends Host with the
  // https default port; an https Origin without the proxy, or an http
  // page of the same name behind it, is another site.
  r = f.Serve(Req("POST", "/models/unload", "192.168.178.50", "{}",
                  {{"Cookie", admin}, {"Origin", "https://192.168.178.198"},
                   {"Host", "192.168.178.198:443"}, {"X-Forwarded-Proto", "https"}}));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("POST", "/models/unload", "192.168.178.50", "{}",
                  {{"Cookie", admin}, {"Origin", "https://192.168.178.198"},
                   {"Host", "192.168.178.198:443"}}));
  CHECK_EQ(r.status, 403);
  r = f.Serve(Req("POST", "/models/unload", "192.168.178.50", "{}",
                  {{"Cookie", admin}, {"Origin", "http://192.168.178.198"},
                   {"Host", "192.168.178.198"}, {"X-Forwarded-Proto", "https"}}));
  CHECK_EQ(r.status, 403);
  r = f.Serve(Req("POST", "/models/unload", "192.168.178.50", "{}",
                  {{"Cookie", admin}, {"Origin", "https://192.168.178.198"},
                   {"Host", "192.168.178.198:80"}, {"X-Forwarded-Proto", "https"}}));
  CHECK_EQ(r.status, 403);
  // Pages on another port and a proxy named in csrf-allow-origins pass,
  // in any spelling; another proxy does not.
  {
    auth::Settings proxied = f.gate->settings();
    proxied.AllowOrigins({"HTTPS://Gufo.Proxy.Local:443/",
                          "http://192.168.178.198:8096", ""});
    CHECK_EQ(proxied.csrf_allow_origins.size(), std::size_t(2));
    auth::Gate behind(proxied, f.store, {});
    // Sessions live in the gate: sign in at this one.
    const std::string admin_behind = CookieOf(behind.Serve(
        Req("POST", "/login", "192.168.178.50",
            "{\"name\":\"fabian\",\"passwort\":\"geheim123\"}"),
        f.next));
    r = behind.Serve(Req("POST", "/models/unload", "192.168.178.50", "{}",
                         {{"Cookie", admin_behind}, {"Origin", "https://gufo.proxy.local"},
                          {"Host", "192.168.178.198:8081"}}),
                     f.next);
    CHECK_EQ(r.status, 200);
    r = behind.Serve(Req("POST", "/models/unload", "192.168.178.50", "{}",
                         {{"Cookie", admin_behind}, {"Origin", "https://andere.proxy.local"},
                          {"Host", "192.168.178.198:8081"}}),
                     f.next);
    CHECK_EQ(r.status, 403);
    r = behind.Serve(Req("POST", "/models/unload", "192.168.178.50", "{}",
                         {{"Cookie", admin_behind}, {"Origin", "https://gufo.proxy.local:443"},
                          {"Host", "192.168.178.198:8081"}}),
                     f.next);
    CHECK_EQ(r.status, 200);
    r = behind.Serve(Req("POST", "/models/unload", "192.168.178.50", "{}",
                         {{"Cookie", admin_behind}, {"Origin", "http://192.168.178.198:8096"},
                          {"Host", "192.168.178.198:8081"}}),
                     f.next);
    CHECK_EQ(r.status, 200);
  }
  r = f.Serve(Req("DELETE", "/router/users/film-56/tokens/csrf", "127.0.0.1"));
  CHECK_EQ(r.status, 200);
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
  // The session that used the old password is over at once.
  r = f.Serve(Req("GET", "/router/me", "192.168.178.50", "", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 401);
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

void TestPasswordChange() {
  Fixture f;
  f.AddUser("fabian", "admin", "geheim123");
  f.AddUser("tim", "nutzer", "tim-pw");
  f.AddUser("film-56", "dienst", "");
  std::string error;
  const std::string token = f.store->CreateToken("film-56", "p", 0, &error);
  const std::string tim = f.LoginCookie("tim", "tim-pw");
  const std::string tim2 = f.LoginCookie("tim", "tim-pw", true, "192.168.178.51");
  const auto change = [&f](const std::string& cookie, const std::string& alt,
                           const std::string& neu,
                           std::vector<std::pair<std::string, std::string>> more = {}) {
    json::Value body = json::Value::object();
    body["alt"] = alt;
    body["neu"] = neu;
    more.emplace_back("Cookie", cookie);
    return f.Serve(Req("POST", "/router/me/passwort", "192.168.178.50",
                       body.dump(), std::move(more)));
  };
  HttpResponse r = change(tim, "falsch", "neu-pw-1");
  CHECK_EQ(r.status, 403);
  CHECK_EQ(ErrorCode(r), std::string("wrong_password"));
  r = change(tim, "tim-pw", "abc");
  CHECK_EQ(r.status, 400);
  CHECK_EQ(ErrorCode(r), std::string("invalid_password"));
  CHECK(f.store->Verify("tim", "tim-pw").has_value());
  // A browser write: the CSRF rule applies.
  r = change(tim, "tim-pw", "neu-pw-1", {{"Sec-Fetch-Site", "cross-site"}});
  CHECK_EQ(r.status, 403);
  CHECK_EQ(ErrorCode(r), std::string("csrf_rejected"));
  r = change(tim, "tim-pw", "neu-pw-1");
  CHECK_EQ(r.status, 200);
  CHECK(json::parse(r.body).find("ok")->as_bool());
  CHECK(f.store->Verify("tim", "neu-pw-1").has_value());
  CHECK(!f.store->Verify("tim", "tim-pw").has_value());
  // This session stays, the other one is over.
  r = f.Serve(Req("GET", "/router/me", "192.168.178.50", "", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("GET", "/router/me", "192.168.178.51", "", {{"Cookie", tim2}}));
  CHECK_EQ(r.status, 401);
  // A remembered session that changes the password survives a restart.
  const std::string stay = f.LoginCookie("tim", "neu-pw-1", true);
  r = change(stay, "neu-pw-1", "neu-pw-2");
  CHECK_EQ(r.status, 200);
  {
    auth::Gate fresh(f.gate->settings(), f.store, {});
    CHECK(fresh.Identify(Req("GET", "/router/me", "192.168.178.50", "",
                             {{"Cookie", stay}}))
              .authenticated());
  }
  // No account to change: loopback admin, the partner; service accounts
  // have tokens only; nobody without a login.
  r = f.Serve(Req("POST", "/router/me/passwort", "127.0.0.1",
                  "{\"alt\":\"x\",\"neu\":\"yyyy\"}"));
  CHECK_EQ(r.status, 403);
  CHECK_EQ(ErrorCode(r), std::string("no_account"));
  r = f.Serve(Req("POST", "/router/me/passwort", "192.168.178.50",
                  "{\"alt\":\"x\",\"neu\":\"yyyy\"}",
                  {{"Authorization", "Bearer " + token}}));
  CHECK_EQ(r.status, 403);
  CHECK_EQ(ErrorCode(r), std::string("password_change_not_allowed"));
  r = f.Serve(Req("POST", "/router/me/passwort", "192.168.178.50",
                  "{\"alt\":\"x\",\"neu\":\"yyyy\"}"));
  CHECK_EQ(r.status, 401);
  // Five wrong current passwords block the address like failed logins.
  for (int i = 0; i < 5; ++i)
    CHECK_EQ(change(stay, "falsch", "neu-pw-3").status, 403);
  r = change(stay, "neu-pw-2", "neu-pw-3");
  CHECK_EQ(r.status, 429);
  // The log has the change, no passwords.
  bool logged = false;
  for (const std::string& line : f.gate->LogTail(100)) {
    CHECK(line.find("neu-pw") == std::string::npos);
    logged = logged || line.find("passwort_geaendert") != std::string::npos;
  }
  CHECK(logged);
  // An admin changes its own password the same way (the address's block
  // from above is lifted first).
  f.gate->ClearFailures("192.168.178.50");
  const std::string admin = f.LoginCookie("fabian", "geheim123");
  r = change(admin, "geheim123", "geheim456");
  CHECK_EQ(r.status, 200);
  CHECK(f.store->Verify("fabian", "geheim456").has_value());
}

void TestLoopbackBearer() {
  // A bearer without the gufo_ prefix (an SDK's placeholder key) from
  // loopback counts as no header: the machine's own programs stay admin;
  // from the LAN it is still 401; a gufo_ token is always checked.
  Fixture f;
  f.AddUser("fabian", "admin", "geheim123");
  const auto lokal = f.gate->Identify(Req("POST", "/v1/chat/completions", "127.0.0.1",
                                          "{\"model\":\"m\"}",
                                          {{"Authorization", "Bearer sk-no-key"}}));
  CHECK(lokal.source == auth::Identity::Source::kLoopback);
  CHECK(lokal.admin());
  CHECK(lokal.bearer_ignored);
  HttpResponse r = f.Serve(Req("POST", "/v1/chat/completions", "127.0.0.1",
                               "{\"model\":\"m\"}",
                               {{"Authorization", "Bearer LLAMA_API_KEY"}}));
  CHECK_EQ(r.status, 200);
  bool noted = false;
  for (const std::string& line : f.gate->LogTail(20))
    noted = noted || line.find("bearer_ignored=1") != std::string::npos;
  CHECK(noted);
  r = f.Serve(Req("POST", "/v1/chat/completions", "192.168.178.50",
                  "{\"model\":\"m\"}", {{"Authorization", "Bearer sk-no-key"}}));
  CHECK_EQ(r.status, 401);
  CHECK_EQ(ErrorCode(r), std::string("invalid_token"));
  r = f.Serve(Req("POST", "/v1/chat/completions", "127.0.0.1",
                  "{\"model\":\"m\"}",
                  {{"Authorization", "Bearer gufo_" + std::string(64, 'a')}}));
  CHECK_EQ(r.status, 401);
  CHECK_EQ(ErrorCode(r), std::string("invalid_token"));
  // With auth-loopback = login the placeholder key is ignored as well,
  // and the caller is then simply not signed in.
  Fixture g(auth::Mode::kLan, "", auth::LoopbackPolicy::kLogin);
  g.AddUser("fabian", "admin", "geheim123");
  r = g.Serve(Req("POST", "/v1/chat/completions", "127.0.0.1",
                  "{\"model\":\"m\"}", {{"Authorization", "Bearer sk-no-key"}}));
  CHECK_EQ(r.status, 401);
  CHECK_EQ(ErrorCode(r), std::string("not_authenticated"));
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
    // A client's identity headers go even so (they would name a job's
    // owner, and reach a partner with auth = lan beside the pool key).
    r = f.Serve(Req("POST", "/v1/jobs", "192.168.178.50", "{}",
                    {{"X-Gufo-User", "fabian"}, {"X-Gufo-Rights", "rolle=admin"},
                     {"X-Gufo-Internal", "x"}, {"X-Gufo-Pool", "remote"},
                     {"X-Other", "bleibt"}}));
    CHECK_EQ(r.status, 200);
    CHECK_EQ(f.last.header("X-Gufo-User"), std::string(""));
    CHECK_EQ(f.last.header("X-Gufo-Rights"), std::string(""));
    CHECK_EQ(f.last.header("X-Gufo-Internal"), std::string(""));
    CHECK_EQ(f.last.header("X-Gufo-Pool"), std::string("remote"));  // a client's own pin stays
    CHECK_EQ(f.last.header("X-Other"), std::string("bleibt"));
    // Stage 0 too: a browser page on this machine must not reach the
    // admin paths through loopback (the first admin is created here).
    r = f.Serve(Req("POST", "/router/users", "127.0.0.1",
                    "{\"name\":\"csrf\",\"passwort\":\"abcd\"}",
                    {{"Content-Type", "application/json"},
                     {"Sec-Fetch-Site", "cross-site"}}));
    CHECK_EQ(r.status, 403);
    CHECK_EQ(ErrorCode(r), std::string("csrf_rejected"));
    r = f.Serve(Req("POST", "/models/unload", "127.0.0.1", "{}",
                    {{"Origin", "http://boese.example"}, {"Host", "127.0.0.1:8081"}}));
    CHECK_EQ(r.status, 403);
    r = f.Serve(Req("POST", "/router/placement", "127.0.0.1", "{}",
                    {{"Sec-Fetch-Site", "cross-site"}}));
    CHECK_EQ(r.status, 403);
    r = f.Serve(Req("POST", "/router/pool/pull", "127.0.0.1", "{}",
                    {{"Origin", "http://boese.example"}, {"Host", "127.0.0.1:8081"}}));
    CHECK_EQ(r.status, 403);
    CHECK(!f.store->Find("csrf").has_value());
    r = f.Serve(Req("POST", "/models/unload", "127.0.0.1", "{}",
                    {{"Origin", "http://127.0.0.1:8096"}, {"Host", "127.0.0.1:8081"}}));
    CHECK_EQ(r.status, 403);  // another port of this machine: another site
    r = f.Serve(Req("POST", "/models/unload", "127.0.0.1", "{}",
                    {{"Origin", "http://127.0.0.1:8081"}, {"Host", "127.0.0.1:8081"}}));
    CHECK_EQ(r.status, 200);
    r = f.Serve(Req("POST", "/models/unload", "127.0.0.1", "{}"));
    CHECK_EQ(r.status, 200);
    r = f.Serve(Req("GET", "/router/users", "127.0.0.1", "",
                    {{"Sec-Fetch-Site", "cross-site"}}));
    CHECK_EQ(r.status, 200);  // reads stay open
    r = f.Serve(Req("POST", "/v1/images/generations", "192.168.178.50", "{}",
                    {{"Sec-Fetch-Site", "cross-site"}}));
    CHECK_EQ(r.status, 200);  // stage 0: generation is open to everyone anyway
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
  // auth = lan with auth-loopback = login: loopback logs in, the users
  // API and /verwaltung/ included (the emergency path is the preset).
  {
    Fixture f(auth::Mode::kLan, "", auth::LoopbackPolicy::kLogin);
    f.AddUser("fabian", "admin", "geheim123");
    HttpResponse r = f.Serve(Req("GET", "/v1/models", "127.0.0.1"));
    CHECK_EQ(r.status, 401);
    r = f.Serve(Req("GET", "/router/users", "127.0.0.1"));
    CHECK_EQ(r.status, 404);
    r = f.Serve(Req("GET", "/router/zugriff", "127.0.0.1"));
    CHECK_EQ(r.status, 404);
    r = f.Serve(Req("GET", "/verwaltung/", "127.0.0.1"));
    CHECK_EQ(r.status, 404);
    const std::string admin = f.LoginCookie("fabian", "geheim123", false, "127.0.0.1");
    r = f.Serve(Req("GET", "/router/users", "127.0.0.1", "", {{"Cookie", admin}}));
    CHECK_EQ(r.status, 200);
    r = f.Serve(Req("GET", "/verwaltung/", "127.0.0.1", "", {{"Cookie", admin}}));
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
           "users-file = konten\\nutzer.json\npool-secret-env = GUFO_POOL\n"
           "csrf-allow-origins = https://gufo.proxy.local, http://10.0.0.5:80\n";
  }
  CHECK(gufo::router::LoadPreset(file, {}, &models, &error, &section));
  CHECK(error.empty());
  CHECK(section.present);
  CHECK_EQ(section.auth, std::string("lan"));
  CHECK_EQ(section.auth_loopback, std::string("login"));
  CHECK_EQ(section.users_file, std::string("konten\\nutzer.json"));
  CHECK_EQ(section.pool_secret_env, std::string("GUFO_POOL"));
  CHECK_EQ(section.csrf_allow_origins,
           std::string("https://gufo.proxy.local, http://10.0.0.5:80"));
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
                           "[router]\ncsrf-allow-origins =\n",
                           "[router]\nauth-loopback = x\n",
                           "[router]\nauth = lan\n[router]\nauth = lan\n",
                           "[router/x]\nauth = lan\n",
                           "[router/x]\nauth = lan\n[router]\nauth = lan\n"}) {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out << "[remote/m]\nurl = http://192.168.1.2:8081\nmodality = llm\n"
        << body;
    out.close();
    section = {};
    CHECK(!gufo::router::LoadPreset(file, {}, &models, &error, &section));
    CHECK(error.find("[router]") != std::string::npos);
  }
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

void TestModelArgsHidden() {
  // 2.0.5: the worker command lines (paths on this machine) are for the
  // admin and the machine itself; a user or service account sees the
  // state without them. Other fields stay as they came.
  Fixture f;
  f.AddUser("fabian", "admin", "geheim123");
  f.AddUser("tim", "nutzer", "tim-pw");
  f.AddUser("film-56", "dienst", "");
  std::string error;
  const std::string token = f.store->CreateToken("film-56", "p", 0, &error);
  f.next = [](const HttpRequest&) {
    json::Value model = json::Value::object();
    model["id"] = "qwen3.8-27b";
    model["modality"] = "llm";
    model["loaded"] = true;
    json::Value args = json::Value::array();
    args.push_back("--model");
    args.push_back("D:\\modelle\\x.gguf");
    json::Value status = json::Value::object();
    status["value"] = "loaded";
    status["args"] = std::move(args);
    status["failed"] = false;
    model["status"] = std::move(status);
    json::Value data = json::Value::array();
    data.push_back(std::move(model));
    json::Value body = json::Value::object();
    body["object"] = "list";
    body["data"] = std::move(data);
    return HttpResponse{.status = 200, .reason = "OK", .body = body.dump()};
  };
  const auto status_of = [](const HttpResponse& r) {
    return *json::parse(r.body).find("data")->items().front().find("status");
  };
  const std::string tim = f.LoginCookie("tim", "tim-pw");
  HttpResponse r = f.Serve(Req("GET", "/v1/models", "192.168.178.50", "",
                               {{"Cookie", tim}}));
  CHECK_EQ(r.status, 200);
  CHECK(r.body.find("x.gguf") == std::string::npos);
  CHECK(status_of(r).find("args") == nullptr);
  CHECK_EQ(status_of(r).member_str("value"), std::string("loaded"));
  CHECK(json::parse(r.body).find("data")->items().front().find("loaded")->as_bool());
  CHECK_EQ(json::parse(r.body).member_str("object"), std::string("list"));
  r = f.Serve(Req("GET", "/models", "192.168.178.50", "",
                  {{"Authorization", "Bearer " + token}}));
  CHECK_EQ(r.status, 200);
  CHECK(status_of(r).find("args") == nullptr);
  const std::string admin = f.LoginCookie("fabian", "geheim123");
  r = f.Serve(Req("GET", "/v1/models", "192.168.178.50", "",
                  {{"Cookie", admin}}));
  CHECK(status_of(r).find("args") != nullptr);
  r = f.Serve(Req("GET", "/v1/models", "127.0.0.1"));
  CHECK(status_of(r).find("args") != nullptr);
}

void TestStatusJobsHidden() {
  // 2.0.5: /router/status names job ids of the shared queue and the
  // remote jobs; a user sees its own, an admin all, the rest of the
  // status stays.
  Fixture f;
  f.AddUser("fabian", "admin", "geheim123");
  f.AddUser("tim", "nutzer", "tim-pw");
  f.next = [](const HttpRequest&) {
    json::Value ids = json::Value::array();
    ids.push_back("j-tim");
    ids.push_back("j-anna");
    ids.push_back("j-unbekannt");
    json::Value queue = json::Value::object();
    queue["local"] = 3;
    queue["pulling"] = ids;
    queue["pushed_active"] = ids;
    queue["enabled"] = true;
    json::Value jobs = json::Value::object();
    jobs["j-tim"] = json::Value::object();
    jobs["j-anna"] = json::Value::object();
    json::Value remote = json::Value::object();
    remote["router_id"] = "r1";
    remote["jobs"] = std::move(jobs);
    json::Value body = json::Value::object();
    body["models_max"] = 2;
    body["pool_queue"] = std::move(queue);
    body["remote"] = std::move(remote);
    return HttpResponse{.status = 200, .reason = "OK", .body = body.dump()};
  };
  const std::string tim = f.LoginCookie("tim", "tim-pw");
  HttpResponse r = f.Serve(Req("GET", "/router/status", "192.168.178.50", "",
                               {{"Cookie", tim}}));
  CHECK_EQ(r.status, 200);
  json::Value body = json::parse(r.body);
  const json::Value* queue = body.find("pool_queue");
  CHECK(queue != nullptr);
  CHECK_EQ(queue->find("pulling")->items().size(), std::size_t(1));
  CHECK_EQ(queue->find("pulling")->items().front().str(), std::string("j-tim"));
  CHECK_EQ(queue->find("pushed_active")->items().size(), std::size_t(1));
  CHECK_EQ(queue->member_size("local"), std::size_t(3));
  CHECK(queue->find("enabled")->as_bool());
  const json::Value* remote = body.find("remote");
  CHECK(remote != nullptr);
  CHECK_EQ(remote->find("jobs")->members().size(), std::size_t(1));
  CHECK(remote->find("jobs")->find("j-tim") != nullptr);
  CHECK_EQ(remote->member_str("router_id"), std::string("r1"));
  CHECK_EQ(body.member_size("models_max"), std::size_t(2));
  const std::string admin = f.LoginCookie("fabian", "geheim123");
  r = f.Serve(Req("GET", "/router/status", "192.168.178.50", "",
                  {{"Cookie", admin}}));
  body = json::parse(r.body);
  CHECK_EQ(body.find("pool_queue")->find("pulling")->items().size(),
           std::size_t(3));
  CHECK_EQ(body.find("remote")->find("jobs")->members().size(),
           std::size_t(2));
  r = f.Serve(Req("GET", "/router/status", "127.0.0.1"));
  CHECK_EQ(json::parse(r.body).find("pool_queue")->find("pulling")->items().size(),
           std::size_t(3));
  // 2.1 (review finding 3): the partner's status, relayed through
  // /router/partner/status, is filtered the same way.
  r = f.Serve(Req("GET", "/router/partner/status", "192.168.178.50", "",
                  {{"Cookie", tim}}));
  CHECK_EQ(r.status, 200);
  body = json::parse(r.body);
  CHECK_EQ(body.find("pool_queue")->find("pulling")->items().size(),
           std::size_t(1));
  CHECK_EQ(body.find("pool_queue")->find("pushed_active")->items().size(),
           std::size_t(1));
  CHECK_EQ(body.find("remote")->find("jobs")->members().size(),
           std::size_t(1));
  CHECK(body.find("remote")->find("jobs")->find("j-tim") != nullptr);
  CHECK_EQ(body.member_size("models_max"), std::size_t(2));
  r = f.Serve(Req("GET", "/router/partner/status", "192.168.178.50", "",
                  {{"Cookie", admin}}));
  body = json::parse(r.body);
  CHECK_EQ(body.find("pool_queue")->find("pulling")->items().size(),
           std::size_t(3));
}

void TestWebhookRoles() {
  // 2.0.5: a webhook_url in a job is for admin and dienst; a nutzer gets
  // 403 webhook_not_allowed, its job without a webhook goes through.
  Fixture f;
  f.AddUser("fabian", "admin", "geheim123");
  f.AddUser("tim", "nutzer", "tim-pw");
  f.AddUser("film-56", "dienst", "");
  std::string error;
  const std::string token = f.store->CreateToken("film-56", "p", 0, &error);
  const std::string with_hook =
      "{\"request\":{\"path\":\"/v1/images/generations\",\"body\":"
      "{\"model\":\"qwen-image-2.1\",\"prompt\":\"Haus\"}},"
      "\"webhook_url\":\"http://192.168.178.56:8096/fertig\"}";
  const std::string without =
      "{\"request\":{\"path\":\"/v1/images/generations\",\"body\":"
      "{\"model\":\"qwen-image-2.1\",\"prompt\":\"Haus\"}}}";
  const std::string tim = f.LoginCookie("tim", "tim-pw");
  HttpResponse r = f.Serve(Req("POST", "/v1/jobs", "192.168.178.50", with_hook,
                               {{"Cookie", tim}}));
  CHECK_EQ(r.status, 403);
  CHECK_EQ(ErrorCode(r), std::string("webhook_not_allowed"));
  r = f.Serve(Req("POST", "/v1/jobs", "192.168.178.50",
                  "{\"request\":{\"path\":\"/v1/images/generations\"},"
                  "\"webhook_url\":null}",
                  {{"Cookie", tim}}));
  CHECK_EQ(r.status, 200);  // null = none
  r = f.Serve(Req("POST", "/v1/jobs", "192.168.178.50", without,
                  {{"Cookie", tim}}));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("POST", "/v1/jobs", "192.168.178.50", with_hook,
                  {{"Authorization", "Bearer " + token}}));
  CHECK_EQ(r.status, 200);
  const std::string admin = f.LoginCookie("fabian", "geheim123");
  r = f.Serve(Req("POST", "/v1/jobs", "192.168.178.50", with_hook,
                  {{"Cookie", admin}}));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("POST", "/v1/jobs", "127.0.0.1", with_hook));
  CHECK_EQ(r.status, 200);
  bool logged = false;
  for (const std::string& line : f.gate->LogTail(20))
    logged = logged || line.find("abgelehnt:webhook_not_allowed") !=
                           std::string::npos;
  CHECK(logged);
}

void TestChatFilter() {
  // 2.0.5: the word filter reads a chat's user texts (every user turn,
  // text parts, `prompt`, `input`); the answer, code and log line are
  // those of an image; admin and the machine itself are not filtered;
  // a streaming request is refused before any stream; assistant texts
  // are not read, system/developer texts are (2.1).
  Fixture f;
  f.AddUser("fabian", "admin", "geheim123");
  f.AddUser("kind", "nutzer", "kind-pw");
  auth::Rights free_rights;
  free_rights.filter = false;
  f.AddUser("tim", "nutzer", "tim-pw", free_rights);
  const std::string kind = f.LoginCookie("kind", "kind-pw");
  const std::string tim = f.LoginCookie("tim", "tim-pw");
  const std::string admin = f.LoginCookie("fabian", "geheim123");
  const auto chat = [&f](const std::string& path, const std::string& cookie,
                         const std::string& body) {
    return f.Serve(Req("POST", path, "192.168.178.50", body,
                       {{"Cookie", cookie}}));
  };
  HttpResponse r = chat("/v1/chat/completions", kind,
                        "{\"model\":\"m\",\"messages\":[{\"role\":\"user\","
                        "\"content\":\"Ein Haus am See\"}]}");
  CHECK_EQ(r.status, 200);
  r = chat("/v1/chat/completions", kind,
           "{\"model\":\"m\",\"stream\":true,\"messages\":[{\"role\":\"user\","
           "\"content\":\"Eine NACKTE Frau\"}]}");
  CHECK_EQ(r.status, 400);
  CHECK_EQ(ErrorCode(r), std::string("content_filtered"));
  CHECK(!r.streaming_body);
  // Only the last user turn is read (a refused question the page keeps
  // in the history must not block the conversation); text parts count,
  // system/assistant not.
  r = chat("/v1/chat/completions", kind,
           "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":"
           "\"nackt\"},{\"role\":\"assistant\",\"content\":\"ok\"},"
           "{\"role\":\"user\",\"content\":\"weiter\"}]}");
  CHECK_EQ(r.status, 200);
  r = chat("/v1/chat/completions", kind,
           "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":"
           "\"Haus\"},{\"role\":\"assistant\",\"content\":\"ok\"},"
           "{\"role\":\"user\",\"content\":\"nackt\"}]}");
  CHECK_EQ(r.status, 400);
  r = chat("/v1/responses", kind,
           "{\"model\":\"m\",\"input\":[{\"role\":\"user\",\"content\":"
           "\"nackt\"},{\"role\":\"user\",\"content\":\"Haus\"}]}");
  CHECK_EQ(r.status, 200);
  r = chat("/v1/chat/completions", kind,
           "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":"
           "[{\"type\":\"text\",\"text\":\"ein nsfw bild\"}]}]}");
  CHECK_EQ(r.status, 400);
  // 2.1 (review finding 6): system and developer messages, Anthropic
  // `system` and responses `instructions` are read too; assistant turns
  // are not.
  r = chat("/v1/chat/completions", kind,
           "{\"model\":\"m\",\"messages\":[{\"role\":\"assistant\",\"content\":"
           "\"nackt\"},{\"role\":\"user\",\"content\":\"Hallo\"}]}");
  CHECK_EQ(r.status, 200);
  r = chat("/v1/chat/completions", kind,
           "{\"model\":\"m\",\"messages\":[{\"role\":\"system\",\"content\":"
           "\"nackt\"},{\"role\":\"user\",\"content\":\"Hallo\"}]}");
  CHECK_EQ(r.status, 400);
  CHECK_EQ(ErrorCode(r), std::string("content_filtered"));
  r = chat("/v1/chat/completions", kind,
           "{\"model\":\"m\",\"messages\":[{\"role\":\"developer\",\"content\":"
           "[{\"type\":\"text\",\"text\":\"porno\"}]},"
           "{\"role\":\"user\",\"content\":\"Hallo\"}]}");
  CHECK_EQ(r.status, 400);
  r = chat("/v1/responses", kind,
           "{\"model\":\"m\",\"instructions\":\"nackt\",\"input\":\"Haus\"}");
  CHECK_EQ(r.status, 400);
  r = chat("/v1/responses", kind,
           "{\"model\":\"m\",\"input\":[{\"role\":\"system\",\"content\":"
           "\"nackt\"},{\"role\":\"user\",\"content\":\"Haus\"}]}");
  CHECK_EQ(r.status, 400);
  r = chat("/v1/messages", kind,
           "{\"model\":\"m\",\"system\":[{\"type\":\"text\",\"text\":\"nackt\"}],"
           "\"messages\":[{\"role\":\"user\",\"content\":\"Haus\"}]}");
  CHECK_EQ(r.status, 400);
  r = chat("/v1/messages", kind,
           "{\"model\":\"m\",\"system\":\"Sei knapp\","
           "\"messages\":[{\"role\":\"user\",\"content\":\"Haus\"}]}");
  CHECK_EQ(r.status, 200);
  r = chat("/v1/completions", kind, "{\"model\":\"m\",\"prompt\":\"porno\"}");
  CHECK_EQ(r.status, 400);
  r = chat("/v1/completions", kind,
           "{\"model\":\"m\",\"prompt\":[\"Haus\",\"porno\"]}");
  CHECK_EQ(r.status, 400);
  r = chat("/v1/responses", kind, "{\"model\":\"m\",\"input\":\"porno\"}");
  CHECK_EQ(r.status, 400);
  r = chat("/v1/responses", kind,
           "{\"model\":\"m\",\"input\":[{\"role\":\"user\",\"content\":"
           "[{\"type\":\"input_text\",\"text\":\"porno\"}]}]}");
  CHECK_EQ(r.status, 400);
  r = chat("/v1/responses", kind, "{\"model\":\"m\",\"input\":\"Haus\"}");
  CHECK_EQ(r.status, 200);
  r = chat("/v1/messages", kind,
           "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":"
           "\"porno\"}]}");
  CHECK_EQ(r.status, 400);
  // Filter off for tim, for admins, for the machine itself.
  const std::string bad =
      "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"nackt\"}]}";
  CHECK_EQ(chat("/v1/chat/completions", tim, bad).status, 200);
  CHECK_EQ(chat("/v1/chat/completions", admin, bad).status, 200);
  r = f.Serve(Req("POST", "/v1/chat/completions", "127.0.0.1", bad));
  CHECK_EQ(r.status, 200);
  // Not JSON: the worker answers; no crash here.
  r = chat("/v1/chat/completions", kind, "nackt");
  CHECK_EQ(r.status, 200);
  bool logged = false;
  for (const std::string& line : f.gate->LogTail(40))
    logged = logged ||
             line.find("abgelehnt:content_filtered") != std::string::npos;
  CHECK(logged);
}

void TestViewOwner() {
  // 2.0.5: a ComfyUI result file answers 404 to another user; the owner
  // and an admin read it; a file no prompt names stays open.
  Fixture f;
  f.AddUser("fabian", "admin", "geheim123");
  f.AddUser("tim", "nutzer", "tim-pw");
  CHECK(f.gate->RememberOwner("prompt:p-tim", "tim"));
  CHECK(f.gate->RememberOwner("prompt:p-anna", "anna"));
  const std::string tim = f.LoginCookie("tim", "tim-pw");
  HttpResponse r = f.Serve(Req("GET", "/view?filename=tim_00001_.png",
                               "192.168.178.50", "", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("GET", "/view?filename=anna_00001_.png&type=output",
                  "192.168.178.50", "", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 404);
  r = f.Serve(Req("GET", "/view?filename=hochgeladen.png&type=input",
                  "192.168.178.50", "", {{"Cookie", tim}}));
  CHECK_EQ(r.status, 200);
  const std::string admin = f.LoginCookie("fabian", "geheim123");
  r = f.Serve(Req("GET", "/view?filename=anna_00001_.png", "192.168.178.50",
                  "", {{"Cookie", admin}}));
  CHECK_EQ(r.status, 200);
  r = f.Serve(Req("GET", "/view?filename=anna_00001_.png", "127.0.0.1"));
  CHECK_EQ(r.status, 200);
}

}  // namespace

int main() {
  TestCrypto();
  TestRights();
  TestStoreRoundTrip();
  TestOldFileFormat();
  TestRelocateAndCreate();
  TestClassify();
  TestGateOrder();
  TestAreasAndJobs();
  TestPoolSecret();
  TestFailedLogins();
  TestFilterAndLimit();
  TestUsersApi();
  TestPasswordChange();
  TestLoopbackBearer();
  TestModes();
  TestPresetSection();
  TestModelArgsHidden();
  TestStatusJobsHidden();
  TestWebhookRoles();
  TestChatFilter();
  TestViewOwner();
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "router_auth_test: ok\n";
  return 0;
}
