// MCP endpoint (src/cli/serve/router/mcp.*): the JSON-RPC 2.0 frame, the
// initialize handshake, tools/list per role, the tools/call skeleton and
// the gate's rules for /mcp (token of any role, loopback, never a cookie,
// the partner 404, Origin check, [router] mcp = off). Pure state, no
// sockets, no models: a real gate in front of a fake dispatch.

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/auth.hpp"
#include "src/cli/serve/router/mcp.hpp"
#include "src/cli/serve/router/preset.hpp"
#include "src/core/json.hpp"

namespace {

namespace auth = gufo::router::auth;
namespace mcp = gufo::router::mcp;
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
constexpr const char* kLan = "192.168.178.50";
constexpr const char* kLoopback = "127.0.0.1";

std::filesystem::path TempDir() {
  const auto base = std::filesystem::temp_directory_path() /
                    ("gufo-mcp-test-" + auth::RandomHex(4));
  std::filesystem::create_directories(base);
  return base;
}

HttpRequest Req(std::string method, std::string path, std::string client,
                std::string body = {},
                std::vector<std::pair<std::string, std::string>> headers = {}) {
  HttpRequest req;
  req.method = std::move(method);
  req.path = std::move(path);
  req.client_id = std::move(client);
  req.body = std::move(body);
  req.headers = std::move(headers);
  if (!req.body.empty() && req.header("Content-Type").empty())
    req.headers.emplace_back("Content-Type", "application/json");
  return req;
}

json::Value Body(const HttpResponse& r) {
  try {
    return json::parse(r.body);
  } catch (const std::exception&) {
    return json::Value();
  }
}

/// The router's error shape (`error.code` string) of a gate refusal.
std::string ErrorCode(const HttpResponse& r) {
  const json::Value body = Body(r);  // kept alive for the pointer below
  if (const json::Value* e = body.find("error"))
    return e->member_str("code");
  return "";
}

/// The JSON-RPC error code (0 when the answer carries none).
int RpcCode(const HttpResponse& r) {
  const json::Value body = Body(r);
  if (const json::Value* e = body.find("error"))
    return static_cast<int>(e->member_double("code"));
  return 0;
}

std::string RpcMessage(const HttpResponse& r) {
  const json::Value body = Body(r);
  if (const json::Value* e = body.find("error"))
    return e->member_str("message");
  return "";
}

const json::Value* Result(const json::Value& body) {
  const json::Value* result = body.find("result");
  return result != nullptr && result->is_object() ? result : nullptr;
}

std::string Header(const HttpResponse& r, std::string_view name) {
  for (const auto& [k, v] : r.headers)
    if (k == name)
      return v;
  return "";
}

std::string CookieOf(const HttpResponse& r) {
  const std::string set = Header(r, "Set-Cookie");
  return set.substr(0, set.find(';'));
}

std::vector<std::string> ToolNames(const HttpResponse& r) {
  std::vector<std::string> names;
  const json::Value body = Body(r);
  if (const json::Value* result = Result(body))
    if (const json::Value* tools = result->find("tools"))
      for (const json::Value& tool : tools->items())
        names.push_back(tool.member_str("name"));
  return names;
}

bool Has(const std::vector<std::string>& names, std::string_view name) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

/// {"jsonrpc":"2.0","id":<id>,"method":"<method>","params":<params>}
std::string Msg(const std::string& method, const std::string& id = "1",
                const std::string& params = "{}") {
  return "{\"jsonrpc\":\"2.0\",\"id\":" + id + ",\"method\":\"" + method +
         "\",\"params\":" + params + "}";
}

const std::vector<std::string> kAllTools = {
    "modelle",       "chat",           "bild_erzeugen",   "bild_bearbeiten",
    "video_erzeugen", "stimme",        "musik",           "objekt_3d",
    "hochskalieren", "auftrag_status", "auftrag_liste",   "auftrag_abbrechen",
    "ablage_lesen"};

/// Outside Fixture: clang needs the default member initializers of a
/// default argument's type complete before the enclosing class is.
struct FixtureConfig {
  auth::Mode mode{auth::Mode::kLan};
  std::string pool_secret;
  bool enabled{true};
  bool jobs_api{true};
  std::string version;
};

struct Fixture {
  using Config = FixtureConfig;

  std::filesystem::path dir = TempDir();
  std::shared_ptr<auth::UserStore> store =
      std::make_shared<auth::UserStore>(dir / "nutzer" / "nutzer.json",
                                        kTestRounds);
  std::unique_ptr<auth::Gate> gate;
  std::unique_ptr<mcp::McpService> service;
  std::vector<std::string> seen;  // paths the fake router received
  std::function<HttpResponse(const HttpRequest&)> dispatch;

  explicit Fixture(Config config = {}) {
    auth::Settings settings;
    settings.mode = config.mode;
    settings.users_file = store->path();
    settings.pool_secret = config.pool_secret;
    auth::Gate::Hooks hooks;
    hooks.model_of = [](const HttpRequest& req) {
      try {
        return json::parse(req.body).member_str("model");
      } catch (const std::exception&) {
        return std::string();
      }
    };
    gate = std::make_unique<auth::Gate>(settings, store, std::move(hooks));
    dispatch = [this](const HttpRequest& req) {
      if (auto answer = service->Handle(req))
        return std::move(*answer);
      seen.push_back(req.path);
      json::Value body = json::Value::object();
      body["ok"] = true;
      return HttpResponse{.status = 200, .reason = "OK", .body = body.dump()};
    };
    mcp::Options options;
    options.gate = gate.get();
    options.enabled = config.enabled;
    options.jobs_api = config.jobs_api;
    options.version = config.version;
    options.self_name = "http://192.168.178.198:8081";
    options.serve = [this](const HttpRequest& inner) {
      return gate->Serve(inner, dispatch);
    };
    const auto model = [&options](std::string modality, std::string id,
                                  std::vector<std::string> argv = {}) {
      gufo::router::PresetModel m;
      m.modality = std::move(modality);
      m.id = std::move(id);
      m.argv = std::move(argv);
      options.models.push_back(std::move(m));
    };
    model("llm", "qwen3.8-27b");
    model("llm", "qwen3.5-4b");
    model("image", "qwen-image-2.1-turbo");
    model("image", "qwen-image-2.1");
    model("video", "minimax-h3");
    // An [external/...] section: the modality comes from `args = serve tts`.
    model("external", "fish-drama-3", {"serve", "tts", "--x"});
    model("external", "moss-music", {"serve", "moss-music"});
    model("external", "ace-step-1.5-xl-turbo", {"serve", "music"});
    model("external", "triposg", {"serve", "objekt"});
    model("external", "gufo-upscale", {"serve", "upscale"});
    service = std::make_unique<mcp::McpService>(std::move(options));
  }
  ~Fixture() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }

  void AddUser(const std::string& name, const std::string& role,
               auth::Rights rights = {}) {
    auth::User user;
    user.name = name;
    user.rolle = role;
    user.rechte = std::move(rights);
    std::string error;
    CHECK(store->Put(user, &error));
  }

  std::string Token(const std::string& name) {
    std::string error;
    const std::string token = store->CreateToken(name, "t", 0, &error);
    CHECK(!token.empty());
    return token;
  }

  std::string LoginCookie(const std::string& name, const std::string& pw) {
    json::Value body = json::Value::object();
    body["name"] = name;
    body["passwort"] = pw;
    const HttpResponse r = Serve(Req("POST", "/login", kLan, body.dump()));
    CHECK_EQ(r.status, 200);
    return CookieOf(r);
  }

  HttpResponse Serve(const HttpRequest& req) {
    return gate->Serve(req, dispatch);
  }

  /// POST /mcp with `body`.
  HttpResponse Post(const std::string& body, std::string client = kLoopback,
                    std::vector<std::pair<std::string, std::string>> headers = {}) {
    return Serve(Req("POST", "/mcp", std::move(client), body, std::move(headers)));
  }

  static std::vector<std::pair<std::string, std::string>> Bearer(
      const std::string& token) {
    return {{"Authorization", "Bearer " + token}};
  }
};

// ---------------------------------------------------------------------------

void TestClassifyAndPreset() {
  using Kind = auth::Route::Kind;
  CHECK(auth::Classify(Req("POST", "/mcp", "x")).kind == Kind::kMcp);
  CHECK(auth::Classify(Req("GET", "/mcp", "x")).kind == Kind::kMcp);  // no page
  CHECK(auth::Classify(Req("DELETE", "/mcp", "x")).kind == Kind::kMcp);
  CHECK(auth::Classify(Req("GET", "/mcp/", "x")).kind == Kind::kPage);
  CHECK(auth::Classify(Req("POST", "/mcpx", "x")).kind == Kind::kOther);
  CHECK(auth::Classify(Req("POST", "/mcp", "x")).area.empty());
  CHECK_EQ(auth::Classify(Req("POST", "/mcp", "x")).counts, 0);

  CHECK(mcp::IsProtocolVersion("2025-03-26"));
  CHECK(mcp::IsProtocolVersion("2025-06-18"));
  CHECK(mcp::IsProtocolVersion("2025-11-25"));
  CHECK(!mcp::IsProtocolVersion("2026-07-28"));  // modern era: Commit 4
  CHECK(!mcp::IsProtocolVersion(""));

  // [router] mcp = on|off, default on; a typo stops the router.
  const auto dir = TempDir();
  const auto file = dir / "preset.ini";
  std::vector<gufo::router::PresetModel> models;
  gufo::router::RouterSection section;
  std::string error;
  for (const auto& [text, want] :
       std::vector<std::pair<std::string, std::string>>{
           {"[router]\nauth = lan\n", "on"},
           {"[router]\nauth = lan\nmcp = off\n", "off"},
           {"[router]\nmcp = on\n", "on"}}) {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out << "[remote/m]\nurl = http://192.168.1.2:8081\nmodality = llm\n" << text;
    out.close();
    section = {};
    CHECK(gufo::router::LoadPreset(file, {}, &models, &error, &section));
    CHECK_EQ(section.mcp, want);
  }
  {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out << "[remote/m]\nurl = http://192.168.1.2:8081\nmodality = llm\n"
           "[router]\nmcp = maybe\n";
    out.close();
    section = {};
    CHECK(!gufo::router::LoadPreset(file, {}, &models, &error, &section));
    CHECK(error.find("[router] mcp") != std::string::npos);
  }
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

void TestFrame() {
  Fixture f;  // auth = lan, loopback is admin: the frame alone
  // Parse error: 400, -32700, id null.
  HttpResponse r = f.Post("{");
  CHECK_EQ(r.status, 400);
  CHECK_EQ(RpcCode(r), mcp::kParseError);
  CHECK(Body(r).find("id") != nullptr && Body(r).find("id")->is_null());
  CHECK_EQ(Body(r).member_str("jsonrpc"), std::string("2.0"));
  CHECK_EQ(Header(r, "Content-Type"), std::string("application/json"));
  r = f.Post("", kLoopback, {{"Content-Type", "application/json"}});
  CHECK_EQ(r.status, 400);
  CHECK_EQ(RpcCode(r), mcp::kParseError);
  // Batches are refused (one message per POST).
  r = f.Post("[" + Msg("ping") + "]");
  CHECK_EQ(r.status, 400);
  CHECK_EQ(RpcCode(r), mcp::kInvalidRequest);
  r = f.Post("\"ping\"");
  CHECK_EQ(RpcCode(r), mcp::kInvalidRequest);
  // No method, wrong jsonrpc, bad id.
  r = f.Post("{\"jsonrpc\":\"2.0\",\"id\":1}");
  CHECK_EQ(r.status, 400);
  CHECK_EQ(RpcCode(r), mcp::kInvalidRequest);
  CHECK_EQ(Body(r).member_double("id"), 1.0);  // the id is echoed
  r = f.Post("{\"jsonrpc\":\"1.0\",\"id\":1,\"method\":\"ping\"}");
  CHECK_EQ(r.status, 400);
  CHECK_EQ(RpcCode(r), mcp::kInvalidRequest);
  r = f.Post("{\"id\":1,\"method\":\"ping\"}");
  CHECK_EQ(RpcCode(r), mcp::kInvalidRequest);
  r = f.Post("{\"jsonrpc\":\"2.0\",\"id\":{},\"method\":\"ping\"}");
  CHECK_EQ(r.status, 400);
  CHECK_EQ(RpcCode(r), mcp::kInvalidRequest);
  r = f.Post("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":7}");
  CHECK_EQ(RpcCode(r), mcp::kInvalidRequest);
  // Unknown method: 200 with -32601.
  r = f.Post(Msg("resources/list"));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(RpcCode(r), mcp::kMethodNotFound);
  r = f.Post(Msg("server/discover"));  // modern era: Commit 4
  CHECK_EQ(RpcCode(r), mcp::kMethodNotFound);
  // Notifications and client responses: 202 without a body.
  r = f.Post("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}");
  CHECK_EQ(r.status, 202);
  CHECK(r.body.empty());
  r = f.Post("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/cancelled\","
             "\"params\":{\"requestId\":1}}");
  CHECK_EQ(r.status, 202);
  r = f.Post("{\"jsonrpc\":\"2.0\",\"id\":9,\"result\":{}}");
  CHECK_EQ(r.status, 202);
  // ping: {} with the id echoed, numeric and string.
  r = f.Post(Msg("ping", "42"));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(Body(r).member_double("id"), 42.0);
  CHECK(Result(Body(r)) != nullptr && Result(Body(r))->empty());
  r = f.Post(Msg("ping", "\"abc\""));
  CHECK_EQ(Body(r).member_str("id"), std::string("abc"));
  r = f.Post("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}");  // no params
  CHECK_EQ(r.status, 200);
  CHECK(Result(Body(r)) != nullptr);
  r = f.Post(Msg("ping", "1", "[]"));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(RpcCode(r), mcp::kInvalidParams);
  // tools/call without a name or with non-object arguments.
  r = f.Post(Msg("tools/call", "1", "{}"));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(RpcCode(r), mcp::kInvalidParams);
  r = f.Post(Msg("tools/call", "1", "{\"name\":\"chat\",\"arguments\":[]}"));
  CHECK_EQ(RpcCode(r), mcp::kInvalidParams);
  // Transport: only POST with JSON; Accept must allow JSON when given.
  r = f.Serve(Req("GET", "/mcp", kLoopback));
  CHECK_EQ(r.status, 405);
  CHECK_EQ(Header(r, "Allow"), std::string("POST"));
  CHECK_EQ(ErrorCode(r), std::string("method_not_allowed"));
  r = f.Serve(Req("DELETE", "/mcp", kLoopback));
  CHECK_EQ(r.status, 405);
  r = f.Serve(Req("POST", "/mcp", kLoopback, Msg("ping"),
                  {{"Content-Type", "text/plain"}}));
  CHECK_EQ(r.status, 415);
  CHECK_EQ(ErrorCode(r), std::string("json_required"));
  r = f.Post(Msg("ping"), kLoopback, {{"Accept", "text/event-stream"}});
  CHECK_EQ(r.status, 406);
  r = f.Post(Msg("ping"), kLoopback,
             {{"Accept", "application/json, text/event-stream"}});
  CHECK_EQ(r.status, 200);
  r = f.Post(Msg("ping"), kLoopback, {{"Accept", "*/*"}});
  CHECK_EQ(r.status, 200);
  // MCP-Protocol-Version: known or missing passes, unknown is 400.
  r = f.Post(Msg("ping"), kLoopback, {{"MCP-Protocol-Version", "2025-03-26"}});
  CHECK_EQ(r.status, 200);
  r = f.Post(Msg("ping"), kLoopback, {{"MCP-Protocol-Version", "2025-11-25"}});
  CHECK_EQ(r.status, 200);
  r = f.Post(Msg("ping"), kLoopback, {{"MCP-Protocol-Version", "1999-01-01"}});
  CHECK_EQ(r.status, 400);
  CHECK_EQ(RpcCode(r), mcp::kInvalidRequest);
  CHECK(Body(r).find("id") != nullptr && Body(r).find("id")->is_null());
  {
    const json::Value body = Body(r);
    const json::Value* data = body.find("error")->find("data");
    CHECK(data != nullptr && data->find("supported") != nullptr &&
          data->find("supported")->size() == 3);
  }
  // A session id sent anyway is ignored (stateless).
  r = f.Post(Msg("ping"), kLoopback, {{"Mcp-Session-Id", "abc"}});
  CHECK_EQ(r.status, 200);
  CHECK(Header(r, "Mcp-Session-Id").empty());
  // The fake router never saw /mcp: the service answered it.
  CHECK(f.seen.empty());
}

void TestInitialize() {
  Fixture f;
  for (const char* version : {"2025-03-26", "2025-06-18", "2025-11-25"}) {
    const HttpResponse r = f.Post(Msg(
        "initialize", "1",
        std::string("{\"protocolVersion\":\"") + version +
            "\",\"capabilities\":{},\"clientInfo\":{\"name\":\"t\",\"version\":\"1\"}}"));
    CHECK_EQ(r.status, 200);
    const json::Value body = Body(r);
    const json::Value* result = Result(body);
    CHECK(result != nullptr);
    if (result == nullptr)
      continue;
    CHECK_EQ(result->member_str("protocolVersion"), std::string(version));
    const json::Value* caps = result->find("capabilities");
    CHECK(caps != nullptr && caps->find("tools") != nullptr);
    if (caps != nullptr && caps->find("tools") != nullptr)
      CHECK(!caps->find("tools")->find("listChanged")->as_bool(true));
    const json::Value* info = result->find("serverInfo");
    CHECK(info != nullptr);
    if (info != nullptr) {
      CHECK_EQ(info->member_str("name"), std::string("gufo-router"));
      CHECK(!info->member_str("version").empty());
    }
    CHECK(!result->member_str("instructions").empty());
  }
  // An unknown version gets the server's own (the spec's rule: the client
  // then decides); so does a missing one.
  HttpResponse r = f.Post(Msg("initialize", "1", "{\"protocolVersion\":\"1.0\"}"));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(Result(Body(r))->member_str("protocolVersion"),
           std::string(mcp::kDefaultProtocolVersion));
  r = f.Post(Msg("initialize", "1", "{}"));
  CHECK_EQ(Result(Body(r))->member_str("protocolVersion"),
           std::string("2025-06-18"));
  // The version comes from the options when set.
  Fixture::Config config;
  config.version = "2.1.0 (test)";
  Fixture g(config);
  r = g.Post(Msg("initialize", "1", "{\"protocolVersion\":\"2025-06-18\"}"));
  CHECK_EQ(Result(Body(r))->find("serverInfo")->member_str("version"),
           std::string("2.1.0 (test)"));
}

void TestGate() {
  Fixture f;
  f.AddUser("tim", "nutzer");
  f.AddUser("bot", "dienst");
  f.AddUser("chef", "admin");
  const std::string tim = f.Token("tim");
  const std::string bot = f.Token("bot");
  const std::string chef = f.Token("chef");
  {
    // An admin counts once it has a password (else "no_admin" setup).
    std::string error;
    CHECK(f.store->SetPassword("chef", "geheim-123", &error));
  }
  // Without a token: 401, no tool list; GET /mcp is no page (no 302).
  HttpResponse r = f.Post(Msg("tools/list"), kLan);
  CHECK_EQ(r.status, 401);
  CHECK_EQ(ErrorCode(r), std::string("not_authenticated"));
  r = f.Serve(Req("GET", "/mcp", kLan));
  CHECK_EQ(r.status, 401);
  r = f.Post(Msg("tools/list"), kLan, Fixture::Bearer("gufo_nope"));
  CHECK_EQ(r.status, 401);
  CHECK_EQ(ErrorCode(r), std::string("invalid_token"));
  // Any token role: nutzer, dienst, admin.
  for (const std::string& token : {tim, bot, chef}) {
    r = f.Post(Msg("tools/list"), kLan, Fixture::Bearer(token));
    CHECK_EQ(r.status, 200);
    CHECK_EQ(ToolNames(r).size(), kAllTools.size());  // no areas set: all
  }
  // A cookie session is refused even for an admin.
  const std::string cookie = f.LoginCookie("chef", "geheim-123");
  CHECK(!cookie.empty());
  r = f.Post(Msg("tools/list"), kLan, {{"Cookie", cookie}});
  CHECK_EQ(r.status, 403);
  CHECK_EQ(ErrorCode(r), std::string("mcp_token_required"));
  // Loopback without a token (auth = lan, auth-loopback = admin): all.
  r = f.Post(Msg("tools/list"));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(ToolNames(r).size(), kAllTools.size());
  // Origin: a token call with a foreign Origin is refused, the router's
  // own Origin passes; a cross-site fetch too.
  r = f.Post(Msg("tools/list"), kLan,
             {{"Authorization", "Bearer " + tim},
              {"Origin", "http://boese.example"},
              {"Host", "192.168.178.198:8081"}});
  CHECK_EQ(r.status, 403);
  CHECK_EQ(ErrorCode(r), std::string("csrf_rejected"));
  r = f.Post(Msg("tools/list"), kLan,
             {{"Authorization", "Bearer " + tim},
              {"Origin", "http://192.168.178.198:8081"},
              {"Host", "192.168.178.198:8081"}});
  CHECK_EQ(r.status, 200);
  r = f.Post(Msg("tools/list"), kLan,
             {{"Authorization", "Bearer " + tim},
              {"Sec-Fetch-Site", "cross-site"}});
  CHECK_EQ(r.status, 403);
  CHECK_EQ(ErrorCode(r), std::string("csrf_rejected"));
  r = f.Post(Msg("tools/list"), kLoopback,
             {{"Origin", "http://boese.example"}, {"Host", "127.0.0.1:8081"}});
  CHECK_EQ(r.status, 403);
  CHECK_EQ(ErrorCode(r), std::string("csrf_rejected"));
  // The last refusals are in the access log with the mcp path.
  {
    bool logged = false;
    for (const std::string& line : f.gate->LogTail(10))
      if (line.find("POST /mcp") != std::string::npos &&
          line.find("abgelehnt:csrf_rejected") != std::string::npos)
        logged = true;
    CHECK(logged);
  }
  // The partner router (pool key) and a relayed user: 404.
  {
    Fixture::Config config;
    config.pool_secret = "s3cret";
    Fixture p(config);
    r = p.Post(Msg("tools/list"), kLan, {{"X-Gufo-Pool-Key", "s3cret"}});
    CHECK_EQ(r.status, 404);
    CHECK_EQ(ErrorCode(r), std::string("not_found"));
    r = p.Post(Msg("tools/list"), kLan,
               {{"X-Gufo-Pool-Key", "s3cret"}, {"X-Gufo-User", "tim"}});
    CHECK_EQ(r.status, 404);
    r = p.Post(Msg("tools/list"), kLan, {{"X-Gufo-Pool-Key", "wrong"}});
    CHECK_EQ(r.status, 401);
    CHECK_EQ(ErrorCode(r), std::string("pool_key_invalid"));
  }
  // [router] mcp = off: 404 for a signed-in caller as well.
  {
    Fixture::Config config;
    config.enabled = false;
    Fixture off(config);
    off.AddUser("tim", "nutzer");
    const std::string token = off.Token("tim");
    r = off.Post(Msg("tools/list"), kLan, Fixture::Bearer(token));
    CHECK_EQ(r.status, 404);
    CHECK_EQ(ErrorCode(r), std::string("not_found"));
    r = off.Post(Msg("tools/list"));
    CHECK_EQ(r.status, 404);
    r = off.Post(Msg("tools/list"), kLan);
    CHECK_EQ(r.status, 401);  // the gate still comes first
  }
  // auth = off: open like every endpoint, full list from the LAN.
  {
    Fixture::Config config;
    config.mode = auth::Mode::kOff;
    Fixture open(config);
    r = open.Post(Msg("tools/list"), kLan);
    CHECK_EQ(r.status, 200);
    CHECK_EQ(ToolNames(r).size(), kAllTools.size());
    // The Origin rule holds at stage 0 too.
    r = open.Post(Msg("tools/list"), kLan,
                  {{"Origin", "http://boese.example"}, {"Host", "x:8081"}});
    CHECK_EQ(r.status, 403);
    CHECK_EQ(ErrorCode(r), std::string("csrf_rejected"));
    r = open.Serve(Req("GET", "/mcp", kLan));
    CHECK_EQ(r.status, 405);
  }
}

void TestVisibility() {
  Fixture f;
  auth::Rights text;
  text.bereiche = {"text"};
  f.AddUser("leser", "nutzer", text);
  auth::Rights bild;
  bild.bereiche = {"bild"};
  f.AddUser("maler", "nutzer", bild);
  auth::Rights video;
  video.bereiche = {"video"};
  f.AddUser("filmer", "nutzer", video);
  auth::Rights ablage;
  ablage.bereiche = {"ablage"};
  f.AddUser("archiv", "nutzer", ablage);
  f.AddUser("bot", "dienst");
  const std::string leser = f.Token("leser");
  const std::string maler = f.Token("maler");
  const std::string filmer = f.Token("filmer");
  const std::string archiv = f.Token("archiv");
  const std::string bot = f.Token("bot");

  HttpResponse r = f.Post(Msg("tools/list"), kLan, Fixture::Bearer(leser));
  CHECK_EQ(r.status, 200);
  CHECK(ToolNames(r) == std::vector<std::string>({"modelle", "chat"}));
  r = f.Post(Msg("tools/list"), kLan, Fixture::Bearer(maler));
  CHECK(ToolNames(r) == std::vector<std::string>(
                            {"modelle", "bild_erzeugen", "bild_bearbeiten",
                             "hochskalieren"}));
  r = f.Post(Msg("tools/list"), kLan, Fixture::Bearer(filmer));
  CHECK(ToolNames(r) ==
        std::vector<std::string>({"modelle", "video_erzeugen", "hochskalieren"}));
  r = f.Post(Msg("tools/list"), kLan, Fixture::Bearer(archiv));
  CHECK(ToolNames(r) == std::vector<std::string>(
                            {"modelle", "auftrag_status", "auftrag_liste",
                             "auftrag_abbrechen", "ablage_lesen"}));
  // A service account with every area: the whole table, nothing admin-like.
  r = f.Post(Msg("tools/list"), kLan, Fixture::Bearer(bot));
  CHECK(ToolNames(r) == kAllTools);
  for (const std::string& name : ToolNames(r)) {
    CHECK(name.find("nutzer") == std::string::npos);
    CHECK(name.find("laden") == std::string::npos);
    CHECK(name.find("user") == std::string::npos);
    CHECK(name.find("load") == std::string::npos);
  }
  // tools/call of a tool the user may not see: -32602 like an unknown one.
  r = f.Post(Msg("tools/call", "1",
                 "{\"name\":\"bild_erzeugen\",\"arguments\":{\"prompt\":\"x\"}}"),
             kLan, Fixture::Bearer(leser));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(RpcCode(r), mcp::kInvalidParams);
  CHECK(RpcMessage(r).find("Unknown tool") != std::string::npos);
  r = f.Post(Msg("tools/call", "1", "{\"name\":\"foo\"}"), kLan,
             Fixture::Bearer(leser));
  CHECK_EQ(RpcCode(r), mcp::kInvalidParams);
  // A visible tool: the Commit 1 skeleton answers isError (not a
  // protocol error), and the access log carries the mcp line.
  r = f.Post(Msg("tools/call", "7",
                 "{\"name\":\"chat\",\"arguments\":{\"nachricht\":\"hi\"}}"),
             kLan, Fixture::Bearer(leser));
  CHECK_EQ(r.status, 200);
  CHECK_EQ(RpcCode(r), 0);
  {
    const json::Value body = Body(r);
    const json::Value* result = Result(body);
    CHECK(result != nullptr);
    if (result != nullptr) {
      CHECK(result->find("isError") != nullptr &&
            result->find("isError")->as_bool());
      const json::Value* content = result->find("content");
      CHECK(content != nullptr && content->size() == 1);
      if (content != nullptr && content->size() == 1) {
        CHECK_EQ(content->items()[0].member_str("type"), std::string("text"));
        CHECK(content->items()[0].member_str("text").find("noch nicht") !=
              std::string::npos);
      }
    }
    bool logged = false;
    for (const std::string& line : f.gate->LogTail(3))
      if (line.find("\tleser\t") != std::string::npos &&
          line.find("mcp:chat:nicht_umgesetzt") != std::string::npos)
        logged = true;
    CHECK(logged);
  }
  // The fake router saw no inner request yet (Commit 2/3 send them).
  CHECK(f.seen.empty());
  // Schemas: objects with the plan's defaults; model lists from the preset.
  r = f.Post(Msg("tools/list"), kLan, Fixture::Bearer(bot));
  {
    const json::Value body = Body(r);
    const json::Value* tools = Result(body)->find("tools");
    CHECK(tools != nullptr);
    for (const json::Value& tool : tools->items()) {
      const json::Value* input = tool.find("inputSchema");
      CHECK(input != nullptr && input->member_str("type") == "object");
      CHECK(input != nullptr && input->find("properties") != nullptr);
      CHECK(!tool.member_str("description").empty());
      CHECK(!tool.member_str("title").empty());
      const std::string name = tool.member_str("name");
      if (name == "chat") {
        const json::Value& model = *input->find("properties")->find("model");
        CHECK_EQ(model.member_str("default"), std::string("qwen3.8-27b"));
        CHECK(model.member_str("description").find("qwen3.5-4b") !=
              std::string::npos);
        CHECK(model.find("enum") == nullptr);
        CHECK(input->find("required") != nullptr &&
              input->find("required")->items()[0].str() == "nachricht");
      } else if (name == "bild_erzeugen") {
        CHECK_EQ(input->find("properties")->find("model")->member_str("default"),
                 std::string("qwen-image-2.1-turbo"));
        CHECK(input->find("properties")->find("vorrang") != nullptr);
      } else if (name == "stimme") {
        // From an [external/...] section with `args = serve tts`.
        CHECK_EQ(input->find("properties")->find("model")->member_str("default"),
                 std::string("fish-drama-3"));
      } else if (name == "musik") {
        const json::Value& model = *input->find("properties")->find("model");
        CHECK_EQ(model.member_str("default"), std::string("ace-step-1.5-xl-turbo"));
        CHECK(model.member_str("description").find("moss-music") !=
              std::string::npos);
      } else if (name == "objekt_3d") {
        // trellis2 is not in this preset: the first 3D model instead.
        CHECK_EQ(input->find("properties")->find("model")->member_str("default"),
                 std::string("triposg"));
      } else if (name == "video_erzeugen") {
        CHECK_EQ(input->find("properties")->find("qualitaet")->member_str("default"),
                 std::string("genau"));
      }
      if (name == "bild_erzeugen" || name == "video_erzeugen" ||
          name == "stimme" || name == "musik" || name == "objekt_3d" ||
          name == "hochskalieren" || name == "bild_bearbeiten") {
        const json::Value* output = tool.find("outputSchema");
        CHECK(output != nullptr &&
              output->find("properties")->find("auftrag_id") != nullptr);
      }
    }
  }
  // Without the job API only chat and modelle exist, even for an admin.
  {
    Fixture::Config config;
    config.jobs_api = false;
    Fixture nojobs(config);
    r = nojobs.Post(Msg("tools/list"));
    CHECK(ToolNames(r) == std::vector<std::string>({"modelle", "chat"}));
    r = nojobs.Post(Msg("tools/call", "1", "{\"name\":\"bild_erzeugen\"}"));
    CHECK_EQ(RpcCode(r), mcp::kInvalidParams);
  }
  // The test hooks: ToolsFor and the table.
  CHECK_EQ(f.service->tools().size(), kAllTools.size());
  auth::Identity who;
  who.source = auth::Identity::Source::kBearer;
  who.name = "leser";
  who.rolle = "nutzer";
  who.rechte = text;
  CHECK_EQ(f.service->ToolsFor(who).size(), std::size_t(2));
  who.rolle = "admin";
  CHECK_EQ(f.service->ToolsFor(who).size(), kAllTools.size());
}

}  // namespace

int main() {
  TestClassifyAndPreset();
  TestFrame();
  TestInitialize();
  TestGate();
  TestVisibility();
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "router_mcp_test: ok\n";
  return 0;
}
