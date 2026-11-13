// MCP endpoint (src/cli/serve/router/mcp.*): the JSON-RPC 2.0 frame, the
// initialize handshake, tools/list per role, the gate's rules for /mcp
// (token of any role, loopback, never a cookie, the partner 404, Origin
// check, [router] mcp = off) and the Commit 2 tools (modelle, chat,
// auftrag_status/liste/abbrechen, ablage_lesen) as inner requests through
// the real gate to a fake router. Pure state, no sockets, no models.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/auth.hpp"
#include "src/cli/serve/router/comfy_graph.hpp"
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

[[maybe_unused]] bool Has(const std::vector<std::string>& names, std::string_view name) {
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
  std::size_t inline_limit{static_cast<std::size_t>(4) << 20};
};

/// A JSON answer of the fake router.
HttpResponse JsonAnswer(int status, const json::Value& body) {
  return HttpResponse{.status = status,
                      .reason = status == 200 ? "OK" : "Error",
                      .body = body.dump(),
                      .headers = {{"Content-Type", "application/json"}}};
}

/// The router's error shape.
HttpResponse ErrorAnswer(int status, const std::string& code,
                         const std::string& message) {
  json::Value detail = json::Value::object();
  detail["message"] = message;
  detail["type"] = "invalid_request_error";
  detail["code"] = code;
  json::Value body = json::Value::object();
  body["error"] = std::move(detail);
  return JsonAnswer(status, body);
}

/// A file answer of the fake job store.
HttpResponse FileAnswer(std::string bytes, const std::string& type) {
  return HttpResponse{.status = 200,
                      .reason = "OK",
                      .body = std::move(bytes),
                      .headers = {{"Content-Type", type}}};
}

struct Fixture {
  using Config = FixtureConfig;

  std::filesystem::path dir = TempDir();
  std::shared_ptr<auth::UserStore> store =
      std::make_shared<auth::UserStore>(dir / "nutzer" / "nutzer.json",
                                        kTestRounds);
  std::unique_ptr<auth::Gate> gate;
  std::unique_ptr<mcp::McpService> service;
  std::vector<std::string> seen;      // paths the fake router received
  std::vector<HttpRequest> requests;  // the checked copies the gate passed
  /// The fake router's answers (nullopt: a plain {"ok": true}).
  std::function<std::optional<HttpResponse>(const HttpRequest&)> fake;
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
      requests.push_back(req);
      if (fake)
        if (auto answer = fake(req))
          return std::move(*answer);
      json::Value body = json::Value::object();
      body["ok"] = true;
      return HttpResponse{.status = 200, .reason = "OK", .body = body.dump()};
    };
    mcp::Options options;
    options.gate = gate.get();
    options.enabled = config.enabled;
    options.jobs_api = config.jobs_api;
    options.version = config.version;
    options.inline_limit = config.inline_limit;
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

  /// tools/call `name` with `arguments` (JSON text) as `client` with
  /// `headers`.
  HttpResponse Call(const std::string& name, const std::string& arguments,
                    std::string client = kLoopback,
                    std::vector<std::pair<std::string, std::string>> headers = {}) {
    return Post(Msg("tools/call", "1",
                    "{\"name\":\"" + name + "\",\"arguments\":" + arguments + "}"),
                std::move(client), std::move(headers));
  }

  /// The inner requests of one path.
  std::vector<HttpRequest> Requests(std::string_view path) const {
    std::vector<HttpRequest> out;
    for (const HttpRequest& req : requests)
      if (req.path == path)
        out.push_back(req);
    return out;
  }

  /// The last line of zugriff.log that contains every text.
  bool Logged(std::initializer_list<std::string_view> texts,
              std::size_t lines = 6) {
    for (const std::string& line : gate->LogTail(lines)) {
      bool all = true;
      for (const std::string_view text : texts)
        if (line.find(text) == std::string::npos)
          all = false;
      if (all)
        return true;
    }
    return false;
  }
};

/// The `result` of a tools/call answer (null when it is an error frame).
json::Value CallResult(const HttpResponse& r) {
  CHECK_EQ(r.status, 200);
  const json::Value body = Body(r);
  const json::Value* result = Result(body);
  return result != nullptr ? *result : json::Value();
}

/// `result.structuredContent` of a tools/call answer.
json::Value Structured(const HttpResponse& r) {
  const json::Value result = CallResult(r);
  const json::Value* structured = result.find("structuredContent");
  return structured != nullptr ? *structured : json::Value();
}

bool IsToolError(const HttpResponse& r) {
  const json::Value result = CallResult(r);
  const json::Value* flag = result.find("isError");
  return flag != nullptr && flag->as_bool();
}

/// The content blocks of a tools/call answer.
std::vector<json::Value> Blocks(const HttpResponse& r) {
  const json::Value result = CallResult(r);
  const json::Value* content = result.find("content");
  return content != nullptr && content->is_array() ? content->items()
                                                   : std::vector<json::Value>();
}

/// A tool result is well formed: content[0] is the text mirror of
/// structuredContent (or the error text) and the flags fit.
void CheckShape(const HttpResponse& r, bool error) {
  CHECK_EQ(RpcCode(r), 0);
  CHECK_EQ(IsToolError(r), error);
  const std::vector<json::Value> blocks = Blocks(r);
  CHECK(!blocks.empty());
  if (blocks.empty())
    return;
  CHECK_EQ(blocks[0].member_str("type"), std::string("text"));
  const json::Value structured = Structured(r);
  CHECK(structured.is_object());
  if (error) {
    CHECK(structured.find("http_status") != nullptr);
    CHECK(!structured.member_str("code").empty());
    CHECK(!structured.member_str("message").empty());
    CHECK(blocks[0].member_str("text").find(structured.member_str("code")) !=
          std::string::npos);
  } else {
    CHECK_EQ(blocks[0].member_str("text"), structured.dump());
  }
}

/// The error code of a tool failure ("" for a success).
std::string FailureCode(const HttpResponse& r) {
  return IsToolError(r) ? Structured(r).member_str("code") : std::string();
}

int FailureStatus(const HttpResponse& r) {
  return static_cast<int>(Structured(r).member_double("http_status"));
}

/// The parsed JSON body of a request (null when none).
json::Value BodyOf(const HttpRequest& req) {
  try {
    return json::parse(req.body);
  } catch (const std::exception&) {
    return json::Value();
  }
}

/// A job view as the job store answers it (jobs.cpp ViewLocked).
json::Value JobView(const std::string& id, const std::string& status,
                    const std::string& owner, const std::string& path,
                    json::Value files = json::Value::array()) {
  json::Value view = json::Value::object();
  view["id"] = id;
  view["object"] = "job";
  view["status"] = status;
  json::Value request = json::Value::object();
  request["method"] = "POST";
  request["path"] = path;
  request["model"] = "qwen-image-2.1-turbo";
  view["request"] = std::move(request);
  view["running_on"] = status == "in_progress" ? json::Value("local") : json::Value();
  view["priority"] = 0;
  view["created_at"] = 1763000000;
  view["completed_at"] =
      status == "completed" ? json::Value(1763000100) : json::Value();
  view["progress"] = status == "completed" ? json::Value(100.0) : json::Value();
  view["progress_estimated"] = false;
  json::Value metadata = json::Value::object();
  metadata["nutzer"] = owner;
  metadata["titel"] = "Haus am See";
  view["metadata"] = std::move(metadata);
  json::Value result = json::Value::object();
  result["files"] = std::move(files);
  view["result"] = std::move(result);
  view["preview_url"] = nullptr;
  view["error"] = nullptr;
  return view;
}

json::Value FileEntry(const std::string& id, const std::string& name,
                      const std::string& type, long long bytes) {
  json::Value file = json::Value::object();
  file["name"] = name;
  file["url"] = "/v1/jobs/" + id + "/files/" + name;
  file["content_type"] = type;
  file["bytes"] = bytes;
  return file;
}

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
  for (const auto& [text, want, inline_mb] :
       std::vector<std::tuple<std::string, std::string, std::string>>{
           {"[router]\nauth = lan\n", "on", "4"},
           {"[router]\nauth = lan\nmcp = off\n", "off", "4"},
           {"[router]\nmcp = on\nmcp-inline-mb = 8\n", "on", "8"},
           {"[router]\nmcp-inline-mb = 0\n", "on", "0"}}) {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out << "[remote/m]\nurl = http://192.168.1.2:8081\nmodality = llm\n" << text;
    out.close();
    section = {};
    CHECK(gufo::router::LoadPreset(file, {}, &models, &error, &section));
    CHECK_EQ(section.mcp, want);
    CHECK_EQ(section.mcp_inline_mb, inline_mb);
  }
  for (const char* bad : {"mcp = maybe", "mcp-inline-mb = viel",
                          "mcp-inline-mb = 4.5", "mcp-inline-mb = -1"}) {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out << "[remote/m]\nurl = http://192.168.1.2:8081\nmodality = llm\n"
           "[router]\n"
        << bad << "\n";
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
  // A visible media job tool (Commit 3) answers isError (not a protocol
  // error), and the access log carries the mcp line.
  r = f.Post(Msg("tools/call", "7",
                 "{\"name\":\"bild_erzeugen\",\"arguments\":{\"prompt\":\"x\"}}"),
             kLan, Fixture::Bearer(maler));
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
    CHECK(f.Logged({"\tmaler\t", "mcp:bild_erzeugen:nicht_umgesetzt"}, 3));
  }
  // The fake router saw no inner request for it (Commit 3 sends them).
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

// --- Commit 2: the tools as inner requests -------------------------------

void TestModelle() {
  Fixture f;
  auth::Rights one;
  one.modelle = {"qwen3.8-27b"};
  f.AddUser("eng", "nutzer", one);
  const std::string eng = f.Token("eng");
  f.fake = [](const HttpRequest& req) -> std::optional<HttpResponse> {
    if (req.method != "GET" || req.path != "/v1/models")
      return std::nullopt;
    json::Value body = json::parse(R"json({"object":"list","data":[
      {"id":"qwen3.8-27b","object":"model","modality":"llm","loaded":true,
       "status":{"value":"loaded","args":["serve","llm"]}},
      {"id":"qwen-image-2.1-turbo","object":"model","modality":"image","loaded":false,
       "remote":{"url":"http://192.168.178.56:8081","reachable":true}},
      {"id":"fish-drama-3","object":"model","modality":"tts","loaded":false,
       "switchable":true,"placement":"remote"},
      {"id":"ace-step-1.5-xl-turbo","object":"model","modality":"music","loaded":false}
    ]})json");
    return JsonAnswer(200, body);
  };
  // Loopback admin: every model, German modality, load state, placement.
  HttpResponse r = f.Call("modelle", "{}");
  CheckShape(r, false);
  {
    const json::Value s = Structured(r);
    const json::Value* list = s.find("modelle");
    CHECK(list != nullptr && list->size() == 4);
    if (list != nullptr && list->size() == 4) {
      const auto& m = list->items();
      CHECK_EQ(m[0].member_str("id"), std::string("qwen3.8-27b"));
      CHECK_EQ(m[0].member_str("modalitaet"), std::string("text"));
      CHECK(m[0].find("geladen")->as_bool());
      CHECK_EQ(m[0].member_str("ort"), std::string("lokal"));
      CHECK_EQ(m[1].member_str("modalitaet"), std::string("bild"));
      CHECK(!m[1].find("geladen")->as_bool());
      CHECK_EQ(m[1].member_str("ort"), std::string("partner"));
      CHECK_EQ(m[2].member_str("modalitaet"), std::string("stimme"));
      CHECK_EQ(m[2].member_str("ort"), std::string("partner"));
      CHECK_EQ(m[3].member_str("modalitaet"), std::string("musik"));
    }
  }
  CHECK_EQ(f.Requests("/v1/models").size(), std::size_t(1));
  // The filter, a wrong filter.
  r = f.Call("modelle", "{\"modalitaet\":\"bild\"}");
  CheckShape(r, false);
  CHECK_EQ(Structured(r).find("modelle")->size(), std::size_t(1));
  r = f.Call("modelle", "{\"modalitaet\":\"foo\"}");
  CHECK_EQ(RpcCode(r), mcp::kInvalidParams);
  // A user sees only the models of its list.
  r = f.Call("modelle", "{}", kLan, Fixture::Bearer(eng));
  CheckShape(r, false);
  CHECK_EQ(Structured(r).find("modelle")->size(), std::size_t(1));
  CHECK(f.Logged({"\teng\t", "mcp:modelle:ok"}, 2));
  // A router error becomes a tool failure.
  f.fake = [](const HttpRequest&) -> std::optional<HttpResponse> {
    return ErrorAnswer(503, "pool_busy", "every machine is busy");
  };
  r = f.Call("modelle", "{}");
  CheckShape(r, true);
  CHECK_EQ(FailureCode(r), std::string("pool_busy"));
  CHECK_EQ(FailureStatus(r), 503);
  CHECK(f.Logged({"mcp:modelle:abgelehnt:pool_busy"}, 2));
  // A 2xx that is no model list.
  f.fake = [](const HttpRequest&) -> std::optional<HttpResponse> {
    return FileAnswer("<html>", "text/html");
  };
  r = f.Call("modelle", "{}");
  CheckShape(r, true);
  CHECK_EQ(FailureCode(r), std::string("unexpected_answer"));
}

void TestChat() {
  Fixture f;
  auth::Rights text;
  text.bereiche = {"text"};
  f.AddUser("leser", "nutzer", text);
  auth::Rights seher_rights;
  seher_rights.bereiche = {"text", "ablage"};
  f.AddUser("seher", "nutzer", seher_rights);
  auth::Rights only4b;
  only4b.bereiche = {"text"};
  only4b.modelle = {"qwen3.5-4b"};
  f.AddUser("eng", "nutzer", only4b);
  auth::Rights limited;
  limited.bereiche = {"text"};
  limited.tageslimit = 1;
  f.AddUser("knapp", "nutzer", limited);
  const std::string leser = f.Token("leser");
  const std::string seher = f.Token("seher");
  const std::string eng = f.Token("eng");
  const std::string knapp = f.Token("knapp");

  bool stream = false;
  std::string content = "Hallo!";
  std::string reasoning = "kurz nachgedacht";
  f.fake = [&](const HttpRequest& req) -> std::optional<HttpResponse> {
    if (req.method == "GET" && req.path == "/v1/jobs/j1/files/a.png")
      return FileAnswer("PNGDATA", "image/png");
    if (req.method != "POST" || req.path != "/v1/chat/completions")
      return std::nullopt;
    json::Value message = json::Value::object();
    message["role"] = "assistant";
    message["content"] = content;
    if (!reasoning.empty())
      message["reasoning_content"] = reasoning;
    json::Value choice = json::Value::object();
    choice["index"] = 0;
    choice["message"] = std::move(message);
    choice["finish_reason"] = "stop";
    json::Value choices = json::Value::array();
    choices.push_back(std::move(choice));
    json::Value usage = json::Value::object();
    usage["prompt_tokens"] = 12;
    usage["completion_tokens"] = 3;
    json::Value reply = json::Value::object();
    reply["id"] = "c1";
    reply["object"] = "chat.completion";
    reply["model"] = BodyOf(req).member_str("model");
    reply["choices"] = std::move(choices);
    reply["usage"] = std::move(usage);
    HttpResponse r = JsonAnswer(200, reply);
    if (stream) {
      // The relay streams the worker's answer even without `stream`.
      const std::string body = std::move(r.body);
      r.body.clear();
      r.streaming_body = [body](const HttpResponse::BodyWriter& write) {
        write(body.substr(0, 10));
        write(body.substr(10));
      };
    }
    return r;
  };

  // Case 5: the inner request and the answer.
  HttpResponse r = f.Call(
      "chat",
      R"json({"nachricht":"Hallo","system":"Sei knapp",
              "verlauf":[{"rolle":"user","text":"a"},{"rolle":"assistant","text":"b"}],
              "denken":true,"max_tokens":50,"temperature":0.2,"json":true})json",
      kLan, Fixture::Bearer(leser));
  CheckShape(r, false);
  {
    const json::Value s = Structured(r);
    CHECK_EQ(s.member_str("text"), std::string("Hallo!"));
    CHECK_EQ(s.member_str("gedanken"), std::string("kurz nachgedacht"));
    CHECK_EQ(s.member_str("model"), std::string("qwen3.8-27b"));
    CHECK_EQ(s.member_str("finish_reason"), std::string("stop"));
    const json::Value* usage = s.find("usage");
    CHECK(usage != nullptr && usage->member_double("prompt_tokens") == 12.0);
    CHECK(usage != nullptr && usage->member_double("completion_tokens") == 3.0);
  }
  {
    const auto inner = f.Requests("/v1/chat/completions");
    CHECK_EQ(inner.size(), std::size_t(1));
    if (inner.size() == 1) {
      const HttpRequest& req = inner[0];
      CHECK_EQ(req.method, std::string("POST"));
      CHECK_EQ(req.header("X-Gufo-User"), std::string("leser"));
      CHECK(req.header("Authorization").starts_with("Bearer gufo_"));
      CHECK_EQ(req.header("Content-Type"), std::string("application/json"));
      CHECK_EQ(req.client_id, std::string(kLan));
      const json::Value body = BodyOf(req);
      CHECK(body.find("stream") != nullptr && !body.find("stream")->as_bool(true));
      CHECK_EQ(body.member_str("model"), std::string("qwen3.8-27b"));
      CHECK_EQ(body.member_double("max_tokens"), 50.0);
      CHECK(std::abs(body.member_double("temperature") - 0.2) < 1e-9);
      CHECK(body.find("chat_template_kwargs") != nullptr &&
            body.find("chat_template_kwargs")->find("enable_thinking")->as_bool());
      CHECK(body.find("response_format") != nullptr &&
            body.find("response_format")->member_str("type") == "json_object");
      const json::Value* messages = body.find("messages");
      CHECK(messages != nullptr && messages->size() == 4);
      if (messages != nullptr && messages->size() == 4) {
        const auto& m = messages->items();
        CHECK_EQ(m[0].member_str("role"), std::string("system"));
        CHECK_EQ(m[0].member_str("content"), std::string("Sei knapp"));
        CHECK_EQ(m[1].member_str("role"), std::string("user"));
        CHECK_EQ(m[1].member_str("content"), std::string("a"));
        CHECK_EQ(m[2].member_str("role"), std::string("assistant"));
        CHECK_EQ(m[3].member_str("role"), std::string("user"));
        CHECK_EQ(m[3].member_str("content"), std::string("Hallo"));
      }
    }
  }
  // Case 8: the access log has the mcp line (with the model) and the
  // inner request's own line.
  CHECK(f.Logged({"\tleser\t", "POST /mcp", "qwen3.8-27b", "mcp:chat:ok"}, 3));
  CHECK(f.Logged({"\tleser\t", "POST /v1/chat/completions", "qwen3.8-27b", "\tok"}, 3));
  // Minimal call: no thinking, no extras, the default model.
  r = f.Call("chat", "{\"nachricht\":\"Hi\"}", kLan, Fixture::Bearer(leser));
  CheckShape(r, false);
  {
    const json::Value body = BodyOf(f.Requests("/v1/chat/completions").back());
    CHECK(!body.find("chat_template_kwargs")->find("enable_thinking")->as_bool(true));
    CHECK(body.find("max_tokens") == nullptr);
    CHECK(body.find("response_format") == nullptr);
    CHECK(body.find("messages")->size() == 1);
  }
  // A streamed answer is collected.
  stream = true;
  r = f.Call("chat", "{\"nachricht\":\"Hi\",\"model\":\"qwen3.5-4b\"}", kLan,
             Fixture::Bearer(leser));
  CheckShape(r, false);
  CHECK_EQ(Structured(r).member_str("text"), std::string("Hallo!"));
  CHECK_EQ(Structured(r).member_str("model"), std::string("qwen3.5-4b"));
  CHECK(f.Logged({"mcp:chat:ok", "qwen3.5-4b"}, 2));
  stream = false;
  // Thinking left in the text is split off.
  reasoning.clear();
  content = "<think>\nDenk...\n</think>\n\nAntwort";
  r = f.Call("chat", "{\"nachricht\":\"Hi\"}", kLan, Fixture::Bearer(leser));
  CheckShape(r, false);
  CHECK_EQ(Structured(r).member_str("text"), std::string("Antwort"));
  CHECK_EQ(Structured(r).member_str("gedanken"), std::string("Denk..."));
  content = "Hallo!";
  reasoning = "kurz nachgedacht";
  // Images: a data URL goes through, an Ablage path is read through the
  // gate (area ablage) and sent as a data URL.
  r = f.Call("chat",
             "{\"nachricht\":\"Was ist das?\",\"bilder\":[\"data:image/png;base64,QUJD\"]}",
             kLan, Fixture::Bearer(leser));
  CheckShape(r, false);
  {
    const json::Value body = BodyOf(f.Requests("/v1/chat/completions").back());
    const json::Value& user = body.find("messages")->items().back();
    const json::Value* parts = user.find("content");
    CHECK(parts != nullptr && parts->is_array() && parts->size() == 2);
    if (parts != nullptr && parts->is_array() && parts->size() == 2) {
      CHECK_EQ(parts->items()[0].member_str("type"), std::string("text"));
      CHECK_EQ(parts->items()[0].member_str("text"), std::string("Was ist das?"));
      CHECK_EQ(parts->items()[1].member_str("type"), std::string("image_url"));
      CHECK_EQ(parts->items()[1].find("image_url")->member_str("url"),
               std::string("data:image/png;base64,QUJD"));
    }
  }
  r = f.Call("chat", "{\"nachricht\":\"Bild?\",\"bilder\":[\"/v1/jobs/j1/files/a.png\"]}",
             kLan, Fixture::Bearer(leser));
  CheckShape(r, true);  // leser has no area ablage
  CHECK_EQ(FailureCode(r), std::string("forbidden"));
  CHECK_EQ(FailureStatus(r), 403);
  r = f.Call("chat", "{\"nachricht\":\"Bild?\",\"bilder\":[\"/v1/jobs/j1/files/a.png\"]}",
             kLan, Fixture::Bearer(seher));
  CheckShape(r, false);
  {
    const json::Value body = BodyOf(f.Requests("/v1/chat/completions").back());
    const json::Value& user = body.find("messages")->items().back();
    CHECK_EQ(user.find("content")->items()[1].find("image_url")->member_str("url"),
             "data:image/png;base64," + gufo::router::comfy::Base64Encode("PNGDATA"));
    CHECK_EQ(f.Requests("/v1/jobs/j1/files/a.png").size(), std::size_t(1));
  }
  // Bad arguments: -32602 naming the field, nothing sent.
  const std::size_t sent = f.requests.size();
  for (const char* bad :
       {"{}", "{\"nachricht\":7}", "{\"nachricht\":\"\"}",
        "{\"nachricht\":\"x\",\"verlauf\":[{\"rolle\":\"system\",\"text\":\"a\"}]}",
        "{\"nachricht\":\"x\",\"verlauf\":\"a\"}",
        "{\"nachricht\":\"x\",\"bilder\":[\"http://x/y.png\"]}",
        "{\"nachricht\":\"x\",\"max_tokens\":0}",
        "{\"nachricht\":\"x\",\"temperature\":-1}",
        "{\"nachricht\":\"x\",\"denken\":\"ja\"}"}) {
    r = f.Call("chat", bad, kLan, Fixture::Bearer(leser));
    CHECK_EQ(r.status, 200);
    CHECK_EQ(RpcCode(r), mcp::kInvalidParams);
    CHECK(RpcMessage(r).find("arguments.") != std::string::npos);
  }
  CHECK_EQ(f.requests.size(), sent);
  CHECK(f.Logged({"mcp:chat:abgelehnt:invalid_params"}, 2));
  // The gate's refusals of the inner request: model list, word filter,
  // daily limit (counted by the inner request alone).
  r = f.Call("chat", "{\"nachricht\":\"Hi\",\"model\":\"qwen3.8-27b\"}", kLan,
             Fixture::Bearer(eng));
  CheckShape(r, true);
  CHECK_EQ(FailureCode(r), std::string("model_forbidden"));
  CHECK_EQ(FailureStatus(r), 403);
  CHECK_EQ(f.requests.size(), sent);
  CHECK(f.Logged({"\teng\t", "mcp:chat:abgelehnt:model_forbidden"}, 3));
  r = f.Call("chat", "{\"nachricht\":\"Hi\",\"model\":\"qwen3.5-4b\"}", kLan,
             Fixture::Bearer(eng));
  CheckShape(r, false);
  r = f.Call("chat", "{\"nachricht\":\"Eine NACKTE Frau\"}", kLan,
             Fixture::Bearer(leser));
  CheckShape(r, true);
  CHECK_EQ(FailureCode(r), std::string("content_filtered"));
  CHECK_EQ(FailureStatus(r), 400);
  r = f.Call("chat", "{\"nachricht\":\"Hi\"}", kLan, Fixture::Bearer(knapp));
  CheckShape(r, false);
  r = f.Call("chat", "{\"nachricht\":\"Hi\"}", kLan, Fixture::Bearer(knapp));
  CheckShape(r, true);
  CHECK_EQ(FailureCode(r), std::string("daily_limit"));
  CHECK_EQ(FailureStatus(r), 429);
  // The gate sends no Retry-After for the daily limit: no retry_after.
  CHECK(Structured(r).find("retry_after") == nullptr);
  CHECK(f.Logged({"\tknapp\t", "mcp:chat:abgelehnt:daily_limit"}, 3));
  // An answer without choices.
  f.fake = [](const HttpRequest&) -> std::optional<HttpResponse> {
    return JsonAnswer(200, json::parse("{\"object\":\"list\"}"));
  };
  r = f.Call("chat", "{\"nachricht\":\"Hi\"}", kLan, Fixture::Bearer(leser));
  CheckShape(r, true);
  CHECK_EQ(FailureCode(r), std::string("unexpected_answer"));
}

void TestJobs() {
  Fixture f;
  auth::Rights ablage;
  ablage.bereiche = {"ablage"};
  f.AddUser("tim", "nutzer", ablage);
  f.AddUser("bob", "nutzer", ablage);
  const std::string tim = f.Token("tim");
  const std::string bob = f.Token("bob");
  const std::string png = std::string("\x89PNG\r\n\x1a\n", 8) + std::string(92, 'x');
  const std::string wav = "RIFF" + std::string(56, 'w');
  const std::string mp4 = std::string(30, 'v');
  json::Value files = json::Value::array();
  files.push_back(FileEntry("j1", "request.body", "application/json", 40));
  files.push_back(FileEntry("j1", "image-1.png", "image/png", 100));
  files.push_back(FileEntry("j1", "ton.wav", "audio/wav", 60));
  files.push_back(FileEntry("j1", "video.mp4", "video/mp4", 30));
  const json::Value j1 = JobView("j1", "completed", "tim", "/v1/images/generations", files);
  json::Value j2 = JobView("j2", "in_progress", "tim", "/v1/videos");
  j2["progress"] = 42.5;
  j2["progress_estimated"] = true;
  j2["position"] = 1;
  j2["preview_url"] = "/v1/previews/p1";
  const json::Value j3 = JobView("j3", "queued", "tim", "/v1/audio/speech");
  json::Value j4 = JobView("j4", "failed", "tim", "/v1/audio/music");
  j4["error"] = "worker died";
  const json::Value jx = JobView("jx", "completed", "bob", "/v1/images/edits");
  f.fake = [&](const HttpRequest& req) -> std::optional<HttpResponse> {
    const std::string& p = req.path;
    if (req.method == "GET") {
      if (p == "/v1/jobs") {
        json::Value body = json::Value::object();
        json::Value data = json::Value::array();
        data.push_back(j1);
        data.push_back(j2);
        data.push_back(jx);
        body["object"] = "list";
        body["data"] = std::move(data);
        return JsonAnswer(200, body);
      }
      if (p == "/v1/jobs/j1") return JsonAnswer(200, j1);
      if (p == "/v1/jobs/j2") return JsonAnswer(200, j2);
      if (p == "/v1/jobs/j3") return JsonAnswer(200, j3);
      if (p == "/v1/jobs/j4") return JsonAnswer(200, j4);
      if (p == "/v1/jobs/jx") return JsonAnswer(200, jx);
      if (p == "/v1/jobs/j1/files/image-1.png") return FileAnswer(png, "image/png");
      if (p == "/v1/jobs/j1/files/ton.wav") return FileAnswer(wav, "audio/wav");
      if (p == "/v1/jobs/j1/files/video.mp4") return FileAnswer(mp4, "video/mp4");
      if (p == "/v1/jobs/j1/files/extra.jpg") return FileAnswer("JPG", "image/jpeg");
      if (p.starts_with("/v1/jobs/j1/files/"))
        return ErrorAnswer(404, "file_not_found", "file not found");
      if (p.starts_with("/v1/jobs/"))
        return ErrorAnswer(404, "job_not_found", "job not found");
    }
    if (req.method == "DELETE") {
      if (p == "/v1/jobs/j1") {
        json::Value body = json::Value::object();
        body["id"] = "j1";
        body["object"] = "job";
        body["deleted"] = true;
        return JsonAnswer(200, body);
      }
      if (p == "/v1/jobs/j3") return JsonAnswer(200, JobView("j3", "cancelled", "tim", "/v1/audio/speech"));
      return ErrorAnswer(404, "job_not_found", "job not found");
    }
    return std::nullopt;
  };

  // Case 7: auftrag_status maps the view.
  HttpResponse r = f.Call("auftrag_status", "{\"auftrag_id\":\"j1\"}", kLan,
                          Fixture::Bearer(tim));
  CheckShape(r, false);
  {
    const json::Value s = Structured(r);
    CHECK_EQ(s.member_str("auftrag_id"), std::string("j1"));
    CHECK_EQ(s.member_str("status"), std::string("completed"));
    CHECK_EQ(s.member_str("art"), std::string("image"));
    CHECK_EQ(s.member_str("model"), std::string("qwen-image-2.1-turbo"));
    CHECK_EQ(s.member_str("titel"), std::string("Haus am See"));
    CHECK_EQ(s.member_double("fortschritt"), 100.0);
    CHECK_EQ(s.member_str("url"), std::string("/v1/jobs/j1"));
    CHECK_EQ(s.member_str("erstellt").size(), std::size_t(20));
    CHECK(s.member_str("erstellt").starts_with("2025-11-13T"));
    CHECK_EQ(s.member_str("fertig").size(), std::size_t(20));
    CHECK(s.find("fehler") == nullptr);
    const json::Value* dateien = s.find("dateien");
    CHECK(dateien != nullptr && dateien->size() == 4);
    if (dateien != nullptr && dateien->size() == 4) {
      CHECK_EQ(dateien->items()[1].member_str("name"), std::string("image-1.png"));
      CHECK_EQ(dateien->items()[1].member_str("url"),
               std::string("/v1/jobs/j1/files/image-1.png"));
      CHECK_EQ(dateien->items()[1].member_double("bytes"), 100.0);
    }
  }
  r = f.Call("auftrag_status", "{\"auftrag_id\":\"j2\"}", kLan, Fixture::Bearer(tim));
  CheckShape(r, false);
  {
    const json::Value s = Structured(r);
    CHECK_EQ(s.member_str("status"), std::string("in_progress"));
    CHECK_EQ(s.member_str("art"), std::string("video"));
    CHECK_EQ(s.member_double("fortschritt"), 42.5);
    CHECK(s.find("geschaetzt")->as_bool());
    CHECK_EQ(s.member_str("laeuft_auf"), std::string("lokal"));
    CHECK_EQ(s.member_double("position"), 1.0);
    CHECK_EQ(s.member_str("vorschau_url"), std::string("/v1/previews/p1"));
    CHECK(s.find("fertig") == nullptr);
    CHECK(s.find("dateien")->size() == 0);
  }
  r = f.Call("auftrag_status", "{\"auftrag_id\":\"j3\"}", kLan, Fixture::Bearer(tim));
  CheckShape(r, false);
  CHECK(Structured(r).find("fortschritt")->is_null());
  CHECK(Structured(r).find("laeuft_auf") == nullptr);
  r = f.Call("auftrag_status", "{\"auftrag_id\":\"j4\"}", kLan, Fixture::Bearer(tim));
  CheckShape(r, false);
  CHECK_EQ(Structured(r).member_str("fehler"), std::string("worker died"));
  CHECK_EQ(Structured(r).member_str("art"), std::string("music"));
  // Another user's job: the gate answers 404 (hidden), an unknown id too.
  r = f.Call("auftrag_status", "{\"auftrag_id\":\"j1\"}", kLan, Fixture::Bearer(bob));
  CheckShape(r, true);
  CHECK_EQ(FailureStatus(r), 404);
  CHECK_EQ(FailureCode(r), std::string("not_found"));
  CHECK(f.Logged({"\tbob\t", "mcp:auftrag_status:abgelehnt:not_found"}, 3));
  r = f.Call("auftrag_status", "{\"auftrag_id\":\"j9\"}", kLan, Fixture::Bearer(tim));
  CheckShape(r, true);
  CHECK_EQ(FailureStatus(r), 404);
  CHECK_EQ(FailureCode(r), std::string("job_not_found"));
  for (const char* bad : {"{}", "{\"auftrag_id\":\"\"}", "{\"auftrag_id\":\"a/b\"}",
                          "{\"auftrag_id\":\"..\"}", "{\"auftrag_id\":\"a?x=1\"}",
                          "{\"auftrag_id\":5}"}) {
    r = f.Call("auftrag_status", bad, kLan, Fixture::Bearer(tim));
    CHECK_EQ(RpcCode(r), mcp::kInvalidParams);
  }

  // auftrag_liste: the gate keeps the caller's jobs; the filters travel
  // as query parameters.
  r = f.Call("auftrag_liste", "{}", kLan, Fixture::Bearer(tim));
  CheckShape(r, false);
  {
    const json::Value s = Structured(r);  // keeps the pointer below alive
    const json::Value* list = s.find("auftraege");
    CHECK(list != nullptr && list->size() == 2);
    if (list != nullptr && list->size() == 2) {
      CHECK_EQ(list->items()[0].member_str("auftrag_id"), std::string("j1"));
      CHECK_EQ(list->items()[0].member_str("titel"), std::string("Haus am See"));
      CHECK_EQ(list->items()[0].member_str("url"), std::string("/v1/jobs/j1"));
      CHECK(list->items()[0].find("dateien") == nullptr);  // the short view
      CHECK_EQ(list->items()[1].member_str("auftrag_id"), std::string("j2"));
    }
    CHECK_EQ(f.Requests("/v1/jobs").back().query, std::string("limit=20"));
  }
  r = f.Call("auftrag_liste",
             "{\"status\":\"completed\",\"art\":\"image\",\"limit\":5,\"suche\":\"Haus am See\"}",
             kLan, Fixture::Bearer(tim));
  CheckShape(r, false);
  CHECK_EQ(f.Requests("/v1/jobs").back().query,
           std::string("limit=5&status=completed&kind=image&q=Haus%20am%20See"));
  CHECK_EQ(f.Requests("/v1/jobs").back().query_param("q"), std::string("Haus am See"));
  r = f.Call("auftrag_liste", "{}");  // loopback admin: every job
  CheckShape(r, false);
  CHECK_EQ(Structured(r).find("auftraege")->size(), std::size_t(3));
  for (const char* bad : {"{\"limit\":0}", "{\"limit\":51}", "{\"status\":\"done\"}",
                          "{\"art\":\"foo\"}", "{\"suche\":3}"}) {
    r = f.Call("auftrag_liste", bad, kLan, Fixture::Bearer(tim));
    CHECK_EQ(RpcCode(r), mcp::kInvalidParams);
  }

  // auftrag_abbrechen: a finished job is not touched without loeschen.
  std::size_t deletes = 0;
  const auto count_deletes = [&] {
    deletes = 0;
    for (const HttpRequest& req : f.requests)
      if (req.method == "DELETE")
        ++deletes;
    return deletes;
  };
  r = f.Call("auftrag_abbrechen", "{\"auftrag_id\":\"j1\"}", kLan, Fixture::Bearer(tim));
  CheckShape(r, true);
  CHECK_EQ(FailureStatus(r), 409);
  CHECK_EQ(FailureCode(r), std::string("job_finished"));
  CHECK_EQ(Structured(r).member_str("status"), std::string("completed"));
  CHECK_EQ(count_deletes(), std::size_t(0));
  r = f.Call("auftrag_abbrechen", "{\"auftrag_id\":\"j1\",\"loeschen\":true}", kLan,
             Fixture::Bearer(tim));
  CheckShape(r, false);
  CHECK(Structured(r).find("geloescht")->as_bool());
  CHECK_EQ(Structured(r).member_str("status"), std::string("completed"));
  CHECK_EQ(count_deletes(), std::size_t(1));
  r = f.Call("auftrag_abbrechen", "{\"auftrag_id\":\"j3\"}", kLan, Fixture::Bearer(tim));
  CheckShape(r, false);
  CHECK(!Structured(r).find("geloescht")->as_bool());
  CHECK_EQ(Structured(r).member_str("status"), std::string("cancelled"));
  CHECK_EQ(count_deletes(), std::size_t(2));
  r = f.Call("auftrag_abbrechen", "{\"auftrag_id\":\"j1\",\"loeschen\":true}", kLan,
             Fixture::Bearer(bob));
  CheckShape(r, true);
  CHECK_EQ(FailureStatus(r), 404);
  CHECK_EQ(count_deletes(), std::size_t(2));
  r = f.Call("auftrag_abbrechen", "{\"auftrag_id\":\"j1\",\"loeschen\":\"ja\"}", kLan,
             Fixture::Bearer(tim));
  CHECK_EQ(RpcCode(r), mcp::kInvalidParams);
  CHECK(f.Logged({"\ttim\t", "mcp:auftrag_abbrechen:ok"}, 4));

  // Case 7: ablage_lesen: the link by default, the content on request.
  const std::string base = "http://192.168.178.198:8081/v1/jobs/j1/files/";
  const std::size_t fetched_before = f.Requests("/v1/jobs/j1/files/image-1.png").size();
  r = f.Call("ablage_lesen", "{\"auftrag_id\":\"j1\"}", kLan, Fixture::Bearer(tim));
  CheckShape(r, false);
  {
    const json::Value s = Structured(r);
    CHECK_EQ(s.member_str("auftrag_id"), std::string("j1"));
    CHECK_EQ(s.member_str("name"), std::string("image-1.png"));  // first media
    CHECK_EQ(s.member_str("url"), base + "image-1.png");
    CHECK_EQ(s.member_str("content_type"), std::string("image/png"));
    CHECK_EQ(s.member_double("bytes"), 100.0);
    CHECK(!s.find("eingebettet")->as_bool());
    const std::vector<json::Value> blocks = Blocks(r);
    CHECK_EQ(blocks.size(), std::size_t(2));
    if (blocks.size() == 2) {
      CHECK_EQ(blocks[1].member_str("type"), std::string("resource_link"));
      CHECK_EQ(blocks[1].member_str("uri"), base + "image-1.png");
      CHECK_EQ(blocks[1].member_str("name"), std::string("image-1.png"));
      CHECK_EQ(blocks[1].member_str("mimeType"), std::string("image/png"));
      CHECK_EQ(blocks[1].member_double("size"), 100.0);
    }
    // Nothing was fetched for a link.
    CHECK_EQ(f.Requests("/v1/jobs/j1/files/image-1.png").size(), fetched_before);
  }
  r = f.Call("ablage_lesen", "{\"auftrag_id\":\"j1\",\"inhalt\":true}", kLan,
             Fixture::Bearer(tim));
  CheckShape(r, false);
  {
    CHECK(Structured(r).find("eingebettet")->as_bool());
    const std::vector<json::Value> blocks = Blocks(r);
    CHECK_EQ(blocks.size(), std::size_t(2));
    if (blocks.size() == 2) {
      CHECK_EQ(blocks[1].member_str("type"), std::string("image"));
      CHECK_EQ(blocks[1].member_str("mimeType"), std::string("image/png"));
      CHECK_EQ(blocks[1].member_str("data"), gufo::router::comfy::Base64Encode(png));
    }
    CHECK_EQ(f.Requests("/v1/jobs/j1/files/image-1.png").size(), fetched_before + 1);
    // The gate logs no GET (the file read has no line of its own); the
    // mcp line is there.
    CHECK(f.Logged({"\ttim\t", "POST /mcp", "mcp:ablage_lesen:ok"}, 2));
  }
  // Over max_bytes: the link only, nothing fetched.
  r = f.Call("ablage_lesen", "{\"auftrag_id\":\"j1\",\"inhalt\":true,\"max_bytes\":50}",
             kLan, Fixture::Bearer(tim));
  CheckShape(r, false);
  CHECK(!Structured(r).find("eingebettet")->as_bool());
  CHECK_EQ(Blocks(r)[1].member_str("type"), std::string("resource_link"));
  CHECK_EQ(f.Requests("/v1/jobs/j1/files/image-1.png").size(), fetched_before + 1);
  // Audio inline, video never.
  r = f.Call("ablage_lesen", "{\"auftrag_id\":\"j1\",\"datei\":\"ton.wav\",\"inhalt\":true}",
             kLan, Fixture::Bearer(tim));
  CheckShape(r, false);
  CHECK_EQ(Blocks(r)[1].member_str("type"), std::string("audio"));
  CHECK_EQ(Blocks(r)[1].member_str("mimeType"), std::string("audio/wav"));
  CHECK_EQ(Blocks(r)[1].member_str("data"), gufo::router::comfy::Base64Encode(wav));
  r = f.Call("ablage_lesen", "{\"auftrag_id\":\"j1\",\"datei\":\"video.mp4\",\"inhalt\":true}",
             kLan, Fixture::Bearer(tim));
  CheckShape(r, false);
  CHECK_EQ(Blocks(r)[1].member_str("type"), std::string("resource_link"));
  CHECK_EQ(Blocks(r)[1].member_str("uri"), base + "video.mp4");
  CHECK_EQ(Structured(r).member_str("content_type"), std::string("video/mp4"));
  CHECK_EQ(f.Requests("/v1/jobs/j1/files/video.mp4").size(), std::size_t(0));
  // A file the list does not know is fetched for its type and size; a
  // missing one is the store's 404.
  r = f.Call("ablage_lesen", "{\"auftrag_id\":\"j1\",\"datei\":\"extra.jpg\"}", kLan,
             Fixture::Bearer(tim));
  CheckShape(r, false);
  CHECK_EQ(Structured(r).member_str("content_type"), std::string("image/jpeg"));
  CHECK_EQ(Structured(r).member_double("bytes"), 3.0);
  r = f.Call("ablage_lesen", "{\"auftrag_id\":\"j1\",\"datei\":\"nix.png\"}", kLan,
             Fixture::Bearer(tim));
  CheckShape(r, true);
  CHECK_EQ(FailureCode(r), std::string("file_not_found"));
  // A job without media files, another user's job, bad names.
  r = f.Call("ablage_lesen", "{\"auftrag_id\":\"j3\"}", kLan, Fixture::Bearer(tim));
  CheckShape(r, true);
  CHECK_EQ(FailureCode(r), std::string("file_not_found"));
  CHECK_EQ(Structured(r).member_str("status"), std::string("queued"));
  r = f.Call("ablage_lesen", "{\"auftrag_id\":\"j1\"}", kLan, Fixture::Bearer(bob));
  CheckShape(r, true);
  CHECK_EQ(FailureStatus(r), 404);
  for (const char* bad : {"{\"auftrag_id\":\"j1\",\"datei\":\"../x\"}",
                          "{\"auftrag_id\":\"j1\",\"datei\":\"a/b.png\"}",
                          "{\"auftrag_id\":\"j1\",\"max_bytes\":0}",
                          "{\"auftrag_id\":\"j1\",\"inhalt\":1}"}) {
    r = f.Call("ablage_lesen", bad, kLan, Fixture::Bearer(tim));
    CHECK_EQ(RpcCode(r), mcp::kInvalidParams);
  }
  // The preset's limit caps a call's max_bytes.
  {
    Fixture::Config config;
    config.inline_limit = 10;
    Fixture g(config);
    g.fake = f.fake;
    r = g.Call("ablage_lesen",
               "{\"auftrag_id\":\"j1\",\"inhalt\":true,\"max_bytes\":1000000}");
    CheckShape(r, false);
    CHECK(!Structured(r).find("eingebettet")->as_bool());
    CHECK_EQ(Blocks(r)[1].member_str("type"), std::string("resource_link"));
    CHECK_EQ(g.Requests("/v1/jobs/j1/files/image-1.png").size(), std::size_t(0));
  }
  // The job tools need the job API.
  {
    Fixture::Config config;
    config.jobs_api = false;
    Fixture nojobs(config);
    r = nojobs.Call("ablage_lesen", "{\"auftrag_id\":\"j1\"}");
    CHECK_EQ(RpcCode(r), mcp::kInvalidParams);
  }
}

}  // namespace

int main() {
  TestClassifyAndPreset();
  TestFrame();
  TestInitialize();
  TestGate();
  TestVisibility();
  TestModelle();
  TestChat();
  TestJobs();
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "router_mcp_test: ok\n";
  return 0;
}
