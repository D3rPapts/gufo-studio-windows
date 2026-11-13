#ifndef GUFO_CLI_SERVE_ROUTER_MCP_HPP_
#define GUFO_CLI_SERVE_ROUTER_MCP_HPP_

// MCP server endpoint of the router front (gufo-studio-windows 2.1, plan
// MCP-PLAN-2026-11-13): `POST /mcp` speaks the Model Context Protocol over
// its Streamable HTTP transport so Claude Code, the Regie and other agents
// use the router's abilities as tools. Stateless: one JSON-RPC 2.0 message
// per POST, answered as application/json; no sessions (Mcp-Session-Id is
// never issued and ignored), no server-initiated stream (GET and DELETE
// are 405). Legacy handshake (protocol versions 2025-03-26, 2025-06-18 and
// 2025-11-25): `initialize`, `notifications/initialized` (202), `ping`,
// `tools/list`, `tools/call`.
//
// The gate runs first (Route::Kind::kMcp): any bearer token (admin, nutzer,
// dienst), loopback as admin and auth = off may call it; a cookie session
// is 403 mcp_token_required, the partner router 404. An Origin header must
// pass the gate's CSRF rule for every identity (403 csrf_rejected).
// `tools/list` shows a user only the tools of its areas; `tools/call` of a
// tool it may not see is -32602 like an unknown name.
//
// Commit 1 (this file): the frame, the handshake, the tool table with its
// schemas and a `tools/call` skeleton that answers every known tool with
// isError (not implemented yet). Commit 2/3 add the builders that run each
// tool as one inner request through gate and dispatch (Options::serve).

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/auth.hpp"
#include "src/cli/serve/router/preset.hpp"
#include "src/core/json.hpp"

namespace gufo::router::mcp {

/// Protocol versions answered by `initialize` (the client's when it is one
/// of these, else kDefaultProtocolVersion) and accepted in the
/// MCP-Protocol-Version header (a missing header counts as the oldest).
inline constexpr std::string_view kProtocolVersions[] = {
    "2025-03-26", "2025-06-18", "2025-11-25"};
inline constexpr std::string_view kDefaultProtocolVersion = "2025-06-18";
inline constexpr const char* kServerName = "gufo-router";

/// JSON-RPC 2.0 error codes the endpoint uses.
inline constexpr int kParseError = -32700;
inline constexpr int kInvalidRequest = -32600;
inline constexpr int kMethodNotFound = -32601;
inline constexpr int kInvalidParams = -32602;
inline constexpr int kInternalError = -32603;

[[nodiscard]] bool IsProtocolVersion(std::string_view version);

/// One entry of the tool table.
struct Tool {
  std::string name;
  std::string title;
  /// The area a user needs (Rights::AllowsArea; "" = every signed-in
  /// user, "bild|video" = either).
  std::string area;
  std::string description;
  json::Value input_schema;
  json::Value output_schema;
  /// Needs the job API (--jobs-dir): left out when the router has none.
  bool needs_jobs{false};
};

struct Options {
  /// One request through the gate and the router, in process:
  /// gate->Serve(inner, Dispatch). No sockets. Used by the tool builders
  /// (Commit 2/3).
  std::function<server::HttpResponse(const server::HttpRequest&)> serve;
  /// Identify, Log, CsrfCheck. May be null in tests (then everyone is
  /// admin).
  auth::Gate* gate{nullptr};
  /// The preset's models: defaults and the model lists in the schemas.
  std::vector<PresetModel> models;
  /// "http://192.168.178.198:8081": the address file links carry.
  std::string self_name;
  /// serverInfo.version; "" = this build's release version.
  std::string version;
  /// ablage_lesen: a file is embedded inline up to this size.
  std::size_t inline_limit{static_cast<std::size_t>(4) << 20};
  /// `[router] mcp = on|off`: off answers 404.
  bool enabled{true};
  /// The router runs the job API (--jobs-dir); without it the job tools
  /// are not listed.
  bool jobs_api{true};
};

class McpService {
public:
  explicit McpService(Options options);

  /// The answer for /mcp (any method), nullopt for every other path.
  [[nodiscard]] std::optional<server::HttpResponse> Handle(
      const server::HttpRequest& req);

  /// Test hook: the `tools/list` entries `who` may see.
  [[nodiscard]] json::Value ToolsFor(const auth::Identity& who) const;
  /// Test hook: the tool table (every tool, before the area filter).
  [[nodiscard]] const std::vector<Tool>& tools() const { return tools_; }
  /// Test hook: a `tools/call` without the HTTP frame: the JSON-RPC
  /// `result`, or an `error` object (member "code") for an unknown or
  /// invisible tool and bad arguments. `outer` is the /mcp request (its
  /// headers and address go into the inner request).
  [[nodiscard]] json::Value Call(std::string_view name, const json::Value& args,
                                 const auth::Identity& who,
                                 const server::HttpRequest& outer);

private:
  [[nodiscard]] server::HttpResponse Rpc(const server::HttpRequest& req,
                                         const auth::Identity& who);
  [[nodiscard]] json::Value Initialize(const json::Value& params) const;
  [[nodiscard]] const Tool* FindTool(std::string_view name,
                                     const auth::Identity& who) const;
  [[nodiscard]] bool Visible(const Tool& tool,
                             const auth::Identity& who) const;

  Options options_;
  std::vector<Tool> tools_;
};

}  // namespace gufo::router::mcp

#endif  // GUFO_CLI_SERVE_ROUTER_MCP_HPP_
