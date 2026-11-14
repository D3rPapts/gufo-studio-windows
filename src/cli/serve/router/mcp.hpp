#ifndef GUFO_CLI_SERVE_ROUTER_MCP_HPP_
#define GUFO_CLI_SERVE_ROUTER_MCP_HPP_

// MCP server endpoint of the router front (gufo-studio-windows 2.1, plan
// MCP-PLAN-2026-11-13): `POST /mcp` speaks the Model Context Protocol over
// its Streamable HTTP transport so Claude Code, the Regie and other agents
// use the router's abilities as tools. Stateless: one JSON-RPC 2.0 message
// per POST, answered as application/json; no sessions (Mcp-Session-Id is
// never issued and ignored), no server-initiated stream (GET and DELETE
// are 405). Dual-era (spec 2026-07-28, Versioning -> "Backward
// Compatibility with Initialization-Based Versions"): the legacy handshake
// (protocol versions 2025-03-26, 2025-06-18 and 2025-11-25: `initialize`,
// `notifications/initialized` (202), `ping`, `tools/list`, `tools/call`)
// and the modern era 2026-07-28 (no handshake, no sessions: every request
// carries its version in `params._meta["io.modelcontextprotocol/
// protocolVersion"]` and the `MCP-Protocol-Version` header, the server
// answers `server/discover`; Commit 4, see Rpc in mcp.cpp).
//
// The gate runs first (Route::Kind::kMcp): any bearer token (admin, nutzer,
// dienst), loopback as admin and auth = off may call it; a cookie session
// is 403 mcp_token_required, the partner router 404. An Origin header must
// pass the gate's CSRF rule for every identity (403 csrf_rejected).
// `tools/list` shows a user only the tools of its areas; `tools/call` of a
// tool it may not see is -32602 like an unknown name.
//
// Commit 1: the frame, the handshake, the tool table with its schemas and
// the `tools/call` skeleton. Commit 2: the builders of `modelle`, `chat`,
// `auftrag_status`, `auftrag_liste`, `auftrag_abbrechen` and
// `ablage_lesen`. Every tool call is one inner request (a fixed path list:
// /v1/models, /v1/chat/completions, /v1/jobs...) through the same gate and
// dispatch as an HTTP client's (Options::serve), under the caller's own
// token: area, model list, word filter, daily limit (counted once, by the
// inner request), own jobs, X-Gufo-User and the access log hold unchanged.
// A refusal or router error (status >= 400) is a tool result with
// `isError: true` and `structuredContent {http_status, code, message,
// retry_after?}`; every result carries `structuredContent` and its JSON as
// `content[0]` text. The access log gets `mcp:<tool>:ok|abgelehnt:<code>`
// beside the inner request's own line.
//
// Commit 3: the media job tools `bild_erzeugen`, `bild_bearbeiten`,
// `video_erzeugen`, `stimme`, `musik`, `objekt_3d` and `hochskalieren`.
// Each is one `POST /v1/jobs` around the endpoint the matching page uses
// (pages/bild|video|stimme|musik|objekt|hochskalieren, with the field
// names those pages send), `vorrang` as X-Gufo-Priority, `metadata
// {quelle: "mcp", seite, titel, ...}` for the Ablage, answering
// `{auftrag_id, status, url, modell, position?, geschaetzt?}` plus the text
// "Auftrag <id> angenommen ...". Input files are Ablage paths
// (`/v1/jobs/<id>/files/<name>`, read through the gate: own jobs only) or
// `data:` URLs; `bild_bearbeiten` builds the multipart body of
// `/v1/images/edits` (comfy::BuildMultipart) and submits it as
// `request.body_base64` + `content_type`.

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

/// The legacy protocol versions (`initialize` handshake; spec 2025-03-26
/// .. 2025-11-25): `initialize` answers the client's when it is one of
/// these, else kDefaultProtocolVersion (Lifecycle -> Version Negotiation).
inline constexpr std::string_view kLegacyProtocolVersions[] = {
    "2025-03-26", "2025-06-18", "2025-11-25"};
inline constexpr std::string_view kDefaultProtocolVersion = "2025-06-18";
/// The modern era (spec 2026-07-28, Versioning -> Terminology: "protocol
/// versions that convey version, identity, and capabilities as per-request
/// metadata"): no handshake, the version in every request's `_meta`.
inline constexpr std::string_view kModernProtocolVersion = "2026-07-28";
/// Every version the endpoint speaks, modern first: what the
/// MCP-Protocol-Version header and `_meta` may carry and the `supported`
/// list of the -32022 error (a missing header counts as the oldest legacy
/// version, as 2026-07-28 Streamable HTTP -> Protocol Version Header
/// allows for servers that support clients before 2025-06-18).
inline constexpr std::string_view kProtocolVersions[] = {
    "2026-07-28", "2025-11-25", "2025-06-18", "2025-03-26"};
inline constexpr const char* kServerName = "gufo-router";

/// The reserved `_meta` keys of 2026-07-28 (Base Protocol -> `_meta` ->
/// "Per-request protocol fields" and "Per-response protocol fields").
inline constexpr const char* kMetaProtocolVersion =
    "io.modelcontextprotocol/protocolVersion";
inline constexpr const char* kMetaClientInfo =
    "io.modelcontextprotocol/clientInfo";
inline constexpr const char* kMetaServerInfo =
    "io.modelcontextprotocol/serverInfo";

/// JSON-RPC 2.0 error codes the endpoint uses; -32020 and -32022 are the
/// MCP-defined ones of 2026-07-28 (Base Protocol -> Error Codes:
/// `HeaderMismatch`, `UnsupportedProtocolVersion`).
inline constexpr int kParseError = -32700;
inline constexpr int kInvalidRequest = -32600;
inline constexpr int kMethodNotFound = -32601;
inline constexpr int kInvalidParams = -32602;
inline constexpr int kInternalError = -32603;
inline constexpr int kHeaderMismatch = -32020;
inline constexpr int kUnsupportedProtocolVersion = -32022;

/// `version` is one of kProtocolVersions / kLegacyProtocolVersions /
/// kModernProtocolVersion.
[[nodiscard]] bool IsProtocolVersion(std::string_view version);
[[nodiscard]] bool IsLegacyProtocolVersion(std::string_view version);
[[nodiscard]] bool IsModernProtocolVersion(std::string_view version);

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
  /// ablage_lesen: an image or audio file is embedded inline up to this
  /// size (`[router] mcp-inline-mb`, default 4); a call's max_bytes may
  /// lower it, never raise it.
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

  /// The collected answer of an inner request (a streamed body is read to
  /// the end, as the job runner does).
  struct InnerAnswer {
    int status{0};
    std::string content_type;
    std::string body;
    /// Retry-After header in seconds, 0 without one.
    long long retry_after{0};
  };

  /// An input file of a media tool: the bytes of an Ablage path (read
  /// through the gate) or a data: URL. `failure` is set when the gate or
  /// the job store refused the path (the tool answers that failure).
  struct Media {
    std::string bytes;
    std::string content_type;
    std::string name;
    std::optional<json::Value> failure;
  };

private:
  [[nodiscard]] server::HttpResponse Rpc(const server::HttpRequest& req,
                                         const auth::Identity& who);
  [[nodiscard]] json::Value Initialize(const json::Value& params) const;
  /// The modern era's `server/discover` result (2026-07-28, Server ->
  /// Discovery: DiscoverResult) for `who`.
  [[nodiscard]] json::Value Discover(const auth::Identity& who) const;
  /// `{name, title, version}` of this server (serverInfo of `initialize`
  /// and `_meta[kMetaServerInfo]` of every modern result).
  [[nodiscard]] json::Value ServerInfo() const;
  [[nodiscard]] const Tool* FindTool(std::string_view name,
                                     const auth::Identity& who) const;
  [[nodiscard]] bool Visible(const Tool& tool,
                             const auth::Identity& who) const;
  /// One inner request through Options::serve: `outer`'s token, address
  /// and cancellation, a JSON body when `body` is set, `extra` headers
  /// (an X-Gufo-Priority there replaces the outer one). `path` must be
  /// one of the tool paths (never /mcp itself).
  [[nodiscard]] InnerAnswer Inner(
      const server::HttpRequest& outer, std::string method, std::string path,
      std::string query = {}, std::string body = {},
      std::vector<std::pair<std::string, std::string>> extra = {}) const;
  /// The documented default of a tool's `model` field.
  [[nodiscard]] std::string DefaultModel(std::string_view tool) const;

  [[nodiscard]] Media ReadMedia(const server::HttpRequest& outer,
                                const std::string& value,
                                const std::string& field) const;
  /// `POST /v1/jobs` around `path` with a JSON `body` or a raw body of
  /// `content_type` (base64 in the submit), `vorrang` as X-Gufo-Priority,
  /// `metadata` (quelle mcp is added): the uniform job tool result, whose
  /// `modell` is the store's `request.model` or `model`.
  [[nodiscard]] json::Value SubmitJob(const server::HttpRequest& outer,
                                      const std::string& path,
                                      const json::Value& body,
                                      const std::string& raw_body,
                                      const std::string& content_type,
                                      const std::string& model,
                                      std::optional<long long> vorrang,
                                      json::Value metadata) const;

  // The builders (Commit 2). Each returns a tool result (`content`,
  // `structuredContent`, maybe `isError`); bad arguments throw
  // InvalidParams (mcp.cpp) and become -32602.
  [[nodiscard]] json::Value RunModelle(const json::Value& args,
                                       const auth::Identity& who,
                                       const server::HttpRequest& outer) const;
  [[nodiscard]] json::Value RunChat(const json::Value& args,
                                    const server::HttpRequest& outer) const;
  [[nodiscard]] json::Value RunAuftragStatus(
      const json::Value& args, const server::HttpRequest& outer) const;
  [[nodiscard]] json::Value RunAuftragListe(
      const json::Value& args, const server::HttpRequest& outer) const;
  [[nodiscard]] json::Value RunAuftragAbbrechen(
      const json::Value& args, const server::HttpRequest& outer) const;
  [[nodiscard]] json::Value RunAblageLesen(
      const json::Value& args, const server::HttpRequest& outer) const;

  // The media job builders (Commit 3).
  [[nodiscard]] json::Value RunBildErzeugen(
      const json::Value& args, const server::HttpRequest& outer) const;
  [[nodiscard]] json::Value RunBildBearbeiten(
      const json::Value& args, const server::HttpRequest& outer) const;
  [[nodiscard]] json::Value RunVideoErzeugen(
      const json::Value& args, const server::HttpRequest& outer) const;
  [[nodiscard]] json::Value RunStimme(const json::Value& args,
                                      const server::HttpRequest& outer) const;
  [[nodiscard]] json::Value RunMusik(const json::Value& args,
                                     const server::HttpRequest& outer) const;
  [[nodiscard]] json::Value RunObjekt3d(const json::Value& args,
                                        const server::HttpRequest& outer) const;
  [[nodiscard]] json::Value RunHochskalieren(
      const json::Value& args, const server::HttpRequest& outer) const;

  Options options_;
  std::vector<Tool> tools_;
};

}  // namespace gufo::router::mcp

#endif  // GUFO_CLI_SERVE_ROUTER_MCP_HPP_
