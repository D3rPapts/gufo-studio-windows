#ifndef GUFO_CLI_SERVE_ROUTER_STRINGS_HPP_
#define GUFO_CLI_SERVE_ROUTER_STRINGS_HPP_

// Small text helpers the router sources share (jobs, auth, the front):
// ASCII case folding, trimming, splitting and the "is this a JSON body"
// test. Header-only.

#include <cctype>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/http_server.hpp"

namespace gufo::router {

/// ASCII lower case (bytes >= 0x80 unchanged).
[[nodiscard]] inline std::string Lower(std::string_view text) {
  std::string out(text);
  for (char& c : out)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

/// ASCII case-insensitive equality (header names, fixed tokens).
[[nodiscard]] inline bool EqualsIgnoreCase(std::string_view a,
                                           std::string_view b) {
  if (a.size() != b.size())
    return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::tolower(static_cast<unsigned char>(a[i])) !=
        std::tolower(static_cast<unsigned char>(b[i])))
      return false;
  return true;
}

/// Without leading and trailing ASCII whitespace.
[[nodiscard]] inline std::string_view Trim(std::string_view text) {
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.front())))
    text.remove_prefix(1);
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
    text.remove_suffix(1);
  return text;
}

/// Trimmed, non-empty parts between `sep`.
[[nodiscard]] inline std::vector<std::string> Split(std::string_view text,
                                                    char sep) {
  std::vector<std::string> out;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const auto end = std::min(text.find(sep, begin), text.size());
    const std::string_view part = Trim(text.substr(begin, end - begin));
    if (!part.empty())
      out.emplace_back(part);
    begin = end + 1;
  }
  return out;
}

[[nodiscard]] inline std::string Join(const std::vector<std::string>& parts,
                                      char sep) {
  std::string out;
  for (const std::string& part : parts) {
    if (!out.empty())
      out += sep;
    out += part;
  }
  return out;
}

/// A POST with a JSON object body (Content-Type application/json, or no
/// type and a body that starts with '{').
[[nodiscard]] inline bool IsJsonBody(const server::HttpRequest& req) {
  if (req.method != "POST" || req.body.empty())
    return false;
  const std::string type = Lower(req.header("Content-Type"));
  if (type.starts_with("application/json"))
    return true;
  return type.empty() && req.body.front() == '{';
}

}  // namespace gufo::router

#endif  // GUFO_CLI_SERVE_ROUTER_STRINGS_HPP_
