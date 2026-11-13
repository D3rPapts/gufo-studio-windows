#ifndef GUFO_CLI_SERVE_SERVE_OPTIONS_HPP_
#define GUFO_CLI_SERVE_SERVE_OPTIONS_HPP_

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/serve/logging.hpp"

/// The part of `gufo serve`'s option handling the router shares: the front
/// server options and validation of a preset section against a modality.
namespace gufo::cli {

/// `--log-level` and its `-v/--verbose` shorthand share one sink. `level`
/// holds the canonical spelling when given and stays empty otherwise; the two
/// spellings together are a conflict, reported where the level is armed.
struct ServerLogOptions {
  bool verbose = false;
  std::optional<server::LogLevel> level;
};

/// Registers --host, --port, [--sessions], --max-connections,
/// --max-request-bytes, --log-level and -v. `sessions_or_null` omits
/// --sessions. Upstream keeps its own copy private to serve.cpp; this one
/// serves `gufo router` with the same spelling and help text. There is no
/// --api-key: upstream dropped API-key auth (#967) and the router follows.
void AddServerOptions(ArgParser& parser, std::string* host, int* port,
                      std::size_t* sessions_or_null,
                      std::size_t* max_connections,
                      std::size_t* max_request_bytes, ServerLogOptions* log);

/// Checks one preset section: "" when valid, otherwise the message
/// `gufo serve <modality> argv...` would print. On release/1.4 serve.cpp
/// installed a parse-only check; upstream v0.31.0 serve.cpp has none, so
/// (WAVE-A stub) no validator is installed and every section is accepted
/// here. A bad option then surfaces when the worker starts.
using ServeArgvValidator = std::string (*)(std::string_view modality,
                                           const std::vector<std::string>& argv);
void SetServeArgvValidator(ServeArgvValidator validator);
[[nodiscard]] std::string ValidateServeArgv(
    std::string_view modality, const std::vector<std::string>& argv);

}  // namespace gufo::cli

#endif  // GUFO_CLI_SERVE_SERVE_OPTIONS_HPP_
