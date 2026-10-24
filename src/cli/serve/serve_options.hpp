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
/// --max-request-bytes, --api-key, --log-level and -v. `sessions_or_null`
/// omits --sessions.
void AddServerOptions(ArgParser& parser, std::string* host, int* port,
                      std::size_t* sessions_or_null,
                      std::size_t* max_connections,
                      std::size_t* max_request_bytes, std::string* api_key,
                      ServerLogOptions* log);

/// Checks one preset section: "" when valid, otherwise the message
/// `gufo serve <modality> argv...` would print. The gufo binary installs the
/// real check (serve.cpp parses without loading); until one is installed every
/// section is accepted, which only unit tests without serve.cpp observe.
using ServeArgvValidator = std::string (*)(std::string_view modality,
                                           const std::vector<std::string>& argv);
void SetServeArgvValidator(ServeArgvValidator validator);
[[nodiscard]] std::string ValidateServeArgv(
    std::string_view modality, const std::vector<std::string>& argv);

}  // namespace gufo::cli

#endif  // GUFO_CLI_SERVE_SERVE_OPTIONS_HPP_
