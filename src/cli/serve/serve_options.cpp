#include "src/cli/serve/serve_options.hpp"

#include <atomic>

namespace gufo::cli {
namespace {

std::atomic<ServeArgvValidator> validator{nullptr};

constexpr std::string_view kLogLevelHelp =
    "Log verbosity: error, warn, info or debug (default: info)";
constexpr std::string_view kVerboseHelp = "Shorthand for --log-level=debug";

}  // namespace

void AddServerOptions(ArgParser& parser, std::string* host, int* port,
                      std::size_t* session_count, std::size_t* max_connections,
                      std::size_t* max_request_body_bytes,
                      ServerLogOptions* log) {
  parser.AddOption("-i", "--host", "IP", "Bind address", "Server", host);
  parser.AddOption("-p", "--port", "N", "Port to listen on", "Server", port);
  if (session_count != nullptr)
    parser.AddOption("-j", "--sessions", "N",
                     "Preallocated GPU request sessions", "Server",
                     session_count);
  parser.AddOption("", "--max-connections", "N",
                   "Maximum simultaneous HTTP connections", "Server",
                   max_connections);
  parser.AddOption("", "--max-request-bytes", "N",
                   "Maximum HTTP request body bytes", "Server",
                   max_request_body_bytes);
  parser.AddCustomOption(
      "", "--log-level", "LEVEL", kLogLevelHelp, "Logging",
      [log](std::string_view flag, std::string_view value, std::string* error) {
        const auto level = server::LogLevelFromName(value);
        if (!level.has_value()) {
          *error = std::string(flag) + " must be error, warn, info or debug";
          return false;
        }
        log->level = *level;
        return true;
      });
  parser.AddFlag("-v", "--verbose", kVerboseHelp, "Logging", &log->verbose);
}

void SetServeArgvValidator(ServeArgvValidator check) {
  validator.store(check);
}

std::string ValidateServeArgv(std::string_view modality,
                              const std::vector<std::string>& argv) {
  const ServeArgvValidator check = validator.load();
  return check == nullptr ? std::string{} : check(modality, argv);
}

}  // namespace gufo::cli
