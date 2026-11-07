#include "src/cli/serve/serve_options.hpp"

#include <atomic>

namespace gufo::cli {
namespace {

std::atomic<ServeArgvValidator> validator{nullptr};

}  // namespace

void SetServeArgvValidator(ServeArgvValidator check) {
  validator.store(check);
}

std::string ValidateServeArgv(std::string_view modality,
                              const std::vector<std::string>& argv) {
  const ServeArgvValidator check = validator.load();
  return check == nullptr ? std::string{} : check(modality, argv);
}

}  // namespace gufo::cli
