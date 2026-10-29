#include "src/cli/serve/router/preset.hpp"

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <set>
#include <utility>

#include "src/cli/serve/serve_options.hpp"

namespace gufo::router {
namespace {

constexpr std::string_view kModalities[] = {"llm", "video", "image", "tts",
                                            "asr", "embedding", "omni",
                                            "external"};

bool IsKnownModality(std::string_view modality) {
  for (const auto candidate : kModalities) {
    if (candidate == modality) {
      return true;
    }
  }
  return false;
}

// Keys whose values name files or directories, so a relative value can be
// resolved against the models directory. Derived from the per-modality
// registrations: `model` everywhere, `<x>-model` sidecars, the llm vision
// sidecar and disk cache, and the video job roots.
bool IsPathKey(std::string_view key) {
  if (key == "model" || key == "root" || key == "manifest" || key == "mmproj" ||
      key == "cache-disk") {
    return true;
  }
  return key.size() > 6 && key.ends_with("-model");
}

std::string_view TrimSpaces(std::string_view text) {
  const auto begin = text.find_first_not_of(' ');
  if (begin == std::string_view::npos) {
    return {};
  }
  const auto end = text.find_last_not_of(' ');
  return text.substr(begin, end - begin + 1);
}

std::string TrimLeadingSpaces(std::string_view text) {
  const auto begin = text.find_first_not_of(' ');
  if (begin == std::string_view::npos) {
    return {};
  }
  return std::string(text.substr(begin));
}

struct Section {
  std::string modality;
  std::string id;
  std::vector<std::pair<std::string, std::string>> entries;
  bool has_served_name = false;
};

bool SectionError(const Section& section, const std::string& message,
                  std::string* error) {
  *error = "[" + section.modality + "/" + section.id + "] " + message;
  return false;
}

std::string ResolvePathValue(std::string_view value,
                             const std::filesystem::path& models_dir,
                             std::string_view key) {
  if (models_dir.empty() || value.empty() || !IsPathKey(key)) {
    return std::string(value);
  }
  const std::filesystem::path candidate(value);
  if (candidate.is_absolute()) {
    return std::string(value);
  }
  return (models_dir / candidate).string();
}

bool ParseNumber(std::string_view text, double* value) {
  const std::string copy(TrimSpaces(text));
  if (copy.empty()) {
    return false;
  }
  char* end = nullptr;
  *value = std::strtod(copy.c_str(), &end);
  return end == copy.c_str() + copy.size() && *value >= 0.0;
}

/// Router-only `router-*` keys; they configure residency, not the worker.
bool ParseHint(const std::string& key, std::string_view value,
               PresetHints* hints, std::string* error) {
  double number = 0.0;
  if (key == "router-gpu-gb" || key == "router-host-gb" ||
      key == "router-idle-seconds") {
    if (!ParseNumber(value, &number)) {
      *error = "'" + key + "' needs a non-negative number";
      return false;
    }
    if (key == "router-gpu-gb") {
      hints->gpu_gb = number;
    } else if (key == "router-host-gb") {
      hints->host_gb = number;
    } else {
      hints->idle_seconds = static_cast<long long>(number);
    }
    return true;
  }
  if (key == "router-exclusive") {
    if (value != "true" && value != "false") {
      *error = "'router-exclusive' takes true|false";
      return false;
    }
    hints->exclusive = value == "true" ? 1 : 0;
    return true;
  }
  *error = "unknown router key '" + key + "'";
  return false;
}

}  // namespace

bool IsReservedKey(std::string_view key) {
  return key == "host" || key == "port" || key == "api-key";
}

bool LoadPreset(const std::filesystem::path& file,
                const std::filesystem::path& models_dir,
                std::vector<PresetModel>* models, std::string* error) {
  std::ifstream input(file, std::ios::binary);
  if (!input) {
    *error = "cannot open preset file '" + file.string() + "'";
    return false;
  }
  models->clear();

  const auto flush = [&](const Section& section) -> bool {
    if (section.entries.empty()) {
      return SectionError(section, "empty section body", error);
    }
    const bool external = section.modality == "external";
    std::vector<std::string> argv;
    std::string command;
    PresetHints hints;
    for (const auto& [key, value] : section.entries) {
      if (key.starts_with("router-")) {
        std::string hint_error;
        if (!ParseHint(key, value, &hints, &hint_error)) {
          return SectionError(section, hint_error, error);
        }
        continue;
      }
      if (IsReservedKey(key)) {
        return SectionError(
            section, "reserved key '" + key + "' is managed by the router",
            error);
      }
      if (external && key == "command") {
        command = ResolvePathValue(value, models_dir, "model");
        continue;
      }
      if (external && key == "args") {
        // Leading positional arguments, e.g. "serve llm" to run another gufo
        // build. Space-separated; no quoting.
        std::vector<std::string> leading;
        std::size_t begin = 0;
        while ((begin = value.find_first_not_of(' ', begin)) !=
               std::string::npos) {
          const auto end = value.find(' ', begin);
          leading.emplace_back(value.substr(begin, end - begin));
          begin = end;
        }
        argv.insert(argv.begin(), leading.begin(), leading.end());
        continue;
      }
      if (value == "true") {
        argv.push_back("--" + key);
      } else if (value == "false") {
        continue;
      } else {
        argv.push_back("--" + key);
        argv.push_back(ResolvePathValue(value, models_dir, key));
      }
    }
    if (external) {
      // Another server's options are its own business; only the program
      // must exist.
      if (command.empty()) {
        return SectionError(section, "external sections need 'command'",
                            error);
      }
      // Non-throwing: an app execution alias or a denied path is an error
      // message, not a crash.
      std::error_code exists_error;
      if (!std::filesystem::exists(command, exists_error)) {
        return SectionError(section, "command '" + command + "' not found",
                            error);
      }
      models->push_back(
          {section.modality, section.id, std::move(argv), std::move(command)});
      models->back().hints = hints;
      return true;
    }
    // Video workers have no `--served-model-name`; the routing id still
    // reaches every other modality as the public model name default.
    if (!section.has_served_name && section.modality != "video") {
      argv.push_back("--served-model-name");
      argv.push_back(section.id);
    }
    const std::string validation =
        cli::ValidateServeArgv(section.modality, argv);
    if (!validation.empty()) {
      return SectionError(section, validation, error);
    }
    models->push_back({section.modality, section.id, std::move(argv), {}});
    models->back().hints = hints;
    return true;
  };

  std::set<std::string> ids;
  Section section;
  bool in_section = false;
  std::string raw;
  bool first_line = true;
  while (std::getline(input, raw)) {
    if (!raw.empty() && raw.back() == '\r') {
      raw.pop_back();
    }
    if (first_line) {
      first_line = false;
      if (raw.rfind("\xEF\xBB\xBF", 0) == 0) {
        raw.erase(0, 3);
      }
    }
    const std::string_view line = TrimSpaces(raw);
    if (line.empty() || line.front() == '#') {
      continue;
    }
    if (line.front() == '[') {
      if (in_section && !flush(section)) {
        return false;
      }
      section = Section();
      const auto close = line.find(']');
      const std::string_view header = close == std::string_view::npos
                                          ? line.substr(1)
                                          : line.substr(1, close - 1);
      const auto slash = header.find('/');
      if (close == std::string_view::npos || slash == std::string_view::npos ||
          !TrimSpaces(line.substr(close + 1)).empty()) {
        *error = "invalid section header '" + std::string(line) +
                 "': expected [modality/id]";
        return false;
      }
      section.modality = std::string(TrimSpaces(header.substr(0, slash)));
      section.id = std::string(TrimSpaces(header.substr(slash + 1)));
      in_section = true;
      if (!IsKnownModality(section.modality)) {
        return SectionError(
            section, "unknown modality '" + section.modality + "'", error);
      }
      if (section.id.empty()) {
        return SectionError(section, "empty model id", error);
      }
      if (!ids.insert(section.id).second) {
        return SectionError(section, "duplicate model id '" + section.id + "'",
                            error);
      }
      continue;
    }
    if (!in_section) {
      *error =
          "key/value line outside any section: '" + std::string(line) + "'";
      return false;
    }
    const auto equals = line.find('=');
    if (equals == std::string_view::npos) {
      return SectionError(
          section, "expected 'key = value', got '" + std::string(line) + "'",
          error);
    }
    const std::string key(TrimSpaces(line.substr(0, equals)));
    if (key.empty()) {
      return SectionError(
          section, "empty key in line '" + std::string(line) + "'", error);
    }
    if (key == "served-model-name") {
      section.has_served_name = true;
    }
    section.entries.emplace_back(key,
                                 TrimLeadingSpaces(line.substr(equals + 1)));
  }
  if (in_section && !flush(section)) {
    return false;
  }
  return true;
}

}  // namespace gufo::router
