#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/router/preset.hpp"
#include "src/cli/serve/serve_options.hpp"

using gufo::router::IsReservedKey;
using gufo::router::LoadPreset;
using gufo::router::ParseRemoteUrl;
using gufo::router::PresetModel;
using gufo::router::RemoteEndpoint;

namespace {

std::filesystem::path WriteFixture(const std::string& name,
                                   const std::string& contents) {
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << contents;
  out.close();
  return path;
}

bool SameModels(const std::vector<PresetModel>& lhs,
                const std::vector<PresetModel>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].modality != rhs[i].modality || lhs[i].id != rhs[i].id ||
        lhs[i].argv != rhs[i].argv) {
      return false;
    }
  }
  return true;
}

constexpr const char* kHappyBody =
    "# comment\n"
    "\n"
    "[llm/a]\n"
    "model = rel/x.gguf\n"
    "sessions = 2\n"
    "log-progress = true\n"
    "vite = false\n"
    "\n"
    "[asr/b]\n"
    "model = rel/asr\n";

}  // namespace

void TestHappyPath() {
  const auto file = WriteFixture("gufo_router_preset_happy.conf", kHappyBody);
  std::vector<PresetModel> models;
  std::string error;
  assert(LoadPreset(file, "/models", &models, &error));
  assert(error.empty());
  assert(models.size() == 2);
  // Relative paths resolve as models_dir / value with the native separator
  // ("/models\\rel/x.gguf" on Windows), exactly as the worker receives them.
  const auto resolved = [](const char* value) {
    return (std::filesystem::path("/models") / value).string();
  };
  const std::vector<std::string> expected_a = {
      "--model",        resolved("rel/x.gguf"), "--sessions", "2",
      "--log-progress", "--served-model-name",  "a"};
  assert(models[0].modality == "llm");
  assert(models[0].id == "a");
  assert(models[0].argv == expected_a);
  const std::vector<std::string> expected_b = {"--model", resolved("rel/asr"),
                                               "--served-model-name", "b"};
  assert(models[1].modality == "asr");
  assert(models[1].id == "b");
  assert(models[1].argv == expected_b);
}

void TestErrorMentionsSectionAndToken(const char* name, const char* body,
                                      const std::string& section_token,
                                      const std::string& offending_token) {
  const auto file = WriteFixture(name, body);
  std::vector<PresetModel> models;
  std::string error;
  assert(!LoadPreset(file, "/models", &models, &error));
  assert(error.find(section_token) != std::string::npos);
  assert(error.find(offending_token) != std::string::npos);
}

void TestErrors() {
  TestErrorMentionsSectionAndToken("gufo_router_preset_bad_modality.conf",
                                   "[sr/x]\nmodel = m\n", "sr/x", "sr");
  TestErrorMentionsSectionAndToken("gufo_router_preset_dup_id.conf",
                                   "[llm/a]\nmodel = m\n[asr/a]\nmodel = m\n",
                                   "asr/a", "duplicate");
  TestErrorMentionsSectionAndToken("gufo_router_preset_reserved.conf",
                                   "[llm/a]\nport = 1\n", "llm/a", "port");
  TestErrorMentionsSectionAndToken("gufo_router_preset_unknown_key.conf",
                                   "[llm/a]\nmodel = m\nbogus = 1\n", "llm/a",
                                   "bogus");
  TestErrorMentionsSectionAndToken("gufo_router_preset_missing_model.conf",
                                   "[llm/a]\nsessions = 2\n", "llm/a",
                                   "--model");
  TestErrorMentionsSectionAndToken("gufo_router_preset_empty_body.conf",
                                   "[llm/a]\n[llm/b]\nmodel = m\n", "llm/a",
                                   "empty");
  TestErrorMentionsSectionAndToken("gufo_router_preset_no_slash.conf",
                                   "[llmx]\nmodel = m\n", "llmx", "section");
}

void TestCrlfAndBomMatchLf() {
  const auto lf = WriteFixture("gufo_router_preset_lf.conf", kHappyBody);
  std::string crlf = "\xEF\xBB\xBF";
  for (const char* p = kHappyBody; *p != '\0'; ++p) {
    if (*p == '\n') {
      crlf.push_back('\r');
    }
    crlf.push_back(*p);
  }
  const auto crlf_file = WriteFixture("gufo_router_preset_crlf_bom.conf", crlf);
  std::vector<PresetModel> lf_models;
  std::vector<PresetModel> crlf_models;
  std::string error;
  assert(LoadPreset(lf, "/models", &lf_models, &error));
  assert(LoadPreset(crlf_file, "/models", &crlf_models, &error));
  assert(SameModels(lf_models, crlf_models));
}

void TestEmptyModelsDirPassesValuesThrough() {
  const auto file = WriteFixture("gufo_router_preset_happy.conf", kHappyBody);
  std::vector<PresetModel> models;
  std::string error;
  assert(LoadPreset(file, "", &models, &error));
  assert(models.size() == 2);
  assert(models[0].argv.size() >= 2);
  assert(models[0].argv[1] == "rel/x.gguf");
  assert(models[1].argv[1] == "rel/asr");
}

void TestIsReservedKey() {
  assert(IsReservedKey("host"));
  assert(IsReservedKey("port"));
  assert(IsReservedKey("api-key"));
  assert(!IsReservedKey("model"));
  assert(!IsReservedKey("sessions"));
  assert(!IsReservedKey("hostname"));
}

void SetEnv(const char* name, const char* value) {
#ifdef _WIN32
  assert(::_putenv_s(name, value) == 0);
#else
  assert(::setenv(name, value, 1) == 0);
#endif
}

void TestRemoteSection() {
  SetEnv("GUFO_PRESET_TEST_KEY", "secret-56");
  const auto file = WriteFixture(
      "gufo_router_preset_remote.conf",
      "[llm/a]\nmodel = m\n\n"
      "[remote/qwen-image-2.1]\n"
      "url = http://192.168.100.56:8081\n"
      "modality = image\n"
      "api-key-env = GUFO_PRESET_TEST_KEY\n"
      "connect-timeout-seconds = 2\n"
      "head-timeout-seconds = 30\n"
      "gpu-slot = studio56\n"
      "busy-check-url = http://192.168.100.56:8085/queue\n"
      "queue-timeout-seconds = 60\n"
      "max-request-bytes = 67108864\n"
      "\n"
      "[remote/tts]\n"
      "url = http://10.0.0.2:9000/\n"
      "modality = tts\n");
  std::vector<PresetModel> models;
  std::string error;
  assert(LoadPreset(file, "/models", &models, &error));
  assert(models.size() == 3);
  assert(!models[0].remote.has_value());
  const PresetModel& image = models[1];
  assert(image.id == "qwen-image-2.1" && image.modality == "image");
  assert(image.argv.empty() && image.command.empty());
  assert(image.remote.has_value());
  const auto& spec = *image.remote;
  assert(spec.url.host == "192.168.100.56" && spec.url.port == 8081);
  assert(spec.url.address == 0xC0A86438u && spec.url.path.empty());
  assert(spec.api_key_env == "GUFO_PRESET_TEST_KEY");
  assert(spec.api_key == "secret-56");
  assert(spec.connect_timeout_seconds == 2);
  assert(spec.head_timeout_seconds == 30);
  assert(spec.gpu_slot == "studio56");
  assert(spec.busy_check.has_value() && spec.busy_check->port == 8085 &&
         spec.busy_check->path == "/queue");
  assert(spec.queue_timeout_seconds == 60);
  assert(spec.max_request_bytes == 67108864u);
  const auto& tts = *models[2].remote;
  // Defaults; a bare trailing slash is no path.
  assert(tts.url.path.empty() && tts.api_key.empty());
  assert(tts.connect_timeout_seconds == 3 && tts.head_timeout_seconds == 0);
  assert(tts.gpu_slot.empty() && !tts.busy_check.has_value());
  assert(tts.queue_timeout_seconds == 900 && tts.max_request_bytes == 0);
}

void TestRemoteErrors() {
  const auto bad = [](const char* name, const std::string& keys,
                      const std::string& token) {
    TestErrorMentionsSectionAndToken(
        name, ("[remote/r]\n" + keys).c_str(), "remote/r", token);
  };
  bad("gufo_router_preset_remote_no_url.conf", "modality = image\n", "url");
  bad("gufo_router_preset_remote_no_modality.conf",
      "url = http://10.0.0.2:8081\n", "modality");
  bad("gufo_router_preset_remote_bad_modality.conf",
      "url = http://10.0.0.2:8081\nmodality = remote\n", "modality");
  bad("gufo_router_preset_remote_https.conf",
      "url = https://10.0.0.2:8081\nmodality = image\n", "http://");
  bad("gufo_router_preset_remote_host.conf",
      "url = http://studio.local:8081\nmodality = image\n", "IPv4");
  bad("gufo_router_preset_remote_no_port.conf",
      "url = http://10.0.0.2\nmodality = image\n", "port");
  bad("gufo_router_preset_remote_path.conf",
      "url = http://10.0.0.2:8081/v1\nmodality = image\n", "path");
  bad("gufo_router_preset_remote_hint.conf",
      "url = http://10.0.0.2:8081\nmodality = image\nrouter-gpu-gb = 9\n",
      "router-gpu-gb");
  bad("gufo_router_preset_remote_model.conf",
      "url = http://10.0.0.2:8081\nmodality = image\nremote-model = x\n",
      "same id");
  bad("gufo_router_preset_remote_api_key.conf",
      "url = http://10.0.0.2:8081\nmodality = image\napi-key = x\n",
      "api-key-env");
  bad("gufo_router_preset_remote_unknown.conf",
      "url = http://10.0.0.2:8081\nmodality = image\nmodel = x\n", "model");
  bad("gufo_router_preset_remote_timeout.conf",
      "url = http://10.0.0.2:8081\nmodality = image\n"
      "connect-timeout-seconds = 0\n",
      "connect-timeout-seconds");
  bad("gufo_router_preset_remote_env.conf",
      "url = http://10.0.0.2:8081\nmodality = image\n"
      "api-key-env = GUFO_PRESET_TEST_UNSET_KEY\n",
      "GUFO_PRESET_TEST_UNSET_KEY");
  // Remote ids share the id space with local sections.
  TestErrorMentionsSectionAndToken(
      "gufo_router_preset_remote_dup.conf",
      "[llm/a]\nmodel = m\n[remote/a]\nurl = http://10.0.0.2:8081\n"
      "modality = llm\n",
      "remote/a", "duplicate");
}

void TestParseRemoteUrl() {
  RemoteEndpoint endpoint;
  std::string error;
  assert(ParseRemoteUrl("http://127.0.0.1:65535", false, &endpoint, &error));
  assert(endpoint.address == 0x7F000001u && endpoint.port == 65535);
  assert(ParseRemoteUrl("http://10.1.2.3:8188/queue?x=1", true, &endpoint,
                        &error));
  assert(endpoint.path == "/queue?x=1");
  assert(ParseRemoteUrl("http://10.1.2.3:8188", true, &endpoint, &error));
  assert(endpoint.path == "/");
  for (const char* text :
       {"http://10.1.2:80", "http://10.1.2.3.4:80", "http://256.1.2.3:80",
        "http://10..2.3:80", "http://10.1.2.3:0", "http://10.1.2.3:70000",
        "http://10.1.2.3:", "http://user@10.1.2.3:80", "http://[::1]:80",
        "10.1.2.3:80", "http://10.1.2.3.:80"}) {
    assert(!ParseRemoteUrl(text, true, &endpoint, &error));
  }
}

// serve.cpp installs the real check in the gufo binary; this stand-in rejects
// the one unknown key the tests use and a missing model, with serve's wording.
std::string RejectBogus(std::string_view, const std::vector<std::string>& argv) {
  bool has_model = false;
  for (const std::string& value : argv) {
    if (value == "--bogus")
      return "unknown option: --bogus";
    has_model = has_model || value == "--model";
  }
  if (!has_model)
    return "--model <PATH> is required";
  return {};
}

int main() {
  gufo::cli::SetServeArgvValidator(&RejectBogus);
  TestHappyPath();
  TestErrors();
  TestCrlfAndBomMatchLf();
  TestEmptyModelsDirPassesValuesThrough();
  TestIsReservedKey();
  TestRemoteSection();
  TestRemoteErrors();
  TestParseRemoteUrl();
  std::cout << "All router preset tests passed.\n";
  return 0;
}
