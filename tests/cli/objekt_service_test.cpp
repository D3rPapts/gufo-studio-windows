// gufo serve objekt (ObjektService) with a fake runner, no model, no GPU:
// numbers are range-checked before the int conversion (1e300 steps, a
// negative seed) and a DELETE while the job runs leaves no job directory.
// AI-assisted (Claude Code), 2026-11-12.
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "src/cli/serve/objekt_service.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen_image_21/model.hpp"

namespace {

int failures = 0;

#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::cerr << __FILE__ << ":" << __LINE__ << " CHECK " #cond "\n"; \
      ++failures;                                                       \
    }                                                                   \
  } while (0)

namespace json = gufo::json;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;

// Waits for the cancel, then still returns a result: a runner that notices
// the cancel only after its last step.
struct FakeRunner : gufo::triposg::ObjektRunner {
  std::atomic<bool> started{false};
  bool Run(const gufo::triposg::ObjektRequest&,
           const gufo::triposg::ObjektProgress&,
           const std::atomic<bool>* cancel, gufo::triposg::ObjektResult* result,
           std::string*) override {
    started = true;
    for (int i = 0; i < 2000 && !cancel->load(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    result->glb = {'g', 'l', 'T', 'F'};
    result->preview_png = {1, 2, 3};
    return true;
  }
  std::size_t weight_bytes() const override { return 0; }
};

std::string Base64(const std::vector<std::uint8_t>& data) {
  static constexpr char kTable[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  std::size_t i = 0;
  for (; i + 2 < data.size(); i += 3) {
    const unsigned v = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
    out += kTable[(v >> 18) & 63];
    out += kTable[(v >> 12) & 63];
    out += kTable[(v >> 6) & 63];
    out += kTable[v & 63];
  }
  if (i + 1 == data.size()) {
    const unsigned v = data[i] << 16;
    out += kTable[(v >> 18) & 63];
    out += kTable[(v >> 12) & 63];
    out += "==";
  } else if (i + 2 == data.size()) {
    const unsigned v = (data[i] << 16) | (data[i + 1] << 8);
    out += kTable[(v >> 18) & 63];
    out += kTable[(v >> 12) & 63];
    out += kTable[(v >> 6) & 63];
    out += '=';
  }
  return out;
}

std::string Png() {
  gufo::models::qwen_image_21::Image image;
  image.width = 4;
  image.height = 4;
  image.rgba.assign(4 * 4 * 4, 200);
  return Base64(gufo::models::qwen_image_21::EncodePng(image));
}

HttpResponse Post(gufo::server::ObjektService& service, const std::string& extra) {
  HttpRequest request;
  request.method = "POST";
  request.path = "/v1/3d/generations";
  request.body = R"({"model":"triposg","image":")" + Png() + "\"" + extra + "}";
  return service.Handle(request);
}

std::string Code(const HttpResponse& response) {
  try {
    return json::parse(response.body).find("error")->member_str("code");
  } catch (const std::exception&) {
    return "";
  }
}

void TestNumbersRangeChecked(gufo::server::ObjektService& service) {
  CHECK(Code(Post(service, R"(,"steps":1e300)")) == "invalid_steps");
  CHECK(Code(Post(service, R"(,"steps":-1e300)")) == "invalid_steps");
  CHECK(Code(Post(service, R"(,"steps":0)")) == "invalid_steps");
  CHECK(Code(Post(service, R"(,"seed":-1)")) == "invalid_seed");
  CHECK(Code(Post(service, R"(,"seed":1e30)")) == "invalid_seed");
  CHECK(Code(Post(service, R"(,"guidance":1e300)")) == "invalid_guidance");
  CHECK(Code(Post(service, R"(,"resolution":1e20)")) == "invalid_resolution");
  CHECK(Code(Post(service, R"(,"resolution":384)")) == "invalid_resolution");
  CHECK(Code(Post(service, R"(,"max_faces":1e20)")) == "invalid_max_faces");
  CHECK(Code(Post(service, R"(,"max_faces":-1)")) == "invalid_max_faces");
}

void TestDeleteWhileRunning(gufo::server::ObjektService& service,
                            FakeRunner& runner,
                            const std::filesystem::path& root) {
  const HttpResponse created = Post(service, R"(,"steps":10,"seed":7)");
  CHECK(created.status == 202);
  const std::string id = json::parse(created.body).member_str("id");
  for (int i = 0; i < 400 && !runner.started.load(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  CHECK(runner.started.load());
  CHECK(std::filesystem::exists(root / id));  // job.json while it runs
  HttpRequest remove;
  remove.method = "DELETE";
  remove.path = "/v1/3d/generations/" + id;
  CHECK(service.Handle(remove).status == 200);
  // The worker confirms the cancel by finishing Run(); then the directory
  // is gone and no result file was written.
  bool gone = false;
  for (int i = 0; i < 400 && !gone; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    gone = !std::filesystem::exists(root / id);
  }
  CHECK(gone);
  HttpRequest status;
  status.method = "GET";
  status.path = "/v1/3d/generations/" + id;
  CHECK(service.Handle(status).status == 404);
}

}  // namespace

int main() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "gufo-objekt-service-test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  auto runner = std::make_shared<FakeRunner>();
  {
    gufo::server::ObjektServiceOptions options;
    options.runner = runner;
    options.storage_root = root;
    gufo::server::ObjektService service(options);
    CHECK(service.ready());
    TestNumbersRangeChecked(service);
    TestDeleteWhileRunning(service, *runner, root);
  }
  std::filesystem::remove_all(root);
  if (failures != 0) {
    std::cerr << failures << " objekt service check(s) failed\n";
    return 1;
  }
  std::cout << "objekt_service_test: all passed\n";
  return 0;
}
