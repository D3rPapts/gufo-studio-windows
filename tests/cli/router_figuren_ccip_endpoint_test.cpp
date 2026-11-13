// POST /v1/figuren "verfahren": "ccip" through FigurenService with a scripted
// chat fetch: request validation, the 503 without models, unchanged dispatch
// for other requests, and (with GUFO_CCIP_MODEL, GUFO_CCIP_FACE,
// GUFO_CCIP_FRAME and GUFO_CCIP_CROP_DIR) a real frame whose own head crops
// (<frame stem>_p<i>.png from ccip_ref.py) are the references: every detected
// person must get the figure made from its own crop (diff ~ 0).
// Written with AI assistance (Claude Code).
#undef NDEBUG
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "src/cli/serve/router/comfy_graph.hpp"
#include "src/cli/serve/router/figuren.hpp"
#include "src/core/json.hpp"

namespace json = gufo::json;
namespace figuren = gufo::router::figuren;
using gufo::server::HttpRequest;

namespace {

HttpRequest Post(const json::Value& body) {
  HttpRequest request;
  request.method = "POST";
  request.path = "/v1/figuren";
  request.body = body.dump();
  return request;
}

std::string FileDataUrl(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  assert(in);
  const std::string bytes((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
  return "data:image/png;base64," + gufo::router::comfy::Base64Encode(bytes);
}

struct Chats {
  int calls = 0;
  std::vector<std::string> bodies;
  gufo::router::comfy::FetchResult operator()(const HttpRequest& request) {
    ++calls;
    bodies.push_back(request.body);
    gufo::router::comfy::FetchResult result;
    result.status = 200;
    result.content_type = "application/json";
    const bool figure = request.body.find("Character description") != std::string::npos;
    const std::string content =
        figure ? "{\"schnurrbart\": false, \"vollbart\": false, \"brille\": \"keine\", \"haar\": \"grau\"}"
               : "{\"schnurrbart\": false, \"vollbart\": false, \"brille\": \"keine\", \"haar\": \"dunkel\"}";
    json::Value message = json::Value::object();
    message["role"] = "assistant";
    message["content"] = content;
    json::Value choice = json::Value::object();
    choice["message"] = std::move(message);
    json::Value choices = json::Value::array();
    choices.push_back(std::move(choice));
    json::Value body = json::Value::object();
    body["choices"] = std::move(choices);
    result.body = body.dump();
    return result;
  }
};

json::Value MinimalCcip() {
  json::Value ref = json::Value::object();
  ref["name"] = "lars";
  ref["kopf"] = "AAAA";
  json::Value refs = json::Value::array();
  refs.push_back(std::move(ref));
  json::Value body = json::Value::object();
  body["verfahren"] = "ccip";
  body["bild"] = "AAAA";
  body["referenzen"] = std::move(refs);
  return body;
}

}  // namespace

int main() {
  auto chats = std::make_shared<Chats>();
  figuren::Options options;
  options.fetch = [chats](const HttpRequest& r) { return (*chats)(r); };
  {
    const figuren::FigurenService service(options);
    // Unknown method: 400, nothing asked.
    json::Value body = MinimalCcip();
    body["verfahren"] = "zufall";
    auto response = service.Handle(Post(body));
    assert(response->status == 400);
    // ccip without referenzen: 400.
    body = MinimalCcip();
    body["referenzen"] = json::Value::array();
    response = service.Handle(Post(body));
    assert(response->status == 400);
    // galerie with an unknown name: 400.
    body = MinimalCcip();
    json::Value entry = json::Value::object();
    entry["name"] = "mayk";
    entry["bild"] = "AAAA";
    json::Value gallery = json::Value::array();
    gallery.push_back(std::move(entry));
    body["galerie"] = std::move(gallery);
    response = service.Handle(Post(body));
    assert(response->status == 400);
    assert(response->body.find("galerie") != std::string::npos);
    // Bad cap: 400.
    body = MinimalCcip();
    body["ccip_cap"] = "viel";
    response = service.Handle(Post(body));
    assert(response->status == 400);
    // Valid request but no model dirs configured: 503 ccip_unavailable.
    response = service.Handle(Post(MinimalCcip()));
    assert(response->status == 503);
    assert(response->body.find("ccip_unavailable") != std::string::npos);
    // Missing files: 503 naming the file.
    figuren::Options missing = options;
    missing.ccip_model_dir = "Z:/does-not-exist/ccip";
    missing.ccip_face_dir = "Z:/does-not-exist/face";
    const figuren::FigurenService service_missing(missing);
    response = service_missing.Handle(Post(MinimalCcip()));
    assert(response->status == 503);
    assert(response->body.find("ccip_caformer.safetensors") != std::string::npos);
    assert(chats->calls == 0);
  }
  std::cout << "validation: ok\n";

  const char* model_dir = std::getenv("GUFO_CCIP_MODEL");
  const char* face_dir = std::getenv("GUFO_CCIP_FACE");
  const char* frame = std::getenv("GUFO_CCIP_FRAME");
  const char* crop_dir = std::getenv("GUFO_CCIP_CROP_DIR");
  if (!model_dir || !face_dir || !frame || !crop_dir) {
    std::cout << "real-model part skipped (set GUFO_CCIP_MODEL, GUFO_CCIP_FACE, "
                 "GUFO_CCIP_FRAME, GUFO_CCIP_CROP_DIR)\nPASS\n";
    return 0;
  }
  figuren::Options real = options;
  real.ccip_model_dir = model_dir;
  real.ccip_face_dir = face_dir;
  const figuren::FigurenService service(real);
  json::Value refs = json::Value::array();
  json::Value profiles = json::Value::object();
  const std::string stem = std::filesystem::path(frame).stem().string();
  int figures = 0;
  for (;; ++figures) {
    const auto crop = std::filesystem::path(crop_dir) /
                      (stem + "_p" + std::to_string(figures) + ".png");
    if (!std::filesystem::exists(crop))
      break;
    json::Value ref = json::Value::object();
    ref["name"] = "f" + std::to_string(figures);
    ref["kopf"] = FileDataUrl(crop);
    refs.push_back(std::move(ref));
    // One Steckbrief so that the 27B path runs for that figure.
    if (figures == 0)
      profiles["f0"] = "grey hair, no glasses, no beard";
  }
  assert(figures > 0);
  json::Value body = json::Value::object();
  body["verfahren"] = "ccip";
  body["bild"] = FileDataUrl(frame);
  body["referenzen"] = std::move(refs);
  body["steckbrief"] = std::move(profiles);
  for (int round = 0; round < 2; ++round) {
    chats->calls = 0;
    const auto t0 = std::chrono::steady_clock::now();
    const auto response = service.Handle(Post(body));
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    if (response->status != 200)
      std::cerr << response->body << '\n';
    assert(response->status == 200);
    const json::Value out = json::parse(response->body);
    const auto& people = out.find("figuren")->items();
    std::cout << "round " << round << ": " << people.size() << " people, "
              << chats->calls << " chats, " << ms << " ms\n";
    assert(static_cast<int>(people.size()) == figures);
    std::vector<bool> used(static_cast<std::size_t>(figures), false);
    for (const json::Value& p : people) {
      // The figure made from this person's own crop has diff ~ 0.
      std::string own;
      for (const auto& [name, d] : p.find("diff")->members())
        if (d.as_double() < 1e-4)
          own = name;
      std::cout << "  " << p.find("name")->str() << " box " << p.find("box")->dump()
                << " own " << own << " llm " << p.find("llm_gefragt")->dump()
                << " | " << p.find("begruendung")->str() << '\n';
      assert(!own.empty());
      assert(p.find("name")->str() == own);
      const int index = std::stoi(own.substr(1));
      assert(!used[static_cast<std::size_t>(index)]);
      used[static_cast<std::size_t>(index)] = true;
      assert(p.find("beste")->str() == own);
    }
  }
  std::cout << "PASS\n";
  return 0;
}
