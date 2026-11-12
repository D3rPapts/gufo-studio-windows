// gufo serve objekt: asynchronous image -> 3D jobs (TripoSG) behind
//   POST   /v1/3d/generations            {model, image (base64 | data URL), steps, guidance,
//                                         seed, resolution 256|512, max_faces, preview}
//                                         -> 202 {id, status, progress, ...} + Location
//   GET    /v1/3d/generations/{id}       -> status snapshot (data[] once completed)
//   GET    /v1/3d/generations/{id}/content  -> model/gltf-binary (attachment {id}.glb)
//   GET    /v1/3d/generations/{id}/preview  -> image/png (three views)
//   DELETE /v1/3d/generations/{id}       -> cancel / remove
// One worker thread, queue capacity 1 (one running + one waiting), results persisted
// under <storage_root>/<id>/ with a TTL like the video jobs.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <chrono>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/models/triposg/runner.hpp"

namespace gufo::server {

struct ObjektServiceOptions {
    std::shared_ptr<triposg::ObjektRunner> runner;          // default model (model_id)
    std::map<std::string, std::shared_ptr<triposg::ObjektRunner>> runners;   // further models by id
    std::string model_id = "triposg";
    std::filesystem::path storage_root;
    std::size_t queue_capacity = 1;
    std::chrono::seconds artifact_ttl{86400};
    std::size_t max_image_bytes = 20u << 20u;
};

constexpr std::string_view kObjektApiPath = "/v1/3d/generations";
constexpr std::string_view kObjektApiSchema = "gufo.3d-api.v1";

class ObjektService {
public:
    explicit ObjektService(ObjektServiceOptions options);
    ~ObjektService();
    ObjektService(const ObjektService&) = delete;
    ObjektService& operator=(const ObjektService&) = delete;

    bool ready() const noexcept;
    std::string model_id() const;
    std::vector<std::string> model_ids() const;
    std::string capability_note(const std::string& model_id) const;
    std::string initialization_error() const;
    std::size_t weight_bytes() const;
    HttpResponse Handle(const HttpRequest& request);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool IsObjektApiPath(std::string_view path) noexcept;

}  // namespace gufo::server
