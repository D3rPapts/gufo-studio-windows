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
// A gufo-studio-windows media service (src/cli/serve/media/media_service.hpp):
// HttpServer lists it in /ready and /v1/models ("capability": "image_to_3d").
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

#include "src/cli/serve/media/media_service.hpp"
#include "src/models/triposg/runner.hpp"

namespace gufo::server {

struct ObjektServiceOptions {
    std::shared_ptr<triposg::ObjektRunner> runner;
    std::string model_id = "triposg";
    std::filesystem::path storage_root;
    std::size_t queue_capacity = 1;
    std::chrono::seconds artifact_ttl{86400};
    std::size_t max_image_bytes = 20u << 20u;
};

constexpr std::string_view kObjektApiPath = "/v1/3d/generations";
constexpr std::string_view kObjektApiSchema = "gufo.3d-api.v1";

class ObjektService final : public MediaService {
public:
    explicit ObjektService(ObjektServiceOptions options);
    ~ObjektService() override;
    ObjektService(const ObjektService&) = delete;
    ObjektService& operator=(const ObjektService&) = delete;

    [[nodiscard]] bool ready() const noexcept override;
    [[nodiscard]] std::string model_id() const override;
    std::string initialization_error() const;
    std::size_t weight_bytes() const;

    [[nodiscard]] std::string_view capability() const override { return "image_to_3d"; }
    [[nodiscard]] std::string_view owned_by() const override { return "operator-supplied-triposg"; }
    /// /ready: "capability": "image_to_3d" (as in 1.4.23; /v1/models gets it
    /// from capability()).
    void Describe(json::Value* entry, bool ready) const override;
    [[nodiscard]] bool HandlesPath(std::string_view path) const override;
    [[nodiscard]] std::string_view unavailable_message() const override { return "3D service is not ready"; }
    [[nodiscard]] std::string_view unavailable_code() const override { return "objekt_service_unavailable"; }
    HttpResponse Handle(const HttpRequest& request) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool IsObjektApiPath(std::string_view path) noexcept;

}  // namespace gufo::server
