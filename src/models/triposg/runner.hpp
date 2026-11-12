// TripoSG: HIP-free interface of the image -> GLB generator for the server
// (gufo serve objekt). CreateObjektRunner lives in the HIP library; the CPU-only
// build links a stub that reports the missing GPU path.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace gufo::triposg {

struct ObjektRequest {
    std::vector<std::uint8_t> rgba;  // packed RGBA8
    int width = 0, height = 0;
    int steps = 50;
    float guidance = 7.f;
    std::uint64_t seed = 42;
    int resolution = 512;
    int max_faces = 200000;
    bool preview = true;
    bool texture = false;   // TRELLIS.2: texture flow + decoder (vertex colours)
};

struct ObjektResult {
    std::vector<std::uint8_t> glb, preview_png;
    std::size_t vertices = 0, faces = 0, faces_before_decimation = 0;
    float bbox_min[3] = {0, 0, 0}, bbox_max[3] = {0, 0, 0};
    double volume = 0;
    bool closed = false;
    double seconds_encode = 0, seconds_denoise = 0, seconds_decode = 0, seconds_mesh = 0;
};

using ObjektProgress = std::function<void(const std::string& phase, int completed, int total)>;

class ObjektRunner {
public:
    virtual ~ObjektRunner() = default;
    virtual bool Run(const ObjektRequest& request, const ObjektProgress& progress, const std::atomic<bool>* cancel,
                     ObjektResult* result, std::string* error) = 0;
    virtual std::size_t weight_bytes() const = 0;
    // Sampler default when the request omits `steps` (TripoSG 50, TRELLIS.2 12).
    virtual int default_steps() const { return 50; }
    virtual std::string capability_note() const { return "shape"; }
    // Attribution the model's licences ask for, shown in /v1/models ("" = none).
    virtual std::string attribution() const { return ""; }
};

// Loads the converted triposg.safetensors onto the GPU (HIP build) or fails with *error.
std::unique_ptr<ObjektRunner> CreateObjektRunner(const std::filesystem::path& weights, std::string* error);

}  // namespace gufo::triposg
