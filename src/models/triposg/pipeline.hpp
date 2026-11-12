// TripoSG: the whole image -> GLB pipeline on the GPU (HIP build only):
// preprocess -> DINOv2 -> rectified-flow DiT (CFG, Euler) -> VAE decoder on a
// hierarchical SDF grid -> marching cubes -> decimation -> GLB + preview PNG.
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

struct GenerateOptions {
    int steps = 50;
    float guidance = 7.f;
    std::uint64_t seed = 42;
    int resolution = 512;      // final grid: 256 or 512 (coarse level is 256)
    int max_faces = 200000;    // 0 = no decimation
    bool preview = true;       // render preview.png (three views)
    int preview_size = 384;
};

struct GenerateResult {
    std::vector<std::uint8_t> glb;
    std::vector<std::uint8_t> preview_png;  // empty unless requested
    std::size_t vertices = 0, faces = 0, faces_before_decimation = 0;
    float bbox_min[3] = {0, 0, 0}, bbox_max[3] = {0, 0, 0};
    double volume = 0;
    bool closed = false;
    double seconds_encode = 0, seconds_denoise = 0, seconds_decode = 0, seconds_mesh = 0;
};

// phase: "encode" | "denoise" | "decode" | "mesh" | "export"; completed/total within the phase.
using GenerateProgress = std::function<void(const std::string& phase, int completed, int total)>;

class Pipeline {
public:
    ~Pipeline();
    // Loads DINOv2 (FP32), DiT and VAE (BF16) from the converted triposg.safetensors.
    static std::unique_ptr<Pipeline> Load(const std::filesystem::path& weights, std::string* error);
    // rgba: packed RGBA8 (alpha required for the foreground). Thread-safe for one call at a time.
    bool Generate(const std::uint8_t* rgba, int width, int height, const GenerateOptions& options,
                  const GenerateProgress& progress, const std::atomic<bool>* cancel, GenerateResult* result,
                  std::string* error);
    std::size_t weight_bytes() const;

private:
    struct Impl;
    explicit Pipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::triposg
