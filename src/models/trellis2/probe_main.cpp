// gufo-trellis2-probe: runs single stages of the TRELLIS.2 port and writes .npy dumps
// for the comparison with the PyTorch reference (tools/trellis2/vergleich.py).
//
//   gufo-trellis2-probe --stage dino --weights trellis2-dinov3.safetensors --image bild.png --out DIR
//       [--device cpu|gpu] [--image-size 512]
//     writes preproc_512.npy [3,512,512], dino_tokens_512.npy [1029,1024], square.png (the crop)
//   gufo-trellis2-probe --selftest-dino [--layers 2] [--image-size 512]
//     random weights: CPU vs HIP (relative RMS)
//   gufo-trellis2-probe --stage ss --weights trellis2-ss_flow.safetensors --dumps DIR --out DIR
//       [--device cpu|gpu] [--seed 42] [--steps 12] [--zero-cond] [--cpu-check] [--layers N]
//     DIR holds dino_tokens_512.npy and optionally ss_noise.npy [8,16,16,16]; writes ss_noise.npy
//     (if generated), ss_block00.npy / ss_block29.npy [4096,1536] (step 0, conditional),
//     ss_v_step00_{cond,uncond}.npy [8,16,16,16] and after --steps Euler steps ss_latent_final.npy.
//     --cpu-check runs the FP32 CPU reference for step 0 (first --layers blocks) and prints the deviation.
//   gufo-trellis2-probe --stage ssdec --weights trellis2-ss_dec.safetensors --dumps DIR --out DIR
//       [--device cpu|gpu] [--cpu-check]
//     DIR holds ss_latent_final.npy; writes ss_logits64.npy, ss_occupancy64.npy (0/1), ss_coords32.npy [N,3],
//     ss_slices.png (mosaic of the 64 z-slices) and prints the active voxel counts.
//   gufo-trellis2-probe --stage shape --weights trellis2-shape_flow_512.safetensors --dumps DIR --out DIR
//       [--device cpu|gpu] [--seed 42] [--steps 12] [--cpu-check] [--layers N]
//     DIR holds dino_tokens_512.npy + ss_coords32.npy (+ optional shape_noise.npy [N,32]); writes shape_noise.npy,
//     shape_block00/29.npy, shape_v_step00_cond.npy and shape_latent_final.npy [N,32] (after * std + mean) + shape_coords.npy.
//   gufo-trellis2-probe --stage shapedec --weights trellis2-shape_dec.safetensors --dumps DIR --out DIR
//       [--device cpu|gpu] [--cpu-check] [--cpu-levels 2]
//     DIR holds shape_latent_final.npy + shape_coords.npy; writes shape_dec_out.npy [M,7], shape_dec_coords.npy,
//     shape_subdiv_{0..3}.npy, mesh.glb (dual grid, 512^3) and preview.png.
//
// No server, no port; the GPU path needs a HIP build (GUFO_TRELLIS2_HIP).
// AI-assisted (Claude Code), 2026-11-12.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/qwen_image_21/model.hpp"
#include "src/models/trellis2/config.hpp"
#include "src/models/trellis2/dinov3.hpp"
#include "src/models/trellis2/dit.hpp"
#include "src/models/trellis2/dual_grid_mesh.hpp"
#include "src/models/trellis2/glb_writer.hpp"
#include "src/models/trellis2/mesh_preview.hpp"
#include "src/models/trellis2/npy.hpp"
#include "src/models/trellis2/png.hpp"
#include "src/models/trellis2/preprocess.hpp"
#include "src/models/trellis2/scheduler.hpp"
#include "src/models/trellis2/sparse_dec.hpp"
#include "src/models/trellis2/ss_dec.hpp"

namespace {

using Clock = std::chrono::steady_clock;
namespace tr = gufo::trellis2;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double rel_rms(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        num += static_cast<double>(a[i] - b[i]) * (a[i] - b[i]);
        den += static_cast<double>(b[i]) * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double ab = 0, aa = 0, bb = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
    return ab / std::max(std::sqrt(aa * bb), 1e-30);
}

int usage() {
    std::cerr << "usage: see the header of probe_main.cpp\n";
    return 2;
}

bool save(const std::string& dir, const std::string& name, const std::vector<float>& data, const std::vector<std::size_t>& shape) {
    std::string error;
    if (!tr::npy::write_f32(std::filesystem::path(dir) / name, data.data(), shape, &error)) {
        std::cerr << error << '\n';
        return false;
    }
    std::cout << "  wrote " << name << " [";
    for (std::size_t i = 0; i < shape.size(); ++i) std::cout << (i ? "," : "") << shape[i];
    std::cout << "]\n";
    return true;
}

// torch layout [C, R, R, R] (channel-major) <-> voxel-major [R^3, C]
std::vector<float> to_voxel_major(const std::vector<float>& chw, int channels, int voxels) {
    std::vector<float> out(chw.size());
    for (int c = 0; c < channels; ++c)
        for (int v = 0; v < voxels; ++v) out[static_cast<std::size_t>(v) * channels + c] = chw[static_cast<std::size_t>(c) * voxels + v];
    return out;
}
std::vector<float> to_channel_major(const std::vector<float>& vm, int channels, int voxels) {
    std::vector<float> out(vm.size());
    for (int c = 0; c < channels; ++c)
        for (int v = 0; v < voxels; ++v) out[static_cast<std::size_t>(c) * voxels + v] = vm[static_cast<std::size_t>(v) * channels + c];
    return out;
}

// ---------------------------------------------------------------- dino
int run_dino(const std::string& weights, const std::string& image, const std::string& out, const std::string& device, int image_size) {
    std::string error;
    auto t0 = Clock::now();
    gufo::models::qwen_image_21::Image rgba;
    try {
        rgba = gufo::models::qwen_image_21::DecodeImage(gufo::core::ReadImageFile(image));
    } catch (const std::exception& e) {
        std::cerr << "image: " << e.what() << '\n';
        return 1;
    }
    tr::Preprocessed pre;
    if (!tr::preprocess_rgba(rgba.rgba.data(), rgba.width, rgba.height, image_size, &pre, &error)) {
        std::cerr << "preprocess: " << error << '\n';
        return 1;
    }
    std::cout << "preprocess: " << rgba.width << "x" << rgba.height << " bbox " << pre.bbox[0] << "," << pre.bbox[1] << " "
              << pre.bbox[2] << "x" << pre.bbox[3] << " square " << pre.square_size << " in " << ms_since(t0) << " ms\n";
    std::filesystem::create_directories(out);
    const std::size_t S = static_cast<std::size_t>(image_size);
    if (!save(out, "preproc_" + std::to_string(image_size) + ".npy", pre.pixels, {3, S, S})) return 1;
    tr::write_png_rgb(std::filesystem::path(out) / "square.png", pre.square.data(), pre.square_size, static_cast<int>(pre.square.size() / 3 / static_cast<std::size_t>(pre.square_size)), &error);

    t0 = Clock::now();
    tr::Dinov3Weights w;
    if (!tr::Dinov3Weights::Load(weights, &w, &error)) {
        std::cerr << "weights: " << error << '\n';
        return 1;
    }
    std::cout << "weights: " << w.parameter_count() << " parameters in " << ms_since(t0) << " ms\n";
    std::vector<float> tokens;
    t0 = Clock::now();
    if (device == "gpu") {
#if GUFO_TRELLIS2_HIP
        auto enc = tr::Dinov3Hip::Create(w, image_size, &error);
        if (!enc) { std::cerr << "hip: " << error << '\n'; return 1; }
        std::cout << "upload: " << enc->device_bytes() / (1024.0 * 1024.0) << " MiB in " << ms_since(t0) << " ms\n";
        t0 = Clock::now();
        if (!enc->Encode(pre.pixels.data(), &tokens, &error)) { std::cerr << "encode: " << error << '\n'; return 1; }
#else
        std::cerr << "no HIP build\n";
        return 1;
#endif
    } else if (!tr::dinov3_encode_cpu(w, pre.pixels.data(), image_size, &tokens, &error)) {
        std::cerr << "encode: " << error << '\n';
        return 1;
    }
    const std::size_t T = static_cast<std::size_t>(tr::DinoV3Config::tokens(image_size));
    std::cout << "dino (" << device << "): " << T << " tokens in " << ms_since(t0) << " ms\n";
    return save(out, "dino_tokens_" + std::to_string(image_size) + ".npy", tokens, {T, 1024}) ? 0 : 1;
}

int selftest_dino(int layers, int image_size) {
#if GUFO_TRELLIS2_HIP
    std::string error;
    auto w = tr::Dinov3Weights::Random(11, 0.02f, layers);
    std::vector<float> pixels(static_cast<std::size_t>(3) * image_size * image_size);
    for (std::size_t i = 0; i < pixels.size(); ++i) pixels[i] = std::sin(static_cast<float>(i) * 0.013f) * 1.5f;
    std::vector<float> cpu, gpu;
    auto t0 = Clock::now();
    if (!tr::dinov3_encode_cpu(w, pixels.data(), image_size, &cpu, &error)) { std::cerr << error << '\n'; return 1; }
    std::cout << "cpu " << layers << " layers: " << ms_since(t0) << " ms\n";
    auto enc = tr::Dinov3Hip::Create(w, image_size, &error);
    if (!enc) { std::cerr << error << '\n'; return 1; }
    t0 = Clock::now();
    if (!enc->Encode(pixels.data(), &gpu, &error)) { std::cerr << error << '\n'; return 1; }
    std::cout << "gpu: " << ms_since(t0) << " ms (first call)\n";
    t0 = Clock::now();
    enc->Encode(pixels.data(), &gpu, &error);
    std::cout << "gpu: " << ms_since(t0) << " ms (warm)\n";
    const double r = rel_rms(gpu, cpu);
    std::cout << "dino selftest: rel RMS " << r << " cosine " << cosine(gpu, cpu) << (r < 1e-3 ? "  OK\n" : "  FAIL\n");
    return r < 1e-3 ? 0 : 1;
#else
    (void)layers; (void)image_size;
    std::cerr << "no HIP build\n";
    return 1;
#endif
}

// ---------------------------------------------------------------- ss flow
struct SsRun {
    std::string weights, dumps, out, device;
    int seed = 42, steps = 12, layers = 0;
    bool zero_cond = false, cpu_check = false;
};

int run_ss(const SsRun& o) {
    std::string error;
    std::filesystem::create_directories(o.out);
    const int R = tr::SsDecConfig::latent_res, C = 8, N = R * R * R;
    const int M = tr::DinoV3Config::tokens_512;
    std::vector<float> cond;
    std::vector<std::size_t> shape;
    if (!o.zero_cond) {
        if (!tr::npy::read_f32(std::filesystem::path(o.dumps) / "dino_tokens_512.npy", &cond, &shape, &error) || shape.size() != 2 ||
            shape[0] != static_cast<std::size_t>(M) || shape[1] != 1024) {
            std::cerr << "dino_tokens_512.npy: " << error << " (expected [1029,1024])\n";
            return 1;
        }
    } else {
        cond.assign(static_cast<std::size_t>(M) * 1024, 0.f);
    }
    // Noise in torch layout [8,16,16,16] (seed) unless provided.
    std::vector<float> noise_cm;
    if (tr::npy::read_f32(std::filesystem::path(o.dumps) / "ss_noise.npy", &noise_cm, &shape, &error) && noise_cm.size() == static_cast<std::size_t>(C) * N) {
        std::cout << "noise: ss_noise.npy from the dumps\n";
    } else {
        std::mt19937_64 rng(static_cast<std::uint64_t>(o.seed));
        std::normal_distribution<float> nd(0.f, 1.f);
        noise_cm.resize(static_cast<std::size_t>(C) * N);
        for (auto& v : noise_cm) v = nd(rng);
        if (!save(o.out, "ss_noise.npy", noise_cm, {8, 16, 16, 16})) return 1;
    }
    std::vector<float> x = to_voxel_major(noise_cm, C, N);   // [4096, 8]
    const std::vector<tr::Coord> coords = tr::dense_grid_coords(R);
    tr::DitShape shape_dit;
    shape_dit.in_channels = 8;
    shape_dit.out_channels = 8;
    tr::DitForwardOptions opts;
    opts.tap_layers = {0, 29};
    opts.layer_limit = o.layers;

    std::unique_ptr<tr::DitHip> hip;
    tr::DitHostWeights host;
    host.shape = shape_dit;
    auto t0 = Clock::now();
    if (o.device == "gpu") {
#if GUFO_TRELLIS2_HIP
        hip = tr::DitHip::Create(o.weights, shape_dit, &error);
        if (!hip) { std::cerr << error << '\n'; return 1; }
#else
        std::cerr << "no HIP build\n";
        return 1;
#endif
    }
    if (o.device == "cpu" || o.cpu_check) {
        if (!tr::DitHostWeights::FromFile(o.weights, &host, &error)) { std::cerr << error << '\n'; return 1; }
    }
    std::cout << "weights ready in " << ms_since(t0) << " ms\n";

    auto forward = [&](const float* xin, const float* c, float t, std::vector<float>* v, std::map<int, std::vector<float>>* taps) {
        if (hip) return hip->Forward(xin, coords, c, M, t, v, taps, &error, opts);
        return tr::dit_forward_cpu(host, xin, coords, c, M, t, v, taps, &error, opts);
    };

    const tr::SamplerParams p = tr::kSsSampler;
    const std::vector<float> times = tr::flow_times(o.steps, p.rescale_t);
    std::vector<float> zero_cond(cond.size(), 0.f);
    // Step 0 with taps (conditional) and the unconditional velocity.
    std::vector<float> v_cond, v_uncond;
    std::map<int, std::vector<float>> taps;
    t0 = Clock::now();
    if (!forward(x.data(), cond.data(), tr::model_timestep(times[0]), &v_cond, &taps)) { std::cerr << error << '\n'; return 1; }
    std::cout << "step 0 forward (" << o.device << "): " << ms_since(t0) << " ms\n";
    for (const auto& [layer, h] : taps) {
        char name[32];
        std::snprintf(name, sizeof(name), "ss_block%02d.npy", layer);
        if (!save(o.out, name, h, {static_cast<std::size_t>(N), 1536})) return 1;
    }
    if (!save(o.out, "ss_v_step00_cond.npy", to_channel_major(v_cond, C, N), {8, 16, 16, 16})) return 1;
    if (!forward(x.data(), zero_cond.data(), tr::model_timestep(times[0]), &v_uncond, nullptr)) { std::cerr << error << '\n'; return 1; }
    if (!save(o.out, "ss_v_step00_uncond.npy", to_channel_major(v_uncond, C, N), {8, 16, 16, 16})) return 1;

    if (o.cpu_check && hip) {
        std::vector<float> v_cpu;
        std::map<int, std::vector<float>> taps_cpu;
        t0 = Clock::now();
        if (!tr::dit_forward_cpu(host, x.data(), coords, cond.data(), M, tr::model_timestep(times[0]), &v_cpu, &taps_cpu, &error, opts)) {
            std::cerr << error << '\n';
            return 1;
        }
        std::cout << std::defaultfloat << std::setprecision(6);   // the HIP runtime leaves cout in fixed(2)
        std::cout << "cpu check (" << (o.layers > 0 ? o.layers : 30) << " layers): " << ms_since(t0) << " ms\n";
        for (const auto& [layer, h] : taps_cpu)
            if (taps.count(layer)) std::cout << "  block " << layer << ": rel RMS " << rel_rms(taps[layer], h) << " cosine " << cosine(taps[layer], h) << '\n';
        std::cout << "  v: rel RMS " << rel_rms(v_cond, v_cpu) << " cosine " << cosine(v_cond, v_cpu) << '\n';
    }
    if (o.steps <= 0 || o.layers > 0) return 0;

    // Full Euler run with CFG (interval + rescale) in FP32 on the host.
    t0 = Clock::now();
    for (int i = 0; i < o.steps; ++i) {
        const float t = times[static_cast<std::size_t>(i)], t_prev = times[static_cast<std::size_t>(i) + 1];
        std::vector<float> pos, neg;
        if (i == 0) { pos = v_cond; neg = v_uncond; }
        else if (!forward(x.data(), cond.data(), tr::model_timestep(t), &pos, nullptr)) { std::cerr << error << '\n'; return 1; }
        if (tr::guidance_active(t, p)) {
            if (i != 0 && !forward(x.data(), zero_cond.data(), tr::model_timestep(t), &neg, nullptr)) { std::cerr << error << '\n'; return 1; }
            tr::apply_guidance(&pos, neg, x, t, p);
        }
        tr::euler_step(x, pos, t, t_prev);
        double rms = 0;
        for (float e : x) rms += static_cast<double>(e) * e;
        std::cout << "  step " << i << " t " << t << " -> " << t_prev << (tr::guidance_active(t, p) ? " cfg" : "    ") << " x rms "
                  << std::sqrt(rms / x.size()) << '\n';
    }
    std::cout << o.steps << " steps in " << ms_since(t0) << " ms\n";
    return save(o.out, "ss_latent_final.npy", to_channel_major(x, C, N), {8, 16, 16, 16}) ? 0 : 1;
}

// ---------------------------------------------------------------- ss decoder
int run_ssdec(const std::string& weights, const std::string& dumps, const std::string& out, const std::string& device, bool cpu_check) {
    std::string error;
    std::filesystem::create_directories(out);
    std::vector<float> latent_cm;
    std::vector<std::size_t> shape;
    if (!tr::npy::read_f32(std::filesystem::path(dumps) / "ss_latent_final.npy", &latent_cm, &shape, &error) || latent_cm.size() != 8 * 4096) {
        std::cerr << "ss_latent_final.npy: " << error << " (expected [8,16,16,16])\n";
        return 1;
    }
    const std::vector<float> latent = to_voxel_major(latent_cm, 8, 4096);
    auto t0 = Clock::now();
    tr::SsDecWeights w;
    if (!tr::SsDecWeights::Load(weights, &w, &error)) { std::cerr << error << '\n'; return 1; }
    std::cout << "weights: " << w.parameter_count() << " parameters in " << ms_since(t0) << " ms\n";
    std::vector<float> logits;
    t0 = Clock::now();
    if (device == "gpu") {
#if GUFO_TRELLIS2_HIP
        auto dec = tr::SsDecHip::Create(w, &error);
        if (!dec) { std::cerr << error << '\n'; return 1; }
        std::cout << "upload " << dec->device_bytes() / (1024.0 * 1024.0) << " MiB in " << ms_since(t0) << " ms\n";
        t0 = Clock::now();
        if (!dec->Forward(latent.data(), &logits, &error)) { std::cerr << error << '\n'; return 1; }
#else
        std::cerr << "no HIP build\n";
        return 1;
#endif
    } else if (!tr::ss_dec_forward_cpu(w, latent.data(), &logits, &error)) {
        std::cerr << error << '\n';
        return 1;
    }
    std::cout << "ss_dec (" << device << "): 64^3 in " << ms_since(t0) << " ms\n";
    if (cpu_check && device == "gpu") {
        std::vector<float> cpu;
        t0 = Clock::now();
        if (!tr::ss_dec_forward_cpu(w, latent.data(), &cpu, &error)) { std::cerr << error << '\n'; return 1; }
        std::cout << std::defaultfloat << std::setprecision(6);
        std::cout << "cpu check: " << ms_since(t0) << " ms, rel RMS " << rel_rms(logits, cpu) << " cosine " << cosine(logits, cpu);
        std::size_t agree = 0;
        for (std::size_t i = 0; i < cpu.size(); ++i) agree += (cpu[i] > 0) == (logits[i] > 0);
        std::cout << " sign agreement " << 100.0 * agree / cpu.size() << " %\n";
    }
    std::vector<float> occ(logits.size());
    std::size_t active = 0;
    for (std::size_t i = 0; i < logits.size(); ++i) { occ[i] = logits[i] > 0 ? 1.f : 0.f; active += logits[i] > 0; }
    const auto coords32 = tr::occupancy_to_coords(logits.data(), 64, 32);
    std::cout << "occupancy: " << active << " / 262144 active at 64^3, " << coords32.size() << " at 32^3 (tokens for the 512 path)\n";
    if (!save(out, "ss_logits64.npy", logits, {64, 64, 64}) || !save(out, "ss_occupancy64.npy", occ, {64, 64, 64})) return 1;
    std::vector<float> cf;
    for (const auto& c : coords32) { cf.push_back(static_cast<float>(c.x)); cf.push_back(static_cast<float>(c.y)); cf.push_back(static_cast<float>(c.z)); }
    if (!save(out, "ss_coords32.npy", cf, {coords32.size(), 3})) return 1;
    int wpx, hpx;
    const auto rgb = tr::slice_mosaic(occ.data(), 64, 0.f, 1.f, 3, &wpx, &hpx);
    if (!tr::write_png_rgb(std::filesystem::path(out) / "ss_slices.png", rgb.data(), wpx, hpx, &error)) { std::cerr << error << '\n'; return 1; }
    std::cout << "  wrote ss_slices.png " << wpx << "x" << hpx << '\n';
    return 0;
}

// ---------------------------------------------------------------- shape flow (sparse DiT)
std::vector<tr::Coord> read_coords(const std::string& path, std::string* error) {
    std::vector<float> cf;
    std::vector<std::size_t> shape;
    std::vector<tr::Coord> coords;
    if (!tr::npy::read_f32(path, &cf, &shape, error) || shape.size() != 2 || shape[1] != 3) {
        if (error && error->empty()) *error = "expected [N,3]";
        return coords;
    }
    coords.reserve(shape[0]);
    for (std::size_t i = 0; i < shape[0]; ++i)
        coords.push_back({static_cast<int>(cf[i * 3]), static_cast<int>(cf[i * 3 + 1]), static_cast<int>(cf[i * 3 + 2])});
    return coords;
}

int run_shape(const SsRun& o) {
    std::string error;
    std::filesystem::create_directories(o.out);
    const int M = tr::DinoV3Config::tokens_512, C = 32;
    std::vector<float> cond;
    std::vector<std::size_t> shape;
    if (!tr::npy::read_f32(std::filesystem::path(o.dumps) / "dino_tokens_512.npy", &cond, &shape, &error) || cond.size() != static_cast<std::size_t>(M) * 1024) {
        std::cerr << "dino_tokens_512.npy: " << error << '\n';
        return 1;
    }
    const std::vector<tr::Coord> coords = read_coords((std::filesystem::path(o.dumps) / "ss_coords32.npy").string(), &error);
    if (coords.empty()) { std::cerr << "ss_coords32.npy: " << error << '\n'; return 1; }
    const int N = static_cast<int>(coords.size());
    std::cout << "shape flow: " << N << " tokens\n";
    std::vector<float> x;
    if (tr::npy::read_f32(std::filesystem::path(o.dumps) / "shape_noise.npy", &x, &shape, &error) && x.size() == static_cast<std::size_t>(N) * C) {
        std::cout << "noise: shape_noise.npy from the dumps\n";
    } else {
        std::mt19937_64 rng(static_cast<std::uint64_t>(o.seed) + 1);
        std::normal_distribution<float> nd(0.f, 1.f);
        x.resize(static_cast<std::size_t>(N) * C);
        for (auto& v : x) v = nd(rng);
        if (!save(o.out, "shape_noise.npy", x, {static_cast<std::size_t>(N), 32})) return 1;
    }
    tr::DitShape shape_dit;
    shape_dit.in_channels = 32;
    shape_dit.out_channels = 32;
    tr::DitForwardOptions opts;
    opts.tap_layers = {0, 29};
    opts.layer_limit = o.layers;
    std::unique_ptr<tr::DitHip> hip;
    tr::DitHostWeights host;
    host.shape = shape_dit;
    auto t0 = Clock::now();
    if (o.device == "gpu") {
#if GUFO_TRELLIS2_HIP
        hip = tr::DitHip::Create(o.weights, shape_dit, &error);
        if (!hip) { std::cerr << error << '\n'; return 1; }
#else
        std::cerr << "no HIP build\n";
        return 1;
#endif
    }
    if (o.device == "cpu" || o.cpu_check) {
        if (!tr::DitHostWeights::FromFile(o.weights, &host, &error)) { std::cerr << error << '\n'; return 1; }
    }
    std::cout << "weights ready in " << ms_since(t0) << " ms\n";
    auto forward = [&](const float* xin, const float* c, float t, std::vector<float>* v, std::map<int, std::vector<float>>* taps) {
        if (hip) return hip->Forward(xin, coords, c, M, t, v, taps, &error, opts);
        return tr::dit_forward_cpu(host, xin, coords, c, M, t, v, taps, &error, opts);
    };
    const tr::SamplerParams p = tr::kShapeSampler;
    const std::vector<float> times = tr::flow_times(o.steps, p.rescale_t);
    std::vector<float> zero_cond(cond.size(), 0.f);
    std::vector<float> v_cond, v_uncond;
    std::map<int, std::vector<float>> taps;
    t0 = Clock::now();
    if (!forward(x.data(), cond.data(), tr::model_timestep(times[0]), &v_cond, &taps)) { std::cerr << error << '\n'; return 1; }
    std::cout << "step 0 forward (" << o.device << "): " << ms_since(t0) << " ms\n";
    for (const auto& [layer, h] : taps) {
        char name[32];
        std::snprintf(name, sizeof(name), "shape_block%02d.npy", layer);
        if (!save(o.out, name, h, {static_cast<std::size_t>(N), 1536})) return 1;
    }
    if (!save(o.out, "shape_v_step00_cond.npy", v_cond, {static_cast<std::size_t>(N), 32})) return 1;
    if (o.cpu_check && hip) {
        std::vector<float> v_cpu;
        std::map<int, std::vector<float>> taps_cpu;
        t0 = Clock::now();
        if (!tr::dit_forward_cpu(host, x.data(), coords, cond.data(), M, tr::model_timestep(times[0]), &v_cpu, &taps_cpu, &error, opts)) {
            std::cerr << error << '\n';
            return 1;
        }
        std::cout << std::defaultfloat << std::setprecision(6);
        std::cout << "cpu check (" << (o.layers > 0 ? o.layers : 30) << " layers): " << ms_since(t0) << " ms\n";
        for (const auto& [layer, h] : taps_cpu)
            if (taps.count(layer)) std::cout << "  block " << layer << ": rel RMS " << rel_rms(taps[layer], h) << " cosine " << cosine(taps[layer], h) << '\n';
        std::cout << "  v: rel RMS " << rel_rms(v_cond, v_cpu) << " cosine " << cosine(v_cond, v_cpu) << '\n';
    }
    if (o.steps <= 0 || o.layers > 0) return 0;
    t0 = Clock::now();
    for (int i = 0; i < o.steps; ++i) {
        const float t = times[static_cast<std::size_t>(i)], t_prev = times[static_cast<std::size_t>(i) + 1];
        std::vector<float> pos, neg;
        if (i == 0) pos = v_cond;
        else if (!forward(x.data(), cond.data(), tr::model_timestep(t), &pos, nullptr)) { std::cerr << error << '\n'; return 1; }
        if (tr::guidance_active(t, p)) {
            if (!forward(x.data(), zero_cond.data(), tr::model_timestep(t), &neg, nullptr)) { std::cerr << error << '\n'; return 1; }
            tr::apply_guidance(&pos, neg, x, t, p);
        }
        tr::euler_step(x, pos, t, t_prev);
    }
    std::cout << o.steps << " steps in " << ms_since(t0) << " ms\n";
    // slat = z * std + mean
    for (int i = 0; i < N; ++i)
        for (int c = 0; c < C; ++c) x[static_cast<std::size_t>(i) * C + c] = x[static_cast<std::size_t>(i) * C + c] * tr::kShapeStd[c] + tr::kShapeMean[c];
    std::vector<float> cf;
    for (const auto& c : coords) { cf.push_back(static_cast<float>(c.x)); cf.push_back(static_cast<float>(c.y)); cf.push_back(static_cast<float>(c.z)); }
    return save(o.out, "shape_latent_final.npy", x, {static_cast<std::size_t>(N), 32}) && save(o.out, "shape_coords.npy", cf, {static_cast<std::size_t>(N), 3}) ? 0 : 1;
}

// ---------------------------------------------------------------- shape decoder + mesh
int run_shapedec(const std::string& weights, const std::string& dumps, const std::string& out, const std::string& device, bool cpu_check, int cpu_levels) {
    std::string error;
    std::filesystem::create_directories(out);
    std::vector<float> latent;
    std::vector<std::size_t> shape;
    if (!tr::npy::read_f32(std::filesystem::path(dumps) / "shape_latent_final.npy", &latent, &shape, &error) || shape.size() != 2 || shape[1] != 32) {
        std::cerr << "shape_latent_final.npy: " << error << '\n';
        return 1;
    }
    const std::vector<tr::Coord> coords = read_coords((std::filesystem::path(dumps) / "shape_coords.npy").string(), &error);
    if (coords.size() != shape[0]) { std::cerr << "shape_coords.npy: " << error << '\n'; return 1; }
    tr::SparseDecShape ds;
    tr::SparseDecResult res;
    tr::SparseDecOptions opts;
    auto t0 = Clock::now();
    if (device == "gpu") {
#if GUFO_TRELLIS2_HIP
        auto dec = tr::SparseDecHip::Create(weights, ds, &error);
        if (!dec) { std::cerr << error << '\n'; return 1; }
        if (!dec->Forward(latent.data(), coords, &res, &error, opts)) { std::cerr << error << '\n'; return 1; }
        std::cout << "shape decoder (gpu): " << coords.size() << " -> " << res.coords.size() << " voxels in " << ms_since(t0) << " ms, weights "
                  << dec->weight_bytes() / (1024.0 * 1024.0) << " MiB\n";
        if (cpu_check) {
            tr::SparseDecHostWeights host;
            if (!tr::SparseDecHostWeights::FromFile(weights, &host, &error)) { std::cerr << error << '\n'; return 1; }
            tr::SparseDecOptions lim;
            lim.level_limit = cpu_levels > 0 ? cpu_levels : 2;
            lim.capture_levels = true;
            tr::SparseDecResult gpu_taps, cpu_taps;
            t0 = Clock::now();
            if (!dec->Forward(latent.data(), coords, &gpu_taps, &error, lim)) { std::cerr << error << '\n'; return 1; }
            const double gpu_ms = ms_since(t0);
            t0 = Clock::now();
            if (!tr::sparse_dec_forward_cpu(host, latent.data(), coords, &cpu_taps, &error, lim)) { std::cerr << error << '\n'; return 1; }
            std::cout << std::defaultfloat << std::setprecision(6);
            std::cout << "cpu check (" << lim.level_limit << " levels): gpu " << gpu_ms << " ms, cpu " << ms_since(t0) << " ms\n";
            for (std::size_t l = 0; l < cpu_taps.level_feats.size() && l < gpu_taps.level_feats.size(); ++l) {
                const bool same_coords = cpu_taps.level_coords[l] == gpu_taps.level_coords[l];
                std::cout << "  level " << l << ": " << cpu_taps.level_coords[l].size() << " voxels, coords " << (same_coords ? "equal" : "DIFFER")
                          << ", feats rel RMS " << rel_rms(gpu_taps.level_feats[l], cpu_taps.level_feats[l]) << " cosine "
                          << cosine(gpu_taps.level_feats[l], cpu_taps.level_feats[l]) << '\n';
            }
            for (std::size_t l = 0; l < cpu_taps.masks.size() && l < gpu_taps.masks.size(); ++l) {
                std::size_t agree = 0;
                for (std::size_t i = 0; i < cpu_taps.masks[l].size(); ++i) agree += cpu_taps.masks[l][i] == gpu_taps.masks[l][i];
                std::cout << "  mask " << l << ": " << 100.0 * agree / std::max<std::size_t>(1, cpu_taps.masks[l].size()) << " % equal\n";
            }
        }
#else
        std::cerr << "no HIP build\n";
        return 1;
#endif
    } else {
        tr::SparseDecHostWeights host;
        if (!tr::SparseDecHostWeights::FromFile(weights, &host, &error)) { std::cerr << error << '\n'; return 1; }
        if (!tr::sparse_dec_forward_cpu(host, latent.data(), coords, &res, &error, opts)) { std::cerr << error << '\n'; return 1; }
        std::cout << "shape decoder (cpu): " << res.coords.size() << " voxels in " << ms_since(t0) << " ms\n";
    }
    const std::size_t Mv = res.coords.size();
    if (!save(out, "shape_dec_out.npy", res.outputs, {Mv, 7})) return 1;
    std::vector<float> cf;
    for (const auto& c : res.coords) { cf.push_back(static_cast<float>(c.x)); cf.push_back(static_cast<float>(c.y)); cf.push_back(static_cast<float>(c.z)); }
    if (!save(out, "shape_dec_coords.npy", cf, {Mv, 3})) return 1;
    for (std::size_t l = 0; l < res.masks.size(); ++l) {
        std::vector<float> mf(res.masks[l].begin(), res.masks[l].end());
        if (!save(out, "shape_subdiv_" + std::to_string(l) + ".npy", mf, {mf.size() / 8, 8})) return 1;
    }
    // mesh
    std::vector<float> dual, wgt;
    std::vector<std::uint8_t> inter;
    tr::decode_shape_outputs(res.outputs.data(), Mv, &dual, &inter, &wgt);
    std::size_t intersections = 0;
    for (auto v : inter) intersections += v;
    tr::DualGridInput in;
    in.coords = &res.coords; in.dual = dual.data(); in.intersected = inter.data(); in.split_weight = wgt.data();
    in.resolution = 512;
    tr::DualGridStats st;
    t0 = Clock::now();
    const tr::Mesh mesh = tr::dual_grid_to_mesh(in, &st);
    std::cout << "mesh: " << mesh.vertices.size() << " vertices, " << mesh.faces.size() << " triangles from " << st.quads << " quads ("
              << intersections << " intersected edges, " << st.skipped_edges << " skipped), " << st.open_edges << " open edges, "
              << ms_since(t0) << " ms\n";
    tr::GlbWriteOptions go;
    if (!tr::write_glb_file(std::filesystem::path(out) / "mesh.glb", mesh, &error, go)) { std::cerr << error << '\n'; return 1; }
    std::cout << "  wrote mesh.glb\n";
    tr::PreviewOptions po;
    int pw, ph;
    const auto rgb = tr::render_preview(mesh, po, &pw, &ph);
    if (!tr::write_png_rgb(std::filesystem::path(out) / "preview.png", rgb.data(), pw, ph, &error)) { std::cerr << error << '\n'; return 1; }
    std::cout << "  wrote preview.png " << pw << "x" << ph << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << std::setprecision(6) << std::defaultfloat;   // the HIP runtime may leave cout in fixed(2)
    std::string stage, weights, image, out, dumps, device = "gpu";
    int image_size = 512, layers = 0, seed = 42, steps = 12, cpu_levels = 2;
    bool selftest = false, zero_cond = false, cpu_check = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--stage") stage = next();
        else if (a == "--weights") weights = next();
        else if (a == "--image") image = next();
        else if (a == "--out") out = next();
        else if (a == "--dumps") dumps = next();
        else if (a == "--device") device = next();
        else if (a == "--image-size") image_size = std::stoi(next());
        else if (a == "--layers") layers = std::stoi(next());
        else if (a == "--seed") seed = std::stoi(next());
        else if (a == "--steps") steps = std::stoi(next());
        else if (a == "--cpu-levels") cpu_levels = std::stoi(next());
        else if (a == "--selftest-dino") selftest = true;
        else if (a == "--zero-cond") zero_cond = true;
        else if (a == "--cpu-check") cpu_check = true;
        else return usage();
    }
    if (selftest) return selftest_dino(layers > 0 ? layers : 2, image_size);
    if (stage == "dino") {
        if (weights.empty() || image.empty() || out.empty()) return usage();
        return run_dino(weights, image, out, device, image_size);
    }
    if (stage == "ss") {
        if (weights.empty() || out.empty() || (dumps.empty() && !zero_cond)) return usage();
        SsRun o;
        o.weights = weights; o.dumps = dumps.empty() ? out : dumps; o.out = out; o.device = device;
        o.seed = seed; o.steps = steps; o.layers = layers; o.zero_cond = zero_cond; o.cpu_check = cpu_check;
        return run_ss(o);
    }
    if (stage == "ssdec") {
        if (weights.empty() || dumps.empty() || out.empty()) return usage();
        return run_ssdec(weights, dumps, out, device, cpu_check);
    }
    if (stage == "shape") {
        if (weights.empty() || out.empty() || dumps.empty()) return usage();
        SsRun o;
        o.weights = weights; o.dumps = dumps; o.out = out; o.device = device;
        o.seed = seed; o.steps = steps; o.layers = layers; o.cpu_check = cpu_check;
        return run_shape(o);
    }
    if (stage == "shapedec") {
        if (weights.empty() || dumps.empty() || out.empty()) return usage();
        return run_shapedec(weights, dumps, out, device, cpu_check, cpu_levels);
    }
    return usage();
}
