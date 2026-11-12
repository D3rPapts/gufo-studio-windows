// gufo-triposg-probe: runs single stages of the TripoSG port for the comparison with
// the PyTorch reference dumps (tools/triposg/vergleich.py).
//
//   gufo-triposg-probe --stage dino --weights triposg.safetensors --image bild.png \
//       --out dumps_gufo/bild [--device cpu|gpu]
//     writes preproc_224.npy [3,224,224], dino_tokens.npy [257,1024], square.png-size info
//   gufo-triposg-probe --selftest [--device gpu]
//     random weights: CPU path vs HIP path (relative RMS), timings
//   gufo-triposg-probe --stage dit --weights F --dumps DIR --out DIR [--device cpu|gpu]
//       [--seed 42] [--steps 0] [--guidance 7] [--only-cond]
//   gufo-triposg-probe --stage vae|mesh --weights F --dumps DIR --out DIR [--device cpu|gpu]
//       [--grid 512] [--coarse 256] [--chunk 32768] [--max-faces 200000]
//     vae: latents_final.npy -> vae_kv.npy (+ sdf_values.npy at sdf_points.npy if present);
//     mesh: additionally coarse grid + band refinement -> mesh.glb, mesh.obj, preview.png
//     DIR holds dino_tokens.npy (ours or the reference's) and optionally dit_in.npy;
//     writes dit_in.npy (if generated), dit_block00/20.npy, dit_v_step00_{cond,uncond}.npy
//     and with --steps N > 0 latents_final.npy after N Euler steps with CFG.
//
// No server, no port; the GPU path needs a HIP build (GUFO_TRIPOSG_HIP).
// AI-assisted (Claude Code), 2026-11-12.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/qwen_image_21/model.hpp"
#include "src/models/triposg/dinov2.hpp"
#include "src/models/triposg/dit.hpp"
#include "src/models/triposg/glb_writer.hpp"
#include "src/models/triposg/marching_cubes.hpp"
#include "src/models/triposg/mesh_decimate.hpp"
#include "src/models/triposg/mesh_preview.hpp"
#include "src/models/triposg/npy.hpp"
#include "src/models/triposg/preprocess.hpp"
#include "src/models/triposg/scheduler.hpp"
#include "src/models/triposg/sdf_grid.hpp"
#include "src/models/triposg/vae.hpp"

namespace {

using Clock = std::chrono::steady_clock;

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

int usage() {
    std::cerr << "usage: gufo-triposg-probe --stage dino --weights F --image PNG --out DIR [--device cpu|gpu]\n"
                 "       gufo-triposg-probe --stage dit --weights F --dumps DIR --out DIR [--device cpu|gpu]\n"
                 "                          [--seed N] [--steps N] [--guidance G] [--only-cond]\n"
                 "       gufo-triposg-probe --selftest [--device gpu] [--layers N]\n";
    return 2;
}

std::vector<float> gaussian_noise(std::size_t count, std::uint64_t seed) {
    std::vector<float> out(count);
    std::uint64_t state = seed * 0x9E3779B97F4A7C15ULL + 0x1234567ULL;
    auto uniform = [&]() {
        state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
        return (static_cast<double>((state * 0x2545F4914F6CDD1DULL) >> 11) + 0.5) / 9007199254740992.0;
    };
    for (std::size_t i = 0; i + 1 < count; i += 2) {
        const double u1 = uniform(), u2 = uniform();
        const double r = std::sqrt(-2.0 * std::log(u1));
        out[i] = static_cast<float>(r * std::cos(6.283185307179586 * u2));
        out[i + 1] = static_cast<float>(r * std::sin(6.283185307179586 * u2));
    }
    if (count % 2) out[count - 1] = static_cast<float>(std::sqrt(-2.0 * std::log(uniform())) * std::cos(6.283185307179586 * uniform()));
    return out;
}

struct DitRunner {
    std::unique_ptr<gufo::triposg::DitHostWeights> cpu;
#if GUFO_TRIPOSG_HIP
    std::unique_ptr<gufo::triposg::DitHip> gpu;
#endif
    bool forward(const std::vector<float>& x, const std::vector<float>& cond, float t, std::vector<float>* v,
                 std::map<int, std::vector<float>>* taps, const gufo::triposg::DitForwardOptions& opt, std::string* error) {
        const int tokens = static_cast<int>(x.size() / 64), cond_tokens = static_cast<int>(cond.size() / 1024);
#if GUFO_TRIPOSG_HIP
        if (gpu) return gpu->Forward(x.data(), tokens, cond.data(), cond_tokens, t, v, taps, error, opt);
#endif
        return gufo::triposg::dit_forward_cpu(*cpu, x.data(), tokens, cond.data(), cond_tokens, t, v, taps, error, opt);
    }
};

// --stage vae: KV cache and logits at the reference point set (CPU or GPU);
// --stage mesh: dense grid -> marching cubes -> GLB/OBJ + preview PNG (GPU).
int run_vae(const std::string& weights, const std::string& dumps, const std::string& out, const std::string& device,
            const std::string& stage, int grid_n, int coarse_n, int chunk, int max_faces) {
    namespace tp = gufo::triposg;
    std::string error;
    std::vector<float> z;
    std::vector<std::size_t> shape;
    if (!tp::npy::read_f32(std::filesystem::path(dumps) / "latents_final.npy", &z, &shape, &error) || z.size() != 2048u * 64) {
        std::cerr << "latents_final.npy: " << error << '\n';
        return 1;
    }
    std::filesystem::create_directories(out);
    auto t0 = Clock::now();
    std::unique_ptr<tp::VaeHostWeights> cpu;
#if GUFO_TRIPOSG_HIP
    std::unique_ptr<tp::VaeHip> gpu;
#endif
    std::vector<float> kv;
    if (device == "gpu") {
#if GUFO_TRIPOSG_HIP
        gpu = tp::VaeHip::Create(weights, &error);
        if (!gpu || !gpu->Prepare(z.data(), 2048, &error)) {
            std::cerr << "gpu: " << error << '\n';
            return 1;
        }
        kv = gpu->DownloadKv();
#else
        std::cerr << "gpu path not built\n";
        return 1;
#endif
    } else {
        cpu = std::make_unique<tp::VaeHostWeights>();
        if (!tp::VaeHostWeights::FromFile(weights, cpu.get(), &error) || !tp::vae_kv_cpu(*cpu, z.data(), 2048, &kv, &error)) {
            std::cerr << "cpu: " << error << '\n';
            return 1;
        }
    }
    std::cout << "vae (" << device << "): kv cache [2048,1024] in " << ms_since(t0) << " ms, kv[0..3] = " << kv[0] << ' ' << kv[1] << ' ' << kv[2] << '\n';
    tp::npy::write_f32(std::filesystem::path(out) / "vae_kv.npy", kv.data(), {2048, 1024}, &error);
    auto query = [&](const float* points, int count, std::vector<float>* logits) {
#if GUFO_TRIPOSG_HIP
        if (gpu) return gpu->Query(points, count, logits, &error, chunk);
#endif
        return tp::vae_query_cpu(*cpu, kv.data(), 2048, points, count, logits, &error);
    };
    const auto points_path = std::filesystem::path(dumps) / "sdf_points.npy";
    if (std::filesystem::exists(points_path)) {
        std::vector<float> points, logits;
        if (!tp::npy::read_f32(points_path, &points, &shape, &error) || shape.size() != 2 || shape[1] != 3) {
            std::cerr << "sdf_points.npy: " << error << '\n';
            return 1;
        }
        t0 = Clock::now();
        if (!query(points.data(), static_cast<int>(shape[0]), &logits)) {
            std::cerr << "query: " << error << '\n';
            return 1;
        }
        std::size_t inside = 0;
        for (float l : logits) inside += l > 0.f;
        std::cout << "sdf: " << shape[0] << " points in " << ms_since(t0) << " ms, inside " << inside << '\n';
        tp::npy::write_f32(std::filesystem::path(out) / "sdf_values.npy", logits.data(), {shape[0]}, &error);
    }
    if (stage != "mesh") return 0;

    // Hierarchical grid: dense coarse level, band refinement up to grid_n.
    tp::SdfGridOptions go;
    go.levels.clear();
    for (int n = coarse_n; n <= grid_n; n *= 2) go.levels.push_back(n);
    go.chunk = static_cast<std::size_t>(chunk);
    go.progress = [&](const std::string& what) { std::cout << "  " << what << " after " << ms_since(t0) / 1000 << " s\n"; };
    tp::SdfGridResult gr;
    t0 = Clock::now();
    if (!tp::extract_sdf_grid([&](const float* p, int n, std::vector<float>* v, std::string* e) {
            if (!query(p, n, v)) { if (e) *e = error; return false; }
            return true;
        }, go, &gr, &error)) {
        std::cerr << "grid: " << error << '\n';
        return 1;
    }
    std::size_t inside = 0;
    for (auto h : gr.half) inside += tp::ScalarGrid::half_to_float(h) < 0.f;
    std::cout << "grid " << gr.n << "^3 in " << ms_since(t0) / 1000 << " s, queries";
    for (auto q : gr.queried_per_level) std::cout << ' ' << q;
    std::cout << ", inside (negative) " << 100.0 * static_cast<double>(inside) / static_cast<double>(gr.half.size()) << " %\n";
    {
        std::vector<float> coarse;  // the final grid is large; save a float copy only for small grids
        if (gr.n <= 256) {
            coarse.resize(gr.half.size());
            for (std::size_t i = 0; i < coarse.size(); ++i) coarse[i] = tp::ScalarGrid::half_to_float(gr.half[i]);
            tp::npy::write_f32(std::filesystem::path(out) / ("grid" + std::to_string(gr.n) + ".npy"), coarse.data(),
                               {static_cast<std::size_t>(gr.n), static_cast<std::size_t>(gr.n), static_cast<std::size_t>(gr.n)}, &error);
        }
    }

    t0 = Clock::now();
    tp::ScalarGrid sg;
    sg.half_values = gr.half.data();
    sg.nx = sg.ny = sg.nz = static_cast<std::size_t>(gr.n);
    const float b = tp::GeometryConfig::bounds;
    for (int a = 0; a < 3; ++a) { sg.min[a] = -b; sg.max[a] = b; }
    // Measured on the mug latents: the decoder output (-proj_out) is NEGATIVE inside the
    // shape and positive outside, i.e. SDF-like.
    tp::MarchingCubesOptions mo;
    mo.inside_positive = false;
    tp::Mesh mesh = tp::marching_cubes(sg, mo);
    gr.half.clear();
    gr.half.shrink_to_fit();
    float bmin[3] = {0, 0, 0}, bmax[3] = {0, 0, 0};
    tp::bounding_box(mesh, bmin, bmax);
    std::cout << "mesh: " << mesh.vertices.size() << " vertices, " << mesh.faces.size() << " faces, open edges "
              << tp::count_open_edges(mesh) << ", volume " << tp::signed_volume(mesh) << ", bbox [" << bmin[0] << ' ' << bmin[1]
              << ' ' << bmin[2] << "] .. [" << bmax[0] << ' ' << bmax[1] << ' ' << bmax[2] << "] in " << ms_since(t0) << " ms\n";
    if (max_faces > 0 && mesh.faces.size() > static_cast<std::size_t>(max_faces)) {
        t0 = Clock::now();
        tp::DecimateOptions dopt;
        dopt.target_faces = static_cast<std::size_t>(max_faces);
        mesh = tp::decimate(mesh, dopt);
        std::cout << "decimated: " << mesh.vertices.size() << " vertices, " << mesh.faces.size() << " faces, open edges "
                  << tp::count_open_edges(mesh) << ", volume " << tp::signed_volume(mesh) << " in " << ms_since(t0) << " ms\n";
    }
    if (!tp::write_glb_file(std::filesystem::path(out) / "mesh.glb", mesh, &error)) {
        std::cerr << error << '\n';
        return 1;
    }
    {
        std::ofstream obj(std::filesystem::path(out) / "mesh.obj");
        obj << tp::write_obj(mesh);
    }
    int pw = 0, ph = 0;
    const auto rgb = tp::render_preview(mesh, {}, &pw, &ph);
    if (!tp::write_png_rgb(std::filesystem::path(out) / "preview.png", rgb.data(), pw, ph, &error)) {
        std::cerr << error << '\n';
        return 1;
    }
    std::cout << "wrote " << out << "/mesh.glb, mesh.obj, preview.png (" << pw << "x" << ph << ")\n";
    return 0;
}

int run_dit(const std::string& weights, const std::string& dumps, const std::string& out, const std::string& device,
            std::uint64_t seed, int steps, float guidance, bool only_cond, bool trace, float t_override) {
    namespace tp = gufo::triposg;
    std::string error;
    std::vector<float> cond, x;
    std::vector<std::size_t> shape;
    if (!tp::npy::read_f32(std::filesystem::path(dumps) / "dino_tokens.npy", &cond, &shape, &error) || shape.size() != 2 ||
        shape[0] != 257 || shape[1] != 1024) {
        std::cerr << "dino_tokens.npy: " << error << '\n';
        return 1;
    }
    std::filesystem::create_directories(out);
    const auto noise_path = std::filesystem::path(dumps) / "dit_in.npy";
    if (std::filesystem::exists(noise_path)) {
        if (!tp::npy::read_f32(noise_path, &x, &shape, &error) || x.size() != 2048u * 64) {
            std::cerr << "dit_in.npy: " << error << '\n';
            return 1;
        }
        std::cout << "noise: " << noise_path.string() << '\n';
    } else {
        x = gaussian_noise(2048u * 64, seed);
        tp::npy::write_f32(std::filesystem::path(out) / "dit_in.npy", x.data(), {2048, 64}, &error);
        std::cout << "noise: generated with seed " << seed << " (not torch-identical)\n";
    }
    DitRunner runner;
    auto t0 = Clock::now();
    if (device == "gpu") {
#if GUFO_TRIPOSG_HIP
        runner.gpu = tp::DitHip::Create(weights, &error);
        if (!runner.gpu) {
            std::cerr << "gpu: " << error << '\n';
            return 1;
        }
#else
        std::cerr << "gpu path not built\n";
        return 1;
#endif
    } else {
        runner.cpu = std::make_unique<tp::DitHostWeights>();
        if (!tp::DitHostWeights::FromFile(weights, runner.cpu.get(), &error)) {
            std::cerr << "weights: " << error << '\n';
            return 1;
        }
    }
    std::cout << "dit (" << device << "): weights mapped in " << ms_since(t0) << " ms\n";
    const std::vector<float> sigmas = tp::flow_sigmas(steps > 0 ? steps : 50);
    const std::vector<float> zeros(cond.size(), 0.f);
    tp::DitForwardOptions taps_opt;
    taps_opt.tap_layers = {0, 20};
    std::map<int, std::vector<float>> taps;
    std::vector<float> v_cond, v_uncond;
    const float t_first = t_override >= 0.f ? t_override : tp::flow_timestep(sigmas, 0);
    t0 = Clock::now();
    if (!runner.forward(x, cond, t_first, &v_cond, &taps, taps_opt, &error)) {
        std::cerr << "forward: " << error << '\n';
        return 1;
    }
    std::cout << "step 0 cond: " << ms_since(t0) << " ms, v[0..3] = " << v_cond[0] << ' ' << v_cond[1] << ' ' << v_cond[2] << '\n';
    tp::npy::write_f32(std::filesystem::path(out) / "dit_block00.npy", taps[0].data(), {2049, 2048}, &error);
    tp::npy::write_f32(std::filesystem::path(out) / "dit_block20.npy", taps[20].data(), {2049, 2048}, &error);
    tp::npy::write_f32(std::filesystem::path(out) / "dit_v_step00_cond.npy", v_cond.data(), {2048, 64}, &error);
    if (!only_cond) {
        t0 = Clock::now();
        if (!runner.forward(x, zeros, t_first, &v_uncond, nullptr, {}, &error)) {
            std::cerr << "forward: " << error << '\n';
            return 1;
        }
        std::cout << "step 0 uncond: " << ms_since(t0) << " ms\n";
        tp::npy::write_f32(std::filesystem::path(out) / "dit_v_step00_uncond.npy", v_uncond.data(), {2048, 64}, &error);
    }
    if (steps > 0) {
        if (only_cond) {
            std::cerr << "--steps needs the unconditional branch\n";
            return 1;
        }
        t0 = Clock::now();
        const std::vector<float> noise = x;
        for (int i = 0; i < steps; ++i) {
            const float t = tp::flow_timestep(sigmas, i);
            if (i > 0) {
                if (!runner.forward(x, cond, t, &v_cond, nullptr, {}, &error) ||
                    !runner.forward(x, zeros, t, &v_uncond, nullptr, {}, &error)) {
                    std::cerr << "step " << i << ": " << error << '\n';
                    return 1;
                }
            }
            // Trace: how the guided velocity relates to the start noise and the sample.
            double vv = 0, vn = 0, nn = 0, xx = 0, xn = 0, cu = 0, cc = 0, uu = 0;
            for (std::size_t k = 0; k < x.size(); ++k) {
                const double n = noise[k], c = v_cond[k], u = v_uncond[k];
                nn += n * n; xx += static_cast<double>(x[k]) * x[k]; xn += x[k] * n;
                cc += c * c; uu += u * u; cu += c * u; vn += c * n; vv += c * c;
            }
            tp::apply_guidance(&v_cond, v_uncond, guidance);
            tp::euler_step(x, v_cond, sigmas, i);
            if (trace)
                std::cout << "  t=" << t << " rms(v_cond) " << std::sqrt(cc / x.size()) << " corr(v_cond,noise) " << vn / std::sqrt(vv * nn)
                          << " corr(cond,uncond) " << cu / std::sqrt(cc * uu) << " rms(x) " << std::sqrt(xx / x.size())
                          << " corr(x,noise) " << xn / std::sqrt(xx * nn) << '\n';
            else if (i % 10 == 9 || i + 1 == steps)
                std::cout << "  step " << i + 1 << "/" << steps << " after " << ms_since(t0) / 1000 << " s\n";
        }
        tp::npy::write_f32(std::filesystem::path(out) / "latents_final.npy", x.data(), {2048, 64}, &error);
        double energy = 0;
        for (float f : x) energy += static_cast<double>(f) * f;
        std::cout << "latents_final: " << steps << " steps, guidance " << guidance << ", rms " << std::sqrt(energy / x.size()) << '\n';
    }
    std::cout << "wrote " << out << "/dit_*.npy\n";
    return 0;
}

int selftest(const std::string& device, int layers) {
    gufo::triposg::Dinov2Weights w = gufo::triposg::Dinov2Weights::Random(7, 0.02f, layers);
    std::vector<float> pixels(3 * 224 * 224);
    for (std::size_t i = 0; i < pixels.size(); ++i) pixels[i] = std::sin(static_cast<float>(i) * 0.013f) * 1.5f;
    std::vector<float> cpu;
    std::string error;
    auto t0 = Clock::now();
    if (!gufo::triposg::dinov2_encode_cpu(w, pixels.data(), &cpu, &error)) {
        std::cerr << "cpu: " << error << '\n';
        return 1;
    }
    std::cout << "cpu: " << layers << " layers in " << ms_since(t0) << " ms, token0[0..3] = " << cpu[0] << ' '
              << cpu[1] << ' ' << cpu[2] << '\n';
    if (device != "gpu") return 0;
#if GUFO_TRIPOSG_HIP
    t0 = Clock::now();
    auto hip = gufo::triposg::Dinov2Hip::Create(w, &error);
    if (!hip) {
        std::cerr << "gpu: " << error << '\n';
        return 1;
    }
    std::cout << "gpu: upload " << hip->device_bytes() / (1 << 20) << " MiB in " << ms_since(t0) << " ms\n";
    std::vector<float> gpu;
    for (int run = 0; run < 3; ++run) {
        t0 = Clock::now();
        if (!hip->Encode(pixels.data(), &gpu, &error)) {
            std::cerr << "gpu: " << error << '\n';
            return 1;
        }
        std::cout << "gpu: encode " << ms_since(t0) << " ms\n";
    }
    const double e = rel_rms(gpu, cpu);
    std::cout << "gpu vs cpu rel RMS " << e << (e < 1e-4 ? "  ok" : "  MISMATCH") << '\n';
    return e < 1e-4 ? 0 : 1;
#else
    std::cerr << "gpu path not built (GUFO_TRIPOSG_HIP)\n";
    return 1;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    std::string stage = "dino", weights, image, out, device = "cpu", dumps;
    bool self = false, only_cond = false, trace = false;
    float t_override = -1.f;
    int layers = 24, steps = 0, grid_n = 512, coarse_n = 256, chunk = 32768, max_faces = 200000;
    std::uint64_t seed = 42;
    float guidance = 7.f;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--stage") stage = next();
        else if (a == "--weights") weights = next();
        else if (a == "--image") image = next();
        else if (a == "--out") out = next();
        else if (a == "--device") device = next();
        else if (a == "--layers") layers = std::stoi(next());
        else if (a == "--selftest") self = true;
        else if (a == "--dumps") dumps = next();
        else if (a == "--seed") seed = std::stoull(next());
        else if (a == "--steps") steps = std::stoi(next());
        else if (a == "--guidance") guidance = std::stof(next());
        else if (a == "--only-cond") only_cond = true;
        else if (a == "--trace") trace = true;
        else if (a == "--t") t_override = std::stof(next());
        else if (a == "--grid") grid_n = std::stoi(next());
        else if (a == "--coarse") coarse_n = std::stoi(next());
        else if (a == "--max-faces") max_faces = std::stoi(next());
        else if (a == "--chunk") chunk = std::stoi(next());
        else return usage();
    }
    if (self) return selftest(device, layers);
    if (stage == "dit") {
        if (weights.empty() || dumps.empty() || out.empty()) return usage();
        return run_dit(weights, dumps, out, device, seed, steps, guidance, only_cond, trace, t_override);
    }
    if (stage == "vae" || stage == "mesh") {
        if (weights.empty() || dumps.empty() || out.empty()) return usage();
        return run_vae(weights, dumps, out, device, stage, grid_n, coarse_n, chunk, max_faces);
    }
    if (stage != "dino" || weights.empty() || image.empty() || out.empty()) return usage();

    std::string error;
    auto t0 = Clock::now();
    gufo::models::qwen_image_21::Image rgba;
    try {
        rgba = gufo::models::qwen_image_21::DecodeImage(gufo::core::ReadImageFile(image));
    } catch (const std::exception& e) {
        std::cerr << "image: " << e.what() << '\n';
        return 1;
    }
    gufo::triposg::Preprocessed pre;
    if (!gufo::triposg::preprocess_rgba(rgba.rgba.data(), rgba.width, rgba.height, &pre, &error)) {
        std::cerr << "preprocess: " << error << '\n';
        return 1;
    }
    std::cout << "preprocess: " << rgba.width << "x" << rgba.height << " bbox " << pre.bbox[0] << "," << pre.bbox[1]
              << " " << pre.bbox[2] << "x" << pre.bbox[3] << " square " << pre.square_width << "x" << pre.square_height
              << " in " << ms_since(t0) << " ms\n";
    std::filesystem::create_directories(out);
    if (!gufo::triposg::npy::write_f32(std::filesystem::path(out) / "preproc_224.npy", pre.pixels.data(), {3, 224, 224}, &error)) {
        std::cerr << error << '\n';
        return 1;
    }

    t0 = Clock::now();
    gufo::triposg::Dinov2Weights w;
    if (!gufo::triposg::Dinov2Weights::Load(weights, &w, &error)) {
        std::cerr << "weights: " << error << '\n';
        return 1;
    }
    std::cout << "weights: " << w.parameter_count() << " parameters in " << ms_since(t0) << " ms\n";
    std::vector<float> tokens;
    t0 = Clock::now();
    if (device == "gpu") {
#if GUFO_TRIPOSG_HIP
        auto hip = gufo::triposg::Dinov2Hip::Create(w, &error);
        if (!hip || !hip->Encode(pre.pixels.data(), &tokens, &error)) {
            std::cerr << "gpu: " << error << '\n';
            return 1;
        }
#else
        std::cerr << "gpu path not built\n";
        return 1;
#endif
    } else if (!gufo::triposg::dinov2_encode_cpu(w, pre.pixels.data(), &tokens, &error)) {
        std::cerr << "cpu: " << error << '\n';
        return 1;
    }
    std::cout << "dino (" << device << "): 257x1024 in " << ms_since(t0) << " ms, cls[0..3] = " << tokens[0] << ' '
              << tokens[1] << ' ' << tokens[2] << '\n';
    if (!gufo::triposg::npy::write_f32(std::filesystem::path(out) / "dino_tokens.npy", tokens.data(), {257, 1024}, &error)) {
        std::cerr << error << '\n';
        return 1;
    }
    std::cout << "wrote " << out << "/preproc_224.npy and dino_tokens.npy\n";
    return 0;
}
