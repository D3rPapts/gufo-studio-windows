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
//     DIR holds dino_tokens.npy (ours or the reference's) and optionally dit_in.npy;
//     writes dit_in.npy (if generated), dit_block00/20.npy, dit_v_step00_{cond,uncond}.npy
//     and with --steps N > 0 latents_final.npy after N Euler steps with CFG.
//
// No server, no port; the GPU path needs a HIP build (GUFO_TRIPOSG_HIP).
// AI-assisted (Claude Code), 2026-11-12.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/qwen_image_21/model.hpp"
#include "src/models/triposg/dinov2.hpp"
#include "src/models/triposg/dit.hpp"
#include "src/models/triposg/npy.hpp"
#include "src/models/triposg/preprocess.hpp"
#include "src/models/triposg/scheduler.hpp"

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

int run_dit(const std::string& weights, const std::string& dumps, const std::string& out, const std::string& device,
            std::uint64_t seed, int steps, float guidance, bool only_cond) {
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
    t0 = Clock::now();
    if (!runner.forward(x, cond, tp::flow_timestep(sigmas, 0), &v_cond, &taps, taps_opt, &error)) {
        std::cerr << "forward: " << error << '\n';
        return 1;
    }
    std::cout << "step 0 cond: " << ms_since(t0) << " ms, v[0..3] = " << v_cond[0] << ' ' << v_cond[1] << ' ' << v_cond[2] << '\n';
    tp::npy::write_f32(std::filesystem::path(out) / "dit_block00.npy", taps[0].data(), {2049, 2048}, &error);
    tp::npy::write_f32(std::filesystem::path(out) / "dit_block20.npy", taps[20].data(), {2049, 2048}, &error);
    tp::npy::write_f32(std::filesystem::path(out) / "dit_v_step00_cond.npy", v_cond.data(), {2048, 64}, &error);
    if (!only_cond) {
        t0 = Clock::now();
        if (!runner.forward(x, zeros, tp::flow_timestep(sigmas, 0), &v_uncond, nullptr, {}, &error)) {
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
        for (int i = 0; i < steps; ++i) {
            const float t = tp::flow_timestep(sigmas, i);
            if (i > 0) {
                if (!runner.forward(x, cond, t, &v_cond, nullptr, {}, &error) ||
                    !runner.forward(x, zeros, t, &v_uncond, nullptr, {}, &error)) {
                    std::cerr << "step " << i << ": " << error << '\n';
                    return 1;
                }
            }
            tp::apply_guidance(&v_cond, v_uncond, guidance);
            tp::euler_step(x, v_cond, sigmas, i);
            if (i % 10 == 9 || i + 1 == steps) std::cout << "  step " << i + 1 << "/" << steps << " after " << ms_since(t0) / 1000 << " s\n";
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
    bool self = false, only_cond = false;
    int layers = 24, steps = 0;
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
        else return usage();
    }
    if (self) return selftest(device, layers);
    if (stage == "dit") {
        if (weights.empty() || dumps.empty() || out.empty()) return usage();
        return run_dit(weights, dumps, out, device, seed, steps, guidance, only_cond);
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
