// gufo-triposg-probe: runs single stages of the TripoSG port for the comparison with
// the PyTorch reference dumps (tools/triposg/vergleich.py).
//
//   gufo-triposg-probe --stage dino --weights triposg.safetensors --image bild.png \
//       --out dumps_gufo/bild [--device cpu|gpu]
//     writes preproc_224.npy [3,224,224], dino_tokens.npy [257,1024], square.png-size info
//   gufo-triposg-probe --selftest [--device gpu]
//     random weights: CPU path vs HIP path (relative RMS), timings
//
// No server, no port; the GPU path needs a HIP build (GUFO_TRIPOSG_HIP).
// AI-assisted (Claude Code), 2026-11-12.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/qwen_image_21/model.hpp"
#include "src/models/triposg/dinov2.hpp"
#include "src/models/triposg/npy.hpp"
#include "src/models/triposg/preprocess.hpp"

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
                 "       gufo-triposg-probe --selftest [--device gpu] [--layers N]\n";
    return 2;
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
    std::string stage = "dino", weights, image, out, device = "cpu";
    bool self = false;
    int layers = 24;
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
        else return usage();
    }
    if (self) return selftest(device, layers);
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
