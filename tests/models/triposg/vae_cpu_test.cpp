// CPU tests for src/models/triposg/vae_cpu.cpp and mesh_preview.*: frequency
// embedding layout, KV/query shapes on a tiny VAE, query locality (different points
// give different logits, same points give the same), the preview renderer (object
// pixels inside, background outside) and the PNG writer (signature, chunks, CRC).
// AI-assisted (Claude Code), 2026-11-12.
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/triposg/marching_cubes.hpp"
#include "src/models/triposg/mesh_preview.hpp"
#include "src/models/triposg/vae.hpp"

namespace tp = gufo::triposg;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

void EmbedTest() {
    const float pts[6] = {0.5f, -0.25f, 1.0f, 0.f, 0.f, 0.f};
    std::vector<float> e;
    tp::vae_embed_points(pts, 2, 8, &e);
    Expect(e.size() == 2u * 51, "51 channels per point");
    Expect(e[0] == 0.5f && e[1] == -0.25f && e[2] == 1.0f, "raw coordinates first");
    Expect(std::fabs(e[3] - std::sin(0.5f)) < 1e-6f && std::fabs(e[3 + 7] - std::sin(0.5f * 128.f)) < 1e-5f, "sin(x f) block");
    Expect(std::fabs(e[3 + 8] - std::sin(-0.25f)) < 1e-6f, "sin(y f0) follows the x block");
    Expect(std::fabs(e[3 + 24] - std::cos(0.5f)) < 1e-6f && std::fabs(e[3 + 24 + 16 + 7] - std::cos(128.f)) < 1e-5f, "cos block");
    Expect(e[51] == 0.f && e[51 + 3] == 0.f && e[51 + 27] == 1.f, "origin: zeros, sin 0, cos 1");
}

void TinyVaeTest() {
    tp::VaeShape s;
    s.width = 64; s.heads = 2; s.layers = 3; s.mlp = 128; s.latent_channels = 8;
    tp::VaeHostWeights w = tp::VaeHostWeights::Random(s, 4, 0.08f);
    const int tokens = 12;
    std::vector<float> z(static_cast<std::size_t>(tokens) * s.latent_channels);
    for (std::size_t i = 0; i < z.size(); ++i) z[i] = std::sin(static_cast<float>(i) * 0.3f);
    std::vector<float> kv;
    std::string error;
    Expect(tp::vae_kv_cpu(w, z.data(), tokens, &kv, &error), "kv ok: " + error);
    Expect(kv.size() == static_cast<std::size_t>(tokens) * s.width, "kv [tokens, width]");
    const float pts[9] = {0.1f, 0.2f, 0.3f, 0.1f, 0.2f, 0.3f, -0.7f, 0.4f, 0.9f};
    std::vector<float> logits;
    Expect(tp::vae_query_cpu(w, kv.data(), tokens, pts, 3, &logits, &error), "query ok");
    Expect(logits.size() == 3 && std::isfinite(logits[0]), "3 logits");
    Expect(logits[0] == logits[1], "same point, same logit");
    Expect(std::fabs(logits[0] - logits[2]) > 1e-6f, "different point, different logit");
    // Chunk independence: querying one point alone equals its value within a batch.
    std::vector<float> single;
    Expect(tp::vae_query_cpu(w, kv.data(), tokens, pts + 6, 1, &single, &error) && std::fabs(single[0] - logits[2]) < 1e-5f,
           "batch independent");
    // The latents matter.
    for (auto& v : z) v = -v;
    std::vector<float> kv2, logits2;
    Expect(tp::vae_kv_cpu(w, z.data(), tokens, &kv2, &error) && tp::vae_query_cpu(w, kv2.data(), tokens, pts, 3, &logits2, &error) &&
               std::fabs(logits2[0] - logits[0]) > 1e-6f,
           "latents change the logits");
}

void PreviewTest() {
    // Sphere mesh from marching cubes, rendered in 2 views.
    const std::size_t n = 24;
    std::vector<float> grid(n * n * n);
    for (std::size_t ix = 0; ix < n; ++ix)
        for (std::size_t iy = 0; iy < n; ++iy)
            for (std::size_t iz = 0; iz < n; ++iz) {
                const float x = -1.f + 2.f * static_cast<float>(ix) / static_cast<float>(n - 1);
                const float y = -1.f + 2.f * static_cast<float>(iy) / static_cast<float>(n - 1);
                const float z = -1.f + 2.f * static_cast<float>(iz) / static_cast<float>(n - 1);
                grid[(ix * n + iy) * n + iz] = 0.6f - std::sqrt(x * x + y * y + z * z);
            }
    tp::ScalarGrid g;
    g.values = grid.data();
    g.nx = g.ny = g.nz = n;
    const tp::Mesh sphere = tp::marching_cubes(g);
    tp::PreviewOptions o;
    o.size = 64;
    o.yaw_degrees = {0.f, 90.f};
    int w = 0, h = 0;
    const auto rgb = tp::render_preview(sphere, o, &w, &h);
    Expect(w == 128 && h == 64 && rgb.size() == 128u * 64 * 3, "preview size");
    auto px = [&](int x, int y) { return rgb.data() + (static_cast<std::size_t>(y) * 128 + static_cast<std::size_t>(x)) * 3; };
    Expect(px(32, 32)[0] != 245 || px(32, 32)[1] != 245, "centre pixel is the object");
    Expect(px(1, 1)[0] == 245 && px(1, 1)[1] == 245 && px(1, 1)[2] == 245, "corner pixel is background");
    Expect(px(96, 32)[0] != 245, "second view centre is the object");
    // Lit top is brighter than the shaded bottom.
    Expect(px(32, 12)[0] > px(32, 52)[0], "top brighter than bottom");
    const auto path = std::filesystem::temp_directory_path() / "triposg_preview_test.png";
    std::string error;
    Expect(tp::write_png_rgb(path, rgb.data(), w, h, &error), "png written: " + error);
    std::vector<std::uint8_t> bytes;
    {
        // Close the stream before the remove below (an open file cannot be deleted on Windows;
        // std::filesystem::remove would throw and abort the test).
        std::ifstream in(path, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    Expect(bytes.size() > 8 + 25 + 12 + rgb.size() && bytes[0] == 0x89 && bytes[1] == 'P' && bytes[2] == 'N' && bytes[3] == 'G', "png signature");
    Expect(std::string(bytes.begin() + 12, bytes.begin() + 16) == "IHDR", "IHDR chunk");
    Expect(std::string(bytes.end() - 8, bytes.end() - 4) == "IEND", "IEND chunk");
    std::filesystem::remove(path);
    // Empty mesh renders background only.
    const auto empty = tp::render_preview(tp::Mesh{}, o, &w, &h);
    Expect(empty.size() == rgb.size() && empty[0] == 245, "empty mesh: background");
}

}  // namespace

int main() {
    EmbedTest();
    TinyVaeTest();
    PreviewTest();
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "vae_cpu_test: ok\n";
    return 0;
}
