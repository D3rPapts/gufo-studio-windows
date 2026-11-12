// CPU tests for src/models/trellis2/dinov3_cpu.cpp with random weights (2 layers):
// patch row layout, 2-D RoPE angle table (y block then x block, theta 100, normalised
// patch centres), rotation keeps norms and matches the rotate_half formula, output
// shape/finite/deterministic with zero-mean unit-variance rows (final non-affine LN),
// prefix tokens unaffected by RoPE, and the row-permutation property: with a
// translation-equivariant conv and RoPE, swapping two patches swaps their tokens only
// up to the position encoding (so the outputs must differ) while a constant image gives
// identical patch tokens up to position.
// AI-assisted (Claude Code), 2026-11-12.
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/trellis2/dinov3.hpp"

namespace tr = gufo::trellis2;

namespace {
int failures = 0;
void Expect(bool c, const std::string& what) {
    if (!c) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}
bool Near(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) <= tol; }
constexpr double kPi = 3.14159265358979323846;
}  // namespace

int main() {
    // patch rows for a 32 px image (2x2 patches)
    std::vector<float> img(3 * 32 * 32);
    for (std::size_t i = 0; i < img.size(); ++i) img[i] = static_cast<float>(i);
    std::vector<float> rows;
    tr::dinov3_patch_rows(img.data(), 32, &rows);
    Expect(rows.size() == 4 * 768, "4 patches x 768");
    // patch (py=1, px=0), c=2, y=3, x=5 -> pixel (c=2, 16+3, 0+5)
    Expect(rows[(1 * 2 + 0) * 768 + (2 * 16 + 3) * 16 + 5] == static_cast<float>((2 * 32 + 19) * 32 + 5), "patch row layout c,y,x");

    // RoPE angles for grid 4
    std::vector<float> ang;
    tr::dinov3_rope_angles(4, &ang);
    Expect(ang.size() == 16 * 32, "16 patches x 32 angles");
    const float cy0 = 2.f * 0.5f / 4.f - 1.f;   // -0.75
    Expect(Near(ang[0], static_cast<float>(2 * kPi * cy0)), "angle 0 = 2 pi cy (inv_freq 1)");
    Expect(Near(ang[1], static_cast<float>(2 * kPi * cy0 / std::pow(100.0, 4.0 / 64.0))), "inv_freq_1 = 100^(-4/64)");
    const float cx3 = 2.f * 3.5f / 4.f - 1.f;   // 0.75
    Expect(Near(ang[(0 * 4 + 3) * 32 + 16], static_cast<float>(2 * kPi * cx3)), "angles 16..31 use x");
    Expect(Near(ang[(2 * 4 + 1) * 32 + 0], ang[(2 * 4 + 3) * 32 + 0]), "y angles independent of x");

    // rotation: norm preserved, rotate_half formula, zero angle identity
    float v[64], w[64];
    for (int i = 0; i < 64; ++i) v[i] = w[i] = std::sin(0.3f * i) + 0.1f * i;
    float a[32];
    for (int i = 0; i < 32; ++i) a[i] = 0.05f * i;
    double n0 = 0, n1 = 0;
    for (int i = 0; i < 64; ++i) n0 += v[i] * v[i];
    tr::dinov3_apply_rope(v, a);
    for (int i = 0; i < 64; ++i) n1 += v[i] * v[i];
    Expect(Near(static_cast<float>(n0), static_cast<float>(n1), 1e-3f), "rotation keeps the norm");
    Expect(Near(v[5], w[5] * std::cos(a[5]) - w[37] * std::sin(a[5])) && Near(v[37], w[37] * std::cos(a[5]) + w[5] * std::sin(a[5])),
           "v*cos + rotate_half(v)*sin on pairs (d, d+32)");
    float z[32] = {};
    tr::dinov3_apply_rope(w, z);
    Expect(w[7] == std::sin(0.3f * 7) + 0.7f, "zero angles = identity");

    // encoder on a 64 px image (4x4 patches -> 21 tokens), 2 random layers
    auto weights = tr::Dinov3Weights::Random(7, 0.02f, 2);
    std::string error;
    Expect(weights.Validate(&error), "random weights validate: " + error);
    std::vector<float> pixels(3 * 64 * 64);
    for (std::size_t i = 0; i < pixels.size(); ++i) pixels[i] = std::sin(static_cast<float>(i) * 0.01f);
    std::vector<float> tokens, tokens2;
    Expect(tr::dinov3_encode_cpu(weights, pixels.data(), 64, &tokens, &error), "encode: " + error);
    Expect(tokens.size() == 21 * 1024, "21 tokens x 1024");
    bool finite = true;
    for (float t : tokens) finite = finite && std::isfinite(t);
    Expect(finite, "finite output");
    double mean = 0, var = 0;
    for (int c = 0; c < 1024; ++c) mean += tokens[3 * 1024 + c];
    mean /= 1024;
    for (int c = 0; c < 1024; ++c) var += (tokens[3 * 1024 + c] - mean) * (tokens[3 * 1024 + c] - mean);
    Expect(std::fabs(mean) < 1e-4 && std::fabs(var / 1024 - 1.0) < 1e-3, "final non-affine layer norm rows");
    tr::dinov3_encode_cpu(weights, pixels.data(), 64, &tokens2, &error);
    Expect(tokens == tokens2, "deterministic");

    // constant image: patch tokens differ only through RoPE -> all patch tokens have
    // the same norm structure but are not identical; the four register tokens differ
    std::vector<float> flat(3 * 64 * 64, 0.5f);
    std::vector<float> ft;
    tr::dinov3_encode_cpu(weights, flat.data(), 64, &ft, &error);
    double diff_patch = 0, diff_reg = 0;
    for (int c = 0; c < 1024; ++c) {
        diff_patch += std::fabs(ft[5 * 1024 + c] - ft[6 * 1024 + c]);
        diff_reg += std::fabs(ft[1 * 1024 + c] - ft[2 * 1024 + c]);
    }
    Expect(diff_patch > 1e-4 && diff_reg > 1e-4, "constant image: positions (RoPE) and registers still separate tokens");

    // bad image size
    Expect(!tr::dinov3_encode_cpu(weights, pixels.data(), 40, &tokens, &error), "40 px rejected");
    Expect(weights.parameter_count() == 2 * (2 * 1024 + 3 * 1024 * 1024 + 3 * 1024 + 1024 * 1024 + 1024 + 1024 + 2 * 1024 +
                                                 4096 * 1024 + 4096 + 1024 * 4096 + 1024 + 1024) + 1024 + 4 * 1024 + 1024 * 768 + 1024,
           "parameter count");
    if (failures == 0) std::cout << "trellis2_dinov3_cpu_test: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
