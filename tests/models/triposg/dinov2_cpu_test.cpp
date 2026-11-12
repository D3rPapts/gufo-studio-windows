// CPU tests for src/models/triposg/dinov2_cpu.cpp with random weights: shapes and
// validation, patch-row layout, the LayerScale-zero identity (blocks disabled ->
// tokens = LayerNorm(embeddings)), finite outputs through 2 real blocks, and the
// .npy round trip used by the probe tool.
// AI-assisted (Claude Code), 2026-11-12.
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/triposg/dinov2.hpp"
#include "src/models/triposg/npy.hpp"

namespace tp = gufo::triposg;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

std::vector<float> Pixels() {
    std::vector<float> p(3 * 224 * 224);
    for (std::size_t i = 0; i < p.size(); ++i) p[i] = std::sin(static_cast<float>(i) * 0.017f) * 2.f;
    return p;
}

void ShapesTest() {
    tp::Dinov2Weights w = tp::Dinov2Weights::Random(3, 0.02f, 2);
    std::string error;
    Expect(w.Validate(&error), "random weights validate: " + error);
    Expect(w.layers.size() == 2, "two layers");
    const std::size_t per_layer = 2 * 1024 + 3 * 1024 * 1024 + 3 * 1024 + 1024 * 1024 + 1024 + 1024 + 2 * 1024 +
                                  4096 * 1024 + 4096 + 1024 * 4096 + 1024 + 1024;
    const std::size_t stem = 1024 + 1024 * 588 + 1024 + 257 * 1024 + 2 * 1024;
    Expect(w.parameter_count() == stem + 2 * per_layer, "parameter count " + std::to_string(w.parameter_count()));
    tp::Dinov2Weights full = tp::Dinov2Weights::Random(1, 0.02f, 24);
    // Real model: 304,368,640 parameters minus mask_token (1024) and the 1370-row position
    // table, plus the baked 257-row table = 303,227,904 (24 x 12,598,272 + 869,376).
    const std::size_t expected = 24 * per_layer + stem;
    Expect(expected == 303227904u, "expected count matches the HF model");
    Expect(full.parameter_count() == expected, "24-layer parameter count " + std::to_string(full.parameter_count()));
    w.layers[1].fc2_b.pop_back();
    Expect(!w.Validate(&error), "broken layer fails validation");
}

void PatchRowsTest() {
    std::vector<float> pixels(3 * 224 * 224);
    for (std::size_t i = 0; i < pixels.size(); ++i) pixels[i] = static_cast<float>(i);
    std::vector<float> rows;
    tp::dinov2_patch_rows(pixels.data(), &rows);
    Expect(rows.size() == 256u * 588, "256 rows x 588");
    // Patch (py=1, px=2), channel 1, y=3, x=4 -> pixel index (1*224 + 14+3)*224 + 28+4.
    const float want = static_cast<float>((1 * 224 + 14 + 3) * 224 + 28 + 4);
    Expect(rows[(1 * 16 + 2) * 588 + (1 * 14 + 3) * 14 + 4] == want, "patch row layout (c, y, x)");
}

void IdentityTest() {
    // LayerScale 0 disables both branches: output = LN(cls/patch + pos) with weight 1, bias 0.
    tp::Dinov2Weights w = tp::Dinov2Weights::Random(5, 0.05f, 3);
    for (auto& l : w.layers) { l.ls1.assign(1024, 0.f); l.ls2.assign(1024, 0.f); }
    const auto pixels = Pixels();
    std::vector<float> tokens;
    std::string error;
    Expect(tp::dinov2_encode_cpu(w, pixels.data(), &tokens, &error), "encode ok: " + error);
    Expect(tokens.size() == 257u * 1024, "257 x 1024 tokens");
    // Row 0: LN(cls + pos[0]).
    std::vector<double> row(1024);
    double mean = 0;
    for (int c = 0; c < 1024; ++c) { row[c] = w.cls[c] + w.pos224[c]; mean += row[c]; }
    mean /= 1024;
    double var = 0;
    for (int c = 0; c < 1024; ++c) var += (row[c] - mean) * (row[c] - mean);
    var /= 1024;
    float worst = 0;
    for (int c = 0; c < 1024; ++c)
        worst = std::max(worst, static_cast<float>(std::fabs((row[c] - mean) / std::sqrt(var + 1e-6) - tokens[c])));
    Expect(worst < 1e-4f, "cls token = LayerNorm(cls + pos) (worst " + std::to_string(worst) + ")");
    // Row 5 (patch 4): LN(patch_embed(row 4) + pos[5]).
    std::vector<float> rows;
    tp::dinov2_patch_rows(pixels.data(), &rows);
    for (int c = 0; c < 1024; ++c) {
        double acc = w.patch_b[c];
        for (int k = 0; k < 588; ++k) acc += static_cast<double>(rows[4 * 588 + k]) * w.patch_w[c * 588 + k];
        row[c] = acc + w.pos224[5 * 1024 + c];
    }
    mean = 0;
    for (double v : row) mean += v;
    mean /= 1024;
    var = 0;
    for (double v : row) var += (v - mean) * (v - mean);
    var /= 1024;
    worst = 0;
    for (int c = 0; c < 1024; ++c)
        worst = std::max(worst, static_cast<float>(std::fabs((row[c] - mean) / std::sqrt(var + 1e-6) - tokens[5 * 1024 + c])));
    Expect(worst < 1e-3f, "patch token = LayerNorm(embed + pos) (worst " + std::to_string(worst) + ")");
}

void BlocksTest() {
    tp::Dinov2Weights w = tp::Dinov2Weights::Random(9, 0.02f, 2);
    const auto pixels = Pixels();
    std::vector<float> tokens, again;
    std::string error;
    Expect(tp::dinov2_encode_cpu(w, pixels.data(), &tokens, &error), "2 blocks ok: " + error);
    bool finite = true;
    double energy = 0;
    for (float v : tokens) { finite = finite && std::isfinite(v); energy += static_cast<double>(v) * v; }
    Expect(finite, "finite output");
    // Final LayerNorm with weight 1 / bias 0 -> unit variance per row.
    Expect(std::fabs(energy / tokens.size() - 1.0) < 0.05, "unit variance after final norm (" + std::to_string(energy / tokens.size()) + ")");
    Expect(tp::dinov2_encode_cpu(w, pixels.data(), &again, &error) && again == tokens, "deterministic");
    // Blocks change the result relative to the identity model.
    tp::Dinov2Weights off = w;
    for (auto& l : off.layers) { l.ls1.assign(1024, 0.f); l.ls2.assign(1024, 0.f); }
    std::vector<float> base;
    Expect(tp::dinov2_encode_cpu(off, pixels.data(), &base, &error), "identity ok");
    double diff = 0;
    for (std::size_t i = 0; i < base.size(); ++i) diff += std::fabs(base[i] - tokens[i]);
    Expect(diff / base.size() > 0.01, "blocks contribute");
}

void NpyTest() {
    const auto path = std::filesystem::temp_directory_path() / "triposg_npy_test.npy";
    std::vector<float> data(257 * 1024);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i) * 0.5f;
    std::string error;
    Expect(tp::npy::write_f32(path, data.data(), {257, 1024}, &error), "npy write: " + error);
    std::vector<float> back;
    std::vector<std::size_t> shape;
    Expect(tp::npy::read_f32(path, &back, &shape, &error), "npy read: " + error);
    Expect(shape.size() == 2 && shape[0] == 257 && shape[1] == 1024, "npy shape");
    Expect(back == data, "npy data round trip");
    std::filesystem::remove(path);
}

}  // namespace

int main() {
    ShapesTest();
    PatchRowsTest();
    IdentityTest();
    BlocksTest();
    NpyTest();
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "dinov2_cpu_test: ok\n";
    return 0;
}
