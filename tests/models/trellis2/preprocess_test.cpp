// CPU tests for src/models/trellis2/preprocess.cpp: LANCZOS resize keeps constant
// images, the alpha bounding box / square crop / black compositing, output layout and
// normalisation, rejection of images without alpha, downscale of inputs above 1024 px.
// AI-assisted (Claude Code), 2026-11-12.
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/trellis2/preprocess.hpp"

namespace tr = gufo::trellis2;

namespace {
int failures = 0;
void Expect(bool c, const std::string& what) {
    if (!c) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}
}  // namespace

int main() {
    // constant image stays constant through LANCZOS (weights sum to 1)
    std::vector<std::uint8_t> flat(40 * 30 * 3, 177);
    auto r = tr::resize_pil_lanczos(flat.data(), 3, 40, 30, 17, 11);
    bool same = r.size() == 17 * 11 * 3;
    for (auto v : r) same = same && v == 177;
    Expect(same, "constant image unchanged by the resize");
    // upscale of a constant RGBA too
    std::vector<std::uint8_t> flat4(8 * 8 * 4, 90);
    auto r4 = tr::resize_pil_lanczos(flat4.data(), 4, 8, 8, 20, 20);
    same = r4.size() == 20 * 20 * 4;
    for (auto v : r4) same = same && v == 90;
    Expect(same, "constant RGBA upscale");

    // 64x64 RGBA: opaque red square at x 10..29, y 20..49 (w 20, h 30), transparent elsewhere
    const int W = 64, H = 64;
    std::vector<std::uint8_t> rgba(static_cast<std::size_t>(W) * H * 4, 0);
    for (int y = 20; y < 50; ++y)
        for (int x = 10; x < 30; ++x) {
            std::uint8_t* p = rgba.data() + (static_cast<std::size_t>(y) * W + x) * 4;
            p[0] = 200; p[1] = 100; p[2] = 50; p[3] = 255;
        }
    tr::Preprocessed pre;
    std::string error;
    Expect(tr::preprocess_rgba(rgba.data(), W, H, 32, &pre, &error), "preprocess: " + error);
    Expect(pre.bbox[0] == 10 && pre.bbox[1] == 20 && pre.bbox[2] == 19 && pre.bbox[3] == 29, "bbox from alpha > 204 (max - min)");
    // size = 29, half 14; cx = 19.5, cy = 34.5 -> box round(5.5)=6 .. round(33.5)=34 (half to even), 20.5->20 .. 48.5->48
    Expect(pre.square_size == 28 && pre.square.size() == 28 * 28 * 3, "square crop 28 px (round half to even)");
    // top-left of the crop (6, 20) is transparent -> black; centre is the red square
    Expect(pre.square[0] == 0 && pre.square[1] == 0, "transparent area composited to black");
    const std::uint8_t* centre = pre.square.data() + (14 * 28 + 14) * 3;
    Expect(centre[0] == 200 && centre[1] == 100 && centre[2] == 50, "opaque pixels keep their colour");
    Expect(pre.pixels.size() == 3 * 32 * 32 && pre.image == 32, "pixels [3,32,32]");
    // normalised value of the centre pixel channel 0: 200/255 -> (0.784 - 0.485)/0.229
    const float v = pre.pixels[(0 * 32 + 16) * 32 + 16];
    Expect(std::fabs(v - (200.f / 255.f - 0.485f) / 0.229f) < 0.05f, "normalisation (ImageNet)");

    // half-transparent pixel: RGB * alpha
    std::vector<std::uint8_t> rgba2 = rgba;
    for (int y = 20; y < 50; ++y)
        for (int x = 10; x < 30; ++x) rgba2[(static_cast<std::size_t>(y) * W + x) * 4 + 3] = 210;
    tr::Preprocessed pre2;
    Expect(tr::preprocess_rgba(rgba2.data(), W, H, 32, &pre2, &error), "alpha 210 still counts as opaque (> 204)");
    const std::uint8_t* c2 = pre2.square.data() + (14 * 28 + 14) * 3;
    Expect(c2[0] == static_cast<std::uint8_t>(200.f / 255.f * (210.f / 255.f) * 255.f), "RGB * alpha truncated");

    // no alpha at all -> rejected
    std::vector<std::uint8_t> opaque(static_cast<std::size_t>(W) * H * 4, 255);
    Expect(!tr::preprocess_rgba(opaque.data(), W, H, 32, &pre, &error), "fully opaque image rejected (needs RMBG)");

    // > 1024 px input is downscaled first (1500 x 1000 -> 1024 x 682)
    const int BW = 1500, BH = 1000;
    std::vector<std::uint8_t> big(static_cast<std::size_t>(BW) * BH * 4, 0);
    for (int y = 100; y < 900; ++y)
        for (int x = 200; x < 1300; ++x) {
            std::uint8_t* p = big.data() + (static_cast<std::size_t>(y) * BW + x) * 4;
            p[0] = 10; p[1] = 20; p[2] = 30; p[3] = 255;
        }
    tr::Preprocessed pre3;
    Expect(tr::preprocess_rgba(big.data(), BW, BH, 64, &pre3, &error), "large image: " + error);
    Expect(pre3.bbox[0] >= 135 && pre3.bbox[0] <= 138 && pre3.bbox[2] >= 748 && pre3.bbox[2] <= 752, "bbox measured on the 1024 px version");
    if (failures == 0) std::cout << "trellis2_preprocess_test: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
