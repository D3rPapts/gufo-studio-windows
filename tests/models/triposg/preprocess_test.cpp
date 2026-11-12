// CPU tests for src/models/triposg/preprocess.*: alpha validity rule (20-bin
// histogram, 1 %), bounding box + 10 % padding exactly like image_process.load_image,
// white compositing, PIL bicubic resize identities (same size = identity, constant
// image stays constant, downscale averages), centre crop and normalisation.
// AI-assisted (Claude Code), 2026-11-12.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/triposg/preprocess.hpp"

namespace tp = gufo::triposg;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

std::vector<std::uint8_t> Rgba(int w, int h, std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) {
    std::vector<std::uint8_t> v(static_cast<std::size_t>(w) * h * 4);
    for (std::size_t i = 0; i < v.size(); i += 4) { v[i] = r; v[i + 1] = g; v[i + 2] = b; v[i + 3] = a; }
    return v;
}

void Paint(std::vector<std::uint8_t>& img, int w, int x0, int y0, int x1, int y1, std::uint8_t r, std::uint8_t g,
           std::uint8_t b, std::uint8_t a) {
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) {
            std::uint8_t* p = img.data() + (static_cast<std::size_t>(y) * w + x) * 4;
            p[0] = r; p[1] = g; p[2] = b; p[3] = a;
        }
}

void AlphaRuleTest() {
    const int w = 100, h = 100;
    auto img = Rgba(w, h, 10, 20, 30, 0);
    Expect(!tp::alpha_is_valid(img.data(), w, h), "all transparent: invalid");
    Paint(img, w, 0, 0, 10, 10, 1, 2, 3, 255);  // 1 % opaque
    Expect(tp::alpha_is_valid(img.data(), w, h), "1 % opaque + rest transparent: valid");
    auto opaque = Rgba(w, h, 1, 2, 3, 255);
    Expect(!tp::alpha_is_valid(opaque.data(), w, h), "all opaque: invalid (no alpha path)");
    Paint(opaque, w, 0, 0, 10, 9, 0, 0, 0, 5);  // 0.9 % in bin 0
    Expect(!tp::alpha_is_valid(opaque.data(), w, h), "0.9 % transparent: invalid");
    Paint(opaque, w, 0, 0, 10, 10, 0, 0, 0, 12);  // 1 % in bin 0 (12 < 12.8)
    Expect(tp::alpha_is_valid(opaque.data(), w, h), "1 % with alpha 12: valid (bin 0)");
    Paint(opaque, w, 0, 0, 10, 10, 0, 0, 0, 13);
    Expect(!tp::alpha_is_valid(opaque.data(), w, h), "alpha 13 is bin 1: invalid");
    auto semi = Rgba(w, h, 1, 2, 3, 243);
    Paint(semi, w, 0, 0, 10, 10, 0, 0, 0, 0);
    Expect(!tp::alpha_is_valid(semi.data(), w, h), "alpha 243 is bin 18: invalid");
    Paint(semi, w, 50, 50, 60, 60, 0, 0, 0, 244);
    Expect(tp::alpha_is_valid(semi.data(), w, h), "alpha 244 is bin 19: valid");
    tp::Preprocessed pre;
    std::string error;
    Expect(!tp::preprocess_rgba(Rgba(w, h, 1, 2, 3, 255).data(), w, h, &pre, &error) && !error.empty(),
           "preprocess rejects images without usable alpha");
}

void BoundingBoxAndPaddingTest() {
    // 300x200 image, opaque red box at x 40..139 (w 100), y 70..129 (h 60): w > h.
    const int w = 300, h = 200;
    auto img = Rgba(w, h, 0, 0, 0, 0);
    Paint(img, w, 40, 70, 140, 130, 200, 30, 30, 255);
    tp::Preprocessed pre;
    std::string error;
    Expect(tp::preprocess_rgba(img.data(), w, h, &pre, &error), "preprocess ok: " + error);
    Expect(pre.used_alpha, "alpha path used");
    Expect(pre.bbox[0] == 40 && pre.bbox[1] == 70 && pre.bbox[2] == 100 && pre.bbox[3] == 60, "bbox 40,70 100x60");
    // pad_left = int(100 * 0.1) = 10; pad_top = int(10 + (100 - 60) / 2) = 30 -> 120 x 120.
    Expect(pre.square_width == 120 && pre.square_height == 120,
           "square 120x120 (" + std::to_string(pre.square_width) + "x" + std::to_string(pre.square_height) + ")");
    // Corner is white, centre is the box colour.
    Expect(pre.square[0] == 255 && pre.square[1] == 255 && pre.square[2] == 255, "padding is white");
    const std::size_t centre = (static_cast<std::size_t>(60) * 120 + 60) * 3;
    Expect(pre.square[centre] == 200 && pre.square[centre + 1] == 30 && pre.square[centre + 2] == 30,
           "opaque pixel keeps its colour");
    Expect(pre.pixels.size() == 3u * 224 * 224, "output [3,224,224]");
    // Centre of the crop: box colour normalised; corner: white normalised.
    const float mean[3] = {0.485f, 0.456f, 0.406f}, sd[3] = {0.229f, 0.224f, 0.225f};
    const float scale = static_cast<float>(0.00392156862745098);
    for (int c = 0; c < 3; ++c) {
        const float got = pre.pixels[(static_cast<std::size_t>(c) * 224 + 112) * 224 + 112];
        const std::uint8_t colour = c == 0 ? 200 : 30;
        const float want = (static_cast<float>(colour) * scale - mean[c]) / sd[c];
        Expect(std::fabs(got - want) < 1e-5f, "centre pixel normalised channel " + std::to_string(c));
        const float corner = pre.pixels[(static_cast<std::size_t>(c) * 224) * 224];
        const float want_white = (255.f * scale - mean[c]) / sd[c];
        Expect(std::fabs(corner - want_white) < 1e-5f, "corner pixel white channel " + std::to_string(c));
    }

    // Tall box (h > w): 60 wide, 100 tall -> pad_top = 10, pad_left = int(10 + 20) = 30 -> 120x120.
    auto tall = Rgba(w, h, 0, 0, 0, 0);
    Paint(tall, w, 100, 50, 160, 150, 10, 200, 10, 255);
    Expect(tp::preprocess_rgba(tall.data(), w, h, &pre, &error), "tall preprocess ok");
    Expect(pre.square_width == 120 && pre.square_height == 120, "tall square 120x120");

    // Semi-transparent pixel composites on white in float32 like the reference:
    // alpha 128 of (0,0,0) -> 1 - fp32(128/255) = 0.49803919 -> * 255 = 126.99999 -> 126
    // after (img * 255).astype(np.uint8) truncation (127 would be the rounded value).
    auto semi = Rgba(w, h, 0, 0, 0, 0);
    Paint(semi, w, 0, 0, 100, 100, 0, 0, 0, 255);
    Paint(semi, w, 50, 50, 51, 51, 0, 0, 0, 128);
    Expect(tp::preprocess_rgba(semi.data(), w, h, &pre, &error), "semi preprocess ok");
    const std::size_t at = (static_cast<std::size_t>(10 + 50) * pre.square_width + 10 + 50) * 3;
    Expect(pre.square[at] == 126, "alpha 128 over white truncates to 126 (" + std::to_string(pre.square[at]) + ")");
    Paint(semi, w, 50, 50, 51, 51, 0, 0, 0, 64);  // 1 - fp32(64/255) = 0.7490196 -> 191.0 exactly? -> 190 or 191
    Expect(tp::preprocess_rgba(semi.data(), w, h, &pre, &error), "semi preprocess ok 2");
    Expect(pre.square[at] == 190 || pre.square[at] == 191, "alpha 64 over white ~ 191 (" + std::to_string(pre.square[at]) + ")");
}

void ResizeTest() {
    // Identity at equal size.
    std::vector<std::uint8_t> img(16 * 16 * 3);
    for (std::size_t i = 0; i < img.size(); ++i) img[i] = static_cast<std::uint8_t>((i * 37) % 251);
    const auto same = tp::resize_rgb_pil_bicubic(img.data(), 16, 16, 16, 16);
    Expect(same == img, "resize to the same size is the identity");
    // Constant stays constant on down- and upscale.
    std::vector<std::uint8_t> flat(40 * 30 * 3, 77);
    for (auto v : tp::resize_rgb_pil_bicubic(flat.data(), 40, 30, 13, 9))
        if (v != 77) { Expect(false, "constant downscale"); break; }
    for (auto v : tp::resize_rgb_pil_bicubic(flat.data(), 40, 30, 90, 70))
        if (v != 77) { Expect(false, "constant upscale"); break; }
    // Half black / half white, downscaled 8x: the border pixel becomes mid grey, far pixels stay.
    std::vector<std::uint8_t> half(64 * 8 * 3, 0);
    for (int y = 0; y < 8; ++y)
        for (int x = 32; x < 64; ++x)
            for (int c = 0; c < 3; ++c) half[(static_cast<std::size_t>(y) * 64 + x) * 3 + c] = 255;
    const auto down = tp::resize_rgb_pil_bicubic(half.data(), 64, 8, 8, 1);
    Expect(down.size() == 8 * 3, "downscale size");
    Expect(down[0] == 0 && down[7 * 3] == 255, "far pixels keep black / white");
    Expect(down[3 * 3] < 60 && down[4 * 3] > 195, "edge pixels lean to their side (" + std::to_string(down[3 * 3]) + "," + std::to_string(down[4 * 3]) + ")");
}

}  // namespace

int main() {
    AlphaRuleTest();
    BoundingBoxAndPaddingTest();
    ResizeTest();
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "preprocess_test: ok\n";
    return 0;
}
