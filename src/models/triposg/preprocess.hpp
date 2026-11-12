// TripoSG: image preprocessing for the DINOv2 conditioning, bit-for-bit after the
// reference (scripts/image_process.py alpha path + HF BitImageProcessor):
//   1. alpha from the PNG (valid when >= 1 % fully transparent and >= 1 % fully
//      opaque pixels, 20-bin histogram), composite on white, crop to the alpha
//      bounding box, pad to a square with 10 % margin (white), truncate to uint8;
//   2. PIL bicubic resize (shortest edge 256, antialiased, 8-bit per pass),
//      centre crop 224, x/255, ImageNet mean/std -> float [3, 224, 224].
// No background removal network (RMBG-1.4 is non-commercial); images without a
// usable alpha channel are rejected until the threshold matting of stage 6.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gufo::triposg {

struct PreprocessOptions {
    float padding_ratio = 0.1f;
    int resize_shortest = 256;
    int crop = 224;
};

struct Preprocessed {
    std::vector<float> pixels;        // [3, crop, crop], normalised (CHW)
    std::vector<std::uint8_t> square; // composited, cropped, padded RGB8 (before the resize)
    int square_width = 0, square_height = 0;
    int bbox[4] = {0, 0, 0, 0};       // x, y, w, h of the alpha > 0 region in the input
    bool used_alpha = false;
};

// True when the alpha plane qualifies for the reference's alpha path.
bool alpha_is_valid(const std::uint8_t* rgba, int width, int height, float min_ratio = 0.01f);

// rgba: packed RGBA8, row-major. Returns false (with *error) when no usable alpha.
bool preprocess_rgba(const std::uint8_t* rgba, int width, int height, Preprocessed* out,
                     std::string* error, const PreprocessOptions& options = {});

// PIL-exact bicubic resize of packed RGB8 (antialiased when downscaling, each pass
// rounded to 8 bits), exposed for tests.
std::vector<std::uint8_t> resize_rgb_pil_bicubic(const std::uint8_t* rgb, int width, int height,
                                                 int out_width, int out_height);

}  // namespace gufo::triposg
