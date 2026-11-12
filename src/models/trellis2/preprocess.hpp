// TRELLIS.2: image preprocessing after Trellis2ImageTo3DPipeline.preprocess_image and
// DinoV3FeatureExtractor.__call__ (PIL semantics reproduced in fixed point like the
// TripoSG port, filter LANCZOS = sinc * sinc(x/3), support 3):
//   1. RGBA with a real alpha channel (otherwise the original runs RMBG-2.0 = BRIA, not
//      commercial -> rejected here); if max(w, h) > 1024 the RGBA image is LANCZOS-resized
//      to int(w * s), int(h * s) with s = 1024 / max;
//   2. bounding box of alpha > 0.8 * 255, centre, size = max(bw, bh), crop
//      [cx - size//2, cy - size//2, cx + size//2, cy + size//2] (PIL crop pads outside
//      pixels with zeros = black, transparent);
//   3. RGB * alpha / 255 -> uint8 (truncation) -> black background;
//   4. LANCZOS resize to image x image (512 or 1024), /255, ImageNet mean/std -> [3, image, image].
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gufo::trellis2 {

struct Preprocessed {
    std::vector<std::uint8_t> square;   // composited RGB8 after the crop (before the final resize)
    int square_size = 0;
    int bbox[4] = {0, 0, 0, 0};         // x, y, w, h of alpha > 204 in the (possibly downscaled) input
    std::vector<float> pixels;          // [3, image, image] normalised (CHW)
    int image = 0;
};

// PIL-exact LANCZOS resize of packed 8-bit pixels with `channels` channels (RGB or RGBA).
std::vector<std::uint8_t> resize_pil_lanczos(const std::uint8_t* pixels, int channels, int width, int height,
                                             int out_width, int out_height);

// rgba: packed RGBA8. Returns false (with *error) when the alpha channel is unusable.
bool preprocess_rgba(const std::uint8_t* rgba, int width, int height, int image, Preprocessed* out, std::string* error);

}  // namespace gufo::trellis2
