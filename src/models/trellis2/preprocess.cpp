// TRELLIS.2: image preprocessing (see preprocess.hpp). PIL ImagingResample with 22-bit
// fixed-point coefficients, antialiasing (support scaled by the downscale factor),
// each pass rounded to 8 bits, horizontal pass first.
// AI-assisted (Claude Code), 2026-11-12.
#include "src/models/trellis2/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace gufo::trellis2 {
namespace {

struct Coefficients {
    std::vector<int> first, count;
    std::vector<std::int32_t> weights;  // [out, ksize]
    int ksize = 0;
};

double sinc(double x) {
    if (x == 0.0) return 1.0;
    x *= 3.14159265358979323846;
    return std::sin(x) / x;
}

double lanczos(double x) {
    x = std::fabs(x);
    if (x < 3.0) return sinc(x) * sinc(x / 3.0);
    return 0.0;
}

Coefficients make_coefficients(int in_size, int out_size) {
    constexpr int kPrecision = 22;
    constexpr double kSupport = 3.0;
    const double scale = static_cast<double>(in_size) / out_size;
    const double filterscale = std::max(scale, 1.0);
    const double support = kSupport * filterscale;
    Coefficients c;
    c.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    c.first.resize(static_cast<std::size_t>(out_size));
    c.count.resize(static_cast<std::size_t>(out_size));
    c.weights.assign(static_cast<std::size_t>(out_size) * static_cast<std::size_t>(c.ksize), 0);
    std::vector<double> k(static_cast<std::size_t>(c.ksize));
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale;
        const double ss = 1.0 / filterscale;
        int xmin = static_cast<int>(center - support + 0.5);
        xmin = std::max(xmin, 0);
        int xmax = static_cast<int>(center + support + 0.5);
        xmax = std::min(xmax, in_size) - xmin;
        double ww = 0;
        for (int x = 0; x < xmax; ++x) {
            k[static_cast<std::size_t>(x)] = lanczos((x + xmin - center + 0.5) * ss);
            ww += k[static_cast<std::size_t>(x)];
        }
        for (int x = 0; x < xmax; ++x) {
            const double w = ww != 0.0 ? k[static_cast<std::size_t>(x)] / ww : k[static_cast<std::size_t>(x)];
            c.weights[static_cast<std::size_t>(xx) * static_cast<std::size_t>(c.ksize) + static_cast<std::size_t>(x)] =
                static_cast<std::int32_t>(w < 0 ? w * (1 << kPrecision) - 0.5 : w * (1 << kPrecision) + 0.5);
        }
        c.first[static_cast<std::size_t>(xx)] = xmin;
        c.count[static_cast<std::size_t>(xx)] = xmax;
    }
    return c;
}

std::uint8_t clip8(std::int64_t value) {
    value >>= 22;
    return static_cast<std::uint8_t>(std::clamp<std::int64_t>(value, 0, 255));
}

}  // namespace

std::vector<std::uint8_t> resize_pil_lanczos(const std::uint8_t* px, int ch, int width, int height, int out_width,
                                             int out_height) {
    const Coefficients cx = make_coefficients(width, out_width);
    const Coefficients cy = make_coefficients(height, out_height);
    const std::size_t C = static_cast<std::size_t>(ch);
    std::vector<std::uint8_t> horizontal(static_cast<std::size_t>(height) * static_cast<std::size_t>(out_width) * C);
    for (int y = 0; y < height; ++y)
        for (int xx = 0; xx < out_width; ++xx)
            for (std::size_t c = 0; c < C; ++c) {
                std::int64_t sum = std::int64_t{1} << 21;
                const std::size_t wbase = static_cast<std::size_t>(xx) * static_cast<std::size_t>(cx.ksize);
                for (int x = 0; x < cx.count[static_cast<std::size_t>(xx)]; ++x)
                    sum += static_cast<std::int64_t>(cx.weights[wbase + static_cast<std::size_t>(x)]) *
                           px[(static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                               static_cast<std::size_t>(cx.first[static_cast<std::size_t>(xx)] + x)) * C + c];
                horizontal[(static_cast<std::size_t>(y) * static_cast<std::size_t>(out_width) + static_cast<std::size_t>(xx)) * C + c] = clip8(sum);
            }
    std::vector<std::uint8_t> out(static_cast<std::size_t>(out_height) * static_cast<std::size_t>(out_width) * C);
    for (int yy = 0; yy < out_height; ++yy)
        for (int x = 0; x < out_width; ++x)
            for (std::size_t c = 0; c < C; ++c) {
                std::int64_t sum = std::int64_t{1} << 21;
                const std::size_t wbase = static_cast<std::size_t>(yy) * static_cast<std::size_t>(cy.ksize);
                for (int y = 0; y < cy.count[static_cast<std::size_t>(yy)]; ++y)
                    sum += static_cast<std::int64_t>(cy.weights[wbase + static_cast<std::size_t>(y)]) *
                           horizontal[(static_cast<std::size_t>(cy.first[static_cast<std::size_t>(yy)] + y) * static_cast<std::size_t>(out_width) +
                                       static_cast<std::size_t>(x)) * C + c];
                out[(static_cast<std::size_t>(yy) * static_cast<std::size_t>(out_width) + static_cast<std::size_t>(x)) * C + c] = clip8(sum);
            }
    return out;
}

bool preprocess_rgba(const std::uint8_t* rgba_in, int width, int height, int image, Preprocessed* out, std::string* error) {
    if (width <= 0 || height <= 0) {
        if (error) *error = "empty image";
        return false;
    }
    std::vector<std::uint8_t> rgba(rgba_in, rgba_in + static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4);
    bool any_transparent = false;
    for (std::size_t i = 3; i < rgba.size(); i += 4)
        if (rgba[i] != 255) { any_transparent = true; break; }
    if (!any_transparent) {
        if (error) *error = "image has no alpha channel (background removal with RMBG-2.0 is not available)";
        return false;
    }
    // Downscale to at most 1024 px (LANCZOS on RGBA, PIL int() truncation of the sizes).
    const int max_size = std::max(width, height);
    if (max_size > 1024) {
        const double s = 1024.0 / max_size;
        const int nw = static_cast<int>(width * s), nh = static_cast<int>(height * s);
        rgba = resize_pil_lanczos(rgba.data(), 4, width, height, nw, nh);
        width = nw;
        height = nh;
    }
    // Bounding box of alpha > 0.8 * 255 (np.argwhere on alpha > 204).
    int x0 = width, y0 = height, x1 = -1, y1 = -1;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            if (rgba[(static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * 4 + 3] > 204) {
                x0 = std::min(x0, x); x1 = std::max(x1, x);
                y0 = std::min(y0, y); y1 = std::max(y1, y);
            }
    if (x1 < 0) {
        if (error) *error = "no opaque pixels (alpha > 204)";
        return false;
    }
    out->bbox[0] = x0; out->bbox[1] = y0; out->bbox[2] = x1 - x0; out->bbox[3] = y1 - y0;
    // center = ((x0 + x1) / 2, (y0 + y1) / 2) as float; size = int(max(x1 - x0, y1 - y0) * 1);
    // crop box = (cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2); PIL's
    // Image.crop maps the float box with round() (half to even) before int().
    const double cx = (x0 + x1) / 2.0, cy = (y0 + y1) / 2.0;
    const int size = std::max(x1 - x0, y1 - y0);
    const int half = size / 2;
    const int left = static_cast<int>(std::nearbyint(cx - half)), top = static_cast<int>(std::nearbyint(cy - half));
    const int right = static_cast<int>(std::nearbyint(cx + half)), bottom = static_cast<int>(std::nearbyint(cy + half));
    const int sw = right - left, sh = bottom - top;
    if (sw <= 0 || sh <= 0) {
        if (error) *error = "degenerate crop";
        return false;
    }
    // Composite RGB * alpha (float / 255, product, * 255, astype uint8 -> truncation); outside = 0.
    std::vector<std::uint8_t> crop(static_cast<std::size_t>(sw) * static_cast<std::size_t>(sh) * 3, 0);
    for (int y = 0; y < sh; ++y)
        for (int x = 0; x < sw; ++x) {
            const int sx = left + x, sy = top + y;
            if (sx < 0 || sy < 0 || sx >= width || sy >= height) continue;
            const std::uint8_t* p = rgba.data() + (static_cast<std::size_t>(sy) * static_cast<std::size_t>(width) + static_cast<std::size_t>(sx)) * 4;
            const float a = static_cast<float>(p[3]) / 255.f;
            std::uint8_t* q = crop.data() + (static_cast<std::size_t>(y) * static_cast<std::size_t>(sw) + static_cast<std::size_t>(x)) * 3;
            for (int c = 0; c < 3; ++c) {
                const float v = static_cast<float>(p[c]) / 255.f * a * 255.f;
                q[c] = static_cast<std::uint8_t>(std::clamp(v, 0.f, 255.f));
            }
        }
    // The square can be non-square by one pixel when the box is odd; PIL resize handles it.
    out->square = crop;
    out->square_size = sw;
    // DinoV3FeatureExtractor: resize to (image, image) LANCZOS, / 255, ImageNet normalisation.
    const std::vector<std::uint8_t> resized = resize_pil_lanczos(crop.data(), 3, sw, sh, image, image);
    constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
    out->image = image;
    out->pixels.resize(static_cast<std::size_t>(3) * static_cast<std::size_t>(image) * static_cast<std::size_t>(image));
    for (int y = 0; y < image; ++y)
        for (int x = 0; x < image; ++x)
            for (int c = 0; c < 3; ++c) {
                const float v = static_cast<float>(resized[(static_cast<std::size_t>(y) * static_cast<std::size_t>(image) + static_cast<std::size_t>(x)) * 3 + static_cast<std::size_t>(c)]) / 255.f;
                out->pixels[(static_cast<std::size_t>(c) * static_cast<std::size_t>(image) + static_cast<std::size_t>(y)) * static_cast<std::size_t>(image) + static_cast<std::size_t>(x)] =
                    (v - kMean[c]) / kStd[c];
            }
    return true;
}

}  // namespace gufo::trellis2
