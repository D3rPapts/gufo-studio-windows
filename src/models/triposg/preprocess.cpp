// TripoSG: image preprocessing (see preprocess.hpp). The PIL resampling follows
// wan22/clip_vision.hip (22-bit fixed point, bicubic a = -0.5, antialiasing).
// AI-assisted (Claude Code), 2026-11-12.
#include "preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace gufo::triposg {
namespace {

struct Coefficients {
    std::vector<int> first, count;
    std::vector<std::int32_t> weights;  // [out, ksize]
    int ksize = 0;
};

double bicubic(double x) {
    constexpr double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

Coefficients make_coefficients(int in_size, int out_size) {
    constexpr int kPrecision = 22;
    const double scale = static_cast<double>(in_size) / out_size;
    const double filterscale = std::max(scale, 1.0);
    const double support = 2.0 * filterscale;
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
            k[static_cast<std::size_t>(x)] = bicubic((x + xmin - center + 0.5) * ss);
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

std::vector<std::uint8_t> resize_rgb_pil_bicubic(const std::uint8_t* rgb, int width, int height,
                                                 int out_width, int out_height) {
    const Coefficients cx = make_coefficients(width, out_width);
    const Coefficients cy = make_coefficients(height, out_height);
    std::vector<std::uint8_t> horizontal(static_cast<std::size_t>(height) * static_cast<std::size_t>(out_width) * 3);
    for (int y = 0; y < height; ++y)
        for (int xx = 0; xx < out_width; ++xx)
            for (int c = 0; c < 3; ++c) {
                std::int64_t sum = std::int64_t{1} << 21;
                const std::size_t wbase = static_cast<std::size_t>(xx) * static_cast<std::size_t>(cx.ksize);
                for (int x = 0; x < cx.count[static_cast<std::size_t>(xx)]; ++x)
                    sum += static_cast<std::int64_t>(cx.weights[wbase + static_cast<std::size_t>(x)]) *
                           rgb[(static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                static_cast<std::size_t>(cx.first[static_cast<std::size_t>(xx)] + x)) * 3 + static_cast<std::size_t>(c)];
                horizontal[(static_cast<std::size_t>(y) * static_cast<std::size_t>(out_width) + static_cast<std::size_t>(xx)) * 3 +
                           static_cast<std::size_t>(c)] = clip8(sum);
            }
    std::vector<std::uint8_t> out(static_cast<std::size_t>(out_height) * static_cast<std::size_t>(out_width) * 3);
    for (int yy = 0; yy < out_height; ++yy)
        for (int x = 0; x < out_width; ++x)
            for (int c = 0; c < 3; ++c) {
                std::int64_t sum = std::int64_t{1} << 21;
                const std::size_t wbase = static_cast<std::size_t>(yy) * static_cast<std::size_t>(cy.ksize);
                for (int y = 0; y < cy.count[static_cast<std::size_t>(yy)]; ++y)
                    sum += static_cast<std::int64_t>(cy.weights[wbase + static_cast<std::size_t>(y)]) *
                           horizontal[(static_cast<std::size_t>(cy.first[static_cast<std::size_t>(yy)] + y) * static_cast<std::size_t>(out_width) +
                                       static_cast<std::size_t>(x)) * 3 + static_cast<std::size_t>(c)];
                out[(static_cast<std::size_t>(yy) * static_cast<std::size_t>(out_width) + static_cast<std::size_t>(x)) * 3 +
                    static_cast<std::size_t>(c)] = clip8(sum);
            }
    return out;
}

bool alpha_is_valid(const std::uint8_t* rgba, int width, int height, float min_ratio) {
    // cv2.calcHist with 20 bins over [0, 256): bin 0 = alpha < 12.8 (0..12), bin 19 = alpha >= 243.2 (244..255).
    const std::size_t n = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::size_t low = 0, high = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int a = rgba[i * 4 + 3];
        if (a * 20 / 256 == 0) ++low;
        if (a * 20 / 256 == 19) ++high;
    }
    const double need = static_cast<double>(n) * min_ratio;
    return static_cast<double>(low) >= need && static_cast<double>(high) >= need;
}

bool preprocess_rgba(const std::uint8_t* rgba, int width, int height, Preprocessed* out,
                     std::string* error, const PreprocessOptions& options) {
    if (width <= 0 || height <= 0) {
        if (error) *error = "empty image";
        return false;
    }
    if (!alpha_is_valid(rgba, width, height)) {
        if (error) *error = "image has no usable alpha channel (need >= 1 % transparent and >= 1 % opaque pixels)";
        return false;
    }
    // Bounding box of alpha > 0.
    int x0 = width, y0 = height, x1 = -1, y1 = -1;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            if (rgba[(static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * 4 + 3] > 0) {
                x0 = std::min(x0, x); x1 = std::max(x1, x);
                y0 = std::min(y0, y); y1 = std::max(y1, y);
            }
    if (x1 < 0) {
        if (error) *error = "input image too small (alpha is empty)";
        return false;
    }
    const int w = x1 - x0 + 1, h = y1 - y0 + 1;
    out->bbox[0] = x0; out->bbox[1] = y0; out->bbox[2] = w; out->bbox[3] = h;
    out->used_alpha = true;

    // Padding exactly as image_process.load_image (F.pad order left, right, top, bottom).
    int pad_left, pad_top;
    if (w > h) {
        pad_left = static_cast<int>(static_cast<float>(w) * options.padding_ratio);
        pad_top = static_cast<int>(static_cast<float>(pad_left) + static_cast<float>(w - h) / 2.f);
    } else {
        pad_top = static_cast<int>(static_cast<float>(h) * options.padding_ratio);
        pad_left = static_cast<int>(static_cast<float>(pad_top) + static_cast<float>(h - w) / 2.f);
    }
    const int sw = w + 2 * pad_left, sh = h + 2 * pad_top;
    out->square_width = sw;
    out->square_height = sh;
    out->square.assign(static_cast<std::size_t>(sw) * static_cast<std::size_t>(sh) * 3, 255);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const std::uint8_t* p = rgba + (static_cast<std::size_t>(y0 + y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x0 + x)) * 4;
            const float alpha = static_cast<float>(p[3]) / 255.f;
            const float inv = 1.f - alpha;
            std::uint8_t* q = out->square.data() + (static_cast<std::size_t>(pad_top + y) * static_cast<std::size_t>(sw) + static_cast<std::size_t>(pad_left + x)) * 3;
            for (int c = 0; c < 3; ++c) {
                const float unit = static_cast<float>(p[c]) / 255.f;
                const float fg = unit * alpha;
                const float bg = 1.f * inv;
                const float v = fg + bg;
                // (img * 255).astype(np.uint8): truncation toward zero.
                q[c] = static_cast<std::uint8_t>(std::clamp(v * 255.f, 0.f, 255.f));
            }
        }

    // BitImageProcessor: shortest edge -> resize_shortest (other edge int(scaled)), centre crop.
    const int short_edge = std::min(sw, sh);
    const int long_edge = std::max(sw, sh);
    const int new_short = options.resize_shortest;
    const int new_long = static_cast<int>(static_cast<double>(new_short) * long_edge / short_edge);
    const int rw = sw <= sh ? new_short : new_long;
    const int rh = sw <= sh ? new_long : new_short;
    const std::vector<std::uint8_t> resized = resize_rgb_pil_bicubic(out->square.data(), sw, sh, rw, rh);
    const int crop = options.crop;
    if (rw < crop || rh < crop) {
        if (error) *error = "resized image smaller than the crop";
        return false;
    }
    const int top = (rh - crop) / 2, left = (rw - crop) / 2;
    constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
    const float scale = static_cast<float>(0.00392156862745098);  // rescale_factor as float32
    out->pixels.resize(static_cast<std::size_t>(3) * static_cast<std::size_t>(crop) * static_cast<std::size_t>(crop));
    for (int y = 0; y < crop; ++y)
        for (int x = 0; x < crop; ++x)
            for (int c = 0; c < 3; ++c) {
                const float v = static_cast<float>(resized[(static_cast<std::size_t>(top + y) * static_cast<std::size_t>(rw) + static_cast<std::size_t>(left + x)) * 3 + static_cast<std::size_t>(c)]) * scale;
                out->pixels[(static_cast<std::size_t>(c) * static_cast<std::size_t>(crop) + static_cast<std::size_t>(y)) * static_cast<std::size_t>(crop) + static_cast<std::size_t>(x)] =
                    (v - kMean[c]) / kStd[c];
            }
    return true;
}

}  // namespace gufo::triposg
