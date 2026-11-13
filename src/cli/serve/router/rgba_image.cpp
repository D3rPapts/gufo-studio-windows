// Copied from src/models/qwen_image_21/image.cpp (upstream v0.31.0):
// CheckImage, PngOutput/WritePng, LanczosFilters, EncodePng and ResizeImage.
// Keep in sync by hand; see rgba_image.hpp for why the router has its own.
#include "src/cli/serve/router/rgba_image.hpp"

#include <png.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <numbers>
#include <stdexcept>

#include "src/core/image.hpp"

namespace gufo::router::image {
namespace {

using Image = RgbaImage;

void CheckImage(const Image& image) {
  if (image.width <= 0 || image.height <= 0 ||
      static_cast<std::uint64_t>(image.width) * image.height >
          core::kMaxImagePixels ||
      image.rgba.size() !=
          static_cast<std::size_t>(image.width) * image.height * 4)
    throw std::invalid_argument("invalid RGBA image dimensions");
}

struct Filter {
  int begin{0};
  std::vector<std::int32_t> weights;
};

struct PngOutput {
  std::vector<std::uint8_t> bytes;
  std::size_t used{0};
};

void WritePng(png_structp png, png_bytep data, png_size_t size) {
  auto& output = *static_cast<PngOutput*>(png_get_io_ptr(png));
  if (size > output.bytes.size() - output.used)
    png_error(png, "output PNG exceeds its allocation");
  std::memcpy(output.bytes.data() + output.used, data, size);
  output.used += size;
}

std::vector<Filter> LanczosFilters(int input, int output) {
  std::vector<Filter> result(output);
  const double scale = static_cast<double>(input) / output;
  const double filter_scale = std::max(scale, 1.0);
  const double support = filter_scale * 3;
  for (int dst = 0; dst < output; ++dst) {
    const double center = (dst + 0.5) * scale;
    const int start = std::max(0, static_cast<int>(center - support + 0.5));
    const int end = std::min(input, static_cast<int>(center + support + 0.5));
    auto& f = result[dst];
    f.begin = start;
    std::vector<double> values;
    double sum = 0;
    for (int i = start; i < end; ++i) {
      const double x = (i - center + 0.5) / filter_scale;
      const auto sinc = [](double z) {
        return z == 0 ? 1.0
                      : std::sin(z * std::numbers::pi) / (z * std::numbers::pi);
      };
      const double w = (x >= -3 && x < 3) ? sinc(x) * sinc(x / 3) : 0;
      values.push_back(w);
      sum += w;
    }
    for (double value : values)
      f.weights.push_back(static_cast<std::int32_t>(value / sum * (1 << 22) +
                                                    (value < 0 ? -0.5 : 0.5)));
  }
  return result;
}

}  // namespace

std::vector<std::uint8_t> EncodePng(const Image& image) {
  CheckImage(image);
  png_image geometry{};
  geometry.width = image.width;
  geometry.height = image.height;
  geometry.format = PNG_FORMAT_RGBA;
  // Allocate before setjmp; callbacks only mutate heap state and cannot throw.
  const auto output = std::make_unique<PngOutput>();
  output->bytes.resize(PNG_IMAGE_PNG_SIZE_MAX(geometry));
  auto png = png_create_write_struct(
      PNG_LIBPNG_VER_STRING, nullptr,
      [](png_structp p, png_const_charp) { png_longjmp(p, 1); }, nullptr);
  if (!png)
    throw std::runtime_error("cannot allocate PNG encoder");
  auto info = png_create_info_struct(png);
  if (!info) {
    png_destroy_write_struct(&png, nullptr);
    throw std::runtime_error("cannot allocate PNG metadata");
  }
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_write_struct(&png, &info);
    throw std::runtime_error("cannot encode output PNG");
  }
  png_set_write_fn(png, output.get(), WritePng, nullptr);
  png_set_IHDR(png, info, image.width, image.height, 8, PNG_COLOR_TYPE_RGBA,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_set_sRGB(png, info, PNG_sRGB_INTENT_PERCEPTUAL);
  // Low compression avoids blocking the response on an expensive zlib search.
  // Adaptive filtering and every RGBA byte are preserved.
  png_set_compression_level(png, 1);
  png_set_filter(png, PNG_FILTER_TYPE_BASE, PNG_ALL_FILTERS);
  png_write_info(png, info);
  for (int row = 0; row < image.height; ++row)
    png_write_row(png, const_cast<png_bytep>(image.rgba.data() +
                                             static_cast<std::size_t>(row) *
                                                 image.width * 4));
  png_write_end(png, info);
  png_destroy_write_struct(&png, &info);
  output->bytes.resize(output->used);
  return std::move(output->bytes);
}

Image ResizeImage(const Image& image, int width, int height) {
  CheckImage(image);
  if (width <= 0 || height <= 0 ||
      static_cast<std::uint64_t>(width) * height > core::kMaxImagePixels)
    throw std::invalid_argument("invalid resized image dimensions");
  if (width == image.width && height == image.height)
    return image;
  // Pillow resizes RGBA in premultiplied space using 22-bit integer Lanczos
  // coefficients. Retain each pass's uint8 rounding, including alpha.
  Image source = image;
  for (std::size_t i = 0; i < source.rgba.size(); i += 4)
    for (int c = 0; c < 3; ++c) {
      const int v = source.rgba[i + c] * source.rgba[i + 3] + 128;
      source.rgba[i + c] = static_cast<std::uint8_t>((v + (v >> 8)) >> 8);
    }
  const auto horizontal = LanczosFilters(image.width, width);
  const auto vertical = LanczosFilters(image.height, height);
  Image temp{width, image.height,
             std::vector<std::uint8_t>(static_cast<std::size_t>(width) *
                                       image.height * 4)};
  Image output{
      width, height,
      std::vector<std::uint8_t>(static_cast<std::size_t>(width) * height * 4)};
  for (int y = 0; y < image.height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 4; ++c) {
        std::int64_t sum = 1 << 21;
        const auto& f = horizontal[x];
        for (std::size_t k = 0; k < f.weights.size(); ++k)
          sum += static_cast<std::int64_t>(f.weights[k]) *
                 source.rgba[(static_cast<std::size_t>(y) * image.width +
                              f.begin + k) *
                                 4 +
                             c];
        temp.rgba[(static_cast<std::size_t>(y) * width + x) * 4 + c] =
            static_cast<std::uint8_t>(
                std::clamp<std::int64_t>(sum >> 22, 0, 255));
      }
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 4; ++c) {
        std::int64_t sum = 1 << 21;
        const auto& f = vertical[y];
        for (std::size_t k = 0; k < f.weights.size(); ++k)
          sum += static_cast<std::int64_t>(f.weights[k]) *
                 temp.rgba[((f.begin + k) * width + x) * 4 + c];
        output.rgba[(static_cast<std::size_t>(y) * width + x) * 4 + c] =
            static_cast<std::uint8_t>(
                std::clamp<std::int64_t>(sum >> 22, 0, 255));
      }
  for (std::size_t i = 0; i < output.rgba.size(); i += 4) {
    const int alpha = output.rgba[i + 3];
    if (alpha != 0 && alpha != 255)
      for (int c = 0; c < 3; ++c)
        output.rgba[i + c] = static_cast<std::uint8_t>(
            std::min(255, output.rgba[i + c] * 255 / alpha));
  }
  return output;
}

}  // namespace gufo::router::image
