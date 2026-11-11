#include "src/core/image.hpp"

#include <png.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <stdexcept>

#include "src/core/crypto/sha256.hpp"
#include "src/models/qwen_image_21/control.hpp"
#include "src/models/qwen_image_21/model.hpp"

namespace gufo::models::qwen_image_21 {
namespace {

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

std::uint64_t HashSeed(std::span<const std::uint8_t> domain, std::uint64_t seed,
                       std::span<const Image> images) {
  crypto::Sha256Hasher hash;
  hash.Update(domain);
  const auto append = [&](std::uint64_t value) {
    std::array<std::uint8_t, 8> bytes{};
    for (int i = 0; i < 8; ++i)
      bytes[i] = static_cast<std::uint8_t>(value >> (8 * i));
    hash.Update(bytes);
  };
  append(seed);
  append(images.size());
  for (const auto& image : images) {
    CheckImage(image);
    append(image.width);
    append(image.height);
    hash.Update(image.rgba);
  }
  const auto digest = hash.Finish();
  std::uint64_t result = 0;
  for (int i = 0; i < 8; ++i)
    result |= static_cast<std::uint64_t>(digest[i]) << (8 * i);
  return result;
}

}  // namespace

std::uint64_t InitialNoiseSeed(std::uint64_t seed,
                               std::span<const Image> images,
                               const Image* init_image) {
  if (init_image) {
    constexpr std::uint8_t domain[] = "Qwen-Image-2.1 img2img noise v1";
    return HashSeed(domain, InitialNoiseSeed(seed, images),
                    std::span(init_image, 1));
  }
  if (images.empty())
    return seed;
  constexpr std::uint8_t domain[] = "Qwen-Image-2.1 edit noise v1";
  return HashSeed(domain, seed, images);
}

Image DecodeImage(std::span<const std::uint8_t> bytes) {
  if (bytes.size() > core::kMaxEncodedImageBytes || bytes.size() < 8)
    throw std::invalid_argument("invalid or oversized input image");
  if (png_sig_cmp(bytes.data(), 0, 8) != 0) {
    const auto rgb = core::DecodeImage(bytes);
    Image image{static_cast<int>(rgb.width), static_cast<int>(rgb.height), {}};
    image.rgba.resize(static_cast<std::size_t>(rgb.width) * rgb.height * 4);
    for (std::size_t i = 0; i < image.rgba.size() / 4; ++i) {
      std::memcpy(image.rgba.data() + 4 * i, rgb.pixels.data() + 3 * i, 3);
      image.rgba[4 * i + 3] = 255;
    }
    return image;
  }
  png_image png{};
  png.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_memory(&png, bytes.data(), bytes.size()))
    throw std::invalid_argument("cannot decode PNG image");
  const auto cleanup = std::unique_ptr<png_image, decltype(&png_image_free)>(
      &png, png_image_free);
  if (!png.width || !png.height ||
      static_cast<std::uint64_t>(png.width) * png.height >
          core::kMaxImagePixels)
    throw std::invalid_argument("PNG image dimensions exceed limits");
  png.format = PNG_FORMAT_RGBA;
  Image image{static_cast<int>(png.width), static_cast<int>(png.height), {}};
  image.rgba.resize(PNG_IMAGE_SIZE(png));
  if (!png_image_finish_read(&png, nullptr, image.rgba.data(), 0, nullptr))
    throw std::invalid_argument("invalid PNG image data");
  return image;
}

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

MaskChannel ParseMaskChannel(std::string_view name) {
  if (name == "auto")
    return MaskChannel::kAuto;
  if (name == "alpha")
    return MaskChannel::kAlpha;
  if (name == "red")
    return MaskChannel::kRed;
  if (name == "green")
    return MaskChannel::kGreen;
  if (name == "blue")
    return MaskChannel::kBlue;
  throw std::invalid_argument(
      "mask channel must be auto, alpha, red, green or blue");
}

namespace {

// Edit value (0 keep .. 255 edit) of every mask pixel for `channel`.
std::vector<std::uint8_t> EditValues(const Image& mask, MaskChannel channel) {
  CheckImage(mask);
  // kAuto without a translucent pixel reads luminance (white = edit).
  bool luminance = false;
  if (channel == MaskChannel::kAuto) {
    luminance = true;
    for (std::size_t i = 3; i < mask.rgba.size(); i += 4)
      if (mask.rgba[i] != 255) {
        channel = MaskChannel::kAlpha;
        luminance = false;
        break;
      }
  }
  std::vector<std::uint8_t> values(mask.rgba.size() / 4);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto* p = mask.rgba.data() + 4 * i;
    int v = p[0];
    if (luminance)  // ITU-R BT.601, as Pillow's convert("L")
      v = (p[0] * 299 + p[1] * 587 + p[2] * 114 + 500) / 1000;
    else if (channel == MaskChannel::kAlpha)
      v = 255 - p[3];
    else if (channel == MaskChannel::kGreen)
      v = p[1];
    else if (channel == MaskChannel::kBlue)
      v = p[2];
    values[i] = static_cast<std::uint8_t>(v);
  }
  return values;
}

// w = max(w, max over the 3 x 3 neighbors - step), `radius` times: the
// max-plus dilation by a Chebyshev cone of slope 1 / (radius + 1).
void Spread(std::vector<float>& w, int rows, int cols, int radius) {
  const float step = 1.0F / static_cast<float>(radius + 1);
  std::vector<float> row_max(w.size());
  for (int k = 0; k < radius; ++k) {
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c) {
        const std::size_t i = static_cast<std::size_t>(r) * cols + c;
        float m = w[i];
        if (c > 0)
          m = std::max(m, w[i - 1]);
        if (c + 1 < cols)
          m = std::max(m, w[i + 1]);
        row_max[i] = m;
      }
    bool changed = false;
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c) {
        const std::size_t i = static_cast<std::size_t>(r) * cols + c;
        float m = row_max[i];
        if (r > 0)
          m = std::max(m, row_max[i - cols]);
        if (r + 1 < rows)
          m = std::max(m, row_max[i + cols]);
        if (m - step > w[i]) {
          w[i] = m - step;
          changed = true;
        }
      }
    if (!changed)
      break;
  }
}

}  // namespace

std::vector<float> MaskWeights(const Image& mask, MaskChannel channel,
                               int width, int height) {
  if (width < 16 || height < 16 || width % 16 || height % 16)
    throw std::invalid_argument("masked image size must be multiples of 16");
  const auto edit = EditValues(mask, channel);
  const int rows = height / 16, cols = width / 16;
  std::vector<float> weights(static_cast<std::size_t>(rows) * cols);
  // Cell [c * 16, c * 16 + 16) covers mask pixels [c * 16 * size / output,
  // (c + 1) * 16 * size / output), at least one.
  const auto span = [](int cell, int size, int output) {
    const auto begin = static_cast<int>(
        static_cast<std::int64_t>(cell) * 16 * size / output);
    const auto end = static_cast<int>(
        (static_cast<std::int64_t>(cell + 1) * 16 * size + output - 1) /
        output);
    return std::pair{std::min(begin, size - 1), std::max(begin + 1, end)};
  };
  for (int r = 0; r < rows; ++r) {
    const auto [y0, y1] = span(r, mask.height, height);
    for (int c = 0; c < cols; ++c) {
      const auto [x0, x1] = span(c, mask.width, width);
      int most = 0;
      for (int y = y0; y < std::min(y1, mask.height) && most < 255; ++y)
        for (int x = x0; x < std::min(x1, mask.width); ++x)
          most = std::max<int>(
              most, edit[static_cast<std::size_t>(y) * mask.width + x]);
      weights[static_cast<std::size_t>(r) * cols + c] =
          static_cast<float>(most) / 255.0F;
    }
  }
  return weights;
}

std::vector<float> SpreadMask(std::span<const float> weights, int rows,
                              int cols, int radius) {
  if (rows <= 0 || cols <= 0 ||
      weights.size() != static_cast<std::size_t>(rows) * cols || radius < 0)
    throw std::invalid_argument("invalid mask spread");
  std::vector<float> result(weights.begin(), weights.end());
  Spread(result, rows, cols, radius);
  return result;
}

std::vector<float> CompositeWeights(const Image& mask, MaskChannel channel,
                                    int width, int height, int feather) {
  if (width <= 0 || height <= 0 || feather < 0)
    throw std::invalid_argument("invalid mask composite");
  const auto edit = EditValues(mask, channel);
  std::vector<float> weights(static_cast<std::size_t>(width) * height);
  for (int y = 0; y < height; ++y) {
    // Nearest mask pixel to the output pixel's center.
    const int my = static_cast<int>(
        (2 * static_cast<std::int64_t>(y) + 1) * mask.height / (2 * height));
    for (int x = 0; x < width; ++x) {
      const int mx = static_cast<int>(
          (2 * static_cast<std::int64_t>(x) + 1) * mask.width / (2 * width));
      weights[static_cast<std::size_t>(y) * width + x] =
          static_cast<float>(
              edit[static_cast<std::size_t>(my) * mask.width + mx]) /
          255.0F;
    }
  }
  Spread(weights, height, width, feather);
  return weights;
}

void CompositeMasked(Image& result, const Image& original,
                     std::span<const float> weights) {
  CheckImage(result);
  CheckImage(original);
  if (result.width != original.width || result.height != original.height ||
      weights.size() != static_cast<std::size_t>(result.width) * result.height)
    throw std::invalid_argument("mask composite shape mismatch");
  for (std::size_t i = 0; i < weights.size(); ++i) {
    const float w = weights[i];
    if (w >= 1)
      continue;
    for (int c = 0; c < 4; ++c) {
      auto& out = result.rgba[4 * i + c];
      out = w <= 0 ? original.rgba[4 * i + c]
                   : static_cast<std::uint8_t>(std::lround(
                         w * out + (1 - w) * original.rgba[4 * i + c]));
    }
  }
}
void BlendMasked(std::span<float> latent, std::span<const float> kept,
                 std::span<const float> weights, int cols) {
  if (cols <= 0 || latent.size() != kept.size() ||
      latent.size() != weights.size() * static_cast<std::size_t>(cols))
    throw std::invalid_argument("masked blend shape mismatch");
  for (std::size_t i = 0; i < latent.size(); ++i) {
    const float m = weights[i / cols];
    latent[i] = m * latent[i] + (1.0F - m) * kept[i];
  }
}

void ValidateRequest(const Request& request) {
  if (request.prompt.size() > (64U << 10U))
    throw std::invalid_argument("image prompt exceeds 64 KiB");
  if (request.width < 32 || request.height < 32 || request.width % 32 ||
      request.height % 32 || request.width > 4096 || request.height > 4096 ||
      static_cast<std::uint64_t>(request.width) * request.height > 4194304)
    throw std::invalid_argument(
        "image size must be multiples of 32, at most 4096 per side and 4 "
        "megapixels");
  if (request.steps != 0 && (request.steps < 2 || request.steps > 100))
    throw std::invalid_argument("image steps must be between 2 and 100");
  if (request.images.size() > 10)
    throw std::invalid_argument(
        "Qwen-Image supports at most 10 reference images");
  std::size_t pixels = 0;
  for (const auto& image : request.images) {
    CheckImage(image);
    pixels += static_cast<std::size_t>(image.width) * image.height;
  }
  if (pixels > core::kMaxImagePixels)
    throw std::invalid_argument(
        "reference images exceed 32 megapixels combined");
  if (request.negative_prompt.size() > (64U << 10U))
    throw std::invalid_argument("negative prompt exceeds 64 KiB");
  if (!std::isfinite(request.guidance) || request.guidance < 0 ||
      request.guidance > 20)
    throw std::invalid_argument("guidance must be between 0 and 20");
  if (!(request.strength >= 0.01 && request.strength <= 1))
    throw std::invalid_argument("strength must be between 0.01 and 1");
  if (request.strength <= kFullStrength && !request.init_image)
    throw std::invalid_argument("strength below 1 requires an input image");
  if (request.init_image)
    CheckImage(*request.init_image);
  if (request.mask_feather < 0 || request.mask_feather > 64)
    throw std::invalid_argument("mask feather must be 0 to 64 pixels");
  if (request.mask_blur < 0 || request.mask_blur > 4)
    throw std::invalid_argument("mask blur must be 0 to 4 cells");
  if (request.mask) {
    if (!request.init_image)
      throw std::invalid_argument("a mask requires an input image to edit");
    CheckImage(*request.mask);
  } else if (request.mask_channel != MaskChannel::kAuto ||
             !request.mask_composite ||
             request.mask_feather != kDefaultMaskFeather ||
             request.mask_blur != kDefaultMaskBlur) {
    throw std::invalid_argument("mask options require a mask");
  }
  if (request.regions.size() > kMaxRegions)
    throw std::invalid_argument("image requests take at most " +
                                std::to_string(kMaxRegions) + " regions");
  if (!std::isfinite(request.base_strength) || request.base_strength < 0 ||
      request.base_strength > 10 ||
      (request.regions.empty() && request.base_strength != 1))
    throw std::invalid_argument(
        "base strength must be 0 to 10 and needs regions");
  std::size_t region_images = 0;
  for (const auto& region : request.regions) {
    if (region.prompt.size() > (64U << 10U))
      throw std::invalid_argument("region prompt exceeds 64 KiB");
    if (region.images.size() > 10)
      throw std::invalid_argument(
          "a region takes at most 10 reference images");
    region_images += region.images.size();
    for (const auto& image : region.images) {
      CheckImage(image);
      pixels += static_cast<std::size_t>(image.width) * image.height;
    }
    if (!std::isfinite(region.strength) || region.strength < 0 ||
        region.strength > 10)
      throw std::invalid_argument("region strength must be 0 to 10");
    if (const auto& area = region.area) {
      for (const double v : {area->x, area->y, area->width, area->height})
        if (!std::isfinite(v) || v < 0 || v > (area->fraction ? 1 : 16384))
          throw std::invalid_argument(
              "region area values must be 0..1 (fraction) or pixels");
      if (area->width <= 0 || area->height <= 0)
        throw std::invalid_argument("region area must not be empty");
    }
    if (const auto& mask = region.mask) {
      if (mask->width <= 0 || mask->height <= 0 ||
          static_cast<std::uint64_t>(mask->width) * mask->height >
              (16ULL << 20) ||
          mask->values.size() !=
              static_cast<std::size_t>(mask->width) * mask->height)
        throw std::invalid_argument("invalid region mask size");
      for (const float v : mask->values)
        if (!(v >= 0 && v <= 1))
          throw std::invalid_argument("region mask values must be 0 to 1");
    } else if (region.mask_bounds) {
      throw std::invalid_argument("region mask_bounds requires a mask");
    }
  }
  if (request.images.size() + region_images > kMaxReferenceImages)
    throw std::invalid_argument(
        "image requests take at most " + std::to_string(kMaxReferenceImages) +
        " reference images including regions");
  if (pixels > core::kMaxImagePixels)
    throw std::invalid_argument(
        "reference images exceed 32 megapixels combined");
  if (request.loras.size() > 16)
    throw std::invalid_argument("image requests apply at most 16 LoRAs");
  for (std::size_t i = 0; i < request.loras.size(); ++i) {
    const auto& lora = request.loras[i];
    if (lora.name.empty())
      throw std::invalid_argument("LoRA name must not be empty");
    for (std::size_t j = 0; j < i; ++j)
      if (request.loras[j].name == lora.name)
        throw std::invalid_argument("duplicate LoRA " + lora.name);
    if (!std::isfinite(lora.scale) || std::abs(lora.scale) > 10)
      throw std::invalid_argument(
          "LoRA scale must be finite and at most 10 in magnitude");
  }
  if (request.control)
    ValidateControl(*request.control, request);
}

std::string ConditionPrompt(std::string_view prompt,
                            ext::ImageBackground background) {
  if (background != ext::ImageBackground::kTransparent)
    return std::string(prompt);
  // Qwen-Image-2.1 produces RGBA output when the prompt names the alpha
  // channel; the framing is model-specific, so it lives in this provider.
  return "This is an RGBA image with transparency. " + std::string(prompt) +
         " The image has alpha channel and the background is transparent.";
}

namespace {

std::vector<float> Schedule(int steps, int image_tokens) {
  const double mu = 0.5 + (image_tokens - 256) * (0.9 - 0.5) / (8192 - 256);
  const float shift = static_cast<float>(std::exp(mu));
  std::vector<float> sigmas(steps + 1);
  for (int i = 0; i < steps; ++i) {
    const float t = static_cast<float>(1.0 - static_cast<double>(i) / steps);
    sigmas[i] = shift / (shift + (1.0F / t - 1.0F));
  }
  const float scale = (1.0F - sigmas[steps - 1]) / (1.0F - 0.02F);
  for (int i = 0; i < steps; ++i)
    sigmas[i] = 1.0F - (1.0F - sigmas[i]) / scale;
  sigmas[steps] = 0;
  return sigmas;
}

}  // namespace

std::vector<float> FlowSigmas(int steps, int image_tokens) {
  if (steps < 2 || steps > 100 || image_tokens < 1)
    throw std::invalid_argument("invalid image flow schedule");
  return Schedule(steps, image_tokens);
}

std::vector<float> DenoiseSigmas(int steps, double strength, int image_tokens) {
  if (!(strength >= 0.01 && strength <= 1))
    throw std::invalid_argument("invalid image denoise strength");
  if (strength > kFullStrength)
    return FlowSigmas(steps, image_tokens);
  if (steps < 2 || steps > 100 || image_tokens < 1)
    throw std::invalid_argument("invalid image flow schedule");
  // ComfyUI KSampler: int(steps / denoise) steps, keep the last steps + 1.
  const int total = static_cast<int>(steps / strength);
  auto sigmas = Schedule(total, image_tokens);
  sigmas.erase(sigmas.begin(), sigmas.end() - (steps + 1));
  return sigmas;
}

int SamplingConfig::ResolveSteps(int requested) const {
  const int steps = requested == 0 ? DefaultSteps() : requested;
  if (IsTurbo() && steps != 8)
    throw std::invalid_argument(
        "Qwen-Image-2.1-Turbo requires its fixed eight-step schedule");
  if (steps < 2 || steps > 100)
    throw std::invalid_argument("image steps must be between 2 and 100");
  return steps;
}

std::vector<float> SamplingConfig::Sigmas(int requested,
                                          int image_tokens) const {
  const int steps = ResolveSteps(requested);
  if (!IsTurbo())
    return FlowSigmas(steps, image_tokens);
  auto result = sample_sigmas;
  result.push_back(0);
  return result;
}

}  // namespace gufo::models::qwen_image_21

// Regional conditioning (ComfyUI area / mask conditionings).
namespace gufo::models::qwen_image_21 {
namespace {

struct CellRect {
  int top{0}, left{0}, rows{0}, cols{0};
};

void CheckRegionGrid(int width, int height) {
  if (width < 16 || height < 16 || width % 16 || height % 16)
    throw std::invalid_argument("regional image size must be multiples of 16");
}

/// The mask averaged over each cell of a rows x cols grid (stretched like
/// ComfyUI's resize of a mask to the latent size).
std::vector<float> MaskCells(const RegionMask& mask, int rows, int cols) {
  if (mask.width <= 0 || mask.height <= 0 ||
      mask.values.size() != static_cast<std::size_t>(mask.width) * mask.height)
    throw std::invalid_argument("invalid region mask");
  const auto span = [](int cell, int size, int cells) {
    const auto begin =
        static_cast<int>(static_cast<std::int64_t>(cell) * size / cells);
    const auto end = static_cast<int>(
        (static_cast<std::int64_t>(cell + 1) * size + cells - 1) / cells);
    return std::pair{std::min(begin, size - 1),
                     std::min(size, std::max(begin + 1, end))};
  };
  std::vector<float> cells(static_cast<std::size_t>(rows) * cols);
  for (int r = 0; r < rows; ++r) {
    const auto [y0, y1] = span(r, mask.height, rows);
    for (int c = 0; c < cols; ++c) {
      const auto [x0, x1] = span(c, mask.width, cols);
      double sum = 0;
      for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x)
          sum += mask.values[static_cast<std::size_t>(y) * mask.width + x];
      cells[static_cast<std::size_t>(r) * cols + c] =
          static_cast<float>(sum / ((y1 - y0) * (x1 - x0)));
    }
  }
  return cells;
}

/// Area in cells, rounded outward (pixels) or to the nearest cell
/// (fractions, as ComfyUI rounds percentages); empty when outside.
CellRect AreaCells(const RegionArea& area, int grid_rows, int grid_cols) {
  int top, left, bottom, right;
  if (area.fraction) {
    top = static_cast<int>(std::lround(area.y * grid_rows));
    left = static_cast<int>(std::lround(area.x * grid_cols));
    bottom = top + std::max(1, static_cast<int>(
                                   std::lround(area.height * grid_rows)));
    right = left + std::max(1, static_cast<int>(
                                   std::lround(area.width * grid_cols)));
  } else {
    top = static_cast<int>(std::floor(area.y / 16));
    left = static_cast<int>(std::floor(area.x / 16));
    bottom = static_cast<int>(std::ceil((area.y + area.height) / 16));
    right = static_cast<int>(std::ceil((area.x + area.width) / 16));
  }
  top = std::clamp(top, 0, grid_rows);
  left = std::clamp(left, 0, grid_cols);
  bottom = std::clamp(bottom, top, grid_rows);
  right = std::clamp(right, left, grid_cols);
  if (bottom == top || right == left)
    return {};
  return {top, left, bottom - top, right - left};
}

}  // namespace

bool RegionCells::Empty() const {
  return rows <= 0 || cols <= 0 ||
         std::none_of(weights.begin(), weights.end(),
                      [](float w) { return w > 0; });
}

RegionCells FullRegion(int width, int height, float strength) {
  CheckRegionGrid(width, height);
  RegionCells cells;
  cells.rows = height / 16;
  cells.cols = width / 16;
  cells.weights.assign(static_cast<std::size_t>(cells.rows) * cells.cols,
                       strength);
  return cells;
}

RegionCells PlanRegion(const Region& region, int width, int height) {
  CheckRegionGrid(width, height);
  const int grid_rows = height / 16, grid_cols = width / 16;
  const float strength = region.strength;
  std::vector<float> mask;
  if (region.mask)
    mask = MaskCells(*region.mask, grid_rows, grid_cols);
  CellRect core{0, 0, grid_rows, grid_cols};
  bool crop = false;
  if (region.mask && region.mask_bounds) {
    // ComfyUI's set_area_to_bounds: the bounding box of the nonzero mask.
    int top = grid_rows, left = grid_cols, bottom = 0, right = 0;
    for (int r = 0; r < grid_rows; ++r)
      for (int c = 0; c < grid_cols; ++c)
        if (mask[static_cast<std::size_t>(r) * grid_cols + c] > 0) {
          top = std::min(top, r);
          left = std::min(left, c);
          bottom = std::max(bottom, r + 1);
          right = std::max(right, c + 1);
        }
    core = bottom > top ? CellRect{top, left, bottom - top, right - left}
                        : CellRect{};
    crop = true;
  } else if (region.area) {
    core = AreaCells(*region.area, grid_rows, grid_cols);
    crop = true;
  }
  RegionCells cells;
  if (core.rows == 0 || core.cols == 0)
    return cells;  // nothing of the image: skipped
  std::vector<float> weights(static_cast<std::size_t>(core.rows) * core.cols);
  for (int r = 0; r < core.rows; ++r)
    for (int c = 0; c < core.cols; ++c) {
      const auto g =
          static_cast<std::size_t>(core.top + r) * grid_cols + core.left + c;
      weights[static_cast<std::size_t>(r) * core.cols + c] =
          region.mask ? mask[g] * strength : strength;
    }
  if (region.area && !region.mask) {
    // ComfyUI fades an area's edges inside the image over 8 latent pixels
    // (the t-th from such an edge scaled by (t + 1) / 8); here per cell.
    const int edge = kRegionEdgeCells;
    for (int r = 0; r < core.rows; ++r)
      for (int c = 0; c < core.cols; ++c) {
        float f = 1;
        if (core.top > 0 && r < edge)
          f *= static_cast<float>(r + 1) / edge;
        if (core.top + core.rows < grid_rows && core.rows - 1 - r < edge)
          f *= static_cast<float>(core.rows - r) / edge;
        if (core.left > 0 && c < edge)
          f *= static_cast<float>(c + 1) / edge;
        if (core.left + core.cols < grid_cols && core.cols - 1 - c < edge)
          f *= static_cast<float>(core.cols - c) / edge;
        weights[static_cast<std::size_t>(r) * core.cols + c] *= f;
      }
  }
  CellRect eval = core;
  if (crop) {
    // Grow small crops (attention tiles) symmetrically inside the grid.
    bool grow_rows = eval.rows <= eval.cols;
    while (eval.rows * eval.cols < kMinRegionCells &&
           (eval.rows < grid_rows || eval.cols < grid_cols)) {
      if ((grow_rows && eval.rows < grid_rows) || eval.cols == grid_cols) {
        if (eval.top > 0 &&
            (eval.top + eval.rows == grid_rows ||
             eval.top <= grid_rows - eval.top - eval.rows))
          --eval.top;
        ++eval.rows;
      } else {
        if (eval.left > 0 &&
            (eval.left + eval.cols == grid_cols ||
             eval.left <= grid_cols - eval.left - eval.cols))
          --eval.left;
        ++eval.cols;
      }
      grow_rows = !grow_rows;
    }
    if (eval.rows == grid_rows && eval.cols == grid_cols)
      crop = false;
  }
  cells.top = eval.top;
  cells.left = eval.left;
  cells.rows = eval.rows;
  cells.cols = eval.cols;
  cells.crop = crop;
  cells.weights.assign(static_cast<std::size_t>(eval.rows) * eval.cols, 0);
  for (int r = 0; r < core.rows; ++r)
    for (int c = 0; c < core.cols; ++c)
      cells.weights[static_cast<std::size_t>(core.top - eval.top + r) *
                        eval.cols +
                    core.left - eval.left + c] =
          weights[static_cast<std::size_t>(r) * core.cols + c];
  return cells;
}

std::vector<float> CropLatent(std::span<const float> latent, int grid_cols,
                              int channels, const RegionCells& cells) {
  if (grid_cols <= 0 || channels <= 0 || cells.top < 0 || cells.left < 0 ||
      cells.left + cells.cols > grid_cols ||
      latent.size() % (static_cast<std::size_t>(grid_cols) * channels) ||
      static_cast<std::size_t>(cells.top + cells.rows) * grid_cols * channels >
          latent.size())
    throw std::invalid_argument("region crop outside the latent");
  std::vector<float> out(static_cast<std::size_t>(cells.rows) * cells.cols *
                         channels);
  for (int r = 0; r < cells.rows; ++r)
    std::copy_n(latent.begin() + (static_cast<std::size_t>(cells.top + r) *
                                      grid_cols +
                                  cells.left) *
                                     channels,
                static_cast<std::size_t>(cells.cols) * channels,
                out.begin() +
                    static_cast<std::size_t>(r) * cells.cols * channels);
  return out;
}

void AccumulateRegion(std::span<float> sum, std::span<float> total,
                      std::span<const float> velocity, int grid_cols,
                      int channels, const RegionCells& cells) {
  const std::size_t rect = static_cast<std::size_t>(cells.rows) * cells.cols;
  if (grid_cols <= 0 || channels <= 0 || cells.weights.size() != rect ||
      cells.top < 0 || cells.left < 0 ||
      sum.size() != total.size() * channels ||
      total.size() % grid_cols != 0 || cells.left + cells.cols > grid_cols ||
      static_cast<std::size_t>(cells.top + cells.rows) * grid_cols >
          total.size() ||
      velocity.size() != (cells.crop ? rect : total.size()) * channels)
    throw std::invalid_argument("region blend shape mismatch");
  for (int r = 0; r < cells.rows; ++r)
    for (int c = 0; c < cells.cols; ++c) {
      const std::size_t i = static_cast<std::size_t>(r) * cells.cols + c;
      const float w = cells.weights[i];
      if (w == 0)
        continue;
      const std::size_t g =
          static_cast<std::size_t>(cells.top + r) * grid_cols + cells.left + c;
      const std::size_t source = cells.crop ? i : g;
      for (int k = 0; k < channels; ++k)
        sum[g * channels + k] += w * velocity[source * channels + k];
      total[g] += w;
    }
}

std::vector<float> FinishRegions(std::span<const float> sum,
                                 std::span<const float> total, int channels) {
  if (channels <= 0 || sum.size() != total.size() * channels)
    throw std::invalid_argument("region blend shape mismatch");
  std::vector<float> out(sum.size());
  for (std::size_t g = 0; g < total.size(); ++g)
    if (total[g] > 0)
      for (int k = 0; k < channels; ++k)
        out[g * channels + k] = sum[g * channels + k] / total[g];
  return out;
}

RegionMask RegionMaskFromImage(const Image& mask, MaskChannel channel,
                               bool invert) {
  const auto edit = EditValues(mask, channel);
  RegionMask out{mask.width, mask.height, std::vector<float>(edit.size())};
  for (std::size_t i = 0; i < edit.size(); ++i) {
    const float v = static_cast<float>(edit[i]) / 255.0F;
    out.values[i] = invert ? 1.0F - v : v;
  }
  return out;
}

}  // namespace gufo::models::qwen_image_21
