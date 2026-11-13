#ifndef GUFO_CLI_SERVE_ROUTER_RGBA_IMAGE_HPP_
#define GUFO_CLI_SERVE_ROUTER_RGBA_IMAGE_HPP_

#include <cstdint>
#include <gufo/ext/image.hpp>
#include <vector>

/// PNG encoding and Pillow-compatible Lanczos resizing of RGBA images for the
/// router's /v1/figuren checks. On release/1.4 these came from the in-tree
/// Qwen-Image model; on v0.31.0 Qwen-Image is an extension the router must
/// not link, so the two helpers are copied here unchanged from
/// src/models/qwen_image_21/image.cpp (upstream v0.31.0).
namespace gufo::router::image {

using RgbaImage = ext::ImageBuffer;

[[nodiscard]] std::vector<std::uint8_t> EncodePng(const RgbaImage& image);
[[nodiscard]] RgbaImage ResizeImage(const RgbaImage& image, int width,
                                    int height);

}  // namespace gufo::router::image

#endif  // GUFO_CLI_SERVE_ROUTER_RGBA_IMAGE_HPP_
