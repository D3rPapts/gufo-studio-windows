#ifndef GUFO_CLI_VIDEO_UPSCALE_HPP_
#define GUFO_CLI_VIDEO_UPSCALE_HPP_

#include <span>

namespace gufo::cli {

/// `gufo video-upscale`: Real-ESRGAN upscaling of an MP4 or an image.
[[nodiscard]] int RunVideoUpscale(std::span<const char* const> args);

}  // namespace gufo::cli

#endif  // GUFO_CLI_VIDEO_UPSCALE_HPP_
