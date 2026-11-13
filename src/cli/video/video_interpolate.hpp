#ifndef GUFO_CLI_VIDEO_INTERPOLATE_HPP_
#define GUFO_CLI_VIDEO_INTERPOLATE_HPP_

#include <span>

namespace gufo::cli {

/// `gufo video-interpolate`: RIFE frame interpolation of an MP4.
[[nodiscard]] int RunVideoInterpolate(std::span<const char* const> args);

}  // namespace gufo::cli

#endif  // GUFO_CLI_VIDEO_INTERPOLATE_HPP_
