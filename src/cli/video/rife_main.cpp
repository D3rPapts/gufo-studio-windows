#include <cstddef>
#include <span>

#include "src/cli/video/video_interpolate.hpp"

int main(int argc, char* argv[]) {
  const std::span<const char* const> args(argv, static_cast<std::size_t>(argc));
  return gufo::cli::RunVideoInterpolate(args.subspan(1));
}
