// Tile plan of the HIP Real-ESRGAN upscaler (host side, no HIP): the frame
// is cut into nx x ny cores; every core is padded by a halo of `halo`
// pixels (one per 3x3 convolution, clipped at the frame edge) so that the
// network's receptive field is complete and tiling reproduces the whole-
// frame result. The grid is the smallest one whose largest padded tile fits
// the pixel budget. Shared by upscaler.hip and the CPU tests.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_REALESRGAN_TILE_PLAN_HPP_
#define GUFO_MODELS_REALESRGAN_TILE_PLAN_HPP_

#include <algorithm>
#include <cstddef>

namespace gufo::models::realesrgan {

/// Largest padded tile (pixels) per network pass by default: 832x480 runs
/// as one tile.
inline constexpr std::size_t kDefaultTilePixels = 512 * 1024;

/// Upper bound of tiles per axis; beyond it the halo dominates and more
/// splitting cannot shrink the tile.
inline constexpr int kMaxTilesPerAxis = 64;

struct TileGrid {
  int width{0};
  int height{0};
  int halo{0};
  int nx{1};
  int ny{1};
  int core_w{0};  // ceil(width / nx)
  int core_h{0};  // ceil(height / ny)
  /// Pixels of the largest padded tile (what the activation buffers hold).
  std::size_t max_tile_pixels{0};
};

/// One tile: the core [cx0, cx1) x [cy0, cy1) whose output is kept and the
/// padded input [x0, x1) x [y0, y1) that the network sees.
struct TileRect {
  int cx0{0}, cy0{0}, cx1{0}, cy1{0};
  int x0{0}, y0{0}, x1{0}, y1{0};
  [[nodiscard]] constexpr bool empty() const noexcept {
    return cx0 >= cx1 || cy0 >= cy1;
  }
  [[nodiscard]] constexpr int width() const noexcept { return x1 - x0; }
  [[nodiscard]] constexpr int height() const noexcept { return y1 - y0; }
  [[nodiscard]] constexpr int core_width() const noexcept {
    return cx1 - cx0;
  }
  [[nodiscard]] constexpr int core_height() const noexcept {
    return cy1 - cy0;
  }
};

/// Padded side length of the widest tile when `size` pixels are split `n`
/// ways: an interior core carries the halo on both sides, an edge core only
/// on its inner side (the frame edge clips the other), and a trailing slot
/// the ceil division left short counts with its real width.
[[nodiscard]] constexpr int PaddedTileSide(int n, int size, int halo) noexcept {
  if (n <= 1 || size <= 0)
    return size;
  const int core = (size + n - 1) / n;
  int widest = 0;
  for (int i = 0; i < n; ++i) {
    const int c0 = i * core;
    if (c0 >= size)
      break;
    const int c1 = std::min(size, c0 + core);
    widest = std::max(widest, std::min(size, c1 + halo) - std::max(0, c0 - halo));
  }
  return widest;
}

/// Fewest tiles whose padded size fits `tile_pixels` (0: the default): the
/// longer tile side is split until the tile fits or the halo dominates.
[[nodiscard]] constexpr TileGrid PlanTileGrid(int width, int height, int halo,
                                              std::size_t tile_pixels =
                                                  0) noexcept {
  TileGrid grid;
  grid.width = width;
  grid.height = height;
  grid.halo = halo;
  if (tile_pixels == 0)
    tile_pixels = kDefaultTilePixels;
  const auto side_x = [&](int n) { return PaddedTileSide(n, width, halo); };
  const auto side_y = [&](int n) { return PaddedTileSide(n, height, halo); };
  int nx = 1, ny = 1;
  while (static_cast<std::size_t>(side_x(nx)) * side_y(ny) > tile_pixels &&
         (nx < width || ny < height)) {
    if ((side_x(nx) >= side_y(ny) && nx < width) || ny >= height)
      ++nx;
    else
      ++ny;
    if (nx > kMaxTilesPerAxis || ny > kMaxTilesPerAxis)
      break;
  }
  grid.nx = nx;
  grid.ny = ny;
  grid.core_w = width > 0 ? (width + nx - 1) / nx : 0;
  grid.core_h = height > 0 ? (height + ny - 1) / ny : 0;
  grid.max_tile_pixels =
      static_cast<std::size_t>(side_x(nx)) * static_cast<std::size_t>(side_y(ny));
  return grid;
}

/// Tile (tx, ty) of the grid; empty() for a trailing slot the ceil-division
/// left without pixels.
[[nodiscard]] constexpr TileRect TileAt(const TileGrid& grid, int tx,
                                        int ty) noexcept {
  TileRect tile;
  tile.cx0 = tx * grid.core_w;
  tile.cx1 = std::min(grid.width, tile.cx0 + grid.core_w);
  tile.cy0 = ty * grid.core_h;
  tile.cy1 = std::min(grid.height, tile.cy0 + grid.core_h);
  tile.x0 = std::max(0, tile.cx0 - grid.halo);
  tile.x1 = std::min(grid.width, tile.cx1 + grid.halo);
  tile.y0 = std::max(0, tile.cy0 - grid.halo);
  tile.y1 = std::min(grid.height, tile.cy1 + grid.halo);
  return tile;
}

}  // namespace gufo::models::realesrgan

#endif  // GUFO_MODELS_REALESRGAN_TILE_PLAN_HPP_
