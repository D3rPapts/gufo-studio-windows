// CPU tests for the HIP upscaler's tile plan (tile_plan.hpp): every pixel
// of the frame belongs to exactly one core, every core carries a halo of
// one pixel per convolution (clipped at the frame edge), the largest padded
// tile fits the budget, and the familiar frame sizes split as documented.
// Written with AI assistance (Claude Code).
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/realesrgan/tile_plan.hpp"

namespace esr = gufo::models::realesrgan;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
  if (!condition) {
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
  }
}

std::string Describe(const esr::TileGrid& grid) {
  return std::to_string(grid.width) + "x" + std::to_string(grid.height) +
         " halo " + std::to_string(grid.halo) + " -> " +
         std::to_string(grid.nx) + "x" + std::to_string(grid.ny) +
         " tiles, max " + std::to_string(grid.max_tile_pixels) + " px";
}

// Coverage, overlap and budget invariants of one plan.
void CheckInvariants(int w, int h, int halo, std::size_t budget) {
  const esr::TileGrid grid = esr::PlanTileGrid(w, h, halo, budget);
  const std::string name = Describe(grid);
  std::vector<int> covered(static_cast<std::size_t>(w) * h, 0);
  std::size_t largest = 0;
  int non_empty = 0;
  for (int ty = 0; ty < grid.ny; ++ty) {
    for (int tx = 0; tx < grid.nx; ++tx) {
      const esr::TileRect tile = esr::TileAt(grid, tx, ty);
      if (tile.empty())
        continue;
      ++non_empty;
      // The core lies inside the frame, the padded tile inside the frame.
      Expect(tile.cx0 >= 0 && tile.cy0 >= 0 && tile.cx1 <= w && tile.cy1 <= h,
             name + ": core inside the frame");
      Expect(tile.x0 >= 0 && tile.y0 >= 0 && tile.x1 <= w && tile.y1 <= h,
             name + ": padded tile inside the frame");
      // The halo is exactly `halo` pixels unless the frame edge clips it.
      Expect(tile.x0 == std::max(0, tile.cx0 - halo) &&
                 tile.x1 == std::min(w, tile.cx1 + halo) &&
                 tile.y0 == std::max(0, tile.cy0 - halo) &&
                 tile.y1 == std::min(h, tile.cy1 + halo),
             name + ": halo of " + std::to_string(halo));
      // Adjacent cores overlap in their padded inputs by 2 * halo.
      if (tx + 1 < grid.nx) {
        const esr::TileRect right = esr::TileAt(grid, tx + 1, ty);
        if (!right.empty())
          Expect(tile.x1 - right.x0 == std::min(tile.cx1 + halo, w) -
                                           std::max(0, tile.cx1 - halo),
                 name + ": horizontal overlap is 2 * halo (edge-clipped)");
      }
      for (int y = tile.cy0; y < tile.cy1; ++y)
        for (int x = tile.cx0; x < tile.cx1; ++x)
          ++covered[static_cast<std::size_t>(y) * w + x];
      largest = std::max(largest, static_cast<std::size_t>(tile.width()) *
                                      static_cast<std::size_t>(tile.height()));
    }
  }
  bool once = true;
  for (int c : covered)
    once = once && c == 1;
  Expect(once, name + ": every pixel in exactly one core");
  Expect(largest == grid.max_tile_pixels,
         name + ": max_tile_pixels equals the largest padded tile (" +
             std::to_string(largest) + ")");
  Expect(largest <= budget || (grid.nx >= esr::kMaxTilesPerAxis ||
                               grid.ny >= esr::kMaxTilesPerAxis ||
                               (grid.nx >= w && grid.ny >= h)),
         name + ": largest tile fits the budget " + std::to_string(budget));
  Expect(non_empty >= 1, name + ": at least one tile");
}

void TestSingleTile() {
  // 832x480 with the default budget runs as one tile, no halo added.
  const esr::TileGrid grid = esr::PlanTileGrid(832, 480, 33);
  Expect(grid.nx == 1 && grid.ny == 1, "832x480: one tile");
  Expect(grid.max_tile_pixels == 832U * 480U, "832x480: tile is the frame");
  const esr::TileRect tile = esr::TileAt(grid, 0, 0);
  Expect(tile.x0 == 0 && tile.y0 == 0 && tile.x1 == 832 && tile.y1 == 480 &&
             tile.cx0 == 0 && tile.cy0 == 0 && tile.cx1 == 832 &&
             tile.cy1 == 480,
         "832x480: tile covers the frame without halo");
  Expect(esr::TileAt(grid, 1, 0).empty(), "832x480: slot (1,0) is empty");
}

void TestSplitsLongerSide() {
  // 1920x1080 (~2 Mi pixels) does not fit 512 Ki: the longer side splits
  // first; the halo of 33 widens each tile by 66 on the split axis.
  const esr::TileGrid grid = esr::PlanTileGrid(1920, 1080, 33);
  Expect(grid.nx >= 2, "1920x1080: width is split");
  Expect(grid.max_tile_pixels <= esr::kDefaultTilePixels,
         "1920x1080: tiles fit the default budget");
  const esr::TileRect first = esr::TileAt(grid, 0, 0);
  const esr::TileRect second = esr::TileAt(grid, 1, 0);
  Expect(first.x0 == 0 && first.x1 == std::min(1920, first.cx1 + 33),
         "1920x1080: first tile has a right halo only");
  Expect(second.x0 == second.cx0 - 33, "1920x1080: second tile has a left halo");
  Expect(first.cx1 == second.cx0, "1920x1080: cores are adjacent");
  Expect(esr::PaddedTileSide(grid.nx, 1920, 33) == first.width() ||
             esr::PaddedTileSide(grid.nx, 1920, 33) == second.width(),
         "1920x1080: PaddedTileSide matches a real tile width");
}

void TestHaloDominates() {
  // A budget below the halo footprint stops at kMaxTilesPerAxis instead of
  // looping forever.
  const esr::TileGrid grid = esr::PlanTileGrid(4000, 4000, 33, 100);
  Expect(grid.nx > esr::kMaxTilesPerAxis || grid.ny > esr::kMaxTilesPerAxis,
         "tiny budget: planning stops at the per-axis limit");
  CheckInvariants(4000, 4000, 33, 100);
}

void TestTinyFrames() {
  // Frames smaller than the halo still produce one full tile.
  const esr::TileGrid grid = esr::PlanTileGrid(8, 6, 17, 16);
  Expect(grid.max_tile_pixels == 48, "8x6 with budget 16: one tile anyway");
  CheckInvariants(8, 6, 17, 16);
  CheckInvariants(1, 1, 33, 1);
}

void TestInvariantsSweep() {
  const int halos[] = {17, 33};
  const std::size_t budgets[] = {esr::kDefaultTilePixels, 65536, 4096};
  const int sizes[][2] = {{832, 480}, {896, 512}, {1280, 720}, {1920, 1080},
                          {3840, 2160}, {641, 359}, {100, 1000}, {1000, 100}};
  for (int halo : halos)
    for (std::size_t budget : budgets)
      for (const auto& size : sizes)
        CheckInvariants(size[0], size[1], halo, budget);
}

}  // namespace

int main() {
  TestSingleTile();
  TestSplitsLongerSide();
  TestHaloDominates();
  TestTinyFrames();
  TestInvariantsSweep();
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "realesrgan tile_plan: all tests passed\n";
  return 0;
}
