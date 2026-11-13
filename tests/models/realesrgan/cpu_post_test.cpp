// CPU tests for the NPU / CPU Real-ESRGAN host side (cpu_post.hpp): Take
// One's tiling, the pixel shuffle with the residual, and the OpenCV-exact
// INTER_AREA fit. Expected hashes come from Take One's numpy code and
// cv2.resize(..., INTER_AREA) (opencv-python 4.13.0, numpy 2.4.2) on the
// same deterministic inputs (LCG pixels, patterned body output).
// Written with AI assistance (Claude Code).
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/realesrgan/cpu_post.hpp"

namespace cpu = gufo::models::realesrgan::cpu;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
  if (!condition) {
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
  }
}

std::vector<std::uint8_t> LcgBytes(std::size_t n, std::uint32_t seed) {
  std::vector<std::uint8_t> out(n);
  std::uint32_t s = seed;
  for (std::size_t i = 0; i < n; ++i) {
    s = s * 1103515245U + 12345U;
    out[i] = static_cast<std::uint8_t>(s >> 24);
  }
  return out;
}

std::uint64_t Fnv1a(const std::vector<std::uint8_t>& bytes) {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  for (std::uint8_t b : bytes) {
    h ^= b;
    h *= 0x100000001b3ULL;
  }
  return h;
}

std::string HexOf(std::uint64_t v) {
  char text[24];
  std::snprintf(text, sizeof(text), "0x%016llx",
                static_cast<unsigned long long>(v));
  return text;
}

void ExpectHash(const std::vector<std::uint8_t>& bytes, std::uint64_t want,
                const std::string& what) {
  const std::uint64_t got = Fnv1a(bytes);
  Expect(got == want, what + ": hash " + HexOf(got) + " != " + HexOf(want));
}

/// Body output [48, h, w]: (k % 401 - 200) * float(1 / 1000).
std::vector<float> PatternBody(int w, int h) {
  std::vector<float> body(static_cast<std::size_t>(48) * h * w);
  const float step = static_cast<float>(1.0 / 1000.0);
  for (std::size_t k = 0; k < body.size(); ++k)
    body[k] = static_cast<float>(static_cast<int>(k % 401) - 200) * step;
  return body;
}

std::vector<std::uint8_t> Upscaled(int w, int h, std::uint32_t seed) {
  const auto rgb = LcgBytes(static_cast<std::size_t>(w) * h * 3, seed);
  const auto body = PatternBody(w, h);
  std::vector<std::uint8_t> up(static_cast<std::size_t>(w) * h * 16 * 3);
  cpu::ShuffleResidual(body.data(), rgb.data(), w, h, up.data());
  return up;
}

void TestTiles() {
  // Take One 896x512: two 288-row passes, rows 0..287 keep 0..255, rows
  // 224..511 keep 256..511 (output rows 32..287 of the second pass).
  const auto t = cpu::PlanRowTiles(512, 288);
  Expect(t.size() == 2, "896 plan has two passes");
  if (t.size() == 2) {
    Expect(t[0].src_row == 0 && t[0].keep_begin == 0 && t[0].keep_end == 256 &&
               t[0].dst_row == 0,
           "first 288 pass");
    Expect(t[1].src_row == 224 && t[1].keep_begin == 32 &&
               t[1].keep_end == 288 && t[1].dst_row == 256,
           "second 288 pass");
  }
  const auto t280 = cpu::PlanRowTiles(512, 280);  // 832x512
  Expect(t280.size() == 2 && t280[1].src_row == 232 &&
             t280[1].keep_begin == 24 && t280[1].keep_end == 280,
         "280 plan");
  Expect(cpu::PlanRowTiles(512, 512).size() == 1, "single pass");
  Expect(cpu::PlanRowTiles(512, 200).empty(), "tile below half is refused");
  Expect(cpu::PlanRowTiles(511, 300).empty(), "odd height is refused");
  Expect(cpu::PlanRowTiles(512, 600).empty(), "tile above frame is refused");
}

void TestTileAssembly() {
  // Tiles assembled from a per-row "network" give the same body as one
  // pass over the frame when every output row depends on its input row.
  const int w = 5, h = 8, tile = 6;
  const auto rgb = LcgBytes(static_cast<std::size_t>(w) * h * 3, 3);
  std::vector<float> full(static_cast<std::size_t>(48) * h * w);
  std::vector<float> in(static_cast<std::size_t>(3) * h * w);
  cpu::PackTileInput(rgb.data(), w, 0, h, in.data());
  for (int c = 0; c < 48; ++c)
    for (int i = 0; i < h * w; ++i)
      full[static_cast<std::size_t>(c) * h * w + i] =
          in[static_cast<std::size_t>(c % 3) * h * w + i] * (c + 1);
  std::vector<float> body(full.size(), -1.0F);
  for (const cpu::TilePass& pass : cpu::PlanRowTiles(h, tile)) {
    std::vector<float> tin(static_cast<std::size_t>(3) * tile * w);
    cpu::PackTileInput(rgb.data(), w, pass.src_row, tile, tin.data());
    std::vector<float> tout(static_cast<std::size_t>(48) * tile * w);
    for (int c = 0; c < 48; ++c)
      for (int i = 0; i < tile * w; ++i)
        tout[static_cast<std::size_t>(c) * tile * w + i] =
            tin[static_cast<std::size_t>(c % 3) * tile * w + i] * (c + 1);
    cpu::StoreTileOutput(tout.data(), w, tile, pass, h, body.data());
  }
  Expect(body == full, "tiled body equals the full body");
  // numpy: uint8 * float(1/255), exact float32 products.
  Expect(in[0] == static_cast<float>(rgb[0]) * static_cast<float>(1.0 / 255.0),
         "pack scale");
  Expect(in[static_cast<std::size_t>(h) * w] ==
             static_cast<float>(rgb[1]) * static_cast<float>(1.0 / 255.0),
         "pack is planar RGB");
}

void TestShuffle() {
  // DepthToSpace(4, CRD): channel c * 16 + i * 4 + j lands at (4y + i,
  // 4x + j, c); the input pixel is added to all 16; clip; *255 + 0.5;
  // truncate.
  const int w = 2, h = 1;
  std::vector<std::uint8_t> rgb = {10, 20, 30, 250, 128, 0};
  std::vector<float> body(static_cast<std::size_t>(48) * h * w, 0.0F);
  body[(0 * 16 + 1 * 4 + 2) * w + 1] = 0.5F;   // R, i=1, j=2, x=1
  body[(2 * 16 + 3 * 4 + 0) * w + 0] = -1.0F;  // B, i=3, j=0, x=0 -> 0
  body[(1 * 16 + 0) * w + 1] = 1.0F;           // G, i=0, j=0, x=1 -> 255
  std::vector<std::uint8_t> up(static_cast<std::size_t>(w) * 4 * h * 4 * 3);
  cpu::ShuffleResidual(body.data(), rgb.data(), w, h, up.data());
  const auto at = [&](int y, int x, int c) {
    return up[(static_cast<std::size_t>(y) * w * 4 + x) * 3 + c];
  };
  Expect(at(0, 0, 0) == 10 && at(0, 0, 1) == 20 && at(0, 0, 2) == 30,
         "residual only");
  Expect(at(1, 4 + 2, 0) == 255, "R shuffled and clipped");
  Expect(at(3, 0, 2) == 0, "B shuffled and clipped at 0");
  Expect(at(0, 4, 1) == 255, "G shuffled");
  Expect(at(2, 5, 0) == 250 && at(2, 5, 1) == 128 && at(2, 5, 2) == 0,
         "nearest residual");
  ExpectHash(Upscaled(24, 16, 7), 0xeae336a749fe6d1eULL,
             "Take One numpy shuffle 24x16");
  ExpectHash(Upscaled(26, 16, 9), 0xd7ef73307bf77403ULL,
             "Take One numpy shuffle 26x16");
}

void TestAreaResize() {
  std::string error;
  const auto run = [&](const std::vector<std::uint8_t>& src, int w, int h,
                       int ow, int oh) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(ow) * oh * 3);
    Expect(cpu::ResizeAreaRgb8(src.data(), w, h, w * 3, ow, oh, out.data(),
                               &error),
           "resize " + error);
    return out;
  };
  ExpectHash(run(LcgBytes(60 * 40 * 3, 3), 60, 40, 30, 20),
             0xf03b9832020f3816ULL, "cv2 INTER_AREA 2x2");
  ExpectHash(run(LcgBytes(60 * 45 * 3, 5), 60, 45, 20, 15),
             0x432185ac18271233ULL, "cv2 INTER_AREA 3x3");
  ExpectHash(run(LcgBytes(112 * 56 * 3, 11), 112, 56, 60, 30),
             0x3c3b4612a530925eULL, "cv2 INTER_AREA 28/15");
  std::vector<std::uint8_t> out(12);
  Expect(!cpu::ResizeAreaRgb8(out.data(), 2, 2, 6, 4, 4, out.data(), &error),
         "enlarging is refused");
}

void TestFit() {
  // Take One's two formats.
  const auto p896 = cpu::PlanAreaCrop(3584, 2048, 1920, 1080);
  Expect(p896.crop_y == 16 && p896.crop_h == 2016 && p896.crop_x == 0 &&
             p896.crop_w == 3584 && p896.resize_w == 1920 &&
             p896.resize_h == 1080 && p896.out_x == 0 && p896.out_y == 0,
         "896x512 -> rows 16..2031 -> 1920x1080");
  const auto p832 = cpu::PlanAreaCrop(3328, 2048, 1920, 1200);
  Expect(p832.crop_w == 3328 && p832.crop_h == 2048 && p832.resize_w == 1950 &&
             p832.resize_h == 1200 && p832.out_x == 15 && p832.out_y == 0,
         "832x512 -> 1950x1200 -> columns 15..1934");
  std::string error;
  std::vector<std::uint8_t> out;
  const auto up = Upscaled(24, 16, 7);  // 96x64
  Expect(cpu::FitAreaCrop(up.data(), 96, 64, 60, 30, &out, &error),
         "fit crop " + error);
  ExpectHash(out, 0xf0e9b7f5039befe8ULL, "crop rows then INTER_AREA");
  const auto up2 = Upscaled(26, 16, 9);  // 104x64
  Expect(cpu::FitAreaCrop(up2.data(), 104, 64, 75, 47, &out, &error),
         "fit crop 2 " + error);
  ExpectHash(out, 0x3f14952e811cba25ULL, "INTER_AREA then crop columns");
  Expect(cpu::FitAreaPad(up.data(), 96, 64, 64, 64, &out, &error) &&
             out.size() == 64u * 64u * 3u && out[0] == 0,
         "pad keeps black bars");
}

}  // namespace

int main() {
  TestTiles();
  TestTileAssembly();
  TestShuffle();
  TestAreaResize();
  TestFit();
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "realesrgan cpu_post: all tests passed\n";
  return 0;
}
