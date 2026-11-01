// CPU pre- and post-processing for the Real-ESRGAN "body" ONNX models that
// run on the NPU (or the ONNX Runtime CPU provider). It reproduces Take One's
// npu_upscale_worker.py (FilmStudio, .56) operation by operation, so a frame
// upscaled here matches the Take One output pixel for pixel when the body
// output is the same:
//   input   RGB8 -> float32 CHW x * float(1/255)
//   tiles   two row tiles of `tile` rows (0..tile-1 and h-tile..h-1); the
//           first keeps rows 0..h/2-1, the second h/2..h-1
//   body    [1, 48, h, w] float32 (the 16 / 32 convolutions before the
//           pixel shuffle) - this part runs in ONNX Runtime
//   post    DepthToSpace(4, CRD) + nearest x4 input, clip [0, 1],
//           uint8(v * 255 + 0.5) (float32, truncation)
//   fit     cv2.resize INTER_AREA (OpenCV's float area tables, round half to
//           even) plus a centre crop, see FitAreaCrop.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_REALESRGAN_CPU_POST_HPP_
#define GUFO_MODELS_REALESRGAN_CPU_POST_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gufo::models::realesrgan::cpu {

inline constexpr int kScale = 4;
inline constexpr int kBodyChannels = 3 * kScale * kScale;  // 48

/// One body pass: input rows [src_row, src_row + rows) of the frame; output
/// rows [keep_begin, keep_end) of the pass become frame rows
/// [dst_row, dst_row + keep_end - keep_begin).
struct TilePass {
  int src_row{0};
  int keep_begin{0};
  int keep_end{0};
  int dst_row{0};
};

/// Row tiles for a frame of `h` rows and a body compiled for `tile` rows.
/// tile == h: one pass. tile < h: Take One's two overlapping tiles (needs
/// tile >= h / 2 and h even); every tile keeps at least `tile - h / 2` rows
/// of context beyond the middle. Returns an empty plan when the sizes do not
/// fit.
[[nodiscard]] std::vector<TilePass> PlanRowTiles(int h, int tile);

/// RGB8 [h, w, 3] rows [row, row + rows) -> float32 [3, rows, w], each value
/// v * float(1 / 255) as numpy computes it.
void PackTileInput(const std::uint8_t* rgb24, int w, int row, int rows,
                   float* chw);

/// Copies the kept rows of one body output [48, rows, w] into the full body
/// output [48, h, w].
void StoreTileOutput(const float* tile_out, int w, int tile_rows,
                     const TilePass& pass, int h, float* body);

/// Body output [48, h, w] + RGB8 input [h, w, 3] -> RGB8 [4h, 4w, 3]:
/// DepthToSpace(4, CRD) + nearest x4 residual, clip, *255 + 0.5, truncate.
void ShuffleResidual(const float* body, const std::uint8_t* rgb24, int w,
                     int h, std::uint8_t* out24);

/// cv2.resize(src, (dst_w, dst_h), interpolation=INTER_AREA) for RGB8 when
/// both axes shrink (src >= dst). Integer factors use OpenCV's fast path
/// (2x2: (sum + 2) >> 2; else round(sum * float(1 / area))), the others its
/// float area tables. Returns false (with *error) for an enlarging axis.
bool ResizeAreaRgb8(const std::uint8_t* src, int src_w, int src_h, int src_stride,
                    int dst_w, int dst_h, std::uint8_t* dst, std::string* error);

/// How the x4 picture (in_w x in_h) becomes out_w x out_h with the aspect
/// ratio kept and the output filled:
///   - if cropping the source to the output aspect ratio is exact in whole
///     pixels, crop first (centre) and then area-resize: Take One 896x512 ->
///     rows 16..2031 of 3584x2048 -> 1920x1080;
///   - otherwise area-resize to the covering size round(in * s) and crop the
///     centre: Take One 832x512 -> 1950x1200 -> columns 15..1934 -> 1920x1200.
struct AreaCropPlan {
  int crop_x{0}, crop_y{0}, crop_w{0}, crop_h{0};  // source crop (step 1)
  int resize_w{0}, resize_h{0};                     // area resize target
  int out_x{0}, out_y{0};                           // crop of the resized
};
[[nodiscard]] AreaCropPlan PlanAreaCrop(int in_w, int in_h, int out_w,
                                        int out_h);

/// Plan + resize + crop: RGB8 in_w x in_h -> RGB8 out_w x out_h.
bool FitAreaCrop(const std::uint8_t* in24, int in_w, int in_h, int out_w,
                 int out_h, std::vector<std::uint8_t>* out24,
                 std::string* error);

/// Keeps the whole picture: area resize to the largest size that fits and
/// centres it on black bars (even offsets).
bool FitAreaPad(const std::uint8_t* in24, int in_w, int in_h, int out_w,
                int out_h, std::vector<std::uint8_t>* out24,
                std::string* error);

}  // namespace gufo::models::realesrgan::cpu

#endif  // GUFO_MODELS_REALESRGAN_CPU_POST_HPP_
