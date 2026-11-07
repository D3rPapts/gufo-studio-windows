// CCIP (deepghs, "ccip-caformer-24-randaug-pruned") character fingerprints and
// the deepghs anime face detector (YOLOv8n, face_detect_v1.4_n), native CPU
// fp32 runtime. Weights come from reference/convert_weights.py (ONNX ->
// safetensors + JSON architecture). Written with AI assistance (Claude Code).
//
//   CcipModel   CAFormer-S36-like backbone (SepConv stages 0-1, attention
//               stages 2-3, res_scale in 2-3, StarReLU, bias-free LayerNorm)
//               + CLIP AttentionPool2d head -> 768 features.
//   FaceDetector YOLOv8n, letterbox 640, DFL decode, conf 0.25, NMS IoU 0.7.
//   CcipDiff    model_metrics.onnx: clip((1 - cos) / 2, 0, 1); same
//               character below kCcipThreshold.
#ifndef GUFO_MODELS_CCIP_CCIP_HPP_
#define GUFO_MODELS_CCIP_CCIP_HPP_

#include <array>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "src/models/ccip/image_ops.hpp"

namespace gufo::models::ccip {

inline constexpr double kCcipThreshold = 0.17847511429108218;

// Intermediate activations (NHWC, flattened) for oracle tests.
using Taps = std::map<std::string, std::vector<float>>;

class CcipModel {
public:
  virtual ~CcipModel() = default;
  // `dir` holds ccip_caformer.safetensors + ccip_caformer.json.
  [[nodiscard]] static std::unique_ptr<CcipModel> Load(const std::filesystem::path& dir,
                                                       std::string* error);
  // Raw 768 features for a 3x384x384 normalised input. Taps receive "stem",
  // "stage0".."stage3" and "stage2_block0" when non-null.
  [[nodiscard]] virtual std::vector<float> Forward(const float* chw, Taps* taps = nullptr) const = 0;
  // L2-normalised features for an RGB image (CcipInput preprocessing).
  [[nodiscard]] std::vector<float> Embed(const RgbImage& image) const;
  [[nodiscard]] virtual int OutputDim() const = 0;
};

// diff = clip((1 - cos) / 2, 0, 1) on (not necessarily normalised) features.
[[nodiscard]] double CcipDiff(const std::vector<float>& a, const std::vector<float>& b);

struct Face {
  std::array<float, 4> box{};  // x0, y0, x1, y1 in image pixels, clamped
  float score{0};
};

class FaceDetector {
public:
  virtual ~FaceDetector() = default;
  // `dir` holds face_yolov8n.safetensors + face_yolov8n.json.
  [[nodiscard]] static std::unique_ptr<FaceDetector> Load(const std::filesystem::path& dir,
                                                          std::string* error);
  // Raw head output [5, anchors] (cx, cy, w, h, score) for a 3xSxS input.
  [[nodiscard]] virtual std::vector<float> Raw(const float* chw, int size, int* anchors) const = 0;
  [[nodiscard]] std::vector<Face> Detect(const RgbImage& image, float conf = 0.25f,
                                         float iou = 0.7f) const;
};

// OpenMP threads for CCIP work started from the calling thread (no-op
// without OpenMP). The setting is per calling thread.
void SetCpuThreads(int threads);

// The face nearest to the horizontal centre (and the upper half) of the
// image, as ccip_eval.head picks it: |cx - W/2| / W + 0.3 * max(0, cy/H - 0.5).
// Index into `faces`, -1 when empty.
[[nodiscard]] int CentreFace(const std::vector<Face>& faces, int width, int height);

// ccip_ref.faces post-processing of a raw output (exposed for tests).
[[nodiscard]] std::vector<Face> DecodeFaces(const std::vector<float>& raw, int anchors,
                                            const LetterboxInput& lb, int width, int height,
                                            float conf, float iou);

}  // namespace gufo::models::ccip

#endif  // GUFO_MODELS_CCIP_CCIP_HPP_
