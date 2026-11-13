// CCIP + face detector against onnxruntime (reference/make_oracle.py).
// Env: GUFO_CCIP_MODEL (dir with ccip_caformer.*), GUFO_CCIP_FACE (dir with
// face_yolov8n.*), GUFO_CCIP_ORACLE (dir with manifest.json), optional
// GUFO_CCIP_THREADS. Exit 77 (skip) without them.
// Pass: preprocessing bit-exact, final embedding cos >= 0.999 per input,
// same number of faces and every box within 1 px.
// Written with AI assistance (Claude Code).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gufo/ext/json.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ccip/ccip.hpp"
#include "ccip/tensors.hpp"

namespace ccip = gufo::models::ccip;

namespace {

struct Cmp {
  double max_abs{0}, rel_l2{0}, cos{0};
};

Cmp Compare(const float* a, const float* b, std::size_t n) {
  Cmp c;
  double ab = 0, aa = 0, bb = 0, dd = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    c.max_abs = std::max(c.max_abs, std::fabs(d));
    dd += d * d;
    ab += static_cast<double>(a[i]) * b[i];
    aa += static_cast<double>(a[i]) * a[i];
    bb += static_cast<double>(b[i]) * b[i];
  }
  c.rel_l2 = bb > 0 ? std::sqrt(dd / bb) : std::sqrt(dd);
  c.cos = (aa > 0 && bb > 0) ? ab / std::sqrt(aa * bb) : 0;
  return c;
}

const char* Env(const char* name) {
  const char* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? v : nullptr;
}

ccip::RgbImage ImageFrom(const ccip::TensorFile& f) {
  const ccip::Tensor* t = f.Find("rgb");
  if (t == nullptr || t->shape.size() != 3 || t->shape[2] != 3) throw std::runtime_error("rgb tensor");
  ccip::RgbImage img;
  img.height = static_cast<int>(t->shape[0]);
  img.width = static_cast<int>(t->shape[1]);
  img.rgb = t->u8;
  return img;
}

double Ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

int main() {
  const char* model_dir = Env("GUFO_CCIP_MODEL");
  const char* face_dir = Env("GUFO_CCIP_FACE");
  const char* oracle_dir = Env("GUFO_CCIP_ORACLE");
  if (!model_dir || !face_dir || !oracle_dir) {
    std::printf("skip: set GUFO_CCIP_MODEL, GUFO_CCIP_FACE, GUFO_CCIP_ORACLE\n");
    return 77;
  }
#ifdef _OPENMP
  if (const char* t = Env("GUFO_CCIP_THREADS")) omp_set_num_threads(std::atoi(t));
  std::printf("threads %d\n", omp_get_max_threads());
#endif
  const std::filesystem::path odir(oracle_dir);
  std::ifstream mf(odir / "manifest.json", std::ios::binary);
  if (!mf) {
    std::printf("missing manifest.json\n");
    return 1;
  }
  std::stringstream ss;
  ss << mf.rdbuf();
  const gufo::json::Value manifest = gufo::json::parse(ss.str());
  bool ok = true;
  std::string error;

  auto t0 = std::chrono::steady_clock::now();
  auto model = ccip::CcipModel::Load(model_dir, &error);
  if (!model) {
    std::printf("load ccip: %s\n", error.c_str());
    return 1;
  }
  auto t1 = std::chrono::steady_clock::now();
  std::printf("ccip loaded in %.0f ms\n", Ms(t0, t1));

  for (const auto& item : manifest.find("ccip")->items()) {
    ccip::TensorFile f;
    if (!f.Load(odir / item.member_str("file"), &error)) {
      std::printf("oracle: %s\n", error.c_str());
      return 1;
    }
    const ccip::RgbImage img = ImageFrom(f);
    const std::vector<float> x = ccip::CcipInput(img);
    const auto& ref_x = f.F32("input");
    std::size_t px_diff = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      // One uint8 step is 2/255 after normalisation.
      if (std::fabs(x[i] - ref_x[i]) > 1e-6f) ++px_diff;
    }
    std::printf("\n[ccip] %s (%dx%d): preprocessing %zu of %zu values differ\n",
                item.member_str("source").c_str(), img.width, img.height, px_diff, x.size());
    if (px_diff != 0) ok = false;
    ccip::Taps taps;
    const auto a = std::chrono::steady_clock::now();
    const std::vector<float> emb = model->Forward(x.data(), &taps);
    const auto b = std::chrono::steady_clock::now();
    for (const char* tap : {"stem", "stage0", "stage1", "stage2_block0", "stage2", "stage3"}) {
      const auto& ref = f.F32(tap);
      const auto& got = taps.at(tap);
      if (got.size() != ref.size()) {
        std::printf("  %-14s size %zu != %zu\n", tap, got.size(), ref.size());
        ok = false;
        continue;
      }
      const Cmp c = Compare(got.data(), ref.data(), ref.size());
      std::printf("  %-14s max_abs %.3e  rel_l2 %.3e  cos %.7f\n", tap, c.max_abs, c.rel_l2, c.cos);
    }
    const auto& ref_e = f.F32("embedding");
    const Cmp c = Compare(emb.data(), ref_e.data(), ref_e.size());
    std::printf("  %-14s max_abs %.3e  rel_l2 %.3e  cos %.7f  diff %.6f  (%.0f ms)\n", "embedding",
                c.max_abs, c.rel_l2, c.cos, ccip::CcipDiff(emb, ref_e), Ms(a, b));
    if (!(c.cos >= 0.999)) ok = false;
  }

  t0 = std::chrono::steady_clock::now();
  auto det = ccip::FaceDetector::Load(face_dir, &error);
  if (!det) {
    std::printf("load face detector: %s\n", error.c_str());
    return 1;
  }
  t1 = std::chrono::steady_clock::now();
  std::printf("\nface detector loaded in %.0f ms\n", Ms(t0, t1));
  for (const auto& item : manifest.find("faces")->items()) {
    ccip::TensorFile f;
    if (!f.Load(odir / item.member_str("file"), &error)) {
      std::printf("oracle: %s\n", error.c_str());
      return 1;
    }
    const ccip::RgbImage img = ImageFrom(f);
    const auto a = std::chrono::steady_clock::now();
    const ccip::LetterboxInput lb = ccip::Letterbox(img, 640);
    int anchors = 0;
    const std::vector<float> raw = det->Raw(lb.chw.data(), lb.size, &anchors);
    const std::vector<ccip::Face> faces =
        ccip::DecodeFaces(raw, anchors, lb, img.width, img.height, 0.25f, 0.7f);
    const auto b = std::chrono::steady_clock::now();
    const auto& ref_x = f.F32("input");
    std::size_t px_diff = 0;
    for (std::size_t i = 0; i < lb.chw.size() && i < ref_x.size(); ++i) {
      if (std::fabs(lb.chw[i] - ref_x[i]) > 1e-6f) ++px_diff;
    }
    std::printf("\n[faces] %s (%dx%d): letterbox %zu values differ (r %.6f dx %d dy %d)\n",
                item.member_str("source").c_str(), img.width, img.height, px_diff, lb.ratio, lb.dx,
                lb.dy);
    if (px_diff != 0 || lb.chw.size() != ref_x.size()) ok = false;
    const auto& ref_raw = f.F32("raw");
    if (ref_raw.size() != raw.size()) {
      std::printf("  raw size %zu != %zu\n", raw.size(), ref_raw.size());
      ok = false;
      continue;
    }
    double coord_max = 0, score_max = 0;
    for (std::size_t i = 0; i < raw.size(); ++i) {
      const double d = std::fabs(static_cast<double>(raw[i]) - ref_raw[i]);
      if (i < static_cast<std::size_t>(4) * anchors) {
        coord_max = std::max(coord_max, d);
      } else {
        score_max = std::max(score_max, d);
      }
    }
    std::printf("  raw: anchors %d, max |coord| diff %.3e px, max |score| diff %.3e\n", anchors,
                coord_max, score_max);
    const ccip::Tensor* rb = f.Find("boxes");
    const std::size_t nref = rb ? static_cast<std::size_t>(rb->shape[0]) : 0;
    std::printf("  faces: %zu (oracle %zu)  %.0f ms incl. letterbox + NMS\n", faces.size(), nref,
                Ms(a, b));
    if (faces.size() != nref) ok = false;
    for (std::size_t i = 0; i < faces.size() && i < nref; ++i) {
      double dmax = 0;
      for (int k = 0; k < 4; ++k) {
        dmax = std::max(dmax, std::fabs(static_cast<double>(faces[i].box[k]) - rb->f32[i * 5 + k]));
      }
      const double ds = std::fabs(static_cast<double>(faces[i].score) - rb->f32[i * 5 + 4]);
      std::printf("   #%zu [%.1f %.1f %.1f %.1f] score %.4f  max box diff %.4f px  score diff %.2e\n", i,
                  faces[i].box[0], faces[i].box[1], faces[i].box[2], faces[i].box[3], faces[i].score,
                  dmax, ds);
      if (!(dmax <= 1.0)) ok = false;
    }
  }
  std::printf("\n%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
