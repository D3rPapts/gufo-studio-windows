#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include "src/models/wan22/point_render.hpp"

namespace gufo::models::wan22 {
namespace {

constexpr int kThreads = 256;

struct View {
  float r[9];  // camera-to-world rotation, row-major
  float c[3];  // centre
  float fx, fy, cx, cy;
  int width, height;
  float near;
};

// Splats every point: key = (depth bits << 32) | point index, atomic min.
__global__ void SplatKernel(const float4* points, int count, View v,
                            unsigned long long* zbuffer) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float4 p = points[i];  // xyz + world footprint (one source pixel)
  const float dx = p.x - v.c[0], dy = p.y - v.c[1], dz = p.z - v.c[2];
  // world -> camera: R^T (p - c)
  const float x = v.r[0] * dx + v.r[3] * dy + v.r[6] * dz;
  const float y = v.r[1] * dx + v.r[4] * dy + v.r[7] * dz;
  const float z = v.r[2] * dx + v.r[5] * dy + v.r[8] * dz;
  if (z <= v.near)
    return;
  const float u = v.fx * x / z + v.cx;
  const float w = v.fy * y / z + v.cy;
  const float half = fminf(fmaxf(0.5F * 1.15F * v.fx * p.w / z, 0.5F), 4.0F);
  const int x0 = max(static_cast<int>(floorf(u - half + 0.5F)), 0);
  const int x1 = min(static_cast<int>(floorf(u + half - 0.5F)), v.width - 1);
  const int y0 = max(static_cast<int>(floorf(w - half + 0.5F)), 0);
  const int y1 = min(static_cast<int>(floorf(w + half - 0.5F)), v.height - 1);
  const unsigned long long key =
      (static_cast<unsigned long long>(__float_as_uint(z)) << 32) |
      static_cast<unsigned int>(i);
  for (int yy = y0; yy <= y1; ++yy)
    for (int xx = x0; xx <= x1; ++xx)
      atomicMin(zbuffer + static_cast<std::size_t>(yy) * v.width + xx, key);
}

__global__ void ResolveKernel(const unsigned long long* zbuffer,
                              const uchar4* colours, int pixels,
                              std::uint8_t* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pixels)
    return;
  const unsigned long long key = zbuffer[i];
  std::uint8_t r = 0, g = 0, b = 0;
  if (key != ~0ULL) {
    const uchar4 c = colours[key & 0xFFFFFFFFULL];
    r = c.x;
    g = c.y;
    b = c.z;
  }
  out[i * 3] = r;
  out[i * 3 + 1] = g;
  out[i * 3 + 2] = b;
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

bool Check(hipError_t status, const char* operation, std::string* error) {
  if (status == hipSuccess)
    return true;
  SetError(error, std::string("point render ") + operation + ": " +
                      hipGetErrorString(status));
  return false;
}

}  // namespace

double SceneDepth(const MogeResult& g) {
  std::vector<float> valid;
  for (std::size_t i = 0; i < g.depth.size(); ++i)
    if (g.mask[i] != 0 && std::isfinite(g.depth[i]))
      valid.push_back(g.depth[i]);
  if (valid.empty())
    return 0;
  std::nth_element(valid.begin(), valid.begin() + valid.size() / 2, valid.end());
  return valid[valid.size() / 2];
}

bool RenderGuide(const MogeResult& g, std::span<const std::uint8_t> rgb,
                 const std::vector<CameraPose>& poses,
                 std::vector<std::uint8_t>* frames, std::string* error) {
  const int w = g.width, h = g.height;
  const std::size_t pixels = static_cast<std::size_t>(w) * h;
  if (rgb.size() != pixels * 3 || g.depth.size() != pixels || poses.empty()) {
    SetError(error, "point render: image and geometry differ in size");
    return false;
  }
  float far = 0;
  for (std::size_t i = 0; i < pixels; ++i)
    if (g.mask[i] != 0 && std::isfinite(g.depth[i]))
      far = std::max(far, g.depth[i]);
  if (!(far > 0)) {
    SetError(error, "point render: no valid depth");
    return false;
  }
  far *= 3.0F;
  // One point per pixel: (x, y, z, footprint = depth / fx), colour.
  std::vector<float> points(pixels * 4);
  std::vector<std::uint8_t> colours(pixels * 4, 255);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * w + x;
      double depth = g.depth[i];
      if (g.mask[i] == 0 || !std::isfinite(depth))
        depth = far;
      const double u = (x + 0.5 - g.cx) / g.fx, v = (y + 0.5 - g.cy) / g.fy;
      points[i * 4] = static_cast<float>(u * depth);
      points[i * 4 + 1] = static_cast<float>(v * depth);
      points[i * 4 + 2] = static_cast<float>(depth);
      points[i * 4 + 3] = static_cast<float>(depth / g.fx);
      for (int c = 0; c < 3; ++c)
        colours[i * 4 + c] = rgb[i * 3 + c];
    }
  void *points_d = nullptr, *colours_d = nullptr, *zbuffer = nullptr,
       *frame_d = nullptr;
  struct Free {
    void** p[4];
    ~Free() {
      for (void** q : p)
        if (*q != nullptr)
          (void)hipFree(*q);
    }
  } release{{&points_d, &colours_d, &zbuffer, &frame_d}};
  if (!Check(hipMalloc(&points_d, points.size() * 4), "allocation", error) ||
      !Check(hipMalloc(&colours_d, colours.size()), "allocation", error) ||
      !Check(hipMalloc(&zbuffer, pixels * 8), "allocation", error) ||
      !Check(hipMalloc(&frame_d, pixels * 3), "allocation", error) ||
      !Check(hipMemcpy(points_d, points.data(), points.size() * 4,
                       hipMemcpyHostToDevice),
             "upload", error) ||
      !Check(hipMemcpy(colours_d, colours.data(), colours.size(),
                       hipMemcpyHostToDevice),
             "upload", error))
    return false;
  frames->assign(pixels * 3 * poses.size(), 0);
  const double scene = SceneDepth(g);
  for (std::size_t f = 0; f < poses.size(); ++f) {
    const CameraPose& pose = poses[f];
    View v{};
    for (int j = 0; j < 9; ++j)
      v.r[j] = static_cast<float>(pose.rotation[j]);
    for (int j = 0; j < 3; ++j)
      v.c[j] = static_cast<float>(pose.centre[j]);
    v.fx = static_cast<float>(g.fx * pose.zoom);
    v.fy = static_cast<float>(g.fy * pose.zoom);
    v.cx = static_cast<float>(g.cx);
    v.cy = static_cast<float>(g.cy);
    v.width = w;
    v.height = h;
    v.near = static_cast<float>(0.01 * scene);
    if (!Check(hipMemset(zbuffer, 0xFF, pixels * 8), "clear", error))
      return false;
    SplatKernel<<<static_cast<unsigned>((pixels + kThreads - 1) / kThreads),
                  kThreads>>>(static_cast<const float4*>(points_d),
                              static_cast<int>(pixels), v,
                              static_cast<unsigned long long*>(zbuffer));
    ResolveKernel<<<static_cast<unsigned>((pixels + kThreads - 1) / kThreads),
                    kThreads>>>(static_cast<const unsigned long long*>(zbuffer),
                                static_cast<const uchar4*>(colours_d),
                                static_cast<int>(pixels),
                                static_cast<std::uint8_t*>(frame_d));
    if (!Check(hipGetLastError(), "render", error) ||
        !Check(hipMemcpy(frames->data() + pixels * 3 * f, frame_d, pixels * 3,
                         hipMemcpyDeviceToHost),
               "readback", error))
      return false;
  }
  return true;
}

}  // namespace gufo::models::wan22
