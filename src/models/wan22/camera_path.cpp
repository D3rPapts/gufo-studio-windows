#include "src/models/wan22/camera_path.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace gufo::models::wan22 {
namespace {

constexpr double kPi = 3.14159265358979323846;

using Matrix = std::array<double, 9>;

Matrix Multiply(const Matrix& a, const Matrix& b) {
  Matrix c{};
  for (int r = 0; r < 3; ++r)
    for (int k = 0; k < 3; ++k)
      for (int j = 0; j < 3; ++j)
        c[r * 3 + j] += a[r * 3 + k] * b[k * 3 + j];
  return c;
}

// Camera-to-world rotation from yaw (turn right), pitch (tilt up) and roll
// (clockwise), in degrees: Ry(yaw) Rx(pitch) Rz(roll), OpenCV axes.
Matrix Rotation(double yaw, double pitch, double roll) {
  const double y = yaw * kPi / 180, p = pitch * kPi / 180, r = roll * kPi / 180;
  const Matrix ry{std::cos(y),  0, std::sin(y), 0, 1, 0,
                  -std::sin(y), 0, std::cos(y)};
  const Matrix rx{1, 0,           0,          0, std::cos(p), -std::sin(p),
                  0, std::sin(p), std::cos(p)};
  const Matrix rz{
      std::cos(r), -std::sin(r), 0, std::sin(r), std::cos(r), 0, 0, 0, 1};
  return Multiply(Multiply(ry, rx), rz);
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

}  // namespace

bool ParseKeyframes(const std::string& text,
                    std::vector<CameraKeyframe>* keyframes,
                    std::string* error) {
  keyframes->clear();
  std::stringstream items(text);
  std::string item;
  while (std::getline(items, item, ';')) {
    if (item.empty())
      continue;
    std::vector<double> values;
    std::stringstream fields(item);
    std::string field;
    while (std::getline(fields, field, ','))
      values.push_back(std::atof(field.c_str()));
    if (values.size() < 4 || values.size() > 8) {
      SetError(
          error,
          "camera keyframe needs frame,x,y,z[,yaw,pitch,roll,zoom]: " + item);
      return false;
    }
    CameraKeyframe k;
    k.frame = static_cast<int>(values[0]);
    k.position = {values[1], values[2], values[3]};
    k.yaw = values.size() > 4 ? values[4] : 0;
    k.pitch = values.size() > 5 ? values[5] : 0;
    k.roll = values.size() > 6 ? values[6] : 0;
    k.zoom = values.size() > 7 ? values[7] : 1;
    keyframes->push_back(k);
  }
  std::sort(keyframes->begin(), keyframes->end(),
            [](const auto& a, const auto& b) { return a.frame < b.frame; });
  if (keyframes->empty()) {
    SetError(error, "no camera keyframes");
    return false;
  }
  return true;
}

bool BuildCameraPath(const CameraSpec& spec, int frames, double d,
                     std::vector<CameraPose>* poses, std::string* error) {
  poses->assign(std::max(frames, 1), CameraPose{});
  if (frames < 1 || !(d > 0)) {
    SetError(error, "camera path: invalid frame count or scene depth");
    return false;
  }
  const std::string& t = spec.type;
  if (t == "poses") {
    const auto& k = spec.keyframes;
    if (k.empty()) {
      SetError(error, "camera path: no keyframes");
      return false;
    }
    for (int f = 0; f < frames; ++f) {
      std::size_t i = 0;
      while (i + 1 < k.size() && k[i + 1].frame <= f)
        ++i;
      CameraKeyframe a = k[i], b = i + 1 < k.size() ? k[i + 1] : k[i];
      const double span = std::max(1, b.frame - a.frame);
      const double s = (f <= a.frame || b.frame == a.frame)
                           ? 0.0
                           : std::min(1.0, (f - a.frame) / span);
      CameraPose& pose = (*poses)[f];
      for (int j = 0; j < 3; ++j)
        pose.centre[j] =
            d * (a.position[j] + s * (b.position[j] - a.position[j]));
      pose.rotation = Rotation(a.yaw + s * (b.yaw - a.yaw),
                               a.pitch + s * (b.pitch - a.pitch),
                               a.roll + s * (b.roll - a.roll));
      pose.zoom = a.zoom + s * (b.zoom - a.zoom);
    }
    return true;
  }
  struct Default {
    const char* type;
    double amount;
  };
  static constexpr Default kDefaults[] = {
      {"static", 0},       {"pan_left", 15},     {"pan_right", 15},
      {"tilt_up", 10},     {"tilt_down", 10},    {"zoom_in", 0.4},
      {"zoom_out", 0.4},   {"dolly_in", 0.3},    {"dolly_out", 0.3},
      {"truck_left", 0.2}, {"truck_right", 0.2}, {"crane_up", 0.2},
      {"crane_down", 0.2}, {"orbit_left", 20},   {"orbit_right", 20}};
  const Default* found = nullptr;
  for (const auto& entry : kDefaults)
    if (t == entry.type)
      found = &entry;
  if (found == nullptr) {
    SetError(error, "unknown camera type " + t);
    return false;
  }
  if (spec.ease != "linear" && spec.ease != "ease_in_out") {
    SetError(error, "camera ease must be linear or ease_in_out");
    return false;
  }
  const double a =
      (spec.amount >= 0 ? spec.amount : found->amount) * spec.speed;
  for (int f = 0; f < frames; ++f) {
    double p = frames > 1 ? static_cast<double>(f) / (frames - 1) : 0.0;
    if (spec.ease == "ease_in_out")
      p = p * p * (3 - 2 * p);
    CameraPose& pose = (*poses)[f];
    const double v = a * p;
    if (t == "pan_left" || t == "pan_right")
      pose.rotation = Rotation(t == "pan_right" ? v : -v, 0, 0);
    else if (t == "tilt_up" || t == "tilt_down")
      pose.rotation = Rotation(0, t == "tilt_up" ? v : -v, 0);
    else if (t == "zoom_in")
      pose.zoom = 1 + v;
    else if (t == "zoom_out")
      pose.zoom = 1 / (1 + v);
    else if (t == "dolly_in" || t == "dolly_out")
      pose.centre[2] = (t == "dolly_in" ? v : -v) * d;
    else if (t == "truck_left" || t == "truck_right")
      pose.centre[0] = (t == "truck_right" ? v : -v) * d;
    else if (t == "crane_up" || t == "crane_down")
      pose.centre[1] = (t == "crane_up" ? -v : v) * d;
    else if (t == "orbit_left" || t == "orbit_right") {
      // Around a pivot one scene depth ahead; orbit_right moves the camera to
      // the right (turning left to keep the pivot centred).
      const double theta = (t == "orbit_right" ? -v : v) * kPi / 180;
      pose.centre = {-d * std::sin(theta), 0, d - d * std::cos(theta)};
      pose.rotation = Rotation(theta * 180 / kPi, 0, 0);
    }
  }
  return true;
}

}  // namespace gufo::models::wan22
