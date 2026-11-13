#ifndef GUFO_MODELS_WAN22_CAMERA_PATH_HPP_
#define GUFO_MODELS_WAN22_CAMERA_PATH_HPP_

#include <array>
#include <string>
#include <vector>

namespace gufo::models::wan22 {

/// Camera of one frame in the start camera's frame (OpenCV axes: x right,
/// y down, z forward): camera-to-world rotation (row-major), centre and a
/// digital zoom (focal multiplier).
struct CameraPose {
  std::array<double, 9> rotation{1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::array<double, 3> centre{0, 0, 0};
  double zoom{1.0};
};

/// One explicit keyframe; positions in units of the scene depth (median
/// depth of the start image), angles in degrees (yaw right, pitch up, roll
/// clockwise), zoom as focal multiplier. Frames between keyframes are
/// interpolated linearly.
struct CameraKeyframe {
  int frame{0};
  std::array<double, 3> position{0, 0, 0};
  double yaw{0}, pitch{0}, roll{0};
  double zoom{1.0};
};

/// A camera move over the clip. Types and their `amount` (default):
///   static
///   pan_left|pan_right     yaw in degrees (15)
///   tilt_up|tilt_down      pitch in degrees (10)
///   zoom_in|zoom_out       focal change, 0.4 = 1.4x (0.4)
///   dolly_in|dolly_out     forward/back in scene depths (0.3)
///   truck_left|truck_right sideways in scene depths (0.2)
///   crane_up|crane_down    up/down in scene depths (0.2)
///   orbit_left|orbit_right degrees around a pivot one scene depth ahead (20)
///   poses                  the explicit `keyframes`
/// The magnitude reached at the last frame is amount * speed; `ease` is
/// "linear" or "ease_in_out" (smoothstep).
struct CameraSpec {
  std::string type{"static"};
  double amount{-1};  // < 0: the type's default
  double speed{1.0};
  std::string ease{"linear"};
  std::vector<CameraKeyframe> keyframes;
};

/// Poses for `frames` frames (frame 0 is the start camera).
bool BuildCameraPath(const CameraSpec& spec, int frames, double scene_depth,
                     std::vector<CameraPose>* poses, std::string* error);

/// Parses "frame,x,y,z,yaw,pitch,roll,zoom;..." into keyframes.
bool ParseKeyframes(const std::string& text,
                    std::vector<CameraKeyframe>* keyframes, std::string* error);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_CAMERA_PATH_HPP_
