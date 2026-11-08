#include "src/models/qwen_image_21/control.hpp"

#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

#include "src/models/qwen_image_21/weights.hpp"

namespace gufo::models::qwen_image_21 {
namespace {

constexpr int kBaseBlocks = 32;
constexpr int kInner = 4096;
constexpr int kHeadDim = 128;
constexpr int kMlp = 12288;

using Tensors = std::map<std::string, ControlTensor, std::less<>>;

void Expect(const Tensors& tensors, const std::string& name,
            std::initializer_list<int> shape, ControlnetInfo* info,
            std::set<std::string>* used) {
  const auto it = tensors.find(name);
  if (it == tensors.end())
    throw std::runtime_error("ControlNet tensor missing: " + name);
  if (!it->second.bf16 || it->second.shape != std::vector<int>(shape))
    throw std::runtime_error("ControlNet tensor " + name +
                             " has an unexpected shape or dtype");
  info->bytes += it->second.bytes;
  used->insert(name);
}

void CheckCanvas(const Image& image) {
  if (image.width <= 0 || image.height <= 0 ||
      image.rgba.size() !=
          static_cast<std::size_t>(image.width) * image.height * 4)
    throw std::invalid_argument("invalid RGBA image dimensions");
}

}  // namespace

ControlnetInfo InspectControlnet(const Tensors& tensors) {
  ControlnetInfo info;
  std::set<std::string> used;
  Expect(tensors, "control_img_in.weight", {kInner, kControlInChannels}, &info,
         &used);
  Expect(tensors, "control_img_in.bias", {kInner}, &info, &used);
  while (tensors.contains("control_blocks." + std::to_string(info.blocks) +
                          ".after_proj.weight"))
    ++info.blocks;
  if (info.blocks < 1 || kBaseBlocks % info.blocks != 0)
    throw std::runtime_error("ControlNet needs a block count dividing 32, not " +
                             std::to_string(info.blocks));
  for (int i = 0; i < info.blocks; ++i) {
    const auto b = "control_blocks." + std::to_string(i) + ".";
    for (const char* name :
         {"attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
          "after_proj"})
      Expect(tensors, b + name + ".weight", {kInner, kInner}, &info, &used);
    Expect(tensors, b + "attn.norm_q.weight", {kHeadDim}, &info, &used);
    Expect(tensors, b + "attn.norm_k.weight", {kHeadDim}, &info, &used);
    Expect(tensors, b + "img_mlp.gate_layer.weight", {kMlp, kInner}, &info,
           &used);
    Expect(tensors, b + "img_mlp.proj.weight", {kMlp, kInner}, &info, &used);
    Expect(tensors, b + "img_mlp.out.weight", {kInner, kMlp}, &info, &used);
    Expect(tensors, b + "after_proj.bias", {kInner}, &info, &used);
    if (i == 0) {
      Expect(tensors, b + "before_proj.weight", {kInner, kInner}, &info,
             &used);
      Expect(tensors, b + "before_proj.bias", {kInner}, &info, &used);
    }
  }
  // Anything else (a bias on a block projection, before_proj on a later
  // block, fused gate_up, a block without after_proj) needs other code.
  for (const auto& [name, tensor] : tensors) {
    (void)tensor;
    if (!used.contains(name))
      throw std::runtime_error("ControlNet file has an unsupported tensor: " +
                               name);
  }
  info.in_channels = kControlInChannels;
  info.inner = kInner;
  info.stride = kBaseBlocks / info.blocks;
  return info;
}

ControlnetInfo InspectControlnet(const Weights& weights,
                                 std::string_view prefix) {
  Tensors tensors;
  for (const auto& [name, weight] : weights.Group(prefix))
    tensors.emplace(std::string(name.substr(prefix.size())),
                    ControlTensor{weight->shape, weight->bf16, weight->bytes});
  return InspectControlnet(tensors);
}

Image ControlCanvas(const Image& image, int width, int height) {
  CheckCanvas(image);
  Image opaque = image;
  for (std::size_t i = 3; i < opaque.rgba.size(); i += 4)
    opaque.rgba[i] = 255;
  return ResizeImage(opaque, width, height);
}

std::vector<float> ControlKeepMask(const Image& mask, MaskChannel channel,
                                   int width, int height) {
  if (width < 16 || height < 16 || width % 16 || height % 16)
    throw std::invalid_argument("control canvas must be multiples of 16");
  const auto edit = CompositeWeights(mask, channel, width, height, 0);
  const int rows = height / 16, cols = width / 16;
  std::vector<float> keep(static_cast<std::size_t>(rows) * cols);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      keep[static_cast<std::size_t>(r) * cols + c] =
          edit[static_cast<std::size_t>(r) * 16 * width + c * 16] >= 0.5F
              ? 0.0F
              : 1.0F;
  return keep;
}

Image ControlInpaintSource(const Image& image, const Image& mask,
                           MaskChannel channel) {
  CheckCanvas(image);
  const auto edit =
      CompositeWeights(mask, channel, image.width, image.height, 0);
  Image source = image;
  for (std::size_t i = 0; i < edit.size(); ++i) {
    auto* p = source.rgba.data() + 4 * i;
    if (edit[i] >= 0.5F)
      p[0] = p[1] = p[2] = 128;
    p[3] = 255;
  }
  return source;
}

std::vector<float> PackControlContext(std::span<const float> control,
                                      std::span<const float> keep,
                                      std::span<const float> inpaint,
                                      int rows) {
  const auto n = static_cast<std::size_t>(rows);
  if (rows <= 0 || control.size() != n * kControlLatentChannels ||
      (!keep.empty() && keep.size() != n) ||
      (!inpaint.empty() && inpaint.size() != n * kControlLatentChannels))
    throw std::invalid_argument("control context shape mismatch");
  std::vector<float> packed(n * kControlInChannels, 0.0F);
  for (std::size_t r = 0; r < n; ++r) {
    float* out = packed.data() + r * kControlInChannels;
    for (int c = 0; c < kControlLatentChannels; ++c)
      out[c] = control[r * kControlLatentChannels + c];
    if (!keep.empty())
      out[kControlLatentChannels] = keep[r];
    if (!inpaint.empty())
      for (int c = 0; c < kControlLatentChannels; ++c)
        out[kControlLatentChannels + 1 + c] =
            inpaint[r * kControlLatentChannels + c];
  }
  return packed;
}

bool ControlActiveAt(const ControlInput& control, int step, int steps) {
  if (steps <= 0 || step < 0 || step >= steps || control.strength == 0)
    return false;
  // Compare step / steps against the bounds without rounding the fraction.
  const double at = static_cast<double>(step);
  return at >= control.start * steps - 1e-9 && at < control.end * steps - 1e-9;
}

bool ControlHasImage(const ControlInput& control) {
  return !(control.image.width == 0 && control.image.height == 0 &&
           control.image.rgba.empty());
}

void ValidateControl(const ControlInput& control, const Request& request) {
  if (ControlHasImage(control))
    CheckCanvas(control.image);
  else if (!control.inpaint)
    throw std::invalid_argument(
        "a control without control image needs the inpaint recipe "
        "(init_image and mask)");
  if (!std::isfinite(control.strength) || control.strength < 0 ||
      control.strength > 2)
    throw std::invalid_argument("control strength must be between 0 and 2");
  if (!(control.start >= 0 && control.start <= 1 && control.end >= 0 &&
        control.end <= 1 && control.start < control.end))
    throw std::invalid_argument(
        "control start and end must satisfy 0 <= start < end <= 1");
  if (!request.images.empty())
    throw std::invalid_argument(
        "control images do not combine with reference images yet");
  // The control branch runs on the full canvas of a text-only pass; regional
  // passes (crops, own references) do not carry it.
  if (!request.regions.empty())
    throw std::invalid_argument(
        "control images do not combine with regional prompts yet");
  if (control.inpaint && !(request.init_image && request.mask))
    throw std::invalid_argument("control inpaint needs init_image and mask");
}

}  // namespace gufo::models::qwen_image_21
