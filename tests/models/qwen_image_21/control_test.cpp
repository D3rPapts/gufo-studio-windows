// CPU contract of the Qwen-Image-2.1 Fun ControlNet-Union branch: layout
// check, host-side conditioning (canvas, keep mask, inpaint source, 129
// channel packing), step range and request validation. With a file argument
// (or GUFO_QWEN_IMAGE_CONTROLNET) the real checkpoint's header is checked;
// the payload is only mapped, never read.

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen_image_21/control.hpp"
#include "src/models/qwen_image_21/weights.hpp"

namespace {

using namespace gufo::models::qwen_image_21;

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
  }
}

template <class E = std::exception>
void Throws(const std::function<void()>& f, const std::string& what) {
  try {
    f();
  } catch (const E&) {
    return;
  }
  std::cerr << "FAIL (no exception): " << what << '\n';
  ++failures;
}

using Tensors = std::map<std::string, ControlTensor, std::less<>>;

void Put(Tensors& t, const std::string& name, std::vector<int> shape) {
  std::size_t bytes = 2;
  for (int d : shape)
    bytes *= static_cast<std::size_t>(d);
  t[name] = ControlTensor{std::move(shape), true, bytes};
}

// The published layout: control_img_in + 16 blocks, before_proj in block 0.
Tensors Layout(int blocks) {
  Tensors t;
  Put(t, "control_img_in.weight", {4096, 129});
  Put(t, "control_img_in.bias", {4096});
  for (int i = 0; i < blocks; ++i) {
    const auto b = "control_blocks." + std::to_string(i) + ".";
    for (const char* n : {"attn.to_q", "attn.to_k", "attn.to_v",
                          "attn.to_out.0", "after_proj"})
      Put(t, b + n + ".weight", {4096, 4096});
    Put(t, b + "attn.norm_q.weight", {128});
    Put(t, b + "attn.norm_k.weight", {128});
    Put(t, b + "img_mlp.gate_layer.weight", {12288, 4096});
    Put(t, b + "img_mlp.proj.weight", {12288, 4096});
    Put(t, b + "img_mlp.out.weight", {4096, 12288});
    Put(t, b + "after_proj.bias", {4096});
    if (i == 0) {
      Put(t, b + "before_proj.weight", {4096, 4096});
      Put(t, b + "before_proj.bias", {4096});
    }
  }
  return t;
}

void TestLayout() {
  const auto info = InspectControlnet(Layout(16));
  Check(info.blocks == 16 && info.stride == 2 && info.in_channels == 129 &&
            info.inner == 4096,
        "16-block layout");
  // Header sizes of the published file: 7,550,979,904 bytes minus the
  // 8-byte length and the 20,280-byte JSON header.
  Check(info.bytes == 7550979904ULL - 8 - 20280, "published payload bytes");
  Check(InspectControlnet(Layout(8)).stride == 4, "8 blocks -> every 4th");
  Throws([] { InspectControlnet(Layout(5)); }, "5 blocks do not divide 32");
  Throws([] { InspectControlnet(Layout(0)); }, "no blocks");
  auto t = Layout(16);
  t.erase("control_blocks.7.attn.norm_k.weight");
  Throws([&] { InspectControlnet(t); }, "missing tensor");
  t = Layout(16);
  Put(t, "control_img_in.weight", {4096, 128});
  Throws([&] { InspectControlnet(t); }, "control_in_dim 128");
  t = Layout(16);
  t["control_blocks.3.attn.to_q.weight"].bf16 = false;
  Throws([&] { InspectControlnet(t); }, "F32 block weight");
  t = Layout(16);
  Put(t, "control_blocks.1.before_proj.weight", {4096, 4096});
  Throws([&] { InspectControlnet(t); }, "before_proj outside block 0");
  t = Layout(16);
  Put(t, "control_blocks.2.attn.to_q.bias", {4096});
  Throws([&] { InspectControlnet(t); }, "biased block projection");
  t = Layout(16);
  Put(t, "control_blocks.16.attn.to_q.weight", {4096, 4096});
  Throws([&] { InspectControlnet(t); }, "block without after_proj");
  t = Layout(16);
  Put(t, "control_blocks.0.img_mlp.gate_up.weight", {24576, 4096});
  Throws([&] { InspectControlnet(t); }, "fused gate_up (other converter)");
}

Image Solid(int w, int h, std::uint8_t r, std::uint8_t g, std::uint8_t b,
            std::uint8_t a) {
  Image image{w, h, {}};
  for (int i = 0; i < w * h; ++i)
    image.rgba.insert(image.rgba.end(), {r, g, b, a});
  return image;
}

void TestCanvas() {
  // Alpha is dropped before resizing: a transparent white pixel stays white
  // (premultiplied resizing would otherwise blacken it).
  auto image = Solid(64, 32, 255, 255, 255, 0);
  const auto canvas = ControlCanvas(image, 128, 64);
  Check(canvas.width == 128 && canvas.height == 64, "canvas size");
  bool ok = true;
  for (std::size_t i = 0; i < canvas.rgba.size(); i += 4)
    ok &= canvas.rgba[i] == 255 && canvas.rgba[i + 3] == 255;
  Check(ok, "canvas opaque white");
  Throws<std::invalid_argument>(
      [] { ControlCanvas(Image{2, 2, {}}, 32, 32); }, "broken image");
}

void TestKeepMask() {
  // 64 x 32 canvas = 4 x 2 cells; white (edit) on the right half.
  Image mask = Solid(64, 32, 0, 0, 0, 255);
  for (int y = 0; y < 32; ++y)
    for (int x = 32; x < 64; ++x)
      for (int c = 0; c < 3; ++c)
        mask.rgba[(y * 64 + x) * 4 + c] = 255;
  const auto keep = ControlKeepMask(mask, MaskChannel::kAuto, 64, 32);
  Check(keep == std::vector<float>{1, 1, 0, 0, 1, 1, 0, 0},
        "keep 1 left, regenerate right");
  // The cell takes its top-left pixel (nearest downsampling): a mark
  // elsewhere in the cell does not regenerate it.
  Image dot = Solid(32, 32, 0, 0, 0, 255);
  for (int c = 0; c < 3; ++c)
    dot.rgba[(5 * 32 + 5) * 4 + c] = 255;
  Check(ControlKeepMask(dot, MaskChannel::kAuto, 32, 32) ==
            std::vector<float>(4, 1.0F),
        "off-corner mark keeps");
  for (int c = 0; c < 3; ++c)
    dot.rgba[c] = 255;  // pixel (0, 0)
  Check(ControlKeepMask(dot, MaskChannel::kAuto, 32, 32) ==
            std::vector<float>({0, 1, 1, 1}),
        "corner mark regenerates");
  // Threshold 0.5: 127 keeps, 128 regenerates (red channel mask).
  Check(ControlKeepMask(Solid(16, 16, 127, 0, 0, 255), MaskChannel::kRed, 16,
                        16)[0] == 1.0F,
        "127 keeps");
  Check(ControlKeepMask(Solid(16, 16, 128, 0, 0, 255), MaskChannel::kRed, 16,
                        16)[0] == 0.0F,
        "128 regenerates");
  // OpenAI-style alpha mask: transparent = edit.
  Check(ControlKeepMask(Solid(16, 16, 0, 0, 0, 0), MaskChannel::kAuto, 16,
                        16)[0] == 0.0F,
        "transparent regenerates");
  // A smaller mask scales to the canvas.
  Image small = Solid(2, 1, 0, 0, 0, 255);
  for (int c = 0; c < 3; ++c)
    small.rgba[4 + c] = 255;
  Check(ControlKeepMask(small, MaskChannel::kAuto, 64, 16) ==
            std::vector<float>({1, 1, 0, 0}),
        "scaled mask");
  Throws<std::invalid_argument>(
      [&] { ControlKeepMask(small, MaskChannel::kAuto, 40, 16); },
      "canvas not multiple of 16");
}

void TestInpaintSource() {
  const auto image = Solid(4, 2, 10, 20, 30, 40);
  Image mask = Solid(4, 2, 0, 0, 0, 255);
  for (int c = 0; c < 3; ++c)
    mask.rgba[(1 * 4 + 3) * 4 + c] = 255;  // pixel (3, 1)
  const auto source = ControlInpaintSource(image, mask, MaskChannel::kAuto);
  bool ok = true;
  for (int i = 0; i < 8; ++i) {
    const auto* p = source.rgba.data() + 4 * i;
    if (i == 7)
      ok &= p[0] == 128 && p[1] == 128 && p[2] == 128;
    else
      ok &= p[0] == 10 && p[1] == 20 && p[2] == 30;
    ok &= p[3] == 255;
  }
  Check(ok, "inpaint source gray under the mask, opaque");
}

void TestPack() {
  const int rows = 2;
  std::vector<float> control(rows * 64), inpaint(rows * 64);
  for (int i = 0; i < rows * 64; ++i) {
    control[i] = static_cast<float>(i);
    inpaint[i] = static_cast<float>(-i);
  }
  const auto pure = PackControlContext(control, {}, {}, rows);
  Check(pure.size() == 2 * 129, "pure size");
  Check(pure[0] == 0 && pure[63] == 63 && pure[64] == 0 && pure[128] == 0 &&
            pure[129] == 64 && pure[129 + 63] == 127 && pure[129 + 64] == 0,
        "pure layout: latent, zero keep, zero inpaint");
  const std::vector<float> keep{1, 0};
  const auto full = PackControlContext(control, keep, inpaint, rows);
  Check(full[64] == 1 && full[129 + 64] == 0 && full[65] == 0 &&
            full[65 + 63] == -63 && full[129 + 65] == -64 &&
            full[2 * 129 - 1] == -127,
        "inpaint layout: latent | keep | masked latent");
  Throws<std::invalid_argument>(
      [&] { PackControlContext(control, keep, inpaint, 3); }, "row mismatch");
  Throws<std::invalid_argument>(
      [&] {
        PackControlContext(control, std::vector<float>{1}, inpaint, rows);
      },
      "keep mismatch");
}

void TestRange() {
  ControlInput control;
  control.image = Solid(32, 32, 0, 0, 0, 255);
  int active = 0;
  for (int s = 0; s < 40; ++s)
    active += ControlActiveAt(control, s, 40);
  Check(active == 40, "full range");
  control.end = 0.5;
  bool ok = true;
  for (int s = 0; s < 40; ++s)
    ok &= ControlActiveAt(control, s, 40) == (s < 20);
  Check(ok, "first half");
  control.start = 0.25;
  control.end = 1;
  ok = true;
  for (int s = 0; s < 8; ++s)
    ok &= ControlActiveAt(control, s, 8) == (s >= 2);
  Check(ok, "Turbo 8 steps from 25 %");
  control.start = 0;
  control.strength = 0;
  Check(!ControlActiveAt(control, 0, 8), "strength 0 is off");
  control.strength = 1;
  Check(!ControlActiveAt(control, 8, 8) && !ControlActiveAt(control, -1, 8),
        "outside the schedule");
}

void TestValidate() {
  Request request;
  request.prompt = "a dancer";
  ControlInput control;
  control.image = Solid(32, 32, 0, 0, 0, 255);
  request.control = control;
  ValidateRequest(request);
  auto bad = request;
  bad.control->strength = 2.5F;
  Throws<std::invalid_argument>([&] { ValidateRequest(bad); }, "strength 2.5");
  bad = request;
  bad.control->strength = NAN;
  Throws<std::invalid_argument>([&] { ValidateRequest(bad); }, "strength NaN");
  bad = request;
  bad.control->start = 0.6;
  bad.control->end = 0.5;
  Throws<std::invalid_argument>([&] { ValidateRequest(bad); }, "start > end");
  // Reference images combine with the control (zero rows in the stream).
  auto with_reference = request;
  with_reference.images.push_back(Solid(32, 32, 1, 2, 3, 255));
  ValidateRequest(with_reference);
  bad = request;
  bad.control->inpaint = true;
  Throws<std::invalid_argument>([&] { ValidateRequest(bad); },
                                "inpaint without mask");
  bad.init_image = Solid(32, 32, 1, 2, 3, 255);
  bad.mask = Solid(32, 32, 255, 255, 255, 255);
  ValidateRequest(bad);
  bad = request;
  bad.control->image = Image{3, 3, {}};
  Throws<std::invalid_argument>([&] { ValidateRequest(bad); }, "broken image");
  // Inpaint-only: no control image (0x0), start image and mask given.
  auto only = request;
  only.control->image = Image{};
  Throws<std::invalid_argument>([&] { ValidateRequest(only); },
                                "no control image without inpaint");
  only.control->inpaint = true;
  only.init_image = Solid(32, 32, 1, 2, 3, 255);
  only.mask = Solid(32, 32, 255, 255, 255, 255);
  ValidateRequest(only);
  Check(!ControlHasImage(*only.control) && ControlHasImage(*request.control),
        "ControlHasImage");
}

void TestRealHeader(const std::filesystem::path& file) {
  // Maps the file and parses its JSON header only (no payload reads).
  auto weights = Weights::FromFiles(
      std::vector<SafetensorsSource>{{file, std::string(kControlnetPrefix)}});
  const auto info = InspectControlnet(weights);
  Check(info.blocks == 16 && info.stride == 2 && info.in_channels == 129 &&
            info.inner == 4096,
        "real header layout");
  Check(info.bytes + 8 + 20280 == std::filesystem::file_size(file),
        "real header covers the whole payload");
  Check(weights.Get("controlnet.control_img_in.weight").shape ==
            std::vector<int>({4096, 129}),
        "real control_img_in");
  // AddFile on top of other weights keeps them apart by prefix.
  auto twice = Weights::FromFiles(
      std::vector<SafetensorsSource>{{file, "other."}});
  twice.AddFile(file, std::string(kControlnetPrefix));
  Check(InspectControlnet(twice).blocks == 16 &&
            InspectControlnet(twice, "other.").blocks == 16,
        "AddFile with a second prefix");
  Throws([&] { twice.AddFile(file, "other."); }, "duplicate prefix");
  std::cout << "real header: " << info.blocks << " blocks, "
            << (info.bytes >> 20) << " MiB\n";
}

}  // namespace

int main(int argc, char** argv) {
  TestLayout();
  TestCanvas();
  TestKeepMask();
  TestInpaintSource();
  TestPack();
  TestRange();
  TestValidate();
  std::filesystem::path file;
  if (argc > 1)
    file = argv[1];
  else if (const char* env = std::getenv("GUFO_QWEN_IMAGE_CONTROLNET"))
    file = env;
  if (!file.empty())
    TestRealHeader(file);
  else
    std::cout << "real header: skipped (no file)\n";
  if (failures) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "qwen_image_21_control_test: all passed\n";
  return 0;
}
