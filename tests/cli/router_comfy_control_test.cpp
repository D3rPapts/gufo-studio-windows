// ComfyUI mapping of the Qwen-Image 2.1 Fun ControlNet-Union:
// ModelPatchLoader -> QwenImageDiffsynthControlnet / ZImageFunControlnet ->
// KSampler becomes ImageTask::control and gufo.control_* fields. Graphs use
// the node signatures of ComfyUI's comfy_extras/nodes_model_patch.py
// (PR #16519).

#include <iostream>
#include <string>
#include <variant>

#include "src/cli/serve/router/comfy_graph.hpp"
#include "src/core/json.hpp"

namespace comfy = gufo::router::comfy;
namespace json = gufo::json;

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
  }
}

std::string Replace(std::string text, const std::string& from,
                    const std::string& to) {
  const auto at = text.find(from);
  if (at == std::string::npos) {
    std::cerr << "fixture text not found: " << from << '\n';
    ++failures;
    return text;
  }
  return text.replace(at, from.size(), to);
}

// Text to image with a pose control: the KSampler model comes through the
// ControlNet apply node.
constexpr const char* kControl = R"({
 "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
 "2": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
 "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
 "p1": {"class_type": "ModelPatchLoader", "inputs": {"name": "qwen_image_2.1_fun_controlnet_union_bf16.safetensors"}},
 "p2": {"class_type": "LoadImage", "inputs": {"image": "pose_tanz.png"}},
 "p3": {"class_type": "QwenImageDiffsynthControlnet", "inputs": {"model": ["4", 0], "model_patch": ["p1", 0], "vae": ["2", 0], "image": ["p2", 0], "strength": 0.8, "start_percent": 0.0, "end_percent": 1.0}},
 "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["1", 0], "prompt": "a dancer on a stage", "negative_prompt": "", "vae": ["2", 0], "resolution": 1024}},
 "6": {"class_type": "KSampler", "inputs": {"model": ["p3", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["5", 2], "seed": 43, "steps": 40, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
 "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "keyframes/pose"}}
})";

bool Plan(const std::string& text, comfy::Plan* plan,
          comfy::PlanError* error) {
  return comfy::BuildPlan(json::parse(text), {}, plan, error);
}

comfy::ImageTask ImageOf(const std::string& text, comfy::Plan* out = nullptr) {
  comfy::Plan plan;
  comfy::PlanError error;
  if (!Plan(text, &plan, &error)) {
    std::cerr << "unexpected plan error: " << error.message << '\n';
    ++failures;
    return {};
  }
  if (out)
    *out = plan;
  return std::get<comfy::ImageTask>(plan.tasks.at(0).spec);
}

std::string ErrorOf(const std::string& text) {
  comfy::Plan plan;
  comfy::PlanError error;
  if (Plan(text, &plan, &error)) {
    std::cerr << "expected a plan error\n";
    ++failures;
    return {};
  }
  return error.message;
}

bool Has(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}

void TestPose() {
  comfy::Plan plan;
  const auto image = ImageOf(kControl, &plan);
  Check(image.control.has_value(), "control mapped");
  if (!image.control)
    return;
  Check((image.control->image == comfy::FileRef{"pose_tanz.png", "input"}),
        "control image file");
  Check(image.control->strength == 0.8 && image.control->start == 0 &&
            image.control->end == 1 && !image.control->inpaint,
        "control values");
  Check(image.references.empty() && image.prompt == "a dancer on a stage" &&
            image.steps == 40,
        "rest of the task unchanged");
  const auto extension = comfy::ImageExtension(image);
  Check(extension.find("control_strength") &&
            extension.find("control_strength")->as_double() == 0.8 &&
            !extension.find("control_start") && !extension.find("control_end") &&
            !extension.find("control_inpaint") &&
            !extension.find("control_image"),
        "extension fields");
  for (const auto& warning : plan.warnings)
    Check(!Has(warning, "ControlNet"), "no warning: " + warning);
}

void TestOptions() {
  // A partial range maps with a warning; LoadImageOutput works too.
  std::string text = Replace(kControl, R"("start_percent": 0.0, "end_percent": 1.0)",
                             R"("start_percent": 0.1, "end_percent": 0.6)");
  text = Replace(text,
                 R"("p2": {"class_type": "LoadImage", "inputs": {"image": "pose_tanz.png"}})",
                 R"("p2": {"class_type": "LoadImageOutput", "inputs": {"image": "posen/kf1.png [output]"}})");
  comfy::Plan plan;
  auto image = ImageOf(text, &plan);
  Check(image.control && image.control->start == 0.1 &&
            image.control->end == 0.6 &&
            (image.control->image == comfy::FileRef{"posen/kf1.png", "output"}),
        "range and output image");
  bool warned = false;
  for (const auto& warning : plan.warnings)
    warned |= Has(warning, "fractions of the steps");
  Check(warned, "range warning");
  const auto extension = comfy::ImageExtension(image);
  Check(extension.find("control_start") &&
            extension.find("control_start")->as_double() == 0.1 &&
            extension.find("control_end")->as_double() == 0.6,
        "range fields");
  // Strength 0: ComfyUI returns the model unchanged.
  image = ImageOf(Replace(kControl, R"("strength": 0.8)", R"("strength": 0.0)"),
                  &plan);
  Check(!image.control, "strength 0 drops the control");
  // A LoRA between the ControlNet and the loader still maps both.
  text = Replace(kControl, R"("model": ["4", 0], "model_patch")",
                 R"("model": ["L1", 0], "model_patch")");
  text = Replace(text, R"( "5": {)",
                 R"( "L1": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["4", 0], "lora_name": "anime.safetensors", "strength_model": 0.7}},
 "5": {)");
  image = ImageOf(text);
  Check(image.control && image.loras.size() == 1 &&
            image.loras[0].first == "anime",
        "LoRA and ControlNet");
  // Turbo checkpoint: the control rides along.
  text = Replace(kControl, "qwen_image_2.1_int8_convrot.safetensors",
                 "qwen_image_2.1_turbo_bf16.safetensors");
  text = Replace(text, R"("steps": 40)", R"("steps": 8)");
  image = ImageOf(text);
  Check(image.control && image.model == "qwen-image-2.1-turbo", "turbo");
  // An unknown patch file only warns.
  ImageOf(Replace(kControl, "qwen_image_2.1_fun_controlnet_union_bf16",
                  "something_else"),
          &plan);
  warned = false;
  for (const auto& warning : plan.warnings)
    warned |= Has(warning, "registered");
  Check(warned, "unknown patch file warns");
}

void TestErrors() {
  // Preprocessor nodes are not part of gufo.
  std::string text = Replace(
      kControl, R"("image": ["p2", 0], "strength")",
      R"("image": ["d1", 0], "strength")");
  text = Replace(text, R"( "5": {)",
                 R"( "d1": {"class_type": "DWPreprocessor", "inputs": {"image": ["p2", 0]}},
 "5": {)");
  Check(!ErrorOf(text).empty(), "preprocessor node rejected");
  // model_patch must come from ModelPatchLoader.
  text = Replace(kControl, R"("model_patch": ["p1", 0])",
                 R"("model_patch": ["p2", 0])");
  Check(Has(ErrorOf(text), "ModelPatchLoader"), "patch loader required");
  // Out-of-range values.
  Check(Has(ErrorOf(Replace(kControl, R"("strength": 0.8)",
                            R"("strength": 3.0)")),
            "strength"),
        "strength 3");
  Check(Has(ErrorOf(Replace(kControl, R"("end_percent": 1.0)",
                            R"("end_percent": 0.0)")),
            "start_percent"),
        "empty range");
  // Two ControlNets in a row.
  text = Replace(kControl, R"("model": ["p3", 0], "positive")",
                 R"("model": ["p4", 0], "positive")");
  text = Replace(text, R"( "5": {)",
                 R"( "p4": {"class_type": "QwenImageDiffsynthControlnet", "inputs": {"model": ["p3", 0], "model_patch": ["p1", 0], "vae": ["2", 0], "image": ["p2", 0], "strength": 1.0}},
 "5": {)");
  Check(Has(ErrorOf(text), "two Qwen-Image ControlNet"), "two ControlNets");
  // ZImageFunControlnet without an image.
  text = Replace(kControl, "QwenImageDiffsynthControlnet", "ZImageFunControlnet");
  text = Replace(text, R"("image": ["p2", 0], )", "");
  Check(Has(ErrorOf(text), "control image"), "control image required");
  // Reference images combine with the ControlNet (1.4.7).
  text = Replace(kControl, R"("resolution": 1024})",
                 R"("resolution": 1024, "images.image_1": ["p2", 0]})");
  {
    const auto with_reference = ImageOf(text);
    Check(with_reference.control && with_reference.references.size() == 1,
          "references with the ControlNet");
  }
}

void TestInpaint() {
  // SetLatentNoiseMask over VAEEncode plus ZImageFunControlnet with the
  // same start image and mask: the checkpoint's inpaint recipe.
  std::string text = Replace(kControl, "QwenImageDiffsynthControlnet",
                             "ZImageFunControlnet");
  text = Replace(text, R"("strength": 0.8,)",
                 R"("strength": 0.8, "inpaint_image": ["c1", 0], "mask": ["c1", 1],)");
  text = Replace(text, R"("latent_image": ["5", 2])",
                 R"("latent_image": ["m2", 0])");
  text = Replace(text, R"( "6": {)",
                 R"( "c1": {"class_type": "LoadImage", "inputs": {"image": "leinwand.png"}},
 "c2": {"class_type": "VAEEncode", "inputs": {"pixels": ["c1", 0], "vae": ["2", 0]}},
 "m2": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["c2", 0], "mask": ["c1", 1]}},
 "6": {)");
  auto image = ImageOf(text);
  Check(image.control && image.control->inpaint &&
            (image.init_image == comfy::FileRef{"leinwand.png", "input"}) &&
            !image.mask.name.empty(),
        "inpaint recipe");
  Check(comfy::ImageExtension(image).find("control_inpaint") != nullptr,
        "control_inpaint field");
  // A different inpaint image than the edited one is refused.
  std::string other = Replace(text, R"("inpaint_image": ["c1", 0])",
                              R"("inpaint_image": ["p2", 0])");
  Check(Has(ErrorOf(other), "masked edit"), "other inpaint image");
  // inpaint_image without mask is refused.
  other = Replace(text, R"(, "mask": ["c1", 1],)", ",");
  Check(Has(ErrorOf(other), "both"), "inpaint without mask");
  // Without "image" (optional in ComfyUI): inpaint_image + mask alone, the
  // control channels stay zero; no control image file is sent.
  std::string only = Replace(text, R"("image": ["p2", 0], )", "");
  auto inpaint_only = ImageOf(only);
  Check(inpaint_only.control && inpaint_only.control->inpaint &&
            inpaint_only.control->image.name.empty(),
        "inpaint-only control");
  Check(comfy::ImageExtension(inpaint_only).find("control_inpaint") != nullptr,
        "inpaint-only control_inpaint field");
  // Without image, inpaint_image and mask it is still refused.
  other = Replace(only, R"(, "inpaint_image": ["c1", 0], "mask": ["c1", 1],)", ",");
  Check(Has(ErrorOf(other), "control image"), "no image and no inpaint");
}

}  // namespace

int main() {
  TestPose();
  TestOptions();
  TestErrors();
  TestInpaint();
  if (failures) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "router_comfy_control_test: all passed\n";
  return 0;
}
