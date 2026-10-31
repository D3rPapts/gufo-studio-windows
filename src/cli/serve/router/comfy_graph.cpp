#include "src/cli/serve/router/comfy_graph.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gufo::router::comfy {
namespace {

// Nodes that only load, wire or decode: the gufo endpoint does their work.
constexpr std::string_view kPlumbing[] = {
    "CLIPLoader",
    "DualCLIPLoader",
    "VAELoader",
    "UNETLoader",
    "CheckpointLoaderSimple",
    "LoraLoaderModelOnly",
    "LoraLoader",
    "ModelSamplingAuraFlow",
    "ModelSamplingSD3",
    "MiniMaxH3SigmaShift",
    "LTXVAudioVAELoader",
    "VAEDecode",
    "VAEDecodeTiled",
    "VAEEncode",
    "VAEEncodeForInpaint",
    "SetLatentNoiseMask",
    "LoadImageMask",
    "LTXVSeparateAVLatent",
    "LTXVAudioVAEDecode",
    "VAEDecodeAudio",
    "ConditioningZeroOut",
    "CLIPTextEncode",
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "ReferenceLatent",
    "LoadImage",
    "LoadImageOutput",
    "LoadAudio",
    "CreateVideo",
    "PrimitiveString",
    "PrimitiveStringMultiline",
    "PrimitiveInt",
    "PrimitiveFloat",
    "PrimitiveBoolean",
    "TextEncodeQwenImage21",
    "MiniMaxH3AddGuide",
    "RIFE VFI",
    "FB_Qwen3TTSVoiceClonePrompt",
    "FB_Qwen3TTSRoleBank",
    "TextEncodeAceStepAudio1.5",
    "EmptyAceStep1.5LatentAudio",
    "WanImageToVideo",
    "WanCameraImageToVideo",
    "WanCameraEmbedding",
    "WanUni3CControlnetApply",
    "ModelPatchLoader",
    "CLIPVisionLoader",
    "CLIPVisionEncode",
    "LoadVideo",
    "GetVideoComponents",
    "VHS_LoadVideo",
    "GufoCameraMove",
    "UpscaleModelLoader",
    "ImageUpscaleWithModel",
    "ImageScale",
    "ImageScaleBy",
    "WanVaceToVideo",
    "TrimVideoLatent",
    "ImageToMask",
    "InvertMask",
    // InfiniteTalk (talk family).
    "WanInfiniteTalkToVideo",
    "AudioEncoderLoader",
    "AudioEncoderEncode",
    "ImageBatch",
    "ImageFromBatch",
    // MMAudio (kijai/ComfyUI-MMAudio): the loaders only name files.
    "MMAudioModelLoader",
    "MMAudioFeatureUtilsLoader",
    "MMAudioVoCoderLoader",
};

constexpr std::string_view kGenerators[] = {
    "KSampler",
    "KSamplerAdvanced",
    "MiniMaxH3ImageToVideo",
    "Wan22ImageToVideoLatent",
    "FB_Qwen3TTSVoiceDesign",
    "FB_Qwen3TTSVoiceClone",
    "FB_Qwen3TTSCustomVoice",
    "FB_Qwen3TTSDialogueInference",
    "MMAudioSampler",
};

constexpr std::string_view kOutputs[] = {
    "SaveImage", "PreviewImage", "SaveAudio", "PreviewAudio", "SaveVideo",
    "VHS_VideoCombine"};

// Node families other gufo ports are bringing; named so the error says so.
// (ACE-Step 1.0; gufo serves ACE-Step 1.5.)
constexpr std::string_view kPending[] = {"TextEncodeAceStepAudio",
                                         "EmptyAceStepLatentAudio"};

template<std::size_t N>
bool Contains(const std::string_view (&set)[N], std::string_view value) {
  return std::find(std::begin(set), std::end(set), value) != std::end(set);
}

bool IsTts(std::string_view class_type) {
  return class_type == "FB_Qwen3TTSVoiceDesign" ||
         class_type == "FB_Qwen3TTSVoiceClone" ||
         class_type == "FB_Qwen3TTSCustomVoice" ||
         class_type == "FB_Qwen3TTSDialogueInference";
}

bool IsSampler(std::string_view class_type) {
  return class_type == "KSampler" || class_type == "KSamplerAdvanced";
}

bool IsFoley(std::string_view class_type) {
  return class_type == "MMAudioSampler";
}

std::string Lower(std::string_view text) {
  std::string out(text);
  for (char& c : out)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

/// Validation failure inside one node.
struct NodeFailure : std::runtime_error {
  NodeFailure(std::string node, std::string type, const std::string& message)
      : std::runtime_error(message),
        node_id(std::move(node)),
        error_type(std::move(type)) {}
  std::string node_id;
  std::string error_type;
};

struct Link {
  std::string node;
  int slot{0};
};

class Graph {
public:
  explicit Graph(const json::Value& prompt) : prompt_(prompt) {}

  const json::Value* Node(const std::string& id) const {
    return prompt_.find(id);
  }

  std::string Class(const std::string& id) const {
    const json::Value* node = Node(id);
    return node == nullptr ? "" : node->member_str("class_type");
  }

  const json::Value* Raw(const std::string& id, const std::string& key) const {
    const json::Value* node = Node(id);
    const json::Value* inputs =
        node == nullptr ? nullptr : node->find("inputs");
    return inputs == nullptr ? nullptr : inputs->find(key);
  }

  static std::optional<Link> AsLink(const json::Value& value) {
    if (!value.is_array() || value.size() != 2 || !value.items()[1].is_number())
      return std::nullopt;
    const json::Value& id = value.items()[0];
    Link link;
    if (id.is_string()) {
      link.node = id.str();
    } else if (id.is_number()) {
      link.node = std::to_string(static_cast<long long>(id.as_double()));
    } else {
      return std::nullopt;
    }
    link.slot = static_cast<int>(value.items()[1].as_double());
    return link;
  }

  std::optional<Link> LinkOf(const std::string& id,
                             const std::string& key) const {
    const json::Value* raw = Raw(id, key);
    if (raw == nullptr)
      return std::nullopt;
    auto link = AsLink(*raw);
    if (link && Node(link->node) == nullptr)
      throw NodeFailure(
          id, "invalid_link",
          "input '" + key + "' links to missing node " + link->node);
    return link;
  }

  Link RequireLink(const std::string& id, const std::string& key) const {
    auto link = LinkOf(id, key);
    if (!link)
      throw NodeFailure(id, "required_input_missing",
                        "required input '" + key + "' is not connected");
    return *link;
  }

  /// The literal behind `key`, following links to Primitive* nodes.
  const json::Value* Literal(const std::string& id,
                             const std::string& key) const {
    const json::Value* raw = Raw(id, key);
    if (raw == nullptr)
      return nullptr;
    if (auto link = AsLink(*raw)) {
      const std::string source = Class(link->node);
      if (!source.starts_with("Primitive"))
        throw NodeFailure(id, "unsupported_link",
                          "input '" + key +
                              "' must be a value or come from a "
                              "Primitive node, not " +
                              source);
      return Literal(link->node, "value");
    }
    return raw;
  }

  std::string Str(const std::string& id, const std::string& key,
                  std::optional<std::string> fallback = std::nullopt) const {
    const json::Value* value = Literal(id, key);
    if (value == nullptr) {
      if (fallback)
        return *fallback;
      throw NodeFailure(id, "required_input_missing",
                        "required input '" + key + "' is missing");
    }
    if (!value->is_string())
      throw NodeFailure(id, "invalid_input_type",
                        "input '" + key + "' must be a string");
    return value->str();
  }

  double Num(const std::string& id, const std::string& key,
             std::optional<double> fallback = std::nullopt) const {
    const json::Value* value = Literal(id, key);
    if (value == nullptr) {
      if (fallback)
        return *fallback;
      throw NodeFailure(id, "required_input_missing",
                        "required input '" + key + "' is missing");
    }
    if (!value->is_number() || !std::isfinite(value->as_double()))
      throw NodeFailure(id, "invalid_input_type",
                        "input '" + key + "' must be a number");
    return value->as_double();
  }

  bool Bool(const std::string& id, const std::string& key,
            bool fallback) const {
    const json::Value* value = Literal(id, key);
    if (value == nullptr)
      return fallback;
    if (!value->is_bool())
      throw NodeFailure(id, "invalid_input_type",
                        "input '" + key + "' must be true or false");
    return value->as_bool();
  }

  /// Every node upstream of `id` (links only).
  std::set<std::string> Ancestors(const std::string& id) const {
    std::set<std::string> seen;
    std::vector<std::string> stack{id};
    while (!stack.empty()) {
      const std::string current = stack.back();
      stack.pop_back();
      const json::Value* node = Node(current);
      const json::Value* inputs =
          node == nullptr ? nullptr : node->find("inputs");
      if (inputs == nullptr)
        continue;
      for (const auto& [key, value] : inputs->members()) {
        (void)key;
        if (auto link = AsLink(value); link && Node(link->node) != nullptr &&
                                       seen.insert(link->node).second)
          stack.push_back(link->node);
      }
    }
    return seen;
  }

private:
  const json::Value& prompt_;
};

std::uint64_t Seed(double value) {
  // ComfyUI seeds span 64 bits; JSON numbers and gufo keep 53.
  if (value < 0)
    return 0;
  constexpr double kLimit = 9007199254740991.0;
  return static_cast<std::uint64_t>(std::min(value, kLimit));
}

/// Image a node input receives, which must come from a Load* node.
FileRef ImageFile(const Graph& graph, const std::string& id,
                  const std::string& key, const Link& link) {
  const std::string source = graph.Class(link.node);
  if (source == "LoadImage")
    return ParseFileRef(graph.Str(link.node, "image"), "input");
  if (source == "LoadImageOutput")
    return ParseFileRef(graph.Str(link.node, "image"), "output");
  throw NodeFailure(
      id, "unsupported_link",
      "input '" + key + "' must come from LoadImage, not " + source);
}

/// Mask a node input receives: LoadImage / LoadImageOutput MASK (1 - alpha)
/// or LoadImageMask (its channel), sent as the file plus gufo.mask_channel.
void MaskFile(const Graph& graph, const std::string& id, const Link& link,
              ImageTask* task) {
  const std::string source = graph.Class(link.node);
  if ((source == "LoadImage" || source == "LoadImageOutput") &&
      link.slot == 1) {
    task->mask = ImageFile(graph, id, "mask", link);
    task->mask_channel = "alpha";
  } else if (source == "LoadImageMask") {
    task->mask = ParseFileRef(graph.Str(link.node, "image"), "input");
    task->mask_channel = graph.Str(link.node, "channel", "alpha");
    if (task->mask_channel != "alpha" && task->mask_channel != "red" &&
        task->mask_channel != "green" && task->mask_channel != "blue")
      throw NodeFailure(link.node, "value_not_in_list",
                        "channel must be alpha, red, green or blue");
  } else {
    throw NodeFailure(id, "unsupported_link",
                      "input 'mask' must come from the MASK output of "
                      "LoadImage or from LoadImageMask, not " +
                          source);
  }
}

/// The LoadVideo file behind a GetVideoComponents output (0 = images,
/// 1 = audio) that feeds input `key` of node `id`.
FileRef VideoComponentsFile(const Graph& graph, const std::string& id,
                            const std::string& key, const Link& link,
                            int slot) {
  if (link.slot != slot)
    throw NodeFailure(id, "unsupported_link",
                      "input '" + key + "' must come from output " +
                          std::to_string(slot) + " of GetVideoComponents");
  const Link file = graph.RequireLink(link.node, "video");
  if (graph.Class(file.node) != "LoadVideo")
    throw NodeFailure(link.node, "unsupported_link",
                      "the video of GetVideoComponents must come from "
                      "LoadVideo, not " +
                          graph.Class(file.node));
  return ParseFileRef(graph.Str(file.node, "file"), "input");
}

FileRef AudioFile(const Graph& graph, const std::string& id,
                  const std::string& key, const Link& link) {
  const std::string source = graph.Class(link.node);
  if (source == "LoadAudio")
    return ParseFileRef(graph.Str(link.node, "audio"), "input");
  throw NodeFailure(
      id, "unsupported_link",
      "input '" + key + "' must come from LoadAudio, not " + source);
}

struct ModelChain {
  std::string loader;  // unet/ckpt name
  std::string loader_node;  // UNETLoader / CheckpointLoaderSimple node id
  std::vector<std::pair<std::string, double>> loras;  // sampler side first
  std::vector<std::string> lora_nodes;                 // node ids, same order
  std::string uni3c;  // WanUni3CControlnetApply node id, "" if none
  std::string talk;   // WanInfiniteTalkToVideo node id, "" if none
};

ModelChain FollowModel(const Graph& graph, const std::string& sampler) {
  ModelChain chain;
  Link link = graph.RequireLink(sampler, "model");
  for (int guard = 0; guard < 64; ++guard) {
    const std::string cls = graph.Class(link.node);
    if (cls == "LoraLoaderModelOnly" || cls == "LoraLoader") {
      chain.loras.emplace_back(graph.Str(link.node, "lora_name"),
                               graph.Num(link.node, "strength_model", 1.0));
      chain.lora_nodes.push_back(link.node);
    } else if (cls == "WanUni3CControlnetApply") {
      if (!chain.uni3c.empty())
        throw NodeFailure(sampler, "unsupported_link",
                          "the model input passes through two "
                          "WanUni3CControlnetApply nodes");
      chain.uni3c = link.node;
    } else if (cls == "WanInfiniteTalkToVideo") {
      if (chain.talk.empty())
        chain.talk = link.node;
    } else if (cls == "UNETLoader") {
      chain.loader = graph.Str(link.node, "unet_name", "");
      chain.loader_node = link.node;
      return chain;
    } else if (cls == "CheckpointLoaderSimple") {
      chain.loader = graph.Str(link.node, "ckpt_name", "");
      chain.loader_node = link.node;
      return chain;
    } else if (cls != "ModelSamplingAuraFlow" && cls != "ModelSamplingSD3" &&
               cls != "MiniMaxH3SigmaShift") {
      throw NodeFailure(
          sampler, "unsupported_link",
          "the model input passes through unsupported node " + cls);
    }
    link = graph.RequireLink(link.node, "model");
  }
  throw NodeFailure(sampler, "invalid_prompt", "model chain is too long");
}

struct Conditioning {
  std::string encoder;  // node id of the text encoder / H3 conditioning
  int slot{0};
  bool zeroed{false};
  std::vector<std::string> guides;      // MiniMaxH3AddGuide ids, outer first
  std::vector<Link> reference_latents;  // ReferenceLatent "latent" inputs
};

Conditioning FollowConditioning(const Graph& graph, const std::string& sampler,
                                const std::string& key) {
  Conditioning cond;
  auto link = graph.LinkOf(sampler, key);
  if (!link)
    return cond;
  for (int guard = 0; guard < 256; ++guard) {
    const std::string cls = graph.Class(link->node);
    if (cls == "ReferenceLatent") {
      if (auto latent = graph.LinkOf(link->node, "latent"))
        cond.reference_latents.push_back(*latent);
      link = graph.RequireLink(link->node, "conditioning");
    } else if (cls == "MiniMaxH3AddGuide") {
      cond.guides.push_back(link->node);
      link = graph.RequireLink(link->node, "positive");
    } else if (cls == "ConditioningZeroOut") {
      cond.zeroed = true;
      link = graph.RequireLink(link->node, "conditioning");
    } else if (cls == "WanImageToVideo" || cls == "WanCameraImageToVideo" ||
               cls == "WanVaceToVideo") {
      // Outputs (positive, negative, latent): follow the matching input.
      link = graph.RequireLink(link->node,
                               link->slot == 1 ? "negative" : "positive");
    } else if (cls == "WanInfiniteTalkToVideo") {
      // Outputs (model, positive, negative, latent, trim_image).
      link = graph.RequireLink(link->node,
                               link->slot == 2 ? "negative" : "positive");
    } else if (cls == "TextEncodeQwenImage21" || cls == "CLIPTextEncode" ||
               cls == "MiniMaxH3ImageToVideo" ||
               cls == "TextEncodeAceStepAudio1.5") {
      cond.encoder = link->node;
      cond.slot = link->slot;
      return cond;
    } else {
      throw NodeFailure(
          sampler, "unsupported_link",
          "input '" + key + "' passes through unsupported node " + cls);
    }
  }
  throw NodeFailure(sampler, "invalid_prompt", "conditioning chain too long");
}

/// Text of a CLIPTextEncode conditioning, "" when zeroed or absent.
std::string ClipText(const Graph& graph, const Conditioning& cond) {
  if (cond.encoder.empty() || cond.zeroed)
    return "";
  if (graph.Class(cond.encoder) == "CLIPTextEncode")
    return graph.Str(cond.encoder, "text", "");
  return "";
}

std::string SeedKey(const std::string& cls) {
  return cls == "KSamplerAdvanced" ? "noise_seed" : "seed";
}

/// The node behind `key` of `id`, which must be one of `classes`.
Link PartOf(const Graph& graph, const std::string& id, const std::string& key,
            std::initializer_list<std::string_view> classes) {
  const Link link = graph.RequireLink(id, key);
  const std::string cls = graph.Class(link.node);
  if (std::find(classes.begin(), classes.end(), cls) != classes.end())
    return link;
  std::string names;
  for (const std::string_view name : classes)
    names += (names.empty() ? "" : " or ") + std::string(name);
  throw NodeFailure(id, "unsupported_link",
                    "input '" + key + "' must come from " + names + ", not " +
                        cls);
}

/// A copy of input `key` of `id` (link or literal); throws when absent.
json::Value InputOf(const Graph& graph, const std::string& id,
                    const std::string& key) {
  const json::Value* raw = graph.Raw(id, key);
  if (raw == nullptr)
    throw NodeFailure(id, "required_input_missing",
                      "required input '" + key + "' is missing");
  return *raw;
}

/// ComfyUI's split sampler: SamplerCustomAdvanced <- RandomNoise (seed),
/// BasicGuider / CFGGuider (model, conditioning, cfg), KSamplerSelect
/// (sampler) and BasicScheduler (steps, scheduler, denoise). Returns `prompt`
/// with each one folded into a KSampler under the same node id, its parts
/// removed and links to its second output (denoised_output) moved to the
/// first; a null value when the prompt has no SamplerCustomAdvanced.
json::Value LowerCustomSamplers(const json::Value& prompt,
                                std::vector<std::string>* warnings) {
  const Graph graph(prompt);
  std::map<std::string, json::Value> lowered;
  std::set<std::string> parts;
  for (const auto& [id, node] : prompt.members()) {
    if (node.member_str("class_type") != "SamplerCustomAdvanced")
      continue;
    const Link noise = PartOf(graph, id, "noise", {"RandomNoise"});
    const Link guider = PartOf(graph, id, "guider", {"BasicGuider", "CFGGuider"});
    const Link sampler = PartOf(graph, id, "sampler", {"KSamplerSelect"});
    const Link sigmas = PartOf(graph, id, "sigmas", {"BasicScheduler"});
    const bool cfg_guider = graph.Class(guider.node) == "CFGGuider";
    json::Value inputs = json::Value::object();
    inputs["model"] = InputOf(graph, guider.node, "model");
    if (cfg_guider) {
      inputs["positive"] = InputOf(graph, guider.node, "positive");
      inputs["negative"] = InputOf(graph, guider.node, "negative");
      inputs["cfg"] = InputOf(graph, guider.node, "cfg");
    } else {
      inputs["positive"] = InputOf(graph, guider.node, "conditioning");
      inputs["cfg"] = 1.0;
    }
    inputs["latent_image"] = InputOf(graph, id, "latent_image");
    inputs["seed"] = InputOf(graph, noise.node, "noise_seed");
    inputs["sampler_name"] = InputOf(graph, sampler.node, "sampler_name");
    inputs["steps"] = InputOf(graph, sigmas.node, "steps");
    inputs["scheduler"] = InputOf(graph, sigmas.node, "scheduler");
    if (const json::Value* denoise = graph.Raw(sigmas.node, "denoise"))
      inputs["denoise"] = *denoise;
    // BasicScheduler reads the model only for its sigma shift; gufo takes
    // the shift from the guider's model chain, so the two should agree.
    if (auto scheduler_model = graph.LinkOf(sigmas.node, "model")) {
      if (const auto guider_model = graph.LinkOf(guider.node, "model");
          !guider_model || guider_model->node != scheduler_model->node ||
          guider_model->slot != scheduler_model->slot)
        warnings->push_back("node " + sigmas.node +
                            ": BasicScheduler and the guider use different "
                            "models; gufo follows the guider's");
    }
    json::Value folded = json::Value::object();
    folded["class_type"] = "KSampler";
    folded["inputs"] = std::move(inputs);
    lowered.emplace(id, std::move(folded));
    for (const Link* part : {&noise, &guider, &sampler, &sigmas})
      parts.insert(part->node);
  }
  if (lowered.empty())
    return json::Value();
  // A part that also feeds another node stays, so the class check reports it.
  for (const auto& [id, node] : prompt.members()) {
    if (parts.contains(id) || lowered.contains(id))
      continue;
    const json::Value* inputs = node.find("inputs");
    if (inputs == nullptr)
      continue;
    for (const auto& [key, value] : inputs->members()) {
      (void)key;
      if (auto link = Graph::AsLink(value))
        parts.erase(link->node);
    }
  }
  json::Value out = json::Value::object();
  for (const auto& [id, node] : prompt.members()) {
    if (parts.contains(id))
      continue;
    if (auto found = lowered.find(id); found != lowered.end()) {
      out.append_member(id, found->second);
      continue;
    }
    json::Value copy = json::Value::object();
    for (const auto& [field, value] : node.members()) {
      if (field != "inputs" || !value.is_object()) {
        copy.append_member(field, value);
        continue;
      }
      json::Value inputs = json::Value::object();
      for (const auto& [key, input] : value.members()) {
        auto link = Graph::AsLink(input);
        if (link && link->slot == 1 && lowered.contains(link->node)) {
          json::Value moved = json::Value::array();
          moved.push_back(input.items()[0]);
          moved.push_back(0);
          inputs.append_member(key, std::move(moved));
        } else {
          inputs.append_member(key, input);
        }
      }
      copy.append_member(field, std::move(inputs));
    }
    out.append_member(id, std::move(copy));
  }
  return out;
}

}  // namespace

std::string LoraName(std::string_view lora_name) {
  // ComfyUI lists files below models/loras, subfolders included
  // ("style/anime.safetensors"); gufo registers adapters by bare name.
  if (const auto slash = lora_name.find_last_of("/\\");
      slash != std::string_view::npos)
    lora_name.remove_prefix(slash + 1);
  for (const std::string_view ext : {".safetensors", ".sft"}) {
    if (lora_name.size() > ext.size() &&
        Lower(lora_name.substr(lora_name.size() - ext.size())) ==
            ext) {
      lora_name.remove_suffix(ext.size());
      break;
    }
  }
  return std::string(lora_name);
}

bool IsQwenImageTurboFile(std::string_view name) {
  return Lower(name).find("turbo") != std::string::npos;
}

namespace {

ImageTask BuildImage(const Graph& graph, const std::string& sampler,
                     const ModelIds& models, const Link& latent,
                     std::vector<std::string>* warnings) {
  const std::string cls = graph.Class(sampler);
  ImageTask task;
  task.model = models.image;
  task.steps = static_cast<int>(graph.Num(sampler, "steps"));
  task.seed = Seed(graph.Num(sampler, SeedKey(cls), 0));
  if (const double denoise = graph.Num(sampler, "denoise", 1.0);
      cls == "KSampler" && denoise < 0.999) {
    if (denoise <= 0.0)
      throw NodeFailure(sampler, "unsupported_value", "denoise must be > 0");
    task.strength = denoise;
  }
  if (const json::Value* cfg = graph.Literal(sampler, "cfg");
      cfg != nullptr && cfg->is_number())
    task.guidance = cfg->as_double();
  const ModelChain chain = FollowModel(graph, sampler);
  if (IsQwenImageTurboFile(chain.loader)) {
    // A Turbo checkpoint is a different transformer with a fixed 8-step
    // schedule: run it on the Turbo model or fail, never on `image`. Like the
    // other value rules here, settings Turbo cannot honor fail validation
    // instead of being changed silently.
    const bool served =
        !models.image_turbo.empty() &&
        (models.available.empty() ||
         std::find(models.available.begin(), models.available.end(),
                   models.image_turbo) != models.available.end());
    if (!served) {
      const std::string id = models.image_turbo.empty()
                                 ? std::string("qwen-image-2.1-turbo")
                                 : models.image_turbo;
      throw NodeFailure(chain.loader_node, "value_not_in_list",
                        chain.loader +
                            " is a Qwen-Image Turbo checkpoint, but the "
                            "router serves no model '" + id +
                            "' (add an [external/" + id +
                            "] section to the router preset)");
    }
    task.model = models.image_turbo;
    if (task.steps != 8)
      throw NodeFailure(sampler, "unsupported_value",
                        "Qwen-Image-2.1-Turbo (" + chain.loader +
                            ") runs exactly 8 steps, not " +
                            std::to_string(task.steps) + "; set steps to 8");
    if (task.strength)
      throw NodeFailure(sampler, "unsupported_value",
                        "Qwen-Image-2.1-Turbo has no partial-strength "
                        "image to image; set denoise to 1");
  }
  // LoraLoader / LoraLoaderModelOnly -> "loras", in the order ComfyUI applies
  // them (loader side first). The worker merges them into the transformer;
  // the CLIP half of LoraLoader has no gufo counterpart.
  for (std::size_t i = chain.loras.size(); i-- > 0;) {
    const auto& [file, strength] = chain.loras[i];
    const std::string& node = chain.lora_nodes[i];
    const std::string name = LoraName(file);
    if (name.empty())
      throw NodeFailure(node, "unsupported_value", "lora_name is empty");
    if (!std::isfinite(strength) || std::abs(strength) > 10)
      throw NodeFailure(node, "unsupported_value",
                        "strength_model must be finite and at most 10 in "
                        "magnitude for gufo Qwen-Image");
    if (graph.Class(node) == "LoraLoader" &&
        graph.Num(node, "strength_clip", 1.0) != 0.0)
      warnings->push_back("node " + node +
                          ": strength_clip is ignored (gufo applies " + name +
                          " to the Qwen-Image transformer only)");
    if (strength == 0.0)
      continue;
    auto same = std::find_if(task.loras.begin(), task.loras.end(),
                             [&](const auto& l) { return l.first == name; });
    if (same != task.loras.end())
      same->second += strength;  // the same adapter twice adds its deltas
    else
      task.loras.emplace_back(name, strength);
  }
  if (task.loras.size() > 16)
    throw NodeFailure(sampler, "unsupported_value",
                      "gufo Qwen-Image takes at most 16 LoRAs per image");
  if (!chain.loader.empty() &&
      Lower(chain.loader).find("qwen_image") == std::string::npos)
    warnings->push_back("model " + chain.loader + " is served by " +
                        task.model);
  const Conditioning positive = FollowConditioning(graph, sampler, "positive");
  if (positive.encoder.empty())
    throw NodeFailure(sampler, "required_input_missing",
                      "positive conditioning is not connected");
  const std::string encoder_class = graph.Class(positive.encoder);
  if (encoder_class == "TextEncodeQwenImage21") {
    task.prompt = graph.Str(positive.encoder, "prompt");
    // Reference images "images.image_N", in N order.
    std::map<int, Link> ordered;
    const json::Value* node = graph.Node(positive.encoder);
    for (const auto& [key, value] : node->find("inputs")->members()) {
      if (!key.starts_with("images."))
        continue;
      const auto digits = key.find_last_not_of("0123456789");
      const int index = digits + 1 < key.size()
                            ? std::atoi(key.c_str() + digits + 1)
                            : static_cast<int>(ordered.size()) + 1000;
      if (auto link = Graph::AsLink(value))
        ordered.emplace(index, *link);
    }
    for (const auto& [index, link] : ordered) {
      (void)index;
      task.references.push_back(
          ImageFile(graph, positive.encoder, "images", link));
    }
    task.resolution =
        static_cast<int>(graph.Num(positive.encoder, "resolution", 1024));
  } else if (encoder_class == "CLIPTextEncode") {
    task.prompt = graph.Str(positive.encoder, "text");
  } else {
    throw NodeFailure(sampler, "unsupported_link",
                      "positive conditioning must come from "
                      "TextEncodeQwenImage21 or CLIPTextEncode");
  }
  for (const Link& reference : positive.reference_latents) {
    if (graph.Class(reference.node) != "VAEEncode")
      throw NodeFailure(sampler, "unsupported_link",
                        "ReferenceLatent latents must come from VAEEncode");
    task.references.push_back(
        ImageFile(graph, reference.node, "pixels",
                  graph.RequireLink(reference.node, "pixels")));
  }
  if (task.references.size() > 10)
    throw NodeFailure(positive.encoder, "unsupported_value",
                      "gufo Qwen-Image takes at most 10 reference images");
  // Negative prompt: sent as gufo.negative_prompt (with true CFG).
  const Conditioning negative = FollowConditioning(graph, sampler, "negative");
  if (!negative.encoder.empty() && !negative.zeroed) {
    if (graph.Class(negative.encoder) == "TextEncodeQwenImage21")
      task.negative_prompt = graph.Str(negative.encoder, "negative_prompt", "");
    else
      task.negative_prompt = ClipText(graph, negative);
  }
  // Masked editing: SetLatentNoiseMask over VAEEncode, or VAEEncodeForInpaint.
  // gufo keeps the unmasked latent cells on the start image every step.
  Link source = latent;
  if (graph.Class(latent.node) == "SetLatentNoiseMask") {
    MaskFile(graph, latent.node, graph.RequireLink(latent.node, "mask"),
             &task);
    source = graph.RequireLink(latent.node, "samples");
    if (graph.Class(source.node) != "VAEEncode")
      throw NodeFailure(latent.node, "unsupported_link",
                        "SetLatentNoiseMask samples must come from VAEEncode "
                        "of a LoadImage");
  }
  const std::string latent_class = graph.Class(source.node);
  if (latent_class == "VAEEncodeForInpaint") {
    MaskFile(graph, source.node, graph.RequireLink(source.node, "mask"),
             &task);
    task.size_source = SizeSource::kCanvas;
    task.canvas = ImageFile(graph, source.node, "pixels",
                            graph.RequireLink(source.node, "pixels"));
    task.init_image = task.canvas;
    // ComfyUI blanks the masked pixels to gray for inpainting checkpoints and
    // grows the mask; gufo starts from the original pixels (the same at
    // denoise 1) and rounds the mask out to 16-pixel cells.
    warnings->push_back("node " + source.node +
                        ": VAEEncodeForInpaint runs as SetLatentNoiseMask "
                        "(original pixels under the mask, mask rounded out "
                        "to 16-pixel cells; grow_mask_by ignored)");
    return task;
  }
  if (latent_class == "TextEncodeQwenImage21") {
    task.size_source = task.references.empty() ? SizeSource::kSquare
                                               : SizeSource::kFirstReference;
    if (task.resolution <= 0 && task.references.empty())
      task.resolution = 1024;
  } else if (latent_class == "VAEEncode") {
    task.size_source = SizeSource::kCanvas;
    task.canvas = ImageFile(graph, source.node, "pixels",
                            graph.RequireLink(source.node, "pixels"));
    if (task.strength || !task.mask.name.empty())
      task.init_image = task.canvas;
  } else if (latent_class == "EmptyLatentImage" ||
             latent_class == "EmptySD3LatentImage") {
    task.size_source = SizeSource::kExplicit;
    task.width = static_cast<int>(graph.Num(latent.node, "width"));
    task.height = static_cast<int>(graph.Num(latent.node, "height"));
    task.batch = static_cast<int>(graph.Num(latent.node, "batch_size", 1));
    if (task.batch < 1 || task.batch > 10)
      throw NodeFailure(latent.node, "unsupported_value",
                        "batch_size must be 1..10");
  } else {
    throw NodeFailure(sampler, "unsupported_link",
                      "latent_image must come from TextEncodeQwenImage21, "
                      "VAEEncode or an empty latent, not " +
                          latent_class);
  }
  if (task.strength && task.init_image.name.empty())
    throw NodeFailure(sampler, "unsupported_value",
                      "denoise below 1 needs latent_image from VAEEncode of a "
                      "LoadImage (image to image)");
  return task;
}

VideoTask BuildH3(const Graph& graph, const std::string& sampler,
                  const ModelIds& models, const Link& latent) {
  const std::string video = latent.node;
  VideoTask task;
  task.family = VideoFamily::kH3;
  task.model = models.h3;
  task.prompt = graph.Str(video, "prompt");
  task.width = static_cast<int>(graph.Num(video, "width", 1344));
  task.height = static_cast<int>(graph.Num(video, "height", 768));
  const int length = static_cast<int>(graph.Num(video, "length", 124));
  task.frames = SnapH3Frames(length);
  if (task.frames < 22)
    task.frames = 22;
  if (task.frames > 345)
    throw NodeFailure(video, "unsupported_value",
                      "gufo MiniMax H3 renders 22..345 frames (17k + 5, up to "
                      "14.4 s at 24 fps), not " +
                          std::to_string(task.frames));
  task.steps = static_cast<int>(graph.Num(sampler, "steps"));
  task.seed = Seed(graph.Num(sampler, SeedKey(graph.Class(sampler)), 0));
  // gufo H3 sizes: multiples of 32, 256..1344 per side, at most 768x1344.
  if (task.width % 32 != 0 || task.height % 32 != 0 || task.width < 256 ||
      task.height < 256 || task.width > 1344 || task.height > 1344 ||
      task.width * task.height > 768 * 1344)
    throw NodeFailure(video, "unsupported_value",
                      "gufo MiniMax H3 needs width/height in multiples of 32, "
                      "256..1344, at most 768x1344 pixels, not " +
                          std::to_string(task.width) + "x" +
                          std::to_string(task.height));
  if (auto first = graph.LinkOf(video, "first_frame"))
    task.first_frame = ImageFile(graph, video, "first_frame", *first);
  if (auto last = graph.LinkOf(video, "last_frame"))
    task.last_frame = ImageFile(graph, video, "last_frame", *last);
  const Conditioning positive = FollowConditioning(graph, sampler, "positive");
  if (positive.encoder != video)
    throw NodeFailure(sampler, "unsupported_link",
                      "positive conditioning must come from the "
                      "MiniMaxH3ImageToVideo that makes the latent");
  for (const std::string& guide : positive.guides) {
    const int frame = static_cast<int>(graph.Num(guide, "frame_idx", 0));
    if (auto audio = graph.LinkOf(guide, "audio")) {
      // A LoadAudio file, or the soundtrack of a LoadVideo through
      // GetVideoComponents (the anchor clip's own sound).
      const bool from_video =
          graph.Class(audio->node) == "GetVideoComponents";
      const FileRef file =
          from_video ? VideoComponentsFile(graph, guide, "audio", *audio, 1)
                     : AudioFile(graph, guide, "audio", *audio);
      if (task.audio_reference &&
          !(*task.audio_reference == file &&
            task.audio_reference_frame == frame &&
            task.audio_reference_from_video == from_video))
        throw NodeFailure(guide, "unsupported_value",
                          "gufo H3 takes one audio anchor per video");
      task.audio_reference = file;
      task.audio_reference_frame = frame;
      task.audio_reference_from_video = from_video;
    }
    auto image = graph.LinkOf(guide, "image");
    if (!image)
      continue;
    if (graph.Class(image->node) == "GetVideoComponents") {
      // A frame batch: gufo anchors its first 17k + 5 frames from `frame`
      // on (fewer than 5 frames: the first one), like ComfyUI.
      const FileRef file =
          VideoComponentsFile(graph, guide, "image", *image, 0);
      if (task.guide_video && !(*task.guide_video == file &&
                                task.guide_video_frame == frame))
        throw NodeFailure(guide, "unsupported_value",
                          "gufo H3 takes one guide clip per video");
      task.guide_video = file;
      task.guide_video_frame = frame;
      continue;
    }
    const FileRef file = ImageFile(graph, guide, "image", *image);
    const bool is_last = frame < 0 ? frame == -1 : frame >= task.frames - 1;
    if (frame == 0) {
      if (task.first_frame && !(*task.first_frame == file))
        throw NodeFailure(guide, "unsupported_value",
                          "two different first frames");
      task.first_frame = file;
    } else if (is_last) {
      if (task.last_frame && !(*task.last_frame == file))
        throw NodeFailure(guide, "unsupported_value",
                          "two different last frames");
      task.last_frame = file;
    } else {
      for (const auto& [at, existing] : task.keyframes)
        if (at == frame && !(existing == file))
          throw NodeFailure(
              guide, "unsupported_value",
              "two different guide images at frame " + std::to_string(frame));
      task.keyframes.emplace_back(frame, file);
    }
  }
  std::sort(task.keyframes.begin(), task.keyframes.end(),
            [&](const auto& a, const auto& b) {
              const auto absolute = [&](int f) {
                return f < 0 ? task.frames + f : f;
              };
              return absolute(a.first) < absolute(b.first);
            });
  task.keyframes.erase(
      std::unique(task.keyframes.begin(), task.keyframes.end()),
      task.keyframes.end());
  const std::size_t anchors = task.keyframes.size() +
                              (task.first_frame ? 1 : 0) +
                              (task.last_frame ? 1 : 0) +
                              (task.guide_video ? 1 : 0);
  if (anchors > 8)
    throw NodeFailure(video, "unsupported_value",
                      "gufo H3 takes at most 8 guide images and clips "
                      "including the first and last frame");
  const ModelChain chain = FollowModel(graph, sampler);
  for (const auto& [name, strength] : chain.loras) {
    if (Lower(name).find("turbo") == std::string::npos)
      throw NodeFailure(sampler, "unsupported_value",
                        "gufo H3 knows only the turbo LoRA, not " + name);
    task.lora = "turbo";
    if (strength != 1.0)
      task.lora_strength = strength;
  }
  return task;
}

VideoTask BuildWan(const Graph& graph, const std::string& sampler,
                   const ModelIds& models, const Link& latent) {
  VideoTask task;
  task.family = VideoFamily::kWan;
  task.model = models.wan;
  task.width = static_cast<int>(graph.Num(latent.node, "width", 1280));
  task.height = static_cast<int>(graph.Num(latent.node, "height", 704));
  task.frames = static_cast<int>(graph.Num(latent.node, "length", 121));
  if (task.frames < 5 || (task.frames - 1) % 4 != 0 || task.frames > 121)
    throw NodeFailure(latent.node, "unsupported_value",
                      "Wan 2.2 TI2V length must be 4k+1 frames, 5..121");
  if (auto start = graph.LinkOf(latent.node, "start_image"))
    task.first_frame = ImageFile(graph, latent.node, "start_image", *start);
  const Conditioning positive = FollowConditioning(graph, sampler, "positive");
  if (positive.encoder.empty() ||
      graph.Class(positive.encoder) != "CLIPTextEncode")
    throw NodeFailure(sampler, "unsupported_link",
                      "Wan positive conditioning must come from "
                      "CLIPTextEncode");
  task.prompt = graph.Str(positive.encoder, "text");
  task.negative_prompt =
      ClipText(graph, FollowConditioning(graph, sampler, "negative"));
  task.steps = static_cast<int>(graph.Num(sampler, "steps"));
  task.seed = Seed(graph.Num(sampler, SeedKey(graph.Class(sampler)), 0));
  task.guidance = graph.Num(sampler, "cfg", 5.0);
  const ModelChain chain = FollowModel(graph, sampler);
  if (!chain.loras.empty())
    throw NodeFailure(sampler, "unsupported_value",
                      "LoRAs are not supported for Wan 2.2 TI2V in gufo");
  return task;
}

/// ref_audio / ref_text / x_vector_only of a clone node (VoiceClone or
/// VoiceClonePrompt) into `task`.
void ApplyCloneReference(const Graph& graph, const std::string& id,
                         SpeechTask* task) {
  task->reference_audio =
      AudioFile(graph, id, "ref_audio", graph.RequireLink(id, "ref_audio"));
  task->reference_text = graph.Str(id, "ref_text", "");
  task->embedding_only =
      graph.Bool(id, "x_vector_only", false) || task->reference_text.empty();
}

void ApplyClonePrompt(const Graph& graph, const std::string& id,
                      const Link& link, SpeechTask* task) {
  if (graph.Class(link.node) != "FB_Qwen3TTSVoiceClonePrompt")
    throw NodeFailure(id, "unsupported_link",
                      "voice_clone_prompt must come from "
                      "FB_Qwen3TTSVoiceClonePrompt, not " +
                          graph.Class(link.node));
  ApplyCloneReference(graph, link.node, task);
}

/// Language, seed and sampling fields shared by every TTS node.
void ApplySpeechCommon(const Graph& graph, const std::string& id,
                       SpeechTask* task, std::vector<std::string>* warnings) {
  // ComfyUI "Auto" lets Qwen3-TTS pick the language (codec "nothink"
  // prefix). gufo's own default for an omitted language is English, so
  // "Auto" must be sent as "auto" rather than left out.
  task->language = Lower(graph.Str(id, "language", "Auto"));
  if (const json::Value* seed = graph.Literal(id, "seed");
      seed != nullptr && seed->is_number())
    task->seed = Seed(seed->as_double());
  if (graph.Str(id, "model_choice", "1.7B") != "1.7B")
    warnings->push_back("node " + id + ": gufo serves the 1.7B Qwen3-TTS");
  for (const char* key : {"temperature", "top_p", "top_k", "repetition_penalty",
                          "max_new_tokens"}) {
    if (const json::Value* value = graph.Literal(id, key);
        value != nullptr && value->is_number())
      task->sampling[key] = value->as_double();
  }
}

std::string Trim(std::string text) {
  while (!text.empty() &&
         (text.back() == ' ' || text.back() == '\t' || text.back() == '\r'))
    text.pop_back();
  std::size_t first = 0;
  while (first < text.size() && (text[first] == ' ' || text[first] == '\t'))
    ++first;
  return text.substr(first);
}

/// FB_Qwen3TTSDialogueInference: each "Role: text" script line becomes one
/// clone request with the role's FB_Qwen3TTSVoiceClonePrompt.
DialogueTask BuildDialogue(const Graph& graph, const std::string& id,
                           const ModelIds& models,
                           std::vector<std::string>* warnings) {
  const Link bank = graph.RequireLink(id, "role_bank");
  if (graph.Class(bank.node) != "FB_Qwen3TTSRoleBank")
    throw NodeFailure(id, "unsupported_link",
                      "role_bank must come from FB_Qwen3TTSRoleBank");
  std::map<std::string, SpeechTask> roles;
  for (int i = 1; i <= 8; ++i) {
    const std::string index = std::to_string(i);
    auto prompt = graph.LinkOf(bank.node, "prompt_" + index);
    if (!prompt)
      continue;
    SpeechTask role;
    role.model = models.tts_clone;
    role.voice = "voice-clone";
    ApplyClonePrompt(graph, bank.node, *prompt, &role);
    const std::string name =
        Trim(graph.Str(bank.node, "role_name_" + index, "Role" + index));
    roles[Lower(name)] = std::move(role);
  }
  if (roles.empty())
    throw NodeFailure(bank.node, "required_input_missing",
                      "the role bank has no voice prompts");
  SpeechTask common;
  ApplySpeechCommon(graph, id, &common, warnings);
  if (const json::Value* tokens = graph.Literal(id, "max_new_tokens_per_line");
      tokens != nullptr && tokens->is_number())
    common.sampling["max_new_tokens"] = tokens->as_double();
  DialogueTask task;
  task.pause_seconds = graph.Num(id, "pause_linebreak", 0.5);
  task.merge = graph.Bool(id, "merge_outputs", true);
  const std::string script = graph.Str(id, "script");
  std::size_t start = 0;
  while (start <= script.size()) {
    std::size_t end = script.find('\n', start);
    if (end == std::string::npos)
      end = script.size();
    const std::string line = Trim(script.substr(start, end - start));
    start = end + 1;
    if (line.empty())
      continue;
    // "Role: text"; the full-width colon counts too.
    std::size_t colon = line.find(':');
    std::size_t skip = 1;
    if (const std::size_t wide = line.find("\xEF\xBC\x9A");
        wide != std::string::npos &&
        (colon == std::string::npos || wide < colon)) {
      colon = wide;
      skip = 3;
    }
    if (colon == std::string::npos)
      throw NodeFailure(id, "invalid_input",
                        "script line without 'Role:' prefix: " + line);
    const std::string role = Trim(line.substr(0, colon));
    const std::string text = Trim(line.substr(colon + skip));
    const auto found = roles.find(Lower(role));
    if (found == roles.end())
      throw NodeFailure(id, "invalid_input",
                        "script role '" + role + "' is not in the role bank");
    if (text.empty())
      continue;
    SpeechTask spoken = found->second;
    spoken.text = text;
    spoken.language = common.language;
    spoken.seed = common.seed;
    spoken.sampling = common.sampling;
    task.lines.push_back(std::move(spoken));
  }
  if (task.lines.empty())
    throw NodeFailure(id, "required_input_missing", "the script has no lines");
  for (const char* key :
       {"period_pause", "comma_pause", "question_pause", "hyphen_pause"}) {
    if (const json::Value* value = graph.Literal(id, key);
        value != nullptr && value->is_number() && value->as_double() > 0)
      warnings->push_back("node " + id + ": " + key +
                          " is not applied (gufo pauses only between lines)");
  }
  return task;
}

SpeechTask BuildSpeech(const Graph& graph, const std::string& id,
                       const ModelIds& models,
                       std::vector<std::string>* warnings) {
  const std::string cls = graph.Class(id);
  SpeechTask task;
  if (cls == "FB_Qwen3TTSVoiceDesign") {
    task.model = models.tts_design;
    task.voice = "voice-design";
    task.text = graph.Str(id, "text");
    task.instructions = graph.Str(id, "instruct");
  } else if (cls == "FB_Qwen3TTSVoiceClone") {
    task.model = models.tts_clone;
    task.voice = "voice-clone";
    task.text = graph.Str(id, "target_text");
    // A FB_Qwen3TTSVoiceClonePrompt carries the reference instead of
    // ref_audio/ref_text on the node itself.
    if (auto prompt = graph.LinkOf(id, "voice_clone_prompt"))
      ApplyClonePrompt(graph, id, *prompt, &task);
    else
      ApplyCloneReference(graph, id, &task);
    task.instructions = graph.Str(id, "instruct", "");
  } else if (cls == "FB_Qwen3TTSCustomVoice") {
    task.model = models.tts_custom;
    task.text = graph.Str(id, "text");
    task.voice = Lower(graph.Str(id, "speaker", "ryan"));
    task.instructions = graph.Str(id, "instruct", "");
    if (!graph.Str(id, "custom_model_path", "").empty())
      throw NodeFailure(id, "unsupported_value",
                        "custom_model_path is not supported by gufo");
  } else {
    throw NodeFailure(id, "invalid_prompt", "not a TTS node");
  }
  if (task.text.empty())
    throw NodeFailure(id, "required_input_missing", "text is empty");
  ApplySpeechCommon(graph, id, &task, warnings);
  return task;
}

/// ACE-Step 1.5 (ComfyUI's template): KSampler <- EmptyAceStep1.5LatentAudio,
/// positive TextEncodeAceStepAudio1.5, negative its ConditioningZeroOut.
MusicTask BuildMusic(const Graph& graph, const std::string& sampler,
                     const ModelIds& models, const Link& latent,
                     std::vector<std::string>* warnings) {
  MusicTask task;
  task.model = models.music;
  task.duration = graph.Num(latent.node, "seconds", 120.0);
  if (task.duration < 1.0 || task.duration > 600.0)
    throw NodeFailure(latent.node, "unsupported_value",
                      "gufo ACE-Step renders 1..600 seconds");
  if (graph.Num(latent.node, "batch_size", 1) != 1)
    throw NodeFailure(latent.node, "unsupported_value",
                      "gufo ACE-Step renders batch_size 1");
  task.steps = static_cast<int>(graph.Num(sampler, "steps", 8));
  if (task.steps < 1 || task.steps > 50)
    throw NodeFailure(sampler, "unsupported_value",
                      "gufo ACE-Step takes 1..50 steps");
  task.seed = Seed(graph.Num(sampler, SeedKey(graph.Class(sampler)), 0));
  if (graph.Num(sampler, "denoise", 1.0) < 0.999)
    throw NodeFailure(sampler, "unsupported_value",
                      "gufo ACE-Step has no audio-to-audio (denoise < 1)");
  const ModelChain chain = FollowModel(graph, sampler);
  if (!chain.loras.empty() || !chain.uni3c.empty())
    throw NodeFailure(sampler, "unsupported_value",
                      "LoRAs are not supported for ACE-Step in gufo");
  if (!chain.loader.empty() &&
      Lower(chain.loader).find("acestep_v1.5_xl_turbo") == std::string::npos)
    warnings->push_back("model " + chain.loader + " is served by " +
                        task.model);
  const Conditioning positive = FollowConditioning(graph, sampler, "positive");
  if (positive.encoder.empty() ||
      graph.Class(positive.encoder) != "TextEncodeAceStepAudio1.5" ||
      positive.zeroed)
    throw NodeFailure(sampler, "unsupported_link",
                      "positive conditioning must come from "
                      "TextEncodeAceStepAudio1.5");
  const std::string& text = positive.encoder;
  task.tags = graph.Str(text, "tags");
  if (task.tags.empty())
    throw NodeFailure(text, "required_input_missing", "tags are empty");
  task.lyrics = graph.Str(text, "lyrics", "");
  task.bpm = static_cast<int>(graph.Num(text, "bpm", 0));
  task.timesignature = graph.Str(text, "timesignature", "");
  task.language = graph.Str(text, "language", "");
  if (task.language == "unknown")
    task.language.clear();
  task.keyscale = graph.Str(text, "keyscale", "");
  if (const json::Value* codes = graph.Literal(text, "generate_audio_codes");
      codes != nullptr && codes->is_bool())
    task.audio_codes = codes->as_bool();
  for (const auto& [key, field] :
       {std::pair<const char*, std::optional<double>*>{"cfg_scale",
                                                       &task.lm_cfg},
        {"temperature", &task.lm_temperature},
        {"top_p", &task.lm_top_p}}) {
    if (const json::Value* value = graph.Literal(text, key);
        value != nullptr && value->is_number())
      *field = value->as_double();
  }
  for (const char* key : {"top_k", "min_p"}) {
    if (const json::Value* value = graph.Literal(text, key);
        value != nullptr && value->is_number() && value->as_double() != 0)
      warnings->push_back("node " + text + ": " + key +
                          " is not applied (gufo ACE-Step has no such field)");
  }
  const double duration = graph.Num(text, "duration", task.duration);
  if (std::abs(duration - task.duration) > 0.05)
    warnings->push_back("node " + text + ": duration " +
                        std::to_string(duration) + " differs from the latent; "
                        "gufo renders the latent's " +
                        std::to_string(task.duration) + " s");
  if (const json::Value* seed = graph.Literal(text, "seed");
      seed != nullptr && seed->is_number() &&
      Seed(seed->as_double()) != task.seed)
    warnings->push_back("node " + text +
                        ": gufo ACE-Step takes one seed (the KSampler's)");
  return task;
}

/// gufo.camera types (wan2.1-uni3c, models/wan22/camera_path).
bool IsCameraType(std::string_view type) {
  static constexpr std::string_view kTypes[] = {
      "static",     "pan_left",    "pan_right",  "tilt_up",   "tilt_down",
      "zoom_in",    "zoom_out",    "dolly_in",   "dolly_out", "truck_left",
      "truck_right", "crane_up",   "crane_down", "orbit_left", "orbit_right"};
  return Contains(kTypes, type);
}

CameraMove GufoCameraMoveOf(const Graph& graph, const std::string& id) {
  CameraMove move;
  move.type = Lower(graph.Str(id, "camera", graph.Str(id, "type", "static")));
  if (!IsCameraType(move.type))
    throw NodeFailure(id, "unsupported_value",
                      "unknown camera move '" + move.type + "'");
  move.speed = graph.Num(id, "speed", 1.0);
  move.ease = graph.Str(id, "ease", "linear");
  if (const json::Value* amount = graph.Literal(id, "amount");
      amount != nullptr && amount->is_number() && amount->as_double() >= 0)
    move.amount = amount->as_double();
  return move;
}

/// ComfyUI's WanCameraEmbedding presets (Wan Fun Camera trajectories: "Pan"
/// and "Zoom" move the camera, they do not turn it) as gufo moves.
CameraMove WanCameraEmbeddingOf(const Graph& graph, const std::string& id,
                                std::vector<std::string>* warnings) {
  static constexpr std::pair<std::string_view, std::string_view> kMap[] = {
      {"Static", "static"},        {"Pan Up", "crane_up"},
      {"Pan Down", "crane_down"},  {"Pan Left", "truck_left"},
      {"Pan Right", "truck_right"}, {"Zoom In", "dolly_in"},
      {"Zoom Out", "dolly_out"}};
  const std::string pose = graph.Str(id, "camera_pose", "Static");
  CameraMove move;
  move.type.clear();
  for (const auto& [comfy_name, gufo_name] : kMap)
    if (pose == comfy_name)
      move.type = std::string(gufo_name);
  if (move.type.empty())
    throw NodeFailure(id, "unsupported_value",
                      "camera_pose '" + pose +
                          "' (roll) has no gufo preset; use GufoCameraMove");
  move.speed = graph.Num(id, "speed", 1.0);
  warnings->push_back("node " + id + ": WanCameraEmbedding '" + pose +
                      "' runs as gufo camera " + move.type +
                      " on wan2.1-uni3c (Uni3C, not Wan Fun Camera)");
  return move;
}

/// Wan 2.1 I2V + Uni3C camera moves (wan2.1-uni3c, 16 fps).
VideoTask BuildCamera(const Graph& graph, const std::string& sampler,
                      const ModelIds& models, const Link& latent,
                      std::vector<std::string>* warnings) {
  const std::string video = latent.node;
  const std::string video_class = graph.Class(video);
  VideoTask task;
  task.family = VideoFamily::kCamera;
  task.model = models.camera;
  task.fps = 16;
  task.width = static_cast<int>(graph.Num(video, "width", 832));
  task.height = static_cast<int>(graph.Num(video, "height", 480));
  if (task.width % 16 != 0 || task.height % 16 != 0 || task.width < 16 ||
      task.height < 16 || task.width * task.height > 832 * 480)
    throw NodeFailure(video, "unsupported_value",
                      "gufo wan2.1-uni3c needs width/height in multiples of "
                      "16 and at most 832x480 pixels, not " +
                          std::to_string(task.width) + "x" +
                          std::to_string(task.height));
  task.frames = static_cast<int>(graph.Num(video, "length", 81));
  if (task.frames < 5 || task.frames > 81 || (task.frames - 1) % 4 != 0)
    throw NodeFailure(video, "unsupported_value",
                      "gufo wan2.1-uni3c length must be 4k+1 frames, 5..81 "
                      "(16 fps)");
  if (graph.Num(video, "batch_size", 1) != 1)
    throw NodeFailure(video, "unsupported_value",
                      "gufo wan2.1-uni3c renders batch_size 1");
  auto start = graph.LinkOf(video, "start_image");
  if (!start)
    throw NodeFailure(video, "required_input_missing",
                      "gufo wan2.1-uni3c needs a start_image");
  task.first_frame = ImageFile(graph, video, "start_image", *start);
  const Conditioning positive = FollowConditioning(graph, sampler, "positive");
  if (positive.encoder.empty() ||
      graph.Class(positive.encoder) != "CLIPTextEncode")
    throw NodeFailure(sampler, "unsupported_link",
                      "Wan positive conditioning must come from "
                      "CLIPTextEncode");
  task.prompt = graph.Str(positive.encoder, "text");
  if (!ClipText(graph, FollowConditioning(graph, sampler, "negative")).empty())
    warnings->push_back("node " + sampler +
                        ": the negative prompt is not applied "
                        "(gufo wan2.1-uni3c has no negative prompt)");
  task.steps = static_cast<int>(graph.Num(sampler, "steps"));
  if (task.steps < 1 || task.steps > 20)
    throw NodeFailure(sampler, "unsupported_value",
                      "gufo wan2.1-uni3c takes 1..20 steps");
  task.seed = Seed(graph.Num(sampler, SeedKey(graph.Class(sampler)), 0));
  if (graph.Num(sampler, "cfg", 1.0) != 1.0)
    warnings->push_back("node " + sampler +
                        ": cfg is not applied (gufo wan2.1-uni3c runs the "
                        "distilled model)");
  const ModelChain chain = FollowModel(graph, sampler);
  for (const auto& [name, strength] : chain.loras) {
    (void)strength;
    const std::string lower = Lower(name);
    if (lower.find("lightx2v") == std::string::npos &&
        lower.find("distill") == std::string::npos)
      throw NodeFailure(sampler, "unsupported_value",
                        "gufo wan2.1-uni3c has its step-distill LoRA built "
                        "in and takes no other LoRA, not " +
                            name);
  }
  // The camera: WanUni3CControlnetApply.render_video, or the Fun Camera
  // conditions of WanCameraImageToVideo.
  std::optional<Link> source;
  std::string source_node;
  if (!chain.uni3c.empty()) {
    for (const auto& [key, fallback] :
         {std::pair<const char*, double>{"strength", 1.0},
          {"start_percent", 0.0},
          {"end_percent", 1.0}})
      if (graph.Num(chain.uni3c, key, fallback) != fallback)
        warnings->push_back("node " + chain.uni3c + ": " + key +
                            " is not applied (gufo runs Uni3C at full "
                            "strength over all steps)");
    source = graph.RequireLink(chain.uni3c, "render_video");
    source_node = chain.uni3c;
  }
  if (video_class == "WanCameraImageToVideo") {
    if (auto conditions = graph.LinkOf(video, "camera_conditions")) {
      if (source)
        throw NodeFailure(video, "unsupported_link",
                          "camera_conditions and a Uni3C render_video "
                          "exclude each other");
      source = conditions;
      source_node = video;
    }
  }
  if (source) {
    const std::string cls = graph.Class(source->node);
    if (cls == "GufoCameraMove") {
      task.camera = GufoCameraMoveOf(graph, source->node);
    } else if (cls == "WanCameraEmbedding") {
      task.camera = WanCameraEmbeddingOf(graph, source->node, warnings);
    } else if (cls == "VHS_LoadVideo") {
      task.render_video =
          ParseFileRef(graph.Str(source->node, "video"), "input");
    } else if (cls == "GetVideoComponents") {
      const Link file = graph.RequireLink(source->node, "video");
      if (graph.Class(file.node) != "LoadVideo")
        throw NodeFailure(source->node, "unsupported_link",
                          "the guide video must come from LoadVideo");
      task.render_video = ParseFileRef(graph.Str(file.node, "file"), "input");
    } else {
      throw NodeFailure(source_node, "unsupported_link",
                        "the camera must come from GufoCameraMove, "
                        "WanCameraEmbedding or a loaded guide video, not " +
                            cls);
    }
  }
  if (task.camera) {
    if (!(task.camera->speed >= 0.05 && task.camera->speed <= 4.0))
      throw NodeFailure(source->node, "unsupported_value",
                        "camera speed must be 0.05..4");
    if (task.camera->ease != "linear" && task.camera->ease != "ease_in_out")
      throw NodeFailure(source->node, "unsupported_value",
                        "camera ease must be linear or ease_in_out");
    if (task.camera->amount && *task.camera->amount > 360)
      throw NodeFailure(source->node, "unsupported_value",
                        "camera amount must be 0..360");
  }
  return task;
}

/// A video an IMAGE input receives: GetVideoComponents <- LoadVideo, or
/// VHS_LoadVideo (gufo decodes and fits the file itself).
FileRef VideoFile(const Graph& graph, const std::string& id,
                  const std::string& key, const Link& link,
                  std::vector<std::string>* warnings) {
  const std::string source = graph.Class(link.node);
  if (source == "VHS_LoadVideo") {
    // gufo reads the file's frames in order from the start.
    for (const char* option :
         {"force_rate", "skip_first_frames", "select_every_nth"}) {
      const double fallback = std::string_view(option) == "select_every_nth"
                                  ? 1.0
                                  : 0.0;
      if (graph.Num(link.node, option, fallback) != fallback)
        warnings->push_back("node " + link.node + ": " + option +
                            " is not applied (gufo reads every frame from "
                            "the start; prepare the video at 16 fps)");
    }
    return ParseFileRef(graph.Str(link.node, "video"), "input");
  }
  if (source == "GetVideoComponents") {
    const Link file = graph.RequireLink(link.node, "video");
    if (graph.Class(file.node) != "LoadVideo")
      throw NodeFailure(link.node, "unsupported_link",
                        "the video must come from LoadVideo");
    return ParseFileRef(graph.Str(file.node, "file"), "input");
  }
  throw NodeFailure(id, "unsupported_link",
                    "input '" + key +
                        "' must come from LoadVideo (GetVideoComponents) or "
                        "VHS_LoadVideo, not " +
                        source +
                        " (run preprocessors in ComfyUI and load their "
                        "result as a video)");
}

/// Wan 2.1 VACE 14B (wan2.1-vace-14b, 16 fps): KSampler <- WanVaceToVideo,
/// TrimVideoLatent behind the sampler (gufo always trims the reference
/// frames).
VideoTask BuildVace(const Graph& graph, const std::string& sampler,
                    const ModelIds& models, const Link& latent,
                    std::vector<std::string>* warnings) {
  const std::string vace = latent.node;
  VideoTask task;
  task.family = VideoFamily::kVace;
  task.model = models.vace;
  task.fps = 16;
  task.width = static_cast<int>(graph.Num(vace, "width", 832));
  task.height = static_cast<int>(graph.Num(vace, "height", 480));
  if (task.width % 16 != 0 || task.height % 16 != 0 || task.width < 256 ||
      task.height < 256 || task.width > 1280 || task.height > 1280 ||
      task.width * task.height > 1280 * 720)
    throw NodeFailure(vace, "unsupported_value",
                      "gufo wan2.1-vace-14b needs width/height in multiples "
                      "of 16, 256..1280 and at most 1280x720 pixels, not " +
                          std::to_string(task.width) + "x" +
                          std::to_string(task.height));
  task.frames = static_cast<int>(graph.Num(vace, "length", 81));
  if (task.frames < 5 || task.frames > 81 || (task.frames - 1) % 4 != 0)
    throw NodeFailure(vace, "unsupported_value",
                      "gufo wan2.1-vace-14b length must be 4k+1 frames, "
                      "5..81 (16 fps)");
  if (graph.Num(vace, "batch_size", 1) != 1)
    throw NodeFailure(vace, "unsupported_value",
                      "gufo wan2.1-vace-14b renders batch_size 1");
  const double strength = graph.Num(vace, "strength", 1.0);
  if (strength < 0 || strength > 1000)
    throw NodeFailure(vace, "unsupported_value",
                      "strength must be 0..1000");
  if (strength != 1.0)
    task.vace_strength = strength;
  if (auto control = graph.LinkOf(vace, "control_video"))
    task.control_video =
        VideoFile(graph, vace, "control_video", *control, warnings);
  if (auto mask = graph.LinkOf(vace, "control_masks")) {
    Link link = *mask;
    for (int guard = 0; guard < 8 && graph.Class(link.node) == "InvertMask";
         ++guard) {
      task.vace_invert_mask = !task.vace_invert_mask;
      link = graph.RequireLink(link.node, "mask");
    }
    const std::string source = graph.Class(link.node);
    std::string channel;
    if (source == "ImageToMask") {
      channel = graph.Str(link.node, "channel", "red");
      task.vace_mask =
          VideoFile(graph, link.node, "image",
                    graph.RequireLink(link.node, "image"), warnings);
    } else if (source == "LoadImageMask") {
      channel = graph.Str(link.node, "channel", "alpha");
      task.vace_mask = ParseFileRef(graph.Str(link.node, "image"), "input");
      warnings->push_back("node " + vace +
                          ": gufo applies the still mask to every frame "
                          "(ComfyUI masks only the first frame and pads "
                          "the rest with 1)");
    } else {
      throw NodeFailure(vace, "unsupported_link",
                        "control_masks must come from ImageToMask (on a "
                        "loaded video), LoadImageMask or InvertMask, not " +
                            source);
    }
    if (channel != "red" && channel != "green" && channel != "blue")
      throw NodeFailure(link.node, "unsupported_value",
                        "gufo reads VACE masks from the red, green or blue "
                        "channel, not " +
                            channel);
    task.vace_mask_channel = channel;
  }
  if (auto reference = graph.LinkOf(vace, "reference_image"))
    task.reference_images.push_back(
        ImageFile(graph, vace, "reference_image", *reference));
  // Prompts and sampling.
  const Conditioning positive = FollowConditioning(graph, sampler, "positive");
  if (positive.encoder.empty() ||
      graph.Class(positive.encoder) != "CLIPTextEncode")
    throw NodeFailure(sampler, "unsupported_link",
                      "Wan positive conditioning must come from "
                      "CLIPTextEncode");
  task.prompt = graph.Str(positive.encoder, "text");
  task.negative_prompt =
      ClipText(graph, FollowConditioning(graph, sampler, "negative"));
  task.steps = static_cast<int>(graph.Num(sampler, "steps"));
  if (task.steps < 1 || task.steps > 50)
    throw NodeFailure(sampler, "unsupported_value",
                      "gufo wan2.1-vace-14b takes 1..50 steps");
  task.seed = Seed(graph.Num(sampler, SeedKey(graph.Class(sampler)), 0));
  const double cfg = graph.Num(sampler, "cfg", 1.0);
  if (cfg < 1.0 || cfg > 20.0)
    throw NodeFailure(sampler, "unsupported_value",
                      "gufo wan2.1-vace-14b takes cfg 1..20");
  if (cfg != 1.0)
    task.guidance = cfg;
  const std::string sampler_name = graph.Str(sampler, "sampler_name", "lcm");
  task.vace_sampler = sampler_name == "lcm" ? "lcm" : "euler";
  if (sampler_name != "lcm" && sampler_name != "euler")
    warnings->push_back("node " + sampler + ": sampler " + sampler_name +
                        " runs as euler on gufo wan2.1-vace-14b");
  const std::string scheduler = graph.Str(sampler, "scheduler", "simple");
  if (scheduler != "simple" && scheduler != "normal")
    warnings->push_back("node " + sampler + ": scheduler " + scheduler +
                        " runs as gufo's shifted linear (simple) schedule");
  if (graph.Num(sampler, "denoise", 1.0) != 1.0)
    throw NodeFailure(sampler, "unsupported_value",
                      "gufo wan2.1-vace-14b samples with denoise 1");
  const ModelChain chain = FollowModel(graph, sampler);
  if (!chain.uni3c.empty())
    throw NodeFailure(chain.uni3c, "unsupported_link",
                      "Uni3C does not combine with VACE in gufo");
  task.vace_lora = !chain.loras.empty();
  for (const auto& [name, lora_strength] : chain.loras) {
    const std::string lower = Lower(name);
    if (lower.find("lightx2v") == std::string::npos &&
        lower.find("distill") == std::string::npos &&
        lower.find("causvid") == std::string::npos)
      throw NodeFailure(sampler, "unsupported_value",
                        "gufo wan2.1-vace-14b has its step-distill LoRA "
                        "(lightx2v) built in and takes no other LoRA, not " +
                            name);
    if (lower.find("lightx2v") == std::string::npos || lora_strength != 1.0)
      warnings->push_back("node " + sampler + ": LoRA " + name +
                          " runs as gufo's built-in lightx2v T2V LoRA at "
                          "strength 1");
  }
  // ModelSamplingSD3 shift on the model path -> gufo.vace.shift.
  Link link = graph.RequireLink(sampler, "model");
  for (int guard = 0; guard < 64; ++guard) {
    const std::string cls = graph.Class(link.node);
    if (cls == "ModelSamplingSD3") {
      task.vace_shift = graph.Num(link.node, "shift", 8.0);
      if (*task.vace_shift < 1.0 || *task.vace_shift > 20.0)
        throw NodeFailure(link.node, "unsupported_value",
                          "gufo wan2.1-vace-14b takes shift 1..20");
      break;
    }
    if (cls == "UNETLoader" || cls == "CheckpointLoaderSimple")
      break;
    link = graph.RequireLink(link.node, "model");
  }
  return task;
}

/// A DynamicCombo sub-input: ComfyUI names it "<combo>.<input>" in the API
/// prompt; older exports use the bare name.
std::optional<Link> ComboLink(const Graph& graph, const std::string& id,
                              const std::string& combo,
                              const std::string& key) {
  if (auto link = graph.LinkOf(id, combo + "." + key))
    return link;
  return graph.LinkOf(id, key);
}

/// The LoadAudio file behind an AudioEncoderEncode output.
FileRef TalkAudio(const Graph& graph, const std::string& id,
                  const std::string& key, const Link& link) {
  if (graph.Class(link.node) != "AudioEncoderEncode")
    throw NodeFailure(id, "unsupported_link",
                      "input '" + key + "' must come from AudioEncoderEncode, "
                      "not " + graph.Class(link.node));
  const Link loader = graph.RequireLink(link.node, "audio_encoder");
  if (graph.Class(loader.node) != "AudioEncoderLoader")
    throw NodeFailure(link.node, "unsupported_link",
                      "audio_encoder must come from AudioEncoderLoader "
                      "(gufo runs chinese-wav2vec2-base)");
  return AudioFile(graph, link.node, "audio",
                   graph.RequireLink(link.node, "audio"));
}

/// A speaker mask: LoadImageMask (its channel) or LoadImage's MASK output
/// (1 - alpha).
std::pair<FileRef, std::string> TalkMask(const Graph& graph,
                                         const std::string& id,
                                         const std::string& key,
                                         const Link& link) {
  const std::string source = graph.Class(link.node);
  if ((source == "LoadImage" || source == "LoadImageOutput") &&
      link.slot == 1)
    return {ImageFile(graph, id, key, link), "inverted_alpha"};
  if (source == "LoadImageMask") {
    const std::string channel = graph.Str(link.node, "channel", "alpha");
    if (channel != "alpha" && channel != "red" && channel != "green" &&
        channel != "blue")
      throw NodeFailure(link.node, "value_not_in_list",
                        "channel must be alpha, red, green or blue");
    // ComfyUI's LoadImageMask inverts the alpha channel (1 - alpha), as
    // LoadImage's MASK does.
    return {ParseFileRef(graph.Str(link.node, "image"), "input"),
            channel == "alpha" ? "inverted_alpha" : channel};
  }
  throw NodeFailure(id, "unsupported_link",
                    "input '" + key +
                        "' must come from LoadImageMask or the MASK output of "
                        "LoadImage, not " +
                        source);
}

/// Wan 2.1 I2V + InfiniteTalk (wan2.1-infinitetalk-14b, 25 fps). A chain of
/// WanInfiniteTalkToVideo windows (previous_frames from the decoded frames
/// of the window before) is one gufo job: gufo windows the video itself
/// (81 frames, continued from the last motion frames).
VideoTask BuildTalk(const Graph& graph, const std::string& sampler,
                    const ModelIds& models, const Link& latent,
                    std::vector<std::string>* warnings) {
  const std::string node = latent.node;
  VideoTask task;
  task.family = VideoFamily::kTalk;
  task.model = models.talk;
  task.fps = 25;
  task.width = static_cast<int>(graph.Num(node, "width", 832));
  task.height = static_cast<int>(graph.Num(node, "height", 480));
  if (task.width % 16 != 0 || task.height % 16 != 0 || task.width < 256 ||
      task.height < 256 || task.width * task.height > 832 * 480)
    throw NodeFailure(node, "unsupported_value",
                      "gufo wan2.1-infinitetalk-14b needs width/height in "
                      "multiples of 16 (at least 256) and at most 832x480 "
                      "pixels, not " +
                          std::to_string(task.width) + "x" +
                          std::to_string(task.height));
  const auto start = graph.LinkOf(node, "start_image");
  if (!start)
    throw NodeFailure(node, "required_input_missing",
                      "gufo wan2.1-infinitetalk-14b needs a start_image");
  task.first_frame = ImageFile(graph, node, "start_image", *start);
  const Link patch = graph.RequireLink(node, "model_patch");
  if (graph.Class(patch.node) != "ModelPatchLoader")
    throw NodeFailure(node, "unsupported_link",
                      "model_patch must come from ModelPatchLoader (an "
                      "InfiniteTalk patch)");
  // Speakers.
  task.talk_audio.push_back(TalkAudio(
      graph, node, "audio_encoder_output_1",
      graph.RequireLink(node, "audio_encoder_output_1")));
  const std::string mode = graph.Str(node, "mode", "single_speaker");
  if (mode == "two_speakers") {
    const auto second =
        ComboLink(graph, node, "mode", "audio_encoder_output_2");
    const auto mask1 = ComboLink(graph, node, "mode", "mask_1");
    const auto mask2 = ComboLink(graph, node, "mode", "mask_2");
    if (second) {
      if (!mask1 || !mask2)
        throw NodeFailure(node, "required_input_missing",
                          "two speakers need mask_1 and mask_2");
      task.talk_audio.push_back(
          TalkAudio(graph, node, "audio_encoder_output_2", *second));
      for (const auto& [key, link] :
           {std::pair<const char*, Link>{"mask_1", *mask1}, {"mask_2", *mask2}}) {
        auto [file, channel] = TalkMask(graph, node, key, link);
        task.talk_masks.push_back(std::move(file));
        task.talk_mask_channels.push_back(std::move(channel));
      }
    } else if (mask1 || mask2) {
      throw NodeFailure(node, "required_input_missing",
                        "masks without audio_encoder_output_2");
    }
  } else if (mode != "single_speaker") {
    throw NodeFailure(node, "value_not_in_list",
                      "mode must be single_speaker or two_speakers");
  }
  task.audio_scale = graph.Num(node, "audio_scale", 1.0);
  if (task.audio_scale < -10 || task.audio_scale > 10)
    throw NodeFailure(node, "unsupported_value", "audio_scale must be -10..10");
  const int motion = static_cast<int>(graph.Num(node, "motion_frame_count", 9));
  task.motion_frames = std::clamp(1 + 4 * ((motion - 1 + 2) / 4), 1, 33);
  if (task.motion_frames != motion)
    warnings->push_back("node " + node + ": motion_frame_count " +
                        std::to_string(motion) + " -> gufo uses " +
                        std::to_string(task.motion_frames) + " (4k+1)");
  // The window chain: this node and every WanInfiniteTalkToVideo behind its
  // previous_frames, oldest first by depth.
  std::vector<std::string> chain{node};
  if (auto previous = graph.LinkOf(node, "previous_frames")) {
    std::set<std::string> upstream = graph.Ancestors(previous->node);
    upstream.insert(previous->node);
    for (const std::string& id : upstream)
      if (graph.Class(id) == "WanInfiniteTalkToVideo")
        chain.push_back(id);
  }
  std::sort(chain.begin(), chain.end(),
            [&](const std::string& a, const std::string& b) {
              return graph.Ancestors(a).size() < graph.Ancestors(b).size();
            });
  int frames = 0;
  for (std::size_t i = 0; i < chain.size(); ++i) {
    const std::string& id = chain[i];
    const int length = static_cast<int>(graph.Num(id, "length", 81));
    if (length < 1 || length > 1500)
      throw NodeFailure(id, "unsupported_value",
                        "length must be 1..1500 frames (25 fps)");
    const int overlap =
        i == 0 ? 0 : static_cast<int>(graph.Num(id, "motion_frame_count", 9));
    frames += length - overlap;
    if (id != node) {
      const auto audio = graph.LinkOf(id, "audio_encoder_output_1");
      if (!audio || !(TalkAudio(graph, id, "audio_encoder_output_1", *audio) ==
                      task.talk_audio.front()))
        throw NodeFailure(id, "unsupported_link",
                          "chained InfiniteTalk windows must share the same "
                          "audio (gufo runs them as one job)");
    }
  }
  task.frames = frames;
  if (task.frames < 1 || task.frames > 1500)
    throw NodeFailure(node, "unsupported_value",
                      "gufo wan2.1-infinitetalk-14b renders 1..1500 frames "
                      "(60 s at 25 fps)");
  if (chain.size() > 1)
    warnings->push_back("node " + node + ": " + std::to_string(chain.size()) +
                        " chained InfiniteTalk windows run as one gufo job "
                        "of " + std::to_string(task.frames) + " frames");
  const Conditioning positive = FollowConditioning(graph, sampler, "positive");
  if (positive.encoder.empty() ||
      graph.Class(positive.encoder) != "CLIPTextEncode")
    throw NodeFailure(sampler, "unsupported_link",
                      "InfiniteTalk positive conditioning must come from "
                      "CLIPTextEncode");
  task.prompt = graph.Str(positive.encoder, "text");
  if (!ClipText(graph, FollowConditioning(graph, sampler, "negative")).empty())
    warnings->push_back("node " + sampler +
                        ": the negative prompt is not applied (gufo "
                        "wan2.1-infinitetalk-14b runs the distilled model)");
  task.steps = static_cast<int>(graph.Num(sampler, "steps"));
  if (task.steps < 1 || task.steps > 20)
    throw NodeFailure(sampler, "unsupported_value",
                      "gufo wan2.1-infinitetalk-14b takes 1..20 steps");
  task.seed = Seed(graph.Num(sampler, SeedKey(graph.Class(sampler)), 0));
  if (graph.Num(sampler, "cfg", 1.0) != 1.0)
    warnings->push_back("node " + sampler +
                        ": cfg is not applied (gufo wan2.1-infinitetalk-14b "
                        "runs the distilled model)");
  const ModelChain model = FollowModel(graph, sampler);
  if (model.talk != node)
    throw NodeFailure(sampler, "unsupported_link",
                      "the sampler's model must come from the same "
                      "WanInfiniteTalkToVideo as its latent");
  for (const auto& [name, strength] : model.loras) {
    (void)strength;
    const std::string lower = Lower(name);
    if (lower.find("lightx2v") == std::string::npos &&
        lower.find("distill") == std::string::npos)
      throw NodeFailure(sampler, "unsupported_value",
                        "gufo wan2.1-infinitetalk-14b has its step-distill "
                        "LoRA built in and takes no other LoRA, not " +
                            name);
  }
  return task;
}

/// kijai/ComfyUI-MMAudio: MMAudioSampler with MMAudioModelLoader,
/// MMAudioFeatureUtilsLoader and optional images (VHS_LoadVideo, LoadVideo
/// via GetVideoComponents, or LoadImage).
FoleyTask BuildFoley(const Graph& graph, const std::string& id,
                     const ModelIds& models,
                     std::vector<std::string>* warnings) {
  FoleyTask task;
  task.model = models.foley;
  const Link model = graph.RequireLink(id, "mmaudio_model");
  if (graph.Class(model.node) != "MMAudioModelLoader")
    throw NodeFailure(id, "unsupported_link",
                      "mmaudio_model must come from MMAudioModelLoader");
  const std::string file = Lower(graph.Str(model.node, "mmaudio_model", ""));
  if (file.find("16k") != std::string::npos)
    throw NodeFailure(model.node, "unsupported_value",
                      "gufo serves MMAudio 44k (large_44k_v2), not 16k");
  if (!file.empty() && file.find("large_44k_v2") == std::string::npos)
    warnings->push_back("node " + model.node + ": " + file +
                        " is served by " + task.model +
                        " (large_44k_v2)");
  const Link features = graph.RequireLink(id, "feature_utils");
  if (graph.Class(features.node) != "MMAudioFeatureUtilsLoader")
    throw NodeFailure(id, "unsupported_link",
                      "feature_utils must come from MMAudioFeatureUtilsLoader");
  if (graph.Str(features.node, "mode", "44k") != "44k")
    throw NodeFailure(features.node, "unsupported_value",
                      "gufo serves MMAudio in 44k mode only");
  task.duration = graph.Num(id, "duration", 8.0);
  if (!(task.duration >= 1.0 && task.duration <= 120.0))
    throw NodeFailure(id, "unsupported_value",
                      "gufo MMAudio renders 1..120 seconds");
  task.steps = static_cast<int>(graph.Num(id, "steps", 25));
  if (task.steps < 1 || task.steps > 200)
    throw NodeFailure(id, "unsupported_value", "steps must be 1..200");
  task.cfg = graph.Num(id, "cfg", 4.5);
  task.seed = Seed(graph.Num(id, "seed", 0));
  task.prompt = graph.Str(id, "prompt", "");
  task.negative_prompt = graph.Str(id, "negative_prompt", "");
  task.mask_away_clip = graph.Bool(id, "mask_away_clip", false);
  if (auto images = graph.LinkOf(id, "images")) {
    const std::string cls = graph.Class(images->node);
    if (cls == "VHS_LoadVideo") {
      task.video = ParseFileRef(graph.Str(images->node, "video"), "input");
      task.force_rate = graph.Num(images->node, "force_rate", 0);
      task.skip_first_frames =
          static_cast<int>(graph.Num(images->node, "skip_first_frames", 0));
      task.select_every_nth =
          static_cast<int>(graph.Num(images->node, "select_every_nth", 1));
      task.frame_load_cap =
          static_cast<int>(graph.Num(images->node, "frame_load_cap", 0));
      if (graph.Num(images->node, "custom_width", 0) != 0 ||
          graph.Num(images->node, "custom_height", 0) != 0)
        warnings->push_back("node " + images->node +
                            ": custom_width / custom_height are ignored "
                            "(MMAudio resizes the frames itself)");
    } else if (cls == "GetVideoComponents") {
      const Link file = graph.RequireLink(images->node, "video");
      if (graph.Class(file.node) != "LoadVideo")
        throw NodeFailure(images->node, "unsupported_link",
                          "the video must come from LoadVideo");
      task.video = ParseFileRef(graph.Str(file.node, "file"), "input");
    } else if (cls == "LoadImage" || cls == "LoadImageOutput") {
      task.image = ImageFile(graph, id, "images", *images);
      warnings->push_back("node " + id +
                          ": one image runs as MMAudio's image mode (CLIP "
                          "only); ComfyUI-MMAudio needs a video here");
    } else {
      throw NodeFailure(id, "unsupported_link",
                        "images must come from VHS_LoadVideo, LoadVideo "
                        "(GetVideoComponents) or LoadImage, not " +
                            cls);
    }
  }
  return task;
}

OutputKind KindOf(std::string_view class_type) {
  if (class_type == "SaveAudio" || class_type == "PreviewAudio")
    return OutputKind::kAudio;
  if (class_type == "SaveVideo" || class_type == "VHS_VideoCombine")
    return OutputKind::kVideo;
  return OutputKind::kImage;
}

/// What the frame nodes between a video decode and one output ask for:
/// RIFE multiplier or target frame rate, and Real-ESRGAN upscaling.
struct Delivery {
  int multiplier{1};
  int fps_num{0};
  int fps_den{1};
  std::optional<VideoUpscale> upscale;
  bool operator==(const Delivery&) const = default;
};

/// 60 -> 60/1, 59.94 -> 60000/1001, 12.5 -> 25/2.
std::pair<int, int> FpsRational(double fps) {
  if (std::abs(fps - std::round(fps)) < 1e-6)
    return {static_cast<int>(std::lround(fps)), 1};
  const double ntsc = fps * 1.001;
  if (std::abs(ntsc - std::round(ntsc)) < 0.01)
    return {static_cast<int>(std::lround(ntsc)) * 1000, 1001};
  const int num = static_cast<int>(std::lround(fps * 1000));
  const int divisor = std::gcd(num, 1000);
  return {num / divisor, 1000 / divisor};
}

/// gufo's Real-ESRGAN checkpoints (BSD-3-Clause) by ComfyUI model file.
std::string UpscaleModelName(std::string_view file) {
  const std::string lower = Lower(file);
  for (const char* name : {"realesr-general-wdn-x4v3", "realesr-general-x4v3",
                           "realesr-animevideov3"})
    if (lower.find(name) != std::string::npos)
      return name;
  return "";
}

int RoundEven(double value) {
  return std::max(2, static_cast<int>(std::lround(value / 2.0)) * 2);
}

/// RIFE VFI, ImageUpscaleWithModel and ImageScale / ImageScaleBy between
/// the decode of `video` and output `id` (`upstream` = its ancestors).
Delivery DeliveryOf(const Graph& graph, const std::string& id,
                    const std::set<std::string>& upstream,
                    const VideoTask* video,
                    std::vector<std::string>* warnings) {
  std::vector<std::string> rife, upscalers, scalers;
  for (const std::string& node_id : upstream) {
    const std::string cls = graph.Class(node_id);
    if (cls == "RIFE VFI")
      rife.push_back(node_id);
    else if (cls == "ImageUpscaleWithModel")
      upscalers.push_back(node_id);
    else if (cls == "ImageScale" || cls == "ImageScaleBy")
      scalers.push_back(node_id);
  }
  Delivery delivery;
  if (video == nullptr) {
    if (!rife.empty())
      throw NodeFailure(rife.front(), "unsupported_link",
                        "RIFE VFI is served only after an H3 or Wan video "
                        "(or a camera move) in the same workflow");
    if (!upscalers.empty())
      throw NodeFailure(upscalers.front(), "unsupported_link",
                        "ImageUpscaleWithModel is served only after an H3, "
                        "Wan or camera video (gufo.upscale)");
    if (!scalers.empty())
      throw NodeFailure(scalers.front(), "unsupported_link",
                        "ImageScale on an output is served only after "
                        "ImageUpscaleWithModel on a video");
    return delivery;
  }
  const int base = VideoFamilyFps(video->family);
  if (!rife.empty()) {
    if (rife.size() > 1)
      throw NodeFailure(rife.back(), "unsupported_link",
                        "chained RIFE VFI nodes are not supported");
    delivery.multiplier =
        static_cast<int>(graph.Num(rife.front(), "multiplier", 2));
    if (delivery.multiplier < 1 || delivery.multiplier > 8)
      throw NodeFailure(rife.front(), "unsupported_value",
                        "gufo interpolates with a multiplier of 1..8");
    const std::string ckpt = graph.Str(rife.front(), "ckpt_name", "");
    // gufo serves RIFE 4.7-architecture weights (rife47 or rife49).
    if (!ckpt.empty() && ckpt != "rife47.pth" && ckpt != "rife49.pth")
      warnings->push_back("node " + rife.front() +
                          ": gufo runs its RIFE 4.7/4.9 weights instead of " +
                          ckpt);
    if (graph.Num(rife.front(), "scale_factor", 1.0) != 1.0)
      warnings->push_back("node " + rife.front() +
                          ": scale_factor is ignored");
  }
  // The playback rate the output asks for: VHS_VideoCombine frame_rate or
  // the CreateVideo fps in front of a SaveVideo.
  double out_fps = 0;
  std::string fps_node;
  if (graph.Class(id) == "VHS_VideoCombine") {
    out_fps = graph.Num(id, "frame_rate", 8);
    fps_node = id;
  } else if (graph.Class(id) == "SaveVideo") {
    for (const std::string& node_id : upstream)
      if (graph.Class(node_id) == "CreateVideo") {
        out_fps = graph.Num(node_id, "fps", 0);
        fps_node = node_id;
      }
  }
  const double natural = static_cast<double>(base * delivery.multiplier);
  if (out_fps > 0 && std::abs(out_fps - natural) > 1e-6) {
    if (rife.empty() || out_fps <= base) {
      // A rate at or below the generated one (e.g. CreateVideo left at 16 or
      // 24 after RIFE) is not a delivery rate: keep real time.
      warnings->push_back(
          "node " + fps_node + ": fps " + std::to_string(out_fps) +
          " ignored; gufo writes " +
          std::to_string(static_cast<int>(natural)) +
          " fps at the original speed" +
          (rife.empty() ? " (add a RIFE VFI node to resample)" : ""));
    } else {
      // With RIFE in the chain a higher playback rate is the delivery rate:
      // gufo resamples to it in real time. (ComfyUI would play the
      // interpolated frames at that rate, i.e. change the speed.)
      const auto [num, den] = FpsRational(out_fps);
      if (num < 1 || den > 1001 || num > 240 * den)
        throw NodeFailure(fps_node, "unsupported_value",
                          "gufo resamples to 1..240 fps");
      delivery.fps_num = num;
      delivery.fps_den = den;
      warnings->push_back(
          "node " + fps_node + ": " + std::to_string(out_fps) +
          " fps with RIFE x" + std::to_string(delivery.multiplier) +
          " -> gufo resamples the video to " +
          (den == 1 ? std::to_string(num)
                    : std::to_string(num) + "/" + std::to_string(den)) +
          " fps at the original speed");
    }
  }
  if (upscalers.size() > 1)
    throw NodeFailure(upscalers.back(), "unsupported_link",
                      "chained ImageUpscaleWithModel nodes are not supported");
  if (upscalers.empty()) {
    if (!scalers.empty())
      throw NodeFailure(scalers.front(), "unsupported_link",
                        "ImageScale on a video output is served only after "
                        "ImageUpscaleWithModel (gufo.upscale)");
    return delivery;
  }
  const std::string& upscaler = upscalers.front();
  const Link loader = graph.RequireLink(upscaler, "upscale_model");
  if (graph.Class(loader.node) != "UpscaleModelLoader")
    throw NodeFailure(upscaler, "unsupported_link",
                      "upscale_model must come from UpscaleModelLoader");
  const std::string file = graph.Str(loader.node, "model_name");
  VideoUpscale upscale;
  upscale.model = UpscaleModelName(file);
  if (upscale.model.empty())
    throw NodeFailure(loader.node, "unsupported_value",
                      "upscale model '" + file +
                          "' is not served by gufo; gufo upscales with the "
                          "Real-ESRGAN models realesr-animevideov3, "
                          "realesr-general-x4v3 and realesr-general-wdn-x4v3 "
                          "(BSD-3-Clause)");
  const double w4 = 4.0 * video->width;
  const double h4 = 4.0 * video->height;
  double width = w4;
  double height = h4;
  std::vector<std::string> after;
  for (const std::string& scaler : scalers) {
    if (!graph.Ancestors(scaler).contains(upscaler))
      throw NodeFailure(scaler, "unsupported_link",
                        "ImageScale before ImageUpscaleWithModel is not "
                        "supported (gufo scales the upscaled video)");
    after.push_back(scaler);
  }
  if (after.size() > 1)
    throw NodeFailure(after.back(), "unsupported_link",
                      "one ImageScale / ImageScaleBy after the upscaler");
  if (!after.empty()) {
    const std::string& scaler = after.front();
    if (graph.Class(scaler) == "ImageScale") {
      width = graph.Num(scaler, "width", 0);
      height = graph.Num(scaler, "height", 0);
      if (width <= 0 && height <= 0) {
        width = w4;
        height = h4;
      } else if (width <= 0) {
        width = height * w4 / h4;
      } else if (height <= 0) {
        height = width * h4 / w4;
      }
      upscale.fit =
          graph.Str(scaler, "crop", "disabled") == "center" ? "crop" : "stretch";
    } else {
      const double by = graph.Num(scaler, "scale_by", 1.0);
      if (!(by > 0))
        throw NodeFailure(scaler, "unsupported_value", "scale_by must be > 0");
      width = w4 * by;
      height = h4 * by;
    }
  }
  upscale.width = RoundEven(width);
  upscale.height = RoundEven(height);
  if (upscale.width > 7680 || upscale.height > 4320)
    throw NodeFailure(after.empty() ? upscaler : after.front(),
                      "unsupported_value",
                      "gufo upscales to at most 7680x4320, not " +
                          std::to_string(upscale.width) + "x" +
                          std::to_string(upscale.height));
  delivery.upscale = upscale;
  return delivery;
}

json::Value NodeError(const std::string& class_type, const std::string& type,
                      const std::string& message) {
  json::Value error = json::Value::object();
  error["type"] = type;
  error["message"] = message;
  error["details"] = "";
  error["extra_info"] = json::Value::object();
  json::Value errors = json::Value::array();
  errors.push_back(std::move(error));
  json::Value entry = json::Value::object();
  entry["errors"] = std::move(errors);
  entry["dependent_outputs"] = json::Value::array();
  entry["class_type"] = class_type;
  return entry;
}

}  // namespace

const std::vector<std::string>& SupportedClassTypes() {
  static const std::vector<std::string> kAll = [] {
    std::vector<std::string> all;
    for (auto name : kPlumbing)
      all.emplace_back(name);
    for (auto name : kGenerators)
      all.emplace_back(name);
    for (auto name : kOutputs)
      all.emplace_back(name);
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    return all;
  }();
  return kAll;
}

bool IsOutputClass(std::string_view class_type) {
  return Contains(kOutputs, class_type);
}

int VideoFamilyFps(VideoFamily family) {
  return family == VideoFamily::kCamera || family == VideoFamily::kVace ? 16
         : family == VideoFamily::kTalk                                   ? 25
                                                                          : 24;
}

json::Value VideoRequestBody(const VideoTask& task) {
  json::Value body = json::Value::object();
  body["model"] = task.model;
  json::Value gufo = json::Value::object();
  if (task.family != VideoFamily::kProcess) {
    body["prompt"] = task.prompt;
    gufo["seed"] = task.seed;
    if (task.steps > 0)
      gufo["steps"] = task.steps;
  }
  if (task.family == VideoFamily::kProcess) {
    // Only gufo.interpolate / gufo.upscale below; the executor adds
    // source_video.
  } else if (task.family == VideoFamily::kH3) {
    body["size"] =
        std::to_string(task.width) + "x" + std::to_string(task.height);
    // gufo aligns ceil(seconds * 24) up to 17k + 5 and wants gufo.frames to
    // match: seconds 1 = 22 frames, otherwise up to 14.375 s. (frames - 8) / 24
    // lands inside the 17-frame step below `frames` without rounding trouble.
    double seconds = 1.0;
    if (task.frames > 22)
      seconds = std::round((task.frames - 8) / 24.0 * 1000.0) / 1000.0;
    body["seconds"] = seconds;
    gufo["frames"] = task.frames;
    if (!task.lora.empty())
      gufo["lora"] = task.lora;
    if (task.lora_strength)
      gufo["lora_strength"] = *task.lora_strength;
    if (task.audio_reference)
      gufo["audio_reference_frame"] = task.audio_reference_frame;
  } else if (task.family == VideoFamily::kCamera) {
    // 16 fps, 4k+1 frames: gufo maps seconds back to 4 * round(4 s) + 1.
    body["size"] =
        std::to_string(task.width) + "x" + std::to_string(task.height);
    body["seconds"] = (task.frames - 1) / 16.0;
    gufo["frames"] = task.frames;
    if (task.camera) {
      json::Value camera = json::Value::object();
      camera["type"] = task.camera->type;
      if (task.camera->amount)
        camera["amount"] = *task.camera->amount;
      camera["speed"] = task.camera->speed;
      camera["ease"] = task.camera->ease;
      gufo["camera"] = std::move(camera);
    }
  } else if (task.family == VideoFamily::kVace) {
    // 16 fps like the camera family. The executor adds control_video, mask
    // and reference_images to gufo.vace.
    body["size"] =
        std::to_string(task.width) + "x" + std::to_string(task.height);
    body["seconds"] = (task.frames - 1) / 16.0;
    gufo["frames"] = task.frames;
    if (task.guidance)
      gufo["guidance"] = *task.guidance;
    if (!task.negative_prompt.empty())
      gufo["negative_prompt"] = task.negative_prompt;
    json::Value vace = json::Value::object();
    if (task.vace_strength)
      vace["strength"] = *task.vace_strength;
    if (task.vace_mask) {
      vace["mask_channel"] = task.vace_mask_channel;
      if (task.vace_invert_mask)
        vace["invert_mask"] = true;
    }
    if (!task.vace_sampler.empty())
      vace["sampler"] = task.vace_sampler;
    if (task.vace_shift)
      vace["shift"] = *task.vace_shift;
    if (!task.vace_lora)
      vace["lora"] = false;
    gufo["vace"] = std::move(vace);
  } else if (task.family == VideoFamily::kTalk) {
    // 25 fps: gufo maps seconds back to round(25 s) frames.
    body["size"] =
        std::to_string(task.width) + "x" + std::to_string(task.height);
    body["seconds"] = task.frames / 25.0;
    json::Value talk = json::Value::object();
    talk["audio_scale"] = task.audio_scale;
    talk["motion_frames"] = task.motion_frames;
    talk["audio_mode"] = "add";
    gufo["talk"] = std::move(talk);
  } else {
    body["size"] =
        std::to_string(task.width) + "x" + std::to_string(task.height);
    body["seconds"] = (task.frames - 1) / 24.0;
    if (task.guidance)
      gufo["guidance"] = *task.guidance;
    if (!task.negative_prompt.empty())
      gufo["negative_prompt"] = task.negative_prompt;
  }
  if (task.fps_num > 0) {
    json::Value interpolate = json::Value::object();
    if (task.fps_den == 1)
      interpolate["fps"] = task.fps_num;
    else
      interpolate["fps"] = std::to_string(task.fps_num) + "/" +
                           std::to_string(task.fps_den);
    gufo["interpolate"] = std::move(interpolate);
  } else if (task.interpolate > 1) {
    gufo["interpolate"] = task.interpolate;
  }
  if (task.upscale) {
    json::Value upscale = json::Value::object();
    upscale["width"] = task.upscale->width;
    upscale["height"] = task.upscale->height;
    upscale["fit"] = task.upscale->fit;
    upscale["model"] = task.upscale->model;
    gufo["upscale"] = std::move(upscale);
  }
  body["gufo"] = std::move(gufo);
  return body;
}

json::Value MusicRequestBody(const MusicTask& task) {
  json::Value body = json::Value::object();
  body["model"] = task.model;
  body["prompt"] = task.tags;
  if (!task.lyrics.empty())
    body["lyrics"] = task.lyrics;
  body["duration"] = task.duration;
  body["seed"] = task.seed;
  body["steps"] = task.steps;
  if (task.bpm > 0)
    body["bpm"] = task.bpm;
  if (!task.timesignature.empty())
    body["timesignature"] = task.timesignature;
  if (!task.language.empty())
    body["language"] = task.language;
  if (!task.keyscale.empty())
    body["keyscale"] = task.keyscale;
  if (task.audio_codes)
    body["audio_codes"] = *task.audio_codes;
  if (task.lm_cfg)
    body["lm_cfg"] = *task.lm_cfg;
  if (task.lm_temperature)
    body["lm_temperature"] = *task.lm_temperature;
  if (task.lm_top_p)
    body["lm_top_p"] = *task.lm_top_p;
  body["response_format"] = "wav";
  return body;
}

json::Value FoleyRequestBody(const FoleyTask& task) {
  json::Value body = json::Value::object();
  body["model"] = task.model;
  body["prompt"] = task.prompt;
  body["negative_prompt"] = task.negative_prompt;
  body["duration"] = task.duration;
  body["steps"] = task.steps;
  body["cfg_strength"] = task.cfg;
  body["seed"] = task.seed;
  body["mask_away_clip"] = task.mask_away_clip;
  if (task.video) {
    // ComfyUI-MMAudio feeds the IMAGE batch's first frames, not 8 / 25 fps.
    body["frame_mode"] = "consecutive";
    if (task.force_rate > 0)
      body["force_rate"] = task.force_rate;
    if (task.skip_first_frames > 0)
      body["skip_first_frames"] = task.skip_first_frames;
    if (task.select_every_nth > 1)
      body["select_every_nth"] = task.select_every_nth;
    if (task.frame_load_cap > 0)
      body["frame_load_cap"] = task.frame_load_cap;
  }
  body["response_format"] = "wav";
  return body;
}

json::Value ImageExtension(const ImageTask& task) {
  json::Value gufo = json::Value::object();
  if (!task.negative_prompt.empty())
    gufo["negative_prompt"] = task.negative_prompt;
  if (task.guidance)
    gufo["guidance"] = *task.guidance;
  if (task.strength)
    gufo["strength"] = *task.strength;
  if (!task.mask.name.empty())
    gufo["mask_channel"] = task.mask_channel;
  return gufo;
}

json::Value ImageLoras(const ImageTask& task) {
  if (task.loras.empty())
    return json::Value();
  json::Value loras = json::Value::array();
  for (const auto& [name, scale] : task.loras) {
    json::Value entry = json::Value::object();
    entry["name"] = name;
    entry["scale"] = scale;
    loras.push_back(std::move(entry));
  }
  return loras;
}

namespace {

std::uint32_t Le32(std::string_view bytes, std::size_t at) {
  return static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[at])) |
         static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[at + 1]))
             << 8 |
         static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[at + 2]))
             << 16 |
         static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[at + 3]))
             << 24;
}

void PutLe(std::string* out, std::uint32_t value, int bytes) {
  for (int i = 0; i < bytes; ++i)
    out->push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
}

}  // namespace

std::optional<Wav> ParseWav(std::string_view bytes) {
  if (bytes.size() < 12 || bytes.substr(0, 4) != "RIFF" ||
      bytes.substr(8, 4) != "WAVE")
    return std::nullopt;
  Wav wav;
  bool format = false;
  std::size_t at = 12;
  while (at + 8 <= bytes.size()) {
    const std::string_view id = bytes.substr(at, 4);
    std::size_t size = Le32(bytes, at + 4);
    const std::size_t body = at + 8;
    if (id == "data" && size > bytes.size() - body)
      size = bytes.size() - body;  // streamed WAVs may carry a 0/-1 size
    if (size > bytes.size() - body)
      return std::nullopt;
    if (id == "fmt ") {
      if (size < 16)
        return std::nullopt;
      const unsigned audio_format = static_cast<unsigned char>(bytes[body]) |
                                    static_cast<unsigned char>(bytes[body + 1])
                                        << 8;
      wav.channels = static_cast<unsigned char>(bytes[body + 2]) |
                     static_cast<unsigned char>(bytes[body + 3]) << 8;
      wav.sample_rate = static_cast<int>(Le32(bytes, body + 4));
      const unsigned bits = static_cast<unsigned char>(bytes[body + 14]) |
                            static_cast<unsigned char>(bytes[body + 15]) << 8;
      if (audio_format != 1 || bits != 16 || wav.channels < 1)
        return std::nullopt;
      format = true;
    } else if (id == "data") {
      if (!format)
        return std::nullopt;
      wav.pcm = std::string(bytes.substr(body, size));
      return wav;
    }
    at = body + size + (size & 1);
  }
  return std::nullopt;
}

std::string BuildWav(const Wav& wav) {
  std::string out = "RIFF";
  PutLe(&out, static_cast<std::uint32_t>(36 + wav.pcm.size()), 4);
  out += "WAVEfmt ";
  PutLe(&out, 16, 4);
  PutLe(&out, 1, 2);
  PutLe(&out, static_cast<std::uint32_t>(wav.channels), 2);
  PutLe(&out, static_cast<std::uint32_t>(wav.sample_rate), 4);
  PutLe(&out, static_cast<std::uint32_t>(wav.sample_rate * wav.channels * 2),
        4);
  PutLe(&out, static_cast<std::uint32_t>(wav.channels * 2), 2);
  PutLe(&out, 16, 2);
  out += "data";
  PutLe(&out, static_cast<std::uint32_t>(wav.pcm.size()), 4);
  out += wav.pcm;
  return out;
}

std::optional<std::string> JoinWavs(const std::vector<std::string>& parts,
                                    double pause_seconds) {
  if (parts.empty())
    return std::nullopt;
  Wav joined;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    auto wav = ParseWav(parts[i]);
    if (!wav)
      return std::nullopt;
    if (i == 0) {
      joined.sample_rate = wav->sample_rate;
      joined.channels = wav->channels;
    } else {
      if (wav->sample_rate != joined.sample_rate ||
          wav->channels != joined.channels)
        return std::nullopt;
      const auto frames = static_cast<std::size_t>(
          std::lround(std::max(0.0, pause_seconds) * joined.sample_rate));
      joined.pcm.append(frames * 2 * joined.channels, '\0');
    }
    joined.pcm += wav->pcm;
  }
  return BuildWav(joined);
}

namespace {

/// The LoadVideo node whose frames reach output `id` through
/// GetVideoComponents and a RIFE VFI or ImageUpscaleWithModel, with no
/// generator involved; nullopt when the output is not such a chain.
std::optional<std::string> ProcessSource(const Graph& graph,
                                         const std::string& id,
                                         const std::set<std::string>& upstream) {
  std::vector<std::string> loaders;
  bool processed = false;
  for (const std::string& node_id : upstream) {
    const std::string cls = graph.Class(node_id);
    if (cls == "LoadVideo")
      loaders.push_back(node_id);
    else if (cls == "RIFE VFI" || cls == "ImageUpscaleWithModel")
      processed = true;
  }
  if (loaders.empty() || !processed)
    return std::nullopt;
  if (loaders.size() > 1)
    throw NodeFailure(id, "unsupported_link",
                      "output depends on more than one LoadVideo");
  for (const std::string& node_id : upstream) {
    if (graph.Class(node_id) != "GetVideoComponents")
      continue;
    if (graph.RequireLink(node_id, "video").node != loaders.front())
      throw NodeFailure(node_id, "unsupported_link",
                        "the video must come from LoadVideo");
  }
  return loaders.front();
}

}  // namespace

bool BuildPlan(const json::Value& original, const ModelIds& models, Plan* plan,
               PlanError* error) {
  *plan = Plan{};
  if (!original.is_object() || original.empty()) {
    error->type = "invalid_prompt";
    error->message = "prompt must be a non-empty object of nodes";
    return false;
  }
  // 0. SamplerCustomAdvanced graphs become KSampler graphs.
  json::Value lowered;
  try {
    lowered = LowerCustomSamplers(original, &plan->warnings);
  } catch (const NodeFailure& failure) {
    const std::string cls = Graph(original).Class(failure.node_id);
    error->type = "prompt_outputs_failed_validation";
    error->message = "node " + failure.node_id + " (" + cls + "): " +
                     failure.what();
    error->details = error->message;
    error->node_errors[failure.node_id] =
        NodeError(cls, failure.error_type, failure.what());
    plan->warnings.clear();
    return false;
  }
  const json::Value& prompt = lowered.is_null() ? original : lowered;
  Graph graph(prompt);
  // 1. Every node must be a class the adapter knows.
  std::vector<std::string> unsupported;
  std::vector<std::string> pending;
  for (const auto& [id, node] : prompt.members()) {
    const json::Value* inputs = node.find("inputs");
    const std::string cls = node.member_str("class_type");
    if (!node.is_object() || cls.empty() ||
        (inputs != nullptr && !inputs->is_object())) {
      error->type = "invalid_prompt";
      error->message = "node " + id + " needs class_type and inputs";
      error->details = "Node ID '#" + id + "'";
      return false;
    }
    const auto& all = SupportedClassTypes();
    if (std::binary_search(all.begin(), all.end(), cls))
      continue;
    const bool is_pending = Contains(kPending, cls);
    (is_pending ? pending : unsupported).push_back(cls + " (node " + id + ")");
    error->node_errors[id] = NodeError(
        cls, "unsupported_node",
        is_pending ? cls + " is not available in gufo yet (port pending)"
                   : cls + " is not supported by the gufo ComfyUI adapter");
  }
  if (!unsupported.empty() || !pending.empty()) {
    error->type = "invalid_prompt";
    std::string message = "gufo ComfyUI adapter: ";
    const auto join = [](const std::vector<std::string>& list) {
      std::string out;
      for (const auto& item : list)
        out += (out.empty() ? "" : ", ") + item;
      return out;
    };
    if (!unsupported.empty())
      message += "unsupported node types: " + join(unsupported);
    if (!pending.empty())
      message += std::string(unsupported.empty() ? "" : "; ") +
                 "not yet available in gufo (port pending): " + join(pending);
    error->message = message;
    error->details = message;
    return false;
  }
  // 2. Each output node is fed by exactly one generator.
  std::map<std::string, std::size_t> task_of;
  std::map<std::size_t, std::optional<Delivery>> delivery_of;
  bool any_output = false;
  try {
    for (const auto& [id, node] : prompt.members()) {
      const std::string cls = node.member_str("class_type");
      if (!IsOutputClass(cls))
        continue;
      any_output = true;
      const std::set<std::string> upstream = graph.Ancestors(id);
      std::vector<std::string> generators;
      bool tts = false;
      for (const std::string& node_id : upstream) {
        const std::string up = graph.Class(node_id);
        if (IsSampler(up) || IsTts(up) || IsFoley(up)) {
          generators.push_back(node_id);
          tts = tts || IsTts(up);
        }
      }
      // Without a generator: a loaded video that is only interpolated /
      // upscaled (FilmStudio: LoadVideo -> RIFE VFI -> SaveVideo per shot).
      std::optional<std::string> process;
      if (generators.empty())
        process = ProcessSource(graph, id, upstream);
      if (generators.empty() && !process)
        throw NodeFailure(id, "unsupported_link",
                          "output has no generating node (KSampler or "
                          "Qwen3-TTS) upstream");
      if (process)
        generators.push_back(*process);
      if (generators.size() > 1) {
        // Chained InfiniteTalk windows: the last sampler (the one all others
        // feed) stands for the whole chain.
        const auto is_talk = [&](const std::string& g) {
          const auto latent = graph.LinkOf(g, "latent_image");
          return IsSampler(graph.Class(g)) && latent &&
                 graph.Class(latent->node) == "WanInfiniteTalkToVideo";
        };
        std::string last;
        if (std::all_of(generators.begin(), generators.end(), is_talk)) {
          for (const std::string& g : generators) {
            const std::set<std::string> before = graph.Ancestors(g);
            if (std::all_of(generators.begin(), generators.end(),
                            [&](const std::string& other) {
                              return other == g || before.contains(other);
                            }))
              last = g;
          }
        }
        if (last.empty())
          throw NodeFailure(id, "unsupported_link",
                            "output depends on more than one generating "
                            "node; chained generators are not supported");
        generators = {last};
      }
      const std::string& generator = generators.front();
      OutputTarget target;
      target.node_id = id;
      target.class_type = cls;
      target.kind = KindOf(cls);
      target.temp = cls.starts_with("Preview");
      target.prefix = target.temp ? "ComfyUI_temp"
                                  : graph.Str(id, "filename_prefix", "ComfyUI");
      auto found = task_of.find(generator);
      if (found == task_of.end()) {
        Task task;
        task.node_id = generator;
        task.class_type = graph.Class(generator);
        if (process) {
          VideoTask video;
          video.family = VideoFamily::kProcess;
          video.model =
              models.available.empty() ||
                      std::find(models.available.begin(),
                                models.available.end(),
                                models.process) != models.available.end()
                  ? models.process
                  : models.h3;
          video.source_video =
              ParseFileRef(graph.Str(generator, "file"), "input");
          task.spec = std::move(video);
        } else if (IsFoley(task.class_type)) {
          task.spec = BuildFoley(graph, generator, models, &plan->warnings);
        } else if (tts &&
                   task.class_type == "FB_Qwen3TTSDialogueInference") {
          task.spec = BuildDialogue(graph, generator, models, &plan->warnings);
        } else if (tts) {
          task.spec = BuildSpeech(graph, generator, models, &plan->warnings);
        } else {
          const Link latent = graph.RequireLink(generator, "latent_image");
          const std::string latent_class = graph.Class(latent.node);
          if (latent_class == "MiniMaxH3ImageToVideo")
            task.spec = BuildH3(graph, generator, models, latent);
          else if (latent_class == "Wan22ImageToVideoLatent")
            task.spec = BuildWan(graph, generator, models, latent);
          else if (latent_class == "WanImageToVideo" ||
                   latent_class == "WanCameraImageToVideo")
            task.spec = BuildCamera(graph, generator, models, latent,
                                    &plan->warnings);
          else if (latent_class == "WanVaceToVideo")
            task.spec = BuildVace(graph, generator, models, latent,
                                  &plan->warnings);
          else if (latent_class == "WanInfiniteTalkToVideo")
            task.spec = BuildTalk(graph, generator, models, latent,
                                  &plan->warnings);
          else if (latent_class == "EmptyAceStep1.5LatentAudio")
            task.spec = BuildMusic(graph, generator, models, latent,
                                   &plan->warnings);
          else
            task.spec =
                BuildImage(graph, generator, models, latent, &plan->warnings);
        }
        found = task_of.emplace(generator, plan->tasks.size()).first;
        plan->tasks.push_back(std::move(task));
      }
      Task& task = plan->tasks[found->second];
      const bool speech = std::holds_alternative<SpeechTask>(task.spec) ||
                          std::holds_alternative<DialogueTask>(task.spec) ||
                          std::holds_alternative<MusicTask>(task.spec);
      const bool image = std::holds_alternative<ImageTask>(task.spec);
      if (const auto* foley = std::get_if<FoleyTask>(&task.spec)) {
        // Audio files, or the loaded video with the new sound track.
        if (target.kind == OutputKind::kImage)
          throw NodeFailure(id, "unsupported_link",
                            cls + " cannot store the output of " +
                                task.class_type);
        if (target.kind == OutputKind::kVideo && !foley->video)
          throw NodeFailure(id, "unsupported_link",
                            "a video output of MMAudioSampler needs the "
                            "sampler's images from a loaded video");
      }
      if ((speech && target.kind != OutputKind::kAudio) ||
          (image && target.kind != OutputKind::kImage))
        throw NodeFailure(
            id, "unsupported_link",
            cls + " cannot store the output of " + task.class_type);
      // RIFE VFI / upscale nodes between the decode and this output: gufo
      // interpolates and upscales the whole video job (gufo.interpolate,
      // gufo.upscale). The audio track is neither, so audio outputs do not
      // count.
      auto* video = std::get_if<VideoTask>(&task.spec);
      if (video == nullptr || target.kind != OutputKind::kAudio) {
        const Delivery delivery =
            DeliveryOf(graph, id, upstream, video, &plan->warnings);
        if (video != nullptr) {
          auto& seen = delivery_of[found->second];
          if (seen && !(*seen == delivery))
            throw NodeFailure(id, "unsupported_link",
                              "outputs of one video need the same RIFE "
                              "interpolation and upscale (gufo interpolates "
                              "and upscales the whole job)");
          seen = delivery;
          video->interpolate = delivery.multiplier;
          video->fps_num = delivery.fps_num;
          video->fps_den = delivery.fps_den;
          video->upscale = delivery.upscale;
          const int base = VideoFamilyFps(video->family);
          video->fps = delivery.fps_num > 0
                           ? static_cast<int>(std::lround(
                                 static_cast<double>(delivery.fps_num) /
                                 delivery.fps_den))
                           : base * delivery.multiplier;
        }
      }
      if (cls == "VHS_VideoCombine") {
        target.temp = !graph.Bool(id, "save_output", true);
        target.prefix = graph.Str(id, "filename_prefix", "AnimateDiff");
        const std::string format = graph.Str(id, "format", "video/h264-mp4");
        if (format.find("mp4") == std::string::npos)
          plan->warnings.push_back("node " + id + ": format " + format +
                                   " -> gufo writes an H.264 MP4");
      }      task.outputs.push_back(std::move(target));
    }
  } catch (const NodeFailure& failure) {
    error->type = "prompt_outputs_failed_validation";
    error->message = "node " + failure.node_id + " (" +
                     graph.Class(failure.node_id) + "): " + failure.what();
    error->details = error->message;
    error->node_errors[failure.node_id] = NodeError(
        graph.Class(failure.node_id), failure.error_type, failure.what());
    plan->tasks.clear();
    return false;
  }
  if (!any_output) {
    error->type = "prompt_no_outputs";
    error->message = "Prompt has no outputs";
    error->details = "";
    return false;
  }
  return true;
}

FileRef ParseFileRef(std::string_view value, std::string_view default_folder) {
  FileRef ref;
  ref.folder = std::string(default_folder);
  std::string_view name = value;
  while (!name.empty() && name.back() == ' ')
    name.remove_suffix(1);
  for (std::string_view folder : {"input", "output", "temp"}) {
    const std::string tag = " [" + std::string(folder) + "]";
    if (name.ends_with(tag)) {
      name.remove_suffix(tag.size());
      ref.folder = std::string(folder);
      break;
    }
  }
  ref.name = std::string(name);
  for (char& c : ref.name)
    if (c == '\\')
      c = '/';
  return ref;
}

std::optional<std::string> SafeRelative(std::string_view path) {
  std::string out;
  std::size_t start = 0;
  while (start <= path.size()) {
    std::size_t end = path.find_first_of("/\\", start);
    if (end == std::string_view::npos)
      end = path.size();
    const std::string_view part = path.substr(start, end - start);
    if (part == "..")
      return std::nullopt;
    if (!part.empty() && part != ".") {
      if (part.find(':') != std::string_view::npos ||
          part.find('\0') != std::string_view::npos)
        return std::nullopt;
      if (!out.empty())
        out += '/';
      out += part;
    } else if (part.empty() && start == 0 && end < path.size()) {
      return std::nullopt;  // leading separator: absolute
    }
    start = end + 1;
  }
  return out;
}

std::pair<int, int> ScaleToArea(int width, int height, int resolution) {
  if (width <= 0 || height <= 0)
    return {resolution, resolution};
  if (resolution <= 0) {
    const auto round32 = [](int v) {
      return std::max(32, static_cast<int>(std::lround(v / 32.0)) * 32);
    };
    return {round32(width), round32(height)};
  }
  const double scale = std::sqrt(static_cast<double>(resolution) * resolution /
                                 (static_cast<double>(width) * height));
  const int w =
      std::max(32, static_cast<int>(std::lround(width * scale / 32.0)) * 32);
  const int h =
      std::max(32, static_cast<int>(std::lround(height * scale / 32.0)) * 32);
  return {w, h};
}

std::optional<std::pair<int, int>> ImageDimensions(std::string_view bytes) {
  const auto u8 = [&](std::size_t i) {
    return static_cast<unsigned>(static_cast<unsigned char>(bytes[i]));
  };
  if (bytes.size() >= 24 && bytes.substr(0, 8) == "\x89PNG\r\n\x1a\n" &&
      bytes.substr(12, 4) == "IHDR") {
    const auto be32 = [&](std::size_t i) {
      return (u8(i) << 24) | (u8(i + 1) << 16) | (u8(i + 2) << 8) | u8(i + 3);
    };
    return std::pair<int, int>{static_cast<int>(be32(16)),
                               static_cast<int>(be32(20))};
  }
  if (bytes.size() >= 4 && u8(0) == 0xFF && u8(1) == 0xD8) {
    std::size_t i = 2;
    while (i + 8 < bytes.size()) {
      if (u8(i) != 0xFF) {
        ++i;
        continue;
      }
      const unsigned marker = u8(i + 1);
      if (marker == 0xFF) {
        ++i;
        continue;
      }
      if (marker == 0xD8 || marker == 0x01 ||
          (marker >= 0xD0 && marker <= 0xD7)) {
        i += 2;
        continue;
      }
      const std::size_t length = (u8(i + 2) << 8) | u8(i + 3);
      const bool sof = marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 &&
                       marker != 0xC8 && marker != 0xCC;
      if (sof) {
        const int height = static_cast<int>((u8(i + 5) << 8) | u8(i + 6));
        const int width = static_cast<int>((u8(i + 7) << 8) | u8(i + 8));
        return std::pair<int, int>{width, height};
      }
      i += 2 + length;
    }
  }
  return std::nullopt;
}

int SnapH3Frames(int length) {
  if (length <= 5)
    return 5;
  return 5 + ((length - 5 + 16) / 17) * 17;
}

std::string Base64Encode(std::string_view bytes) {
  static constexpr char kAlphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((bytes.size() + 2) / 3 * 4);
  for (std::size_t i = 0; i < bytes.size(); i += 3) {
    const std::uint32_t a = static_cast<unsigned char>(bytes[i]);
    const std::uint32_t b =
        i + 1 < bytes.size() ? static_cast<unsigned char>(bytes[i + 1]) : 0;
    const std::uint32_t c =
        i + 2 < bytes.size() ? static_cast<unsigned char>(bytes[i + 2]) : 0;
    const std::uint32_t word = (a << 16) | (b << 8) | c;
    out += kAlphabet[(word >> 18) & 63];
    out += kAlphabet[(word >> 12) & 63];
    out += i + 1 < bytes.size() ? kAlphabet[(word >> 6) & 63] : '=';
    out += i + 2 < bytes.size() ? kAlphabet[word & 63] : '=';
  }
  return out;
}

std::optional<std::string> Base64Decode(std::string_view text) {
  std::string out;
  out.reserve(text.size() / 4 * 3);
  std::uint32_t word = 0;
  int bits = 0;
  for (const char c : text) {
    int value = 0;
    if (c >= 'A' && c <= 'Z')
      value = c - 'A';
    else if (c >= 'a' && c <= 'z')
      value = c - 'a' + 26;
    else if (c >= '0' && c <= '9')
      value = c - '0' + 52;
    else if (c == '+' || c == '-')
      value = 62;
    else if (c == '/' || c == '_')
      value = 63;
    else if (c == '=' || c == '\r' || c == '\n' || c == ' ')
      continue;
    else
      return std::nullopt;
    word = (word << 6) | static_cast<std::uint32_t>(value);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out += static_cast<char>((word >> bits) & 0xFF);
    }
  }
  return out;
}

namespace {

/// Value of `key` in a header like `form-data; name="x"; filename="y"`.
std::string HeaderParam(std::string_view header, std::string_view key) {
  std::size_t pos = 0;
  while (pos < header.size()) {
    std::size_t semi = header.find(';', pos);
    if (semi == std::string_view::npos)
      semi = header.size();
    std::string_view part = header.substr(pos, semi - pos);
    while (!part.empty() && part.front() == ' ')
      part.remove_prefix(1);
    if (part.size() > key.size() && part[key.size()] == '=' &&
        Lower(part.substr(0, key.size())) == Lower(key)) {
      std::string_view value = part.substr(key.size() + 1);
      if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
        value = value.substr(1, value.size() - 2);
      return std::string(value);
    }
    pos = semi + 1;
  }
  return "";
}

}  // namespace

bool ParseMultipart(std::string_view content_type, std::string_view body,
                    std::vector<MultipartPart>* parts, std::string* error) {
  parts->clear();
  if (!Lower(content_type).starts_with("multipart/form-data")) {
    *error = "expected multipart/form-data";
    return false;
  }
  const std::string boundary = HeaderParam(content_type, "boundary");
  if (boundary.empty()) {
    *error = "multipart boundary missing";
    return false;
  }
  const std::string delimiter = "--" + boundary;
  std::size_t pos = body.find(delimiter);
  if (pos == std::string_view::npos) {
    *error = "multipart body has no boundary";
    return false;
  }
  for (;;) {
    pos += delimiter.size();
    if (body.substr(pos, 2) == "--")
      return true;
    if (body.substr(pos, 2) != "\r\n") {
      *error = "malformed multipart boundary line";
      return false;
    }
    pos += 2;
    const std::size_t head_end = body.find("\r\n\r\n", pos);
    if (head_end == std::string_view::npos) {
      *error = "multipart part without headers";
      return false;
    }
    MultipartPart part;
    std::string_view headers = body.substr(pos, head_end - pos);
    while (!headers.empty()) {
      std::size_t eol = headers.find("\r\n");
      const std::string_view line = headers.substr(
          0, eol == std::string_view::npos ? headers.size() : eol);
      const std::size_t colon = line.find(':');
      if (colon != std::string_view::npos) {
        const std::string name = Lower(line.substr(0, colon));
        std::string_view value = line.substr(colon + 1);
        while (!value.empty() && value.front() == ' ')
          value.remove_prefix(1);
        if (name == "content-disposition") {
          part.name = HeaderParam(value, "name");
          part.filename = HeaderParam(value, "filename");
        } else if (name == "content-type") {
          part.content_type = std::string(value);
        }
      }
      if (eol == std::string_view::npos)
        break;
      headers.remove_prefix(eol + 2);
    }
    const std::size_t data_start = head_end + 4;
    const std::size_t next = body.find("\r\n" + delimiter, data_start);
    if (next == std::string_view::npos) {
      *error = "multipart part is not terminated";
      return false;
    }
    part.data = std::string(body.substr(data_start, next - data_start));
    parts->push_back(std::move(part));
    pos = next + 2;
  }
}

std::string BuildMultipart(const std::vector<MultipartPart>& parts,
                           std::string_view boundary) {
  std::string out;
  for (const MultipartPart& part : parts) {
    out += "--";
    out += boundary;
    out += "\r\nContent-Disposition: form-data; name=\"" + part.name + "\"";
    if (!part.filename.empty())
      out += "; filename=\"" + part.filename + "\"";
    out += "\r\n";
    if (!part.content_type.empty())
      out += "Content-Type: " + part.content_type + "\r\n";
    out += "\r\n";
    out += part.data;
    out += "\r\n";
  }
  out += "--";
  out += boundary;
  out += "--\r\n";
  return out;
}

std::optional<SaveName> SplitPrefix(std::string_view prefix) {
  auto safe = SafeRelative(prefix);
  if (!safe || safe->empty())
    return std::nullopt;
  SaveName name;
  const std::size_t slash = safe->rfind('/');
  if (slash == std::string::npos) {
    name.base = *safe;
  } else {
    name.subfolder = safe->substr(0, slash);
    name.base = safe->substr(slash + 1);
  }
  return name;
}

int NextCounter(const std::vector<std::string>& existing,
                std::string_view base) {
  int highest = 0;
  const std::string lead = std::string(base) + "_";
  for (const std::string& name : existing) {
    if (!name.starts_with(lead))
      continue;
    std::size_t i = lead.size();
    int value = 0;
    std::size_t digits = 0;
    while (i < name.size() &&
           std::isdigit(static_cast<unsigned char>(name[i])) && digits < 9) {
      value = value * 10 + (name[i] - '0');
      ++i;
      ++digits;
    }
    if (digits == 5)
      highest = std::max(highest, value);
  }
  return highest + 1;
}

std::string OutputFilename(std::string_view base, int counter, OutputKind kind,
                           std::string_view extension) {
  char digits[16];
  std::snprintf(digits, sizeof(digits), "%05d", counter);
  std::string name = std::string(base) + "_" + digits;
  if (kind != OutputKind::kAudio)
    name += "_";
  name += ".";
  name += extension;
  return name;
}

}  // namespace gufo::router::comfy
