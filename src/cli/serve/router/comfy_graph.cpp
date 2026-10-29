#include "src/cli/serve/router/comfy_graph.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
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
};

constexpr std::string_view kGenerators[] = {
    "KSampler",
    "KSamplerAdvanced",
    "MiniMaxH3ImageToVideo",
    "Wan22ImageToVideoLatent",
    "FB_Qwen3TTSVoiceDesign",
    "FB_Qwen3TTSVoiceClone",
    "FB_Qwen3TTSCustomVoice",
};

constexpr std::string_view kOutputs[] = {
    "SaveImage", "PreviewImage", "SaveAudio", "PreviewAudio", "SaveVideo"};

// Node families other gufo ports are bringing; named so the error says so.
constexpr std::string_view kPending[] = {
    "RIFE VFI", "TextEncodeAceStepAudio1.5", "EmptyAceStep1.5LatentAudio",
    "TextEncodeAceStepAudio", "EmptyAceStepLatentAudio"};

template<std::size_t N>
bool Contains(const std::string_view (&set)[N], std::string_view value) {
  return std::find(std::begin(set), std::end(set), value) != std::end(set);
}

bool IsTts(std::string_view class_type) {
  return class_type.starts_with("FB_Qwen3TTS");
}

bool IsSampler(std::string_view class_type) {
  return class_type == "KSampler" || class_type == "KSamplerAdvanced";
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
  std::vector<std::pair<std::string, double>> loras;
};

ModelChain FollowModel(const Graph& graph, const std::string& sampler) {
  ModelChain chain;
  Link link = graph.RequireLink(sampler, "model");
  for (int guard = 0; guard < 64; ++guard) {
    const std::string cls = graph.Class(link.node);
    if (cls == "LoraLoaderModelOnly" || cls == "LoraLoader") {
      chain.loras.emplace_back(graph.Str(link.node, "lora_name"),
                               graph.Num(link.node, "strength_model", 1.0));
    } else if (cls == "UNETLoader") {
      chain.loader = graph.Str(link.node, "unet_name", "");
      return chain;
    } else if (cls == "CheckpointLoaderSimple") {
      chain.loader = graph.Str(link.node, "ckpt_name", "");
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
    } else if (cls == "TextEncodeQwenImage21" || cls == "CLIPTextEncode" ||
               cls == "MiniMaxH3ImageToVideo") {
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

ImageTask BuildImage(const Graph& graph, const std::string& sampler,
                     const ModelIds& models, const Link& latent,
                     std::vector<std::string>* warnings) {
  const std::string cls = graph.Class(sampler);
  ImageTask task;
  task.model = models.image;
  task.steps = static_cast<int>(graph.Num(sampler, "steps"));
  task.seed = Seed(graph.Num(sampler, SeedKey(cls), 0));
  if (cls == "KSampler" && graph.Num(sampler, "denoise", 1.0) < 0.999)
    throw NodeFailure(sampler, "unsupported_value",
                      "denoise below 1 (image to image) is not supported by "
                      "gufo Qwen-Image");
  const ModelChain chain = FollowModel(graph, sampler);
  if (!chain.loras.empty())
    throw NodeFailure(sampler, "unsupported_value",
                      "LoRAs are not supported for Qwen-Image in gufo (" +
                          chain.loras.front().first + ")");
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
  // Negative prompt: kept for the record; gufo Qwen-Image has no field.
  const Conditioning negative = FollowConditioning(graph, sampler, "negative");
  if (!negative.encoder.empty() && !negative.zeroed) {
    if (graph.Class(negative.encoder) == "TextEncodeQwenImage21")
      task.negative_prompt = graph.Str(negative.encoder, "negative_prompt", "");
    else
      task.negative_prompt = ClipText(graph, negative);
  }
  if (!task.negative_prompt.empty())
    warnings->push_back("node " + sampler +
                        ": negative prompt ignored (gufo Qwen-Image has none)");
  if (const double cfg = graph.Num(sampler, "cfg", 1.0);
      cfg != 1.0 && cfg != 2.5)
    warnings->push_back("node " + sampler + ": cfg " + std::to_string(cfg) +
                        " ignored (gufo uses its own guidance)");
  const std::string latent_class = graph.Class(latent.node);
  if (latent_class == "TextEncodeQwenImage21") {
    task.size_source = task.references.empty() ? SizeSource::kSquare
                                               : SizeSource::kFirstReference;
    if (task.resolution <= 0 && task.references.empty())
      task.resolution = 1024;
  } else if (latent_class == "VAEEncode") {
    task.size_source = SizeSource::kCanvas;
    task.canvas = ImageFile(graph, latent.node, "pixels",
                            graph.RequireLink(latent.node, "pixels"));
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
  task.steps = static_cast<int>(graph.Num(sampler, "steps"));
  task.seed = Seed(graph.Num(sampler, SeedKey(graph.Class(sampler)), 0));
  if (!((task.width == 512 && task.height == 512) ||
        (task.width == 1344 && task.height == 768)))
    throw NodeFailure(video, "unsupported_value",
                      "gufo MiniMax H3 renders 512x512 or 1344x768, not " +
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
    if (graph.LinkOf(guide, "audio"))
      throw NodeFailure(guide, "unsupported_value",
                        "audio anchors (MiniMaxH3AddGuide audio) are not "
                        "supported by gufo H3 yet");
    auto image = graph.LinkOf(guide, "image");
    if (!image)
      continue;
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
      throw NodeFailure(guide, "unsupported_value",
                        "intermediate guide frames (frame_idx " +
                            std::to_string(frame) +
                            ") are not supported by gufo H3 yet");
    }
  }
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
    if (graph.LinkOf(id, "voice_clone_prompt"))
      throw NodeFailure(id, "unsupported_value",
                        "voice_clone_prompt is not supported; connect "
                        "ref_audio from LoadAudio");
    task.reference_audio =
        AudioFile(graph, id, "ref_audio", graph.RequireLink(id, "ref_audio"));
    task.reference_text = graph.Str(id, "ref_text", "");
    task.embedding_only =
        graph.Bool(id, "x_vector_only", false) || task.reference_text.empty();
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
  const std::string language = graph.Str(id, "language", "Auto");
  task.language = language == "Auto" ? "" : Lower(language);
  if (const json::Value* seed = graph.Literal(id, "seed");
      seed != nullptr && seed->is_number())
    task.seed = Seed(seed->as_double());
  if (graph.Str(id, "model_choice", "1.7B") != "1.7B")
    warnings->push_back("node " + id + ": gufo serves the 1.7B Qwen3-TTS");
  for (const char* key : {"temperature", "top_p", "top_k", "repetition_penalty",
                          "max_new_tokens"}) {
    if (const json::Value* value = graph.Literal(id, key);
        value != nullptr && value->is_number())
      task.sampling[key] = value->as_double();
  }
  return task;
}

OutputKind KindOf(std::string_view class_type) {
  if (class_type == "SaveAudio" || class_type == "PreviewAudio")
    return OutputKind::kAudio;
  if (class_type == "SaveVideo")
    return OutputKind::kVideo;
  return OutputKind::kImage;
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

json::Value VideoRequestBody(const VideoTask& task) {
  json::Value body = json::Value::object();
  body["model"] = task.model;
  body["prompt"] = task.prompt;
  json::Value gufo = json::Value::object();
  gufo["seed"] = task.seed;
  if (task.steps > 0)
    gufo["steps"] = task.steps;
  if (task.family == VideoFamily::kH3) {
    body["size"] =
        std::to_string(task.width) + "x" + std::to_string(task.height);
    // seconds must be 1 or 5..15; gufo.frames sets the real length.
    const int seconds =
        task.frames <= 22
            ? 1
            : std::clamp(static_cast<int>(std::ceil(task.frames / 24.0)), 5,
                         14);
    body["seconds"] = seconds;
    gufo["frames"] = task.frames;
    if (!task.lora.empty())
      gufo["lora"] = task.lora;
    if (task.lora_strength)
      gufo["lora_strength"] = *task.lora_strength;
  } else {
    body["size"] =
        std::to_string(task.width) + "x" + std::to_string(task.height);
    body["seconds"] = (task.frames - 1) / 24.0;
    if (task.guidance)
      gufo["guidance"] = *task.guidance;
    if (!task.negative_prompt.empty())
      gufo["negative_prompt"] = task.negative_prompt;
  }
  body["gufo"] = std::move(gufo);
  return body;
}

bool BuildPlan(const json::Value& prompt, const ModelIds& models, Plan* plan,
               PlanError* error) {
  *plan = Plan{};
  if (!prompt.is_object() || prompt.empty()) {
    error->type = "invalid_prompt";
    error->message = "prompt must be a non-empty object of nodes";
    return false;
  }
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
        if (IsSampler(up) || IsTts(up)) {
          generators.push_back(node_id);
          tts = tts || IsTts(up);
        }
      }
      if (generators.empty())
        throw NodeFailure(id, "unsupported_link",
                          "output has no generating node (KSampler or "
                          "Qwen3-TTS) upstream");
      if (generators.size() > 1)
        throw NodeFailure(id, "unsupported_link",
                          "output depends on more than one generating node; "
                          "chained generators are not supported");
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
        if (tts) {
          task.spec = BuildSpeech(graph, generator, models, &plan->warnings);
        } else {
          const Link latent = graph.RequireLink(generator, "latent_image");
          const std::string latent_class = graph.Class(latent.node);
          if (latent_class == "MiniMaxH3ImageToVideo")
            task.spec = BuildH3(graph, generator, models, latent);
          else if (latent_class == "Wan22ImageToVideoLatent")
            task.spec = BuildWan(graph, generator, models, latent);
          else
            task.spec =
                BuildImage(graph, generator, models, latent, &plan->warnings);
        }
        found = task_of.emplace(generator, plan->tasks.size()).first;
        plan->tasks.push_back(std::move(task));
      }
      Task& task = plan->tasks[found->second];
      const bool speech = std::holds_alternative<SpeechTask>(task.spec);
      const bool image = std::holds_alternative<ImageTask>(task.spec);
      if ((speech && target.kind != OutputKind::kAudio) ||
          (image && target.kind != OutputKind::kImage))
        throw NodeFailure(
            id, "unsupported_link",
            cls + " cannot store the output of " + task.class_type);
      task.outputs.push_back(std::move(target));
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
