// The 1.4.9 image API request rules (src/cli/serve/image_api.cpp of
// release/1.4) on the provider side of the 2.0 extension: the host passes the
// "gufo"/"loras" members and the named image parts through
// (changes/0.2/image-request-extension); this file turns them into the
// model's Request exactly as 1.4.9 did. gufo-studio-windows.
#include "src/models/qwen_image_21/request.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>

#include "src/core/image.hpp"
#include "src/core/json.hpp"

namespace gufo::models::qwen_image_21 {
namespace {

constexpr std::array<std::string_view, 17> kExtensionFields{
    "negative_prompt",  "guidance",         "strength",      "init_image",
    "preview",          "mask",             "mask_channel",  "mask_composite",
    "mask_feather",     "mask_blur",        "regions",       "base_strength",
    "control_image",    "control_strength", "control_start", "control_end",
    "control_inpaint"};

double Real(const json::Value& fields, const char* name, double fallback,
            double low, double high) {
  const auto* value = fields.find(name);
  if (!value)
    return fallback;
  if (!value->is_number() || !std::isfinite(value->as_double()) ||
      value->as_double() < low || value->as_double() > high)
    throw std::invalid_argument(std::string("invalid gufo.") + name);
  return value->as_double();
}

std::uint64_t Number(const json::Value& fields, const char* name,
                     std::uint64_t fallback, std::uint64_t low,
                     std::uint64_t high) {
  const auto* value = fields.find(name);
  if (!value)
    return fallback;
  if (!value->is_number() || !std::isfinite(value->as_double()) ||
      value->as_double() < static_cast<double>(low) ||
      value->as_double() > static_cast<double>(high) ||
      std::floor(value->as_double()) != value->as_double())
    throw std::invalid_argument(std::string("invalid ") + name);
  return static_cast<std::uint64_t>(value->as_double());
}

std::string String(const json::Value& fields, const char* name,
                   const char* fallback) {
  const auto* value = fields.find(name);
  if (!value)
    return fallback;
  if (!value->is_string())
    throw std::invalid_argument(std::string(name) + " must be a string");
  return value->str();
}

std::vector<std::uint8_t> DecodeBase64(std::string_view text) {
  std::vector<std::uint8_t> out;
  out.reserve(text.size() / 4 * 3);
  std::uint32_t word = 0;
  int bits = 0;
  for (const char c : text) {
    int value;
    if (c >= 'A' && c <= 'Z')
      value = c - 'A';
    else if (c >= 'a' && c <= 'z')
      value = c - 'a' + 26;
    else if (c >= '0' && c <= '9')
      value = c - '0' + 52;
    else if (c == '+')
      value = 62;
    else if (c == '/')
      value = 63;
    else if (c == '=')
      break;
    else
      throw std::invalid_argument("invalid base64 in gufo.regions");
    word = (word << 6) | static_cast<std::uint32_t>(value);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<std::uint8_t>((word >> bits) & 0xFF));
    }
  }
  return out;
}

/// Inline data only: base64 or a data URL, never a server-side fetch.
Image InlineData(const json::Value& value, const std::string& what) {
  if (!value.is_string() || value.str().starts_with("http"))
    throw std::invalid_argument(what + " must be base64 or a data URL");
  const auto& text = value.str();
  return DecodeImage(core::ReadImageUrl(
      text.starts_with("data:") ? text : "data:image/png;base64," + text));
}

Image FromBuffer(const ext::ImageBuffer& buffer) {
  return Image{
      .width = buffer.width, .height = buffer.height, .rgba = buffer.rgba};
}

std::vector<Region> ParseRegions(const json::Value& value,
                                 std::vector<Image>& uploaded) {
  if (!value.is_array() || value.size() > kMaxRegions)
    throw std::invalid_argument("gufo.regions must be an array of at most " +
                                std::to_string(kMaxRegions) + " objects");
  static const std::set<std::string> kKeys{
      "prompt",       "strength",    "area",        "mask", "mask_image",
      "mask_channel", "mask_invert", "mask_bounds", "images"};
  std::vector<Region> regions;
  std::size_t next = 0;
  for (const auto& entry : value.items()) {
    if (!entry.is_object())
      throw std::invalid_argument("gufo.regions entries must be objects");
    for (const auto& [key, member] : entry.members()) {
      (void)member;
      if (!kKeys.contains(key))
        throw std::invalid_argument("unsupported region parameter: " + key);
    }
    auto& region = regions.emplace_back();
    region.prompt = String(entry, "prompt", "");
    region.strength = static_cast<float>(Real(entry, "strength", 1, 0, 10));
    if (const auto* area = entry.find("area")) {
      if (!area->is_object())
        throw std::invalid_argument("region area must be an object");
      const auto unit = String(*area, "unit", "px");
      if (unit != "px" && unit != "fraction")
        throw std::invalid_argument("region area unit must be px or fraction");
      const bool fraction = unit == "fraction";
      const double high = fraction ? 1 : 16384;
      region.area = RegionArea{
          .x = Real(*area, "x", 0, 0, high),
          .y = Real(*area, "y", 0, 0, high),
          .width = Real(*area, "width", fraction ? 1 : 0, 0, high),
          .height = Real(*area, "height", fraction ? 1 : 0, 0, high),
          .fraction = fraction};
    }
    const auto* mask = entry.find("mask");
    const auto* mask_image = entry.find("mask_image");
    if (mask && mask_image)
      throw std::invalid_argument("a region takes mask or mask_image");
    if (mask) {
      if (!mask->is_object())
        throw std::invalid_argument("region mask must be an object");
      const auto width = static_cast<int>(Number(*mask, "width", 0, 1, 16384));
      const auto height =
          static_cast<int>(Number(*mask, "height", 0, 1, 16384));
      if (static_cast<std::uint64_t>(width) * height > (16ULL << 20))
        throw std::invalid_argument("region mask exceeds 16 M values");
      const auto bytes = DecodeBase64(String(*mask, "data", ""));
      if (bytes.size() != static_cast<std::size_t>(width) * height)
        throw std::invalid_argument(
            "region mask data must hold width x height bytes");
      RegionMask grid{width, height, std::vector<float>(bytes.size())};
      for (std::size_t i = 0; i < bytes.size(); ++i)
        grid.values[i] = static_cast<float>(bytes[i]) / 255.0F;
      region.mask = std::move(grid);
    } else if (mask_image) {
      bool invert = false;
      if (const auto* flag = entry.find("mask_invert")) {
        if (!flag->is_bool())
          throw std::invalid_argument("region mask_invert must be a boolean");
        invert = flag->as_bool();
      }
      region.mask = RegionMaskFromImage(
          InlineData(*mask_image, "gufo.regions mask_image"),
          ParseMaskChannel(String(entry, "mask_channel", "auto")), invert);
    } else if (entry.contains("mask_channel") ||
               entry.contains("mask_invert")) {
      throw std::invalid_argument("mask_channel needs a region mask_image");
    }
    if (const auto* bounds = entry.find("mask_bounds")) {
      if (!bounds->is_bool())
        throw std::invalid_argument("region mask_bounds must be a boolean");
      region.mask_bounds = bounds->as_bool();
    }
    if (const auto* images = entry.find("images")) {
      if (images->is_array()) {
        for (const auto& item : images->items())
          region.images.push_back(InlineData(item, "gufo.regions images"));
      } else {
        const auto count = Number(entry, "images", 0, 0, 10);
        if (next + count > uploaded.size())
          throw std::invalid_argument(
              "gufo.regions names more images than region_image[] parts");
        for (std::uint64_t i = 0; i < count; ++i)
          region.images.push_back(std::move(uploaded[next++]));
      }
    }
  }
  if (next != uploaded.size())
    throw std::invalid_argument(
        "region_image[] parts must all belong to gufo.regions");
  return regions;
}

std::vector<Lora> ParseLoras(const json::Value* value,
                             const ProviderState& state) {
  if (!value)
    return state.default_loras;
  if (!value->is_array())
    throw std::invalid_argument("loras must be an array");
  std::vector<Lora> loras;
  for (const auto& entry : value->items()) {
    if (!entry.is_object())
      throw std::invalid_argument("loras entries must be objects");
    for (const auto& [key, member] : entry.members()) {
      (void)member;
      if (key != "name" && key != "scale")
        throw std::invalid_argument("unsupported LoRA parameter: " + key);
    }
    const auto* name = entry.find("name");
    if (!name || !name->is_string())
      throw std::invalid_argument("LoRA name must be a string");
    if (!state.loras.contains(name->str()))
      throw std::invalid_argument("unknown LoRA adapter: " + name->str());
    auto& lora = loras.emplace_back(Lora{name->str()});
    if (const auto* scale = entry.find("scale")) {
      if (!scale->is_number())
        throw std::invalid_argument("LoRA scale must be a number");
      // Converting an out-of-range double to float is undefined.
      const double number = scale->as_double();
      if (!std::isfinite(number) || std::abs(number) > 10)
        throw std::invalid_argument(
            "LoRA scale must be finite and at most 10 in magnitude");
      lora.scale = static_cast<float>(number);
    }
  }
  return loras;
}

}  // namespace

Request BuildRequest(const ext::ImageRequest& in, const ProviderState& state) {
  Request generation;
  generation.prompt = in.prompt;
  generation.width = in.width;
  generation.height = in.height;
  generation.steps = in.steps;
  generation.seed = in.seed;
  generation.images.reserve(in.images.size());
  for (const auto& image : in.images)
    generation.images.push_back(FromBuffer(image));

  std::optional<Image> init_image, mask, control_image;
  std::vector<Image> region_images;
  for (const auto& [name, buffer] : in.named_images) {
    if (name == "init_image")
      init_image = FromBuffer(buffer);
    else if (name == "mask")
      mask = FromBuffer(buffer);
    else if (name == "control_image")
      control_image = FromBuffer(buffer);
    else if (name == "region_image[]")
      region_images.push_back(FromBuffer(buffer));
    else
      throw std::invalid_argument("unsupported image part: " + name);
  }
  if (generation.images.size() + region_images.size() > kMaxReferenceImages)
    throw std::invalid_argument(
        "image inputs exceed 10 images (20 with regions) or 20 MiB");

  json::Value members = json::Value::object();
  if (!in.extension_json.empty()) {
    try {
      members = json::parse(in.extension_json);
    } catch (const std::exception& error) {
      throw std::invalid_argument(error.what());
    }
  }
  // Extensions (JSON "gufo": {...} or form fields "gufo.<name>"):
  // negative_prompt + guidance > 1 enable true CFG; strength < 1 with an
  // input image is image to image (ComfyUI KSampler denoise).
  const json::Value* extension = members.find("gufo");
  if (extension) {
    if (!extension->is_object())
      throw std::invalid_argument("gufo must be an object");
    for (const auto& [name, value] : extension->members()) {
      (void)value;
      if (std::find(kExtensionFields.begin(), kExtensionFields.end(), name) ==
          kExtensionFields.end())
        throw std::invalid_argument("unsupported image parameter: gufo." +
                                    name);
    }
    generation.negative_prompt = String(*extension, "negative_prompt", "");
    generation.guidance =
        static_cast<float>(Real(*extension, "guidance", 1, 0, 20));
    generation.strength = Real(*extension, "strength", 1, 0.01, 1);
    if (const auto* init = extension->find("init_image")) {
      if (init_image)
        throw std::invalid_argument(
            "gufo.init_image must be base64 or a data URL, given once");
      init_image = InlineData(*init, "gufo.init_image");
    }
    if (const auto* inline_mask = extension->find("mask")) {
      if (mask)
        throw std::invalid_argument(
            "gufo.mask must be base64 or a data URL, given once");
      mask = InlineData(*inline_mask, "gufo.mask");
    }
    generation.mask_channel =
        ParseMaskChannel(String(*extension, "mask_channel", "auto"));
    if (const auto* composite = extension->find("mask_composite")) {
      if (!composite->is_bool())
        throw std::invalid_argument("gufo.mask_composite must be a boolean");
      generation.mask_composite = composite->as_bool();
    }
    generation.mask_feather = static_cast<int>(
        Number(*extension, "mask_feather", kDefaultMaskFeather, 0, 64));
    generation.mask_blur = static_cast<int>(
        Number(*extension, "mask_blur", kDefaultMaskBlur, 0, 4));
    if (const auto* regions = extension->find("regions"))
      generation.regions = ParseRegions(*regions, region_images);
    generation.base_strength =
        static_cast<float>(Real(*extension, "base_strength", 1, 0, 10));
    if (const auto* control = extension->find("control_image")) {
      if (control_image)
        throw std::invalid_argument(
            "gufo.control_image must be base64 or a data URL, given once");
      control_image = InlineData(*control, "gufo.control_image");
    }
    if (const auto* preview = extension->find("preview"); preview &&
        !preview->is_object())
      throw std::invalid_argument("gufo.preview must be an object");
  }
  // Fun ControlNet-Union: a ready structure image (pose skeleton, depth,
  // canny, lineart ...); gufo does no preprocessing. "control_inpaint":
  // true without a control image runs the inpaint channels alone (control
  // channels zero, as ComfyUI ZImageFunControlnet without "image").
  const json::Value* inpaint_only =
      extension && !control_image ? extension->find("control_inpaint")
                                  : nullptr;
  if (inpaint_only && inpaint_only->is_bool() && inpaint_only->as_bool()) {
    if (!state.controlnet)
      throw std::invalid_argument(
          "gufo.control_inpaint needs a server started with --controlnet");
    ControlInput control;
    control.inpaint = true;
    control.strength =
        static_cast<float>(Real(*extension, "control_strength", 1, 0, 2));
    control.start = Real(*extension, "control_start", 0, 0, 1);
    control.end = Real(*extension, "control_end", 1, 0, 1);
    generation.control = std::move(control);
  } else if (control_image) {
    if (!state.controlnet)
      throw std::invalid_argument(
          "gufo.control_image needs a server started with --controlnet");
    ControlInput control;
    control.image = std::move(*control_image);
    if (extension) {
      control.strength =
          static_cast<float>(Real(*extension, "control_strength", 1, 0, 2));
      control.start = Real(*extension, "control_start", 0, 0, 1);
      control.end = Real(*extension, "control_end", 1, 0, 1);
      if (const auto* inpaint = extension->find("control_inpaint")) {
        if (!inpaint->is_bool())
          throw std::invalid_argument("gufo.control_inpaint must be a boolean");
        control.inpaint = inpaint->as_bool();
      }
    }
    generation.control = std::move(control);
  } else if (extension) {
    for (const char* key : {"control_strength", "control_start", "control_end",
                            "control_inpaint"})
      if (extension->contains(key))
        throw std::invalid_argument(std::string("gufo.") + key +
                                    " needs gufo.control_image or "
                                    "gufo.control_inpaint true");
  }
  if (!region_images.empty() && (!extension || !extension->contains("regions")))
    throw std::invalid_argument("region_image[] needs gufo.regions");
  if (extension && extension->contains("strength") && !init_image) {
    // A single uploaded "image" is the start image, not a reference.
    if (generation.images.size() != 1)
      throw std::invalid_argument(
          "gufo.strength needs init_image or exactly one image");
    init_image = std::move(generation.images.front());
    generation.images.clear();
  }
  if (mask && !init_image) {
    // Like OpenAI, the mask edits the first image, which stays a reference
    // (the edit model sees the whole original).
    if (generation.images.empty())
      throw std::invalid_argument("a mask needs an image to edit");
    init_image = generation.images.front();
  }
  generation.init_image = std::move(init_image);
  generation.mask = std::move(mask);
  if (in.size_auto) {
    if (generation.init_image) {
      // Keep the start image's canvas, rounded to the 32-pixel grid.
      const auto grid = [](int v) {
        return std::max(32, static_cast<int>(std::nearbyint(v / 32.0)) * 32);
      };
      generation.width = grid(generation.init_image->width);
      generation.height = grid(generation.init_image->height);
    } else if (generation.images.empty() && generation.control) {
      // The control image sets the aspect ratio (the host used the last
      // reference when there is one).
      const auto& last = generation.control->image;
      const double ratio = static_cast<double>(last.width) / last.height;
      const double w = std::sqrt(1024.0 * 1024 * ratio);
      generation.width = static_cast<int>(std::nearbyint(w / 32)) * 32;
      generation.height = static_cast<int>(std::nearbyint(w / ratio / 32)) * 32;
    }
  }
  generation.loras = ParseLoras(members.find("loras"), state);
  ValidateRequest(generation);
  return generation;
}

StartupOptions ParseStartupOptions(
    const std::vector<std::pair<std::string, std::string>>& options) {
  StartupOptions out;
  for (const auto& [name, value] : options) {
    if (name == "lora") {
      const auto separator = value.find('=');
      if (separator == std::string::npos || separator == 0 ||
          separator + 1 == value.size())
        throw std::invalid_argument("--lora expects NAME=PATH");
      out.loras.emplace_back(value.substr(0, separator),
                             value.substr(separator + 1));
    } else if (name == "lora-default") {
      const std::size_t separator = value.rfind(':');
      float scale = 0;
      if (separator == std::string::npos || separator == 0)
        throw std::invalid_argument("--lora-default expects NAME:SCALE");
      const std::string_view text(value.data() + separator + 1,
                                  value.size() - separator - 1);
      const auto [end, ec] =
          std::from_chars(text.data(), text.data() + text.size(), scale);
      if (text.empty() || ec != std::errc{} ||
          end != text.data() + text.size())
        throw std::invalid_argument("--lora-default expects NAME:SCALE");
      out.default_loras.push_back({value.substr(0, separator), scale});
    } else if (name == "controlnet") {
      if (!out.controlnet.empty())
        throw std::invalid_argument("--controlnet given twice");
      out.controlnet = value;
    } else {
      throw std::invalid_argument("Qwen-Image-2.1 has no option --" + name);
    }
  }
  std::set<std::string, std::less<>> names;
  for (const auto& [name, path] : out.loras) {
    (void)path;
    if (name.empty() || name.size() > 64 ||
        name.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrs"
                               "tuvwxyz0123456789._-") != std::string::npos)
      throw std::invalid_argument("LoRA name '" + name +
                                  "' must be 1-64 of [A-Za-z0-9._-]");
    if (!names.insert(name).second)
      throw std::invalid_argument("duplicate --lora name '" + name + "'");
  }
  for (const auto& lora : out.default_loras)
    if (!names.contains(lora.name))
      throw std::invalid_argument("--lora-default names unknown adapter '" +
                                  lora.name + "'");
  Request defaults;
  defaults.loras = out.default_loras;
  ValidateRequest(defaults);
  return out;
}

std::string LorasJson(
    const std::vector<std::pair<std::string, std::string>>& files,
    const std::vector<LoraInfo>& infos, const std::vector<Lora>& defaults,
    const std::vector<Lora>& active) {
  const auto scale = [](const std::vector<Lora>& loras,
                        const std::string& name) -> json::Value {
    for (const auto& lora : loras)
      if (lora.name == name) {
        // Shortest float text, so 0.8 is listed as 0.8.
        char text[32];
        const auto end =
            std::to_chars(text, text + sizeof(text), lora.scale).ptr;
        double value = lora.scale;
        std::from_chars(text, end, value);
        return value;
      }
    return nullptr;
  };
  auto body = json::Value::array();
  for (std::size_t i = 0; i < files.size(); ++i) {
    const auto& [name, path] = files[i];
    const LoraInfo info = i < infos.size() ? infos[i] : LoraInfo{};
    auto metadata = json::Value::object();
    metadata["format"] = info.format;
    metadata["targets"] = info.targets;
    metadata["rank"] =
        std::to_string(info.min_rank) + '-' + std::to_string(info.max_rank);
    auto item = json::Value::object();
    item["name"] = name;
    item["alias"] = name;
    item["path"] = std::filesystem::path(path).filename().string();
    item["metadata"] = std::move(metadata);
    item["default_scale"] = scale(defaults, name);
    item["active_scale"] = scale(active, name);
    body.push_back(std::move(item));
  }
  return body.dump();
}

std::string LoraList(const std::vector<Lora>& loras) {
  std::ostringstream list;
  for (const auto& lora : loras)
    if (lora.scale != 0)
      list << (list.tellp() > 0 ? "," : "") << lora.name << ':' << lora.scale;
  return list.tellp() > 0 ? list.str() : "none";
}

}  // namespace gufo::models::qwen_image_21
