// Safetensors header and SRVGGNetCompact architecture (see srvgg_weights.hpp).
// Written with AI assistance (Claude Code).
#include "src/models/realesrgan/srvgg_weights.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>

#include "src/core/json.hpp"

namespace gufo::models::realesrgan {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

constexpr std::uint64_t kMaxHeaderBytes = 16U << 20U;

std::string Lower(std::string_view text) {
  std::string out(text);
  std::ranges::transform(out, out.begin(), [](char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  });
  return out;
}

/// "body.12.weight" -> 12 and "weight"; false for any other name.
bool SplitBodyName(const std::string& name, int* index, std::string* leaf) {
  constexpr std::string_view prefix = "body.";
  if (!name.starts_with(prefix))
    return false;
  const std::size_t dot = name.find('.', prefix.size());
  if (dot == std::string::npos || dot == prefix.size())
    return false;
  int value = 0;
  for (std::size_t i = prefix.size(); i < dot; ++i) {
    const char c = name[i];
    if (c < '0' || c > '9' || value > 100000)
      return false;
    value = value * 10 + (c - '0');
  }
  *index = value;
  *leaf = name.substr(dot + 1);
  return true;
}

bool ParseMetadataInt(const SafetensorsHeader& header, const char* key,
                      int* value) {
  const auto it = header.metadata.find(key);
  if (it == header.metadata.end())
    return false;
  try {
    std::size_t used = 0;
    const int parsed = std::stoi(it->second, &used);
    if (used != it->second.size())
      return false;
    *value = parsed;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace

std::size_t SafetensorsTensor::count() const noexcept {
  std::size_t n = 1;
  for (std::size_t dim : shape)
    n *= dim;
  return n;
}

std::size_t SafetensorsElementBytes(std::string_view dtype) {
  if (dtype == "F64" || dtype == "I64" || dtype == "U64")
    return 8;
  if (dtype == "F32" || dtype == "I32" || dtype == "U32")
    return 4;
  if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16")
    return 2;
  if (dtype == "I8" || dtype == "U8" || dtype == "BOOL" || dtype == "F8_E4M3" ||
      dtype == "F8_E5M2")
    return 1;
  return 0;
}

bool ParseSafetensorsHeader(std::string_view json_text,
                            std::uint64_t data_bytes,
                            SafetensorsHeader* header, std::string* error) {
  SafetensorsHeader parsed;
  parsed.header_bytes = json_text.size();
  parsed.data_bytes = data_bytes;
  json::Value root;
  try {
    root = json::parse(std::string(json_text));
  } catch (const std::exception& e) {
    SetError(error, std::string("safetensors header is not JSON: ") + e.what());
    return false;
  }
  if (!root.is_object()) {
    SetError(error, "safetensors header is not a JSON object");
    return false;
  }
  for (const auto& [name, entry] : root.members()) {
    if (name == "__metadata__") {
      if (!entry.is_object()) {
        SetError(error, "safetensors __metadata__ is not an object");
        return false;
      }
      for (const auto& [key, value] : entry.members())
        if (value.is_string())
          parsed.metadata[key] = value.str();
      continue;
    }
    if (!entry.is_object()) {
      SetError(error, "safetensors tensor " + name + " is not an object");
      return false;
    }
    const json::Value* shape = entry.find("shape");
    const json::Value* offsets = entry.find("data_offsets");
    SafetensorsTensor tensor;
    tensor.dtype = entry.member_str("dtype");
    if (tensor.dtype.empty() || shape == nullptr || !shape->is_array() ||
        offsets == nullptr || !offsets->is_array() || offsets->size() != 2) {
      SetError(error, "safetensors tensor " + name +
                          " lacks dtype, shape or data_offsets");
      return false;
    }
    for (const json::Value& dim : shape->items()) {
      if (!dim.is_number()) {
        SetError(error, "safetensors tensor " + name + " has a bad shape");
        return false;
      }
      tensor.shape.push_back(dim.as_size());
    }
    if (!offsets->items()[0].is_number() || !offsets->items()[1].is_number()) {
      SetError(error, "safetensors tensor " + name + " has bad data_offsets");
      return false;
    }
    tensor.begin = offsets->items()[0].as_size();
    tensor.end = offsets->items()[1].as_size();
    const std::size_t element = SafetensorsElementBytes(tensor.dtype);
    if (tensor.end < tensor.begin || tensor.end > data_bytes ||
        (element != 0 && tensor.end - tensor.begin != tensor.count() * element)) {
      SetError(error, "safetensors tensor " + name +
                          " has an extent outside the data block or not "
                          "matching its shape");
      return false;
    }
    parsed.tensors[name] = std::move(tensor);
  }
  *header = std::move(parsed);
  return true;
}

bool ReadSafetensorsHeader(const std::filesystem::path& path,
                           SafetensorsHeader* header, std::string* error) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    SetError(error, "cannot open " + path.string());
    return false;
  }
  file.seekg(0, std::ios::end);
  const auto size = static_cast<std::uint64_t>(file.tellg());
  file.seekg(0);
  std::uint64_t header_size = 0;
  if (size < 8 || !file.read(reinterpret_cast<char*>(&header_size), 8) ||
      header_size == 0 || header_size > kMaxHeaderBytes ||
      header_size > size - 8) {
    SetError(error, "invalid safetensors header in " + path.string());
    return false;
  }
  std::string text(static_cast<std::size_t>(header_size), '\0');
  if (!file.read(text.data(), static_cast<std::streamsize>(text.size()))) {
    SetError(error, "short read of " + path.string());
    return false;
  }
  std::string parse_error;
  if (!ParseSafetensorsHeader(text, size - 8 - header_size, header,
                              &parse_error)) {
    SetError(error, path.string() + ": " + parse_error);
    return false;
  }
  return true;
}

bool DescribeSrvgg(const SafetensorsHeader& header,
                   SrvggArchitecture* architecture, std::string* error) {
  // Collect body.N.{weight,bias} and find the last index.
  int last = -1;
  for (const auto& [name, tensor] : header.tensors) {
    int index = 0;
    std::string leaf;
    if (!SplitBodyName(name, &index, &leaf)) {
      SetError(error, "unexpected tensor " + name +
                          " (not a SRVGGNetCompact body tensor)");
      return false;
    }
    if (leaf != "weight" && leaf != "bias") {
      SetError(error, "unexpected tensor " + name);
      return false;
    }
    last = std::max(last, index);
  }
  // body.0 conv, body.1 PReLU, ..., body.2(N+1) conv: last = 2N + 2.
  if (last < 2 || last % 2 != 0) {
    SetError(error, "SRVGGNetCompact body must end in a convolution "
                    "(body.2(N+1).weight)");
    return false;
  }
  const int convolutions = last / 2 - 1;
  const auto tensor = [&](const std::string& name) -> const SafetensorsTensor* {
    const auto it = header.tensors.find(name);
    return it == header.tensors.end() ? nullptr : &it->second;
  };
  const SafetensorsTensor* first = tensor("body.0.weight");
  if (first == nullptr || first->shape.size() != 4 || first->shape[2] != 3 ||
      first->shape[3] != 3) {
    SetError(error, "body.0.weight is not a 3x3 convolution");
    return false;
  }
  SrvggArchitecture arch;
  arch.dtype = first->dtype;
  arch.features = static_cast<int>(first->shape[0]);
  arch.in_channels = static_cast<int>(first->shape[1]);
  std::size_t parameters = 0;
  for (int index = 0; index <= last; ++index) {
    const std::string base = "body." + std::to_string(index) + ".";
    const SafetensorsTensor* weight = tensor(base + "weight");
    if (weight == nullptr) {
      SetError(error, "missing " + base + "weight");
      return false;
    }
    if (weight->dtype != arch.dtype) {
      SetError(error, base + "weight has dtype " + weight->dtype + ", expected " +
                          arch.dtype);
      return false;
    }
    parameters += weight->count();
    if (index % 2 == 1) {
      // PReLU slope per feature; no bias.
      if (weight->shape.size() != 1 ||
          weight->shape[0] != static_cast<std::size_t>(arch.features) ||
          tensor(base + "bias") != nullptr) {
        SetError(error, base + "weight is not a per-feature PReLU slope");
        return false;
      }
      continue;
    }
    const SafetensorsTensor* bias = tensor(base + "bias");
    const std::size_t in = index == 0 ? static_cast<std::size_t>(arch.in_channels)
                                      : static_cast<std::size_t>(arch.features);
    if (weight->shape.size() != 4 || weight->shape[1] != in ||
        weight->shape[2] != 3 || weight->shape[3] != 3 || bias == nullptr ||
        bias->shape.size() != 1 || bias->shape[0] != weight->shape[0] ||
        bias->dtype != arch.dtype) {
      SetError(error, base + "weight/bias do not form a 3x3 convolution of " +
                          std::to_string(in) + " inputs");
      return false;
    }
    parameters += bias->count();
    if (index != last && weight->shape[0] != static_cast<std::size_t>(arch.features)) {
      SetError(error, base + "weight does not keep " +
                          std::to_string(arch.features) + " features");
      return false;
    }
  }
  const std::size_t out = tensor("body." + std::to_string(last) + ".weight")->shape[0];
  // out = out_channels * S^2 with out_channels = in_channels (RGB).
  arch.out_channels = arch.in_channels;
  if (arch.out_channels <= 0 || out % static_cast<std::size_t>(arch.out_channels) != 0) {
    SetError(error, "last convolution has " + std::to_string(out) +
                        " outputs, not a multiple of the channel count");
    return false;
  }
  const std::size_t squared = out / static_cast<std::size_t>(arch.out_channels);
  int scale = 0;
  for (int s = 1; s <= 8; ++s)
    if (static_cast<std::size_t>(s) * s == squared)
      scale = s;
  if (scale == 0) {
    SetError(error, "last convolution's " + std::to_string(out) +
                        " outputs are not channels x S^2");
    return false;
  }
  arch.scale = scale;
  arch.convolutions = convolutions;
  arch.parameters = parameters;
  int meta = 0;
  if (ParseMetadataInt(header, "num_conv", &meta) && meta != convolutions) {
    SetError(error, "metadata num_conv=" + std::to_string(meta) + " but " +
                        std::to_string(convolutions) + " body convolutions");
    return false;
  }
  if (ParseMetadataInt(header, "num_feat", &meta) && meta != arch.features) {
    SetError(error, "metadata num_feat=" + std::to_string(meta) + " but " +
                        std::to_string(arch.features) + " features");
    return false;
  }
  if (ParseMetadataInt(header, "upscale", &meta) && meta != scale) {
    SetError(error, "metadata upscale=" + std::to_string(meta) + " but x" +
                        std::to_string(scale) + " pixel shuffle");
    return false;
  }
  *architecture = std::move(arch);
  return true;
}

std::string VariantFromFileName(std::string_view name) {
  const std::string lower = Lower(name);
  if (lower.find("wdn") != std::string::npos)
    return "wdn";
  if (lower.find("anime") != std::string::npos)
    return "anime";
  if (lower.find("general") != std::string::npos)
    return "general";
  return "";
}

}  // namespace gufo::models::realesrgan
