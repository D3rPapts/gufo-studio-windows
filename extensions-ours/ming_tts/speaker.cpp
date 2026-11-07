#include "ming_tts/speaker.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include <gufo/ext/platform/mapped_file.hpp>
#include "ming_tts/cpu/fft.hpp"

namespace gufo::models::ming_tts {
namespace {

// ---------------------------------------------------------------- ONNX ----
// Minimal protobuf wire-format reader for the parts of an ONNX ModelProto the
// CAM++ port needs: graph nodes (op, inputs, outputs, int/float attributes,
// Constant tensors) and float initializers.

struct OnnxTensor {
  std::vector<std::int64_t> dims;
  std::vector<float> values;  // float or int64 payload converted to float
};

struct OnnxNode {
  std::string op;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::map<std::string, std::vector<std::int64_t>> ints;
  std::map<std::string, std::vector<float>> floats;
  OnnxTensor tensor;  // Constant "value"
};

class Wire {
public:
  Wire(const std::uint8_t* begin, const std::uint8_t* end) : p_(begin), end_(end) {}
  [[nodiscard]] bool done() const { return p_ >= end_; }

  std::uint64_t Varint() {
    std::uint64_t value = 0;
    for (int shift = 0; shift < 64; shift += 7) {
      if (p_ >= end_) {
        throw std::runtime_error("truncated varint");
      }
      const std::uint8_t byte = *p_++;
      value |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
      if ((byte & 0x80U) == 0) {
        return value;
      }
    }
    throw std::runtime_error("varint too long");
  }
  // Returns field number and wire type.
  std::pair<std::uint32_t, std::uint32_t> Key() {
    const std::uint64_t key = Varint();
    return {static_cast<std::uint32_t>(key >> 3U),
            static_cast<std::uint32_t>(key & 7U)};
  }
  Wire Bytes() {
    const std::uint64_t length = Varint();
    if (length > static_cast<std::uint64_t>(end_ - p_)) {
      throw std::runtime_error("truncated length-delimited field");
    }
    Wire sub(p_, p_ + length);
    p_ += length;
    return sub;
  }
  std::string String() {
    Wire sub = Bytes();
    return std::string(reinterpret_cast<const char*>(sub.p_),
                       static_cast<std::size_t>(sub.end_ - sub.p_));
  }
  std::uint32_t Fixed32() {
    if (end_ - p_ < 4) {
      throw std::runtime_error("truncated fixed32");
    }
    std::uint32_t value;
    std::memcpy(&value, p_, 4);
    p_ += 4;
    return value;
  }
  void Skip(std::uint32_t wire_type) {
    switch (wire_type) {
      case 0:
        (void)Varint();
        break;
      case 1:
        p_ += 8;
        break;
      case 2:
        (void)Bytes();
        break;
      case 5:
        p_ += 4;
        break;
      default:
        throw std::runtime_error("unsupported protobuf wire type");
    }
  }
  [[nodiscard]] const std::uint8_t* data() const { return p_; }
  [[nodiscard]] std::size_t size() const { return static_cast<std::size_t>(end_ - p_); }

private:
  const std::uint8_t* p_;
  const std::uint8_t* end_;
};

float BitsToFloat(std::uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, 4);
  return value;
}

void ReadPackedOrSingleInts(Wire* wire, std::uint32_t wire_type,
                            std::vector<std::int64_t>* out) {
  if (wire_type == 2) {
    Wire packed = wire->Bytes();
    while (!packed.done()) {
      out->push_back(static_cast<std::int64_t>(packed.Varint()));
    }
  } else {
    out->push_back(static_cast<std::int64_t>(wire->Varint()));
  }
}

void ReadPackedOrSingleFloats(Wire* wire, std::uint32_t wire_type,
                              std::vector<float>* out) {
  if (wire_type == 2) {
    Wire packed = wire->Bytes();
    while (!packed.done()) {
      out->push_back(BitsToFloat(packed.Fixed32()));
    }
  } else {
    out->push_back(BitsToFloat(wire->Fixed32()));
  }
}

OnnxTensor ParseTensor(Wire wire, std::string* name) {
  OnnxTensor tensor;
  std::int64_t data_type = 0;
  std::vector<std::int64_t> int64_data;
  std::string raw;
  while (!wire.done()) {
    const auto [field, type] = wire.Key();
    if (field == 1) {
      ReadPackedOrSingleInts(&wire, type, &tensor.dims);
    } else if (field == 2 && type == 0) {
      data_type = static_cast<std::int64_t>(wire.Varint());
    } else if (field == 4) {
      ReadPackedOrSingleFloats(&wire, type, &tensor.values);
    } else if (field == 7) {
      ReadPackedOrSingleInts(&wire, type, &int64_data);
    } else if (field == 8 && type == 2) {
      *name = wire.String();
    } else if (field == 9 && type == 2) {
      raw = wire.String();
    } else {
      wire.Skip(type);
    }
  }
  if (!raw.empty()) {
    if (data_type == 1) {  // FLOAT
      tensor.values.resize(raw.size() / 4);
      std::memcpy(tensor.values.data(), raw.data(), tensor.values.size() * 4);
    } else if (data_type == 7) {  // INT64
      const std::size_t count = raw.size() / 8;
      for (std::size_t i = 0; i < count; ++i) {
        std::int64_t value;
        std::memcpy(&value, raw.data() + i * 8, 8);
        tensor.values.push_back(static_cast<float>(value));
      }
    }
  } else if (!int64_data.empty()) {
    for (const std::int64_t value : int64_data) {
      tensor.values.push_back(static_cast<float>(value));
    }
  }
  return tensor;
}

OnnxNode ParseNode(Wire wire) {
  OnnxNode node;
  while (!wire.done()) {
    const auto [field, type] = wire.Key();
    if (field == 1 && type == 2) {
      node.inputs.push_back(wire.String());
    } else if (field == 2 && type == 2) {
      node.outputs.push_back(wire.String());
    } else if (field == 4 && type == 2) {
      node.op = wire.String();
    } else if (field == 5 && type == 2) {
      Wire attribute = wire.Bytes();
      std::string name;
      std::vector<std::int64_t> ints;
      std::vector<float> floats;
      while (!attribute.done()) {
        const auto [afield, atype] = attribute.Key();
        if (afield == 1 && atype == 2) {
          name = attribute.String();
        } else if (afield == 2 && atype == 5) {
          floats.push_back(BitsToFloat(attribute.Fixed32()));
        } else if (afield == 3 && atype == 0) {
          ints.push_back(static_cast<std::int64_t>(attribute.Varint()));
        } else if (afield == 5 && atype == 2) {
          std::string ignored;
          node.tensor = ParseTensor(attribute.Bytes(), &ignored);
        } else if (afield == 7) {
          ReadPackedOrSingleFloats(&attribute, atype, &floats);
        } else if (afield == 8) {
          ReadPackedOrSingleInts(&attribute, atype, &ints);
        } else {
          attribute.Skip(atype);
        }
      }
      if (!ints.empty()) {
        node.ints[name] = std::move(ints);
      }
      if (!floats.empty()) {
        node.floats[name] = std::move(floats);
      }
    } else {
      wire.Skip(type);
    }
  }
  return node;
}

// ------------------------------------------------------------ tensors ----

struct Feature {  // [channels, height, width]; 1-D features use height 1
  std::int64_t c{0}, h{1}, w{0};
  std::vector<float> data;
  float* row(std::int64_t channel, std::int64_t y = 0) {
    return data.data() + (channel * h + y) * w;
  }
  [[nodiscard]] const float* row(std::int64_t channel, std::int64_t y = 0) const {
    return data.data() + (channel * h + y) * w;
  }
};

struct ConvParams {
  const std::vector<float>* weight{nullptr};
  const std::vector<float>* bias{nullptr};
  std::int64_t out_channels{0}, in_channels{0}, kh{1}, kw{1};
  std::int64_t stride_h{1}, stride_w{1}, pad_h{0}, pad_w{0}, dilation_h{1},
      dilation_w{1};
};

Feature Conv(const Feature& x, const ConvParams& p) {
  if (x.c != p.in_channels) {
    throw std::runtime_error("CAM++ conv channel mismatch");
  }
  Feature y;
  y.c = p.out_channels;
  y.h = (x.h + 2 * p.pad_h - p.dilation_h * (p.kh - 1) - 1) / p.stride_h + 1;
  y.w = (x.w + 2 * p.pad_w - p.dilation_w * (p.kw - 1) - 1) / p.stride_w + 1;
  y.data.assign(static_cast<std::size_t>(y.c * y.h * y.w), 0.0F);
#pragma omp parallel for schedule(dynamic)
  for (std::int64_t oc = 0; oc < y.c; ++oc) {
    const float b = p.bias != nullptr ? (*p.bias)[static_cast<std::size_t>(oc)] : 0.0F;
    for (std::int64_t oy = 0; oy < y.h; ++oy) {
      float* out = y.data.data() + (oc * y.h + oy) * y.w;
      std::fill(out, out + y.w, b);
      for (std::int64_t ic = 0; ic < x.c; ++ic) {
        for (std::int64_t ky = 0; ky < p.kh; ++ky) {
          const std::int64_t iy = oy * p.stride_h - p.pad_h + ky * p.dilation_h;
          if (iy < 0 || iy >= x.h) {
            continue;
          }
          const float* in = x.row(ic, iy);
          for (std::int64_t kx = 0; kx < p.kw; ++kx) {
            const float weight =
                (*p.weight)[static_cast<std::size_t>(((oc * p.in_channels + ic) * p.kh + ky) * p.kw + kx)];
            const std::int64_t offset = kx * p.dilation_w - p.pad_w;
            if (p.stride_w == 1) {
              const std::int64_t lo = std::max<std::int64_t>(0, -offset);
              const std::int64_t hi = std::min<std::int64_t>(y.w, x.w - offset);
              for (std::int64_t ox = lo; ox < hi; ++ox) {
                out[ox] += weight * in[ox + offset];
              }
            } else {
              for (std::int64_t ox = 0; ox < y.w; ++ox) {
                const std::int64_t ix = ox * p.stride_w + offset;
                if (ix >= 0 && ix < x.w) {
                  out[ox] += weight * in[ix];
                }
              }
            }
          }
        }
      }
    }
  }
  return y;
}

void Relu(Feature* x) {
  for (float& v : x->data) {
    v = std::max(v, 0.0F);
  }
}

}  // namespace

// --------------------------------------------------------------- CAM++ ----

struct CamPlusPlus::Impl {
  std::unordered_map<std::string, OnnxTensor> initializers;
  std::unordered_map<std::string, OnnxTensor> constants;
  std::unordered_map<std::string, OnnxNode> by_output;

  [[nodiscard]] const OnnxTensor& Param(const std::string& name) const {
    if (const auto found = initializers.find(name); found != initializers.end()) {
      return found->second;
    }
    if (const auto found = constants.find(name); found != constants.end()) {
      return found->second;
    }
    throw std::runtime_error("CAM++ parameter missing: " + name);
  }
  [[nodiscard]] const OnnxNode* Node(const std::string& output) const {
    const auto found = by_output.find(output);
    return found == by_output.end() ? nullptr : &found->second;
  }
  [[nodiscard]] const OnnxNode& RequireNode(const std::string& output,
                                            const char* op) const {
    const OnnxNode* node = Node(output);
    if (node == nullptr || node->op != op) {
      throw std::runtime_error("CAM++ node missing: " + output);
    }
    return *node;
  }

  [[nodiscard]] ConvParams ConvOf(const std::string& output) const {
    const OnnxNode& node = RequireNode(output, "Conv");
    ConvParams p;
    const OnnxTensor& w = Param(node.inputs.at(1));
    p.weight = &w.values;
    if (node.inputs.size() > 2 && !node.inputs[2].empty()) {
      p.bias = &Param(node.inputs[2]).values;
    }
    p.out_channels = w.dims.at(0);
    p.in_channels = w.dims.at(1);
    const auto attr = [&](const char* name) -> std::vector<std::int64_t> {
      const auto found = node.ints.find(name);
      return found == node.ints.end() ? std::vector<std::int64_t>{} : found->second;
    };
    if (const auto group = attr("group"); !group.empty() && group[0] != 1) {
      throw std::runtime_error("CAM++ grouped conv unsupported");
    }
    const auto strides = attr("strides");
    const auto pads = attr("pads");
    const auto dilations = attr("dilations");
    if (w.dims.size() == 4) {
      p.kh = w.dims[2];
      p.kw = w.dims[3];
      p.stride_h = strides.size() == 2 ? strides[0] : 1;
      p.stride_w = strides.size() == 2 ? strides[1] : 1;
      p.pad_h = pads.size() == 4 ? pads[0] : 0;
      p.pad_w = pads.size() == 4 ? pads[1] : 0;
      p.dilation_h = dilations.size() == 2 ? dilations[0] : 1;
      p.dilation_w = dilations.size() == 2 ? dilations[1] : 1;
      if (pads.size() == 4 && (pads[2] != pads[0] || pads[3] != pads[1])) {
        throw std::runtime_error("CAM++ asymmetric 2-D padding unsupported");
      }
    } else {
      p.kw = w.dims.at(2);
      p.stride_w = strides.empty() ? 1 : strides[0];
      p.pad_w = pads.empty() ? 0 : pads[0];
      p.dilation_w = dilations.empty() ? 1 : dilations[0];
      if (pads.size() == 2 && pads[1] != pads[0]) {
        throw std::runtime_error("CAM++ asymmetric 1-D padding unsupported");
      }
    }
    return p;
  }

  void BatchNorm(Feature* x, const std::string& output) const {
    const OnnxNode& node = RequireNode(output, "BatchNormalization");
    const auto& scale = Param(node.inputs.at(1)).values;
    const auto& bias = Param(node.inputs.at(2)).values;
    const auto& mean = Param(node.inputs.at(3)).values;
    const auto& var = Param(node.inputs.at(4)).values;
    float eps = 1e-5F;
    if (const auto found = node.floats.find("epsilon"); found != node.floats.end()) {
      eps = found->second.at(0);
    }
    for (std::int64_t c = 0; c < x->c; ++c) {
      const auto i = static_cast<std::size_t>(c);
      const float inv = 1.0F / std::sqrt(var[i] + eps);
      for (std::int64_t y = 0; y < x->h; ++y) {
        float* row = x->row(c, y);
        for (std::int64_t t = 0; t < x->w; ++t) {
          row[t] = (row[t] - mean[i]) * inv * scale[i] + bias[i];
        }
      }
    }
  }

  // CAMLayer.seg_pooling(seg_len=100, avg) added to the global mean.
  static Feature Context(const Feature& x) {
    constexpr std::int64_t kSegment = 100;
    Feature ctx;
    ctx.c = x.c;
    ctx.w = x.w;
    ctx.data.resize(x.data.size());
    for (std::int64_t c = 0; c < x.c; ++c) {
      const float* in = x.row(c);
      float* out = ctx.row(c);
      double total = 0.0;
      for (std::int64_t t = 0; t < x.w; ++t) {
        total += in[t];
      }
      const auto mean = static_cast<float>(total / static_cast<double>(x.w));
      for (std::int64_t start = 0; start < x.w; start += kSegment) {
        const std::int64_t end = std::min(x.w, start + kSegment);
        double sum = 0.0;
        for (std::int64_t t = start; t < end; ++t) {
          sum += in[t];
        }
        const auto seg = static_cast<float>(sum / static_cast<double>(end - start));
        for (std::int64_t t = start; t < end; ++t) {
          out[t] = mean + seg;
        }
      }
    }
    return ctx;
  }
};

std::unique_ptr<CamPlusPlus> CamPlusPlus::Load(
    const std::filesystem::path& onnx_path, std::string* error) {
  auto set_error = [error](const std::string& message) {
    if (error != nullptr) {
      *error = message;
    }
  };
  std::string map_error;
  auto file = platform::ReadOnlyMappedFile::Open(onnx_path, &map_error);
  if (!file.valid()) {
    set_error("cannot map " + onnx_path.string() + ": " + map_error);
    return nullptr;
  }
  auto result = std::unique_ptr<CamPlusPlus>(new CamPlusPlus());
  result->impl_ = std::make_unique<Impl>();
  Impl& impl = *result->impl_;
  try {
    Wire model(file.data(), file.data() + file.size());
    bool found_graph = false;
    while (!model.done()) {
      const auto [field, type] = model.Key();
      if (field != 7 || type != 2) {
        model.Skip(type);
        continue;
      }
      found_graph = true;
      Wire graph = model.Bytes();
      while (!graph.done()) {
        const auto [gfield, gtype] = graph.Key();
        if (gfield == 1 && gtype == 2) {
          OnnxNode node = ParseNode(graph.Bytes());
          if (node.op == "Constant" && !node.outputs.empty()) {
            impl.constants[node.outputs[0]] = node.tensor;
          }
          if (!node.outputs.empty()) {
            impl.by_output[node.outputs[0]] = std::move(node);
          }
        } else if (gfield == 5 && gtype == 2) {
          std::string name;
          OnnxTensor tensor = ParseTensor(graph.Bytes(), &name);
          impl.initializers[name] = std::move(tensor);
        } else {
          graph.Skip(gtype);
        }
      }
    }
    if (!found_graph || impl.Node("/head/conv1/Conv_output_0") == nullptr ||
        impl.Node("output") == nullptr) {
      throw std::runtime_error("not a CAM++ graph");
    }
  } catch (const std::exception& exception) {
    set_error("campplus.onnx: " + std::string(exception.what()));
    return nullptr;
  }
  return result;
}

CamPlusPlus::~CamPlusPlus() = default;

std::vector<float> CamPlusPlus::Embed(std::span<const float> features,
                                      std::int64_t frames) const {
  const Impl& impl = *impl_;
  const auto conv = [&](const Feature& x, const std::string& path) {
    return Conv(x, impl.ConvOf(path + "/Conv_output_0"));
  };
  // FCM head on [1, 80 (freq), T]
  Feature x;
  x.c = 1;
  x.h = 80;
  x.w = frames;
  x.data.resize(static_cast<std::size_t>(80 * frames));
  for (std::int64_t t = 0; t < frames; ++t) {
    for (std::int64_t f = 0; f < 80; ++f) {
      x.data[static_cast<std::size_t>(f * frames + t)] =
          features[static_cast<std::size_t>(t * 80 + f)];
    }
  }
  x = conv(x, "/head/conv1");
  Relu(&x);
  for (int layer = 1; layer <= 2; ++layer) {
    for (int block = 0; block < 2; ++block) {
      const std::string base = "/head/layer" + std::to_string(layer) + "/layer" +
                               std::to_string(layer) + "." + std::to_string(block);
      Feature y = conv(x, base + "/conv1");
      Relu(&y);
      y = conv(y, base + "/conv2");
      if (impl.Node(base + "/shortcut/shortcut.0/Conv_output_0") != nullptr) {
        const Feature shortcut = conv(x, base + "/shortcut/shortcut.0");
        for (std::size_t i = 0; i < y.data.size(); ++i) {
          y.data[i] += shortcut.data[i];
        }
      } else {
        for (std::size_t i = 0; i < y.data.size(); ++i) {
          y.data[i] += x.data[i];
        }
      }
      Relu(&y);
      x = std::move(y);
    }
  }
  x = conv(x, "/head/conv2");
  Relu(&x);
  // (C, F, T) -> (C*F, T)
  x.c = x.c * x.h;
  x.h = 1;

  x = conv(x, "/xvector/tdnn/linear");
  Relu(&x);
  for (int block = 1;; ++block) {
    const std::string bname = "/xvector/block" + std::to_string(block);
    if (impl.Node(bname + "/tdnnd1/linear1/Conv_output_0") == nullptr) {
      break;
    }
    for (int layer = 1;; ++layer) {
      const std::string name = bname + "/tdnnd" + std::to_string(layer);
      if (impl.Node(name + "/linear1/Conv_output_0") == nullptr) {
        break;
      }
      Feature h = x;
      impl.BatchNorm(&h, name + "/nonlinear1/batchnorm/BatchNormalization_output_0");
      Relu(&h);
      h = conv(h, name + "/linear1");
      Relu(&h);
      const Feature local = conv(h, name + "/cam_layer/linear_local");
      Feature ctx = Impl::Context(h);
      ctx = conv(ctx, name + "/cam_layer/linear1");
      Relu(&ctx);
      ctx = conv(ctx, name + "/cam_layer/linear2");
      Feature out = local;
      for (std::size_t i = 0; i < out.data.size(); ++i) {
        out.data[i] *= 1.0F / (1.0F + std::exp(-ctx.data[i]));
      }
      // concat along channels
      x.data.insert(x.data.end(), out.data.begin(), out.data.end());
      x.c += out.c;
    }
    const std::string transit = "/xvector/transit" + std::to_string(block);
    impl.BatchNorm(&x, transit + "/nonlinear/batchnorm/BatchNormalization_output_0");
    Relu(&x);
    x = conv(x, transit + "/linear");
  }
  Relu(&x);  // out_nonlinear (its batchnorm is folded into the last conv)

  // statistics pooling: mean and unbiased std over time
  Feature stats;
  stats.c = 2 * x.c;
  stats.w = 1;
  stats.data.resize(static_cast<std::size_t>(stats.c));
  for (std::int64_t c = 0; c < x.c; ++c) {
    const float* row = x.row(c);
    double sum = 0.0;
    for (std::int64_t t = 0; t < x.w; ++t) {
      sum += row[t];
    }
    const double mean = sum / static_cast<double>(x.w);
    double sq = 0.0;
    for (std::int64_t t = 0; t < x.w; ++t) {
      sq += (row[t] - mean) * (row[t] - mean);
    }
    const double var = sq / static_cast<double>(x.w) *
                       static_cast<double>(x.w) / static_cast<double>(x.w - 1);
    stats.data[static_cast<std::size_t>(c)] = static_cast<float>(mean);
    stats.data[static_cast<std::size_t>(x.c + c)] = static_cast<float>(std::sqrt(var));
  }
  Feature embedding = conv(stats, "/xvector/dense/linear");
  impl.BatchNorm(&embedding, "output");
  return embedding.data;
}

std::vector<float> CamPlusPlus::EmbedWaveform(std::span<const float> mono,
                                              std::int64_t sample_rate) const {
  std::vector<float> wave16k;
  if (sample_rate != 16000) {
    wave16k = ResampleLikeTorchaudio(mono, sample_rate, 16000);
  } else {
    wave16k.assign(mono.begin(), mono.end());
  }
  std::int64_t frames = 0;
  std::vector<float> feat = KaldiFbank80(wave16k, &frames);
  if (frames < 2) {
    throw std::invalid_argument("reference audio is too short for CAM++");
  }
  for (std::int64_t f = 0; f < 80; ++f) {
    double sum = 0.0;
    for (std::int64_t t = 0; t < frames; ++t) {
      sum += feat[static_cast<std::size_t>(t * 80 + f)];
    }
    const auto mean = static_cast<float>(sum / static_cast<double>(frames));
    for (std::int64_t t = 0; t < frames; ++t) {
      feat[static_cast<std::size_t>(t * 80 + f)] -= mean;
    }
  }
  return Embed(feat, frames);
}

// ------------------------------------------------------------ resample ----

std::vector<float> ResampleLikeTorchaudio(std::span<const float> input,
                                          std::int64_t orig_rate,
                                          std::int64_t new_rate) {
  if (orig_rate <= 0 || new_rate <= 0) {
    throw std::invalid_argument("sample rates must be positive");
  }
  if (orig_rate == new_rate) {
    return {input.begin(), input.end()};
  }
  const std::int64_t gcd = std::gcd(orig_rate, new_rate);
  const std::int64_t orig = orig_rate / gcd;
  const std::int64_t target = new_rate / gcd;
  constexpr double kWidth = 6.0;
  constexpr double kRolloff = 0.99;
  const double base = static_cast<double>(std::min(orig, target)) * kRolloff;
  const auto width = static_cast<std::int64_t>(
      std::ceil(kWidth * static_cast<double>(orig) / base));
  const std::int64_t taps = 2 * width + orig;
  std::vector<float> kernel(static_cast<std::size_t>(target * taps));
  const double scale = base / static_cast<double>(orig);
  for (std::int64_t i = 0; i < target; ++i) {
    // torch.arange(0, -new, -1)[:, None, None] / new  (float32), promoted.
    const auto phase = static_cast<double>(
        static_cast<float>(-i) / static_cast<float>(target));
    for (std::int64_t k = 0; k < taps; ++k) {
      const double idx =
          static_cast<double>(k - width) / static_cast<double>(orig);
      double t = (phase + idx) * base;
      t = std::clamp(t, -kWidth, kWidth);
      const double c = std::cos(t * std::numbers::pi / kWidth / 2.0);
      const double window = c * c;
      t *= std::numbers::pi;
      const double sinc = t == 0.0 ? 1.0 : std::sin(t) / t;
      kernel[static_cast<std::size_t>(i * taps + k)] =
          static_cast<float>(sinc * window * scale);
    }
  }
  const auto length = static_cast<std::int64_t>(input.size());
  std::vector<float> padded(static_cast<std::size_t>(length + 2 * width + orig), 0.0F);
  std::copy(input.begin(), input.end(), padded.begin() + width);
  const std::int64_t blocks =
      (static_cast<std::int64_t>(padded.size()) - taps) / orig + 1;
  const auto target_length = static_cast<std::int64_t>(std::ceil(
      static_cast<double>(target) * static_cast<double>(length) /
      static_cast<double>(orig)));
  std::vector<float> out(static_cast<std::size_t>(blocks * target));
#pragma omp parallel for schedule(static)
  for (std::int64_t b = 0; b < blocks; ++b) {
    const float* src = padded.data() + b * orig;
    for (std::int64_t i = 0; i < target; ++i) {
      const float* k = kernel.data() + i * taps;
      float sum = 0.0F;
      for (std::int64_t j = 0; j < taps; ++j) {
        sum += src[j] * k[j];
      }
      out[static_cast<std::size_t>(b * target + i)] = sum;
    }
  }
  out.resize(static_cast<std::size_t>(std::min(target_length, blocks * target)));
  return out;
}

// --------------------------------------------------------------- fbank ----

std::vector<float> KaldiFbank80(std::span<const float> wave, std::int64_t* frames) {
  constexpr std::int64_t kShift = 160;
  constexpr std::int64_t kSize = 400;
  constexpr std::int64_t kPadded = 512;
  constexpr std::int64_t kBins = 80;
  constexpr std::int64_t kFftBins = kPadded / 2;  // mel banks cover 256 bins
  const auto samples = static_cast<std::int64_t>(wave.size());
  *frames = samples < kSize ? 0 : 1 + (samples - kSize) / kShift;
  std::vector<float> out(static_cast<std::size_t>(*frames * kBins));
  if (*frames == 0) {
    return out;
  }
  // povey window: hann(periodic=False) ** 0.85 in float32
  std::vector<float> window(kSize);
  for (std::int64_t i = 0; i < kSize; ++i) {
    const double hann = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi *
                                             static_cast<double>(i) /
                                             static_cast<double>(kSize - 1));
    window[static_cast<std::size_t>(i)] =
        std::pow(static_cast<float>(hann), 0.85F);
  }
  // mel banks (float32, like get_mel_banks)
  const auto mel_scale = [](float f) { return 1127.0F * std::log(1.0F + f / 700.0F); };
  const double mel_low = 1127.0 * std::log(1.0 + 20.0 / 700.0);
  const double mel_high = 1127.0 * std::log(1.0 + 8000.0 / 700.0);
  const double delta = (mel_high - mel_low) / static_cast<double>(kBins + 1);
  std::vector<float> banks(static_cast<std::size_t>(kBins * kFftBins), 0.0F);
  for (std::int64_t b = 0; b < kBins; ++b) {
    const float left = static_cast<float>(b) * static_cast<float>(delta) +
                       static_cast<float>(mel_low);
    const float center = (static_cast<float>(b) + 1.0F) * static_cast<float>(delta) +
                         static_cast<float>(mel_low);
    const float right = (static_cast<float>(b) + 2.0F) * static_cast<float>(delta) +
                        static_cast<float>(mel_low);
    for (std::int64_t k = 0; k < kFftBins; ++k) {
      const float mel = mel_scale(31.25F * static_cast<float>(k));
      const float up = (mel - left) / (center - left);
      const float down = (right - mel) / (right - center);
      banks[static_cast<std::size_t>(b * kFftBins + k)] =
          std::max(0.0F, std::min(up, down));
    }
  }
  const float epsilon = std::numeric_limits<float>::epsilon();
#pragma omp parallel
  {
    const cpu::ComplexFft fft(kPadded, -1);
    std::vector<std::complex<double>> in(kPadded), spectrum(kPadded);
    std::vector<float> frame(kSize);
    std::vector<float> power(kFftBins + 1);
#pragma omp for schedule(static)
    for (std::int64_t f = 0; f < *frames; ++f) {
      const float* src = wave.data() + f * kShift;
      double sum = 0.0;
      for (std::int64_t i = 0; i < kSize; ++i) {
        sum += src[i];
      }
      const auto mean = static_cast<float>(sum / static_cast<double>(kSize));
      for (std::int64_t i = 0; i < kSize; ++i) {
        frame[static_cast<std::size_t>(i)] = src[i] - mean;
      }
      // pre-emphasis 0.97 with replicate padding, then window
      for (std::int64_t i = kSize - 1; i >= 0; --i) {
        const float previous = frame[static_cast<std::size_t>(i > 0 ? i - 1 : 0)];
        frame[static_cast<std::size_t>(i)] -= 0.97F * previous;
      }
      for (std::int64_t i = 0; i < kPadded; ++i) {
        const float value = i < kSize ? frame[static_cast<std::size_t>(i)] *
                                            window[static_cast<std::size_t>(i)]
                                      : 0.0F;
        in[static_cast<std::size_t>(i)] = std::complex<double>(value, 0.0);
      }
      fft.Run(in.data(), spectrum.data());
      for (std::int64_t k = 0; k <= kFftBins; ++k) {
        power[static_cast<std::size_t>(k)] =
            static_cast<float>(std::norm(spectrum[static_cast<std::size_t>(k)]));
      }
      for (std::int64_t b = 0; b < kBins; ++b) {
        float energy = 0.0F;
        for (std::int64_t k = 0; k < kFftBins; ++k) {
          energy += power[static_cast<std::size_t>(k)] *
                    banks[static_cast<std::size_t>(b * kFftBins + k)];
        }
        out[static_cast<std::size_t>(f * kBins + b)] = std::log(std::max(energy, epsilon));
      }
    }
  }
  return out;
}

}  // namespace gufo::models::ming_tts
