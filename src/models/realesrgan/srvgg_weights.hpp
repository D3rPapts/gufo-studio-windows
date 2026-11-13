// Safetensors header of a converted Real-ESRGAN SRVGGNetCompact file and
// the architecture it describes (host side, no HIP): the server checks the
// weights at start-up and names the variant (general / wdn / anime) before
// any device code loads them. tools/upscale/convert_realesrgan_pth.py writes
// the files; their tensors are
//   body.0.weight [F, 3, 3, 3]  body.0.bias [F]  body.1.weight [F]  (PReLU)
//   body.2k.weight [F, F, 3, 3] body.2k.bias [F] body.2k+1.weight [F]
//   body.2(N+1).weight [3 S^2, F, 3, 3] body.2(N+1).bias [3 S^2]
// for N body convolutions, F features and the pixel-shuffle scale S
// (realesr-general-x4v3 / -wdn-x4v3: N = 32, realesr-animevideov3: N = 16;
// F = 64, S = 4 for all three).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_REALESRGAN_SRVGG_WEIGHTS_HPP_
#define GUFO_MODELS_REALESRGAN_SRVGG_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::realesrgan {

struct SafetensorsTensor {
  std::string dtype;  // "F32", "F16", "BF16", ...
  std::vector<std::size_t> shape;
  std::size_t begin{0};  // byte range in the data block
  std::size_t end{0};
  [[nodiscard]] std::size_t count() const noexcept;
};

struct SafetensorsHeader {
  std::uint64_t header_bytes{0};  // JSON length (the 8-byte prefix)
  std::uint64_t data_bytes{0};    // file size - 8 - header_bytes
  std::map<std::string, std::string> metadata;  // "__metadata__"
  std::map<std::string, SafetensorsTensor> tensors;
};

/// Parses the JSON header text of a safetensors file whose data block holds
/// `data_bytes`. Every tensor needs dtype, shape and data_offsets inside the
/// block; the element count must match the extent for the known dtypes.
[[nodiscard]] bool ParseSafetensorsHeader(std::string_view json_text,
                                          std::uint64_t data_bytes,
                                          SafetensorsHeader* header,
                                          std::string* error);

/// Reads the 8-byte length and the JSON header of `path` (not the data).
[[nodiscard]] bool ReadSafetensorsHeader(const std::filesystem::path& path,
                                         SafetensorsHeader* header,
                                         std::string* error);

/// Bytes per element of a safetensors dtype; 0 when unknown.
[[nodiscard]] std::size_t SafetensorsElementBytes(std::string_view dtype);

struct SrvggArchitecture {
  int in_channels{0};
  int out_channels{0};  // 3
  int features{0};      // F
  int convolutions{0};  // N (body convolutions between the first and last)
  int scale{0};         // S
  std::size_t parameters{0};
  std::string dtype;
};

/// Derives the architecture from the tensor names and shapes (body.0 ...
/// body.2(N+1)) and checks it against the __metadata__ (num_conv, num_feat,
/// upscale) when present.
[[nodiscard]] bool DescribeSrvgg(const SafetensorsHeader& header,
                                 SrvggArchitecture* architecture,
                                 std::string* error);

/// The public variant of a Real-ESRGAN v3 file by its name: "wdn"
/// (realesr-general-wdn-x4v3), "general" (realesr-general-x4v3), "anime"
/// (realesr-animevideov3); "" for anything else.
[[nodiscard]] std::string VariantFromFileName(std::string_view name);

}  // namespace gufo::models::realesrgan

#endif  // GUFO_MODELS_REALESRGAN_SRVGG_WEIGHTS_HPP_
