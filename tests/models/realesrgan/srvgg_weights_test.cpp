// CPU tests for the safetensors header parser and the SRVGGNetCompact
// architecture check (srvgg_weights.hpp): synthetic headers of the three
// Real-ESRGAN v3 files (general / wdn: 32 body convolutions, anime: 16),
// the file on disk when GUFO_REALESRGAN_WEIGHTS points at one, and the
// variant names.
// Written with AI assistance (Claude Code).
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/realesrgan/srvgg_weights.hpp"

namespace esr = gufo::models::realesrgan;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
  if (!condition) {
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
  }
}

/// Header JSON of a SRVGGNetCompact with `convolutions` body convolutions,
/// `features` features, `scale` pixel shuffle; `data_bytes` receives the
/// data block size. `metadata` is inserted verbatim ("" for none).
std::string SrvggHeader(int convolutions, int features, int scale,
                        const std::string& metadata, std::uint64_t* data_bytes,
                        const std::string& dtype = "F32") {
  const std::size_t element = dtype == "F32" ? 4 : 2;
  std::string out = "{";
  if (!metadata.empty())
    out += "\"__metadata__\":" + metadata + ",";
  std::uint64_t offset = 0;
  const auto tensor = [&](const std::string& name,
                          const std::vector<std::size_t>& shape) {
    std::size_t count = 1;
    std::string dims;
    for (std::size_t d : shape) {
      count *= d;
      dims += (dims.empty() ? "" : ",") + std::to_string(d);
    }
    const std::uint64_t end = offset + count * element;
    out += "\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":[" + dims +
           "],\"data_offsets\":[" + std::to_string(offset) + "," +
           std::to_string(end) + "]},";
    offset = end;
  };
  const auto f = static_cast<std::size_t>(features);
  tensor("body.0.weight", {f, 3, 3, 3});
  tensor("body.0.bias", {f});
  tensor("body.1.weight", {f});
  for (int i = 1; i <= convolutions; ++i) {
    tensor("body." + std::to_string(2 * i) + ".weight", {f, f, 3, 3});
    tensor("body." + std::to_string(2 * i) + ".bias", {f});
    tensor("body." + std::to_string(2 * i + 1) + ".weight", {f});
  }
  const auto last = static_cast<std::size_t>(3 * scale * scale);
  tensor("body." + std::to_string(2 * (convolutions + 1)) + ".weight",
         {last, f, 3, 3});
  tensor("body." + std::to_string(2 * (convolutions + 1)) + ".bias", {last});
  out.back() = '}';
  *data_bytes = offset;
  return out;
}

void TestGeneral() {
  std::uint64_t bytes = 0;
  const std::string text = SrvggHeader(
      32, 64, 4,
      "{\"format\":\"realesrgan-srvgg\",\"num_conv\":\"32\","
      "\"num_feat\":\"64\",\"upscale\":\"4\",\"source\":\"x.pth\"}",
      &bytes);
  esr::SafetensorsHeader header;
  std::string error;
  Expect(esr::ParseSafetensorsHeader(text, bytes, &header, &error),
         "general: header parses (" + error + ")");
  Expect(header.tensors.size() == 2 + 1 + 32 * 3 + 2,
         "general: 103 tensors, got " + std::to_string(header.tensors.size()));
  Expect(header.metadata.at("format") == "realesrgan-srvgg",
         "general: metadata format");
  Expect(header.tensors.at("body.66.weight").shape ==
             std::vector<std::size_t>{48, 64, 3, 3},
         "general: last convolution shape [48, 64, 3, 3]");
  esr::SrvggArchitecture arch;
  Expect(esr::DescribeSrvgg(header, &arch, &error),
         "general: architecture (" + error + ")");
  Expect(arch.convolutions == 32 && arch.features == 64 && arch.scale == 4 &&
             arch.in_channels == 3 && arch.out_channels == 3 &&
             arch.dtype == "F32",
         "general: 32 convolutions, 64 features, x4, RGB");
  // 3*64*9+64 + 64 + 32*(64*64*9+64+64) + 48*64*9+48 = 1 213 296 parameters
  // (realesr-general-x4v3.safetensors: 4 853 184 data bytes / 4).
  Expect(arch.parameters == 1213296,
         "general: parameter count " + std::to_string(arch.parameters));
  Expect(bytes == arch.parameters * 4, "general: data block is F32 x params");
}

void TestAnime() {
  std::uint64_t bytes = 0;
  const std::string text = SrvggHeader(16, 64, 4, "", &bytes);
  esr::SafetensorsHeader header;
  std::string error;
  Expect(esr::ParseSafetensorsHeader(text, bytes, &header, &error),
         "anime: header parses (" + error + ")");
  esr::SrvggArchitecture arch;
  Expect(esr::DescribeSrvgg(header, &arch, &error),
         "anime: architecture without metadata (" + error + ")");
  Expect(arch.convolutions == 16 && arch.scale == 4 && arch.features == 64,
         "anime: 16 convolutions");
  // realesr-animevideov3.safetensors: 2 485 696 data bytes / 4.
  Expect(arch.parameters == 621424,
         "anime: parameter count " + std::to_string(arch.parameters));
}

void TestMetadataMismatch() {
  std::uint64_t bytes = 0;
  const std::string text =
      SrvggHeader(16, 64, 4, "{\"num_conv\":\"32\"}", &bytes);
  esr::SafetensorsHeader header;
  std::string error;
  Expect(esr::ParseSafetensorsHeader(text, bytes, &header, &error),
         "mismatch: header parses");
  esr::SrvggArchitecture arch;
  Expect(!esr::DescribeSrvgg(header, &arch, &error),
         "mismatch: num_conv 32 against 16 convolutions is rejected");
  Expect(error.find("num_conv") != std::string::npos,
         "mismatch: error names num_conv (" + error + ")");
}

void TestBrokenHeaders() {
  esr::SafetensorsHeader header;
  std::string error;
  Expect(!esr::ParseSafetensorsHeader("not json", 0, &header, &error),
         "broken: non-JSON is rejected");
  Expect(!esr::ParseSafetensorsHeader("[1,2]", 0, &header, &error),
         "broken: array is rejected");
  // Extent beyond the data block.
  Expect(!esr::ParseSafetensorsHeader(
             "{\"body.0.weight\":{\"dtype\":\"F32\",\"shape\":[2],"
             "\"data_offsets\":[0,8]}}",
             4, &header, &error),
         "broken: extent past the data block is rejected");
  // Extent not matching the shape.
  Expect(!esr::ParseSafetensorsHeader(
             "{\"body.0.weight\":{\"dtype\":\"F32\",\"shape\":[2],"
             "\"data_offsets\":[0,4]}}",
             8, &header, &error),
         "broken: extent not matching the shape is rejected");
  // Missing data_offsets.
  Expect(!esr::ParseSafetensorsHeader(
             "{\"a\":{\"dtype\":\"F32\",\"shape\":[1]}}", 4, &header, &error),
         "broken: missing data_offsets is rejected");
  // A valid header whose tensors are not a SRVGG body.
  Expect(esr::ParseSafetensorsHeader(
             "{\"encoder.weight\":{\"dtype\":\"F16\",\"shape\":[2,2],"
             "\"data_offsets\":[0,8]}}",
             8, &header, &error),
         "foreign: valid header parses");
  esr::SrvggArchitecture arch;
  Expect(!esr::DescribeSrvgg(header, &arch, &error),
         "foreign: non-body tensors are rejected");
  // A body that ends in a PReLU (odd last index) is incomplete.
  Expect(esr::ParseSafetensorsHeader(
             "{\"body.0.weight\":{\"dtype\":\"F32\",\"shape\":[64,3,3,3],"
             "\"data_offsets\":[0,6912]},"
             "\"body.0.bias\":{\"dtype\":\"F32\",\"shape\":[64],"
             "\"data_offsets\":[6912,7168]},"
             "\"body.1.weight\":{\"dtype\":\"F32\",\"shape\":[64],"
             "\"data_offsets\":[7168,7424]}}",
             7424, &header, &error),
         "truncated: header parses");
  Expect(!esr::DescribeSrvgg(header, &arch, &error),
         "truncated: body ending in PReLU is rejected");
  // Unknown dtype: the extent is not checked against the shape.
  Expect(esr::ParseSafetensorsHeader(
             "{\"body.0.weight\":{\"dtype\":\"Q4\",\"shape\":[64,3,3,3],"
             "\"data_offsets\":[0,100]}}",
             100, &header, &error),
         "unknown dtype: extent accepted");
  Expect(esr::SafetensorsElementBytes("F32") == 4 &&
             esr::SafetensorsElementBytes("BF16") == 2 &&
             esr::SafetensorsElementBytes("Q4") == 0,
         "element sizes");
}

void TestVariants() {
  Expect(esr::VariantFromFileName("realesr-general-x4v3") == "general",
         "variant general");
  Expect(esr::VariantFromFileName("realesr-general-wdn-x4v3") == "wdn",
         "variant wdn");
  Expect(esr::VariantFromFileName("realesr-animevideov3") == "anime",
         "variant anime");
  Expect(esr::VariantFromFileName("RealESR-AnimeVideoV3.safetensors") == "anime",
         "variant is case-insensitive");
  Expect(esr::VariantFromFileName("rife49").empty(), "variant unknown");
}

// With GUFO_REALESRGAN_WEIGHTS=<file> the real header is read (optional).
void TestFileOnDisk() {
  const char* path = std::getenv("GUFO_REALESRGAN_WEIGHTS");
  if (path == nullptr || *path == '\0')
    return;
  esr::SafetensorsHeader header;
  std::string error;
  Expect(esr::ReadSafetensorsHeader(path, &header, &error),
         std::string("file: header reads (") + error + ")");
  esr::SrvggArchitecture arch;
  Expect(esr::DescribeSrvgg(header, &arch, &error),
         "file: architecture (" + error + ")");
  std::cout << "file " << path << ": " << arch.convolutions
            << " convolutions, " << arch.features << " features, x"
            << arch.scale << ", " << arch.parameters << " parameters, "
            << arch.dtype << ", variant '"
            << esr::VariantFromFileName(
                   std::filesystem::path(path).stem().string())
            << "'\n";
}

}  // namespace

int main() {
  TestGeneral();
  TestAnime();
  TestMetadataMismatch();
  TestBrokenHeaders();
  TestVariants();
  TestFileOnDisk();
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "realesrgan srvgg_weights: all tests passed\n";
  return 0;
}
