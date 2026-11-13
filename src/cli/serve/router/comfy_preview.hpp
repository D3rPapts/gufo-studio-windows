#ifndef GUFO_CLI_SERVE_ROUTER_COMFY_PREVIEW_HPP_
#define GUFO_CLI_SERVE_ROUTER_COMFY_PREVIEW_HPP_

// ComfyUI /ws binary preview frames (server.py BinaryEventTypes), all
// integers big-endian u32:
//   PREVIEW_IMAGE (1):               [1][format: 1 JPEG, 2 PNG][image]
//   PREVIEW_IMAGE_WITH_METADATA (4): [4][json length][json][image]
// Current frontends paint node previews from type 4 only and announce it
// with {"type": "feature_flags", "data": {"supports_preview_metadata": true}};
// type 1 is for clients that did not.

#include <cstdint>
#include <string>
#include <string_view>

namespace gufo::router::comfy {

inline constexpr std::uint32_t kPreviewImage = 1;
inline constexpr std::uint32_t kPreviewImageWithMetadata = 4;
inline constexpr std::uint32_t kPreviewJpeg = 1;
inline constexpr std::uint32_t kPreviewPng = 2;

inline void AppendBigEndian(std::string* out, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    out->push_back(static_cast<char>((value >> shift) & 0xFF));
}

/// Type 1 frame around a JPEG (or PNG) image.
inline std::string PreviewImageFrame(std::string_view image,
                                     std::uint32_t format = kPreviewJpeg) {
  std::string frame;
  frame.reserve(8 + image.size());
  AppendBigEndian(&frame, kPreviewImage);
  AppendBigEndian(&frame, format);
  frame.append(image);
  return frame;
}

/// Type 4 frame; `metadata` is the JSON object ComfyUI sends (node_id,
/// prompt_id, display_node_id, parent_node_id, real_node_id, image_type).
inline std::string PreviewMetadataFrame(std::string_view image,
                                        std::string_view metadata) {
  std::string frame;
  frame.reserve(8 + metadata.size() + image.size());
  AppendBigEndian(&frame, kPreviewImageWithMetadata);
  AppendBigEndian(&frame, static_cast<std::uint32_t>(metadata.size()));
  frame.append(metadata);
  frame.append(image);
  return frame;
}

/// A worker preview id ([A-Za-z0-9._-], at most 128 characters) for a
/// ComfyUI prompt and node.
inline std::string PreviewId(std::string_view prompt_id,
                             std::string_view node_id) {
  std::string id;
  for (const char c : std::string(prompt_id) + "_" + std::string(node_id)) {
    const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
    id.push_back(ok ? c : '_');
  }
  if (id.size() > 128)
    id.resize(128);
  return id;
}

}  // namespace gufo::router::comfy

#endif  // GUFO_CLI_SERVE_ROUTER_COMFY_PREVIEW_HPP_
