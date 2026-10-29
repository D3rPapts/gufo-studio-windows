#ifndef GUFO_SERVER_OMNI_API_HPP_
#define GUFO_SERVER_OMNI_API_HPP_

#include <string_view>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/omni_service.hpp"

namespace gufo::server {

/// OpenAI Chat Completions for the captioner: the last user message carries
/// one `input_audio` part (base64 WAV), as llama-server accepts it. Text parts
/// are ignored because the captioner takes no instruction.
[[nodiscard]] bool IsOmniApiPath(std::string_view path) noexcept;
[[nodiscard]] HttpResponse HandleOmniApiRequest(const HttpRequest& request,
                                                OmniService& service);

}  // namespace gufo::server

#endif  // GUFO_SERVER_OMNI_API_HPP_
