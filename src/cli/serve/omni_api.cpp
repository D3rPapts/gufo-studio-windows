#include "src/cli/serve/omni_api.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "src/cli/serve/audio_stream.hpp"
#include "src/core/json.hpp"

namespace gufo::server {
namespace {

namespace omni = models::qwen3_omni;

constexpr std::string_view kSchema = "gufo.omni-chat.v1";
constexpr std::size_t kMaximumAudioBytes = static_cast<std::size_t>(48) << 20U;

HttpResponse Error(int status, std::string reason, std::string message,
                   std::string code) {
  json::Value root = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = std::move(message);
  error["type"] = status >= 500 ? "server_error" : "invalid_request_error";
  error["code"] = std::move(code);
  root["error"] = std::move(error);
  return {
      .status = status,
      .reason = std::move(reason),
      .body = root.dump(),
      .headers = {},
      .streaming_body = {},
      .log_details = {},
  };
}

std::int64_t Now() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

/// The single `input_audio` part of the last user message, or an error.
bool FindAudio(const json::Value& messages, const json::Value** audio,
               std::string* error) {
  const json::Value* user = nullptr;
  for (const json::Value& message : messages.items()) {
    if (message.is_object() && message.member_str("role") == "user") {
      user = &message;
    }
  }
  if (user == nullptr) {
    *error = "messages need a user message with an input_audio part";
    return false;
  }
  const json::Value* content = user->find("content");
  if (content == nullptr || !content->is_array()) {
    *error = "the user message content must be an array with input_audio";
    return false;
  }
  *audio = nullptr;
  for (const json::Value& part : content->items()) {
    if (!part.is_object() || part.member_str("type") != "input_audio") {
      continue;
    }
    if (*audio != nullptr) {
      *error = "only one input_audio part per request is supported";
      return false;
    }
    *audio = part.find("input_audio");
  }
  if (*audio == nullptr || !(*audio)->is_object()) {
    *error = "the user message has no input_audio part";
    return false;
  }
  return true;
}

json::Value Usage(const omni::CaptionResult& result) {
  json::Value usage = json::Value::object();
  usage["prompt_tokens"] = result.prompt_tokens;
  usage["completion_tokens"] = result.completion_tokens;
  usage["total_tokens"] = result.prompt_tokens + result.completion_tokens;
  return usage;
}

/// llama-server's `timings` block, so existing benchmark scripts keep working.
json::Value Timings(const omni::CaptionResult& result) {
  const auto per_second = [](std::size_t count, double milliseconds) {
    return milliseconds > 0.0
               ? 1000.0 * static_cast<double>(count) / milliseconds
               : 0.0;
  };
  json::Value timings = json::Value::object();
  timings["audio_seconds"] = result.audio_seconds;
  timings["audio_tokens"] = result.audio_tokens;
  timings["encoder_ms"] = result.encoder_ms;
  timings["prompt_n"] = result.prompt_tokens;
  timings["prompt_ms"] = result.prefill_ms;
  timings["prompt_per_second"] =
      per_second(result.prompt_tokens, result.prefill_ms);
  timings["predicted_n"] = result.completion_tokens;
  timings["predicted_ms"] = result.decode_ms;
  timings["predicted_per_second"] =
      per_second(result.completion_tokens, result.decode_ms);
  return timings;
}

std::string LogDetails(const OmniService& service,
                       const omni::CaptionResult& result) {
  return "model=" + service.model_id() +
         " audio_ms=" +
         std::to_string(static_cast<std::int64_t>(result.audio_seconds * 1000)) +
         " audio_tokens=" + std::to_string(result.audio_tokens) +
         " prompt_tokens=" + std::to_string(result.prompt_tokens) +
         " generated_tokens=" + std::to_string(result.completion_tokens) +
         " encoder_ms=" +
         std::to_string(static_cast<std::int64_t>(result.encoder_ms)) +
         " prefill_ms=" +
         std::to_string(static_cast<std::int64_t>(result.prefill_ms)) +
         " decode_ms=" +
         std::to_string(static_cast<std::int64_t>(result.decode_ms)) +
         " finish=" + result.finish_reason;
}

json::Value Chunk(const std::string& id, std::int64_t created,
                  const std::string& model, json::Value delta,
                  const std::string& finish_reason) {
  json::Value choice = json::Value::object();
  choice["index"] = 0;
  choice["delta"] = std::move(delta);
  if (finish_reason.empty()) {
    choice["finish_reason"] = json::Value();
  } else {
    choice["finish_reason"] = finish_reason;
  }
  json::Value choices = json::Value::array();
  choices.push_back(std::move(choice));
  json::Value chunk = json::Value::object();
  chunk["id"] = id;
  chunk["object"] = "chat.completion.chunk";
  chunk["created"] = created;
  chunk["model"] = model;
  chunk["choices"] = std::move(choices);
  return chunk;
}

HttpResponse ChatCompletions(const HttpRequest& request, OmniService& service) {
  json::Value body;
  try {
    body = json::parse(request.body);
  } catch (const std::exception& exception) {
    return Error(400, "Bad Request",
                 std::string("invalid JSON: ") + exception.what(),
                 "invalid_json");
  }
  if (!body.is_object()) {
    return Error(400, "Bad Request", "request body must be a JSON object",
                 "invalid_request");
  }
  const json::Value* messages = body.find("messages");
  if (messages == nullptr || !messages->is_array()) {
    return Error(400, "Bad Request", "'messages' must be an array",
                 "invalid_messages");
  }
  const json::Value* audio = nullptr;
  std::string error;
  if (!FindAudio(*messages, &audio, &error)) {
    return Error(400, "Bad Request", std::move(error), "missing_audio");
  }
  const std::string format = audio->member_str("format", "wav");
  if (format != "wav") {
    return Error(400, "Bad Request", "input_audio format must be wav",
                 "unsupported_audio_format");
  }
  const json::Value* data = audio->find("data");
  auto wav = std::make_shared<std::string>();
  if (data == nullptr || !data->is_string() ||
      !DecodeAudioBase64(data->str(), kMaximumAudioBytes, wav.get()) ||
      wav->empty()) {
    return Error(400, "Bad Request",
                 "input_audio data must be canonical base64 WAV up to 48 MiB",
                 "invalid_audio");
  }

  omni::CaptionRequest caption;
  caption.max_tokens = service.max_tokens();
  for (const char* key : {"max_tokens", "max_completion_tokens"}) {
    if (const json::Value* value = body.find(key);
        value != nullptr && !value->is_null()) {
      const std::size_t parsed = value->as_size(0);
      if (parsed == 0) {
        return Error(400, "Bad Request",
                     std::string(key) + " must be a positive integer",
                     "invalid_max_tokens");
      }
      caption.max_tokens = std::min(parsed, service.max_tokens());
    }
  }
  caption.temperature =
      static_cast<float>(body.member_double("temperature", 0.0));
  caption.top_p = static_cast<float>(body.member_double("top_p", 0.95));
  caption.top_k = body.member_size("top_k", 20);
  caption.seed = body.member_size("seed", 1234);
  if (!std::isfinite(caption.temperature) || caption.temperature < 0.0F ||
      !std::isfinite(caption.top_p) || caption.top_p <= 0.0F ||
      caption.top_p > 1.0F) {
    return Error(400, "Bad Request",
                 "temperature must be >= 0 and top_p in (0, 1]",
                 "invalid_sampling");
  }
  bool stream = false;
  if (const json::Value* value = body.find("stream"); value != nullptr) {
    if (!value->is_bool() && !value->is_null()) {
      return Error(400, "Bad Request", "'stream' must be a boolean",
                   "invalid_parameter_type");
    }
    stream = value->as_bool(false);
  }
  bool include_usage = false;
  if (const json::Value* options = body.find("stream_options");
      options != nullptr && options->is_object()) {
    if (const json::Value* value = options->find("include_usage")) {
      include_usage = value->as_bool(false);
    }
  }

  const std::string id = "chatcmpl-" + request.request_id;
  const std::int64_t created = Now();
  const std::string model = service.model_id();

  if (stream) {
    HttpResponse response;
    response.headers = {{"Content-Type", "text/event-stream"},
                        {"Cache-Control", "no-cache"},
                        {"X-Accel-Buffering", "no"},
                        {"X-Gufo-Schema", std::string(kSchema)}};
    response.stream_log = std::make_shared<HttpResponse::StreamLog>();
    response.streaming_body =
        [&service, wav, caption, include_usage, id, created, model,
         cancelled = request.is_cancelled,
         log = response.stream_log](const HttpResponse::BodyWriter& write) {
          bool connected = true;
          const auto send = [&](const json::Value& event) {
            connected = connected && write("data: " + event.dump() + "\n\n");
            return connected;
          };
          json::Value role = json::Value::object();
          role["role"] = "assistant";
          role["content"] = "";
          if (!send(Chunk(id, created, model, std::move(role), ""))) {
            return;
          }
          omni::CaptionRequest input = caption;
          input.wav = std::as_bytes(std::span(wav->data(), wav->size()));
          input.on_text = [&](std::string_view piece) {
            json::Value delta = json::Value::object();
            delta["content"] = std::string(piece);
            return send(Chunk(id, created, model, std::move(delta), ""));
          };
          const auto stop = [&] {
            return !connected || (cancelled && cancelled());
          };
          omni::CaptionResult result;
          std::string failure;
          if (!service.Caption(input, stop, &result, &failure)) {
            log->error_code = stop() ? "cancelled" : "caption_failed";
            if (!stop()) {
              json::Value error = json::Value::object();
              error["message"] = failure;
              error["type"] = result.invalid_request ? "invalid_request_error"
                                                     : "server_error";
              error["code"] = log->error_code;
              json::Value event = json::Value::object();
              event["error"] = std::move(error);
              log->error_event_sent = send(event);
            }
            return;
          }
          if (stop()) {
            log->error_code = "cancelled";
            return;
          }
          json::Value last = Chunk(id, created, model, json::Value::object(),
                                   result.finish_reason);
          last["timings"] = Timings(result);
          if (!send(last)) {
            return;
          }
          if (include_usage) {
            json::Value usage = Chunk(id, created, model,
                                      json::Value::object(), "");
            usage["choices"] = json::Value::array();
            usage["usage"] = Usage(result);
            if (!send(usage)) {
              return;
            }
          }
          (void)write("data: [DONE]\n\n");
          log->details = LogDetails(service, result);
        };
    return response;
  }

  caption.wav = std::as_bytes(std::span(wav->data(), wav->size()));
  omni::CaptionResult result;
  if (!service.Caption(caption, request.is_cancelled, &result, &error)) {
    if (request.is_cancelled && request.is_cancelled()) {
      return Error(499, "Client Closed Request", "caption cancelled",
                   "cancelled");
    }
    if (result.invalid_request) {
      return Error(400, "Bad Request", std::move(error), "invalid_audio");
    }
    return Error(500, "Internal Server Error", std::move(error),
                 "caption_failed");
  }
  json::Value message = json::Value::object();
  message["role"] = "assistant";
  message["content"] = result.text;
  json::Value choice = json::Value::object();
  choice["index"] = 0;
  choice["message"] = std::move(message);
  choice["finish_reason"] = result.finish_reason;
  json::Value choices = json::Value::array();
  choices.push_back(std::move(choice));
  json::Value root = json::Value::object();
  root["id"] = id;
  root["object"] = "chat.completion";
  root["created"] = created;
  root["model"] = model;
  root["choices"] = std::move(choices);
  root["usage"] = Usage(result);
  root["timings"] = Timings(result);

  HttpResponse response;
  response.body = root.dump();
  // The server adds Content-Type: application/json; charset=utf-8.
  response.headers = {{"X-Gufo-Schema", std::string(kSchema)}};
  response.log_details = LogDetails(service, result);
  return response;
}

}  // namespace

bool IsOmniApiPath(std::string_view path) noexcept {
  return path == "/v1/chat/completions" || path == "/chat/completions";
}

HttpResponse HandleOmniApiRequest(const HttpRequest& request,
                                  OmniService& service) {
  if (request.method == "POST") {
    return ChatCompletions(request, service);
  }
  return Error(405, "Method Not Allowed",
               "HTTP method is not supported for this endpoint",
               "method_not_allowed");
}

}  // namespace gufo::server
