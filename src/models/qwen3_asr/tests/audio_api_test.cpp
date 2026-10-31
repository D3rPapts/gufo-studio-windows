#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/audio_asr_api.hpp"
#include "src/models/qwen3_asr/audio.hpp"

namespace {

namespace qwen3_asr = gufo::models::qwen3_asr;
using gufo::server::AsrService;
using gufo::server::AsrServiceOptions;
using gufo::server::HandleAudioAsrApiRequest;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL qwen3_asr_audio_api_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

class FakeRunner {
public:
  bool Run(const qwen3_asr::TranscriptionRequest& request,
           const qwen3_asr::CancellationCheck&,
           qwen3_asr::TranscriptionResult* result, std::string* error) {
    ++calls;
    wav_bytes = request.wav.size();
    context = request.context;
    language = request.language;
    maximum_tokens = request.max_new_tokens;
    result->language = "English";
    result->text = "A native transcript.";
    result->audio_samples = 32000;
    result->audio_tokens = 25;
    result->generated_ids = {1, 2, 3};
    result->timings.audio_encoder_ms = 10.0;
    result->timings.text_decoder_ms = 20.0;
    if (!request.pcm16k.empty()) {
      pcm_samples.push_back(request.pcm16k.size());
      if (!segment_texts.empty()) {
        result->text = segment_texts[(pcm_samples.size() - 1U) %
                                     segment_texts.size()];
      }
    }
    if (request.on_text && !request.on_text("A native")) {
      if (error != nullptr)
        *error = "cancelled";
      return false;
    }
    return true;
  }

  int calls{0};
  std::size_t wav_bytes{0};
  std::string context;
  std::optional<std::string> language;
  std::size_t maximum_tokens{0};
  std::vector<std::size_t> pcm_samples;
  std::vector<std::string> segment_texts;
};

AsrService MakeService(FakeRunner* runner) {
  return AsrService(AsrServiceOptions{
      .model_root = {},
      .validate_model = false,
      .model_id = "qwen3-asr-1.7b",
      .runner =
          [runner](const qwen3_asr::TranscriptionRequest& request,
                   const qwen3_asr::CancellationCheck& cancelled,
                   qwen3_asr::TranscriptionResult* result, std::string* error) {
            return runner->Run(request, cancelled, result, error);
          },
  });
}

std::string Multipart(std::string response_format = "json",
                      bool include_unknown = false,
                      std::string_view stream = {},
                      std::string_view file = "RIFFfake",
                      std::string_view granularity = {}) {
  constexpr std::string_view boundary = "gufo-test-boundary";
  std::string body;
  const auto append = [&](std::string_view name, std::string_view value,
                          std::string_view filename = {}) {
    body += "--";
    body += boundary;
    body += "\r\nContent-Disposition: form-data; name=\"";
    body += name;
    body += '"';
    if (!filename.empty()) {
      body += "; filename=\"";
      body += filename;
      body += '"';
    }
    body += "\r\n\r\n";
    body.append(value.data(), value.size());
    body += "\r\n";
  };
  append("file", file, "sample.wav");
  append("model", "qwen3-asr");
  append("language", "en");
  append("prompt", "Names: Gufo.");
  append("response_format", response_format);
  append("max_tokens", "42");
  append("temperature", "0");
  if (!stream.empty())
    append("stream", stream);
  if (!granularity.empty())
    append("timestamp_granularities[]", granularity);
  if (include_unknown) {
    append("typo", "true");
  }
  body += "--";
  body += boundary;
  body += "--\r\n";
  return body;
}

HttpResponse Send(AsrService& service, std::string body,
                  std::string content_type =
                      "multipart/form-data; boundary=gufo-test-boundary") {
  return HandleAudioAsrApiRequest(
      HttpRequest{
          .method = "POST",
          .path = "/v1/audio/transcriptions",
          .query = {},
          .body = std::move(body),
          .headers = {{"Content-Type", std::move(content_type)}},
          .is_cancelled = {},
      },
      service);
}

bool HasHeader(const HttpResponse& response, std::string_view name,
               std::string_view value) {
  return std::ranges::any_of(response.headers, [&](const auto& header) {
    return header.first == name && header.second == value;
  });
}

void TestJsonRequest() {
  FakeRunner runner;
  AsrService service = MakeService(&runner);
  const HttpResponse response = Send(service, Multipart());
  Check(response.status == 200 &&
            response.body.find("\"text\":\"A native transcript.\"") !=
                std::string::npos,
        "valid multipart request returns OpenAI JSON");
  Check(runner.calls == 1 && runner.wav_bytes == 8U &&
            runner.context == "Names: Gufo." &&
            runner.language == std::optional<std::string>("en") &&
            runner.maximum_tokens == 42U,
        "multipart fields reach the model contract");
  Check(HasHeader(response, "X-Gufo-ASR-Backend", "native-hip"),
        "native backend is exposed");
  Check(response.log_details.find("encoder_ms=") != std::string::npos &&
            response.log_details.find("decoder_ms=") != std::string::npos &&
            response.log_details.find("A native transcript.") ==
                std::string::npos &&
            response.log_details.find("Names: Gufo.") == std::string::npos,
        "transcription diagnostics report work without transcript or context");
}

void TestFormatsAndValidation() {
  FakeRunner runner;
  AsrService service = MakeService(&runner);
  const HttpResponse text = Send(service, Multipart("text"));
  Check(text.status == 200 && text.body == "A native transcript." &&
            HasHeader(text, "Content-Type", "text/plain; charset=utf-8"),
        "text response format is supported");
  Check(Send(service, Multipart("srt")).status == 400 &&
            Send(service, Multipart("verbose_json")).status == 400,
        "segmented formats reject audio that is not a WAV");
  Check(Send(service, Multipart("xml")).status == 400,
        "unsupported response formats fail closed");
  Check(Send(service, Multipart("json", true)).status == 400,
        "unknown multipart fields fail closed");
  Check(Send(service, Multipart(), "application/json").status == 400,
        "non-multipart requests fail closed");
  Check(runner.calls == 1, "invalid requests never invoke inference");
}

void TestStreamingAndChunks() {
  FakeRunner runner;
  AsrService service = MakeService(&runner);
  const auto response = Send(service, Multipart("json", false, "true"));
  Check(response.status == 200 && response.streaming_body && runner.calls == 0,
        "streaming defers inference until the response is consumed");
  std::string events;
  response.streaming_body([&](std::string_view part) {
    events += part;
    return true;
  });
  Check(
      runner.calls == 1 &&
          events.find("\"delta\":\"A native\"") != std::string::npos &&
          events.find("\"delta\":\" transcript.\"") != std::string::npos &&
          events.find("\"type\":\"transcript.text.done\"") != std::string::npos,
      "OpenAI SSE emits incremental deltas and a complete transcript");
  const auto disconnected = Send(service, Multipart("json", false, "true"));
  int writes = 0;
  disconnected.streaming_body([&](std::string_view) {
    ++writes;
    return false;
  });
  Check(writes == 1 && disconnected.stream_log->error_code == "cancelled",
        "disconnected stream cancels inference without a done event");
  Check(!Send(service, Multipart("json", false, "false")).streaming_body,
        "stream=false retains the ordinary JSON contract");
  Check(Send(service, Multipart("json", false, "yes")).status == 400 &&
            Send(service, Multipart("text", false, "true")).status == 400,
        "invalid streaming configurations fail closed");

  constexpr std::size_t limit = 4 * qwen3_asr::kAudioSampleRate;
  const auto short_chunk =
      qwen3_asr::SplitAudio(std::vector<float>(1600), 1600);
  Check(short_chunk.size() == 1 && short_chunk.front().samples == 1600,
        "a short unsplit input does not require a half-second chunk budget");
  std::vector<float> waveform(limit * 2 + 137, 0.25F);
  std::fill(waveform.begin() + 3 * qwen3_asr::kAudioSampleRate,
            waveform.begin() + 3 * qwen3_asr::kAudioSampleRate + 1600, 0.0F);
  const auto chunks = qwen3_asr::SplitAudio(waveform, limit);
  std::size_t consumed = 0;
  for (const auto& chunk : chunks) {
    Check(
        chunk.offset == consumed && chunk.samples > 0 && chunk.samples <= limit,
        "long audio partitions exactly within the hard capacity");
    consumed += chunk.samples;
  }
  Check(consumed == waveform.size() &&
            chunks.front().samples == 3 * qwen3_asr::kAudioSampleRate,
        "chunking prefers a quiet boundary without dropping samples");
  for (const std::size_t budget : {13U, 104U, 725U}) {
    const auto samples = qwen3_asr::AudioSamplesForTokenBudget(budget);
    Check(qwen3_asr::AudioEmbeddingTokenCount(samples / 160) <= budget &&
              qwen3_asr::AudioEmbeddingTokenCount((samples + 1) / 160) > budget,
          "audio capacity includes convolutional tails");
  }
}

/// 16-bit mono 16-kHz WAV: 0.5 s silence, 1 s tone, 2 s silence, 1.5 s tone,
/// 1 s silence. Speech therefore lies in [0.5, 1.5) and [3.5, 5.0) seconds.
std::string TwoUtteranceWav() {
  constexpr std::uint32_t rate = qwen3_asr::kAudioSampleRate;
  std::vector<std::int16_t> samples;
  const auto add = [&](double seconds, bool tone) {
    const auto count = static_cast<std::size_t>(seconds * rate);
    for (std::size_t i = 0; i < count; ++i) {
      const double t = static_cast<double>(samples.size()) / rate;
      samples.push_back(
          tone ? static_cast<std::int16_t>(
                     9000.0 * std::sin(2.0 * 3.14159265358979 * 220.0 * t))
               : std::int16_t{0});
    }
  };
  add(0.5, false);
  add(1.0, true);
  add(2.0, false);
  add(1.5, true);
  add(1.0, false);
  std::string wav;
  const auto u32 = [&](std::uint32_t value) {
    for (int i = 0; i < 4; ++i)
      wav.push_back(static_cast<char>((value >> (8 * i)) & 0xFFU));
  };
  const auto u16 = [&](std::uint16_t value) {
    wav.push_back(static_cast<char>(value & 0xFFU));
    wav.push_back(static_cast<char>(value >> 8U));
  };
  const auto data_bytes = static_cast<std::uint32_t>(samples.size() * 2U);
  wav += "RIFF";
  u32(36U + data_bytes);
  wav += "WAVEfmt ";
  u32(16);
  u16(1);
  u16(1);
  u32(rate);
  u32(rate * 2U);
  u16(2);
  u16(16);
  wav += "data";
  u32(data_bytes);
  for (const std::int16_t sample : samples)
    u16(static_cast<std::uint16_t>(sample));
  return wav;
}

void TestSubtitleFormats() {
  const std::string wav = TwoUtteranceWav();
  FakeRunner runner;
  runner.segment_texts = {"Hallo <Welt> & Freunde.", "  \n  ",
                          "Zweiter Satz --> Ende."};
  AsrService service = MakeService(&runner);

  // Default merging would join two cues 2 s apart only if the gap were at
  // most 1 s; here they stay separate.
  const HttpResponse srt = Send(service, Multipart("srt", false, {}, wav));
  Check(srt.status == 200, "srt succeeds for a WAV upload");
  Check(runner.pcm_samples.size() == 2 && runner.calls == 2 &&
            runner.wav_bytes == 0U,
        "each measured speech segment is transcribed separately as PCM");
  Check(runner.pcm_samples[0] == 19200U && runner.pcm_samples[1] == 27200U,
        "segments are the speech plus 100 ms padding on each side");
  Check(srt.body == "1\n00:00:00,400 --> 00:00:01,600\n"
                    "Hallo <Welt> & Freunde.\n\n",
        "srt cue times are the measured boundaries; empty text is dropped");
  Check(HasHeader(srt, "Content-Type", "application/x-subrip; charset=utf-8") &&
            HasHeader(srt, "X-Gufo-ASR-Timestamps", "vad-segments"),
        "srt declares its media type and timing source");
  Check(srt.log_details.find("segments=2") != std::string::npos &&
            srt.log_details.find("cues=1") != std::string::npos &&
            srt.log_details.find("Hallo") == std::string::npos,
        "srt diagnostics count segments without logging the transcript");

  runner.pcm_samples.clear();
  runner.segment_texts = {"Hallo <Welt> & Freunde.", "Zweiter Satz --> Ende."};
  const HttpResponse vtt = Send(service, Multipart("vtt", false, {}, wav));
  Check(vtt.status == 200 &&
            vtt.body == "WEBVTT\n\n"
                        "1\n00:00:00.400 --> 00:00:01.600\n"
                        "Hallo &lt;Welt&gt; &amp; Freunde.\n\n"
                        "2\n00:00:03.400 --> 00:00:05.100\n"
                        "Zweiter Satz --&gt; Ende.\n\n",
        "vtt uses a header, full-stop times and escaped text");
  Check(HasHeader(vtt, "Content-Type", "text/vtt; charset=utf-8"),
        "vtt declares text/vtt");

  runner.pcm_samples.clear();
  const int calls_before_json = runner.calls;
  const HttpResponse plain = Send(service, Multipart("json", false, {}, wav));
  Check(plain.status == 200 && runner.calls == calls_before_json + 1 &&
            runner.pcm_samples.empty(),
        "json transcribes the whole file in one pass");
  const std::size_t plain_text = plain.body.find("\"text\":");
  Check(plain_text != std::string::npos, "json carries a text member");
  // `"text":"..."` up to and including the closing quote.
  const std::string text_member = plain.body.substr(
      plain_text, plain.body.find('"', plain_text + 8U) + 1U - plain_text);
  const int calls_before_verbose = runner.calls;
  const HttpResponse verbose =
      Send(service, Multipart("verbose_json", false, {}, wav, "segment"));
  Check(verbose.status == 200 && runner.calls == calls_before_verbose + 3 &&
            runner.pcm_samples.size() == 2,
        "verbose_json runs the whole file once plus every speech segment");
  Check(verbose.body.find(text_member) != std::string::npos &&
            text_member == "\"text\":\"A native transcript.\"",
        "verbose_json text equals the json text, not the joined segments");
  Check(verbose.body.find("\"duration\":2") != std::string::npos &&
            verbose.body.find("\"language\":\"english\"") !=
                std::string::npos,
        "verbose_json reports the whole-file duration and language");
  Check(verbose.body.find("\"text\":\"Hallo <Welt> & Freunde.\"") !=
                std::string::npos &&
            verbose.body.find("\"text\":\"Zweiter Satz --> Ende.\"") !=
                std::string::npos,
        "verbose_json segments keep their own per-segment text");
  Check(HasHeader(verbose, "X-Gufo-ASR-Timestamps", "vad-segments") &&
            verbose.log_details.find("whole_file_ms=") != std::string::npos &&
            verbose.log_details.find("A native") == std::string::npos,
        "verbose_json diagnostics report the extra pass without the transcript");
  Check(verbose.body.find("\"id\":0") != std::string::npos &&
            verbose.body.find("\"start\":0.4") != std::string::npos &&
            verbose.body.find("\"end\":1.6") != std::string::npos &&
            verbose.body.find("\"start\":3.4") != std::string::npos &&
            verbose.body.find("\"end\":5.1") != std::string::npos &&
            verbose.body.find("\"seek\":340") != std::string::npos &&
            verbose.body.find("\"tokens\":[1,2,3]") != std::string::npos,
        "verbose_json segments carry measured start and end seconds");

  const int calls = runner.calls;
  Check(Send(service, Multipart("verbose_json", false, {}, wav, "word"))
                .status == 400,
        "word timestamps are refused rather than estimated");
  Check(Send(service, Multipart("json", false, {}, wav, "segment")).status ==
            400,
        "timestamp granularities require verbose_json");
  Check(Send(service, Multipart("srt", false, "true", wav)).status == 400,
        "subtitles cannot be streamed");
  Check(runner.calls == calls, "rejected subtitle requests never run");

  std::string silent = wav.substr(0, 44);
  silent.append(16000U * 2U, '\0');
  // Fix the RIFF and data sizes for one second of silence.
  const auto patch = [&](std::size_t at, std::uint32_t value) {
    for (int i = 0; i < 4; ++i)
      silent[at + static_cast<std::size_t>(i)] =
          static_cast<char>((value >> (8 * i)) & 0xFFU);
  };
  patch(4, 36U + 32000U);
  patch(40, 32000U);
  const HttpResponse empty = Send(service, Multipart("srt", false, {}, silent));
  Check(empty.status == 200 && empty.body.empty() && runner.calls == calls,
        "silence yields an empty SRT without running the model");
}

}  // namespace

int main() {
  TestJsonRequest();
  TestFormatsAndValidation();
  TestStreamingAndChunks();
  TestSubtitleFormats();
  std::cout << "PASS qwen3_asr_audio_api_test\n";
  return 0;
}
