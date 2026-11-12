// gufo-moss-music: driver for the MOSS-Music port (stage 2/3 tooling).
//
//   gufo-moss-music --audio-gguf moss-music-8b.audio.gguf --audio clip.wav
//                   --dump PREFIX [--stage frontend|encoder]
//                   [--backend cpu|gpu|both] [--threads N]
//                   [--llm-gguf moss-music-8b.llm-bf16.gguf --text "question"
//                    --max-new 64 --context 4096]
//
// Decodes the WAV, computes the Whisper log-mel features and runs the F32 CPU
// reference (--backend cpu), the HIP encoder (gpu) or both with a stage-wise
// comparison. With --llm-gguf the HIP encoder feeds the Qwen3-8B LLM: the
// ChatML prompt with time markers is built, the prompt is prefilled with the
// adapter rows and the DeepStack additions, and the answer is decoded
// greedily. Dumps (PREFIX.*.f32, .i32, .meta.json) mirror the Regie reference
// package (PLAN.md D1-D9) so tools/moss_music/compare_reference.py can
// compare them.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "src/models/moss_music/config.hpp"
#include "src/models/moss_music/model.hpp"
#include "src/models/moss_music/reference.hpp"
#include "src/models/qwen3_asr/audio.hpp"
#if defined(GUFO_MOSS_MUSIC_HIP)
#include <hip/hip_runtime.h>

#include "src/models/moss_music/hip/audio_encoder.hpp"
#include "src/models/moss_music/hip/llm.hpp"
#include "src/models/qwen/tokenizer.hpp"
#endif

namespace {

namespace moss = gufo::models::moss_music;
namespace asr = gufo::models::qwen3_asr;

struct Arguments {
  std::string audio_gguf;
  std::string audio;
  std::string dump;
  std::string stage{"frontend"};
  std::string backend{"cpu"};
  std::string llm_gguf;
  std::string text{"Please give a detailed musical description of this clip."};
  std::size_t max_new{64};
  std::size_t context{4096};
  unsigned threads{0};
};

void Usage() {
  std::cerr << "usage: gufo-moss-music --audio-gguf FILE.gguf --audio FILE.wav "
               "--dump PREFIX [--stage frontend|encoder] "
               "[--backend cpu|gpu|both] [--threads N] "
               "[--llm-gguf FILE.gguf --text QUESTION --max-new N --context N]\n";
}

bool Parse(int argc, char** argv, Arguments* arguments) {
  for (int index = 1; index < argc; ++index) {
    const std::string name = argv[index];
    if (index + 1 >= argc) {
      return false;
    }
    const std::string value = argv[++index];
    if (name == "--audio-gguf") {
      arguments->audio_gguf = value;
    } else if (name == "--audio") {
      arguments->audio = value;
    } else if (name == "--dump") {
      arguments->dump = value;
    } else if (name == "--stage") {
      arguments->stage = value;
    } else if (name == "--backend") {
      arguments->backend = value;
    } else if (name == "--llm-gguf") {
      arguments->llm_gguf = value;
    } else if (name == "--text") {
      arguments->text = value;
    } else if (name == "--max-new") {
      arguments->max_new = std::stoul(value);
    } else if (name == "--context") {
      arguments->context = std::stoul(value);
    } else if (name == "--threads") {
      arguments->threads = static_cast<unsigned>(std::stoul(value));
    } else {
      return false;
    }
  }
  if (!arguments->llm_gguf.empty()) {
    arguments->backend = "gpu";
    arguments->stage = "encoder";
  }
  return !arguments->audio_gguf.empty() && !arguments->audio.empty() &&
         !arguments->dump.empty() &&
         (arguments->stage == "frontend" || arguments->stage == "encoder") &&
         (arguments->backend == "cpu" || arguments->backend == "gpu" ||
          arguments->backend == "both");
}

template<typename Element>
bool WriteBinary(const std::string& path, std::span<const Element> values) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(Element)));
  return static_cast<bool>(out);
}

/// Stage outputs by name, in dump order.
using Stages = std::map<std::string, std::vector<float>>;

Stages RunCpu(const moss::AudioModel& model, const asr::LogMelFeatures& features,
              const std::string& stage, unsigned threads) {
  const moss::CpuReference reference(model, threads);
  Stages out;
  out["frontend"] = reference.EncodeFrontend(features.values, features.frames);
  if (stage == "encoder") {
    const moss::EncoderOutput encoded =
        reference.EncodeAudio(features.values, features.frames);
    const auto& taps = model.config().audio.deepstack_layers;
    for (std::size_t tap = 0; tap < encoded.deepstack.size(); ++tap) {
      out["enc" + std::to_string(taps[tap])] = encoded.deepstack[tap];
      out["merger" + std::to_string(tap)] = reference.GatedMlp(
          model.mergers()[tap], encoded.deepstack[tap], encoded.tokens);
    }
    out["final"] = encoded.final;
    out["adapter"] =
        reference.GatedMlp(model.adapter(), encoded.final, encoded.tokens);
  }
  return out;
}

void Compare(const Stages& cpu, const Stages& gpu) {
  for (const auto& [name, reference] : cpu) {
    const auto it = gpu.find(name);
    if (it == gpu.end() || it->second.size() != reference.size()) {
      std::cout << "compare " << name << ": missing or size mismatch\n";
      continue;
    }
    const std::vector<float>& candidate = it->second;
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    float max_abs = 0.0F;
    float max_ref = 0.0F;
    std::size_t at = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
      const float d = std::fabs(reference[i] - candidate[i]);
      if (d > max_abs) {
        max_abs = d;
        at = i;
      }
      max_ref = std::max(max_ref, std::fabs(reference[i]));
      dot += static_cast<double>(reference[i]) * candidate[i];
      norm_a += static_cast<double>(reference[i]) * reference[i];
      norm_b += static_cast<double>(candidate[i]) * candidate[i];
    }
    const double cos = dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-30);
    std::cout << "compare " << name << ": n=" << reference.size()
              << " max|cpu-gpu|=" << max_abs << " (at " << at << ", cpu "
              << reference[at] << " gpu " << candidate[at] << ") max|cpu|="
              << max_ref << " rel=" << max_abs / (max_ref + 1e-30)
              << " cos=" << cos << '\n';
  }
}

#if defined(GUFO_MOSS_MUSIC_HIP)
struct GpuRun {
  Stages stages;
  double encoder_seconds{0.0};
  double llm_load_seconds{0.0};
  double prefill_seconds{0.0};
  double decode_seconds{0.0};
  std::vector<std::int32_t> prompt;
  std::vector<std::uint32_t> audio_rows;
  std::vector<float> logits_first;
  std::vector<std::int32_t> generated;
  std::string answer;
  moss::hip::PrefillTrace trace;
};

bool RunGpu(const moss::AudioModel& model, const asr::LogMelFeatures& features,
            const Arguments& arguments, GpuRun* run, std::string* error) {
  const moss::TokenIds ids = model.config().tokens;
  hipStream_t stream = nullptr;
  if (hipStreamCreate(&stream) != hipSuccess) {
    *error = "hipStreamCreate failed";
    return false;
  }
  std::unique_ptr<moss::hip::AudioEncoder> encoder =
      moss::hip::AudioEncoder::Create(model, stream, error);
  if (encoder == nullptr) {
    (void)hipStreamDestroy(stream);
    return false;
  }
  std::cout << "gpu: encoder weights " << (encoder->weight_bytes() >> 20)
            << " MiB\n";
  auto started = std::chrono::steady_clock::now();
  moss::hip::EncodedAudio encoded;
  bool ok = encoder->Encode(features.values, features.frames, &encoded, error);
  run->encoder_seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - started)
                             .count();
  const std::size_t hidden = model.config().audio.d_model;
  const std::size_t adapter_out = model.config().adapter.output;
  const auto& taps = model.config().audio.deepstack_layers;
  Stages& out = run->stages;
  if (ok) {
    ok = encoder->CopyToHost(encoded.frontend, encoded.tokens * hidden,
                             &out["frontend"], error);
  }
  if (ok && arguments.stage == "encoder") {
    for (std::size_t tap = 0; ok && tap < taps.size(); ++tap) {
      ok = encoder->CopyToHost(encoded.taps[tap], encoded.tokens * hidden,
                               &out["enc" + std::to_string(taps[tap])],
                               error) &&
           encoder->CopyToHost(encoded.mergers[tap],
                               encoded.tokens * adapter_out,
                               &out["merger" + std::to_string(tap)], error);
    }
    ok = ok &&
         encoder->CopyToHost(encoded.final, encoded.tokens * hidden,
                             &out["final"], error) &&
         encoder->CopyToHost(encoded.adapter, encoded.tokens * adapter_out,
                             &out["adapter"], error);
  }
  if (ok && !arguments.llm_gguf.empty()) {
    started = std::chrono::steady_clock::now();
    std::unique_ptr<moss::hip::Llm> llm = moss::hip::Llm::Create(
        arguments.llm_gguf, arguments.context, stream, error);
    run->llm_load_seconds = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - started)
                                .count();
    if (llm == nullptr) {
      ok = false;
    } else {
      std::cout << "gpu: LLM weights " << (llm->weight_bytes() >> 20)
                << " MiB loaded in " << run->llm_load_seconds << " s\n";
      std::unique_ptr<gufo::tokenization::QwenTokenizer> tokenizer =
          gufo::tokenization::QwenTokenizer::CreateFromGguf(llm->reader(),
                                                             error);
      if (tokenizer == nullptr) {
        ok = false;
      } else {
        // MossMusicProcessor._build_default_prompt + _build_input_from_prompt.
        const auto append = [&](const std::string& text) {
          for (const auto id : tokenizer->Encode(text)) {
            run->prompt.push_back(static_cast<std::int32_t>(id));
          }
        };
        append("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
               "<|im_start|>user\n");
        run->prompt.push_back(ids.audio_start);
        const std::vector<std::int32_t> span =
            moss::AudioSpanWithMarkers(encoded.tokens, ids);
        run->prompt.insert(run->prompt.end(), span.begin(), span.end());
        run->prompt.push_back(ids.audio_end);
        append("\n" + arguments.text + "<|im_end|>\n<|im_start|>assistant\n");
        for (std::size_t index = 0; index < run->prompt.size(); ++index) {
          if (run->prompt[index] == ids.audio_pad) {
            run->audio_rows.push_back(static_cast<std::uint32_t>(index));
          }
        }
        std::cout << "prompt: " << run->prompt.size() << " tokens, "
                  << run->audio_rows.size() << " audio rows\n";
        if (run->audio_rows.size() != encoded.tokens) {
          *error = "audio row count differs from the encoder token count";
          ok = false;
        } else if (run->prompt.size() + arguments.max_new > arguments.context) {
          *error = "prompt plus answer exceed the context";
          ok = false;
        }
      }
      if (ok) {
        started = std::chrono::steady_clock::now();
        std::vector<float> logits;
        ok = llm->Prefill(run->prompt, run->audio_rows, encoded.adapter,
                          encoded.mergers, &logits, error, &run->trace);
        run->prefill_seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - started)
                                   .count();
        if (ok) {
          run->logits_first = logits;
          started = std::chrono::steady_clock::now();
          while (ok && run->generated.size() < arguments.max_new) {
            const auto next = static_cast<std::int32_t>(
                std::max_element(logits.begin(), logits.end()) -
                logits.begin());
            run->generated.push_back(next);
            if (next == ids.im_end || next == ids.endoftext) {
              break;
            }
            ok = llm->Decode(next, &logits, error);
          }
          run->decode_seconds = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - started)
                                    .count();
          std::vector<gufo::tokenization::TokenId> answer_ids;
          for (const std::int32_t id : run->generated) {
            if (id != ids.im_end && id != ids.endoftext) {
              answer_ids.push_back(static_cast<gufo::tokenization::TokenId>(id));
            }
          }
          run->answer = tokenizer->Decode(answer_ids);
        }
      }
    }
  }
  encoder.reset();
  (void)hipStreamDestroy(stream);
  return ok;
}
#endif

}  // namespace

int main(int argc, char** argv) {
  Arguments arguments;
  if (!Parse(argc, argv, &arguments)) {
    Usage();
    return 2;
  }
  std::ifstream wav_file(arguments.audio, std::ios::binary);
  if (!wav_file) {
    std::cerr << "cannot read " << arguments.audio << '\n';
    return 1;
  }
  const std::vector<char> wav_bytes((std::istreambuf_iterator<char>(wav_file)),
                                    std::istreambuf_iterator<char>());
  asr::AudioBuffer buffer;
  std::string error;
  if (!asr::DecodeWav(std::as_bytes(std::span(wav_bytes)), &buffer, &error)) {
    std::cerr << "WAV: " << error << '\n';
    return 1;
  }
  const std::vector<float> waveform = asr::ResampleMono16k(buffer);
  const asr::LogMelFeatures features = asr::ComputeLogMelFeatures(waveform);
  std::cout << "audio: " << buffer.sample_rate << " Hz, " << buffer.channels
            << " ch -> " << waveform.size() << " samples at 16 kHz, "
            << features.frames << " mel frames\n";

  std::unique_ptr<moss::AudioModel> model =
      moss::AudioModel::Load(arguments.audio_gguf, &error);
  if (model == nullptr) {
    std::cerr << "load: " << error << '\n';
    return 1;
  }
  const moss::AudioEncoderConfig& config = model->config().audio;
  const std::size_t tokens = moss::AudioTokenCount(
      static_cast<std::uint32_t>(features.frames), config);
  const std::vector<std::uint32_t> chunks = moss::ChunkFrameCounts(
      static_cast<std::uint32_t>(features.frames), config);
  std::cout << "chunks: " << chunks.size() << " (" << config.ChunkFrames()
            << " frames each, tail " << chunks.back() << "), tokens: " << tokens
            << ", marker span: " << moss::AudioSpanWithMarkers(tokens).size()
            << '\n';

  Stages cpu;
  double cpu_seconds = 0.0;
  if (arguments.backend != "gpu") {
    const auto started = std::chrono::steady_clock::now();
    cpu = RunCpu(*model, features, arguments.stage, arguments.threads);
    cpu_seconds = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - started)
                      .count();
    std::cout << "cpu: " << cpu_seconds << " s\n";
  }
#if defined(GUFO_MOSS_MUSIC_HIP)
  GpuRun gpu;
  if (arguments.backend != "cpu") {
    if (!RunGpu(*model, features, arguments, &gpu, &error)) {
      std::cerr << "gpu: " << error << '\n';
      return 1;
    }
    std::cout << "gpu: encoder " << gpu.encoder_seconds << " s";
    if (!arguments.llm_gguf.empty()) {
      std::cout << ", prefill " << gpu.prefill_seconds << " s, decode "
                << gpu.generated.size() << " tokens in " << gpu.decode_seconds
                << " s";
    }
    std::cout << '\n';
    if (!gpu.answer.empty()) {
      std::cout << "answer: " << gpu.answer << '\n';
    }
  }
  if (arguments.backend == "both") {
    Compare(cpu, gpu.stages);
  }
  const Stages& primary = arguments.backend == "gpu" ? gpu.stages : cpu;
#else
  if (arguments.backend != "cpu") {
    std::cerr << "this build has no HIP backend\n";
    return 1;
  }
  const Stages& primary = cpu;
#endif

  std::ofstream meta(arguments.dump + ".meta.json");
  meta << "{\n  \"sample_rate\": 16000,\n  \"samples\": " << waveform.size()
       << ",\n  \"mel_bins\": " << config.mel_bins
       << ",\n  \"frames\": " << features.frames
       << ",\n  \"chunk_frames\": " << config.ChunkFrames()
       << ",\n  \"tokens\": " << tokens << ",\n  \"d_model\": " << config.d_model
       << ",\n  \"adapter_out\": " << model->config().adapter.output
       << ",\n  \"stage\": \"" << arguments.stage << "\",\n  \"backend\": \""
       << arguments.backend << "\",\n  \"cpu_seconds\": " << cpu_seconds;
  bool ok = WriteBinary<float>(arguments.dump + ".mel.f32", features.values);
  for (const auto& [name, values] : primary) {
    ok = ok && WriteBinary<float>(arguments.dump + "." + name + ".f32", values);
  }
#if defined(GUFO_MOSS_MUSIC_HIP)
  meta << ",\n  \"gpu_encoder_seconds\": " << gpu.encoder_seconds;
  if (arguments.backend == "both") {
    for (const auto& [name, values] : gpu.stages) {
      ok = ok &&
           WriteBinary<float>(arguments.dump + ".gpu." + name + ".f32", values);
    }
  }
  if (!arguments.llm_gguf.empty()) {
    ok = ok &&
         WriteBinary<std::int32_t>(arguments.dump + ".input_ids.i32",
                                   gpu.prompt) &&
         WriteBinary<std::uint32_t>(arguments.dump + ".audio_rows.u32",
                                    gpu.audio_rows) &&
         WriteBinary<float>(arguments.dump + ".logits_last.f32",
                            gpu.logits_first) &&
         WriteBinary<std::int32_t>(arguments.dump + ".greedy_ids.i32",
                                   gpu.generated) &&
         WriteBinary<float>(arguments.dump + ".hs0.f32", gpu.trace.hs0) &&
         WriteBinary<float>(arguments.dump + ".hs1.f32",
                            gpu.trace.hs_layer[0]) &&
         WriteBinary<float>(arguments.dump + ".hs2.f32",
                            gpu.trace.hs_layer[1]) &&
         WriteBinary<float>(arguments.dump + ".hs3.f32",
                            gpu.trace.hs_layer[2]) &&
         WriteBinary<float>(arguments.dump + ".hs_final.f32",
                            gpu.trace.hs_final);
    std::string escaped;
    for (const char c : gpu.answer) {
      if (c == '"' || c == '\\') {
        escaped += '\\';
        escaped += c;
      } else if (c == '\n') {
        escaped += "\\n";
      } else if (static_cast<unsigned char>(c) < 0x20) {
        escaped += ' ';
      } else {
        escaped += c;
      }
    }
    meta << ",\n  \"prompt_tokens\": " << gpu.prompt.size()
         << ",\n  \"audio_tokens\": " << gpu.audio_rows.size()
         << ",\n  \"generated\": " << gpu.generated.size()
         << ",\n  \"llm_load_seconds\": " << gpu.llm_load_seconds
         << ",\n  \"prefill_seconds\": " << gpu.prefill_seconds
         << ",\n  \"decode_seconds\": " << gpu.decode_seconds
         << ",\n  \"text\": \"" << arguments.text << "\",\n  \"answer\": \""
         << escaped << "\"";
  }
#endif
  meta << "\n}\n";
  if (!ok || !meta) {
    std::cerr << "cannot write dumps at " << arguments.dump << '\n';
    return 1;
  }
  std::cout << "done -> " << arguments.dump << ".*\n";
  return 0;
}
