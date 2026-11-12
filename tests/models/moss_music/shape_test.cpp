// MOSS-Music stage 1: shape math of the Conv2d stem, chunking, the time
// marker layout of the prompt and the weight table. Pure CPU, no weights.
// Expected values come from src/processing_moss_music.py and
// src/modeling_moss_music.py of github.com/OpenMOSS/MOSS-Music (replayed in
// plain Python) and from model.safetensors.index.json of the checkpoint.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/moss_music/config.hpp"
#include "src/models/moss_music/model_description.hpp"

namespace moss = gufo::models::moss_music;

namespace {

int failures = 0;

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL moss_music_shape_test: " << message << '\n';
    ++failures;
  }
}

void CheckEq(std::size_t actual, std::size_t expected, std::string_view what) {
  if (actual != expected) {
    std::cerr << "FAIL moss_music_shape_test: " << what << " = " << actual
              << ", expected " << expected << '\n';
    ++failures;
  }
}

void TestConfigDerived() {
  const moss::AudioEncoderConfig audio;
  CheckEq(audio.HeadDim(), 64, "encoder head_dim");
  CheckEq(audio.ChunkFrames(), 400, "chunk frames (2 * n_window)");
  CheckEq(audio.ChunkTokens(), 50, "tokens per full chunk");
  CheckEq(audio.StemFrequency(), 16, "mel bins after the stem");
  CheckEq(audio.StemFlattened(), 7680, "stem_proj input width");
  const moss::TextConfig text;
  CheckEq(text.QDim(), 4096, "q dim");
  CheckEq(text.KvDim(), 1024, "kv dim");
  // 12.5 tokens/s = 25 tokens per 2 s: 2 * 100 frames/s over 8 frames.
  CheckEq(moss::kSampleRate / moss::kHopLength * 2 / moss::kDownsampleRate,
          moss::kTokensPerTwoSeconds, "tokens per two seconds");
}

void TestConvLengths() {
  // (L - 1) / 2 + 1 for kernel 3, stride 2, padding 1.
  const std::size_t expected[] = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5};
  for (std::uint32_t length = 0; length < 10; ++length) {
    CheckEq(moss::ConvOutLength(length), expected[length],
            "ConvOutLength(" + std::to_string(length) + ")");
  }
  CheckEq(moss::ConvOutLength(400), 200, "conv1 time of a full chunk");
  CheckEq(moss::ConvOutLength(200), 100, "conv2 time of a full chunk");
  CheckEq(moss::ConvOutLength(100), 50, "conv3 time of a full chunk");
  CheckEq(moss::ConvOutLength(128), 64, "conv1 frequency");
  CheckEq(moss::ConvOutLength(64), 32, "conv2 frequency");
  CheckEq(moss::ConvOutLength(32), 16, "conv3 frequency");

  // StemOutLength == three convolutions; 1..8 frames give one token.
  for (std::uint32_t frames = 1; frames <= 8; ++frames) {
    CheckEq(moss::StemOutLength(frames), 1,
            "StemOutLength(" + std::to_string(frames) + ")");
  }
  CheckEq(moss::StemOutLength(9), 2, "StemOutLength(9)");
  CheckEq(moss::StemOutLength(100), 13, "StemOutLength(100)");
  CheckEq(moss::StemOutLength(399), 50, "StemOutLength(399)");
  CheckEq(moss::StemOutLength(400), 50, "StemOutLength(400)");
  CheckEq(moss::StemOutLength(401), 51, "StemOutLength(401)");
  CheckEq(moss::StemOutLength(12000), 1500, "StemOutLength(12000) = 120 s");
}

void TestStemShape() {
  const moss::StemShape full = moss::StemOutput(400);
  CheckEq(full.channels, 480, "stem channels");
  CheckEq(full.frequency, 16, "stem frequency");
  CheckEq(full.time, 50, "stem time");
  CheckEq(full.Flattened(), 7680, "stem flattened width");
  CheckEq(full.Elements(), 480U * 16U * 50U, "stem elements");
  const moss::StemShape tail = moss::StemOutput(7);
  CheckEq(tail.time, 1, "stem time of a 7-frame tail");
  CheckEq(tail.frequency, 16, "stem frequency of a tail");
  // permute(0, 3, 1, 2).flatten(2): channel-major, frequency fastest.
  CheckEq(moss::StemFlatIndex(0, 0), 0, "flat index (0, 0)");
  CheckEq(moss::StemFlatIndex(0, 15), 15, "flat index (0, 15)");
  CheckEq(moss::StemFlatIndex(1, 0), 16, "flat index (1, 0)");
  CheckEq(moss::StemFlatIndex(479, 15), 7679, "flat index (479, 15)");
}

void TestChunking() {
  struct Case {
    std::uint32_t frames;
    std::vector<std::uint32_t> chunks;
    std::size_t tokens;
  };
  const Case cases[] = {
      {1, {1}, 1},
      {8, {8}, 1},
      {9, {9}, 2},
      {100, {100}, 13},
      {399, {399}, 50},
      {400, {400}, 50},
      {401, {400, 1}, 51},
      {800, {400, 400}, 100},
      {801, {400, 400, 1}, 101},
      {1200, {400, 400, 400}, 150},
      {4001, {400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 1}, 501},
      {12007, {}, 1501},
      {12008, {}, 1501},
  };
  for (const Case& c : cases) {
    const std::string tag = "frames=" + std::to_string(c.frames);
    const std::vector<std::uint32_t> chunks = moss::ChunkFrameCounts(c.frames);
    if (!c.chunks.empty()) {
      Check(chunks == c.chunks, tag + ": chunk frame counts");
    }
    std::size_t total = 0;
    for (const std::uint32_t chunk : chunks) {
      Check(chunk >= 1 && chunk <= 400, tag + ": chunk within 1..400");
      total += chunk;
    }
    CheckEq(total, c.frames, tag + ": chunks cover all frames");
    CheckEq(moss::AudioTokenCount(c.frames), c.tokens, tag + ": tokens");
    // The processor counts placeholders with the unchunked formula.
    CheckEq(moss::AudioTokenCount(c.frames), moss::StemOutLength(c.frames),
            tag + ": chunked == unchunked count");
  }
  Check(moss::ChunkFrameCounts(0).empty(), "no chunks without frames");
  CheckEq(moss::AudioTokenCount(12000), 1500, "120 s give 1500 tokens");
  CheckEq(moss::ChunkFrameCounts(12000).size(), 30, "120 s give 30 chunks");
}

void TestTimeMarkers() {
  const moss::TokenIds ids;
  constexpr std::int32_t kAudio = 151654;
  CheckEq(ids.audio_pad, kAudio, "<|AUDIO|> id");
  CheckEq(ids.audio_start, 151669, "<|audio_bos|> id");
  CheckEq(ids.audio_end, 151670, "<|audio_eos|> id");

  // Fewer than 25 tokens (under 2 s): placeholders only.
  const std::vector<std::int32_t> short_span = moss::AudioSpanWithMarkers(24);
  CheckEq(short_span.size(), 24, "24 tokens: no marker");
  Check(short_span == std::vector<std::int32_t>(24, kAudio),
        "24 tokens: all placeholders");

  // 25 tokens = exactly 2 s: 25 placeholders then "2".
  const std::vector<std::int32_t> two_s = moss::AudioSpanWithMarkers(25);
  CheckEq(two_s.size(), 26, "25 tokens: 25 + '2'");
  Check(two_s.back() == ids.digit_zero + 2, "25 tokens: ends with '2'");

  // 26 tokens: 25 placeholders, "2", one placeholder.
  const std::vector<std::int32_t> span26 = moss::AudioSpanWithMarkers(26);
  CheckEq(span26.size(), 27, "26 tokens span length");
  Check(span26[25] == ids.digit_zero + 2 && span26[26] == kAudio,
        "26 tokens: '2' at index 25, placeholder after it");

  // 50 tokens = one full chunk (4 s): 25 A, "2", 25 A, "4".
  const std::vector<std::int32_t> span50 = moss::AudioSpanWithMarkers(50);
  CheckEq(span50.size(), 52, "50 tokens span length");
  Check(span50[25] == ids.digit_zero + 2 && span50[51] == ids.digit_zero + 4,
        "50 tokens: '2' at 25 and '4' at 51");
  CheckEq(moss::MarkerTokenCount(50), 2, "50 tokens: 2 marker tokens");

  // 63 tokens (5.04 s): markers at 2 and 4 only, 13 trailing placeholders.
  const std::vector<std::int32_t> span63 = moss::AudioSpanWithMarkers(63);
  CheckEq(span63.size(), 65, "63 tokens span length");
  for (std::size_t i = 52; i < 65; ++i) {
    Check(span63[i] == kAudio, "63 tokens: trailing placeholders");
  }

  // Second 10 is the first two-digit marker: "1" "0" after 5 * 25 audio
  // tokens and 4 earlier single-digit markers, at index 129.
  const std::vector<std::int32_t> span150 = moss::AudioSpanWithMarkers(150);
  CheckEq(span150.size(), 158, "150 tokens (12 s) span length");
  Check(span150[129] == ids.digit_zero + 1 && span150[130] == ids.digit_zero,
        "150 tokens: '1','0' at 129/130");
  CheckEq(moss::MarkerTokenCount(150), 8, "150 tokens: 8 marker digits");

  // Whole-clip counts replayed from the processor.
  CheckEq(moss::AudioSpanWithMarkers(100).size(), 104, "100 tokens -> 104");
  CheckEq(moss::AudioSpanWithMarkers(101).size(), 105, "101 tokens -> 105");
  CheckEq(moss::AudioSpanWithMarkers(500).size(), 536, "500 tokens -> 536");
  CheckEq(moss::AudioSpanWithMarkers(501).size(), 537, "501 tokens -> 537");
  // 120 s: 60 markers, 4 + 45 * 2 + 11 * 3 = 127 digits.
  const std::vector<std::int32_t> span1500 = moss::AudioSpanWithMarkers(1500);
  CheckEq(span1500.size(), 1627, "1500 tokens -> 1627");
  CheckEq(moss::MarkerTokenCount(1500), 127, "1500 tokens: 127 digits");
  // "1","2","0" are the last three tokens: 120 s ends on a marker.
  Check(span1500[1624] == ids.digit_zero + 1 &&
            span1500[1625] == ids.digit_zero + 2 &&
            span1500[1626] == ids.digit_zero,
        "1500 tokens: ends with '1','2','0'");
  CheckEq(moss::AudioSpanWithMarkers(1501).size(), 1628, "1501 tokens -> 1628");

  // Placeholder count always equals the token count; digits are 15..24.
  for (const std::size_t tokens : {1U, 24U, 25U, 49U, 50U, 777U, 1500U}) {
    std::size_t placeholders = 0;
    bool digits_ok = true;
    for (const std::int32_t id : moss::AudioSpanWithMarkers(tokens)) {
      if (id == kAudio) {
        ++placeholders;
      } else if (id < ids.digit_zero || id > ids.digit_zero + 9) {
        digits_ok = false;
      }
    }
    CheckEq(placeholders, tokens, "placeholder count");
    Check(digits_ok, "marker tokens are digits");
  }
}

void TestPromptRows() {
  const moss::TokenIds ids;
  // A prompt: 3 text tokens, <|audio_bos|>, the span, <|audio_eos|>, text.
  std::vector<std::int32_t> prompt{151644, 8948, 198, ids.audio_start};
  const std::vector<std::int32_t> span = moss::AudioSpan(63, true, ids);
  prompt.insert(prompt.end(), span.begin(), span.end());
  prompt.push_back(ids.audio_end);
  prompt.push_back(198);
  const std::vector<std::uint32_t> rows = moss::AudioRowIndices(prompt, ids);
  CheckEq(rows.size(), 63, "63 audio rows");
  CheckEq(rows.front(), 4, "first audio row after <|audio_bos|>");
  // Marker "2" sits at span index 25 (prompt 29): row 25 jumps over it.
  CheckEq(rows[24], 28, "row 24 before the first marker");
  CheckEq(rows[25], 30, "row 25 after the first marker");
  // Marker "4" after 50 placeholders (span index 51, prompt 55).
  CheckEq(rows[49], 54, "row 49 before the second marker");
  CheckEq(rows[50], 56, "row 50 after the second marker");
  CheckEq(rows.back(), 68, "last audio row");
  for (std::size_t i = 1; i < rows.size(); ++i) {
    Check(rows[i] > rows[i - 1], "rows strictly increasing");
  }
  // Without markers the span is plain and the rows are contiguous.
  const std::vector<std::int32_t> plain = moss::AudioSpan(63, false, ids);
  CheckEq(plain.size(), 63, "plain span length");
  Check(plain == std::vector<std::int32_t>(63, ids.audio_pad),
        "plain span is all placeholders");
}

void TestWeightTable() {
  const std::vector<moss::TensorDescription> tensors = moss::DescribeTensors();
  // 9 stem/norm entries + 32 * 15 layer entries + 2 final norm = 491 encoder,
  // 4 * 3 gated MLPs, 1 + 36 * 11 + 2 language model = 399.
  CheckEq(tensors.size(), 491 + 12 + 399, "tensor count");
  CheckEq(moss::ParameterCount(tensors, false), moss::kCheckpointParameters,
          "parameters without the sinusoid buffer (index total_parameters)");
  CheckEq(moss::ParameterCount(tensors, true) * 2, moss::kCheckpointBytes,
          "BF16 bytes with the buffer (index total_size)");

  std::size_t encoder = 0;
  std::size_t adapters = 0;
  std::size_t llm = 0;
  std::size_t buffers = 0;
  for (const moss::TensorDescription& t : tensors) {
    Check(!t.hf_name.empty() && !t.gufo_name.empty(), "names present");
    Check(!t.shape.empty(), "shape present: " + t.hf_name);
    if (t.buffer) {
      ++buffers;
    }
    switch (t.component) {
      case moss::Component::kAudioEncoder:
        encoder += t.Elements();
        Check(t.gufo_name == t.hf_name, "encoder keeps HF names");
        break;
      case moss::Component::kAudioAdapter:
      case moss::Component::kDeepStackMerger:
        adapters += t.Elements();
        Check(t.gufo_name == t.hf_name, "adapters keep HF names");
        break;
      case moss::Component::kLanguageModel:
        llm += t.Elements();
        Check(t.gufo_name.rfind("language_model", 0) == std::string::npos,
              "LLM uses llama.cpp names: " + t.gufo_name);
        break;
    }
  }
  CheckEq(buffers, 1, "one buffer (inv_timescales)");
  CheckEq(encoder, 643'624'960, "encoder elements incl. buffer");
  CheckEq(adapters, 4 * 54'525'952, "adapter + 3 merger elements");
  CheckEq(llm, 8'190'735'360, "Qwen3-8B elements");

  // Spot checks of shapes the kernels depend on.
  for (const moss::TensorDescription& t : tensors) {
    if (t.hf_name == "audio_encoder.stem_proj.weight") {
      Check(t.shape == std::vector<std::uint64_t>{1280, 7680},
            "stem_proj [1280, 7680]");
    } else if (t.hf_name == "audio_encoder.conv1.weight") {
      Check(t.shape == std::vector<std::uint64_t>{480, 1, 3, 3},
            "conv1 [480, 1, 3, 3]");
    } else if (t.hf_name == "deepstack_audio_merger_list.2.down_proj.weight") {
      Check(t.shape == std::vector<std::uint64_t>{4096, 8192},
            "merger down [4096, 8192]");
    } else if (t.hf_name == "language_model.layers.35.self_attn.k_proj.weight") {
      Check(t.shape == std::vector<std::uint64_t>{1024, 4096} &&
                t.gufo_name == "blk.35.attn_k.weight",
            "layer 35 k_proj [1024, 4096] -> blk.35.attn_k.weight");
    } else if (t.hf_name == "lm_head.weight") {
      Check(t.shape == std::vector<std::uint64_t>{151936, 4096} &&
                t.gufo_name == "output.weight",
            "lm_head [151936, 4096] -> output.weight");
    }
  }
}

}  // namespace

int main() {
  TestConfigDerived();
  TestConvLengths();
  TestStemShape();
  TestChunking();
  TestTimeMarkers();
  TestPromptRows();
  TestWeightTable();
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "moss_music_shape_test: OK\n";
  return 0;
}
