#include "src/models/qwen_image_21/model.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/image_api.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen_image_21/hip/memory.hpp"
#include "src/models/qwen_image_21/tokenizer.hpp"
#include "src/models/qwen_image_21/weights.hpp"

int main(int argc, char** argv) {
  using namespace gufo::models::qwen_image_21;
  if (argc > 2)
    return 2;
  if (argc == 2) {
    // Official tokenizers output from the checkpoint pinned in UPSTREAM.md.
    // Optional local-artifact qualification; hosted tests never download
    // weights.
    Tokenizer tokenizer(argv[1]);
    const std::vector<std::pair<std::string, std::vector<std::uint32_t>>> cases{
        {"A red cube on a white table.",
         {32, 2518, 23739, 389, 264, 4158, 1965, 13}},
        {"  cafe\xcc\x81  dÃ©jÃ  vu", {220, 51950, 220, 45839, 32514}},
        {"à¤¨à¤®à¤¸à¥à¤¤à¥‡ ä¸–ç•Œ ì•ˆë…•í•˜ì„¸ìš”",
         {60096, 87244, 78368, 30484, 97, 34370, 220, 99489, 95170, 144370,
          91145}},
        {"if x == 123:\n    print(\"hello\")\n\n",
         {333, 856, 621, 220, 16, 17, 18, 510, 262, 1173, 445, 14990, 5130}},
        {"I'm don't they've won't",
         {40, 2776, 1513, 944, 807, 3003, 2765, 944}},
        {"<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|"
         ">",
         {151644, 872, 198, 27, 1805, 16, 29, 151652, 151655, 151653}}};
    for (const auto& [text, expected] : cases)
      assert(tokenizer.Encode(text) == expected);
  }
  const Image image{
      2, 2, {255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0, 127, 63, 21, 255}};
  // Initial noise must not repeat the noise that generated a reference at the
  // same seed and size. Re-editing the result must also use a new stream.
  assert(InitialNoiseSeed(42, {}) == 42);
  assert(InitialNoiseSeed(0, {}) == 0);
  const std::array references{image};
  const auto edit_seed = InitialNoiseSeed(42, references);
  // Independent hashlib SHA-256 golden, little-endian fields and digest.
  assert(edit_seed == 9781428742393689377ULL);
  assert(InitialNoiseSeed(42, references) == edit_seed);
  assert(InitialNoiseSeed(43, references) != edit_seed);
  auto changed = image;
  changed.rgba[0] ^= 1;
  assert(InitialNoiseSeed(42, std::array{changed}) != edit_seed);
  changed = image;
  changed.rgba[3] ^= 1;
  assert(InitialNoiseSeed(42, std::array{changed}) != edit_seed);
  const auto pair_seed = InitialNoiseSeed(42, std::array{image, changed});
  assert(pair_seed != edit_seed);
  assert(pair_seed != InitialNoiseSeed(42, std::array{changed, image}));
  changed = image;
  changed.width = 1;
  changed.height = 4;
  assert(InitialNoiseSeed(42, std::array{changed}) != edit_seed);
  const auto roundtrip = DecodeImage(EncodePng(image));
  assert(roundtrip.width == image.width && roundtrip.height == image.height);
  assert(roundtrip.rgba == image.rgba);
  assert(InitialNoiseSeed(42, std::array{roundtrip}) == edit_seed);
  Image textured{65, 17, std::vector<std::uint8_t>(65 * 17 * 4)};
  for (std::size_t i = 0; i < textured.rgba.size(); ++i)
    textured.rgba[i] = static_cast<std::uint8_t>((i * 73) ^ (i >> 3));
  const auto textured_roundtrip = DecodeImage(EncodePng(textured));
  assert(textured_roundtrip.width == textured.width &&
         textured_roundtrip.height == textured.height);
  assert(textured_roundtrip.rgba == textured.rgba);
  assert(ResizeImage(image, 2, 2).rgba == image.rgba);
  assert(ResizeImage(image, 4, 4).rgba.size() == 64);
  // Pillow RGBA Lanczos: premultiplication, two uint8 passes, unpremultiply.
  const std::vector<std::uint8_t> resized{
      255, 0, 0, 255, 225, 33,  0,  254, 87,  180, 0,  151, 0,   255, 0,  80,
      255, 0, 0, 196, 215, 37,  1,  186, 103, 131, 6,  167, 63,  238, 10, 141,
      255, 0, 0, 59,  179, 46,  10, 98,  120, 81,  16, 187, 132, 106, 22, 220,
      0,   0, 0, 0,   8,   102, 51, 30,  130, 48,  23, 203, 175, 61,  29, 255};
  assert(ResizeImage(image, 4, 4).rgba == resized);
  const auto schedule = FlowSigmas(40, 4096);
  assert(schedule.size() == 41);
  assert(schedule.front() == 1 && schedule.back() == 0);
  assert(std::abs(schedule[39] - 0.02F) < 1e-6F);
  for (std::size_t i = 1; i < schedule.size(); ++i)
    assert(std::isfinite(schedule[i]) && schedule[i] < schedule[i - 1]);
  // Independent checkpoint metadata, not a call to the native schedule builder.
  const auto turbo_index = gufo::json::parse(R"({"sample_sigmas":
    [1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568]})");
  const auto turbo_scheduler = gufo::json::parse(R"({
    "_class_name":"FlowMatchEulerDiscreteScheduler",
    "base_image_seq_len":256,"max_image_seq_len":8192,
    "base_shift":0.5,"max_shift":0.9,"num_train_timesteps":1000,
    "time_shift_type":"exponential","shift":1.0,"shift_terminal":null,
    "use_dynamic_shifting":false,"invert_sigmas":false,
    "stochastic_sampling":false,"use_beta_sigmas":false,
    "use_exponential_sigmas":false,"use_karras_sigmas":false})");
  const auto turbo = ReadSamplingConfig(turbo_index, turbo_scheduler);
  const std::vector<float> expected_turbo{1,         0.978453F, 0.95418F,
                                          0.926626F, 0.89508F,  0.845148F,
                                          0.704534F, 0.414568F, 0};
  assert(turbo.IsTurbo() && turbo.DefaultSteps() == 8);
  assert(turbo.ResolveSteps(0) == 8 && turbo.ResolveSteps(8) == 8);
  assert(turbo.Sigmas(0, 256) == expected_turbo);
  assert(turbo.Sigmas(8, 8160) == expected_turbo);
  const SamplingConfig base;
  assert(!base.IsTurbo() && base.ResolveSteps(0) == 40);
  assert(base.Sigmas(0, 4096) == schedule);
  auto base_scheduler = turbo_scheduler;
  base_scheduler["use_dynamic_shifting"] = true;
  base_scheduler["shift_terminal"] = 0.02;
  assert(!ReadSamplingConfig(gufo::json::Value::object(), base_scheduler)
              .IsTurbo());
  const auto reject_config = [](const auto& index, const auto& scheduler) {
    bool rejected = false;
    try {
      (void)ReadSamplingConfig(index, scheduler);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    assert(rejected);
  };
  reject_config(turbo_index, base_scheduler);
  reject_config(gufo::json::Value::object(), turbo_scheduler);
  for (const auto* sigmas :
       {"[]", "[1,0.5]", "[1,1,0.9,0.8,0.7,0.6,0.5,0.4]",
        "[1,0.9,0.8,0.7,0.6,0.5,0.4,0]", "[1,0.9,0.8,0.7,0.6,0.5,0.4,1e-100]",
        "[1,0.9,0.8,0.7,0.6,0.5,0.4,\"x\"]"})
    reject_config(
        gufo::json::parse(std::string("{\"sample_sigmas\":") + sigmas + "}"),
        turbo_scheduler);
  for (const int steps : {2, 7, 9, 40, 100}) {
    bool rejected = false;
    try {
      (void)turbo.ResolveSteps(steps);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  ValidateRequest(Request{.prompt = "a red cube"});
  for (const auto invalid :
       {Request{.width = 31}, Request{.steps = 1}, Request{.height = 8192}}) {
    bool rejected = false;
    try {
      ValidateRequest(invalid);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  ValidateRequest(Request{.prompt = "a red cube",
                          .images = {},
                          .loras = {{"style", 0.8F}, {"ink", -10}}});
  for (auto loras : {std::vector<Lora>(17), std::vector<Lora>{{"", 1}},
                     std::vector<Lora>{{"style", 1}, {"style", 0.5F}},
                     std::vector<Lora>{{"style", NAN}},
                     std::vector<Lora>{{"style", INFINITY}},
                     std::vector<Lora>{{"style", 10.5F}}}) {
    for (std::size_t i = 0; loras.size() == 17 && i < loras.size(); ++i)
      loras[i].name = "a" + std::to_string(i);
    bool rejected = false;
    try {
      ValidateRequest(
          Request{.prompt = "a red cube", .images = {}, .loras = loras});
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  int calls = 0;
  gufo::server::ImageService service(
      {}, "image-test", [&](const Request& request, const CancellationCheck&) {
        ++calls;
        assert(request.prompt == "a red cube");
        assert(request.width == 256 && request.height == 256);
        assert(request.seed == static_cast<std::uint64_t>(calls + 40));
        return Result{.image = image};
      });
  gufo::server::HttpRequest http;
  http.method = "POST";
  http.path = "/v1/images/generations";
  http.headers = {{"Content-Type", "application/json"}};
  http.body =
      R"({"model":"image-test","prompt":"a red cube","size":"256x256","n":2,"seed":41})";
  const auto response = gufo::server::HandleImageApiRequest(http, service);
  assert(response.status == 200 && calls == 2);
  const auto json = gufo::json::parse(response.body);
  assert(json.find("data")->size() == 2);
  for (const auto body :
       {R"({"prompt":)", R"({"prompt":"a red cube","seed":-1})",
        R"({"prompt":"a red cube","n":1.5})",
        R"({"prompt":"a red cube","n":0})",
        R"({"prompt":"a red cube","output_format":"webp"})",
        R"({"prompt":"a red cube","stream":true})",
        R"({"prompt":"a red cube","size":"256x256junk"})",
        R"({"prompt":"a red cube","unknown":true})"}) {
    http.body = body;
    assert(gufo::server::HandleImageApiRequest(http, service).status == 400);
  }
  http.body = R"({"prompt":"a red cube","model":"unknown"})";
  assert(gufo::server::HandleImageApiRequest(http, service).status == 404);
  http.path = "/v1/images/edits";
  http.headers = {
      {"Content-Type", "multipart/form-data; boundary=\"boundary\""}};
  const auto png = EncodePng(image);
  http.body =
      "--boundary\r\nContent-Disposition: form-data; name=\"image[]\"; "
      "filename=\"x.png\"\r\n"
      "Content-Type: image/png\r\n\r\n";
  http.body.append(reinterpret_cast<const char*>(png.data()), png.size());
  for (const auto& [name, value] : {std::pair{"model", "image-test"},
                                    {"prompt", "a red cube"},
                                    {"size", "256x256"},
                                    {"seed", "43"}})
    http.body += "\r\n--boundary\r\nContent-Disposition: form-data; name=\"" +
                 std::string(name) + "\"\r\n\r\n" + value;
  http.body += "\r\n--boundary--\r\n";
  assert(gufo::server::HandleImageApiRequest(http, service).status == 200 &&
         calls == 3);
  http.body.resize(http.body.size() - 8);
  assert(gufo::server::HandleImageApiRequest(http, service).status == 400);

  // Image to image: the start image binds its own noise stream.
  assert(InitialNoiseSeed(42, {}, &image) != 42);
  assert(InitialNoiseSeed(42, {}, &image) != edit_seed);
  assert(InitialNoiseSeed(42, references, &image) !=
         InitialNoiseSeed(42, {}, &image));
  // ComfyUI denoise: the last steps + 1 sigmas of int(steps / strength).
  assert(DenoiseSigmas(40, 1.0, 4096) == schedule);
  const auto full80 = DenoiseSigmas(80, 1.0, 4096);
  const auto half = DenoiseSigmas(40, 0.5, 4096);
  assert(half.size() == 41 && half.back() == 0);
  assert(std::equal(half.begin(), half.end(), full80.end() - 41));
  assert(DenoiseSigmas(25, 0.6, 1024).size() == 26);
  assert(DenoiseSigmas(25, 0.6, 1024).front() ==
         DenoiseSigmas(41, 1.0, 1024)[16]);
  for (const auto invalid : {Request{.guidance = 21}, Request{.strength = 0.5},
                             Request{.strength = 0}, Request{.guidance = -1}}) {
    bool rejected = false;
    try {
      ValidateRequest(invalid);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  ValidateRequest(Request{.init_image = image, .strength = 0.5});

  Request seen;
  gufo::server::ImageService extended(
      {}, "image-test", [&](const Request& request, const CancellationCheck&) {
        seen = request;
        return Result{.image = image};
      });
  http.path = "/v1/images/generations";
  http.headers = {{"Content-Type", "application/json"}};
  // Defaults stay the unguided full-strength generation.
  http.body = R"({"prompt":"a red cube","size":"256x256","seed":1})";
  assert(gufo::server::HandleImageApiRequest(http, extended).status == 200);
  assert(seen.guidance == 1 && seen.negative_prompt.empty() &&
         seen.strength == 1 && !seen.init_image);
  http.body =
      R"({"prompt":"a red cube","size":"256x256","gufo":{"negative_prompt":"blurry","guidance":2.5}})";
  assert(gufo::server::HandleImageApiRequest(http, extended).status == 200);
  assert(seen.guidance == 2.5F && seen.negative_prompt == "blurry");
  std::string encoded;
  constexpr char kDigits[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (std::size_t i = 0; i < png.size(); i += 3) {
    const std::uint32_t v = png[i] << 16 |
                            (i + 1 < png.size() ? png[i + 1] << 8 : 0) |
                            (i + 2 < png.size() ? png[i + 2] : 0);
    const std::size_t digits = std::min<std::size_t>(png.size() - i, 3) + 1;
    for (std::size_t k = 0; k < 4; ++k)
      encoded += k < digits ? kDigits[(v >> (18 - 6 * k)) & 63] : '=';
  }
  http.body =
      R"({"prompt":"a red cube","gufo":{"strength":0.5,"init_image":")" +
      encoded + R"("}})";
  assert(gufo::server::HandleImageApiRequest(http, extended).status == 200);
  assert(seen.strength == 0.5 && seen.init_image &&
         seen.init_image->rgba == image.rgba && seen.images.empty());
  // A 2x2 start image rounds to the smallest 32-pixel canvas.
  assert(seen.width == 32 && seen.height == 32);
  for (const auto body :
       {R"({"prompt":"a","gufo":{"guidance":"2"}})",
        R"({"prompt":"a","gufo":{"strength":0.5}})",
        R"({"prompt":"a","gufo":{"unknown":1}})",
        R"({"prompt":"a","gufo":{"init_image":"https://example.com/x.png"}})",
        R"({"prompt":"a","gufo":[]})"}) {
    http.body = body;
    assert(gufo::server::HandleImageApiRequest(http, extended).status == 400);
  }
  // Multipart: one "image" plus gufo.strength is the start image.
  http.path = "/v1/images/edits";
  http.headers = {
      {"Content-Type", "multipart/form-data; boundary=\"boundary\""}};
  http.body =
      "--boundary\r\nContent-Disposition: form-data; name=\"image\"; "
      "filename=\"x.png\"\r\n"
      "Content-Type: image/png\r\n\r\n";
  http.body.append(reinterpret_cast<const char*>(png.data()), png.size());
  for (const auto& [name, value] : {std::pair{"prompt", "a red cube"},
                                    {"size", "256x256"},
                                    {"gufo.strength", "0.5"},
                                    {"gufo.guidance", "3"},
                                    {"gufo.negative_prompt", "blurry"}})
    http.body += "\r\n--boundary\r\nContent-Disposition: form-data; name=\"" +
                 std::string(name) + "\"\r\n\r\n" + value;
  http.body += "\r\n--boundary--\r\n";
  assert(gufo::server::HandleImageApiRequest(http, extended).status == 200);
  assert(seen.strength == 0.5 && seen.init_image && seen.images.empty() &&
         seen.guidance == 3 && seen.negative_prompt == "blurry" &&
         seen.width == 256);

  // Masked editing: mask channels, cell weights, blend and request parsing.
  assert(ParseMaskChannel("auto") == MaskChannel::kAuto);
  assert(ParseMaskChannel("alpha") == MaskChannel::kAlpha);
  assert(ParseMaskChannel("blue") == MaskChannel::kBlue);
  for (const auto* bad : {"", "Alpha", "luma", "red "}) {
    bool rejected = false;
    try {
      (void)ParseMaskChannel(bad);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  const auto solid = [](int w, int h, std::array<std::uint8_t, 4> rgba) {
    Image result{w, h, {}};
    for (int i = 0; i < w * h; ++i)
      result.rgba.insert(result.rgba.end(), rgba.begin(), rgba.end());
    return result;
  };
  const auto set = [](Image& target, int x, int y,
                      std::array<std::uint8_t, 4> rgba) {
    std::copy(rgba.begin(), rgba.end(),
              target.rgba.begin() + (std::size_t(y) * target.width + x) * 4);
  };
  // OpenAI mask: one transparent pixel edits exactly its 16 x 16 cell.
  auto openai = solid(64, 64, {0, 0, 0, 255});
  set(openai, 40, 8, {0, 0, 0, 0});
  assert((MaskWeights(openai, MaskChannel::kAuto, 64, 64) ==
          std::vector<float>{0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  // Half the size: 32 mask pixels per cell, still exactly one cell.
  assert((MaskWeights(openai, MaskChannel::kAlpha, 32, 32) ==
          std::vector<float>{0, 1, 0, 0}));
  // Twice the size: a cell covers 8 x 8 mask pixels.
  const auto doubled = MaskWeights(openai, MaskChannel::kAlpha, 128, 128);
  assert(doubled.size() == 64);
  for (int r = 0; r < 8; ++r)
    for (int c = 0; c < 8; ++c)
      assert(doubled[r * 8 + c] == ((r == 1 && (c == 5)) ? 1.0F : 0.0F));
  // Partial alpha is a partial weight; the cell takes its largest weight.
  set(openai, 41, 9, {0, 0, 0, 128});
  set(openai, 3, 60, {0, 0, 0, 128});
  const auto partial = MaskWeights(openai, MaskChannel::kAuto, 64, 64);
  assert(partial[2] == 1 && partial[12] == 127.0F / 255.0F);
  // Opaque masks: auto reads luminance (white edits), alpha keeps everything.
  auto painted = solid(32, 48, {0, 0, 0, 255});
  set(painted, 31, 47, {255, 255, 255, 255});
  set(painted, 0, 0, {255, 0, 0, 255});
  const auto luma = MaskWeights(painted, MaskChannel::kAuto, 32, 48);
  assert(luma.size() == 6 && luma[5] == 1 && luma[1] == 0);
  assert(luma[0] == 76.0F / 255.0F);  // (255 * 299 + 500) / 1000
  assert(MaskWeights(painted, MaskChannel::kRed, 32, 48)[0] == 1);
  assert(MaskWeights(painted, MaskChannel::kGreen, 32, 48)[0] == 0);
  for (const float w : MaskWeights(painted, MaskChannel::kAlpha, 32, 48))
    assert(w == 0);
  // A 1 x 1 mask covers every cell.
  for (const float w :
       MaskWeights(solid(1, 1, {0, 0, 0, 0}), MaskChannel::kAuto, 96, 32))
    assert(w == 1);
  for (const auto [w, h] : {std::pair{8, 16}, {24, 32}, {32, 40}, {0, 32}}) {
    bool rejected = false;
    try {
      (void)MaskWeights(openai, MaskChannel::kAuto, w, h);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  // Blend: weight 1 keeps the sample, 0 the start image, exactly.
  std::vector<float> sample{1.5F, -2, 3, 4, 0.25F, -8};
  const std::vector<float> kept{9, 9, 7, 7, -1, 2};
  BlendMasked(sample, kept, std::vector<float>{1, 0, 0.5F}, 2);
  assert((sample == std::vector<float>{1.5F, -2, 7, 7, -0.375F, -3}));
  {
    bool rejected = false;
    try {
      BlendMasked(sample, kept, std::vector<float>{1, 0}, 2);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  // Spread: linear falloff of 1 / (radius + 1) per cell, Chebyshev distance.
  const std::vector<float> center{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  assert(SpreadMask(center, 5, 5, 0) == center);
  const auto spread = SpreadMask(center, 5, 5, 2);
  const auto near = [](float a, float b) { return std::abs(a - b) < 1e-6F; };
  assert(spread[12] == 1 && near(spread[0], 1.0F / 3.0F) &&
         near(spread[6], 2.0F / 3.0F) && spread[8] == spread[6]);
  const auto wide = SpreadMask(center, 5, 5, 4);  // never beyond the grid
  assert(near(wide[0], 0.6F) && near(wide[1], 0.6F) && near(wide[7], 0.8F));
  // Composite: mask pixel nearest to each output pixel, feathered in pixels.
  auto strip = solid(4, 1, {0, 0, 0, 255});
  set(strip, 0, 0, {0, 0, 0, 0});
  const auto exact = CompositeWeights(strip, MaskChannel::kAuto, 8, 1, 0);
  assert((exact == std::vector<float>{1, 1, 0, 0, 0, 0, 0, 0}));
  const auto soft = CompositeWeights(strip, MaskChannel::kAuto, 8, 1, 3);
  assert(soft[1] == 1 && soft[2] == 0.75F && soft[3] == 0.5F &&
         soft[4] == 0.25F && soft[5] == 0 && soft[7] == 0);
  Image generated = solid(8, 1, {200, 100, 0, 255});
  const Image original = solid(8, 1, {0, 100, 200, 255});
  CompositeMasked(generated, original, soft);
  assert(generated.rgba[0] == 200 && generated.rgba[4 * 2] == 150 &&
         generated.rgba[4 * 2 + 2] == 50 && generated.rgba[4 * 3] == 100 &&
         generated.rgba[4 * 7 + 2] == 200);
  assert(std::equal(generated.rgba.begin() + 20, generated.rgba.end(),
                    original.rgba.begin() + 20));
  {
    bool rejected = false;
    try {
      CompositeMasked(generated, solid(4, 1, {0, 0, 0, 255}), soft);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  ValidateRequest(Request{.init_image = image, .mask = openai});
  ValidateRequest(Request{.init_image = image,
                          .mask = openai,
                          .mask_composite = false,
                          .mask_feather = 0,
                          .mask_blur = 0});
  ValidateRequest(Request{.init_image = image, .strength = 0.4, .mask = openai});
  for (const auto& invalid :
       {Request{.mask = openai},
        Request{.init_image = image, .mask_channel = MaskChannel::kRed},
        Request{.init_image = image, .mask_composite = false},
        Request{.init_image = image, .mask_blur = 0},
        Request{.init_image = image, .mask = openai, .mask_feather = 65},
        Request{.init_image = image, .mask = openai, .mask_blur = 5}}) {
    bool rejected = false;
    try {
      ValidateRequest(invalid);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  // Multipart OpenAI edit: "image" + "mask"; the image stays a reference and
  // is the start image; size keeps its canvas.
  const auto mask_png = EncodePng(openai);
  const auto multipart = [&](const std::vector<std::pair<std::string,
                                                         std::string>>& fields,
                             bool with_image, bool with_mask, int masks = 1) {
    std::string body;
    const auto file = [&](const char* name, const std::vector<std::uint8_t>& b) {
      body += std::string("--boundary\r\nContent-Disposition: form-data; "
                          "name=\"") +
              name + "\"; filename=\"x.png\"\r\nContent-Type: image/png\r\n\r\n";
      body.append(reinterpret_cast<const char*>(b.data()), b.size());
      body += "\r\n";
    };
    if (with_image)
      file("image", png);
    for (int i = 0; with_mask && i < masks; ++i)
      file("mask", mask_png);
    for (const auto& [name, value] : fields)
      body += "--boundary\r\nContent-Disposition: form-data; name=\"" + name +
              "\"\r\n\r\n" + value + "\r\n";
    return body + "--boundary--\r\n";
  };
  http.path = "/v1/images/edits";
  http.headers = {{"Content-Type", "multipart/form-data; boundary=boundary"}};
  http.body = multipart({{"prompt", "a red cube"}}, true, true);
  assert(gufo::server::HandleImageApiRequest(http, extended).status == 200);
  assert(seen.mask && seen.mask->rgba == openai.rgba && seen.init_image &&
         seen.init_image->rgba == image.rgba && seen.images.size() == 1 &&
         seen.strength == 1 && seen.mask_channel == MaskChannel::kAuto &&
         seen.width == 32 && seen.height == 32);
  http.body = multipart({{"prompt", "a red cube"},
                         {"size", "256x256"},
                         {"gufo.strength", "0.6"},
                         {"gufo.mask_channel", "alpha"}},
                        true, true);
  assert(gufo::server::HandleImageApiRequest(http, extended).status == 200);
  assert(seen.mask && seen.init_image && seen.images.empty() &&
         seen.strength == 0.6 && seen.mask_channel == MaskChannel::kAlpha &&
         seen.width == 256);
  assert(seen.mask_composite && seen.mask_feather == kDefaultMaskFeather &&
         seen.mask_blur == kDefaultMaskBlur);
  http.body = multipart({{"prompt", "a red cube"},
                         {"gufo.mask_composite", "false"},
                         {"gufo.mask_feather", "4"},
                         {"gufo.mask_blur", "0"}},
                        true, true);
  assert(gufo::server::HandleImageApiRequest(http, extended).status == 200);
  assert(!seen.mask_composite && seen.mask_feather == 4 &&
         seen.mask_blur == 0);
  for (const auto& body :
       {multipart({{"prompt", "a"}}, false, true),
        multipart({{"prompt", "a"}}, true, true, 2),
        multipart({{"prompt", "a"}, {"gufo.mask_channel", "hue"}}, true, true),
        multipart({{"prompt", "a"}, {"gufo.mask_channel", "red"}}, true,
                  false),
        multipart({{"prompt", "a"}, {"gufo.mask_composite", "1"}}, true, true),
        multipart({{"prompt", "a"}, {"gufo.mask_feather", "2.5"}}, true, true),
        multipart({{"prompt", "a"}, {"gufo.mask_blur", "2"}}, true, false),
        multipart({{"prompt", "a"}, {"gufo.mask_release", "0.2"}}, true, true)}) {
    http.body = body;
    assert(gufo::server::HandleImageApiRequest(http, extended).status == 400);
  }
  // JSON: gufo.init_image + gufo.mask (base64), no references.
  std::string mask_encoded;
  for (std::size_t i = 0; i < mask_png.size(); i += 3) {
    const std::uint32_t v =
        mask_png[i] << 16 |
        (i + 1 < mask_png.size() ? mask_png[i + 1] << 8 : 0) |
        (i + 2 < mask_png.size() ? mask_png[i + 2] : 0);
    const std::size_t digits = std::min<std::size_t>(mask_png.size() - i, 3) + 1;
    for (std::size_t k = 0; k < 4; ++k)
      mask_encoded += k < digits ? kDigits[(v >> (18 - 6 * k)) & 63] : '=';
  }
  http.path = "/v1/images/generations";
  http.headers = {{"Content-Type", "application/json"}};
  http.body = R"({"prompt":"a red cube","size":"64x64","gufo":{"init_image":")" +
              encoded + R"(","mask":")" + mask_encoded +
              R"(","mask_channel":"auto"}})";
  assert(gufo::server::HandleImageApiRequest(http, extended).status == 200);
  assert(seen.mask && seen.mask->rgba == openai.rgba && seen.init_image &&
         seen.images.empty() && seen.width == 64 && seen.strength == 1);
  http.body = R"({"prompt":"a","gufo":{"mask":")" + mask_encoded + R"("}})";
  assert(gufo::server::HandleImageApiRequest(http, extended).status == 400);
  // Requests without a mask are unchanged.
  http.body = R"({"prompt":"a red cube","size":"256x256","seed":1})";
  assert(gufo::server::HandleImageApiRequest(http, extended).status == 200);
  assert(!seen.mask && !seen.init_image &&
         seen.mask_channel == MaskChannel::kAuto);

  int turbo_calls = 0;
  std::vector<Request> turbo_requests;
  gufo::server::ImageService turbo_service(
      {}, "turbo-test",
      [&](const Request& request, const CancellationCheck&) {
        assert(request.steps == 8);
        turbo_requests.push_back(request);
        ++turbo_calls;
        return Result{image, {}};
      },
      turbo);
  http.path = "/v1/images/generations";
  http.headers = {{"Content-Type", "application/json"}};
  for (const auto* body : {R"({"prompt":"cube","size":"256x256"})",
                           R"({"prompt":"cube","steps":8,"size":"256x256"})"}) {
    http.body = body;
    assert(gufo::server::HandleImageApiRequest(http, turbo_service).status ==
           200);
  }
  assert(turbo_calls == 2);
  for (const int steps : {0, 2, 7, 9, 40, 100}) {
    http.body = "{\"prompt\":\"cube\",\"steps\":" + std::to_string(steps) + "}";
    assert(gufo::server::HandleImageApiRequest(http, turbo_service).status ==
           400);
  }
  http.path = "/v1/images/edits";
  assert(gufo::server::HandleImageApiRequest(http, turbo_service).status ==
         400);
  assert(turbo_calls == 2);
  // Editing shares the ordered-reference path, including RGBA, masks,
  // background controls and independent outputs; only the schedule differs.
  const auto edit_body = [&](const std::vector<Image>& inputs, int steps,
                             const std::string& background, int count) {
    std::string body;
    for (const auto& input : inputs) {
      const auto bytes = EncodePng(input);
      body +=
          "--boundary\r\nContent-Disposition: form-data; "
          "name=\"image[]\"; filename=\"reference.png\"\r\n"
          "Content-Type: image/png\r\n\r\n";
      body.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      body += "\r\n";
    }
    std::vector<std::pair<std::string, std::string>> fields{
        {"model", "turbo-test"},
        {"prompt", "edit image one using image two"},
        {"size", "256x256"},
        {"seed", "77"},
        {"background", background},
        {"n", std::to_string(count)}};
    if (steps >= 0)
      fields.emplace_back("steps", std::to_string(steps));
    for (const auto& [name, value] : fields)
      body += "--boundary\r\nContent-Disposition: form-data; name=\"" + name +
              "\"\r\n\r\n" + value + "\r\n";
    return body + "--boundary--\r\n";
  };
  http.headers = {{"Content-Type", "multipart/form-data; boundary=boundary"}};
  Image mask{
      2,
      2,
      {255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255}};
  for (const int steps : {-1, 8}) {
    http.body = edit_body({image, mask}, steps, "transparent", 2);
    const auto edited =
        gufo::server::HandleImageApiRequest(http, turbo_service);
    assert(edited.status == 200);
    assert(gufo::json::parse(edited.body).find("data")->size() == 2);
    for (std::size_t i = turbo_requests.size() - 2; i < turbo_requests.size();
         ++i) {
      const auto& captured = turbo_requests[i];
      assert(captured.steps == 8 && captured.images.size() == 2);
      assert(captured.images[0].rgba == image.rgba);
      assert(captured.images[1].rgba == mask.rgba);
      assert(captured.seed == 77 + i % 2);
      assert(captured.prompt.find("RGBA image with transparency") !=
             std::string::npos);
    }
  }
  http.body = edit_body(std::vector<Image>(10, image), 8, "opaque", 1);
  assert(gufo::server::HandleImageApiRequest(http, turbo_service).status ==
         200);
  assert(turbo_requests.back().images.size() == 10);
  const auto accepted = turbo_calls;
  for (const int steps : {0, 7, 9, 40}) {
    http.body = edit_body({image}, steps, "auto", 1);
    assert(gufo::server::HandleImageApiRequest(http, turbo_service).status ==
           400);
  }
  http.body = edit_body(std::vector<Image>(11, image), 8, "auto", 1);
  assert(gufo::server::HandleImageApiRequest(http, turbo_service).status ==
         400);
  http.body = edit_body({}, 8, "auto", 1);
  assert(gufo::server::HandleImageApiRequest(http, turbo_service).status ==
         400);
  assert(turbo_calls == accepted);
  // Adapters: registered by name only (an injected runner opens no files).
  std::vector<Lora> seen_loras;
  gufo::server::ImageService adapters(
      {}, "image-test",
      [&](const Request& request, const CancellationCheck&) {
        seen_loras = request.loras;
        return Result{.image = image, .metrics = {}};
      },
      {.files = {{"style", "/nonexistent/style.safetensors"},
                 {"ink", "/x/ink.safetensors"}},
       .defaults = {{"style", 0.8F}}});
  http.path = "/v1/images/generations";
  http.headers = {{"Content-Type", "application/json"}};
  const auto generate = [&](const std::string& loras) {
    http.body = R"({"prompt":"a red cube","size":"256x256")" +
                (loras.empty() ? "" : ",\"loras\":" + loras) + "}";
    return gufo::server::HandleImageApiRequest(http, adapters).status;
  };
  const std::vector<Lora> defaults{{"style", 0.8F}};
  const std::vector<Lora> both{{"ink", 0.8F}, {"style", 1}};
  const std::vector<Lora> negative{{"ink", -0.25F}};
  assert(generate("") == 200 && seen_loras == defaults);
  assert(generate("[]") == 200 && seen_loras.empty());
  assert(generate(R"([{"name":"ink","scale":0.8},{"name":"style"}])") == 200 &&
         seen_loras == both);
  seen_loras = {};
  for (const auto& bad : std::vector<std::string>{
           R"([{"name":"unknown"}])", R"([{"name":"ink"},{"name":"ink"}])",
           R"([{"name":"ink","scale":"x"}])", R"([{"name":"ink","path":"/x"}])",
           R"({"name":"ink"})", R"("ink")", R"([{"scale":1}])", R"(["ink"])",
           R"([{"name":"ink","scale":11}])", R"([{"name":"ink","scale":1e40}])",
           R"([{"name":"ink","scale":1e300}])"})
    assert(generate(bad) == 400 && seen_loras.empty());
  // More than 16 distinct registered adapters.
  gufo::server::ImageLoraOptions seventeen;
  std::string many = "[";
  for (int i = 0; i < 17; ++i) {
    const auto name = "a" + std::to_string(i);
    seventeen.files.emplace_back(name, "/x/" + name + ".safetensors");
    many += std::string(i ? "," : "") + R"({"name":")" + name + R"("})";
  }
  gufo::server::ImageService capped(
      {}, "image-test",
      [&](const Request&, const CancellationCheck&) {
        return Result{.image = image, .metrics = {}};
      },
      std::move(seventeen));
  http.body = R"({"prompt":"a red cube","loras":)" + many + "]}";
  const auto over = gufo::server::HandleImageApiRequest(http, capped);
  assert(over.status == 400 &&
         over.body.find("at most 16") != std::string::npos);
  http.body = R"({"prompt":"a red cube","loras":[{"name":"unknown"}]})";
  assert(gufo::server::HandleImageApiRequest(http, adapters)
             .body.find("unknown LoRA adapter: unknown") != std::string::npos);
  // Multipart edits carry `loras` as a JSON field.
  http.path = "/v1/images/edits";
  http.headers = {
      {"Content-Type", "multipart/form-data; boundary=\"boundary\""}};
  http.body =
      "--boundary\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\n";
  http.body.append(reinterpret_cast<const char*>(png.data()), png.size());
  for (const auto& [name, value] :
       {std::pair{"prompt", "a red cube"},
        {"size", "256x256"},
        {"loras", R"([{"name":"ink","scale":-0.25}])"}})
    http.body += "\r\n--boundary\r\nContent-Disposition: form-data; name=\"" +
                 std::string(name) + "\"\r\n\r\n" + value;
  http.body += "\r\n--boundary--\r\n";
  assert(gufo::server::HandleImageApiRequest(http, adapters).status == 200 &&
         seen_loras == negative);
  http.method = "GET";
  http.path = "/sdapi/v1/loras";
  http.headers = {};
  http.body.clear();
  const auto listed = gufo::server::HandleImageLorasRequest(http, adapters);
  assert(listed.status == 200);
  const auto list = gufo::json::parse(listed.body);
  assert(list.is_array() && list.size() == 2);
  const auto& style = list.items()[0];
  const auto& ink = list.items()[1];
  assert(style.member_str("name") == "style" &&
         style.member_str("alias") == "style" &&
         style.member_str("path") == "style.safetensors");
  assert(ink.member_str("path") == "ink.safetensors");
  assert(style.find("metadata")->is_object());
  // Listed as the shortest float text; `active_scale` uses the same formatting.
  assert(style.find("default_scale")->as_double() == 0.8);
  assert(ink.find("default_scale")->is_null());
  // Only a loaded model merges adapters; an injected runner lists none.
  assert(style.find("active_scale")->is_null() &&
         ink.find("active_scale")->is_null());
  http.method = "POST";
  assert(gufo::server::HandleImageLorasRequest(http, adapters).status == 400);
  for (const auto& options : std::vector<gufo::server::ImageLoraOptions>{
           {.files = {{"a b", "/x"}}, .defaults = {}},
           {.files = {{"", "/x"}}, .defaults = {}},
           {.files = {{std::string(65, 'a'), "/x"}}, .defaults = {}},
           {.files = {{"a", "/x"}, {"a", "/y"}}, .defaults = {}},
           {.files = {{"a", "/x"}}, .defaults = {{"b", 1}}},
           {.files = {{"a", "/x"}}, .defaults = {{"a", 1}, {"a", 2}}},
           {.files = {{"a", "/x"}}, .defaults = {{"a", 12}}}}) {
    bool rejected = false;
    try {
      gufo::server::ImageService invalid(
          {}, "image-test",
          [&](const Request&, const CancellationCheck&) { return Result{}; },
          options);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  // Large requests keep the text encoder only with a known, sufficient
  // headroom beside it; otherwise (or when forced) it is released as before.
  {
    using hip::KeepsTextEncoder;
    using hip::kKeepTextEncoderMarginBytes;
    constexpr std::uint64_t GiB = 1ULL << 30;
    // Measured keyframe job (9 references): need 23.7 GB, 45 GB free with
    // the encoder resident.
    static_assert(KeepsTextEncoder(23695ULL << 20, 45 * GiB, false));
    static_assert(!KeepsTextEncoder(23695ULL << 20, 45 * GiB, true));
    static_assert(KeepsTextEncoder(
        10 * GiB, 10 * GiB + kKeepTextEncoderMarginBytes, false));
    static_assert(!KeepsTextEncoder(
        10 * GiB, 10 * GiB + kKeepTextEncoderMarginBytes - 1, false));
    static_assert(!KeepsTextEncoder(30 * GiB, 20 * GiB, false));
    static_assert(!KeepsTextEncoder(GiB, UINT64_MAX, false));
  }
  // Regional prompts: cells, weights, crops and the ComfyUI-style blend.
  {
    // 1344 x 768 = 84 x 48 cells. A 268-pixel column (y 250..768) as an
    // area rounds outward to cells 16..33 x 15..47 and fades inside the
    // image over kRegionEdgeCells.
    Region column{.prompt = "B",
                  .images = {},
                  .strength = 2,
                  .area = RegionArea{.x = 268, .y = 250, .width = 268,
                                     .height = 518, .fraction = false},
                  .mask = {},
                  .mask_bounds = false};
    auto cells = PlanRegion(column, 1344, 768);
    assert(cells.crop && cells.top == 15 && cells.left == 16 &&
           cells.rows == 33 && cells.cols == 18 && !cells.Empty());
    const auto at = [&](const RegionCells& c, int r, int k) {
      return c.weights[static_cast<std::size_t>(r) * c.cols + k];
    };
    // Bottom edge is the image border (no fade), top and sides fade.
    assert(at(cells, 32, 9) == 2.0F);
    assert(at(cells, 0, 9) == 2.0F / 4);
    assert(at(cells, 20, 0) == 2.0F / 4 && at(cells, 20, 17) == 2.0F / 4);
    assert(at(cells, 0, 0) == 2.0F / 16);
    assert(at(cells, 20, 3) == 2.0F && at(cells, 3, 9) == 2.0F);
    // Fractions round to the nearest cell (ConditioningSetAreaPercentage).
    Region fraction = column;
    fraction.area = RegionArea{.x = 0.5, .y = 0, .width = 0.5, .height = 1,
                               .fraction = true};
    cells = PlanRegion(fraction, 1344, 768);
    assert(cells.crop && cells.top == 0 && cells.left == 42 &&
           cells.rows == 48 && cells.cols == 42);
    assert(at(cells, 10, 0) == 2.0F / 4 && at(cells, 10, 41) == 2.0F);
    // The whole image is no crop.
    fraction.area = RegionArea{.x = 0, .y = 0, .width = 1, .height = 1,
                               .fraction = true};
    cells = PlanRegion(fraction, 1344, 768);
    assert(!cells.crop && cells.rows == 48 && cells.cols == 84 &&
           at(cells, 0, 0) == 2.0F);
    // A mask (stretched: 84 x 48 values for 1344 x 768) on the whole grid,
    // or on its bounding box with mask_bounds.
    RegionMask mask{84, 48, std::vector<float>(84 * 48, 0)};
    for (int r = 16; r < 48; ++r)
      for (int c = 0; c < 17; ++c)
        mask.values[r * 84 + c] = c == 16 ? 0.5F : 1.0F;
    Region masked{.prompt = "A",
                  .images = {},
                  .strength = 1,
                  .area = {},
                  .mask = mask,
                  .mask_bounds = false};
    cells = PlanRegion(masked, 1344, 768);
    assert(!cells.crop && cells.rows == 48 && cells.cols == 84);
    assert(at(cells, 20, 0) == 1.0F && at(cells, 20, 16) == 0.5F &&
           at(cells, 20, 17) == 0.0F && at(cells, 15, 0) == 0.0F);
    masked.mask_bounds = true;
    cells = PlanRegion(masked, 1344, 768);
    assert(cells.crop && cells.top == 16 && cells.left == 0 &&
           cells.rows == 32 && cells.cols == 17);
    assert(at(cells, 0, 0) == 1.0F && at(cells, 0, 16) == 0.5F);
    // A mask at another size is stretched and averaged per cell.
    masked.mask = RegionMask{2, 1, {1.0F, 0.0F}};
    masked.mask_bounds = false;
    cells = PlanRegion(masked, 64, 32);
    assert(cells.weights == (std::vector<float>{1, 1, 0, 0, 1, 1, 0, 0}));
    // Mask and area: evaluated on the area, weighted by the mask, no fade.
    masked.mask = mask;
    masked.area = RegionArea{.x = 0, .y = 0, .width = 512, .height = 768,
                             .fraction = false};
    cells = PlanRegion(masked, 1344, 768);
    assert(cells.crop && cells.left == 0 && cells.cols == 32 &&
           cells.rows == 48 && at(cells, 20, 0) == 1.0F &&
           at(cells, 0, 0) == 0.0F);
    // Small crops grow to kMinRegionCells with weight 0 around them.
    Region small{.prompt = "door",
                 .images = {},
                 .strength = 1,
                 .area = RegionArea{.x = 640, .y = 384, .width = 32,
                                    .height = 32, .fraction = false},
                 .mask = {},
                 .mask_bounds = false};
    cells = PlanRegion(small, 1344, 768);
    assert(cells.crop && cells.rows * cells.cols >= kMinRegionCells);
    assert(cells.top <= 24 && cells.top + cells.rows >= 26 &&
           cells.left <= 40 && cells.left + cells.cols >= 42);
    float weight = 0;
    for (const float w : cells.weights)
      weight += w;
    // 2 x 2 cells faded on all four sides: (1/4 * 2/4)^2 each, as ComfyUI
    // shrinks very small areas.
    assert(weight == 1.0F / 16);
    // Empty masks and areas outside the image are skipped.
    masked.area.reset();
    masked.mask = RegionMask{4, 4, std::vector<float>(16, 0)};
    assert(PlanRegion(masked, 1344, 768).Empty());
    masked.mask_bounds = true;
    assert(PlanRegion(masked, 1344, 768).Empty());
    small.area->x = 2000;
    assert(PlanRegion(small, 1344, 768).Empty());

    // Blend: overlapping regions average by weight; uncovered cells are 0.
    // Grid 2 x 3 cells, 2 channels.
    const int channels = 2;
    std::vector<float> sum(6 * channels), total(6);
    RegionCells base{.top = 0, .left = 0, .rows = 2, .cols = 3,
                     .crop = false, .weights = {1, 1, 1, 1, 1, 0}};
    std::vector<float> v1(6 * channels, 1.0F);
    AccumulateRegion(sum, total, v1, 3, channels, base);
    RegionCells crop{.top = 0, .left = 1, .rows = 2, .cols = 1, .crop = true,
                     .weights = {3, 0}};
    AccumulateRegion(sum, total, std::vector<float>{5, 9, 7, 7}, 3, channels,
                     crop);
    const auto blended = FinishRegions(sum, total, channels);
    // Cell (0, 1): (1 * 1 + 3 * 5) / 4 = 4 and (1 + 3 * 9) / 4 = 7.
    assert(blended[2] == 4.0F && blended[3] == 7.0F);
    assert(blended[0] == 1.0F && blended[8] == 1.0F);
    assert(blended[10] == 0.0F && blended[11] == 0.0F);  // cell (1, 2)
    // One region at weight 1 everywhere returns its velocity exactly.
    std::vector<float> s2(6 * channels), t2(6);
    std::vector<float> v2{0.1F, -2, 3.5F, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    AccumulateRegion(s2, t2, v2, 3, channels,
                     RegionCells{.top = 0, .left = 0, .rows = 2, .cols = 3,
                                 .crop = false,
                                 .weights = std::vector<float>(6, 1)});
    assert(FinishRegions(s2, t2, channels) == v2);
    // Crops of the latent, row-major.
    const std::vector<float> latent{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    assert((CropLatent(latent, 3, channels, crop) ==
            std::vector<float>{2, 3, 8, 9}));
    bool rejected = false;
    try {
      AccumulateRegion(sum, total, std::vector<float>(3), 3, channels, crop);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
    assert(FullRegion(64, 32, 0.5F).weights ==
           std::vector<float>(8, 0.5F));

    // Mask images: channel and inversion.
    Image red{2, 1, {255, 0, 0, 255, 0, 0, 0, 255}};
    const auto from_image = RegionMaskFromImage(red, MaskChannel::kRed, false);
    assert(from_image.values == (std::vector<float>{1.0F, 0.0F}));
    assert(RegionMaskFromImage(red, MaskChannel::kRed, true).values ==
           (std::vector<float>{0.0F, 1.0F}));

    // Validation.
    ValidateRequest(Request{.prompt = "office",
                            .regions = {column, masked},
                            .base_strength = 0});
    for (const auto& invalid : std::vector<Request>{
             Request{.prompt = "a", .base_strength = 0.5F},
             Request{.prompt = "a",
                     .regions = std::vector<Region>(kMaxRegions + 1)},
             Request{.prompt = "a",
                     .regions = {Region{.prompt = "b", .strength = 11}}},
             Request{.prompt = "a",
                     .regions = {Region{.prompt = "b", .mask_bounds = true}}},
             Request{.prompt = "a",
                     .regions = {Region{
                         .prompt = "b",
                         .mask = RegionMask{2, 2, {0, 1, 2, 0}}}}},
             Request{.prompt = "a",
                     .regions = {Region{
                         .prompt = "b",
                         .area = RegionArea{.x = 0, .y = 0, .width = 0,
                                            .height = 1, .fraction = true}}}},
             Request{.prompt = "a",
                     .regions = {Region{
                         .prompt = "b",
                         .images = std::vector<Image>(11, image)}}},
             Request{.prompt = "a",
                     .images = std::vector<Image>(10, image),
                     .regions = {Region{
                         .prompt = "b",
                         .images = std::vector<Image>(10, image)},
                                 Region{.prompt = "c",
                                        .images = {image}}}}}) {
      bool bad = false;
      try {
        ValidateRequest(invalid);
      } catch (const std::invalid_argument&) {
        bad = true;
      }
      assert(bad);
    }

    // HTTP: gufo.regions with region_image[] parts and a cell mask.
    Request regional;
    gufo::server::ImageService regions_service(
        {}, "image-test",
        [&](const Request& request, const CancellationCheck&) {
          regional = request;
          return Result{.image = image};
        });
    gufo::server::HttpRequest post;
    post.method = "POST";
    post.path = "/v1/images/edits";
    post.headers = {
        {"Content-Type", "multipart/form-data; boundary=\"boundary\""}};
    const auto region_png = EncodePng(image);
    const auto part = [&](const std::string& name, const std::string& value) {
      post.body += "--boundary\r\nContent-Disposition: form-data; name=\"" +
                   name + "\"\r\n\r\n" + value + "\r\n";
    };
    const auto image_part = [&](const std::string& name) {
      post.body += "--boundary\r\nContent-Disposition: form-data; name=\"" +
                   name +
                   "\"; filename=\"r.png\"\r\nContent-Type: image/png\r\n\r\n";
      post.body.append(reinterpret_cast<const char*>(region_png.data()),
                       region_png.size());
      post.body += "\r\n";
    };
    const auto request_with = [&](const std::string& gufo, int images) {
      post.body.clear();
      part("prompt", "office");
      part("size", "64x32");
      part("gufo", gufo);
      for (int i = 0; i < images; ++i)
        image_part("region_image[]");
      post.body += "--boundary--\r\n";
      return gufo::server::HandleImageApiRequest(post, regions_service).status;
    };
    // 4 x 2 cells; "/wAA/wAAAAA=" = 255 0 0 255 0 0 0 0.
    const std::string gufo_field =
        R"j({"guidance":2.5,"negative_prompt":"extra","base_strength":0.5,"regions":[)j"
        R"j({"prompt":"A (<image1>)","images":1,"mask":{"width":4,"height":2,"data":"/wAA/wAAAAA="},"mask_bounds":true},)j"
        R"j({"prompt":"B (<image1>) and (<image2>)","strength":0.8,"images":2,"area":{"x":0.5,"y":0,"width":0.5,"height":1,"unit":"fraction"}},)j"
        R"j({"prompt":"door"}]})j";
    assert(request_with(gufo_field, 3) == 200);
    assert(regional.prompt == "office" && regional.images.empty());
    assert(regional.base_strength == 0.5F && regional.guidance == 2.5F);
    assert(regional.regions.size() == 3);
    assert(regional.regions[0].images.size() == 1 &&
           regional.regions[0].mask_bounds &&
           regional.regions[0].mask->width == 4 &&
           regional.regions[0].mask->values[0] == 1.0F &&
           regional.regions[0].mask->values[1] == 0.0F &&
           regional.regions[0].mask->values[3] == 1.0F);
    assert(regional.regions[1].images.size() == 2 &&
           regional.regions[1].strength == 0.8F &&
           regional.regions[1].area->fraction &&
           regional.regions[1].area->x == 0.5);
    assert(regional.regions[2].prompt == "door" &&
           regional.regions[2].images.empty() && !regional.regions[2].mask);
    // Parts and counts must agree; unknown keys and bad masks fail.
    assert(request_with(gufo_field, 2) == 400);
    assert(request_with(gufo_field, 4) == 400);
    assert(request_with(R"({"regions":[]})", 1) == 400);
    // JSON generations take regions with inline images.
    post.path = "/v1/images/generations";
    post.headers = {{"Content-Type", "application/json"}};
    post.body =
        R"({"prompt":"office","size":"64x32","gufo":{"regions":[{"prompt":"a","area":{"x":0,"y":0,"width":32,"height":32}}]}})";
    assert(gufo::server::HandleImageApiRequest(post, regions_service).status ==
           200);
    assert(regional.regions.size() == 1 && !regional.regions[0].area->fraction &&
           regional.regions[0].area->width == 32);
    for (const auto* body : {
             R"({"prompt":"a","gufo":{"regions":[{"prompt":"a","color":1}]}})",
             R"({"prompt":"a","gufo":{"regions":[{"prompt":"a","mask":{"width":4,"height":2,"data":"/w=="}}]}})",
             R"({"prompt":"a","gufo":{"regions":[{"prompt":"a","mask_channel":"red"}]}})",
             R"({"prompt":"a","gufo":{"regions":[{"prompt":"a","images":1}]}})",
             R"({"prompt":"a","gufo":{"regions":[{"prompt":"a","images":["https://example.com/x.png"]}]}})",
             R"({"prompt":"a","gufo":{"regions":[{"prompt":"a","area":{"x":0,"y":0,"width":2,"height":1,"unit":"fraction"}}]}})",
             R"({"prompt":"a","gufo":{"regions":{}}})",
             R"({"prompt":"a","gufo":{"base_strength":0.5}})"}) {
      post.body = body;
      assert(gufo::server::HandleImageApiRequest(post, regions_service)
                 .status == 400);
    }
  }
  std::cout << "Qwen-Image image and schedule contracts passed\n";
}
