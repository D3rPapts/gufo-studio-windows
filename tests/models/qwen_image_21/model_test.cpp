#include "src/models/qwen_image_21/model.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/image_api.hpp"
#include "src/core/json.hpp"
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
        {"  cafe\xcc\x81  déjà vu", {220, 51950, 220, 45839, 32514}},
        {"नमस्ते 世界 안녕하세요",
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
  ValidateRequest(Request{.init_image = image, .mask = openai});
  ValidateRequest(Request{.init_image = image, .strength = 0.4, .mask = openai});
  for (const auto& invalid :
       {Request{.mask = openai},
        Request{.init_image = image, .mask_channel = MaskChannel::kRed}}) {
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
  for (const auto& body :
       {multipart({{"prompt", "a"}}, false, true),
        multipart({{"prompt", "a"}}, true, true, 2),
        multipart({{"prompt", "a"}, {"gufo.mask_channel", "hue"}}, true, true),
        multipart({{"prompt", "a"}, {"gufo.mask_channel", "red"}}, true,
                  false)}) {
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
  std::cout << "Qwen-Image image and schedule contracts passed\n";
}
