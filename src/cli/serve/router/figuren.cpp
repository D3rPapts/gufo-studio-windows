// POST /v1/figuren: figure check for film keyframes (see figuren.hpp).
// Written with AI assistance (Claude Code).
#include "src/cli/serve/router/figuren.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <set>
#include <span>
#include <stdexcept>
#include <utility>

#include "src/cli/serve/router/comfy_graph.hpp"
#include "src/core/image.hpp"
#include "src/models/qwen_image_21/model.hpp"

namespace gufo::router::figuren {
namespace {

using server::HttpRequest;
using server::HttpResponse;
using RgbaImage = models::qwen_image_21::Image;

constexpr int kFrameSide = 1344;
constexpr int kHeadSide = 768;
constexpr int kMaxTokens = 6000;
constexpr double kDefaultThreshold = 0.6;
constexpr std::size_t kMaxReferences = 32;

HttpResponse JsonResponse(int status, std::string reason, std::string body) {
  HttpResponse response;
  response.status = status;
  response.reason = std::move(reason);
  response.body = std::move(body);
  response.headers = {{"Content-Type", "application/json"}};
  return response;
}

HttpResponse Error(int status, std::string reason, std::string message,
                   std::string code) {
  json::Value detail = json::Value::object();
  detail["message"] = std::move(message);
  detail["type"] = status >= 500 ? "server_error" : "invalid_request_error";
  detail["code"] = std::move(code);
  json::Value body = json::Value::object();
  body["error"] = std::move(detail);
  return JsonResponse(status, std::move(reason), body.dump());
}

/// A chat the router answered with an error: relayed to the client as is
/// (404 unknown model, 503 busy or memory guard, ...).
struct UpstreamError {
  int status{502};
  std::string body;
  std::string content_type;
};

/// The model answered without the JSON a step needs.
struct AnswerError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

bool StartsWith(std::string_view text, std::string_view prefix) {
  return text.substr(0, prefix.size()) == prefix;
}

/// Python float(): numbers, numeric strings and booleans.
std::optional<double> AsNumber(const json::Value& value) {
  if (value.is_number())
    return value.as_double();
  if (value.is_bool())
    return value.as_bool() ? 1.0 : 0.0;
  if (value.is_string()) {
    const std::string& text = value.str();
    const char* begin = text.c_str();
    char* end = nullptr;
    const double parsed = std::strtod(begin, &end);
    if (end == begin)
      return std::nullopt;
    while (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')
      ++end;
    if (*end != '\0' || !std::isfinite(parsed))
      return std::nullopt;
    return parsed;
  }
  return std::nullopt;
}

/// Uppercased (ASCII; German sharp s as SS) letters A-Z only.
std::string BadgeLetters(std::string_view badge) {
  std::string out;
  for (std::size_t i = 0; i < badge.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(badge[i]);
    if (c == 0xC3 && i + 1 < badge.size() &&
        static_cast<unsigned char>(badge[i + 1]) == 0x9F) {
      out += "SS";
      ++i;
    } else if (c >= 'a' && c <= 'z') {
      out.push_back(static_cast<char>(c - 'a' + 'A'));
    } else if (c >= 'A' && c <= 'Z') {
      out.push_back(static_cast<char>(c));
    }
  }
  return out;
}

std::string UpperAscii(std::string_view text) {
  std::string out(text);
  for (char& c : out)
    if (c >= 'a' && c <= 'z')
      c = static_cast<char>(c - 'a' + 'A');
  return out;
}

/// The first `count` UTF-8 code points.
std::string_view FirstCodePoints(std::string_view text, std::size_t count) {
  std::size_t i = 0;
  for (std::size_t n = 0; n < count && i < text.size(); ++n) {
    ++i;
    while (i < text.size() &&
           (static_cast<unsigned char>(text[i]) & 0xC0) == 0x80)
      ++i;
  }
  return text.substr(0, i);
}

std::string Trim(std::string_view text) {
  const auto space = [](char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
           c == '\v';
  };
  while (!text.empty() && space(text.front()))
    text.remove_prefix(1);
  while (!text.empty() && space(text.back()))
    text.remove_suffix(1);
  return std::string(text);
}

json::Value BoxJson(const Box& box) {
  json::Value out = json::Value::array();
  for (const int v : box)
    out.push_back(v);
  return out;
}

// ---- Images ----

RgbaImage ReadImage(const std::string& input, const char* what) {
  // Raw base64 is wrapped as a data URL; the decoder tells PNG, JPEG and
  // WebP apart by their bytes.
  const std::string url = StartsWith(input, "data:") ||
                                  StartsWith(input, "https://")
                              ? input
                              : "data:image/png;base64," + input;
  core::ImageReadBudget budget;
  core::Image rgb;
  try {
    const std::vector<std::uint8_t> bytes = core::ReadImageUrl(url, budget);
    rgb = core::DecodeImage(bytes);
  } catch (const std::exception& error) {
    throw std::invalid_argument(std::string(what) + ": " + error.what());
  }
  RgbaImage image;
  image.width = static_cast<int>(rgb.width);
  image.height = static_cast<int>(rgb.height);
  const std::size_t pixels = static_cast<std::size_t>(rgb.width) * rgb.height;
  image.rgba.resize(pixels * 4);
  for (std::size_t i = 0; i < pixels; ++i) {
    image.rgba[i * 4] = rgb.pixels[i * 3];
    image.rgba[i * 4 + 1] = rgb.pixels[i * 3 + 1];
    image.rgba[i * 4 + 2] = rgb.pixels[i * 3 + 2];
    image.rgba[i * 4 + 3] = 255;
  }
  return image;
}

RgbaImage Crop(const RgbaImage& image, const Box& rect) {
  RgbaImage out;
  out.width = rect[2] - rect[0];
  out.height = rect[3] - rect[1];
  out.rgba.resize(static_cast<std::size_t>(out.width) * out.height * 4);
  for (int y = 0; y < out.height; ++y) {
    const auto* row = image.rgba.data() +
                      (static_cast<std::size_t>(rect[1] + y) * image.width +
                       rect[0]) *
                          4;
    std::copy(row, row + static_cast<std::size_t>(out.width) * 4,
              out.rgba.data() + static_cast<std::size_t>(y) * out.width * 4);
  }
  return out;
}

/// figuren.py data_url: at most `max_side` (Lanczos like Pillow), PNG.
std::string DataUrl(const RgbaImage& image, int max_side) {
  const auto [width, height] = FitSize(image.width, image.height, max_side);
  const std::vector<std::uint8_t> png = models::qwen_image_21::EncodePng(
      width == image.width && height == image.height
          ? image
          : models::qwen_image_21::ResizeImage(image, width, height));
  return "data:image/png;base64," +
         comfy::Base64Encode(std::string_view(
             reinterpret_cast<const char*>(png.data()), png.size()));
}

json::Value ImagePart(std::string url) {
  json::Value image_url = json::Value::object();
  image_url["url"] = std::move(url);
  json::Value part = json::Value::object();
  part["type"] = "image_url";
  part["image_url"] = std::move(image_url);
  return part;
}

json::Value TextPart(std::string text) {
  json::Value part = json::Value::object();
  part["type"] = "text";
  part["text"] = std::move(text);
  return part;
}

// ---- Model calls ----

struct Reference {
  std::string name;
  RgbaImage head;
  std::string profile;
};

struct Verdict {
  double similarity{0};
  json::Value differences = json::Value::array();
};

class Session {
public:
  Session(const comfy::Fetch& fetch, std::string model)
      : fetch_(fetch), model_(std::move(model)) {}

  /// One chat through the router's own API; throws UpstreamError on a non-2xx
  /// answer.
  std::string Ask(json::Value content, bool think) const {
    json::Value message = json::Value::object();
    message["role"] = "user";
    message["content"] = std::move(content);
    json::Value messages = json::Value::array();
    messages.push_back(std::move(message));
    json::Value body = json::Value::object();
    body["model"] = model_;
    body["messages"] = std::move(messages);
    body["max_tokens"] = kMaxTokens;
    body["temperature"] = 0;
    if (!think) {
      json::Value kwargs = json::Value::object();
      kwargs["enable_thinking"] = false;
      body["chat_template_kwargs"] = std::move(kwargs);
    }
    HttpRequest request;
    request.method = "POST";
    request.path = "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace_back("Content-Type", "application/json");
    const comfy::FetchResult answer = fetch_(request);
    if (answer.status < 200 || answer.status >= 300)
      throw UpstreamError{answer.status == 0 ? 502 : answer.status,
                          answer.body, answer.content_type};
    json::Value parsed;
    try {
      parsed = json::parse(answer.body);
    } catch (const std::exception&) {
      throw AnswerError("chat answer is not JSON");
    }
    const json::Value* choices = parsed.find("choices");
    if (choices == nullptr || !choices->is_array() || choices->empty())
      throw AnswerError("chat answer has no choices");
    const json::Value* message_out = choices->items()[0].find("message");
    if (message_out == nullptr)
      throw AnswerError("chat answer has no message");
    const json::Value* text = message_out->find("content");
    return text != nullptr && text->is_string() ? text->str() : std::string();
  }

  std::vector<Person> FindPeople(const RgbaImage& frame) const {
    const std::string prompt =
        "List every person visible in this film frame. For each person give "
        "the bounding box of the whole visible person, the bounding box of "
        "the head (hair to chin), and the text of a name badge if one is "
        "readable on the person or on the desk directly in front of them. "
        "Coordinates are relative from 0 to 1000 (x0, y0, x1, y1). Answer "
        "only with JSON: [{\"person_box\": [x0,y0,x1,y1], \"head_box\": "
        "[x0,y0,x1,y1], \"badge\": \"NAME or empty\"}], ordered from left to "
        "right.";
    json::Value content = json::Value::array();
    content.push_back(ImagePart(DataUrl(frame, kFrameSide)));
    content.push_back(TextPart(prompt));
    const std::string answer = Ask(std::move(content), false);
    const auto people = JsonFrom(answer, true);
    if (!people.has_value())
      throw AnswerError("no JSON in answer: " + Tail(answer));
    std::vector<Person> out;
    for (const json::Value& entry : people->items()) {
      const json::Value* body = entry.find("person_box");
      const json::Value* head = entry.find("head_box");
      if (body == nullptr || head == nullptr)
        continue;
      const auto box = ToPixels(*body, frame.width, frame.height);
      const auto kopf = ToPixels(*head, frame.width, frame.height);
      if (!box.has_value() || !kopf.has_value())
        continue;
      Person person;
      person.box = *box;
      person.kopf_box = *kopf;
      const json::Value* badge = entry.find("badge");
      if (badge != nullptr && badge->is_string())
        person.schild = Trim(badge->str());
      out.push_back(std::move(person));
    }
    return out;
  }

  Verdict Compare(const RgbaImage& frame, const Box& kopf_box,
                  const Reference& reference, bool think) const {
    const RgbaImage head =
        Crop(frame, HeadCrop(kopf_box, frame.width, frame.height));
    std::string prompt =
        "Image 1 is the reference head of the film character " +
        reference.name +
        ". Image 2 is a head from a new film frame that should show the same "
        "character in the same drawing style.";
    if (!reference.profile.empty())
      prompt += " The character's profile: " + reference.profile + ".";
    prompt +=
        " Compare only identity features: face shape, jaw, nose, eyes and "
        "brows, hair colour and style, beard or stubble, glasses or "
        "sunglasses, skin tone, apparent age, build of neck and shoulders. "
        "Ignore expression, pose, light and framing. Answer only with JSON: "
        "{\"aehnlichkeit\": 0-10 (10 = clearly the same character), "
        "\"abweichungen\": [\"short German phrase per clear difference\"]}.";
    json::Value content = json::Value::array();
    content.push_back(ImagePart(DataUrl(reference.head, kHeadSide)));
    content.push_back(ImagePart(DataUrl(head, kHeadSide)));
    content.push_back(TextPart(std::move(prompt)));
    const std::string answer = Ask(std::move(content), think);
    const auto result = JsonFrom(answer, false);
    if (!result.has_value())
      throw AnswerError("no JSON in answer: " + Tail(answer));
    Verdict verdict;
    double score = 0;
    if (const json::Value* value = result->find("aehnlichkeit")) {
      const auto number = AsNumber(*value);
      if (!number.has_value())
        throw AnswerError("aehnlichkeit is not a number");
      score = *number;
    }
    verdict.similarity = std::nearbyint(score / 10 * 100) / 100;
    if (const json::Value* diff = result->find("abweichungen")) {
      if (diff->is_array())
        verdict.differences = *diff;
      else if (diff->is_string())
        verdict.differences.push_back(diff->str());
    }
    return verdict;
  }

private:
  static std::string Tail(const std::string& text) {
    return text.size() <= 300 ? text : text.substr(text.size() - 300);
  }

  const comfy::Fetch& fetch_;
  std::string model_;
};

struct Request {
  RgbaImage frame;
  std::vector<Reference> references;
  double threshold{kDefaultThreshold};
  std::string model;
};

Request ParseRequest(const std::string& text, const std::string& default_model) {
  json::Value body;
  try {
    body = json::parse(text);
  } catch (const std::exception& error) {
    throw std::invalid_argument(std::string("invalid JSON body: ") +
                                error.what());
  }
  if (!body.is_object())
    throw std::invalid_argument("body must be a JSON object");
  Request request;
  const json::Value* bild = body.find("bild");
  if (bild == nullptr || !bild->is_string() || bild->str().empty())
    throw std::invalid_argument("bild (data URL or base64 image) is required");
  const json::Value* refs = body.find("referenzen");
  if (refs == nullptr || !refs->is_array() || refs->empty())
    throw std::invalid_argument(
        "referenzen must be a non-empty list of {name, kopf}");
  if (refs->size() > kMaxReferences)
    throw std::invalid_argument("too many referenzen");
  const json::Value* profiles = body.find("steckbrief");
  if (profiles != nullptr && !profiles->is_null() && !profiles->is_object())
    throw std::invalid_argument("steckbrief must be an object {name: text}");
  std::set<std::string> seen;
  for (const json::Value& entry : refs->items()) {
    const json::Value* name = entry.find("name");
    const json::Value* kopf = entry.find("kopf");
    if (name == nullptr || !name->is_string() || name->str().empty() ||
        kopf == nullptr || !kopf->is_string() || kopf->str().empty())
      throw std::invalid_argument(
          "every referenz needs a name and a kopf image");
    if (!seen.insert(name->str()).second)
      throw std::invalid_argument("duplicate referenz name: " + name->str());
    Reference reference;
    reference.name = name->str();
    reference.head = ReadImage(kopf->str(), "kopf");
    if (profiles != nullptr && profiles->is_object()) {
      if (const json::Value* profile = profiles->find(reference.name);
          profile != nullptr && profile->is_string())
        reference.profile = profile->str();
    }
    request.references.push_back(std::move(reference));
  }
  if (const json::Value* threshold = body.find("schwelle");
      threshold != nullptr && !threshold->is_null()) {
    if (!threshold->is_number())
      throw std::invalid_argument("schwelle must be a number");
    request.threshold = threshold->as_double();
  }
  request.model = default_model;
  if (const json::Value* model = body.find("modell");
      model != nullptr && !model->is_null()) {
    if (!model->is_string() || model->str().empty())
      throw std::invalid_argument("modell must be a model id");
    request.model = model->str();
  }
  request.frame = ReadImage(bild->str(), "bild");
  return request;
}

}  // namespace

std::string StripThinking(std::string_view text) {
  constexpr std::string_view kOpen = "<think>";
  constexpr std::string_view kClose = "</think>";
  std::string out;
  std::size_t pos = 0;
  while (pos < text.size()) {
    const auto open = text.find(kOpen, pos);
    if (open == std::string_view::npos) {
      out.append(text.substr(pos));
      break;
    }
    const auto close = text.find(kClose, open + kOpen.size());
    if (close == std::string_view::npos) {
      // re.sub leaves an unclosed block alone.
      out.append(text.substr(pos));
      break;
    }
    out.append(text.substr(pos, open - pos));
    pos = close + kClose.size();
  }
  // A closing tag without its opener: everything before it is thinking.
  if (const auto lone = out.rfind(kClose);
      lone != std::string::npos && out.find(kOpen) == std::string::npos)
    out.erase(0, lone + kClose.size());
  return out;
}

std::optional<json::Value> JsonFrom(std::string_view raw, bool want_array) {
  const std::string text = StripThinking(raw);
  const char opener = want_array ? '[' : '{';
  const char closer = want_array ? ']' : '}';
  const auto end = text.rfind(closer);
  if (end == std::string::npos || end == 0)
    return std::nullopt;
  auto start = text.rfind(opener, end - 1);
  while (start != std::string::npos) {
    try {
      json::Value value =
          json::parse(std::string_view(text).substr(start, end + 1 - start));
      if (want_array ? value.is_array() : value.is_object())
        return value;
    } catch (const std::exception&) {
      // Not a complete value from here; try the next opener to the left.
    }
    if (start == 0)
      break;
    start = text.rfind(opener, start - 1);
  }
  return std::nullopt;
}

std::optional<Box> ToPixels(const json::Value& box, int width, int height) {
  if (!box.is_array() || box.size() != 4)
    return std::nullopt;
  std::array<double, 4> v{};
  for (std::size_t i = 0; i < 4; ++i) {
    const auto number = AsNumber(box.items()[i]);
    if (!number.has_value())
      return std::nullopt;
    v[i] = *number;
  }
  const auto scale = [](double value, int size) {
    return static_cast<int>(std::nearbyint(value * size / 1000));
  };
  return Box{scale(v[0], width), scale(v[1], height), scale(v[2], width),
             scale(v[3], height)};
}

Box HeadCrop(const Box& head, int width, int height) {
  const auto floor_div3 = [](int value) {
    return value >= 0 ? value / 3 : -((-value + 2) / 3);
  };
  const int pad_x = floor_div3(head[2] - head[0]);
  const int pad_y = floor_div3(head[3] - head[1]);
  const int x0 = std::clamp(head[0] - pad_x, 0, std::max(0, width - 1));
  const int y0 = std::clamp(head[1] - pad_y, 0, std::max(0, height - 1));
  const int x1 = std::clamp(head[2] + pad_x, x0 + 1, std::max(x0 + 1, width));
  const int y1 = std::clamp(head[3] + pad_y, y0 + 1, std::max(y0 + 1, height));
  return Box{x0, y0, x1, y1};
}

std::pair<int, int> FitSize(int width, int height, int max_side) {
  const int longer = std::max(width, height);
  if (longer <= max_side || longer <= 0)
    return {width, height};
  const double scale = static_cast<double>(max_side) / longer;
  return {std::max(1, static_cast<int>(std::lround(width * scale))),
          std::max(1, static_cast<int>(std::lround(height * scale)))};
}

std::vector<Assignment> Assign(const std::vector<Person>& people,
                               const std::vector<std::string>& names) {
  std::vector<std::size_t> order(people.size());
  for (std::size_t i = 0; i < order.size(); ++i)
    order[i] = i;
  std::stable_sort(order.begin(), order.end(),
                   [&people](std::size_t a, std::size_t b) {
                     return people[a].box[0] < people[b].box[0];
                   });
  std::vector<Assignment> out;
  std::vector<std::string> free = names;
  std::vector<bool> taken(people.size(), false);
  for (const std::size_t index : order) {
    const std::string badge = BadgeLetters(people[index].schild);
    if (badge.empty())
      continue;
    const std::string_view badge3 = FirstCodePoints(badge, 3);
    for (auto it = free.begin(); it != free.end(); ++it) {
      const std::string upper = UpperAscii(*it);
      const std::string_view name3 = FirstCodePoints(upper, 3);
      if (StartsWith(badge, name3) || StartsWith(upper, badge3)) {
        out.push_back({*it, index, true});
        taken[index] = true;
        free.erase(it);
        break;
      }
    }
  }
  std::size_t next = 0;
  for (const std::size_t index : order) {
    if (taken[index])
      continue;
    if (next >= free.size())
      break;
    out.push_back({free[next++], index, false});
  }
  return out;
}

FigurenService::FigurenService(Options options) : options_(std::move(options)) {}

std::optional<HttpResponse> FigurenService::Handle(
    const HttpRequest& request) const {
  if (request.path != "/v1/figuren")
    return std::nullopt;
  if (request.method != "POST")
    return Error(405, "Method Not Allowed", "use POST", "method_not_allowed");
  const auto started = std::chrono::steady_clock::now();
  Request parsed;
  try {
    parsed = ParseRequest(request.body, options_.default_model);
  } catch (const std::exception& error) {
    return Error(400, "Bad Request", error.what(), "invalid_request");
  }
  const Session session(options_.fetch, parsed.model);
  try {
    // (1) people, thinking off; (2) assignment.
    const std::vector<Person> people = session.FindPeople(parsed.frame);
    std::vector<std::string> names;
    for (const Reference& reference : parsed.references)
      names.push_back(reference.name);
    const std::vector<Assignment> assigned = Assign(people, names);
    const auto reference_of = [&parsed](const std::string& name) {
      return std::find_if(parsed.references.begin(), parsed.references.end(),
                          [&name](const Reference& r) { return r.name == name; });
    };
    // (3) quick compare for every figure, thinking off.
    std::vector<Verdict> quick;
    for (const Assignment& a : assigned)
      quick.push_back(session.Compare(parsed.frame,
                                      people[a.person].kopf_box,
                                      *reference_of(a.name), false));
    // (4) below the threshold: again with thinking; that verdict counts.
    json::Value figures = json::Value::array();
    for (std::size_t i = 0; i < assigned.size(); ++i) {
      const Assignment& a = assigned[i];
      const Person& person = people[a.person];
      json::Value figure = json::Value::object();
      figure["name"] = a.name;
      figure["box"] = BoxJson(person.box);
      figure["kopf_box"] = BoxJson(person.kopf_box);
      figure["schild"] = person.schild;
      figure["zuordnung"] = a.by_badge ? "schild" : "reihenfolge";
      Verdict verdict = std::move(quick[i]);
      const bool recheck = verdict.similarity < parsed.threshold;
      if (recheck) {
        figure["schnell"] = verdict.similarity;
        verdict = session.Compare(parsed.frame, person.kopf_box,
                                  *reference_of(a.name), true);
      }
      figure["aehnlichkeit"] = verdict.similarity;
      figure["abweichungen"] = std::move(verdict.differences);
      figure["nachgeprueft"] = recheck;
      figures.push_back(std::move(figure));
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      started)
            .count();
    json::Value body = json::Value::object();
    body["figuren"] = std::move(figures);
    body["sekunden"] = std::round(seconds * 10) / 10;
    return JsonResponse(200, "OK", body.dump());
  } catch (const UpstreamError& error) {
    HttpResponse response;
    response.status = error.status;
    response.reason = error.status == 404   ? "Not Found"
                      : error.status == 503 ? "Service Unavailable"
                                            : "Upstream Error";
    response.body = error.body;
    response.headers = {
        {"Content-Type", error.content_type.empty() ? "application/json"
                                                    : error.content_type}};
    return response;
  } catch (const AnswerError& error) {
    return Error(502, "Bad Gateway", error.what(), "model_answer_invalid");
  } catch (const std::exception& error) {
    return Error(500, "Internal Server Error", error.what(), "figuren_failed");
  }
}

}  // namespace gufo::router::figuren
