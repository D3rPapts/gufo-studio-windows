// POST /v1/figuren (router figure check): JSON extraction, badge assignment,
// box conversion and crop math, and the endpoint against a scripted fetch.
// Written with AI assistance (Claude Code).
#undef NDEBUG
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "src/cli/serve/router/comfy_graph.hpp"
#include "src/cli/serve/router/figuren.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen_image_21/model.hpp"

namespace json = gufo::json;
namespace figuren = gufo::router::figuren;
using figuren::Box;
using figuren::Person;
using gufo::server::HttpRequest;

namespace {

std::string PngDataUrl(int width, int height, std::uint8_t shade) {
  gufo::models::qwen_image_21::Image image;
  image.width = width;
  image.height = height;
  image.rgba.assign(static_cast<std::size_t>(width) * height * 4, shade);
  for (std::size_t i = 3; i < image.rgba.size(); i += 4)
    image.rgba[i] = 255;
  const auto png = gufo::models::qwen_image_21::EncodePng(image);
  return "data:image/png;base64," +
         gufo::router::comfy::Base64Encode(std::string_view(
             reinterpret_cast<const char*>(png.data()), png.size()));
}

Person P(Box box, std::string badge) {
  Person person;
  person.box = box;
  person.kopf_box = box;
  person.schild = std::move(badge);
  return person;
}

void TestJsonFrom() {
  auto list = figuren::JsonFrom("text [1,2] more [{\"a\":[1]}] end", true);
  assert(list.has_value() && list->is_array() && list->size() == 1);
  assert(list->items()[0].find("a") != nullptr);

  auto object = figuren::JsonFrom(
      "Answer: {\"aehnlichkeit\": 7, \"abweichungen\": [\"x\"]}", false);
  assert(object.has_value() && object->is_object());
  assert(object->find("aehnlichkeit")->as_double() == 7);

  // Thinking is removed first; a lone closing tag drops what precedes it.
  assert(!figuren::JsonFrom("<think>[9]</think>no json", true).has_value());
  auto after = figuren::JsonFrom("<think>\n{\"a\":1}\n</think>\n{\"b\":2}", false);
  assert(after.has_value() && after->find("b") != nullptr);
  auto lone = figuren::JsonFrom("reasoning [1] </think>[2]", true);
  assert(lone.has_value() && lone->items()[0].as_double() == 2);
  assert(!figuren::JsonFrom("reasoning {\"a\":1}</think> none", false));
  assert(figuren::StripThinking("a<think>x</think>b<think>y</think>c") == "abc");

  assert(!figuren::JsonFrom("", true).has_value());
  assert(!figuren::JsonFrom("]", true).has_value());
  assert(!figuren::JsonFrom("[1, 2", true).has_value());
  // Fenced answers still parse.
  auto fenced = figuren::JsonFrom("```json\n[{\"badge\": \"\"}]\n```", true);
  assert(fenced.has_value() && fenced->size() == 1);
}

void TestBoxes() {
  const auto parse = [](const char* text) { return json::parse(text); };
  // Python round(): half to even (12.5 -> 12, 13.5 -> 14).
  auto px = figuren::ToPixels(parse("[125, 0, 135, 1000]"), 100, 50);
  assert(px.has_value());
  assert((*px == Box{12, 0, 14, 50}));
  auto str = figuren::ToPixels(parse("[\"100\", 0, 300.0, 400]"), 100, 50);
  assert(str.has_value() && (*str == Box{10, 0, 30, 20}));
  assert(!figuren::ToPixels(parse("[1, 2, 3]"), 100, 50).has_value());
  assert(!figuren::ToPixels(parse("[1, 2, 3, \"x\"]"), 100, 50).has_value());
  assert(!figuren::ToPixels(parse("{\"a\": 1}"), 100, 50).has_value());

  // Pad a third of the box (floor), clamp to the image.
  assert((figuren::HeadCrop(Box{10, 0, 30, 20}, 100, 50) == Box{4, 0, 36, 26}));
  assert((figuren::HeadCrop(Box{90, 40, 99, 49}, 100, 50) ==
          Box{87, 37, 100, 50}));
  // Reversed or empty boxes still give at least one pixel inside the image.
  const Box odd = figuren::HeadCrop(Box{50, 30, 40, 30}, 100, 50);
  assert(odd[0] >= 0 && odd[2] > odd[0] && odd[2] <= 100);
  assert(odd[1] >= 0 && odd[3] > odd[1] && odd[3] <= 50);
  const Box outside = figuren::HeadCrop(Box{200, 80, 300, 90}, 100, 50);
  assert(outside[0] < outside[2] && outside[2] <= 100);
  assert(outside[1] < outside[3] && outside[3] <= 50);

  assert((figuren::FitSize(1920, 1080, 1344) == std::pair<int, int>{1344, 756}));
  assert((figuren::FitSize(500, 300, 768) == std::pair<int, int>{500, 300}));
  assert((figuren::FitSize(300, 1000, 768) == std::pair<int, int>{230, 768}));
}

void TestAssign() {
  const std::vector<std::string> names = {"Lars", "Bernd", "Grossmann"};
  // People out of order; the badge of the left one is misspelt.
  std::vector<Person> people = {P({600, 0, 700, 10}, ""),
                                P({100, 0, 200, 10}, "Larz!"),
                                P({300, 0, 400, 10}, "")};
  auto a = figuren::Assign(people, names);
  assert(a.size() == 3);
  assert(a[0].name == "Lars" && a[0].person == 1 && a[0].by_badge);
  // The rest left to right in reference order.
  assert(a[1].name == "Bernd" && a[1].person == 2 && !a[1].by_badge);
  assert(a[2].name == "Grossmann" && a[2].person == 0 && !a[2].by_badge);

  // Short badge matches a name prefix; sharp s counts as SS; case ignored.
  people = {P({0, 0, 1, 1}, "Groß"), P({10, 0, 11, 1}, "be")};
  a = figuren::Assign(people, names);
  assert(a.size() == 2);
  assert(a[0].name == "Grossmann" && a[0].by_badge);
  assert(a[1].name == "Bernd" && a[1].by_badge);

  // More people than names: the extra ones are dropped; fewer: names unused.
  people = {P({0, 0, 1, 1}, ""), P({5, 0, 6, 1}, ""), P({9, 0, 10, 1}, ""),
            P({12, 0, 13, 1}, "")};
  a = figuren::Assign(people, names);
  assert(a.size() == 3 && a[2].person == 2);
  a = figuren::Assign({P({0, 0, 1, 1}, "")}, names);
  assert(a.size() == 1 && a[0].name == "Lars");
  // Unreadable badge text falls back to the order.
  a = figuren::Assign({P({0, 0, 1, 1}, "???")}, names);
  assert(a.size() == 1 && a[0].name == "Lars" && !a[0].by_badge);
}

void TestEndpoint() {
  int calls = 0;
  int thinking_calls = 0;
  int status = 200;
  figuren::Options options;
  options.fetch = [&](const HttpRequest& request) {
    ++calls;
    assert(request.method == "POST" && request.path == "/v1/chat/completions");
    const json::Value body = json::parse(request.body);
    assert(body.find("model")->str() == "test-vlm");
    assert(body.find("temperature")->as_double() == 0);
    const bool think = body.find("chat_template_kwargs") == nullptr;
    if (think)
      ++thinking_calls;
    const auto& content =
        body.find("messages")->items()[0].find("content")->items();
    const std::string text = content.back().find("text")->str();
    std::string answer;
    if (text.starts_with("List every person")) {
      assert(content.size() == 2 && !think);
      answer =
          "[{\"person_box\": [0,0,400,1000], \"head_box\": [100,0,300,400], "
          "\"badge\": \"\"}, {\"person_box\": [500,0,1000,1000], "
          "\"head_box\": [600,100,900,500], \"badge\": \" bernd \"}]";
    } else if (text.find("character Lars.") != std::string::npos) {
      assert(content.size() == 3);
      answer = "{\"aehnlichkeit\": 8, \"abweichungen\": []}";
    } else {
      assert(text.find("character Bernd.") != std::string::npos);
      assert(text.find("profile: kurze Haare, Brille.") != std::string::npos);
      answer = think ? "<think>{\"x\": 1}</think>{\"aehnlichkeit\": 7, "
                       "\"abweichungen\": [\"Brille fehlt\"]}"
                     : "{\"aehnlichkeit\": \"3\", \"abweichungen\": [\"a\"]}";
    }
    gufo::router::comfy::FetchResult result;
    result.status = status;
    if (status != 200) {
      result.body = "{\"error\":{\"code\":\"busy\"}}";
      return result;
    }
    json::Value message = json::Value::object();
    message["role"] = "assistant";
    message["content"] = answer;
    json::Value choice = json::Value::object();
    choice["message"] = std::move(message);
    json::Value choices = json::Value::array();
    choices.push_back(std::move(choice));
    json::Value out = json::Value::object();
    out["choices"] = std::move(choices);
    result.body = out.dump();
    result.content_type = "application/json";
    return result;
  };
  options.default_model = "unused";
  const figuren::FigurenService service(std::move(options));

  HttpRequest other;
  other.method = "POST";
  other.path = "/v1/names";
  assert(!service.Handle(other).has_value());
  HttpRequest get;
  get.method = "GET";
  get.path = "/v1/figuren";
  assert(service.Handle(get)->status == 405);

  json::Value body = json::Value::object();
  body["bild"] = PngDataUrl(100, 50, 40);
  json::Value refs = json::Value::array();
  for (const char* name : {"Lars", "Bernd"}) {
    json::Value ref = json::Value::object();
    ref["name"] = name;
    // Raw base64 without the data URL prefix is accepted as well.
    const std::string url = PngDataUrl(64, 64, 200);
    ref["kopf"] = std::string(name) == "Lars" ? url : url.substr(url.find(',') + 1);
    refs.push_back(std::move(ref));
  }
  body["referenzen"] = std::move(refs);
  json::Value profiles = json::Value::object();
  profiles["Bernd"] = "kurze Haare, Brille";
  body["steckbrief"] = std::move(profiles);
  body["modell"] = "test-vlm";

  HttpRequest post;
  post.method = "POST";
  post.path = "/v1/figuren";
  post.body = body.dump();
  const auto response = service.Handle(post);
  assert(response.has_value());
  if (response->status != 200)
    std::cerr << response->body << '\n';
  assert(response->status == 200);
  // 1 find + 2 quick compares + 1 recheck with thinking (Bernd 0.3 < 0.6).
  assert(calls == 4 && thinking_calls == 1);
  const json::Value result = json::parse(response->body);
  const auto& figures = result.find("figuren")->items();
  assert(figures.size() == 2);
  const json::Value& bernd = figures[0];
  assert(bernd.find("name")->str() == "Bernd");
  assert(bernd.find("zuordnung")->str() == "schild");
  assert(bernd.find("schild")->str() == "bernd");
  assert(bernd.find("nachgeprueft")->as_bool());
  assert(bernd.find("schnell")->as_double() == 0.3);
  assert(bernd.find("aehnlichkeit")->as_double() == 0.7);
  assert(bernd.find("abweichungen")->items()[0].str() == "Brille fehlt");
  assert(bernd.find("box")->items()[0].as_double() == 50);
  assert(bernd.find("kopf_box")->items()[1].as_double() == 5);
  const json::Value& lars = figures[1];
  assert(lars.find("name")->str() == "Lars");
  assert(lars.find("zuordnung")->str() == "reihenfolge");
  assert(!lars.find("nachgeprueft")->as_bool());
  assert(lars.find("schnell") == nullptr);
  assert(lars.find("aehnlichkeit")->as_double() == 0.8);
  assert(lars.find("kopf_box")->dump() == "[10,0,30,20]");
  assert(result.find("sekunden")->is_number());

  // A higher threshold rechecks Lars as well.
  json::Value strict = json::parse(post.body);
  strict["schwelle"] = 0.9;
  post.body = strict.dump();
  calls = thinking_calls = 0;
  assert(service.Handle(post)->status == 200);
  assert(calls == 5 && thinking_calls == 2);

  // The router's own error (busy, memory guard) reaches the client as is.
  status = 503;
  const auto busy = service.Handle(post);
  assert(busy->status == 503 && busy->body.find("busy") != std::string::npos);
  status = 200;

  // Bad requests.
  HttpRequest bad = post;
  bad.body = "{\"referenzen\": []}";
  assert(service.Handle(bad)->status == 400);
  bad.body = "not json";
  assert(service.Handle(bad)->status == 400);
  json::Value broken = json::parse(post.body);
  broken["bild"] = "data:image/png;base64,AAAA";
  bad.body = broken.dump();
  assert(service.Handle(bad)->status == 400);
}

}  // namespace

int main() {
  TestJsonFrom();
  TestBoxes();
  TestAssign();
  TestEndpoint();
  std::cout << "router_figuren_test: ok\n";
  return 0;
}
