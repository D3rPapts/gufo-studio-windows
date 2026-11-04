// POST /v1/figuren (router figure check): JSON extraction, badge assignment,
// box conversion and crop math, the costume colour-family rule, and the
// endpoint (head compare, costume check, nur_kostuem) against a scripted fetch.
// Written with AI assistance (Claude Code).
#undef NDEBUG
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/cli/serve/router/comfy_graph.hpp"
#include "src/cli/serve/router/figuren.hpp"
#include "src/core/image.hpp"
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

void TestRepair() {
  // Expected values are figuren.py repair_json outputs.
  assert(figuren::RepairJson("[75, 3, 702, .948]") == "[75, 3, 702, 0.948]");
  assert(figuren::RepairJson("[-.5,.25]") == "[-0.5,0.25]");
  assert(figuren::RepairJson("{\"a\": 1., \"b\": [2.]}") ==
         "{\"a\": 1, \"b\": [2]}");
  assert(figuren::RepairJson("[1.,2.]") == "[1,2]");
  assert(figuren::RepairJson("[+3, +.5, 7.\n]") == "[3, +.5, 7\n]");
  assert(figuren::RepairJson("v.5 -.5 :.5") == "v.5 -0.5 :0.5");
  // Python \s includes U+00A0 (no-break space).
  assert(figuren::RepairJson("\xc2\xa0.5") == "\xc2\xa0" "0.5");
  // Text outside such numbers stays as it is.
  const std::string text = "Wert 3.5 und x.y, a+1 [1.5] \xc3\xa4.5";
  assert(figuren::RepairJson(text) == text);
  assert(figuren::RepairJson("[1..]") == "[1..]");
  assert(figuren::RepairJson("") == "");
  assert(figuren::RepairJson(".5") == ".5");
  // The lauf_170 answer parses after the repair.
  const std::string lauf =
      "[{\"person_box\": [75, 3, 702, .948], \"head_box\": [1, 2, 3, 4]}]";
  assert(!figuren::JsonFrom(lauf, true).has_value());
  auto repaired = figuren::JsonFrom(figuren::RepairJson(lauf), true);
  assert(repaired.has_value() && repaired->size() == 1);
}

void TestBoxes() {
  const auto parse = [](const char* text) { return json::parse(text); };
  // figuren.py to_px: a single fraction in a 0..1000 box is scaled up ...
  auto frac = figuren::ToPixels(parse("[75, 3, 702, 0.948]"), 1000, 1000);
  assert(frac.has_value() && (*frac == Box{75, 3, 702, 948}));
  // ... 0 and 1 are kept (only 0 < v < 1) ...
  frac = figuren::ToPixels(parse("[0, 1, 500, 0.5]"), 1000, 1000);
  assert(frac.has_value() && (*frac == Box{0, 1, 500, 500}));
  // ... a whole 0..1 box is scaled by 1000 ...
  frac = figuren::ToPixels(parse("[0.1, 0, 0.5, 1]"), 100, 50);
  assert(frac.has_value() && (*frac == Box{10, 0, 50, 50}));
  // ... and a box with a maximum between 1 and 2 is left alone.
  frac = figuren::ToPixels(parse("[0.5, 0, 2, 1.5]"), 1000, 1000);
  assert(frac.has_value() && (*frac == Box{0, 0, 2, 2}));
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

void TestTorsoCrop() {
  // Person box width; head top to head bottom + 3.2 head heights.
  assert((figuren::TorsoCrop(Box{100, 50, 300, 900}, Box{150, 100, 250, 200},
                             1000, 1000) == Box{100, 100, 300, 520}));
  // 3.2 x 7 = 22.4 is truncated to 22.
  assert((figuren::TorsoCrop(Box{0, 0, 50, 900}, Box{10, 10, 20, 17}, 1000,
                             1000) == Box{0, 10, 50, 39}));
  // Clipped at the person box bottom ...
  assert((figuren::TorsoCrop(Box{100, 50, 300, 300}, Box{150, 100, 250, 200},
                             1000, 1000) == Box{100, 100, 300, 300}));
  // ... but at least 8 px below the head.
  assert((figuren::TorsoCrop(Box{100, 50, 300, 150}, Box{150, 100, 250, 200},
                             1000, 1000) == Box{100, 100, 300, 208}));
  // Clipped to the image.
  assert((figuren::TorsoCrop(Box{-20, 50, 1200, 900}, Box{150, -10, 250, 90},
                             1000, 1000) == Box{0, 0, 1000, 410}));
  assert((figuren::TorsoCrop(Box{100, 50, 300, 900}, Box{150, 100, 250, 200},
                             1000, 400) == Box{100, 100, 300, 400}));
  // Degenerate boxes still give at least one pixel inside the image.
  const Box odd =
      figuren::TorsoCrop(Box{90, 0, 10, 5}, Box{50, 60, 40, 55}, 100, 50);
  assert(odd[0] >= 0 && odd[2] > odd[0] && odd[2] <= 100);
  assert(odd[1] >= 0 && odd[3] > odd[1] && odd[3] <= 50);
}

void TestColourRule() {
  assert(figuren::ColourFamily("Dunkelblau") == "blau");
  assert(figuren::ColourFamily("navy") == "blau");
  assert(figuren::ColourFamily("WEISS") == "weiss");
  assert(figuren::ColourFamily("wei\xc3\x9f") == "weiss");
  assert(figuren::ColourFamily("GR\xc3\x9cN") == "gruen");
  assert(figuren::ColourFamily("hellbraun") == "braun");
  assert(figuren::ColourFamily("cremefarben") == "creme");
  assert(figuren::ColourFamily("dunkel") == "schwarz");
  assert(!figuren::ColourFamily("lila").has_value());
  assert(!figuren::ColourFamily("").has_value());

  const json::Value items = json::parse(
      "[{\"teil\": \"Hemd\", \"soll\": \"weiss\", \"ist\": \"creme\", "
      "\"ok\": true},"
      " {\"teil\": \"Krawatte\", \"soll\": \"dunkelblau\", \"ist\": \"blau\", "
      "\"ok\": true},"
      " {\"teil\": \"Brille\", \"soll\": \"gruen getoent\", \"ist\": "
      "\"schwarz\", \"ok\": true},"
      " {\"teil\": \"Shirt collar\", \"soll\": \"white\", \"ist\": \"grey\"},"
      " {\"teil\": \"Tie\", \"soll\": \"rot\", \"ist\": \"gelb\", \"ok\": "
      "false},"
      " {\"teil\": \"Hemd\", \"soll\": \"lila\", \"ist\": \"blau\", \"ok\": "
      "true},"
      " \"no object\"]");
  const json::Value checked = figuren::CheckCostumeItems(items);
  assert(checked.size() == 6);
  const auto& c = checked.items();
  // weiss -> creme: other family, wrong although the model said ok.
  assert(!c[0].find("ok")->as_bool(true));
  assert(c[0].find("regel")->str() == "Farbfamilie weiss != creme");
  // dunkelblau -> blau: same family, stays ok.
  assert(c[1].find("ok")->as_bool() && c[1].find("regel") == nullptr);
  // Glasses are left to the model.
  assert(c[2].find("ok")->as_bool() && c[2].find("regel") == nullptr);
  // A missing ok counts as ok and is checked.
  assert(!c[3].find("ok")->as_bool(true));
  assert(c[3].find("regel")->str() == "Farbfamilie weiss != grau");
  // Already wrong: no rule note.
  assert(!c[4].find("ok")->as_bool(true) && c[4].find("regel") == nullptr);
  // Unknown colour word: no family, no change.
  assert(c[5].find("ok")->as_bool() && c[5].find("regel") == nullptr);
}

void TestEndpoint() {
  int calls = 0;
  int thinking_calls = 0;
  int costume_calls = 0;
  int status = 200;
  std::string costume_answer =
      "<think>{\"items\": []}</think>{\"items\": ["
      "{\"teil\": \"Hemd\", \"soll\": \"weiss\", \"ist\": \"creme\", \"ok\": "
      "true},"
      "{\"teil\": \"Krawatte\", \"soll\": \"dunkelblau\", \"ist\": \"blau\", "
      "\"ok\": true},"
      "{\"teil\": \"Brille\", \"soll\": \"gruen\", \"ist\": \"schwarz\", "
      "\"ok\": true}],"
      " \"widersprueche\": [\"Brille: Text dunkel, Referenz gruen getoent\"],"
      " \"kostuem_ok\": true, \"abweichungen\": []}";
  figuren::Options options;
  options.fetch = [&](const HttpRequest& request) {
    ++calls;
    assert(request.method == "POST" && request.path == "/v1/chat/completions");
    const json::Value body = json::parse(request.body);
    assert(body.find("model")->str() == "test-vlm");
    // 0.3 only for the one retry after an answer without JSON.
    const double temperature = body.find("temperature")->as_double();
    assert(temperature == 0 || temperature == 0.3);
    const bool think = body.find("chat_template_kwargs") == nullptr;
    if (think)
      ++thinking_calls;
    const auto& content =
        body.find("messages")->items()[0].find("content")->items();
    const std::string text = content.back().find("text")->str();
    std::string answer;
    if (text.starts_with("Image 1 is the front reference")) {
      // Costume: thinking on, 14000 tokens, front + torso crop.
      ++costume_calls;
      assert(think);
      assert(body.find("max_tokens")->as_double() == 14000);
      assert(content.size() == 3);
      assert(content[0].find("type")->str() == "image_url" &&
             content[1].find("type")->str() == "image_url");
      assert(text.find("character Lars. Image 2 shows the same character "
                       "(head to belt)") != std::string::npos);
      assert(text.find("generated from: Lars Profil Check only") !=
             std::string::npos);
      // Lars: person [0,0,40,50], head [10,0,30,20] in a 100x50 frame.
      const auto torso = gufo::core::DecodeImage(gufo::core::ReadImageUrl(
          content[1].find("image_url")->find("url")->str()));
      assert(torso.width == 40 && torso.height == 50);
      const auto front = gufo::core::DecodeImage(gufo::core::ReadImageUrl(
          content[0].find("image_url")->find("url")->str()));
      assert(front.width == 30 && front.height == 60);
      answer = costume_answer;
    } else if (text.starts_with("List every person")) {
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
  // No front reference: no costume call, no costume fields.
  assert(costume_calls == 0);
  for (const json::Value& figure : figures) {
    assert(figure.find("kostuem_ok") == nullptr);
    assert(figure.find("kostuem_abweichungen") == nullptr);
    assert(figure.find("kostuem_teile") == nullptr);
    assert(figure.find("steckbrief_widerspricht_referenz") == nullptr);
  }
  const std::string plain_body = post.body;

  // Front reference for Lars: one costume call after the head compares.
  json::Value dressed = json::parse(plain_body);
  {
    json::Value refs_with_front = json::Value::array();
    for (const json::Value& ref : dressed.find("referenzen")->items()) {
      json::Value copy = ref;
      if (copy.find("name")->str() == "Lars")
        copy["front"] = PngDataUrl(30, 60, 90);
      refs_with_front.push_back(std::move(copy));
    }
    dressed["referenzen"] = std::move(refs_with_front);
    dressed["steckbrief"]["Lars"] = "Lars Profil";
  }
  post.body = dressed.dump();
  calls = thinking_calls = costume_calls = 0;
  const auto costume = service.Handle(post);
  if (costume->status != 200)
    std::cerr << costume->body << '\n';
  assert(costume->status == 200);
  // 1 find + 2 quick + 1 recheck + 1 costume (thinking).
  assert(calls == 5 && thinking_calls == 2 && costume_calls == 1);
  {
    const json::Value out = json::parse(costume->body);
    const auto& figs = out.find("figuren")->items();
    assert(figs.size() == 2);
    const json::Value& bernd_out = figs[0];
    assert(bernd_out.find("name")->str() == "Bernd");
    assert(bernd_out.find("kostuem_ok") == nullptr);
    const json::Value& lars_out = figs[1];
    assert(lars_out.find("aehnlichkeit")->as_double() == 0.8);
    // Hemd weiss -> creme is wrong by the family rule although the model
    // passed it; its own empty list is replaced by one phrase per item.
    assert(!lars_out.find("kostuem_ok")->as_bool(true));
    assert(lars_out.find("kostuem_abweichungen")->dump() ==
           "[\"Hemd creme statt weiss\"]");
    const auto& teile = lars_out.find("kostuem_teile")->items();
    assert(teile.size() == 3);
    assert(teile[0].find("regel")->str() == "Farbfamilie weiss != creme");
    assert(teile[1].find("ok")->as_bool() && teile[1].find("regel") == nullptr);
    assert(teile[2].find("ok")->as_bool() && teile[2].find("regel") == nullptr);
    assert(lars_out.find("steckbrief_widerspricht_referenz")->dump() ==
           "[\"Brille: Text dunkel, Referenz gruen getoent\"]");
  }

  // All items ok: kostuem_ok and no differences, even if the model lists some.
  costume_answer =
      "{\"items\": [{\"teil\": \"Krawatte\", \"soll\": \"dunkelblau\", "
      "\"ist\": \"blau\", \"ok\": true}], \"kostuem_ok\": false, "
      "\"abweichungen\": [\"Krawatte heller\"]}";
  {
    const json::Value out = json::parse(service.Handle(post)->body);
    const json::Value& lars_out = out.find("figuren")->items()[1];
    assert(lars_out.find("kostuem_ok")->as_bool());
    assert(lars_out.find("kostuem_abweichungen")->empty());
    assert(lars_out.find("steckbrief_widerspricht_referenz")->empty());
  }
  // No items: the model's kostuem_ok decides, its own phrases are kept.
  costume_answer =
      "{\"items\": [], \"kostuem_ok\": false, \"abweichungen\": [\"Brille "
      "fehlt\"]}";
  {
    const json::Value out = json::parse(service.Handle(post)->body);
    const json::Value& lars_out = out.find("figuren")->items()[1];
    assert(!lars_out.find("kostuem_ok")->as_bool(true));
    assert(lars_out.find("kostuem_abweichungen")->dump() ==
           "[\"Brille fehlt\"]");
  }
  // A costume answer without JSON, also on the retry, is a bad gateway.
  costume_answer = "kein JSON";
  calls = costume_calls = 0;
  assert(service.Handle(post)->status == 502);
  assert(costume_calls == 2);
  costume_answer =
      "{\"items\": [{\"teil\": \"Hemd\", \"soll\": \"weiss\", \"ist\": "
      "\"weiss\", \"ok\": true}]}";

  // nur_kostuem: boxes + costume only, no head compare calls.
  json::Value only = json::parse(post.body);
  only["nur_kostuem"] = true;
  post.body = only.dump();
  calls = thinking_calls = costume_calls = 0;
  {
    const auto answer = service.Handle(post);
    assert(answer->status == 200);
    assert(calls == 2 && thinking_calls == 1 && costume_calls == 1);
    const json::Value out = json::parse(answer->body);
    const auto& figs = out.find("figuren")->items();
    assert(figs.size() == 2);
    for (const json::Value& figure : figs) {
      assert(figure.find("box") != nullptr && figure.find("kopf_box"));
      assert(figure.find("aehnlichkeit") == nullptr);
      assert(figure.find("abweichungen") == nullptr);
      assert(figure.find("nachgeprueft") == nullptr);
    }
    assert(figs[1].find("kostuem_ok")->as_bool());
    assert(figs[0].find("kostuem_ok") == nullptr);
  }
  // With nur_kostuem the head reference may be left out.
  {
    json::Value no_head = json::parse(post.body);
    json::Value refs_out = json::Value::array();
    for (const json::Value& ref : no_head.find("referenzen")->items()) {
      json::Value copy = json::Value::object();
      copy["name"] = ref.find("name")->str();
      if (const json::Value* front = ref.find("front"))
        copy["front"] = front->str();
      refs_out.push_back(std::move(copy));
    }
    no_head["referenzen"] = std::move(refs_out);
    HttpRequest headless = post;
    headless.body = no_head.dump();
    assert(service.Handle(headless)->status == 200);
    no_head["nur_kostuem"] = false;
    headless.body = no_head.dump();
    assert(service.Handle(headless)->status == 400);
  }
  // Bad costume input.
  {
    HttpRequest wrong = post;
    json::Value flag = json::parse(post.body);
    flag["nur_kostuem"] = "ja";
    wrong.body = flag.dump();
    assert(service.Handle(wrong)->status == 400);
    json::Value front = json::parse(plain_body);
    json::Value refs_bad = json::Value::array();
    for (const json::Value& ref : front.find("referenzen")->items()) {
      json::Value copy = ref;
      copy["front"] = 5;
      refs_bad.push_back(std::move(copy));
    }
    front["referenzen"] = std::move(refs_bad);
    wrong.body = front.dump();
    assert(service.Handle(wrong)->status == 400);
    json::Value refs_broken = json::Value::array();
    for (const json::Value& ref : front.find("referenzen")->items()) {
      json::Value copy = ref;
      copy["front"] = "data:image/png;base64,AAAA";
      refs_broken.push_back(std::move(copy));
    }
    front["referenzen"] = std::move(refs_broken);
    wrong.body = front.dump();
    assert(service.Handle(wrong)->status == 400);
  }
  post.body = plain_body;

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

/// Answers to the person search in order; every call's temperature, thinking
/// flag and max_tokens are recorded. Head compares answer 8/10.
struct ScriptedModel {
  std::vector<std::string> find_answers;
  std::vector<std::string> compare_answers;
  std::vector<double> find_temperatures;
  std::vector<bool> find_thinking;
  std::vector<double> find_max_tokens;
  std::vector<double> compare_temperatures;
  std::vector<bool> compare_thinking;
  std::vector<double> compare_max_tokens;

  gufo::router::comfy::FetchResult operator()(const HttpRequest& request) {
    const json::Value body = json::parse(request.body);
    const std::string text = body.find("messages")
                                 ->items()[0]
                                 .find("content")
                                 ->items()
                                 .back()
                                 .find("text")
                                 ->str();
    const double temperature = body.find("temperature")->as_double();
    const bool think = body.find("chat_template_kwargs") == nullptr;
    const double max_tokens = body.find("max_tokens")->as_double();
    std::string answer;
    if (text.starts_with("List every person")) {
      answer = find_answers.at(find_temperatures.size());
      find_temperatures.push_back(temperature);
      find_thinking.push_back(think);
      find_max_tokens.push_back(max_tokens);
    } else {
      answer = compare_answers.at(compare_temperatures.size());
      compare_temperatures.push_back(temperature);
      compare_thinking.push_back(think);
      compare_max_tokens.push_back(max_tokens);
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
    gufo::router::comfy::FetchResult result;
    result.status = 200;
    result.body = out.dump();
    result.content_type = "application/json";
    return result;
  }
};

std::string OneFigureBody() {
  json::Value body = json::Value::object();
  body["bild"] = PngDataUrl(100, 50, 40);
  json::Value ref = json::Value::object();
  ref["name"] = "Lars";
  ref["kopf"] = PngDataUrl(64, 64, 200);
  json::Value refs = json::Value::array();
  refs.push_back(std::move(ref));
  body["referenzen"] = std::move(refs);
  body["modell"] = "test-vlm";
  return body.dump();
}

void TestRetry() {
  HttpRequest post;
  post.method = "POST";
  post.path = "/v1/figuren";
  post.body = OneFigureBody();
  const std::string broken = "[{\"person_box\": [0, 0, 400, 1000}, oops";
  const std::string valid =
      "[{\"person_box\": [0, 0, 400, 1000], \"head_box\": [100, 0, 300, 400], "
      "\"badge\": \"\"}]";

  // Broken beyond repair, then valid at temperature 0.3: success, 2 calls.
  {
    auto model = std::make_shared<ScriptedModel>();
    model->find_answers = {broken, valid};
    model->compare_answers = {"{\"aehnlichkeit\": 8, \"abweichungen\": []}"};
    figuren::Options options;
    options.fetch = [model](const HttpRequest& r) { return (*model)(r); };
    const figuren::FigurenService service(std::move(options));
    const auto response = service.Handle(post);
    if (response->status != 200)
      std::cerr << response->body << '\n';
    assert(response->status == 200);
    assert(model->find_temperatures.size() == 2);
    assert(model->find_temperatures[0] == 0 &&
           model->find_temperatures[1] == 0.3);
    // Same thinking setting and max_tokens on the retry.
    assert(!model->find_thinking[0] && !model->find_thinking[1]);
    assert(model->find_max_tokens[0] == 6000 &&
           model->find_max_tokens[1] == 6000);
    assert(model->compare_temperatures.size() == 1);
    const json::Value out = json::parse(response->body);
    assert(out.find("figuren")->items()[0].find("kopf_box")->dump() ==
           "[10,0,30,20]");
  }
  // Repairable first answer (lauf_170): no second request.
  {
    auto model = std::make_shared<ScriptedModel>();
    model->find_answers = {
        "[{\"person_box\": [0, 0, 400, 1000], \"head_box\": [100, 0, 300, "
        ".4], \"badge\": \"\"}]"};
    model->compare_answers = {"{\"aehnlichkeit\": 8, \"abweichungen\": []}"};
    figuren::Options options;
    options.fetch = [model](const HttpRequest& r) { return (*model)(r); };
    const figuren::FigurenService service(std::move(options));
    const auto response = service.Handle(post);
    assert(response->status == 200);
    assert(model->find_temperatures.size() == 1);
    const json::Value out = json::parse(response->body);
    // .4 -> 0.4 -> 400 (scaled because the box maximum is above 2).
    assert(out.find("figuren")->items()[0].find("kopf_box")->dump() ==
           "[10,0,30,20]");
  }
  // Both answers broken: 502 after exactly 2 calls.
  {
    auto model = std::make_shared<ScriptedModel>();
    model->find_answers = {broken, "still no JSON"};
    figuren::Options options;
    options.fetch = [model](const HttpRequest& r) { return (*model)(r); };
    const figuren::FigurenService service(std::move(options));
    const auto response = service.Handle(post);
    assert(response->status == 502);
    assert(response->body.find("model_answer_invalid") != std::string::npos);
    assert(model->find_temperatures.size() == 2);
    assert(model->compare_temperatures.empty());
  }
  // Head compare: the retry keeps the thinking setting; second answer counts.
  {
    auto model = std::make_shared<ScriptedModel>();
    model->find_answers = {valid};
    model->compare_answers = {"aehnlichkeit: acht",
                              "{\"aehnlichkeit\": 9, \"abweichungen\": []}"};
    figuren::Options options;
    options.fetch = [model](const HttpRequest& r) { return (*model)(r); };
    const figuren::FigurenService service(std::move(options));
    const auto response = service.Handle(post);
    assert(response->status == 200);
    assert(model->compare_temperatures.size() == 2);
    assert(model->compare_temperatures[1] == 0.3);
    assert(!model->compare_thinking[0] && !model->compare_thinking[1]);
    const json::Value out = json::parse(response->body);
    assert(out.find("figuren")->items()[0].find("aehnlichkeit")->as_double() ==
           0.9);
  }
}

}  // namespace

int main() {
  TestJsonFrom();
  TestRepair();
  TestBoxes();
  TestAssign();
  TestTorsoCrop();
  TestColourRule();
  TestEndpoint();
  TestRetry();
  std::cout << "router_figuren_test: ok\n";
  return 0;
}
