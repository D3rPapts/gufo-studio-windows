#include "src/cli/serve/router/mcp.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <utility>

#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/router/comfy_graph.hpp"
#include "src/cli/serve/router/strings.hpp"

#ifndef GUFO_RELEASE_VERSION
#define GUFO_RELEASE_VERSION "development"
#endif
#ifndef GUFO_REVISION
#define GUFO_REVISION "unknown"
#endif

namespace gufo::router::mcp {

using server::HttpRequest;
using server::HttpResponse;
using server::Logger;

namespace {

constexpr const char* kPath = "/mcp";

HttpResponse JsonResponse(
    int status, std::string_view reason, const json::Value& body,
    std::vector<std::pair<std::string, std::string>> headers = {}) {
  headers.emplace_back("Content-Type", "application/json");
  headers.emplace_back("Cache-Control", "no-store");
  return HttpResponse{.status = status,
                      .reason = std::string(reason),
                      .body = body.dump(),
                      .headers = std::move(headers)};
}

/// The router's error shape for the HTTP-level refusals of the endpoint
/// itself (wrong method, wrong content type, endpoint off).
HttpResponse HttpError(int status, std::string_view reason,
                       std::string_view message, std::string_view code,
                       std::vector<std::pair<std::string, std::string>> headers =
                           {}) {
  json::Value detail = json::Value::object();
  detail["message"] = std::string(message);
  detail["type"] = "invalid_request_error";
  detail["code"] = std::string(code);
  json::Value body = json::Value::object();
  body["error"] = std::move(detail);
  return JsonResponse(status, reason, body, std::move(headers));
}

json::Value RpcError(int code, std::string_view message,
                     std::optional<json::Value> data = std::nullopt) {
  json::Value error = json::Value::object();
  error["code"] = code;
  error["message"] = std::string(message);
  if (data.has_value())
    error["data"] = std::move(*data);
  return error;
}

json::Value Envelope(const json::Value& id) {
  json::Value body = json::Value::object();
  body["jsonrpc"] = "2.0";
  body["id"] = id;
  return body;
}

/// HTTP `status` with a JSON-RPC error object (the frame's own errors are
/// 400, everything after the frame is 200).
HttpResponse ErrorResponse(int status, const json::Value& id,
                           json::Value error) {
  json::Value body = Envelope(id);
  body["error"] = std::move(error);
  return JsonResponse(status, status == 400 ? "Bad Request" : "OK", body);
}

HttpResponse ResultResponse(const json::Value& id, json::Value result) {
  json::Value body = Envelope(id);
  body["result"] = std::move(result);
  return JsonResponse(200, "OK", body);
}

json::Value SupportedVersions() {
  json::Value list = json::Value::array();
  for (const std::string_view version : kProtocolVersions)
    list.push_back(std::string(version));
  return list;
}

std::string VersionList() {
  std::string out;
  for (const std::string_view version : kProtocolVersions) {
    if (!out.empty())
      out += ", ";
    out += version;
  }
  return out;
}

/// A tool result that is an error the model can read (spec: execution
/// errors are results with isError, not protocol errors).
json::Value ToolError(std::string_view text) {
  json::Value item = json::Value::object();
  item["type"] = "text";
  item["text"] = std::string(text);
  json::Value content = json::Value::array();
  content.push_back(std::move(item));
  json::Value result = json::Value::object();
  result["content"] = std::move(content);
  result["isError"] = true;
  return result;
}

/// The preset's model ids whose reported modality is one of `modalities`
/// ("external" sections without `args = serve <modality>` are not
/// recognised and stay out: the lists are a hint, not a restriction).
std::vector<std::string> ModelIds(
    const std::vector<PresetModel>& models,
    std::initializer_list<std::string_view> modalities) {
  std::vector<std::string> ids;
  for (const PresetModel& model : models) {
    const std::string modality = ReportedModality(model);
    if (std::find(modalities.begin(), modalities.end(), modality) !=
            modalities.end() &&
        std::find(ids.begin(), ids.end(), model.id) == ids.end())
      ids.push_back(model.id);
  }
  return ids;
}

/// `preferred` when the preset has it, else the preset's first model of
/// the modality, else `preferred` as the documented default.
std::string PickDefault(const std::vector<std::string>& ids,
                        std::string_view preferred) {
  if (std::find(ids.begin(), ids.end(), preferred) != ids.end())
    return std::string(preferred);
  if (!ids.empty())
    return ids.front();
  return std::string(preferred);
}

/// Sets `properties.model.default` and names the preset's models in its
/// description. No `enum`: a model the preset lists under another
/// modality spelling (an `[external/...]` section without `args = serve
/// <modality>`) must stay callable; the gate answers 403 model_forbidden
/// for a model the user may not use.
void SetModelHint(json::Value* schema, const std::vector<std::string>& ids,
                  std::string_view preferred) {
  json::Value& model = (*schema)["properties"]["model"];
  model["default"] = PickDefault(ids, preferred);
  std::string description = model.member_str("description");
  if (!ids.empty()) {
    if (!description.empty())
      description += description.back() == '.' ? " " : ". ";
    description += "Modelle in diesem Preset (models in this preset): ";
    for (std::size_t i = 0; i < ids.size(); ++i)
      description += (i == 0 ? "" : ", ") + ids[i];
    description += ".";
    model["description"] = description;
  }
}

/// The uniform answer of every job tool.
json::Value JobOutputSchema() {
  return json::parse(R"json({
    "type": "object",
    "required": ["auftrag_id", "status", "url"],
    "properties": {
      "auftrag_id": {"type": "string", "description": "Auftrags-ID (job id)"},
      "status": {"type": "string",
                 "enum": ["queued", "in_progress", "completed", "failed", "cancelled"]},
      "position": {"type": "integer", "description": "Platz in der Warteschlange (queue position)"},
      "url": {"type": "string", "description": "/v1/jobs/<id>"}
    }
  })json");
}

const char* kVorrangSchema =
    R"json({"type": "integer", "minimum": 0, "maximum": 9, "default": 0,
            "description": "Vorrang 0-9, 9 zuerst (priority, 9 runs first)"})json";

json::Value Vorrang() { return json::parse(kVorrangSchema); }

std::vector<Tool> BuildTools(const Options& options) {
  std::vector<Tool> tools;
  const auto add = [&tools](Tool tool) { tools.push_back(std::move(tool)); };
  const std::vector<std::string> llm = ModelIds(options.models, {"llm"});
  const std::vector<std::string> image = ModelIds(options.models, {"image"});
  const std::vector<std::string> video = ModelIds(options.models, {"video"});
  const std::vector<std::string> tts = ModelIds(options.models, {"tts"});
  const std::vector<std::string> music =
      ModelIds(options.models, {"music", "moss-music"});
  const std::vector<std::string> objekt = ModelIds(options.models, {"objekt"});

  // modelle ----------------------------------------------------------------
  {
    Tool t;
    t.name = "modelle";
    t.title = "Modelle";
    t.description =
        "Listet die Modelle des Routers (Kennung, Art, geladen, lokal oder "
        "Partner). Lists the router's models with modality, load state and "
        "placement.";
    t.input_schema = json::parse(R"json({
      "type": "object",
      "properties": {
        "modalitaet": {"type": "string",
                       "enum": ["text", "bild", "video", "stimme", "musik", "3d", "hochskalieren"],
                       "description": "Nur diese Art (only this modality)"}
      }
    })json");
    t.output_schema = json::parse(R"json({
      "type": "object",
      "required": ["modelle"],
      "properties": {
        "modelle": {"type": "array", "items": {"type": "object",
          "properties": {
            "id": {"type": "string"},
            "modalitaet": {"type": "string"},
            "geladen": {"type": "boolean"},
            "ort": {"type": "string", "enum": ["lokal", "partner"]}
          }}}
      }
    })json");
    add(std::move(t));
  }
  // chat --------------------------------------------------------------------
  {
    Tool t;
    t.name = "chat";
    t.title = "Chat";
    t.area = "text";
    t.description =
        "Eine Frage an ein Sprachmodell des Routers, Antwort sofort (ohne "
        "Stream); mit bilder auch Bildverstehen. Asks a chat model and "
        "returns the answer synchronously; images for vision models.";
    t.input_schema = json::parse(R"json({
      "type": "object",
      "required": ["nachricht"],
      "properties": {
        "nachricht": {"type": "string", "description": "Die Nutzer-Nachricht (the user message)"},
        "system": {"type": "string", "description": "System-Anweisung (system prompt)"},
        "verlauf": {"type": "array",
                    "description": "Frühere Runden (earlier turns), älteste zuerst",
                    "items": {"type": "object", "required": ["rolle", "text"],
                              "properties": {"rolle": {"type": "string", "enum": ["user", "assistant"]},
                                             "text": {"type": "string"}}}},
        "bilder": {"type": "array", "items": {"type": "string"},
                   "description": "Ablage-Pfade /v1/jobs/<id>/files/<name> oder data:-URLs; nur für Bildversteher wie qwen3.8-27b (file paths or data URLs for vision models)"},
        "model": {"type": "string", "description": "Modell-Kennung (model id)"},
        "denken": {"type": "boolean", "default": false, "description": "Nachdenken einschalten (enable thinking)"},
        "max_tokens": {"type": "integer", "minimum": 1},
        "temperature": {"type": "number", "minimum": 0},
        "json": {"type": "boolean", "default": false,
                 "description": "Antwort als JSON-Objekt erzwingen (force a JSON object answer)"}
      }
    })json");
    SetModelHint(&t.input_schema, llm, "qwen3.8-27b");
    t.output_schema = json::parse(R"json({
      "type": "object",
      "required": ["text", "model"],
      "properties": {
        "text": {"type": "string"},
        "gedanken": {"type": "string", "description": "Gedankengang, wenn denken an (thinking)"},
        "model": {"type": "string"},
        "usage": {"type": "object", "properties": {
          "prompt_tokens": {"type": "integer"}, "completion_tokens": {"type": "integer"}}},
        "finish_reason": {"type": "string"}
      }
    })json");
    add(std::move(t));
  }
  // bild_erzeugen -----------------------------------------------------------
  {
    Tool t;
    t.name = "bild_erzeugen";
    t.title = "Bild erzeugen";
    t.area = "bild";
    t.needs_jobs = true;
    t.description =
        "Erzeugt ein Bild aus Text als Auftrag; Ergebnis über auftrag_status "
        "und ablage_lesen. Generates an image from a prompt as a background "
        "job and returns the job id.";
    t.input_schema = json::parse(R"json({
      "type": "object",
      "required": ["prompt"],
      "properties": {
        "prompt": {"type": "string", "description": "Bildbeschreibung (prompt)"},
        "negativ": {"type": "string", "description": "Was nicht im Bild sein soll (negative prompt)"},
        "model": {"type": "string", "description": "Bildmodell (image model id)"},
        "breite": {"type": "integer", "minimum": 64, "maximum": 4096, "description": "Vielfaches von 32, Breite x Höhe höchstens 4 Megapixel (multiple of 32, at most 4 MP)"},
        "hoehe": {"type": "integer", "minimum": 64, "maximum": 4096},
        "schritte": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Rechenschritte (steps)"},
        "seed": {"type": "integer", "minimum": 0},
        "anzahl": {"type": "integer", "minimum": 1, "maximum": 4, "default": 1, "description": "Bilder je Auftrag (images per job)"},
        "lora": {"type": "array", "items": {"type": "object", "required": ["name"],
                 "properties": {"name": {"type": "string"}, "staerke": {"type": "number", "default": 1.0}}},
                 "description": "LoRA-Zusätze (LoRA adapters)"},
        "titel": {"type": "string", "description": "Titel in der Ablage (job title)"}
      }
    })json");
    t.input_schema["properties"]["vorrang"] = Vorrang();
    SetModelHint(&t.input_schema, image, "qwen-image-2.1-turbo");
    t.output_schema = JobOutputSchema();
    add(std::move(t));
  }
  // bild_bearbeiten ---------------------------------------------------------
  {
    Tool t;
    t.name = "bild_bearbeiten";
    t.title = "Bild bearbeiten";
    t.area = "bild";
    t.needs_jobs = true;
    t.description =
        "Bearbeitet ein oder mehrere Bilder nach Anweisung als Auftrag "
        "(Bilder als Ablage-Pfad oder data:-URL). Edits images by "
        "instruction as a background job and returns the job id.";
    t.input_schema = json::parse(R"json({
      "type": "object",
      "required": ["prompt", "bilder"],
      "properties": {
        "prompt": {"type": "string", "description": "Was geändert werden soll (edit instruction)"},
        "bilder": {"type": "array", "minItems": 1, "maxItems": 10, "items": {"type": "string"},
                   "description": "Ablage-Pfade /v1/jobs/<id>/files/<name> oder data:-URLs, zusammen höchstens 20 MB (file paths or data URLs, 20 MB in total)"},
        "maske": {"type": "string", "description": "Maske als Ablage-Pfad oder data:-URL (mask image)"},
        "staerke": {"type": "number", "minimum": 0, "maximum": 1, "description": "Wie stark das Bild verändert wird (edit strength)"},
        "model": {"type": "string", "description": "Bildmodell (image model id)"},
        "schritte": {"type": "integer", "minimum": 1, "maximum": 100},
        "seed": {"type": "integer", "minimum": 0}
      }
    })json");
    t.input_schema["properties"]["vorrang"] = Vorrang();
    SetModelHint(&t.input_schema, image, "qwen-image-2.1");
    t.output_schema = JobOutputSchema();
    add(std::move(t));
  }
  // video_erzeugen ----------------------------------------------------------
  {
    Tool t;
    t.name = "video_erzeugen";
    t.title = "Video erzeugen";
    t.area = "video";
    t.needs_jobs = true;
    t.description =
        "Erzeugt ein Video aus Text, Startbild, Endbild oder Ton als Auftrag "
        "(zählt doppelt im Tageslimit). Generates a video from a prompt and "
        "optional start/end image or audio as a background job.";
    t.input_schema = json::parse(R"json({
      "type": "object",
      "required": ["prompt"],
      "properties": {
        "prompt": {"type": "string"},
        "model": {"type": "string", "description": "Videomodell (video model id)"},
        "startbild": {"type": "string", "description": "Erstes Bild als Ablage-Pfad oder data:-URL (start frame)"},
        "endbild": {"type": "string", "description": "Letztes Bild (end frame)"},
        "ton": {"type": "string", "description": "Tonspur als Ablage-Pfad oder data:-URL, für Sprechvideos (audio track for talking videos)"},
        "sekunden": {"type": "number", "minimum": 1, "description": "Länge (duration in seconds)"},
        "breite": {"type": "integer", "minimum": 64},
        "hoehe": {"type": "integer", "minimum": 64},
        "schritte": {"type": "integer", "minimum": 1},
        "qualitaet": {"type": "string", "enum": ["genau", "schnell", "schnellste"], "default": "genau",
                      "description": "genau = beste Qualität; schnell/schnellste schwächen Ton und Gesichter (exact is the default; fast modes degrade audio and faces)"},
        "negativ": {"type": "string"},
        "seed": {"type": "integer", "minimum": 0}
      }
    })json");
    t.input_schema["properties"]["vorrang"] = Vorrang();
    SetModelHint(&t.input_schema, video, "minimax-h3");
    t.output_schema = JobOutputSchema();
    add(std::move(t));
  }
  // stimme ------------------------------------------------------------------
  {
    Tool t;
    t.name = "stimme";
    t.title = "Stimme";
    t.area = "stimme";
    t.needs_jobs = true;
    t.description =
        "Spricht einen Text als Auftrag (Ergebnis output.wav in der Ablage); "
        "mit probe und wortlaut wird eine Stimme geklont. Text to speech as "
        "a background job; a reference sample with its transcript clones a "
        "voice.";
    t.input_schema = json::parse(R"json({
      "type": "object",
      "required": ["text"],
      "properties": {
        "text": {"type": "string", "description": "Der zu sprechende Text (text to speak)"},
        "model": {"type": "string", "description": "Stimmmodell (speech model id)"},
        "stimme": {"type": "string", "description": "Name der Stimme (voice name)"},
        "sprache": {"type": "string", "description": "Sprachkürzel, z. B. de, en (language code)"},
        "tempo": {"type": "number", "minimum": 0.5, "maximum": 2.0, "description": "Sprechtempo (speed)"},
        "beschreibung": {"type": "string", "description": "Anweisung zur Sprechweise (style instructions)"},
        "probe": {"type": "string", "description": "Stimmprobe als Ablage-Pfad oder data:-URL (reference audio for cloning)"},
        "wortlaut": {"type": "string", "description": "Was die Probe sagt (transcript of the reference)"},
        "format": {"type": "string", "enum": ["wav", "mp3"], "default": "wav"}
      }
    })json");
    t.input_schema["properties"]["vorrang"] = Vorrang();
    SetModelHint(&t.input_schema, tts, "fish-drama-3");
    t.output_schema = JobOutputSchema();
    add(std::move(t));
  }
  // musik -------------------------------------------------------------------
  {
    Tool t;
    t.name = "musik";
    t.title = "Musik";
    t.area = "musik";
    t.needs_jobs = true;
    t.description =
        "Erzeugt Musik aus Stil und Liedtext als Auftrag. Generates music "
        "from a style description and lyrics as a background job.";
    t.input_schema = json::parse(R"json({
      "type": "object",
      "required": ["stil"],
      "properties": {
        "stil": {"type": "string", "description": "Stil, Instrumente, Stimmung, auf Englisch (style tags, English)"},
        "liedtext": {"type": "string", "description": "Liedtext; [Instrumental] für ohne Gesang (lyrics; [Instrumental] for none)"},
        "sekunden": {"type": "number", "minimum": 5, "maximum": 600, "description": "Länge (duration in seconds)"},
        "model": {"type": "string", "description": "Musikmodell (music model id)"},
        "format": {"type": "string", "enum": ["wav", "mp3"], "default": "wav"},
        "seed": {"type": "integer", "minimum": 0}
      }
    })json");
    t.input_schema["properties"]["vorrang"] = Vorrang();
    SetModelHint(&t.input_schema, music, "ace-step-1.5-xl-turbo");
    t.output_schema = JobOutputSchema();
    add(std::move(t));
  }
  // objekt_3d ---------------------------------------------------------------
  {
    Tool t;
    t.name = "objekt_3d";
    t.title = "3D-Objekt";
    t.area = "3d";
    t.needs_jobs = true;
    t.description =
        "Macht aus einem Bild (PNG mit Transparenz) ein 3D-Objekt (GLB) als "
        "Auftrag. Image to 3D (GLB) as a background job.";
    t.input_schema = json::parse(R"json({
      "type": "object",
      "required": ["bild"],
      "properties": {
        "bild": {"type": "string", "description": "PNG mit Alpha als Ablage-Pfad oder data:-URL (PNG with alpha)"},
        "model": {"type": "string", "description": "3D-Modell (3D model id)"},
        "schritte": {"type": "integer", "minimum": 1},
        "guidance": {"type": "number", "minimum": 0},
        "seed": {"type": "integer", "minimum": 0},
        "aufloesung": {"type": "integer", "enum": [256, 512], "description": "Gitterauflösung (grid resolution)"},
        "max_dreiecke": {"type": "integer", "minimum": 1000, "description": "Höchstzahl Dreiecke (max faces)"}
      }
    })json");
    t.input_schema["properties"]["vorrang"] = Vorrang();
    SetModelHint(&t.input_schema, objekt, "trellis2");
    t.output_schema = JobOutputSchema();
    add(std::move(t));
  }
  // hochskalieren -----------------------------------------------------------
  {
    Tool t;
    t.name = "hochskalieren";
    t.title = "Hochskalieren";
    t.area = "bild|video";
    t.needs_jobs = true;
    t.description =
        "Vergrößert ein Bild oder Video 2x/4x als Auftrag (Video zählt "
        "doppelt). Upscales an image or video 2x/4x as a background job.";
    t.input_schema = json::parse(R"json({
      "type": "object",
      "required": ["datei"],
      "properties": {
        "datei": {"type": "string", "description": "Bild oder Video als Ablage-Pfad /v1/jobs/<id>/files/<name> oder data:-URL (file path or data URL)"},
        "faktor": {"type": "integer", "enum": [2, 4], "default": 2},
        "variante": {"type": "string", "enum": ["allgemein", "entrauscht", "anime"], "default": "allgemein",
                     "description": "Modellvariante (general, denoised, anime)"},
        "ausgabe": {"type": "string", "enum": ["png", "jpg"], "default": "png", "description": "Bildformat der Ausgabe (output format for images)"}
      }
    })json");
    t.input_schema["properties"]["vorrang"] = Vorrang();
    t.output_schema = JobOutputSchema();
    add(std::move(t));
  }
  // auftrag_status ----------------------------------------------------------
  {
    Tool t;
    t.name = "auftrag_status";
    t.title = "Auftragsstand";
    t.area = "ablage";
    t.needs_jobs = true;
    t.description =
        "Stand eines Auftrags: Status, Fortschritt, Position, Dateien, "
        "Fehler. Status, progress and result files of a job.";
    t.input_schema = json::parse(R"json({
      "type": "object",
      "required": ["auftrag_id"],
      "properties": {"auftrag_id": {"type": "string", "description": "Auftrags-ID (job id)"}}
    })json");
    t.output_schema = json::parse(R"json({
      "type": "object",
      "required": ["auftrag_id", "status"],
      "properties": {
        "auftrag_id": {"type": "string"},
        "status": {"type": "string", "enum": ["queued", "in_progress", "completed", "failed", "cancelled"]},
        "fortschritt": {"type": ["number", "null"], "description": "0-100 oder null (percent or null)"},
        "geschaetzt": {"type": "boolean", "description": "Fortschritt ist geschätzt (progress is an estimate)"},
        "laeuft_auf": {"type": "string", "description": "lokal oder Partner (where it runs)"},
        "position": {"type": "integer"},
        "dateien": {"type": "array", "items": {"type": "object", "properties": {
          "name": {"type": "string"}, "url": {"type": "string"},
          "content_type": {"type": "string"}, "bytes": {"type": "integer"}}}},
        "fehler": {"type": "string"},
        "vorschau_url": {"type": "string"},
        "erstellt": {"type": "string"},
        "fertig": {"type": "string"}
      }
    })json");
    add(std::move(t));
  }
  // auftrag_liste -----------------------------------------------------------
  {
    Tool t;
    t.name = "auftrag_liste";
    t.title = "Auftragsliste";
    t.area = "ablage";
    t.needs_jobs = true;
    t.description =
        "Die eigenen Aufträge, neueste zuerst, mit Filtern. Lists the "
        "caller's jobs (admins see all), newest first.";
    t.input_schema = json::parse(R"json({
      "type": "object",
      "properties": {
        "status": {"type": "string", "enum": ["queued", "in_progress", "completed", "failed", "cancelled"]},
        "art": {"type": "string", "enum": ["image", "music", "speech", "video", "other"], "description": "Art des Auftrags (job kind)"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
        "suche": {"type": "string", "description": "Suchwort in Titel und Prompt (search text)"}
      }
    })json");
    t.output_schema = json::parse(R"json({
      "type": "object",
      "required": ["auftraege"],
      "properties": {
        "auftraege": {"type": "array", "items": {"type": "object", "properties": {
          "auftrag_id": {"type": "string"}, "status": {"type": "string"},
          "art": {"type": "string"}, "model": {"type": "string"},
          "titel": {"type": "string"}, "fortschritt": {"type": ["number", "null"]},
          "erstellt": {"type": "string"}}}}
      }
    })json");
    add(std::move(t));
  }
  // auftrag_abbrechen -------------------------------------------------------
  {
    Tool t;
    t.name = "auftrag_abbrechen";
    t.title = "Auftrag abbrechen";
    t.area = "ablage";
    t.needs_jobs = true;
    t.description =
        "Bricht einen wartenden oder laufenden Auftrag ab; ein fertiger wird "
        "nur mit loeschen = true samt Dateien entfernt. Cancels a queued or "
        "running job; a finished one is deleted only with loeschen = true.";
    t.input_schema = json::parse(R"json({
      "type": "object",
      "required": ["auftrag_id"],
      "properties": {
        "auftrag_id": {"type": "string"},
        "loeschen": {"type": "boolean", "default": false, "description": "Fertigen Auftrag mit Dateien löschen (delete a finished job)"}
      }
    })json");
    t.output_schema = json::parse(R"json({
      "type": "object",
      "required": ["auftrag_id", "status"],
      "properties": {"auftrag_id": {"type": "string"}, "status": {"type": "string"},
                     "geloescht": {"type": "boolean"}}
    })json");
    add(std::move(t));
  }
  // ablage_lesen ------------------------------------------------------------
  {
    Tool t;
    t.name = "ablage_lesen";
    t.title = "Ablage lesen";
    t.area = "ablage";
    t.needs_jobs = true;
    t.description =
        "Holt eine Ergebnisdatei eines Auftrags: Vorgabe ist der Link mit "
        "Metadaten, mit inhalt = true ein Bild oder Ton eingebettet (bis "
        "max_bytes). Fetches a job's result file: a link by default, the "
        "image or audio inline on request up to max_bytes.";
    t.input_schema = json::parse(R"json({
      "type": "object",
      "required": ["auftrag_id"],
      "properties": {
        "auftrag_id": {"type": "string"},
        "datei": {"type": "string", "description": "Dateiname, z. B. image-1.png; Vorgabe: die erste Mediendatei (file name; default: the first media file)"},
        "inhalt": {"type": "boolean", "default": false, "description": "Inhalt eingebettet liefern (embed the content)"},
        "max_bytes": {"type": "integer", "minimum": 1, "default": 4194304, "description": "Größte eingebettete Datei; darüber nur der Link (inline limit)"}
      }
    })json");
    t.output_schema = json::parse(R"json({
      "type": "object",
      "required": ["auftrag_id", "name", "url"],
      "properties": {
        "auftrag_id": {"type": "string"}, "name": {"type": "string"},
        "url": {"type": "string"}, "content_type": {"type": "string"},
        "bytes": {"type": "integer"}
      }
    })json");
    add(std::move(t));
  }
  return tools;
}

json::Value ToolJson(const Tool& tool) {
  json::Value out = json::Value::object();
  out["name"] = tool.name;
  out["title"] = tool.title;
  out["description"] = tool.description;
  out["inputSchema"] = tool.input_schema;
  if (!tool.output_schema.is_null())
    out["outputSchema"] = tool.output_schema;
  return out;
}

auth::Identity AdminIdentity() {
  auth::Identity who;
  who.source = auth::Identity::Source::kOff;
  who.name = "lokal";
  who.rolle = "admin";
  who.placeholder = true;
  return who;
}

// --- tools/call: arguments, results, inner answers ------------------------

/// Bad arguments of a tools/call: JSON-RPC -32602 naming the field.
struct InvalidParams : std::runtime_error {
  using std::runtime_error::runtime_error;
};

const json::Value* Arg(const json::Value& args, const std::string& key) {
  const json::Value* v = args.find(key);
  return (v == nullptr || v->is_null()) ? nullptr : v;
}

std::string ArgString(const json::Value& args, const std::string& key,
                      bool required) {
  const json::Value* v = Arg(args, key);
  if (v == nullptr) {
    if (required)
      throw InvalidParams("arguments." + key + " is required");
    return {};
  }
  if (!v->is_string())
    throw InvalidParams("arguments." + key + " must be a string");
  if (required && Trim(v->str()).empty())
    throw InvalidParams("arguments." + key + " must not be empty");
  return v->str();
}

std::optional<bool> ArgBool(const json::Value& args, const std::string& key) {
  const json::Value* v = Arg(args, key);
  if (v == nullptr)
    return std::nullopt;
  if (!v->is_bool())
    throw InvalidParams("arguments." + key + " must be true or false");
  return v->as_bool();
}

std::optional<long long> ArgInt(const json::Value& args, const std::string& key,
                                long long min, long long max) {
  const json::Value* v = Arg(args, key);
  if (v == nullptr)
    return std::nullopt;
  const double d = v->as_double(-1e300);
  if (!v->is_number() || d != static_cast<double>(static_cast<long long>(d)))
    throw InvalidParams("arguments." + key + " must be an integer");
  const auto n = static_cast<long long>(d);
  if (n < min || n > max)
    throw InvalidParams("arguments." + key + " must be between " +
                        std::to_string(min) + " and " + std::to_string(max));
  return n;
}

std::optional<double> ArgNumber(const json::Value& args, const std::string& key,
                                double min) {
  const json::Value* v = Arg(args, key);
  if (v == nullptr)
    return std::nullopt;
  if (!v->is_number() || v->as_double() < min)
    throw InvalidParams("arguments." + key + " must be a number >= " +
                        std::to_string(min));
  return v->as_double();
}

/// Job ids and file names travel in a path: one segment, nothing odd.
bool PlainSegment(std::string_view text) {
  if (text.empty() || text.size() > 200 || text == "." || text == "..")
    return false;
  for (const char c : text)
    if (c == '/' || c == '\\' || c == '?' || c == '#' || c == '%' ||
        static_cast<unsigned char>(c) < 0x20)
      return false;
  return true;
}

std::string ArgSegment(const json::Value& args, const std::string& key,
                       bool required) {
  const std::string value = ArgString(args, key, required);
  if (!value.empty() && !PlainSegment(value))
    throw InvalidParams("arguments." + key + " is not a valid name");
  return value;
}

std::string PercentEncode(std::string_view text) {
  static constexpr char kHex[] = "0123456789ABCDEF";
  std::string out;
  for (const unsigned char c : text) {
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      out += static_cast<char>(c);
    } else {
      out += '%';
      out += kHex[c >> 4];
      out += kHex[c & 15];
    }
  }
  return out;
}

/// Unix seconds as ISO 8601 UTC ("" for 0).
std::string IsoUtc(long long seconds) {
  if (seconds <= 0)
    return {};
  const std::time_t t = static_cast<std::time_t>(seconds);
  std::tm tm{};
#ifdef _WIN32
  if (gmtime_s(&tm, &t) != 0)
    return {};
#else
  if (gmtime_r(&t, &tm) == nullptr)
    return {};
#endif
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ",
                tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour,
                tm.tm_min, tm.tm_sec);
  return buf;
}

long long NumberOf(const json::Value& object, const std::string& key) {
  const json::Value* v = object.find(key);
  return v != nullptr && v->is_number() ? static_cast<long long>(v->as_double())
                                        : 0;
}

/// A text content block.
json::Value TextBlock(std::string text) {
  json::Value item = json::Value::object();
  item["type"] = "text";
  item["text"] = std::move(text);
  return item;
}

/// A tool result: `structured` as structuredContent and, for clients
/// without it, its JSON as the first text block; `extra` blocks (an
/// image, a link) follow.
json::Value ToolResult(json::Value structured,
                       std::vector<json::Value> extra = {}) {
  json::Value content = json::Value::array();
  content.push_back(TextBlock(structured.dump()));
  for (json::Value& block : extra)
    content.push_back(std::move(block));
  json::Value result = json::Value::object();
  result["content"] = std::move(content);
  result["structuredContent"] = std::move(structured);
  return result;
}

/// An execution error with the router's error shape: isError, the text
/// "<status> <code>: <message>" and structuredContent {http_status, code,
/// message, retry_after?} (plus `extra` members).
json::Value ToolFailure(int status, std::string code, std::string message,
                        long long retry_after = 0,
                        json::Value extra = json::Value()) {
  json::Value structured = json::Value::object();
  structured["http_status"] = status;
  structured["code"] = code;
  structured["message"] = message;
  if (retry_after > 0)
    structured["retry_after"] = retry_after;
  if (extra.is_object())
    for (const auto& [key, value] : extra.members())
      structured[key] = value;
  json::Value content = json::Value::array();
  content.push_back(
      TextBlock(std::to_string(status) + " " + code + ": " + message));
  json::Value result = json::Value::object();
  result["content"] = std::move(content);
  result["structuredContent"] = std::move(structured);
  result["isError"] = true;
  return result;
}

/// The tool failure for an inner answer >= 400: code and message from the
/// router's error object, else the status and the body's first 200 bytes.
json::Value HttpFailure(const McpService::InnerAnswer& answer) {
  std::string code;
  std::string message;
  try {
    const json::Value body = json::parse(answer.body);
    if (const json::Value* error = body.find("error");
        error != nullptr && error->is_object()) {
      code = error->member_str("code");
      message = error->member_str("message");
    }
  } catch (const std::exception&) {
  }
  if (code.empty())
    code = "http_" + std::to_string(answer.status);
  if (message.empty())
    message = answer.body.empty() ? "no details" : answer.body.substr(0, 200);
  return ToolFailure(answer.status, std::move(code), std::move(message),
                     answer.retry_after);
}

/// A 2xx answer that is not the JSON the tool expected.
json::Value UnexpectedAnswer(const McpService::InnerAnswer& answer) {
  return ToolFailure(answer.status, "unexpected_answer",
                     "unerwartete Antwort (unexpected answer): " +
                         answer.body.substr(0, 200));
}

/// Parses a 2xx JSON answer; nullopt when it is none.
std::optional<json::Value> JsonOf(const McpService::InnerAnswer& answer) {
  try {
    json::Value body = json::parse(answer.body);
    if (body.is_object())
      return body;
  } catch (const std::exception&) {
  }
  return std::nullopt;
}

bool IsError(const json::Value& result) {
  const json::Value* flag = result.find("isError");
  return flag != nullptr && flag->as_bool();
}

/// The German modality name of a /v1/models entry (the `modelle` tool and
/// its filter).
std::string GermanModality(std::string_view modality) {
  if (modality == "llm" || modality == "omni")
    return "text";
  if (modality == "image")
    return "bild";
  if (modality == "video")
    return "video";
  if (modality == "tts")
    return "stimme";
  if (modality == "music" || modality == "moss-music")
    return "musik";
  if (modality == "objekt")
    return "3d";
  if (modality == "upscale")
    return "hochskalieren";
  return std::string(modality);
}

/// The job kind the list filter uses (image|music|speech|video|other),
/// from the wrapped request's path.
std::string KindOfPath(std::string_view path) {
  if (path.starts_with("/v1/images"))
    return "image";
  if (path.starts_with("/v1/audio/music"))
    return "music";
  if (path.starts_with("/v1/audio/speech"))
    return "speech";
  if (path.starts_with("/v1/videos"))
    return "video";
  return "other";
}

/// Result files of a job view as [{name, url, content_type, bytes}].
json::Value FilesOf(const json::Value& view) {
  json::Value out = json::Value::array();
  const json::Value* result = view.find("result");
  const json::Value* files =
      result != nullptr ? result->find("files") : nullptr;
  if (files == nullptr || !files->is_array())
    return out;
  for (const json::Value& file : files->items()) {
    json::Value entry = json::Value::object();
    entry["name"] = file.member_str("name");
    entry["url"] = file.member_str("url");
    entry["content_type"] = file.member_str("content_type");
    entry["bytes"] = NumberOf(file, "bytes");
    out.push_back(std::move(entry));
  }
  return out;
}

/// The job view (jobs.cpp ViewLocked) as the tool's answer; `full` adds
/// progress details, files and error.
json::Value MapJob(const json::Value& view, bool full) {
  json::Value out = json::Value::object();
  const std::string id = view.member_str("id");
  out["auftrag_id"] = id;
  out["status"] = view.member_str("status");
  const json::Value* request = view.find("request");
  const std::string path =
      request != nullptr ? request->member_str("path") : std::string();
  out["art"] = KindOfPath(path);
  if (request != nullptr && !request->member_str("model").empty())
    out["model"] = request->member_str("model");
  if (const json::Value* metadata = view.find("metadata");
      metadata != nullptr && !metadata->member_str("titel").empty())
    out["titel"] = metadata->member_str("titel");
  const json::Value* progress = view.find("progress");
  out["fortschritt"] = progress != nullptr && progress->is_number()
                           ? json::Value(progress->as_double())
                           : json::Value();
  out["erstellt"] = IsoUtc(NumberOf(view, "created_at"));
  if (!full) {
    out["url"] = "/v1/jobs/" + id;
    return out;
  }
  if (const json::Value* flag = view.find("progress_estimated");
      flag != nullptr && flag->is_bool())
    out["geschaetzt"] = flag->as_bool();
  if (const json::Value* on = view.find("running_on");
      on != nullptr && on->is_string())
    out["laeuft_auf"] = on->str() == "local" ? std::string("lokal") : on->str();
  if (const json::Value* position = view.find("position");
      position != nullptr && position->is_number())
    out["position"] = static_cast<long long>(position->as_double());
  out["dateien"] = FilesOf(view);
  if (const json::Value* error = view.find("error");
      error != nullptr && !error->is_null()) {
    if (error->is_string()) {
      if (!error->str().empty())
        out["fehler"] = error->str();
    } else if (error->is_object() && !error->member_str("message").empty()) {
      out["fehler"] = error->member_str("message");
    } else {
      out["fehler"] = error->dump();
    }
  }
  if (const json::Value* preview = view.find("preview_url");
      preview != nullptr && preview->is_string())
    out["vorschau_url"] = preview->str();
  if (const long long done = NumberOf(view, "completed_at"); done > 0)
    out["fertig"] = IsoUtc(done);
  out["url"] = "/v1/jobs/" + id;
  return out;
}

bool Finished(std::string_view status) {
  return status == "completed" || status == "failed" || status == "cancelled";
}

/// Paths a tool may send inward (never /mcp itself).
bool InnerPathAllowed(std::string_view path) {
  return path == "/v1/models" || path == "/v1/chat/completions" ||
         path == "/v1/jobs" || path.starts_with("/v1/jobs/");
}

}  // namespace

bool IsProtocolVersion(std::string_view version) {
  return std::find(std::begin(kProtocolVersions), std::end(kProtocolVersions),
                   version) != std::end(kProtocolVersions);
}

McpService::McpService(Options options) : options_(std::move(options)) {
  if (options_.version.empty())
    options_.version = GUFO_RELEASE_VERSION " (" GUFO_REVISION ")";
  tools_ = BuildTools(options_);
}

bool McpService::Visible(const Tool& tool, const auth::Identity& who) const {
  if (tool.needs_jobs && !options_.jobs_api)
    return false;
  if (tool.area.empty() || who.admin() || who.trusted())
    return true;
  return who.rechte.AllowsArea(tool.area);
}

const Tool* McpService::FindTool(std::string_view name,
                                 const auth::Identity& who) const {
  for (const Tool& tool : tools_)
    if (tool.name == name)
      return Visible(tool, who) ? &tool : nullptr;
  return nullptr;
}

json::Value McpService::ToolsFor(const auth::Identity& who) const {
  json::Value list = json::Value::array();
  for (const Tool& tool : tools_)
    if (Visible(tool, who))
      list.push_back(ToolJson(tool));
  return list;
}

json::Value McpService::Initialize(const json::Value& params) const {
  // Version rule of the spec: the client's version when supported, else
  // the server's own (the client then decides whether to go on).
  const std::string requested = params.member_str("protocolVersion");
  json::Value result = json::Value::object();
  result["protocolVersion"] = IsProtocolVersion(requested)
                                  ? requested
                                  : std::string(kDefaultProtocolVersion);
  json::Value tools = json::Value::object();
  tools["listChanged"] = false;
  json::Value capabilities = json::Value::object();
  capabilities["tools"] = std::move(tools);
  result["capabilities"] = std::move(capabilities);
  json::Value info = json::Value::object();
  info["name"] = kServerName;
  info["title"] = "gufo Router";
  info["version"] = options_.version;
  result["serverInfo"] = std::move(info);
  result["instructions"] =
      "Werkzeuge des gufo-Routers: Chat, Bild, Video, Stimme, Musik, 3D, "
      "Hochskalieren und die Ablage. Lange Arbeiten laufen als Auftrag: das "
      "Werkzeug gibt eine auftrag_id zurück, auftrag_status zeigt den "
      "Fortschritt, ablage_lesen holt das Ergebnis. (gufo router tools: "
      "chat, image, video, speech, music, 3D, upscaling and the job store. "
      "Long work runs as a job: the tool returns an auftrag_id, "
      "auftrag_status reports progress, ablage_lesen fetches the result.)";
  return result;
}

McpService::InnerAnswer McpService::Inner(const HttpRequest& outer,
                                          std::string method, std::string path,
                                          std::string query,
                                          std::string body) const {
  if (!InnerPathAllowed(path))
    throw std::logic_error("mcp: inner path not allowed: " + path);
  if (!options_.serve)
    throw std::logic_error("mcp: no inner dispatch (Options::serve)");
  HttpRequest inner;
  inner.method = std::move(method);
  inner.path = std::move(path);
  inner.query = std::move(query);
  inner.body = std::move(body);
  inner.client_id = outer.client_id;
  inner.is_cancelled = outer.is_cancelled;
  inner.request_id = outer.request_id;
  // The caller's token (never a cookie: the gate refused it before) and
  // its job priority; nothing else of the outer headers.
  for (const auto& [name, value] : outer.headers) {
    if (EqualsIgnoreCase(name, "Authorization") ||
        EqualsIgnoreCase(name, "X-Gufo-Priority"))
      inner.headers.emplace_back(name, value);
  }
  if (!inner.body.empty())
    inner.headers.emplace_back("Content-Type", "application/json");
  inner.headers.emplace_back("Accept", "application/json");
  HttpResponse response = options_.serve(inner);
  InnerAnswer out;
  out.status = response.status;
  for (const auto& [name, value] : response.headers) {
    if (EqualsIgnoreCase(name, "Content-Type"))
      out.content_type = value;
    else if (EqualsIgnoreCase(name, "Retry-After"))
      out.retry_after = std::atoll(value.c_str());
  }
  if (response.streaming_body) {
    // The worker's answer streams through the relay even without
    // `stream`: read it to the end, as the job runner does.
    response.streaming_body([&out](std::string_view chunk) {
      out.body.append(chunk);
      return true;
    });
  } else {
    out.body = std::move(response.body);
  }
  return out;
}

std::string McpService::DefaultModel(std::string_view tool) const {
  for (const Tool& t : tools_)
    if (t.name == tool)
      if (const json::Value* properties = t.input_schema.find("properties"))
        if (const json::Value* model = properties->find("model"))
          return model->member_str("default");
  return {};
}

json::Value McpService::RunModelle(const json::Value& args,
                                   const auth::Identity& who,
                                   const HttpRequest& outer) const {
  const std::string filter = ArgString(args, "modalitaet", false);
  if (!filter.empty()) {
    static constexpr std::string_view kKinds[] = {
        "text", "bild", "video", "stimme", "musik", "3d", "hochskalieren"};
    if (std::find(std::begin(kKinds), std::end(kKinds), filter) ==
        std::end(kKinds))
      throw InvalidParams(
          "arguments.modalitaet must be one of text, bild, video, stimme, "
          "musik, 3d, hochskalieren");
  }
  const InnerAnswer answer = Inner(outer, "GET", "/v1/models");
  if (answer.status >= 400)
    return HttpFailure(answer);
  const std::optional<json::Value> body = JsonOf(answer);
  const json::Value* data = body.has_value() ? body->find("data") : nullptr;
  if (data == nullptr || !data->is_array())
    return UnexpectedAnswer(answer);
  // Only the models the caller may use (Rights::modelle); the gate hid the
  // worker command lines already.
  const bool all_models = who.admin() || who.trusted();
  json::Value list = json::Value::array();
  for (const json::Value& model : data->items()) {
    const std::string id = model.member_str("id");
    if (id.empty() || (!all_models && !who.rechte.AllowsModel(id)))
      continue;
    const std::string modality = GermanModality(model.member_str("modality"));
    if (!filter.empty() && modality != filter)
      continue;
    json::Value entry = json::Value::object();
    entry["id"] = id;
    entry["modalitaet"] = modality;
    const json::Value* loaded = model.find("loaded");
    entry["geladen"] = loaded != nullptr && loaded->as_bool();
    const bool remote = model.find("remote") != nullptr ||
                        model.member_str("placement") == "remote";
    entry["ort"] = remote ? "partner" : "lokal";
    list.push_back(std::move(entry));
  }
  json::Value structured = json::Value::object();
  structured["modelle"] = std::move(list);
  return ToolResult(std::move(structured));
}

json::Value McpService::RunChat(const json::Value& args,
                                const HttpRequest& outer) const {
  const std::string nachricht = ArgString(args, "nachricht", true);
  const std::string system = ArgString(args, "system", false);
  std::string model = ArgString(args, "model", false);
  if (model.empty())
    model = DefaultModel("chat");
  const bool denken = ArgBool(args, "denken").value_or(false);
  const bool as_json = ArgBool(args, "json").value_or(false);
  const auto max_tokens = ArgInt(args, "max_tokens", 1, 1LL << 31);
  const auto temperature = ArgNumber(args, "temperature", 0.0);

  json::Value messages = json::Value::array();
  const auto message = [](std::string role, json::Value content) {
    json::Value m = json::Value::object();
    m["role"] = std::move(role);
    m["content"] = std::move(content);
    return m;
  };
  if (!system.empty())
    messages.push_back(message("system", json::Value(system)));
  if (const json::Value* verlauf = Arg(args, "verlauf")) {
    if (!verlauf->is_array())
      throw InvalidParams("arguments.verlauf must be an array");
    for (const json::Value& turn : verlauf->items()) {
      if (!turn.is_object())
        throw InvalidParams("arguments.verlauf[] must be {rolle, text}");
      const std::string rolle = turn.member_str("rolle");
      if (rolle != "user" && rolle != "assistant")
        throw InvalidParams("arguments.verlauf[].rolle must be user or assistant");
      const json::Value* text = turn.find("text");
      if (text == nullptr || !text->is_string())
        throw InvalidParams("arguments.verlauf[].text must be a string");
      messages.push_back(message(rolle, json::Value(text->str())));
    }
  }
  // The user turn: text alone, or text plus image parts (vision models).
  // An Ablage path is read through the gate (own files only) and sent as
  // a data URL; the worker cannot fetch router paths itself.
  json::Value user_content;
  if (const json::Value* bilder = Arg(args, "bilder");
      bilder != nullptr && bilder->size() > 0) {
    if (!bilder->is_array())
      throw InvalidParams("arguments.bilder must be an array of strings");
    json::Value parts = json::Value::array();
    json::Value text_part = json::Value::object();
    text_part["type"] = "text";
    text_part["text"] = nachricht;
    parts.push_back(std::move(text_part));
    std::size_t index = 0;
    for (const json::Value& bild : bilder->items()) {
      const std::string where = "arguments.bilder[" + std::to_string(index++) + "]";
      if (!bild.is_string())
        throw InvalidParams(where + " must be a string");
      std::string url = bild.str();
      if (url.starts_with("/v1/jobs/") && url.find("/files/") != std::string::npos) {
        const InnerAnswer file = Inner(outer, "GET", url);
        if (file.status >= 400)
          return HttpFailure(file);
        const std::string type = file.content_type.empty()
                                     ? std::string("application/octet-stream")
                                     : file.content_type;
        url = "data:" + type + ";base64," + comfy::Base64Encode(file.body);
      } else if (!url.starts_with("data:")) {
        throw InvalidParams(where +
                            " must be /v1/jobs/<id>/files/<name> or a data: URL");
      }
      json::Value image = json::Value::object();
      json::Value ref = json::Value::object();
      ref["url"] = std::move(url);
      image["type"] = "image_url";
      image["image_url"] = std::move(ref);
      parts.push_back(std::move(image));
    }
    user_content = std::move(parts);
  } else {
    user_content = nachricht;
  }
  messages.push_back(message("user", std::move(user_content)));

  json::Value body = json::Value::object();
  body["model"] = model;
  body["messages"] = std::move(messages);
  body["stream"] = false;
  json::Value kwargs = json::Value::object();
  kwargs["enable_thinking"] = denken;
  body["chat_template_kwargs"] = std::move(kwargs);
  if (max_tokens.has_value())
    body["max_tokens"] = *max_tokens;
  if (temperature.has_value())
    body["temperature"] = *temperature;
  if (as_json) {
    json::Value format = json::Value::object();
    format["type"] = "json_object";
    body["response_format"] = std::move(format);
  }
  const InnerAnswer answer =
      Inner(outer, "POST", "/v1/chat/completions", {}, body.dump());
  if (answer.status >= 400)
    return HttpFailure(answer);
  const std::optional<json::Value> reply = JsonOf(answer);
  const json::Value* choices = reply.has_value() ? reply->find("choices") : nullptr;
  if (choices == nullptr || !choices->is_array() || choices->size() == 0)
    return UnexpectedAnswer(answer);
  const json::Value& choice = choices->items()[0];
  const json::Value* msg = choice.find("message");
  std::string text;
  std::string gedanken;
  if (msg != nullptr) {
    if (const json::Value* content = msg->find("content")) {
      if (content->is_string()) {
        text = content->str();
      } else if (content->is_array()) {
        for (const json::Value& part : content->items())
          if (part.member_str("type") == "text")
            text += part.member_str("text");
      }
    }
    gedanken = msg->member_str("reasoning_content");
  }
  // A worker that leaves the thinking in the text: split it off.
  if (gedanken.empty() && text.starts_with("<think>")) {
    if (const auto end = text.find("</think>"); end != std::string::npos) {
      gedanken = std::string(Trim(text.substr(7, end - 7)));
      text = std::string(Trim(text.substr(end + 8)));
    }
  }
  json::Value structured = json::Value::object();
  structured["text"] = text;
  if (!gedanken.empty())
    structured["gedanken"] = gedanken;
  structured["model"] = reply->member_str("model").empty()
                            ? model
                            : reply->member_str("model");
  json::Value usage = json::Value::object();
  if (const json::Value* u = reply->find("usage"); u != nullptr && u->is_object()) {
    usage["prompt_tokens"] = NumberOf(*u, "prompt_tokens");
    usage["completion_tokens"] = NumberOf(*u, "completion_tokens");
  }
  structured["usage"] = std::move(usage);
  if (!choice.member_str("finish_reason").empty())
    structured["finish_reason"] = choice.member_str("finish_reason");
  return ToolResult(std::move(structured));
}

json::Value McpService::RunAuftragStatus(const json::Value& args,
                                         const HttpRequest& outer) const {
  const std::string id = ArgSegment(args, "auftrag_id", true);
  const InnerAnswer answer = Inner(outer, "GET", "/v1/jobs/" + id);
  if (answer.status >= 400)
    return HttpFailure(answer);
  const std::optional<json::Value> view = JsonOf(answer);
  if (!view.has_value() || view->member_str("id").empty())
    return UnexpectedAnswer(answer);
  return ToolResult(MapJob(*view, true));
}

json::Value McpService::RunAuftragListe(const json::Value& args,
                                        const HttpRequest& outer) const {
  const std::string status = ArgString(args, "status", false);
  const std::string art = ArgString(args, "art", false);
  const std::string suche = ArgString(args, "suche", false);
  const long long limit = ArgInt(args, "limit", 1, 50).value_or(20);
  static constexpr std::string_view kStatus[] = {
      "queued", "in_progress", "completed", "failed", "cancelled"};
  if (!status.empty() && std::find(std::begin(kStatus), std::end(kStatus),
                                   status) == std::end(kStatus))
    throw InvalidParams("arguments.status must be one of queued, in_progress, "
                        "completed, failed, cancelled");
  static constexpr std::string_view kKinds[] = {"image", "music", "speech",
                                                "video", "other"};
  if (!art.empty() && std::find(std::begin(kKinds), std::end(kKinds), art) ==
                          std::end(kKinds))
    throw InvalidParams(
        "arguments.art must be one of image, music, speech, video, other");
  std::string query = "limit=" + std::to_string(limit);
  if (!status.empty())
    query += "&status=" + status;
  if (!art.empty())
    query += "&kind=" + art;
  if (!suche.empty())
    query += "&q=" + PercentEncode(suche);
  const InnerAnswer answer = Inner(outer, "GET", "/v1/jobs", query);
  if (answer.status >= 400)
    return HttpFailure(answer);
  const std::optional<json::Value> body = JsonOf(answer);
  const json::Value* data = body.has_value() ? body->find("data") : nullptr;
  if (data == nullptr || !data->is_array())
    return UnexpectedAnswer(answer);
  json::Value list = json::Value::array();
  for (const json::Value& job : data->items())
    list.push_back(MapJob(job, false));
  json::Value structured = json::Value::object();
  structured["auftraege"] = std::move(list);
  return ToolResult(std::move(structured));
}

json::Value McpService::RunAuftragAbbrechen(const json::Value& args,
                                            const HttpRequest& outer) const {
  const std::string id = ArgSegment(args, "auftrag_id", true);
  const bool loeschen = ArgBool(args, "loeschen").value_or(false);
  // The state first (own jobs only, the gate answers 404 for others): a
  // finished job is never removed by accident.
  const InnerAnswer current = Inner(outer, "GET", "/v1/jobs/" + id);
  if (current.status >= 400)
    return HttpFailure(current);
  const std::optional<json::Value> view = JsonOf(current);
  if (!view.has_value() || view->member_str("id").empty())
    return UnexpectedAnswer(current);
  const std::string status = view->member_str("status");
  if (Finished(status) && !loeschen) {
    json::Value extra = json::Value::object();
    extra["auftrag_id"] = id;
    extra["status"] = status;
    return ToolFailure(409, "job_finished",
                       "Auftrag " + id + " ist schon " + status +
                           "; zum Löschen samt Dateien loeschen = true setzen "
                           "(job already finished; set loeschen to delete it)",
                       0, std::move(extra));
  }
  const InnerAnswer answer = Inner(outer, "DELETE", "/v1/jobs/" + id);
  if (answer.status >= 400)
    return HttpFailure(answer);
  const std::optional<json::Value> reply = JsonOf(answer);
  if (!reply.has_value())
    return UnexpectedAnswer(answer);
  json::Value structured = json::Value::object();
  structured["auftrag_id"] = id;
  const json::Value* deleted = reply->find("deleted");
  if (deleted != nullptr && deleted->as_bool()) {
    structured["status"] = status;
    structured["geloescht"] = true;
  } else {
    structured["status"] = reply->member_str("status").empty()
                               ? status
                               : reply->member_str("status");
    structured["geloescht"] = false;
  }
  return ToolResult(std::move(structured));
}

json::Value McpService::RunAblageLesen(const json::Value& args,
                                       const HttpRequest& outer) const {
  const std::string id = ArgSegment(args, "auftrag_id", true);
  std::string name = ArgSegment(args, "datei", false);
  const bool inhalt = ArgBool(args, "inhalt").value_or(false);
  const std::size_t limit = std::min(
      options_.inline_limit,
      static_cast<std::size_t>(
          ArgInt(args, "max_bytes", 1, 1LL << 40)
              .value_or(static_cast<long long>(options_.inline_limit))));
  // The job's file list names the first media file, type and size; the
  // gate answers 404 for another user's job.
  const InnerAnswer current = Inner(outer, "GET", "/v1/jobs/" + id);
  if (current.status >= 400)
    return HttpFailure(current);
  const std::optional<json::Value> view = JsonOf(current);
  if (!view.has_value() || view->member_str("id").empty())
    return UnexpectedAnswer(current);
  const json::Value files = FilesOf(*view);
  const auto media_kind = [](std::string_view type) -> std::string {
    if (type.starts_with("image/"))
      return "image";
    if (type.starts_with("audio/"))
      return "audio";
    if (type.starts_with("video/"))
      return "video";
    return "";
  };
  std::string content_type;
  long long bytes = -1;
  for (const json::Value& file : files.items()) {
    const std::string file_name = file.member_str("name");
    if (name.empty()) {
      if (media_kind(file.member_str("content_type")).empty())
        continue;
      name = file_name;
    } else if (file_name != name) {
      continue;
    }
    content_type = file.member_str("content_type");
    bytes = NumberOf(file, "bytes");
    break;
  }
  if (name.empty()) {
    json::Value extra = json::Value::object();
    extra["auftrag_id"] = id;
    extra["status"] = view->member_str("status");
    return ToolFailure(404, "file_not_found",
                       "Auftrag " + id + " hat keine Bild-, Ton- oder "
                           "Videodatei (status " + view->member_str("status") +
                           "; job has no media file yet)",
                       0, std::move(extra));
  }
  const std::string path = "/v1/jobs/" + id + "/files/" + name;
  const std::string link = options_.self_name + path;
  const std::string kind = media_kind(content_type);
  const bool inline_kind = kind == "image" || kind == "audio";
  // Fetch when the content is wanted and may fit, or when the list did
  // not know the file (then its answer gives type and size).
  const bool fetch =
      (inhalt && inline_kind && (bytes < 0 || static_cast<std::size_t>(bytes) <= limit)) ||
      bytes < 0;
  std::string data;
  bool embedded = false;
  if (fetch) {
    const InnerAnswer file = Inner(outer, "GET", path);
    if (file.status >= 400)
      return HttpFailure(file);
    if (content_type.empty())
      content_type = file.content_type;
    bytes = static_cast<long long>(file.body.size());
    if (inhalt && !media_kind(content_type).empty() &&
        media_kind(content_type) != "video" &&
        file.body.size() <= limit) {
      data = comfy::Base64Encode(file.body);
      embedded = true;
    }
  }
  json::Value structured = json::Value::object();
  structured["auftrag_id"] = id;
  structured["name"] = name;
  structured["url"] = link;
  structured["content_type"] = content_type;
  if (bytes >= 0)
    structured["bytes"] = bytes;
  structured["eingebettet"] = embedded;
  json::Value block = json::Value::object();
  if (embedded) {
    block["type"] = media_kind(content_type);  // image | audio
    block["data"] = std::move(data);
    block["mimeType"] = content_type;
  } else {
    block["type"] = "resource_link";
    block["uri"] = link;
    block["name"] = name;
    if (!content_type.empty())
      block["mimeType"] = content_type;
    if (bytes >= 0)
      block["size"] = bytes;
  }
  std::vector<json::Value> extra;
  extra.push_back(std::move(block));
  return ToolResult(std::move(structured), std::move(extra));
}

json::Value McpService::Call(std::string_view name, const json::Value& args,
                             const auth::Identity& who,
                             const HttpRequest& outer) {
  const Tool* tool = FindTool(name, who);
  if (tool == nullptr) {
    if (options_.gate != nullptr)
      options_.gate->Log(outer, who, "mcp:" + std::string(name) + ":unbekannt");
    json::Value out = json::Value::object();
    out["error"] = RpcError(kInvalidParams,
                            "Unknown tool: " + std::string(name));
    return out;
  }
  // The model for the log line: the call's or the tool's default.
  std::string model;
  if (const json::Value* given = args.find("model");
      given != nullptr && given->is_string())
    model = given->str();
  else
    model = DefaultModel(tool->name);
  const auto log = [&](std::string_view result) {
    if (options_.gate != nullptr)
      options_.gate->Log(outer, who, "mcp:" + tool->name + ":" + std::string(result),
                         model);
  };
  json::Value result;
  try {
    if (tool->name == "modelle")
      result = RunModelle(args, who, outer);
    else if (tool->name == "chat")
      result = RunChat(args, outer);
    else if (tool->name == "auftrag_status")
      result = RunAuftragStatus(args, outer);
    else if (tool->name == "auftrag_liste")
      result = RunAuftragListe(args, outer);
    else if (tool->name == "auftrag_abbrechen")
      result = RunAuftragAbbrechen(args, outer);
    else if (tool->name == "ablage_lesen")
      result = RunAblageLesen(args, outer);
    else {
      // The media job tools follow in Commit 3.
      log("nicht_umgesetzt");
      return ToolError("Werkzeug '" + tool->name +
                       "' ist noch nicht umgesetzt (Commit 3). Tool not "
                       "implemented yet.");
    }
  } catch (const InvalidParams& error) {
    log("abgelehnt:invalid_params");
    json::Value out = json::Value::object();
    out["error"] = RpcError(kInvalidParams, error.what());
    return out;
  }
  if (IsError(result)) {
    std::string code = "fehler";
    if (const json::Value* structured = result.find("structuredContent"))
      if (!structured->member_str("code").empty())
        code = structured->member_str("code");
    log("abgelehnt:" + code);
  } else {
    log("ok");
  }
  return result;
}

HttpResponse McpService::Rpc(const HttpRequest& req,
                             const auth::Identity& who) {
  const json::Value null_id;
  json::Value message;
  try {
    message = json::parse(req.body);
  } catch (const std::exception& error) {
    return ErrorResponse(
        400, null_id,
        RpcError(kParseError, std::string("Parse error: ") + error.what()));
  }
  if (message.is_array())
    return ErrorResponse(400, null_id,
                         RpcError(kInvalidRequest,
                                  "batch requests are not supported: one "
                                  "JSON-RPC message per POST"));
  if (!message.is_object())
    return ErrorResponse(
        400, null_id,
        RpcError(kInvalidRequest, "a JSON-RPC request object is required"));
  json::Value id;
  bool has_id = false;
  if (const json::Value* given = message.find("id");
      given != nullptr && !given->is_null()) {
    if (!given->is_string() && !given->is_number())
      return ErrorResponse(
          400, null_id,
          RpcError(kInvalidRequest, "id must be a string or a number"));
    id = *given;
    has_id = true;
  }
  if (message.member_str("jsonrpc") != "2.0")
    return ErrorResponse(400, id,
                         RpcError(kInvalidRequest, "jsonrpc must be \"2.0\""));
  const json::Value* method = message.find("method");
  if (method == nullptr) {
    // A JSON-RPC response from the client (we never ask it anything):
    // accepted and dropped, as the transport says.
    if (message.contains("result") || message.contains("error"))
      return HttpResponse{.status = 202, .reason = "Accepted"};
    return ErrorResponse(400, id,
                         RpcError(kInvalidRequest, "method is required"));
  }
  if (!method->is_string())
    return ErrorResponse(400, id,
                         RpcError(kInvalidRequest, "method must be a string"));
  if (!has_id) {
    // A notification (notifications/initialized, notifications/cancelled,
    // ...): nothing to answer.
    return HttpResponse{.status = 202, .reason = "Accepted"};
  }
  const json::Value empty = json::Value::object();
  const json::Value* params = message.find("params");
  if (params == nullptr || params->is_null())
    params = &empty;
  if (!params->is_object())
    return ErrorResponse(200, id,
                         RpcError(kInvalidParams, "params must be an object"));
  const std::string& name = method->str();
  try {
    if (name == "initialize")
      return ResultResponse(id, Initialize(*params));
    if (name == "ping")
      return ResultResponse(id, json::Value::object());
    if (name == "tools/list") {
      json::Value result = json::Value::object();
      result["tools"] = ToolsFor(who);
      return ResultResponse(id, std::move(result));
    }
    if (name == "tools/call") {
      const std::string tool = params->member_str("name");
      if (tool.empty())
        return ErrorResponse(
            200, id, RpcError(kInvalidParams, "params.name is required"));
      const json::Value* args = params->find("arguments");
      if (args == nullptr || args->is_null())
        args = &empty;
      if (!args->is_object())
        return ErrorResponse(
            200, id,
            RpcError(kInvalidParams, "params.arguments must be an object"));
      json::Value out = Call(tool, *args, who, req);
      if (const json::Value* error = out.find("error");
          error != nullptr && error->is_object())
        return ErrorResponse(200, id, *error);
      return ResultResponse(id, std::move(out));
    }
    return ErrorResponse(200, id,
                         RpcError(kMethodNotFound, "Method not found: " + name));
  } catch (const std::exception& error) {
    Logger::Warn("router", "event=mcp_error method=" + name +
                               " detail=\"" + error.what() + "\"");
    return ErrorResponse(200, id, RpcError(kInternalError, "Internal error"));
  }
}

std::optional<HttpResponse> McpService::Handle(const HttpRequest& req) {
  if (req.path != kPath)
    return std::nullopt;
  if (!options_.enabled)
    return HttpError(404, "Not Found",
                     "MCP endpoint is off ([router] mcp = off)", "not_found");
  if (req.method != "POST")
    return HttpError(405, "Method Not Allowed",
                     "POST one JSON-RPC message; no server stream, no "
                     "sessions (GET and DELETE are not offered)",
                     "method_not_allowed", {{"Allow", "POST"}});
  if (!Lower(req.header("Content-Type")).starts_with("application/json"))
    return HttpError(415, "Unsupported Media Type",
                     "the MCP endpoint takes application/json",
                     "json_required");
  const std::string accept = Lower(req.header("Accept"));
  if (!accept.empty() && accept.find("application/json") == std::string::npos &&
      accept.find("*/*") == std::string::npos)
    return HttpError(406, "Not Acceptable",
                     "the MCP endpoint answers application/json",
                     "not_acceptable");
  const std::string version =
      std::string(Trim(req.header("MCP-Protocol-Version")));
  if (!version.empty() && !IsProtocolVersion(version)) {
    json::Value data = json::Value::object();
    data["supported"] = SupportedVersions();
    return ErrorResponse(
        400, json::Value(),
        RpcError(kInvalidRequest,
                 "unsupported MCP-Protocol-Version '" + version +
                     "' (supported: " + VersionList() + ")",
                 std::move(data)));
  }
  const auth::Identity who =
      options_.gate != nullptr ? options_.gate->Identify(req) : AdminIdentity();
  // DNS-rebinding guard of the spec: an Origin must be this router's own
  // (or listed in csrf-allow-origins) for every identity, tokens included;
  // MCP clients send none, browsers always do.
  if (options_.gate != nullptr) {
    if (auto refusal = options_.gate->CsrfCheck(req)) {
      options_.gate->Log(req, who, "abgelehnt:csrf_rejected");
      return std::move(*refusal);
    }
  }
  return Rpc(req, who);
}

}  // namespace gufo::router::mcp
