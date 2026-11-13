#include "src/cli/serve/router/mcp.hpp"

#include <algorithm>
#include <exception>
#include <initializer_list>
#include <iterator>
#include <utility>

#include "src/cli/serve/logging.hpp"
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

json::Value McpService::Call(std::string_view name, const json::Value& args,
                             const auth::Identity& who,
                             const HttpRequest& outer) {
  (void)args;
  const Tool* tool = FindTool(name, who);
  if (tool == nullptr) {
    if (options_.gate != nullptr)
      options_.gate->Log(outer, who, "mcp:" + std::string(name) + ":unbekannt");
    json::Value out = json::Value::object();
    out["error"] = RpcError(kInvalidParams,
                            "Unknown tool: " + std::string(name));
    return out;
  }
  // Commit 1: the table and the frame only; the builders that turn a call
  // into an inner request follow in Commit 2/3.
  if (options_.gate != nullptr)
    options_.gate->Log(outer, who, "mcp:" + tool->name + ":nicht_umgesetzt");
  return ToolError("Werkzeug '" + tool->name +
                   "' ist noch nicht umgesetzt (Commit 2/3). Tool not "
                   "implemented yet.");
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
