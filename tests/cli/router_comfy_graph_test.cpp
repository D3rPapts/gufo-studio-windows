// Unit tests for the ComfyUI workflow mapping (comfy_graph). The Qwen-Image
// and Qwen3-TTS fixtures are FilmStudio v100 workflows as ComfyUI recorded
// them in /history (long prompt texts shortened); the H3 and Wan graphs are
// built from the node signatures ComfyUI's /object_info reports.

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

#include "src/cli/serve/router/comfy_graph.hpp"
#include "src/core/json.hpp"

namespace comfy = gufo::router::comfy;
namespace json = gufo::json;

namespace {

constexpr const char* kQwenText = R"({
 "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
 "2": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
 "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
 "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["1", 0], "prompt": "A single mature adult male office worker", "negative_prompt": "chibi, childlike", "vae": ["2", 0], "resolution": 1024}},
 "6": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["5", 2], "seed": 1769224114, "steps": 25, "cfg": 2.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
 "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "life3_zukunftsszenarien/v100_lauf_kurz/stil/demo_personen_r1"}}
})";

constexpr const char* kQwenCanvas = R"({
 "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
 "2": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
 "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
 "r1": {"class_type": "LoadImage", "inputs": {"image": "referenz1_seed1141445518_00001_.png"}},
 "r2": {"class_type": "LoadImage", "inputs": {"image": "referenz1_seed1059564922_00001_.png"}},
 "r10": {"class_type": "LoadImage", "inputs": {"image": "door_reader.png [output]"}},
 "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["1", 0], "prompt": "medium shot of an office", "negative_prompt": "extra people", "vae": ["2", 0], "resolution": 1024, "images.image_10": ["r10", 0], "images.image_1": ["r1", 0], "images.image_2": ["r2", 0]}},
 "c1": {"class_type": "LoadImage", "inputs": {"image": "leinwand.png"}},
 "c2": {"class_type": "VAEEncode", "inputs": {"pixels": ["c1", 0], "vae": ["2", 0]}},
 "6": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["c2", 0], "seed": 1865683568, "steps": 25, "cfg": 2.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
 "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "life3/keyframes/a1_s01_shot01_kf0"}}
})";

constexpr const char* kVoiceDesign = R"({
 "2": {"class_type": "FB_Qwen3TTSVoiceDesign", "inputs": {"text": "Ich bin ALIA, das KI-System dieses Bueros.", "instruct": "Frau, 30 Jahre, kalte Stimme", "seed": 236224718, "model_choice": "1.7B", "device": "auto", "precision": "bf16", "language": "German", "unload_model_after_generate": true}},
 "3": {"class_type": "SaveAudio", "inputs": {"audio": ["2", 0], "filename_prefix": "life3/stimmen/alia_probe_seed236224718"}}
})";

constexpr const char* kVoiceClone = R"({
 "1": {"class_type": "LoadAudio", "inputs": {"audio": "fabian_referenz.flac"}},
 "2": {"class_type": "FB_Qwen3TTSVoiceClone", "inputs": {"target_text": "Wir bleiben an unseren Plaetzen.", "seed": 1616559399, "ref_audio": ["1", 0], "ref_text": "Hallo und herzlich willkommen!", "model_choice": "1.7B", "device": "auto", "precision": "bf16", "language": "German", "unload_model_after_generate": true}},
 "3": {"class_type": "SaveAudio", "inputs": {"audio": ["2", 0], "filename_prefix": "life3/stimmen/fabian_s03"}}
})";

constexpr const char* kH3 = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "strength_model": 1.0}},
 "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "type": "minimax_h3", "device": "default"}},
 "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
 "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
 "10": {"class_type": "LoadImage", "inputs": {"image": "kf0.png"}},
 "11": {"class_type": "LoadImage", "inputs": {"image": "kf1.png"}},
 "20": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["3", 0], "vae": ["4", 0], "prompt": "five office workers at CRT screens", "width": 1344, "height": 768, "length": 120, "first_frame": ["10", 0]}},
 "21": {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": ["20", 0], "latent": ["20", 1], "frame_idx": -1, "vae": ["4", 0], "image": ["11", 0]}},
 "22": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["20", 0]}},
 "30": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["21", 0], "negative": ["22", 0], "latent_image": ["20", 1], "seed": 7, "steps": 8, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
 "31": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["30", 0]}},
 "32": {"class_type": "VAEDecode", "inputs": {"samples": ["31", 0], "vae": ["4", 0]}},
 "33": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["31", 1], "audio_vae": ["5", 0]}},
 "40": {"class_type": "SaveImage", "inputs": {"images": ["32", 0], "filename_prefix": "film/clips/a1_s01_shot01"}},
 "41": {"class_type": "SaveAudio", "inputs": {"audio": ["33", 0], "filename_prefix": "film/clips/a1_s01_shot01"}}
})";

constexpr const char* kWan = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_ti2v_5B_fp16.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
 "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
 "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8}},
 "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "a fox in the snow"}},
 "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "blurry"}},
 "9": {"class_type": "LoadImage", "inputs": {"image": "fuchs.png"}},
 "55": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"vae": ["3", 0], "width": 1280, "height": 704, "length": 49, "batch_size": 1, "start_image": ["9", 0]}},
 "3s": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["55", 0], "seed": 42, "steps": 20, "cfg": 5, "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1}},
 "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3s", 0], "vae": ["3", 0]}},
 "57": {"class_type": "CreateVideo", "inputs": {"images": ["8", 0], "fps": 24}},
 "58": {"class_type": "SaveVideo", "inputs": {"video": ["57", 0], "filename_prefix": "video/wan", "format": "auto", "codec": "auto"}}
})";

// ACE-Step 1.5 as ComfyUI's audio_ace_step_1_5 template wires it (node
// signatures from /object_info on .56).
constexpr const char* kAceStep = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "acestep_v1.5_xl_turbo_bf16.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "qwen_0.6b_ace15.safetensors", "clip_name2": "qwen_4b_ace15.safetensors", "type": "ace", "device": "default"}},
 "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ace_1.5_vae.safetensors"}},
 "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3}},
 "5": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {"clip": ["2", 0], "tags": "90s anime opening, synth pop, female vocals", "lyrics": "[verse]\nNeon im Regen", "seed": 11, "bpm": 128, "duration": 45, "timesignature": "4", "language": "de", "keyscale": "A minor", "generate_audio_codes": true, "cfg_scale": 2.0, "temperature": 0.85, "top_p": 0.9, "top_k": 0, "min_p": 0.0}},
 "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
 "7": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {"seconds": 45, "batch_size": 1}},
 "8": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0], "seed": 11, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "9": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
 "10": {"class_type": "SaveAudio", "inputs": {"audio": ["9", 0], "filename_prefix": "film/musik/titel"}}
})";

// Wan 2.1 I2V 14B + Uni3C the ComfyUI-core way (WanUni3CControlnetApply on
// the model, WanImageToVideo for the start image); the camera comes from
// gufo's GufoCameraMove node.
constexpr const char* kUni3C = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "strength_model": 1.0}},
 "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp16.safetensors", "type": "wan", "device": "default"}},
 "4": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
 "5": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
 "6": {"class_type": "ModelPatchLoader", "inputs": {"name": "Wan21_Uni3C_controlnet_fp16.safetensors"}},
 "10": {"class_type": "LoadImage", "inputs": {"image": "kf_shot07.png"}},
 "11": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["5", 0], "image": ["10", 0], "crop": "none"}},
 "12": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "slow orbit around the office desk"}},
 "13": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": ""}},
 "14": {"class_type": "GufoCameraMove", "inputs": {"camera": "orbit_left", "speed": 0.5, "ease": "ease_in_out", "amount": -1}},
 "15": {"class_type": "WanUni3CControlnetApply", "inputs": {"model": ["2", 0], "model_patch": ["6", 0], "vae": ["4", 0], "render_video": ["14", 0], "strength": 1.0, "start_percent": 0.0, "end_percent": 1.0}},
 "16": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["15", 0], "shift": 5}},
 "20": {"class_type": "WanImageToVideo", "inputs": {"positive": ["12", 0], "negative": ["13", 0], "vae": ["4", 0], "width": 832, "height": 480, "length": 81, "batch_size": 1, "clip_vision_output": ["11", 0], "start_image": ["10", 0]}},
 "30": {"class_type": "KSampler", "inputs": {"model": ["16", 0], "positive": ["20", 0], "negative": ["20", 1], "latent_image": ["20", 2], "seed": 99, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "31": {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}},
 "32": {"class_type": "CreateVideo", "inputs": {"images": ["31", 0], "fps": 16}},
 "33": {"class_type": "SaveVideo", "inputs": {"video": ["32", 0], "filename_prefix": "film/kamera/shot07", "format": "mp4", "codec": "h264"}}
})";

comfy::Plan PlanOf(const char* text) {
  comfy::Plan plan;
  comfy::PlanError error;
  const bool ok = comfy::BuildPlan(json::parse(text), {}, &plan, &error);
  if (!ok)
    std::cerr << "unexpected plan error: " << error.message << "\n";
  assert(ok);
  return plan;
}

comfy::PlanError ErrorOf(const std::string& text) {
  comfy::Plan plan;
  comfy::PlanError error;
  assert(!comfy::BuildPlan(json::parse(text), {}, &plan, &error));
  return error;
}

void TestQwenTextToImage() {
  const comfy::Plan plan = PlanOf(kQwenText);
  assert(plan.tasks.size() == 1);
  const auto& task = plan.tasks[0];
  assert(task.node_id == "6");
  const auto& image = std::get<comfy::ImageTask>(task.spec);
  assert(image.model == "qwen-image-2.1");
  assert(image.prompt == "A single mature adult male office worker");
  assert(image.negative_prompt == "chibi, childlike");
  assert(image.seed == 1769224114ULL);
  assert(image.steps == 25);
  assert(image.size_source == comfy::SizeSource::kSquare);
  assert(image.resolution == 1024);
  assert(image.references.empty());
  assert(task.outputs.size() == 1);
  assert(task.outputs[0].node_id == "8");
  assert(task.outputs[0].kind == comfy::OutputKind::kImage);
  assert(task.outputs[0].prefix ==
         "life3_zukunftsszenarien/v100_lauf_kurz/stil/demo_personen_r1");
  // Negative prompt and cfg pass through as gufo fields.
  assert(image.guidance && *image.guidance == 2.5);
  assert(!image.strength);
  const json::Value extension = comfy::ImageExtension(image);
  assert(extension.member_str("negative_prompt") == "chibi, childlike");
  assert(extension.find("guidance")->as_double() == 2.5);
  assert(extension.find("strength") == nullptr);
}

void TestQwenReferencesAndCanvas() {
  const comfy::Plan plan = PlanOf(kQwenCanvas);
  const auto& image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert(image.size_source == comfy::SizeSource::kCanvas);
  assert((image.canvas == comfy::FileRef{"leinwand.png", "input"}));
  // images.image_N in N order, the "[output]" tag picks the folder.
  assert(image.references.size() == 3);
  assert(image.references[0].name == "referenz1_seed1141445518_00001_.png");
  assert(image.references[1].name == "referenz1_seed1059564922_00001_.png");
  assert((image.references[2] == comfy::FileRef{"door_reader.png", "output"}));
}

void TestVoiceDesign() {
  const comfy::Plan plan = PlanOf(kVoiceDesign);
  const auto& speech = std::get<comfy::SpeechTask>(plan.tasks[0].spec);
  assert(speech.model == "qwen3-tts-voicedesign");
  assert(speech.voice == "voice-design");
  assert(speech.language == "german");
  assert(speech.instructions == "Frau, 30 Jahre, kalte Stimme");
  assert(speech.seed && *speech.seed == 236224718ULL);
  assert(!speech.reference_audio);
  assert(plan.tasks[0].outputs[0].kind == comfy::OutputKind::kAudio);
}

void TestVoiceClone() {
  const comfy::Plan plan = PlanOf(kVoiceClone);
  const auto& speech = std::get<comfy::SpeechTask>(plan.tasks[0].spec);
  assert(speech.model == "qwen3-tts-clone");
  assert(speech.voice == "voice-clone");
  assert((speech.reference_audio ==
          comfy::FileRef{"fabian_referenz.flac", "input"}));
  assert(speech.reference_text == "Hallo und herzlich willkommen!");
  assert(!speech.embedding_only);
}

void TestH3() {
  const comfy::Plan plan = PlanOf(kH3);
  assert(plan.tasks.size() == 1);
  const auto& task = plan.tasks[0];
  assert(task.outputs.size() == 2);
  const auto& video = std::get<comfy::VideoTask>(task.spec);
  assert(video.family == comfy::VideoFamily::kH3);
  assert(video.model == "minimax-h3");
  assert(video.width == 1344 && video.height == 768);
  assert(video.frames == 124);  // 120 snapped up to 17k+5
  assert(video.steps == 8 && video.seed == 7);
  assert(video.lora == "turbo" && !video.lora_strength);
  assert((video.first_frame == comfy::FileRef{"kf0.png", "input"}));
  assert((video.last_frame == comfy::FileRef{"kf1.png", "input"}));
  const json::Value body = comfy::VideoRequestBody(video);
  assert(body.member_str("size") == "1344x768");
  assert(body.find("seconds")->as_double() == 5);
  // seconds and gufo.frames must agree after gufo's 24-fps alignment.
  for (const int frames : {124, 141, 243, 345}) {
    comfy::VideoTask longer = video;
    longer.frames = frames;
    const double seconds =
        comfy::VideoRequestBody(longer).find("seconds")->as_double();
    const int aligned =
        comfy::SnapH3Frames(static_cast<int>(std::ceil(seconds * 24)));
    assert(seconds >= 5 && seconds <= 14.375 && aligned == frames);
  }
  std::string medium = kH3;
  medium.replace(medium.find("\"length\": 120"), 13, "\"length\": 73");
  assert(ErrorOf(medium).message.find("124..345") != std::string::npos);
  assert(body.find("gufo")->find("frames")->as_double() == 124);
  assert(body.find("gufo")->member_str("lora") == "turbo");
  assert(body.find("gufo")->find("steps")->as_double() == 8);
}

void TestH3Guides() {
  std::string text = kH3;
  // 832x512 (FilmStudio v100) is a gufo H3 size now.
  std::string resized = text;
  resized.replace(resized.find("\"width\": 1344"), 13, "\"width\": 832");
  resized.replace(resized.find("\"height\": 768"), 13, "\"height\": 512");
  const comfy::Plan plan = PlanOf(resized.c_str());
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.width == 832 && video.height == 512);
  assert(comfy::VideoRequestBody(video).member_str("size") == "832x512");
  // Sizes outside multiples of 32 / 256..1344 / 768x1344 pixels are a 400.
  std::string odd = resized;
  odd.replace(odd.find("\"width\": 832"), 12, "\"width\": 840");
  comfy::PlanError error = ErrorOf(odd);
  assert(error.message.find("840x512") != std::string::npos);
  assert(error.node_errors.find("20") != nullptr);
  std::string big = text;
  big.replace(big.find("\"height\": 768"), 13, "\"height\": 1344");
  assert(ErrorOf(big).message.find("1344x1344") != std::string::npos);
  // Intermediate guide images become gufo.keyframes; an audio guide becomes
  // the audio reference.
  std::string guides = text;
  guides.replace(
      guides.find(" \"30\": {"), 8,
      " \"12\": {\"class_type\": \"LoadImage\", \"inputs\": "
      "{\"image\": \"mid.png\"}},\n"
      " \"13\": {\"class_type\": \"LoadAudio\", \"inputs\": "
      "{\"audio\": \"zeile.flac\"}},\n"
      " \"23\": {\"class_type\": \"MiniMaxH3AddGuide\", \"inputs\": "
      "{\"positive\": [\"21\", 0], \"latent\": [\"20\", 1], "
      "\"frame_idx\": 60, \"vae\": [\"4\", 0], \"image\": [\"12\", 0]}},\n"
      " \"24\": {\"class_type\": \"MiniMaxH3AddGuide\", \"inputs\": "
      "{\"positive\": [\"23\", 0], \"latent\": [\"20\", 1], "
      "\"frame_idx\": 0, \"audio_vae\": [\"5\", 0], \"audio\": [\"13\", 0]}},\n"
      " \"30\": {");
  guides.replace(guides.find("\"positive\": [\"21\", 0], \"negative\""), 21,
                 "\"positive\": [\"24\", 0]");
  const comfy::Plan guided = PlanOf(guides.c_str());
  const auto& h3 = std::get<comfy::VideoTask>(guided.tasks[0].spec);
  assert(h3.keyframes.size() == 1 && h3.keyframes[0].first == 60);
  assert((h3.keyframes[0].second == comfy::FileRef{"mid.png", "input"}));
  assert((h3.first_frame == comfy::FileRef{"kf0.png", "input"}));
  assert((h3.last_frame == comfy::FileRef{"kf1.png", "input"}));
  assert((h3.audio_reference == comfy::FileRef{"zeile.flac", "input"}));
  assert(h3.audio_reference_frame == 0);
  assert(comfy::VideoRequestBody(h3)
             .find("gufo")
             ->find("audio_reference_frame")
             ->as_double() == 0);
}

void TestWan() {
  const comfy::Plan plan = PlanOf(kWan);
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kWan);
  assert(video.model == "wan2.2-ti2v-5b");
  assert(video.prompt == "a fox in the snow");
  assert(video.negative_prompt == "blurry");
  assert(video.frames == 49 && video.guidance && *video.guidance == 5);
  assert((video.first_frame == comfy::FileRef{"fuchs.png", "input"}));
  assert(plan.tasks[0].outputs[0].kind == comfy::OutputKind::kVideo);
  const json::Value body = comfy::VideoRequestBody(video);
  assert(body.find("seconds")->as_double() == 2);
  assert(body.member_str("size") == "1280x704");
  assert(body.find("gufo")->member_str("negative_prompt") == "blurry");
}

void TestUnsupportedNodes() {
  const comfy::PlanError error = ErrorOf(R"({
    "1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
    "2": {"class_type": "EmptyAceStepLatentAudio", "inputs": {"seconds": 30, "batch_size": 1}},
    "3": {"class_type": "FancyUpscaler", "inputs": {"image": ["2", 0]}},
    "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "x"}}
  })");
  assert(error.type == "invalid_prompt");
  assert(error.message.find("FancyUpscaler (node 3)") != std::string::npos);
  assert(error.message.find("EmptyAceStepLatentAudio (node 2)") !=
         std::string::npos);
  assert(error.message.find("port pending") != std::string::npos);
  assert(error.node_errors.find("2") != nullptr);
  assert(error.node_errors.find("3") != nullptr);
  assert(
      ErrorOf(
          R"({"1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}}})")
          .type == "prompt_no_outputs");
  // A save without a generator upstream.
  assert(ErrorOf(R"({
    "1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
    "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}}
  })")
             .type == "prompt_outputs_failed_validation");
}

std::string Replace(std::string text, const std::string& from,
                    const std::string& to) {
  const auto at = text.find(from);
  assert(at != std::string::npos);
  return text.replace(at, from.size(), to);
}

void TestImageToImage() {
  const std::string text =
      Replace(kQwenCanvas, "\"denoise\": 1.0", "\"denoise\": 0.6");
  const comfy::Plan plan = PlanOf(text.c_str());
  const auto& image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert(image.strength && *image.strength == 0.6);
  assert((image.init_image == comfy::FileRef{"leinwand.png", "input"}));
  assert(comfy::ImageExtension(image).find("strength")->as_double() == 0.6);
  // Without an encoded image there is nothing to denoise from.
  const comfy::PlanError error =
      ErrorOf(Replace(kQwenText, "\"denoise\": 1.0", "\"denoise\": 0.5"));
  assert(error.message.find("VAEEncode") != std::string::npos);
}

void TestRife() {
  // VAEDecode -> RIFE VFI x4 -> SaveImage; the audio output stays plain.
  std::string text =
      Replace(kH3,
              "\"40\": {\"class_type\": \"SaveImage\", \"inputs\": "
              "{\"images\": [\"32\", 0]",
              "\"39\": {\"class_type\": \"RIFE VFI\", \"inputs\": "
              "{\"ckpt_name\": \"rife49.pth\", "
              "\"frames\": [\"32\", 0], \"clear_cache_after_n_frames\": 10, "
              "\"multiplier\": 4, "
              "\"fast_mode\": true, \"ensemble\": true, \"scale_factor\": 1.0, "
              "\"dtype\": "
              "\"float32\", \"torch_compile\": false, \"batch_size\": 1}},\n"
              " \"40\": {\"class_type\": \"SaveImage\", \"inputs\": "
              "{\"images\": [\"39\", 0]");
  const comfy::Plan plan = PlanOf(text.c_str());
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.interpolate == 4 && video.fps == 96);
  assert(comfy::VideoRequestBody(video)
             .find("gufo")
             ->find("interpolate")
             ->as_double() == 4);
  // rife49 (FilmStudio) and rife47 are the weights gufo serves: no warning.
  for (const auto& warning : plan.warnings)
    assert(warning.find("rife") == std::string::npos);
  const comfy::Plan other =
      PlanOf(Replace(text, "rife49.pth", "rife426.pth").c_str());
  bool warned = false;
  for (const auto& warning : other.warnings)
    warned = warned || warning.find("rife426") != std::string::npos;
  assert(warned);
  // A second, plain image output of the same video conflicts.
  text = Replace(text, " \"41\": {",
                 " \"42\": {\"class_type\": \"PreviewImage\", \"inputs\": "
                 "{\"images\": [\"32\", 0]}},\n \"41\": {");
  assert(ErrorOf(text).message.find("same RIFE") != std::string::npos);
  // RIFE on a still image has no gufo endpoint.
  const comfy::PlanError still = ErrorOf(
      Replace(kQwenText, "\"images\": [\"7\", 0]",
              "\"images\": [\"9\", 0]}}, \"9\": {\"class_type\": \"RIFE VFI\", "
              "\"inputs\": {\"frames\": [\"7\", 0], \"multiplier\": 2"));
  assert(still.message.find("H3 or Wan") != std::string::npos);
}

constexpr const char* kDialogue = R"({
 "1": {"class_type": "LoadAudio", "inputs": {"audio": "anna.wav"}},
 "2": {"class_type": "LoadAudio", "inputs": {"audio": "ben.flac"}},
 "3": {"class_type": "FB_Qwen3TTSVoiceClonePrompt", "inputs": {"ref_audio": ["1", 0], "ref_text": "Hallo, ich bin Anna.", "model_choice": "1.7B", "device": "auto", "precision": "bf16", "attention": "auto"}},
 "4": {"class_type": "FB_Qwen3TTSVoiceClonePrompt", "inputs": {"ref_audio": ["2", 0], "ref_text": "", "model_choice": "1.7B", "device": "auto", "precision": "bf16", "attention": "auto", "x_vector_only": true}},
 "5": {"class_type": "FB_Qwen3TTSRoleBank", "inputs": {"role_name_1": "Anna", "prompt_1": ["3", 0], "role_name_2": "Ben", "prompt_2": ["4", 0]}},
 "6": {"class_type": "FB_Qwen3TTSDialogueInference", "inputs": {"script": "Anna: Guten Morgen.\n\nben: Morgen!\r\nAnna： Wie geht es dir?", "role_bank": ["5", 0], "model_choice": "1.7B", "device": "auto", "precision": "bf16", "language": "Auto", "pause_linebreak": 0.25, "period_pause": 0.4, "comma_pause": 0.2, "question_pause": 0.6, "hyphen_pause": 0.3, "merge_outputs": true, "batch_size": 4, "seed": 9}},
 "7": {"class_type": "SaveAudio", "inputs": {"audio": ["6", 0], "filename_prefix": "dialog/szene1"}},
 "8": {"class_type": "FB_Qwen3TTSVoiceClone", "inputs": {"target_text": "Allein.", "voice_clone_prompt": ["3", 0], "model_choice": "1.7B", "device": "auto", "precision": "bf16", "language": "German"}},
 "9": {"class_type": "SaveAudio", "inputs": {"audio": ["8", 0], "filename_prefix": "dialog/anna"}}
})";

void TestDialogue() {
  const comfy::Plan plan = PlanOf(kDialogue);
  assert(plan.tasks.size() == 2);
  const comfy::Task* dialogue_task = nullptr;
  const comfy::Task* clone_task = nullptr;
  for (const auto& task : plan.tasks)
    (task.node_id == "6" ? dialogue_task : clone_task) = &task;
  const auto& dialogue = std::get<comfy::DialogueTask>(dialogue_task->spec);
  assert(dialogue.lines.size() == 3 && dialogue.merge);
  assert(dialogue.pause_seconds == 0.25);
  assert(dialogue.lines[0].text == "Guten Morgen.");
  assert((dialogue.lines[0].reference_audio ==
          comfy::FileRef{"anna.wav", "input"}));
  assert(dialogue.lines[0].reference_text == "Hallo, ich bin Anna.");
  assert(!dialogue.lines[0].embedding_only);
  assert(dialogue.lines[1].text == "Morgen!");
  assert((dialogue.lines[1].reference_audio ==
          comfy::FileRef{"ben.flac", "input"}));
  assert(dialogue.lines[1].embedding_only);
  assert(dialogue.lines[2].text == "Wie geht es dir?");
  assert(dialogue.lines[2].model == "qwen3-tts-clone");
  assert(dialogue.lines[2].seed && *dialogue.lines[2].seed == 9);
  const auto& clone = std::get<comfy::SpeechTask>(clone_task->spec);
  assert((clone.reference_audio == comfy::FileRef{"anna.wav", "input"}));
  assert(clone.reference_text == "Hallo, ich bin Anna.");
  assert(clone.language == "german");
  // An unknown role is a validation error.
  const comfy::PlanError error =
      ErrorOf(Replace(kDialogue, "ben: Morgen!", "Carl: Morgen!"));
  assert(error.message.find("Carl") != std::string::npos);
}

void TestAceStep() {
  const comfy::Plan plan = PlanOf(kAceStep);
  assert(plan.tasks.size() == 1);
  const auto& task = plan.tasks[0];
  assert(task.node_id == "8");
  const auto& music = std::get<comfy::MusicTask>(task.spec);
  assert(music.model == "ace-step-1.5-xl-turbo");
  assert(music.tags == "90s anime opening, synth pop, female vocals");
  assert(music.duration == 45 && music.seed == 11 && music.steps == 8);
  assert(task.outputs.size() == 1 &&
         task.outputs[0].kind == comfy::OutputKind::kAudio);
  assert(task.outputs[0].prefix == "film/musik/titel");
  const json::Value body = comfy::MusicRequestBody(music);
  assert(body.member_str("model") == "ace-step-1.5-xl-turbo");
  assert(body.member_str("prompt") == music.tags);
  assert(body.member_str("lyrics") == "[verse]\nNeon im Regen");
  assert(body.find("duration")->as_double() == 45);
  assert(body.find("seed")->as_double() == 11);
  assert(body.find("steps")->as_double() == 8);
  assert(body.find("bpm")->as_double() == 128);
  assert(body.member_str("timesignature") == "4");
  assert(body.member_str("language") == "de");
  assert(body.member_str("keyscale") == "A minor");
  assert(body.find("audio_codes")->as_bool());
  assert(body.find("lm_cfg")->as_double() == 2.0);
  assert(body.find("lm_temperature")->as_double() == 0.85);
  assert(body.find("lm_top_p")->as_double() == 0.9);
  assert(body.member_str("response_format") == "wav");
  assert(plan.warnings.empty());
  // A different encoder duration, seed or a top_k is reported, not fatal.
  std::string text = Replace(kAceStep, "\"duration\": 45", "\"duration\": 60");
  text = Replace(text, "\"top_k\": 0", "\"top_k\": 40");
  assert(PlanOf(text.c_str()).warnings.size() == 2);
  // PreviewAudio works too; a SaveImage of music does not.
  const comfy::Plan preview = PlanOf(
      Replace(kAceStep,
              "\"SaveAudio\", \"inputs\": {\"audio\": [\"9\", 0], "
              "\"filename_prefix\": \"film/musik/titel\"}",
              "\"PreviewAudio\", \"inputs\": {\"audio\": [\"9\", 0]}")
          .c_str());
  assert(preview.tasks[0].outputs[0].temp);
  assert(ErrorOf(Replace(kAceStep, "\"SaveAudio\", \"inputs\": {\"audio\"",
                         "\"SaveImage\", \"inputs\": {\"images\""))
             .message.find("cannot store") != std::string::npos);
  assert(ErrorOf(Replace(kAceStep, "\"seconds\": 45", "\"seconds\": 700"))
             .message.find("1..600") != std::string::npos);
  assert(ErrorOf(Replace(kAceStep, "\"batch_size\": 1", "\"batch_size\": 2"))
             .node_errors.find("7") != nullptr);
}

void TestUni3C() {
  const comfy::Plan plan = PlanOf(kUni3C);
  assert(plan.tasks.size() == 1);
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kCamera);
  assert(video.model == "wan2.1-uni3c");
  assert(video.prompt == "slow orbit around the office desk");
  assert(video.width == 832 && video.height == 480 && video.frames == 81);
  assert(video.fps == 16 && video.steps == 4 && video.seed == 99);
  assert((video.first_frame == comfy::FileRef{"kf_shot07.png", "input"}));
  assert(video.camera && video.camera->type == "orbit_left");
  assert(video.camera->speed == 0.5 && video.camera->ease == "ease_in_out");
  assert(!video.camera->amount && !video.render_video);
  assert(plan.warnings.empty());
  const json::Value body = comfy::VideoRequestBody(video);
  assert(body.member_str("model") == "wan2.1-uni3c");
  assert(body.member_str("size") == "832x480");
  assert(body.find("seconds")->as_double() == 5);
  const json::Value* gufo = body.find("gufo");
  assert(gufo->find("frames")->as_double() == 81);
  assert(gufo->find("steps")->as_double() == 4);
  assert(gufo->find("seed")->as_double() == 99);
  assert(gufo->find("negative_prompt") == nullptr);
  assert(gufo->find("guidance") == nullptr);
  const json::Value* camera = gufo->find("camera");
  assert(camera->member_str("type") == "orbit_left");
  assert(camera->find("speed")->as_double() == 0.5);
  assert(camera->member_str("ease") == "ease_in_out");
  assert(camera->find("amount") == nullptr);
  // 33 frames = 2 s; an explicit amount is sent.
  std::string shorter = Replace(kUni3C, "\"length\": 81", "\"length\": 33");
  shorter = Replace(shorter, "\"amount\": -1", "\"amount\": 30");
  const comfy::Plan two_plan = PlanOf(shorter.c_str());
  const auto& two = std::get<comfy::VideoTask>(two_plan.tasks[0].spec);
  const json::Value two_body = comfy::VideoRequestBody(two);
  assert(two_body.find("seconds")->as_double() == 2);
  assert(two_body.find("gufo")->find("camera")->find("amount")->as_double() ==
         30);
  // A rendered guide video (LoadVideo -> GetVideoComponents) instead.
  std::string rendered = Replace(
      kUni3C,
      "\"14\": {\"class_type\": \"GufoCameraMove\", \"inputs\": {\"camera\": "
      "\"orbit_left\", \"speed\": 0.5, \"ease\": \"ease_in_out\", "
      "\"amount\": -1}}",
      "\"14\": {\"class_type\": \"GetVideoComponents\", \"inputs\": "
      "{\"video\": [\"17\", 0]}},\n \"17\": {\"class_type\": \"LoadVideo\", "
      "\"inputs\": {\"file\": \"render_shot07.mp4\"}}");
  const comfy::Plan guided_plan = PlanOf(rendered.c_str());
  const auto& guided =
      std::get<comfy::VideoTask>(guided_plan.tasks[0].spec);
  assert(!guided.camera);
  assert((guided.render_video ==
          comfy::FileRef{"render_shot07.mp4", "input"}));
  // Errors: unknown move, bad ease, too large, wrong length, no start image,
  // a foreign LoRA.
  assert(ErrorOf(Replace(kUni3C, "\"orbit_left\"", "\"spiral\""))
             .message.find("spiral") != std::string::npos);
  assert(ErrorOf(Replace(kUni3C, "\"ease_in_out\"", "\"bounce\""))
             .message.find("ease") != std::string::npos);
  assert(ErrorOf(Replace(kUni3C, "\"height\": 480", "\"height\": 832"))
             .message.find("832x832") != std::string::npos);
  assert(ErrorOf(Replace(kUni3C, "\"length\": 81", "\"length\": 80"))
             .message.find("4k+1") != std::string::npos);
  assert(ErrorOf(Replace(kUni3C, ", \"start_image\": [\"10\", 0]", ""))
             .message.find("start_image") != std::string::npos);
  assert(ErrorOf(Replace(kUni3C, "lightx2v_I2V_14B_480p_cfg_step_distill",
                         "anime_style"))
             .message.find("LoRA") != std::string::npos);
  // RIFE after the camera video doubles 16 fps.
  const std::string rife = Replace(
      kUni3C, "\"32\": {\"class_type\": \"CreateVideo\", \"inputs\": "
              "{\"images\": [\"31\", 0]",
      "\"34\": {\"class_type\": \"RIFE VFI\", \"inputs\": {\"frames\": "
      "[\"31\", 0], \"multiplier\": 2}},\n \"32\": {\"class_type\": "
      "\"CreateVideo\", \"inputs\": {\"images\": [\"34\", 0]");
  const comfy::Plan smooth_plan = PlanOf(rife.c_str());
  const auto& smooth =
      std::get<comfy::VideoTask>(smooth_plan.tasks[0].spec);
  assert(smooth.interpolate == 2 && smooth.fps == 32);
}

void TestWanFunCamera() {
  // ComfyUI's Wan Fun Camera form: WanCameraEmbedding -> WanCameraImageToVideo.
  std::string text = Replace(
      kUni3C, "\"model\": [\"15\", 0], \"shift\": 5",
      "\"model\": [\"2\", 0], \"shift\": 5");
  text = Replace(
      text,
      "\"20\": {\"class_type\": \"WanImageToVideo\", \"inputs\": "
      "{\"positive\": [\"12\", 0], \"negative\": [\"13\", 0], \"vae\": "
      "[\"4\", 0], \"width\": 832, \"height\": 480, \"length\": 81, "
      "\"batch_size\": 1,",
      "\"18\": {\"class_type\": \"WanCameraEmbedding\", \"inputs\": "
      "{\"camera_pose\": \"Zoom In\", \"width\": 832, \"height\": 480, "
      "\"length\": 81, \"speed\": 2.0}},\n \"20\": {\"class_type\": "
      "\"WanCameraImageToVideo\", \"inputs\": {\"positive\": [\"12\", 0], "
      "\"negative\": [\"13\", 0], \"vae\": [\"4\", 0], \"width\": 832, "
      "\"height\": 480, \"length\": 81, \"batch_size\": 1, "
      "\"camera_conditions\": [\"18\", 0],");
  // Nodes 6, 14 and 15 stay in the graph unconnected: harmless plumbing.
  const comfy::Plan plan = PlanOf(text.c_str());
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kCamera);
  assert(video.camera && video.camera->type == "dolly_in");
  assert(video.camera->speed == 2.0);
  assert(plan.warnings.size() == 1 &&
         plan.warnings[0].find("Fun Camera") != std::string::npos);
  // Roll has no gufo preset.
  assert(ErrorOf(Replace(text, "\"Zoom In\"", "\"ClockWise (CW)\""))
             .message.find("GufoCameraMove") != std::string::npos);
  // Without a camera it is plain image to video on wan2.1-uni3c.
  const comfy::Plan plain = PlanOf(
      Replace(text, ", \"camera_conditions\": [\"18\", 0]", "").c_str());
  const auto& still = std::get<comfy::VideoTask>(plain.tasks[0].spec);
  assert(!still.camera && !still.render_video);
  assert(comfy::VideoRequestBody(still).find("gufo")->find("camera") ==
         nullptr);
}

void TestWavJoin() {
  comfy::Wav a{24000, 1, std::string(4, '\x01')};
  comfy::Wav b{24000, 1, std::string(2, '\x02')};
  const auto joined =
      comfy::JoinWavs({comfy::BuildWav(a), comfy::BuildWav(b)}, 0.001);
  assert(joined);
  const auto parsed = comfy::ParseWav(*joined);
  assert(parsed && parsed->sample_rate == 24000 && parsed->channels == 1);
  // 4 bytes + 24 silent samples (48 bytes) + 2 bytes.
  assert(parsed->pcm.size() == 4 + 48 + 2);
  assert(parsed->pcm.substr(4, 48) == std::string(48, '\0'));
  comfy::Wav other{16000, 1, "xx"};
  assert(!comfy::JoinWavs({comfy::BuildWav(a), comfy::BuildWav(other)}, 0));
  assert(!comfy::ParseWav("RIFF....WAVE"));
}

void TestHelpers() {
  assert(comfy::SafeRelative("a\\b/c.png") ==
         std::optional<std::string>("a/b/c.png"));
  assert(!comfy::SafeRelative("../x"));
  assert(!comfy::SafeRelative("a/../../x"));
  assert(!comfy::SafeRelative("/etc/passwd"));
  assert(!comfy::SafeRelative("C:/Windows"));
  assert(comfy::SafeRelative("") == std::optional<std::string>(""));
  assert((comfy::ScaleToArea(1024, 1024, 1024) ==
          std::pair<int, int>{1024, 1024}));
  assert(
      (comfy::ScaleToArea(1344, 832, 1024) == std::pair<int, int>{1312, 800}));
  assert((comfy::ScaleToArea(1000, 600, 0) == std::pair<int, int>{992, 608}));
  assert(comfy::SnapH3Frames(124) == 124);
  assert(comfy::SnapH3Frames(125) == 141);
  assert(comfy::SnapH3Frames(22) == 22);
  assert(comfy::SnapH3Frames(1) == 5);
  std::string png("\x89PNG\r\n\x1a\n\0\0\0\rIHDR\0\0\x05\x40\0\0\x03\x40", 24);
  assert((comfy::ImageDimensions(png) == std::pair<int, int>{1344, 832}));
  const std::string jpeg(
      "\xFF\xD8\xFF\xE0\x00\x04\x00\x00\xFF\xC0\x00\x11\x08\x02\x00\x03\x40",
      17);
  assert((comfy::ImageDimensions(jpeg) == std::pair<int, int>{832, 512}));
  assert(!comfy::ImageDimensions("nope"));
  const std::string bytes("\x00\x01\xFE\xFFhello", 9);
  assert(comfy::Base64Decode(comfy::Base64Encode(bytes)) == bytes);
  assert(comfy::Base64Encode("ab") == "YWI=");
  assert(!comfy::Base64Decode("**"));
  const auto parts = std::vector<comfy::MultipartPart>{
      {"subfolder", "", "", "x"},
      {"image", "a b.png", "image/png", std::string("\r\n--\0bin", 8)}};
  std::vector<comfy::MultipartPart> parsed;
  std::string error;
  assert(comfy::ParseMultipart("multipart/form-data; boundary=XyZ",
                               comfy::BuildMultipart(parts, "XyZ"), &parsed,
                               &error));
  assert(parsed.size() == 2 && parsed[1].filename == "a b.png");
  assert(parsed[1].data == parts[1].data && parsed[0].data == "x");
  const auto name = comfy::SplitPrefix("life3/stimmen/alia");
  assert(name && name->subfolder == "life3/stimmen" && name->base == "alia");
  assert(!comfy::SplitPrefix("../escape"));
  assert(comfy::NextCounter({"alia_00001.flac", "alia_00007_.png",
                             "alia_x_00009_.png", "other_00020_.png"},
                            "alia") == 8);
  assert(comfy::OutputFilename("kf", 3, comfy::OutputKind::kImage, "png") ==
         "kf_00003_.png");
  assert(comfy::OutputFilename("alia", 1, comfy::OutputKind::kAudio, "flac") ==
         "alia_00001.flac");
  assert((comfy::ParseFileRef("sub\\a.png [temp]", "input") ==
          comfy::FileRef{"sub/a.png", "temp"}));
}

}  // namespace

int main() {
  TestQwenTextToImage();
  TestQwenReferencesAndCanvas();
  TestVoiceDesign();
  TestVoiceClone();
  TestH3();
  TestH3Guides();
  TestWan();
  TestUnsupportedNodes();
  TestImageToImage();
  TestRife();
  TestDialogue();
  TestAceStep();
  TestUni3C();
  TestWanFunCamera();
  TestWavJoin();
  TestHelpers();
  std::cout << "router_comfy_graph_test: all passed\n";
  return 0;
}
