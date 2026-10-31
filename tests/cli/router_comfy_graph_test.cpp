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
#include "src/cli/serve/router/comfy_preview.hpp"
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

// InfiniteTalk as ComfyUI's template wires it (WanInfiniteTalkToVideo with
// a DynamicCombo "mode"; its sub-inputs are named "mode.<input>").
constexpr const char* kTalk = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "strength_model": 1.0}},
 "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp16.safetensors", "type": "wan", "device": "default"}},
 "4": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
 "5": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
 "6": {"class_type": "ModelPatchLoader", "inputs": {"name": "wan2.1_infiniteTalk_single_fp16.safetensors"}},
 "7": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": "wav2vec2-chinese-base_fp16.safetensors"}},
 "8": {"class_type": "LoadAudio", "inputs": {"audio": "anna_satz1.wav"}},
 "9": {"class_type": "AudioEncoderEncode", "inputs": {"audio_encoder": ["7", 0], "audio": ["8", 0]}},
 "10": {"class_type": "LoadImage", "inputs": {"image": "anna_portrait.png"}},
 "11": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["5", 0], "image": ["10", 0], "crop": "none"}},
 "12": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "a woman talks calmly to the camera"}},
 "13": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": ""}},
 "16": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["20", 0], "shift": 5}},
 "20": {"class_type": "WanInfiniteTalkToVideo", "inputs": {"mode": "single_speaker", "model": ["2", 0], "model_patch": ["6", 0], "positive": ["12", 0], "negative": ["13", 0], "vae": ["4", 0], "width": 832, "height": 480, "length": 81, "clip_vision_output": ["11", 0], "start_image": ["10", 0], "audio_encoder_output_1": ["9", 0], "motion_frame_count": 9, "audio_scale": 1.0}},
 "30": {"class_type": "KSampler", "inputs": {"model": ["16", 0], "positive": ["20", 1], "negative": ["20", 2], "latent_image": ["20", 3], "seed": 7, "steps": 4, "cfg": 1.0, "sampler_name": "lcm", "scheduler": "simple", "denoise": 1.0}},
 "31": {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}},
 "32": {"class_type": "CreateVideo", "inputs": {"images": ["31", 0], "audio": ["8", 0], "fps": 25}},
 "33": {"class_type": "SaveVideo", "inputs": {"video": ["32", 0], "filename_prefix": "film/dialog/anna_s01", "format": "mp4", "codec": "h264"}}
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

void TestMaskedEdit() {
  // SetLatentNoiseMask over VAEEncode with LoadImage's MASK output (1 - alpha).
  std::string text = Replace(
      kQwenCanvas, R"("latent_image": ["c2", 0])", R"("latent_image": ["m2", 0])");
  text = Replace(text, R"( "6": {"class_type": "KSampler")",
                 R"( "m1": {"class_type": "LoadImage", "inputs": {"image": "maske.png"}},
 "m2": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["c2", 0], "mask": ["m1", 1]}},
 "6": {"class_type": "KSampler")");
  comfy::Plan plan = PlanOf(text.c_str());
  auto image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert((image.init_image == comfy::FileRef{"leinwand.png", "input"}));
  assert((image.mask == comfy::FileRef{"maske.png", "input"}));
  assert(image.mask_channel == "alpha" && !image.strength);
  assert(image.size_source == comfy::SizeSource::kCanvas);
  assert(image.references.size() == 3);
  assert(comfy::ImageExtension(image).member_str("mask_channel") == "alpha");
  // With denoise below 1 both apply.
  plan = PlanOf(Replace(text, "\"denoise\": 1.0", "\"denoise\": 0.7").c_str());
  image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert(image.strength && *image.strength == 0.7 && !image.mask.name.empty());
  // LoadImageMask keeps its channel.
  const std::string channel = Replace(
      text, R"("m1": {"class_type": "LoadImage", "inputs": {"image": "maske.png"}})",
      R"("m1": {"class_type": "LoadImageMask", "inputs": {"image": "maske.png [output]", "channel": "red"}})");
  plan = PlanOf(Replace(channel, R"("mask": ["m1", 1])", R"("mask": ["m1", 0])")
                    .c_str());
  image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert((image.mask == comfy::FileRef{"maske.png", "output"}));
  assert(image.mask_channel == "red");
  assert(ErrorOf(Replace(channel, "\"red\"", "\"hue\"")).message.find(
             "channel") != std::string::npos);
  // The IMAGE output of LoadImage is not a mask.
  assert(ErrorOf(Replace(text, R"("mask": ["m1", 1])", R"("mask": ["m1", 0])"))
             .message.find("MASK output") != std::string::npos);
  // SetLatentNoiseMask needs an encoded image.
  assert(ErrorOf(Replace(text, R"("samples": ["c2", 0])",
                         R"("samples": ["5", 2])"))
             .message.find("VAEEncode") != std::string::npos);
  // VAEEncodeForInpaint maps to the same masked edit, with a warning.
  std::string inpaint = Replace(
      kQwenCanvas,
      R"("c2": {"class_type": "VAEEncode", "inputs": {"pixels": ["c1", 0], "vae": ["2", 0]}})",
      R"("c2": {"class_type": "VAEEncodeForInpaint", "inputs": {"pixels": ["c1", 0], "vae": ["2", 0], "mask": ["c1", 1], "grow_mask_by": 6}})");
  plan = PlanOf(inpaint.c_str());
  image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert((image.init_image == comfy::FileRef{"leinwand.png", "input"}));
  assert((image.mask == comfy::FileRef{"leinwand.png", "input"}));
  assert(image.mask_channel == "alpha");
  bool warned = false;
  for (const auto& warning : plan.warnings)
    warned |= warning.find("VAEEncodeForInpaint") != std::string::npos;
  assert(warned);
  // Without a mask nothing changes: no mask field is sent.
  plan = PlanOf(kQwenCanvas);
  image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert(image.mask.name.empty() && image.init_image.name.empty());
  assert(comfy::ImageExtension(image).find("mask_channel") == nullptr);
}

void TestQwenLoras() {
  // LoraLoader -> LoraLoaderModelOnly (strength 0) -> LoraLoaderModelOnly ->
  // KSampler: "loras" in the order ComfyUI applies them, loader side first.
  std::string text = Replace(
      kQwenText, R"("model": ["4", 0], "positive")",
      R"("model": ["L3", 0], "positive")");
  text = Replace(text, R"("clip": ["1", 0], "prompt")",
                 R"("clip": ["L1", 1], "prompt")");
  text = Replace(
      text, R"( "5": {)",
      R"( "L1": {"class_type": "LoraLoader", "inputs": {"model": ["4", 0], "clip": ["1", 0], "lora_name": "figuren\\Anime_Consistency.safetensors", "strength_model": 0.8, "strength_clip": 1.0}},
 "L2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["L1", 0], "lora_name": "off.safetensors", "strength_model": 0.0}},
 "L3": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["L2", 0], "lora_name": "styles/detail.sft", "strength_model": -0.5}},
 "5": {)");
  const comfy::Plan plan = PlanOf(text.c_str());
  const auto& image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert(image.loras.size() == 2);
  assert(image.loras[0].first == "Anime_Consistency" &&
         image.loras[0].second == 0.8);
  assert(image.loras[1].first == "detail" && image.loras[1].second == -0.5);
  const json::Value loras = comfy::ImageLoras(image);
  assert(loras.is_array() && loras.size() == 2);
  assert(loras.items()[0].member_str("name") == "Anime_Consistency");
  assert(loras.items()[1].find("scale")->as_double() == -0.5);
  // The CLIP half of LoraLoader has no gufo counterpart: one warning.
  int clip_warnings = 0;
  for (const auto& warning : plan.warnings)
    clip_warnings += warning.find("strength_clip") != std::string::npos;
  assert(clip_warnings == 1);
  // The extension object stays free of LoRA fields.
  assert(comfy::ImageExtension(image).find("loras") == nullptr);

  // Without LoRA nodes nothing is sent, so the worker's defaults apply.
  const comfy::Plan plain = PlanOf(kQwenText);
  assert(comfy::ImageLoras(std::get<comfy::ImageTask>(plain.tasks[0].spec))
             .is_null());

  // The same file twice adds up; strength_clip 0 does not warn.
  std::string twice = Replace(
      kQwenText, R"("model": ["4", 0], "positive")",
      R"("model": ["L2", 0], "positive")");
  twice = Replace(
      twice, R"( "5": {)",
      R"( "L1": {"class_type": "LoraLoader", "inputs": {"model": ["4", 0], "clip": ["1", 0], "lora_name": "a.safetensors", "strength_model": 0.25, "strength_clip": 0.0}},
 "L2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["L1", 0], "lora_name": "sub/a.safetensors", "strength_model": 0.5}},
 "5": {)");
  const comfy::Plan summed = PlanOf(twice.c_str());
  const auto& merged = std::get<comfy::ImageTask>(summed.tasks[0].spec);
  assert(merged.loras.size() == 1 && merged.loras[0].first == "a" &&
         merged.loras[0].second == 0.75);
  assert(summed.warnings.empty());

  // Out-of-range strength fails at the node.
  const comfy::PlanError error = ErrorOf(Replace(
      text, R"("strength_model": 0.8)", R"("strength_model": 12)"));
  assert(error.node_errors.find("L1") != nullptr);

  assert(comfy::LoraName("anime.safetensors") == "anime");
  assert(comfy::LoraName("A/B\\c.SafeTensors") == "c");
  assert(comfy::LoraName("plain") == "plain");
  assert(comfy::LoraName("x.ckpt") == "x.ckpt");
}

void TestQwenTurbo() {
  // A UNETLoader naming a Turbo checkpoint goes to the Turbo model, not to
  // qwen-image-2.1; everything else in the graph maps as before.
  std::string turbo = Replace(kQwenText, "qwen_image_2.1_int8_convrot",
                              "qwen_image_2.1_turbo_bf16");
  turbo = Replace(turbo, "\"steps\": 25, \"cfg\": 2.5",
                  "\"steps\": 8, \"cfg\": 1.0");
  const comfy::Plan plan = PlanOf(turbo.c_str());
  const auto& image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert(image.model == "qwen-image-2.1-turbo");
  assert(image.steps == 8);
  assert(image.prompt == "A single mature adult male office worker");
  assert(plan.warnings.empty());
  assert(comfy::IsQwenImageTurboFile("Qwen_Image_2.1_TURBO.safetensors"));
  assert(!comfy::IsQwenImageTurboFile("qwen_image_2.1_int8_convrot.safetensors"));

  const auto plan_with = [](const std::string& text,
                            const comfy::ModelIds& models, comfy::Plan* plan,
                            comfy::PlanError* error) {
    return comfy::BuildPlan(json::parse(text), models, plan, error);
  };
  // The router serves the Turbo model: same mapping.
  comfy::ModelIds served;
  served.available = {"qwen-image-2.1", "qwen-image-2.1-turbo"};
  {
    comfy::Plan p;
    comfy::PlanError e;
    assert(plan_with(turbo, served, &p, &e));
    assert(std::get<comfy::ImageTask>(p.tasks[0].spec).model ==
           "qwen-image-2.1-turbo");
  }
  // Without it: a validation error at the loader, never a silent fallback.
  comfy::ModelIds base_only;
  base_only.available = {"qwen-image-2.1"};
  {
    comfy::Plan p;
    comfy::PlanError e;
    assert(!plan_with(turbo, base_only, &p, &e));
    assert(p.tasks.empty());
    assert(e.type == "prompt_outputs_failed_validation");
    const json::Value* node = e.node_errors.find("4");
    assert(node != nullptr);
    assert(node->member_str("class_type") == "UNETLoader");
    assert(node->find("errors")->items()[0].member_str("type") ==
           "value_not_in_list");
    assert(e.message.find("qwen-image-2.1-turbo") != std::string::npos);
    // The plain graph still runs on qwen-image-2.1 with that model list.
    comfy::Plan plain;
    assert(plan_with(kQwenText, base_only, &plain, &e));
    assert(std::get<comfy::ImageTask>(plain.tasks[0].spec).model ==
           "qwen-image-2.1");
  }
  // No Turbo model id configured: also an error at the loader.
  comfy::ModelIds no_turbo;
  no_turbo.image_turbo.clear();
  {
    comfy::Plan p;
    comfy::PlanError e;
    assert(!plan_with(turbo, no_turbo, &p, &e));
    assert(e.node_errors.find("4") != nullptr);
  }
  // Turbo runs exactly 8 steps: other counts fail at the sampler.
  const comfy::PlanError steps = ErrorOf(
      Replace(turbo, "\"steps\": 8", "\"steps\": 20"));
  assert(steps.node_errors.find("6") != nullptr);
  assert(steps.message.find("exactly 8 steps, not 20") != std::string::npos);
  // No partial-strength image to image on Turbo.
  std::string canvas = Replace(kQwenCanvas, "qwen_image_2.1_int8_convrot",
                               "qwen_image_2.1_turbo_bf16");
  canvas = Replace(canvas, "\"steps\": 25", "\"steps\": 8");
  assert(std::get<comfy::ImageTask>(PlanOf(canvas.c_str()).tasks[0].spec)
             .model == "qwen-image-2.1-turbo");
  const comfy::PlanError denoise =
      ErrorOf(Replace(canvas, "\"denoise\": 1.0", "\"denoise\": 0.6"));
  assert(denoise.node_errors.find("6") != nullptr);
  assert(denoise.message.find("denoise") != std::string::npos);
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
  // ComfyUI "Auto" stays automatic; dropping it would make gufo speak English.
  assert(dialogue.lines[0].language == "auto");
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

void TestInfiniteTalk() {
  const comfy::Plan plan = PlanOf(kTalk);
  assert(plan.tasks.size() == 1);
  assert(plan.tasks[0].node_id == "30");
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kTalk);
  assert(video.model == "wan2.1-infinitetalk-14b");
  assert(video.prompt == "a woman talks calmly to the camera");
  assert(video.width == 832 && video.height == 480 && video.frames == 81);
  assert(video.fps == 25 && video.steps == 4 && video.seed == 7);
  assert((video.first_frame == comfy::FileRef{"anna_portrait.png", "input"}));
  assert(video.talk_audio.size() == 1);
  assert((video.talk_audio[0] == comfy::FileRef{"anna_satz1.wav", "input"}));
  assert(video.talk_masks.empty());
  assert(plan.warnings.empty());
  const json::Value body = comfy::VideoRequestBody(video);
  assert(body.member_str("model") == "wan2.1-infinitetalk-14b");
  assert(body.member_str("size") == "832x480");
  assert(std::abs(body.find("seconds")->as_double() - 3.24) < 1e-9);
  const json::Value* gufo = body.find("gufo");
  assert(gufo->find("frames") == nullptr);
  assert(gufo->find("steps")->as_double() == 4);
  const json::Value* talk = gufo->find("talk");
  assert(talk->find("audio_scale")->as_double() == 1.0);
  assert(talk->find("motion_frames")->as_double() == 9);
  assert(talk->member_str("audio_mode") == "add");

  // Two speakers: mask_1 from LoadImageMask (alpha), mask_2 from LoadImage's
  // MASK output.
  std::string two = Replace(
      kTalk, "\"mode\": \"single_speaker\",",
      "\"mode\": \"two_speakers\", \"mode.audio_encoder_output_2\": "
      "[\"41\", 0], \"mode.mask_1\": [\"42\", 0], \"mode.mask_2\": "
      "[\"43\", 1],");
  two = Replace(
      two, " \"10\": {",
      " \"40\": {\"class_type\": \"LoadAudio\", \"inputs\": {\"audio\": "
      "\"ben_satz1.wav\"}},\n \"41\": {\"class_type\": "
      "\"AudioEncoderEncode\", \"inputs\": {\"audio_encoder\": [\"7\", 0], "
      "\"audio\": [\"40\", 0]}},\n \"42\": {\"class_type\": "
      "\"LoadImageMask\", \"inputs\": {\"image\": \"maske_anna.png\", "
      "\"channel\": \"alpha\"}},\n \"43\": {\"class_type\": \"LoadImage\", "
      "\"inputs\": {\"image\": \"maske_ben.png\"}},\n \"10\": {");
  const comfy::Plan two_plan = PlanOf(two.c_str());
  const auto& pair = std::get<comfy::VideoTask>(two_plan.tasks[0].spec);
  assert(pair.talk_audio.size() == 2);
  assert((pair.talk_audio[1] == comfy::FileRef{"ben_satz1.wav", "input"}));
  assert(pair.talk_masks.size() == 2);
  assert((pair.talk_masks[0] == comfy::FileRef{"maske_anna.png", "input"}));
  assert((pair.talk_masks[1] == comfy::FileRef{"maske_ben.png", "input"}));
  assert(pair.talk_mask_channels[0] == "inverted_alpha");
  assert(pair.talk_mask_channels[1] == "inverted_alpha");
  // Two speakers without masks, or from an unsupported node, fail.
  assert(ErrorOf(Replace(two, ", \"mode.mask_2\": [\"43\", 1]", ""))
             .message.find("mask_1 and mask_2") != std::string::npos);
  assert(ErrorOf(Replace(two, "\"mode.mask_2\": [\"43\", 1]",
                         "\"mode.mask_2\": [\"43\", 0]"))
             .message.find("LoadImageMask") != std::string::npos);

  // A second window continuing from the first (previous_frames) runs as one
  // job: 81 + (81 - 9) frames.
  std::string chained = Replace(
      kTalk, " \"33\": {",
      " \"50\": {\"class_type\": \"WanInfiniteTalkToVideo\", \"inputs\": "
      "{\"mode\": \"single_speaker\", \"model\": [\"2\", 0], "
      "\"model_patch\": [\"6\", 0], \"positive\": [\"12\", 0], "
      "\"negative\": [\"13\", 0], \"vae\": [\"4\", 0], \"width\": 832, "
      "\"height\": 480, \"length\": 81, \"clip_vision_output\": [\"11\", 0], "
      "\"start_image\": [\"10\", 0], \"audio_encoder_output_1\": [\"9\", 0], "
      "\"motion_frame_count\": 9, \"audio_scale\": 1.0, \"previous_frames\": "
      "[\"31\", 0]}},\n \"51\": {\"class_type\": \"KSampler\", \"inputs\": "
      "{\"model\": [\"50\", 0], \"positive\": [\"50\", 1], \"negative\": "
      "[\"50\", 2], \"latent_image\": [\"50\", 3], \"seed\": 7, \"steps\": 4, "
      "\"cfg\": 1.0, \"sampler_name\": \"lcm\", \"scheduler\": \"simple\", "
      "\"denoise\": 1.0}},\n \"52\": {\"class_type\": \"VAEDecode\", "
      "\"inputs\": {\"samples\": [\"51\", 0], \"vae\": [\"4\", 0]}},\n "
      "\"53\": {\"class_type\": \"ImageFromBatch\", \"inputs\": {\"image\": "
      "[\"52\", 0], \"batch_index\": [\"50\", 4], \"length\": 4096}},\n "
      "\"54\": {\"class_type\": \"ImageBatch\", \"inputs\": {\"image1\": "
      "[\"31\", 0], \"image2\": [\"53\", 0]}},\n \"33\": {");
  chained = Replace(chained, "\"images\": [\"31\", 0], \"audio\"",
                    "\"images\": [\"54\", 0], \"audio\"");
  const comfy::Plan chain_plan = PlanOf(chained.c_str());
  assert(chain_plan.tasks.size() == 1);
  assert(chain_plan.tasks[0].node_id == "51");
  const auto& longer = std::get<comfy::VideoTask>(chain_plan.tasks[0].spec);
  assert(longer.frames == 153);
  assert(chain_plan.warnings.size() == 1 &&
         chain_plan.warnings[0].find("one gufo job") != std::string::npos);
  // Windows with different audio cannot be one job.
  std::string other = Replace(
      chained, "\"previous_frames\": [\"31\", 0]}}",
      "\"previous_frames\": [\"31\", 0]}},\n \"55\": {\"class_type\": "
      "\"LoadAudio\", \"inputs\": {\"audio\": \"other.wav\"}},\n \"56\": "
      "{\"class_type\": \"AudioEncoderEncode\", \"inputs\": "
      "{\"audio_encoder\": [\"7\", 0], \"audio\": [\"55\", 0]}}");
  other = Replace(other,
                  "\"audio_encoder_output_1\": [\"9\", 0], "
                  "\"motion_frame_count\": 9, \"audio_scale\": 1.0, "
                  "\"previous_frames\"",
                  "\"audio_encoder_output_1\": [\"56\", 0], "
                  "\"motion_frame_count\": 9, \"audio_scale\": 1.0, "
                  "\"previous_frames\"");
  assert(ErrorOf(other).message.find("same audio") != std::string::npos);

  // Errors: no start image, too large, foreign LoRA, audio not from the
  // encoder.
  assert(ErrorOf(Replace(kTalk, ", \"start_image\": [\"10\", 0]", ""))
             .message.find("start_image") != std::string::npos);
  assert(ErrorOf(Replace(kTalk, "\"height\": 480", "\"height\": 832"))
             .message.find("832x832") != std::string::npos);
  assert(ErrorOf(Replace(kTalk, "lightx2v_I2V_14B_480p_cfg_step_distill",
                         "anime_style"))
             .message.find("LoRA") != std::string::npos);
  assert(ErrorOf(Replace(kTalk, "\"audio_encoder_output_1\": [\"9\", 0]",
                         "\"audio_encoder_output_1\": [\"8\", 0]"))
             .message.find("AudioEncoderEncode") != std::string::npos);
  // motion_frame_count snaps to 4k+1 with a warning.
  const comfy::Plan snapped = PlanOf(
      Replace(kTalk, "\"motion_frame_count\": 9", "\"motion_frame_count\": 10")
          .c_str());
  assert(std::get<comfy::VideoTask>(snapped.tasks[0].spec).motion_frames ==
         9);
  assert(snapped.warnings.size() == 1);
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

// FilmStudio delivery after Wan 2.2: RIFE x2 (ComfyUI-Frame-Interpolation),
// Real-ESRGAN x4 + ImageScale to 1920x1080, VideoHelperSuite at 60 fps.
constexpr const char* kWanDelivery = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_ti2v_5B_fp16.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
 "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
 "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "a fox in the snow"}},
 "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
 "55": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"vae": ["3", 0], "width": 1280, "height": 704, "length": 49, "batch_size": 1}},
 "3s": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["55", 0], "seed": 42, "steps": 20, "cfg": 5, "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1}},
 "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3s", 0], "vae": ["3", 0]}},
 "60": {"class_type": "RIFE VFI", "inputs": {"ckpt_name": "rife49.pth", "frames": ["8", 0], "clear_cache_after_n_frames": 10, "multiplier": 2, "fast_mode": true, "ensemble": true, "scale_factor": 1.0}},
 "61": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "realesr-animevideov3.pth"}},
 "62": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["61", 0], "image": ["60", 0]}},
 "63": {"class_type": "ImageScale", "inputs": {"image": ["62", 0], "upscale_method": "lanczos", "width": 1920, "height": 1080, "crop": "center"}},
 "64": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["63", 0], "frame_rate": 60, "loop_count": 0, "filename_prefix": "film/lieferung/shot01", "format": "video/h264-mp4", "pingpong": false, "save_output": true}}
})";

void TestDelivery() {
  const comfy::Plan plan = PlanOf(kWanDelivery);
  const auto& wan = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(wan.interpolate == 2 && wan.fps_num == 60 && wan.fps_den == 1);
  assert(wan.fps == 60);
  assert(wan.upscale && wan.upscale->width == 1920 &&
         wan.upscale->height == 1080);
  assert(wan.upscale->fit == "crop" &&
         wan.upscale->model == "realesr-animevideov3");
  const auto& out = plan.tasks[0].outputs[0];
  assert(out.class_type == "VHS_VideoCombine" &&
         out.kind == comfy::OutputKind::kVideo && !out.temp);
  assert(out.prefix == "film/lieferung/shot01");
  const json::Value body = comfy::VideoRequestBody(wan);
  const json::Value* gufo = body.find("gufo");
  assert(gufo->find("interpolate")->find("fps")->as_double() == 60);
  const json::Value* upscale = gufo->find("upscale");
  assert(upscale->find("width")->as_double() == 1920);
  assert(upscale->find("height")->as_double() == 1080);
  assert(upscale->member_str("fit") == "crop");
  assert(upscale->member_str("model") == "realesr-animevideov3");
  bool speed_note = false;
  for (const auto& warning : plan.warnings)
    speed_note = speed_note || warning.find("resamples") != std::string::npos;
  assert(speed_note);
  // 59.94 is NTSC 60000/1001.
  const comfy::Plan ntsc = PlanOf(
      Replace(kWanDelivery, "\"frame_rate\": 60", "\"frame_rate\": 59.94")
          .c_str());
  const auto& ntsc_video = std::get<comfy::VideoTask>(ntsc.tasks[0].spec);
  assert(ntsc_video.fps_num == 60000 && ntsc_video.fps_den == 1001);
  assert(comfy::VideoRequestBody(ntsc_video)
             .find("gufo")
             ->find("interpolate")
             ->member_str("fps") == "60000/1001");
  // frame_rate = 24 * multiplier keeps the plain multiplier.
  const comfy::Plan plain = PlanOf(
      Replace(kWanDelivery, "\"frame_rate\": 60", "\"frame_rate\": 48")
          .c_str());
  const auto& doubled = std::get<comfy::VideoTask>(plain.tasks[0].spec);
  assert(doubled.fps_num == 0 && doubled.interpolate == 2);
  assert(comfy::VideoRequestBody(doubled)
             .find("gufo")
             ->find("interpolate")
             ->as_double() == 2);
  // Without ImageScale the x4 size, stretched; ImageScaleBy scales it; a
  // zero side keeps the aspect ratio.
  std::string bare = Replace(
      kWanDelivery,
      "\"63\": {\"class_type\": \"ImageScale\", \"inputs\": {\"image\": "
      "[\"62\", 0], \"upscale_method\": \"lanczos\", \"width\": 1920, "
      "\"height\": 1080, \"crop\": \"center\"}},\n",
      "");
  bare = Replace(bare, "\"images\": [\"63\", 0]", "\"images\": [\"62\", 0]");
  const comfy::Plan x4 = PlanOf(bare.c_str());
  const auto& big = std::get<comfy::VideoTask>(x4.tasks[0].spec);
  assert(big.upscale->width == 5120 && big.upscale->height == 2816 &&
         big.upscale->fit == "stretch");
  const comfy::Plan by = PlanOf(
      Replace(kWanDelivery,
              "\"ImageScale\", \"inputs\": {\"image\": [\"62\", 0], "
              "\"upscale_method\": \"lanczos\", \"width\": 1920, \"height\": "
              "1080, \"crop\": \"center\"}",
              "\"ImageScaleBy\", \"inputs\": {\"image\": [\"62\", 0], "
              "\"upscale_method\": \"lanczos\", \"scale_by\": 0.375}")
          .c_str());
  const auto& scaled = std::get<comfy::VideoTask>(by.tasks[0].spec);
  assert(scaled.upscale->width == 1920 && scaled.upscale->height == 1056);
  const comfy::Plan aspect = PlanOf(
      Replace(kWanDelivery, "\"height\": 1080, \"crop\": \"center\"",
              "\"height\": 0, \"crop\": \"disabled\"")
          .c_str());
  const auto& kept = std::get<comfy::VideoTask>(aspect.tasks[0].spec);
  assert(kept.upscale->width == 1920 && kept.upscale->height == 1056 &&
         kept.upscale->fit == "stretch");
  // general-x4v3 is served; AnimeSharp (non-commercial) is not.
  const comfy::Plan general = PlanOf(
      Replace(kWanDelivery, "realesr-animevideov3.pth",
              "realesr-general-x4v3.pth")
          .c_str());
  assert(std::get<comfy::VideoTask>(general.tasks[0].spec).upscale->model ==
         "realesr-general-x4v3");
  const comfy::PlanError sharp = ErrorOf(
      Replace(kWanDelivery, "realesr-animevideov3.pth",
              "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors"));
  assert(sharp.message.find("AnimeSharp") != std::string::npos);
  assert(sharp.message.find("not served by gufo") != std::string::npos);
  assert(sharp.node_errors.find("61") != nullptr);
  // A frame rate without RIFE is only a warning; ImageScale without an
  // upscaler and an upscale of a still image are errors.
  std::string no_rife =
      Replace(kWanDelivery, "\"image\": [\"60\", 0]", "\"image\": [\"8\", 0]");
  no_rife = Replace(
      no_rife,
      " \"60\": {\"class_type\": \"RIFE VFI\", \"inputs\": {\"ckpt_name\": "
      "\"rife49.pth\", \"frames\": [\"8\", 0], "
      "\"clear_cache_after_n_frames\": 10, \"multiplier\": 2, \"fast_mode\": "
      "true, \"ensemble\": true, \"scale_factor\": 1.0}},\n",
      "");
  const comfy::Plan unsampled = PlanOf(no_rife.c_str());
  const auto& as_is = std::get<comfy::VideoTask>(unsampled.tasks[0].spec);
  assert(as_is.fps_num == 0 && as_is.interpolate == 1 && as_is.upscale);
  assert(!unsampled.warnings.empty());
  assert(ErrorOf(Replace(kWanDelivery, "\"image\": [\"62\", 0]",
                         "\"image\": [\"60\", 0]"))
             .message.find("ImageUpscaleWithModel") != std::string::npos);
  assert(ErrorOf(Replace(kQwenText, "\"images\": [\"7\", 0]",
                         "\"images\": [\"9\", 0]}}, \"9\": {\"class_type\": "
                         "\"ImageUpscaleWithModel\", \"inputs\": "
                         "{\"upscale_model\": [\"10\", 0], \"image\": [\"7\", "
                         "0]}}, \"10\": {\"class_type\": "
                         "\"UpscaleModelLoader\", \"inputs\": {\"model_name\": "
                         "\"realesr-animevideov3.pth\""))
             .message.find("video") != std::string::npos);
  // H3: RIFE x4 + CreateVideo at 96 fps stays the multiplier; the camera
  // family upscales its 832x480 x4 picture with ImageScaleBy 0.5.
  const std::string h3_rife = Replace(
      kH3,
      "\"40\": {\"class_type\": \"SaveImage\", \"inputs\": {\"images\": "
      "[\"32\", 0]",
      "\"39\": {\"class_type\": \"RIFE VFI\", \"inputs\": {\"frames\": "
      "[\"32\", 0], \"multiplier\": 4}},\n \"38\": {\"class_type\": "
      "\"CreateVideo\", \"inputs\": {\"images\": [\"39\", 0], \"fps\": 96}},\n "
      "\"40\": {\"class_type\": \"SaveVideo\", \"inputs\": {\"video\": "
      "[\"38\", 0]");
  const comfy::Plan h3 = PlanOf(h3_rife.c_str());
  const auto& h3_video = std::get<comfy::VideoTask>(h3.tasks[0].spec);
  assert(h3_video.interpolate == 4 && h3_video.fps_num == 0 &&
         !h3_video.upscale);
  const comfy::Plan h3_60 =
      PlanOf(Replace(h3_rife, "\"fps\": 96", "\"fps\": 60").c_str());
  assert(std::get<comfy::VideoTask>(h3_60.tasks[0].spec).fps_num == 60);
  const std::string camera = Replace(
      kUni3C,
      "\"32\": {\"class_type\": \"CreateVideo\", \"inputs\": {\"images\": "
      "[\"31\", 0]",
      "\"34\": {\"class_type\": \"UpscaleModelLoader\", \"inputs\": "
      "{\"model_name\": \"realesr-general-x4v3.pth\"}},\n \"35\": "
      "{\"class_type\": \"ImageUpscaleWithModel\", \"inputs\": "
      "{\"upscale_model\": [\"34\", 0], \"image\": [\"31\", 0]}},\n \"36\": "
      "{\"class_type\": \"ImageScaleBy\", \"inputs\": {\"image\": [\"35\", "
      "0], \"upscale_method\": \"lanczos\", \"scale_by\": 0.5}},\n \"32\": "
      "{\"class_type\": \"CreateVideo\", \"inputs\": {\"images\": [\"36\", "
      "0]");
  const comfy::Plan cam = PlanOf(camera.c_str());
  const auto& cam_video = std::get<comfy::VideoTask>(cam.tasks[0].spec);
  assert(cam_video.upscale->width == 1664 && cam_video.upscale->height == 960);
  assert(cam_video.upscale->model == "realesr-general-x4v3");
  assert(comfy::VideoRequestBody(cam_video)
             .find("gufo")
             ->find("upscale")
             ->find("width")
             ->as_double() == 1664);
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

// ComfyUI /ws preview frames: big-endian u32 type, then format or metadata
// length, then the bytes (server.py BinaryEventTypes 1 and 4).
void TestPreviewFrames() {
  const std::string jpeg("\xFF\xD8\xFF\xE0jpeg", 8);
  const std::string one = comfy::PreviewImageFrame(jpeg);
  assert(one.size() == 8 + jpeg.size());
  assert(one.substr(0, 8) == std::string("\0\0\0\x01\0\0\0\x01", 8));
  assert(one.substr(8) == jpeg);
  const std::string png = comfy::PreviewImageFrame("x", comfy::kPreviewPng);
  assert(png.substr(4, 4) == std::string("\0\0\0\x02", 4));
  const std::string meta = R"({"node_id":"6","prompt_id":"p"})";
  const std::string four = comfy::PreviewMetadataFrame(jpeg, meta);
  assert(four.substr(0, 4) == std::string("\0\0\0\x04", 4));
  const auto length = (static_cast<unsigned char>(four[4]) << 24) |
                      (static_cast<unsigned char>(four[5]) << 16) |
                      (static_cast<unsigned char>(four[6]) << 8) |
                      static_cast<unsigned char>(four[7]);
  assert(length == meta.size());
  assert(four.substr(8, meta.size()) == meta);
  assert(four.substr(8 + meta.size()) == jpeg);
  // Lengths above 255 use the higher bytes.
  const std::string long_meta(300, 'm');
  const std::string big = comfy::PreviewMetadataFrame("", long_meta);
  assert(big.substr(4, 4) == std::string("\0\0\x01\x2C", 4));
  assert(comfy::PreviewId("a1b2-c3", "12:4") == "a1b2-c3_12_4");
  assert(comfy::PreviewId(std::string(200, 'x'), "1").size() == 128);
}

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
  TestMaskedEdit();
  TestQwenLoras();
  TestQwenTurbo();
  TestRife();
  TestDialogue();
  TestAceStep();
  TestUni3C();
  TestWanFunCamera();
  TestInfiniteTalk();
  TestDelivery();
  TestWavJoin();
  TestHelpers();
  TestPreviewFrames();
  std::cout << "router_comfy_graph_test: all passed\n";
  return 0;
}
