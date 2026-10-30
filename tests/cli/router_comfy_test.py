"""Contract tests for the router's ComfyUI-compatible API (--comfy-dir).

Run through ctest as router_comfy_contract_test with the gufo binary as
argv[1]. The router starts external fake workers (this file with
--fake-worker) for the image, TTS and video model ids; they answer the gufo
endpoints with small generated PNG/WAV/MP4 bytes and log every request so
the tests can check how the ComfyUI workflows were mapped.
"""

import base64
import hashlib
import http.client
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import zlib
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

THIS = os.path.abspath(__file__)
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
SEP = "\\" if os.name == "nt" else "/"


def make_png(width, height, shade=0):
    row = b"\x00" + bytes([shade % 256, 80, 160]) * width
    raw = row * height

    def chunk(kind, data):
        body = kind + data
        return (struct.pack(">I", len(data)) + body +
                struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


def png_size(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n", data[:16]
    return struct.unpack(">II", data[16:24])


def make_wav(seconds=0.25, rate=24000):
    samples = int(seconds * rate)
    pcm = b"".join(struct.pack("<h", (i * 97) % 2000 - 1000) for i in range(samples))
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) +
            b"data" + struct.pack("<I", len(pcm)) + pcm)


# ---------------------------------------------------------------- fake worker

def serve_fake_worker(argv):
    port = int(argv[argv.index("--port") + 1])
    model = argv[argv.index("--served-model-name") + 1]
    log_path = os.environ["COMFY_FAKE_LOG"]
    # One-shot start failure (a load that dies with "bad allocation"): while
    # <dir>/<model>.fail exists, remove it and exit with status 1.
    fail_dir = os.environ.get("COMFY_FAKE_FAIL_DIR")
    if fail_dir:
        marker = os.path.join(fail_dir, model + ".fail")
        if os.path.exists(marker):
            os.remove(marker)
            time.sleep(0.3)
            sys.exit(1)
    lock = threading.Lock()
    videos = {}

    def log(entry):
        entry["model"] = model
        with lock, open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def send(self, status, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def body(self):
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                return self.send(200, {"status": "ok"})
            if path.startswith("/v1/videos/"):
                rest = path[len("/v1/videos/"):]
                if rest.endswith("/content"):
                    return self.send(200, videos[rest[:-8]], "video/mp4")
                return self.send(200, {"id": rest, "object": "video",
                                       "status": "completed", "progress": 100})
            self.send(404, {"error": {"message": "no route"}})

        def do_DELETE(self):
            self.send(200, {"deleted": True})

        def image_reply(self, fields, size, count):
            if "FAIL" in fields.get("prompt", ""):
                return self.send(500, {"error": {"message": "fake failure"}})
            if "OLDGUFO" in fields.get("prompt", "") and "gufo" in fields:
                return self.send(400, {"error": {
                    "message": "unsupported image parameter: gufo"}})
            width, height = (int(v) for v in size.split("x"))
            data = [{"b64_json": base64.b64encode(make_png(width, height, i)).decode()}
                    for i in range(count)]
            self.send(200, {"created": 1, "data": data})

        def do_POST(self):
            path = self.path.split("?")[0]
            raw = self.body()
            if path == "/v1/images/generations":
                fields = json.loads(raw)
                log({"path": path, "fields": fields})
                return self.image_reply(fields, fields["size"], fields.get("n", 1))
            if path == "/v1/images/edits":
                message = BytesParser(policy=HTTP).parsebytes(
                    b"Content-Type: " + self.headers["Content-Type"].encode() +
                    b"\r\n\r\n" + raw)
                fields, images, init = {}, [], None
                for part in message.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    data = part.get_payload(decode=True)
                    if name == "image[]":
                        images.append(hashlib.sha256(data).hexdigest())
                    elif name in ("image", "init_image"):
                        init = name + ":" + hashlib.sha256(data).hexdigest()
                    else:
                        fields[name] = data.decode()
                if "gufo" in fields:
                    fields["gufo"] = json.loads(fields["gufo"])
                log({"path": path, "fields": fields, "images": images, "init": init})
                return self.image_reply(fields, fields["size"], int(fields.get("n", 1)))
            if path == "/v1/audio/speech":
                fields = json.loads(raw)
                if "reference_audio" in fields:
                    fields["reference_audio"] = hashlib.sha256(
                        base64.b64decode(fields["reference_audio"])).hexdigest()
                log({"path": path, "fields": fields})
                return self.send(200, make_wav(), "audio/wav")
            if path == "/v1/audio/music":
                fields = json.loads(raw)
                log({"path": path, "fields": fields})
                return self.send(200, make_wav(0.5, 48000), "audio/wav")
            if path == "/v1/videos":
                fields = json.loads(raw)
                def describe(url):
                    # Images by pixel size, audio by hash.
                    prefix, _, data = url.partition(",")
                    raw = base64.b64decode(data)
                    if raw[:8] == b"\x89PNG\r\n\x1a\n":
                        return prefix + ",%dx%d" % png_size(raw)
                    return prefix + "," + hashlib.sha256(raw).hexdigest()
                for key in ("first_frame", "last_frame", "audio_reference"):
                    if key in fields:
                        fields[key] = describe(fields[key])
                for keyframe in fields.get("gufo", {}).get("keyframes", []):
                    keyframe["image"] = describe(keyframe["image"])
                camera = fields.get("gufo", {}).get("camera", {})
                if "render_video" in camera:
                    camera["render_video"] = describe(camera["render_video"])
                log({"path": path, "fields": fields})
                # Unique across fake workers, like gufo's random job ids: the
                # router routes status/content reads by job id.
                video_id = "video_%s_%d" % (model.replace(".", "_"), len(videos) + 1)
                videos[video_id] = b"FAKEMP4-" + model.encode()
                return self.send(202, {"id": video_id, "object": "video",
                                       "status": "queued", "progress": 0})
            self.send(404, {"error": {"message": "no route"}})

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


# ---------------------------------------------------------------- harness

def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Router:
    def __init__(self, binary, tmp, comfy_dir, ffmpeg=None):
        self.port = free_port()
        self.tmp = tmp
        self.fake_log = os.path.join(tmp, "fake.jsonl")
        preset = os.path.join(tmp, "comfy.preset")
        # An app execution alias (Store Python) is no real file for the
        # router's existence check; COMFY_TEST_PYTHON names another one.
        python = os.environ.get("COMFY_TEST_PYTHON") or sys.executable
        sections = []
        for model in ("qwen-image-2.1", "qwen3-tts", "qwen3-tts-voicedesign",
                      "qwen3-tts-clone", "minimax-h3", "wan2.2-ti2v-5b",
                      "ace-step-1.5-xl-turbo", "wan2.1-uni3c"):
            sections.append("[external/%s]\ncommand = %s\nargs = %s --fake-worker\n"
                            "served-model-name = %s\n" % (model, python, THIS, model))
        with open(preset, "w") as handle:
            handle.write("\n".join(sections))
        env = {k: v for k, v in os.environ.items()
               if k not in {"HOST", "PORT", "GUFO_HOST", "GUFO_PORT"}}
        env["COMFY_FAKE_LOG"] = self.fake_log
        args = [binary, "router", "--models-preset", preset, "--host", "127.0.0.1",
                "--port", str(self.port), "--models-max", "1", "--comfy-dir", comfy_dir]
        if ffmpeg:
            args += ["--comfy-ffmpeg", ffmpeg]
        self.log = open(os.path.join(tmp, "router.log"), "ab")
        self.proc = subprocess.Popen(args, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                if self.request("GET", "/health")[0] == 200:
                    self.ready = True
                    return
            except OSError:
                time.sleep(0.05)
        raise AssertionError("router did not start: " + self.tail())

    def tail(self):
        with open(os.path.join(self.tmp, "router.log"), "rb") as handle:
            return handle.read()[-3000:].decode(errors="replace")

    def stop(self):
        self.proc.terminate()
        self.proc.wait(10)
        self.log.close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
            headers = dict(headers or {}, **{"Content-Type": "application/json"})
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
        except OSError as error:
            if not getattr(self, "ready", False):
                raise  # still starting up
            # Name the router's exit status (e.g. 0xC0000409) next to the error.
            time.sleep(0.5)
            code = self.proc.poll()
            raise AssertionError("%s %s failed: %r; router exit %s; log: %s" % (
                method, path, error, "running" if code is None else hex(code & 0xFFFFFFFF),
                self.tail()))
        payload = response.read()
        conn.close()
        return response.status, payload

    def json(self, method, path, body=None):
        status, payload = self.request(method, path, body)
        return status, (json.loads(payload) if payload else None)

    def upload(self, name, data, overwrite=False, subfolder=""):
        boundary = "testboundary%d" % time.monotonic_ns()
        parts = []
        for field, value in (("overwrite", "true" if overwrite else "false"),
                             ("subfolder", subfolder), ("type", "input")):
            parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                          % (boundary, field, value)).encode())
        parts.append(("--%s\r\nContent-Disposition: form-data; name=\"image\"; "
                      "filename=\"%s\"\r\nContent-Type: application/octet-stream\r\n\r\n"
                      % (boundary, name)).encode() + data + b"\r\n")
        body = b"".join(parts) + ("--%s--\r\n" % boundary).encode()
        status, payload = self.request(
            "POST", "/upload/image", body,
            {"Content-Type": "multipart/form-data; boundary=" + boundary})
        assert status == 200, payload
        return json.loads(payload)

    def fake_entries(self):
        if not os.path.exists(self.fake_log):
            return []
        with open(self.fake_log, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def submit(self, prompt, client_id="tester"):
        status, body = self.json("POST", "/prompt", {"prompt": prompt, "client_id": client_id})
        assert status == 200, body
        assert body["node_errors"] == {}
        return body["prompt_id"]

    def wait(self, prompt_id, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status, body = self.json("GET", "/history/" + prompt_id)
            assert status == 200
            if prompt_id in body:
                return body[prompt_id]
            time.sleep(0.1)
        raise AssertionError("prompt %s did not finish (router exit %s): %s" % (prompt_id, self.proc.poll(), self.tail()))

    def run(self, prompt, **kwargs):
        entry = self.wait(self.submit(prompt, **kwargs))
        assert entry["status"]["status_str"] == "success", entry["status"]
        return entry


# ---------------------------------------------------------------- workflows

def qwen_graph(prompt="an office", refs=(), canvas=None, prefix="film/stil/demo",
               seed=1769224114):
    graph = {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
        "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["1", 0], "prompt": prompt, "negative_prompt": "chibi", "vae": ["2", 0], "resolution": 1024}},
        "6": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["5", 2], "seed": seed, "steps": 25, "cfg": 2.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}},
    }
    for index, name in enumerate(refs, 1):
        graph["r%d" % index] = {"class_type": "LoadImage", "inputs": {"image": name}}
        graph["5"]["inputs"]["images.image_%d" % index] = ["r%d" % index, 0]
    if canvas:
        graph["c1"] = {"class_type": "LoadImage", "inputs": {"image": canvas}}
        graph["c2"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["c1", 0], "vae": ["2", 0]}}
        graph["6"]["inputs"]["latent_image"] = ["c2", 0]
    return graph


def design_graph(text="Ich bin ALIA.", prefix="film/stimmen/alia"):
    return {
        "2": {"class_type": "FB_Qwen3TTSVoiceDesign", "inputs": {"text": text, "instruct": "Frau, 30 Jahre, kalt", "seed": 236224718, "model_choice": "1.7B", "device": "auto", "precision": "bf16", "language": "German", "unload_model_after_generate": True}},
        "3": {"class_type": "SaveAudio", "inputs": {"audio": ["2", 0], "filename_prefix": prefix}},
    }


def clone_graph(reference):
    return {
        "1": {"class_type": "LoadAudio", "inputs": {"audio": reference}},
        "2": {"class_type": "FB_Qwen3TTSVoiceClone", "inputs": {"target_text": "Wir bleiben.", "seed": 5, "ref_audio": ["1", 0], "ref_text": "Hallo und willkommen.", "model_choice": "1.7B", "device": "auto", "precision": "bf16", "language": "German"}},
        "3": {"class_type": "SaveAudio", "inputs": {"audio": ["2", 0], "filename_prefix": "film/stimmen/fabian"}},
    }


def h3_graph(first, last, width=512, height=512, rife=0, middle=None, audio=None):
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "strength_model": 1.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "type": "minimax_h3", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "10": {"class_type": "LoadImage", "inputs": {"image": first}},
        "11": {"class_type": "LoadImage", "inputs": {"image": last}},
        "20": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["3", 0], "vae": ["4", 0], "prompt": "workers at CRT screens", "width": width, "height": height, "length": 124, "first_frame": ["10", 0]}},
        "21": {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": ["20", 0], "latent": ["20", 1], "frame_idx": -1, "vae": ["4", 0], "image": ["11", 0]}},
        "30": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["21", 0], "negative": ["20", 0], "latent_image": ["20", 1], "seed": 7, "steps": 8, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
        "32": {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}},
        "57": {"class_type": "CreateVideo", "inputs": {"images": ["32", 0], "fps": 24}},
        "58": {"class_type": "SaveVideo", "inputs": {"video": ["57", 0], "filename_prefix": "film/clips/shot01", "format": "mp4", "codec": "h264"}},
    }
    if middle or audio:
        positive = ["21", 0]
        if middle:
            graph["12"] = {"class_type": "LoadImage", "inputs": {"image": middle}}
            graph["23"] = {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": positive, "latent": ["20", 1], "frame_idx": 60, "vae": ["4", 0], "image": ["12", 0]}}
            positive = ["23", 0]
        if audio:
            graph["13"] = {"class_type": "LoadAudio", "inputs": {"audio": audio}}
            graph["24"] = {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": positive, "latent": ["20", 1], "frame_idx": 0, "audio": ["13", 0]}}
            positive = ["24", 0]
        graph["30"]["inputs"]["positive"] = positive
    if rife:
        graph["40"] = {"class_type": "RIFE VFI", "inputs": {"ckpt_name": "rife47.pth", "frames": ["32", 0], "clear_cache_after_n_frames": 10, "multiplier": rife, "fast_mode": True, "ensemble": True, "scale_factor": 1.0, "dtype": "float32", "torch_compile": False, "batch_size": 1}}
        graph["57"]["inputs"] = {"images": ["40", 0], "fps": 24 * rife}
    return graph


def ace_graph(prefix="film/musik/titel"):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "acestep_v1.5_xl_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "qwen_0.6b_ace15.safetensors", "clip_name2": "qwen_4b_ace15.safetensors", "type": "ace", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ace_1.5_vae.safetensors"}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3}},
        "5": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {"clip": ["2", 0], "tags": "synth pop, 90s anime", "lyrics": "[verse]\nNeon", "seed": 5, "bpm": 120, "duration": 30, "timesignature": "4", "language": "de", "keyscale": "C major", "generate_audio_codes": False, "cfg_scale": 2.0, "temperature": 0.85, "top_p": 0.9, "top_k": 0, "min_p": 0.0}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {"seconds": 30, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0], "seed": 5, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveAudio", "inputs": {"audio": ["9", 0], "filename_prefix": prefix}},
    }


def camera_graph(start, move=None, render=None, length=81):
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "strength_model": 1.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp16.safetensors", "type": "wan", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "5": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "6": {"class_type": "ModelPatchLoader", "inputs": {"name": "Wan21_Uni3C_controlnet_fp16.safetensors"}},
        "10": {"class_type": "LoadImage", "inputs": {"image": start}},
        "11": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["5", 0], "image": ["10", 0], "crop": "none"}},
        "12": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "slow camera move through the office"}},
        "13": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": ""}},
        "15": {"class_type": "WanUni3CControlnetApply", "inputs": {"model": ["2", 0], "model_patch": ["6", 0], "vae": ["4", 0], "render_video": ["14", 0], "strength": 1.0, "start_percent": 0.0, "end_percent": 1.0}},
        "16": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["15", 0], "shift": 5}},
        "20": {"class_type": "WanImageToVideo", "inputs": {"positive": ["12", 0], "negative": ["13", 0], "vae": ["4", 0], "width": 832, "height": 480, "length": length, "batch_size": 1, "clip_vision_output": ["11", 0], "start_image": ["10", 0]}},
        "30": {"class_type": "KSampler", "inputs": {"model": ["16", 0], "positive": ["20", 0], "negative": ["20", 1], "latent_image": ["20", 2], "seed": 99, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "31": {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}},
        "32": {"class_type": "CreateVideo", "inputs": {"images": ["31", 0], "fps": 16}},
        "33": {"class_type": "SaveVideo", "inputs": {"video": ["32", 0], "filename_prefix": "film/kamera/shot", "format": "mp4", "codec": "h264"}},
    }
    if render:
        graph["14"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["17", 0]}}
        graph["17"] = {"class_type": "LoadVideo", "inputs": {"file": render}}
    else:
        graph["14"] = {"class_type": "GufoCameraMove", "inputs": dict({"camera": "orbit_left", "speed": 1.0, "ease": "linear"}, **(move or {}))}
    return graph


def dialogue_graph():
    return {
        "1": {"class_type": "LoadAudio", "inputs": {"audio": "anna.wav"}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": "ben.wav"}},
        "3": {"class_type": "FB_Qwen3TTSVoiceClonePrompt", "inputs": {"ref_audio": ["1", 0], "ref_text": "Ich bin Anna.", "model_choice": "1.7B", "device": "auto", "precision": "bf16", "attention": "auto"}},
        "4": {"class_type": "FB_Qwen3TTSVoiceClonePrompt", "inputs": {"ref_audio": ["2", 0], "ref_text": "", "model_choice": "1.7B", "device": "auto", "precision": "bf16", "attention": "auto", "x_vector_only": True}},
        "5": {"class_type": "FB_Qwen3TTSRoleBank", "inputs": {"role_name_1": "Anna", "prompt_1": ["3", 0], "role_name_2": "Ben", "prompt_2": ["4", 0]}},
        "6": {"class_type": "FB_Qwen3TTSDialogueInference", "inputs": {"script": "Anna: Guten Morgen.\nBen: Morgen!\nAnna: Wie geht es?", "role_bank": ["5", 0], "model_choice": "1.7B", "device": "auto", "precision": "bf16", "language": "Auto", "pause_linebreak": 0.25, "period_pause": 0, "comma_pause": 0, "question_pause": 0, "hyphen_pause": 0, "merge_outputs": True, "batch_size": 4, "seed": 3}},
        "7": {"class_type": "SaveAudio", "inputs": {"audio": ["6", 0], "filename_prefix": "film/dialog/szene1"}},
    }


# ---------------------------------------------------------------- websocket

class WsClient:
    def __init__(self, port, path):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(("GET %s HTTP/1.1\r\nHost: 127.0.0.1\r\nUpgrade: websocket\r\n"
                           "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
                           "Sec-WebSocket-Version: 13\r\n\r\n" % (path, key)).encode())
        head = b""
        while b"\r\n\r\n" not in head:
            head += self.sock.recv(1)
        assert b" 101 " in head.split(b"\r\n")[0], head
        self.buffer = b""

    def read(self, count):
        while len(self.buffer) < count:
            chunk = self.sock.recv(65536)
            assert chunk, "websocket closed"
            self.buffer += chunk
        out, self.buffer = self.buffer[:count], self.buffer[count:]
        return out

    def message(self):
        first, second = self.read(2)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack(">H", self.read(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", self.read(8))[0]
        payload = self.read(length)
        assert first & 0x0F == 1, first
        return json.loads(payload)

    def close(self):
        self.sock.close()


# ---------------------------------------------------------------- cases

def check_front(router):
    status, stats = router.json("GET", "/system_stats")
    assert status == 200 and stats["system"]["ram_total"] > 0, stats
    status, queue = router.json("GET", "/queue")
    assert queue == {"queue_running": [], "queue_pending": []}, queue
    status, info = router.json("GET", "/prompt")
    assert info == {"exec_info": {"queue_remaining": 0}}, info
    status, objects = router.json("GET", "/object_info")
    assert "TextEncodeQwenImage21" in objects and "SaveImage" in objects
    assert objects["SaveImage"]["output_node"] is True
    status, one = router.json("GET", "/api/object_info/LoadImage")
    assert list(one) == ["LoadImage"], one
    # /v1 routing stays with the router.
    status, models = router.json("GET", "/v1/models")
    assert status == 200 and len(models["data"]) == 8


def check_uploads(router):
    png = make_png(64, 48)
    assert router.upload("ref.png", png) == {"name": "ref.png", "subfolder": "", "type": "input"}
    assert router.upload("ref.png", png)["name"] == "ref.png"  # same bytes: reused
    other = make_png(64, 48, 99)
    assert router.upload("ref.png", other)["name"] == "ref (1).png"
    assert router.upload("ref.png", png, overwrite=True)["name"] == "ref.png"
    assert router.upload("sub/x.png", png, subfolder="chars")["name"] == "x.png"
    status, data = router.request("GET", "/view?filename=x.png&subfolder=chars&type=input")
    assert status == 200 and data == png
    router.upload("leinwand.png", make_png(1344, 832))
    router.upload("voice.wav", make_wav())


def check_qwen(router):
    entry = router.run(qwen_graph())
    image = entry["outputs"]["8"]["images"][0]
    assert image == {"filename": "demo_00001_.png", "subfolder": "film" + SEP + "stil",
                     "type": "output"}, image
    assert entry["prompt"][1] and entry["prompt"][4] == ["8"]
    assert entry["meta"]["8"]["node_id"] == "8"
    query = urllib.parse.urlencode(image)
    status, data = router.request("GET", "/view?" + query)
    assert status == 200 and png_size(data) == (1024, 1024)
    call = router.fake_entries()[-1]
    assert call["path"] == "/v1/images/generations", call
    assert call["fields"] == {"model": "qwen-image-2.1", "prompt": "an office",
                              "size": "1024x1024", "steps": 25, "seed": 1769224114,
                              "n": 1, "response_format": "b64_json",
                              "gufo": {"negative_prompt": "chibi", "guidance": 2.5}}, call
    messages = [m[0] for m in entry["status"]["messages"]]
    assert messages[0] == "execution_start" and messages[-1] == "execution_success"
    assert "gufo_warning" not in messages, messages
    # The counter continues after existing files.
    again = router.run(qwen_graph())
    assert again["outputs"]["8"]["images"][0]["filename"] == "demo_00002_.png"


def check_qwen_references(router):
    router.run(qwen_graph(refs=["ref.png", "ref (1).png"], canvas="leinwand.png",
                          prefix="film/keyframes/kf0"))
    call = router.fake_entries()[-1]
    assert call["path"] == "/v1/images/edits", call
    assert call["fields"]["size"] == "1344x832", call
    assert call["fields"]["steps"] == "25" and call["fields"]["seed"] == "1769224114"
    assert len(call["images"]) == 2 and call["images"][0] != call["images"][1]
    assert call["fields"]["gufo"] == {"negative_prompt": "chibi", "guidance": 2.5}
    assert call["init"] is None
    # denoise < 1: image to image from the canvas with gufo.strength.
    graph = qwen_graph(canvas="leinwand.png", prefix="film/keyframes/i2i")
    graph["6"]["inputs"]["denoise"] = 0.55
    router.run(graph)
    call = router.fake_entries()[-1]
    assert call["path"] == "/v1/images/edits" and call["images"] == [], call
    assert call["init"] == "image:" + hashlib.sha256(make_png(1344, 832)).hexdigest()
    assert call["fields"]["gufo"]["strength"] == 0.55, call
    assert call["fields"]["size"] == "1344x832", call
    # With references the start image is the "init_image" part.
    graph = qwen_graph(refs=["ref.png"], canvas="leinwand.png", prefix="film/keyframes/i2i")
    graph["6"]["inputs"]["denoise"] = 0.7
    router.run(graph)
    call = router.fake_entries()[-1]
    assert len(call["images"]) == 1, call
    assert call["init"] == "init_image:" + hashlib.sha256(make_png(1344, 832)).hexdigest()
    # Canvas sizes round to multiples of 32 (gufo rejects others).
    router.upload("leinwand-odd.png", make_png(1000, 570))
    router.run(qwen_graph(canvas="leinwand-odd.png", prefix="film/keyframes/odd"))
    assert router.fake_entries()[-1]["fields"]["size"] == "992x576"
    # An image worker without the gufo fields: rendered without them + warning.
    entry = router.run(qwen_graph(prompt="OLDGUFO office", prefix="film/stil/old"))
    calls = router.fake_entries()[-2:]
    assert "gufo" in calls[0]["fields"] and "gufo" not in calls[1]["fields"], calls
    warnings = [m[1]["message"] for m in entry["status"]["messages"] if m[0] == "gufo_warning"]
    assert warnings and "rejected" in warnings[0], entry["status"]
    # Without a canvas the latent follows the first reference (64x48 -> ~1024^2).
    router.run(qwen_graph(refs=["ref.png"], prefix="film/keyframes/kf1"))
    assert router.fake_entries()[-1]["fields"]["size"] == "1184x896"


def check_speech(router, audio_ext):
    entry = router.run(design_graph())
    audio = entry["outputs"]["3"]["audio"][0]
    assert audio["filename"] == "alia_00001." + audio_ext, audio
    status, data = router.request("GET", "/view?" + urllib.parse.urlencode(audio))
    assert status == 200 and len(data) > 100
    if audio_ext == "wav":
        assert data[:4] == b"RIFF"
    else:
        assert data[:4] == b"fLaC"
    call = router.fake_entries()[-1]
    assert call["model"] == "qwen3-tts-voicedesign"
    assert call["fields"] == {"model": "qwen3-tts-voicedesign", "input": "Ich bin ALIA.",
                              "voice": "voice-design", "language": "german",
                              "instructions": "Frau, 30 Jahre, kalt", "seed": 236224718,
                              "response_format": "wav"}, call
    router.run(clone_graph("voice.wav"))
    call = router.fake_entries()[-1]
    assert call["fields"]["voice"] == "voice-clone"
    assert call["fields"]["reference_audio"] == hashlib.sha256(make_wav()).hexdigest()
    assert call["fields"]["reference_text"] == "Hallo und willkommen."
    anna, ben = make_wav(0.5), make_wav(0.3)
    router.upload("anna.wav", anna)
    router.upload("ben.wav", ben)
    before = len(router.fake_entries())
    entry = router.run(dialogue_graph())
    calls = router.fake_entries()[before:]
    assert [c["fields"]["input"] for c in calls] == ["Guten Morgen.", "Morgen!", "Wie geht es?"]
    assert calls[0]["fields"]["reference_audio"] == hashlib.sha256(anna).hexdigest()
    assert calls[0]["fields"]["reference_text"] == "Ich bin Anna."
    assert calls[1]["fields"]["reference_audio"] == hashlib.sha256(ben).hexdigest()
    assert calls[1]["fields"]["voice_clone_mode"] == "speaker_embedding_only"
    assert all(c["fields"]["seed"] == 3 and c["model"] == "qwen3-tts-clone" for c in calls)
    files = entry["outputs"]["7"]["audio"]
    assert len(files) == 1 and files[0]["filename"] == "szene1_00001." + audio_ext, files
    status, data = router.request("GET", "/view?" + urllib.parse.urlencode(files[0]))
    assert status == 200
    if audio_ext == "wav":
        # Three 0.25 s fake lines and two 0.25 s pauses at 24 kHz, 16 bit.
        assert len(data) == 44 + 5 * 6000 * 2, len(data)


def check_video(router, stretched):
    entry = router.run(h3_graph("ref.png", "ref (1).png"))
    video = entry["outputs"]["58"]
    assert video["animated"] == [True], video
    assert video["images"][0]["filename"] == "shot01_00001_.mp4", video
    status, data = router.request("GET", "/view?" + urllib.parse.urlencode(video["images"][0]))
    assert status == 200 and data == b"FAKEMP4-minimax-h3"
    call = [e for e in router.fake_entries() if e["path"] == "/v1/videos"][-1]
    fields = call["fields"]
    assert fields["model"] == "minimax-h3" and fields["size"] == "512x512", fields
    assert fields["gufo"] == {"seed": 7, "steps": 8, "frames": 124, "lora": "turbo"}, fields
    assert fields["seconds"] == 5
    # ComfyUI stretches guide images to the canvas: the adapter does too (ffmpeg).
    size = "512x512" if stretched else "64x48"
    assert fields["first_frame"] == "data:image/png;base64," + size, fields
    assert fields["last_frame"] == "data:image/png;base64," + size, fields
    entry = router.run(h3_graph("ref.png", "ref (1).png", rife=4))
    fields = [e for e in router.fake_entries() if e["path"] == "/v1/videos"][-1]["fields"]
    assert fields["gufo"]["interpolate"] == 4, fields
    assert entry["outputs"]["58"]["images"][0]["filename"] == "shot01_00002_.mp4"
    # 832x512, an intermediate guide image and a voice line as audio anchor.
    router.run(h3_graph("ref.png", "ref (1).png", 832, 512, middle="ref.png", audio="voice.wav"))
    fields = [e for e in router.fake_entries() if e["path"] == "/v1/videos"][-1]["fields"]
    assert fields["size"] == "832x512", fields
    size = "832x512" if stretched else "64x48"
    assert fields["gufo"]["keyframes"] == [
        {"frame": 60, "image": "data:image/png;base64," + size}], fields
    assert fields["audio_reference"] == ("data:audio/wav;base64," +
                                         hashlib.sha256(make_wav()).hexdigest()), fields
    assert fields["gufo"]["audio_reference_frame"] == 0, fields


def check_music(router, audio_ext):
    entry = router.run(ace_graph())
    audio = entry["outputs"]["10"]["audio"][0]
    assert audio["filename"] == "titel_00001." + audio_ext, audio
    status, data = router.request("GET", "/view?" + urllib.parse.urlencode(audio))
    assert status == 200 and len(data) > 100
    call = router.fake_entries()[-1]
    assert call["path"] == "/v1/audio/music" and call["model"] == "ace-step-1.5-xl-turbo", call
    assert call["fields"] == {"model": "ace-step-1.5-xl-turbo", "prompt": "synth pop, 90s anime",
                              "lyrics": "[verse]\nNeon", "duration": 30, "seed": 5, "steps": 8,
                              "bpm": 120, "timesignature": "4", "language": "de",
                              "keyscale": "C major", "audio_codes": False, "lm_cfg": 2.0,
                              "lm_temperature": 0.85, "lm_top_p": 0.9,
                              "response_format": "wav"}, call


def check_camera(router):
    entry = router.run(camera_graph("ref.png", {"camera": "dolly_in", "speed": 0.5, "ease": "ease_in_out"}))
    video = entry["outputs"]["33"]
    assert video["images"][0]["filename"] == "shot_00001_.mp4", video
    status, data = router.request("GET", "/view?" + urllib.parse.urlencode(video["images"][0]))
    assert status == 200 and data == b"FAKEMP4-wan2.1-uni3c"
    call = [e for e in router.fake_entries() if e["path"] == "/v1/videos"][-1]
    fields = call["fields"]
    assert call["model"] == "wan2.1-uni3c", call
    assert fields == {"model": "wan2.1-uni3c", "prompt": "slow camera move through the office",
                      "size": "832x480", "seconds": 5,
                      "first_frame": "data:image/png;base64,64x48",
                      "gufo": {"seed": 99, "steps": 4, "frames": 81,
                               "camera": {"type": "dolly_in", "speed": 0.5,
                                          "ease": "ease_in_out"}}}, fields
    # A rendered guide video goes as gufo.camera.render_video.
    render = b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 2
    router.upload("render.mp4", render)
    router.run(camera_graph("ref.png", render="render.mp4", length=33))
    fields = [e for e in router.fake_entries() if e["path"] == "/v1/videos"][-1]["fields"]
    assert fields["seconds"] == 2 and fields["gufo"]["frames"] == 33, fields
    assert fields["gufo"]["camera"] == {
        "render_video": "data:video/mp4;base64," + hashlib.sha256(render).hexdigest()}, fields
    status, body = router.json("POST", "/prompt", {"prompt": camera_graph("ref.png", {"camera": "spiral"})})
    assert status == 400 and "spiral" in body["error"]["message"], body


def wan_delivery_graph(model_file="realesr-animevideov3.pth"):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_ti2v_5B_fp16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "a fox in the snow"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "55": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"vae": ["3", 0], "width": 1280, "height": 704, "length": 49, "batch_size": 1}},
        "9": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["55", 0], "seed": 42, "steps": 20, "cfg": 5, "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "60": {"class_type": "RIFE VFI", "inputs": {"ckpt_name": "rife49.pth", "frames": ["8", 0], "multiplier": 2}},
        "61": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model_file}},
        "62": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["61", 0], "image": ["60", 0]}},
        "63": {"class_type": "ImageScale", "inputs": {"image": ["62", 0], "upscale_method": "lanczos", "width": 1920, "height": 1080, "crop": "center"}},
        "64": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["63", 0], "frame_rate": 60, "loop_count": 0, "filename_prefix": "film/lieferung/shot", "format": "video/h264-mp4", "pingpong": False, "save_output": True}},
    }


def check_delivery(router):
    entry = router.run(wan_delivery_graph())
    gifs = entry["outputs"]["64"]["gifs"]
    assert gifs[0]["filename"] == "shot_00001_.mp4" and gifs[0]["frame_rate"] == 60, gifs
    assert gifs[0]["format"] == "video/h264-mp4", gifs
    status, data = router.request("GET", "/view?" + urllib.parse.urlencode(
        {k: gifs[0][k] for k in ("filename", "subfolder", "type")}))
    assert status == 200 and data == b"FAKEMP4-wan2.2-ti2v-5b"
    call = [e for e in router.fake_entries() if e["path"] == "/v1/videos"][-1]
    assert call["model"] == "wan2.2-ti2v-5b", call
    gufo = call["fields"]["gufo"]
    assert gufo["interpolate"] == {"fps": 60}, gufo
    assert gufo["upscale"] == {"width": 1920, "height": 1080, "fit": "crop",
                               "model": "realesr-animevideov3"}, gufo
    status, body = router.json("POST", "/prompt", {"prompt": wan_delivery_graph(
        "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors")})
    assert status == 400 and "not served by gufo" in body["error"]["message"], body
    assert "61" in body["node_errors"], body


def check_models_control(router):
    # llama.cpp router mode: POST /models/load and /models/unload {"model"}.
    status, body = router.json("POST", "/models/unload", {"model": "nope"})
    assert status == 400 and body == {"error": {"code": 400, "message": "model is not found",
                                                "type": "invalid_request_error"}}, body
    status, body = router.json("POST", "/models/load", {"model": "nope"})
    assert status == 404 and body["error"]["type"] == "not_found_error", body
    status, body = router.json("POST", "/models/load", {"model": "qwen3-tts"})
    assert status == 200 and body == {"success": True}, body
    loaded = {m["id"]: m["status"]["value"] for m in router.json("GET", "/models")[1]["data"]}
    assert loaded["qwen3-tts"] == "loaded", loaded
    status, body = router.json("POST", "/models/load", {"model": "qwen3-tts"})
    assert status == 400 and body["error"]["message"] == "model is already running", body
    status, body = router.json("POST", "/models/unload", {"model": "qwen3-tts"})
    assert status == 200 and body == {"success": True}, body
    loaded = {m["id"]: m["loaded"] for m in router.json("GET", "/v1/models")[1]["data"]}
    assert not loaded["qwen3-tts"], loaded
    status, body = router.json("POST", "/models/unload", {"model": "qwen3-tts"})
    assert status == 400 and body["error"]["message"] == "model is not running", body


def check_errors(router):
    graph = qwen_graph()
    graph["9"] = {"class_type": "FancyUpscaler", "inputs": {"image": ["7", 0]}}
    graph["10"] = {"class_type": "EmptyAceStepLatentAudio", "inputs": {"seconds": 30, "batch_size": 1}}
    status, body = router.json("POST", "/prompt", {"prompt": graph})
    assert status == 400, body
    assert "FancyUpscaler (node 9)" in body["error"]["message"], body
    assert "port pending" in body["error"]["message"], body
    assert set(body["node_errors"]) == {"9", "10"}, body
    status, body = router.json("POST", "/prompt", {"prompt": h3_graph("ref.png", "ref.png", 840, 512)})
    assert status == 400 and "840x512" in body["error"]["message"], body
    status, body = router.json("POST", "/prompt", {"prompt": qwen_graph(refs=["missing.png"])})
    assert status == 400 and "missing.png" in body["error"]["message"], body
    status, body = router.json("POST", "/prompt", {"no": "prompt"})
    assert status == 400 and body["error"]["type"] == "no_prompt", body
    status, _ = router.request("GET", "/view?filename=..%2F..%2Fhistory.json&type=output")
    assert status == 400
    status, _ = router.request("GET", "/view?filename=nothing.png")
    assert status == 404
    status, _ = router.request("GET", "/view?filename=x.png&subfolder=..&type=input")
    assert status == 400
    # A failing gufo request becomes a ComfyUI execution_error.
    prompt_id = router.submit(qwen_graph(prompt="FAIL please"))
    entry = router.wait(prompt_id)
    assert entry["status"]["status_str"] == "error", entry["status"]
    error = [m for m in entry["status"]["messages"] if m[0] == "execution_error"][0][1]
    assert error["node_id"] == "6" and "fake failure" in error["exception_message"], error


def check_websocket(router):
    ws = WsClient(router.port, "/ws?clientId=wsclient")
    hello = ws.message()
    assert hello["type"] == "status" and hello["data"]["sid"] == "wsclient", hello
    prompt_id = router.submit(design_graph(text="Hallo Socket."), client_id="wsclient")
    seen = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        message = ws.message()
        if message["data"].get("prompt_id") != prompt_id:
            continue
        seen.append(message["type"])
        if message["type"] == "executing" and message["data"]["node"] is None:
            break
    ws.close()
    assert seen[0] == "execution_start" and "executed" in seen, seen
    assert "execution_success" in seen, seen


def check_queue_history_free(router):
    status, _ = router.request("POST", "/interrupt", b"")
    assert status == 200
    status, _ = router.request("POST", "/queue", {"clear": True})
    assert status == 200
    status, history = router.json("GET", "/history?max_items=2")
    assert len(history) == 2, list(history)
    victim = list(history)[0]
    status, _ = router.request("POST", "/history", {"delete": [victim]})
    assert router.json("GET", "/history/" + victim)[1] == {}
    status, _ = router.request("POST", "/free", {"unload_models": True, "free_memory": True})
    assert status == 200
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        models = router.json("GET", "/v1/models")[1]["data"]
        if not any(m["loaded"] for m in models):
            break
        time.sleep(0.1)
    assert not any(m["loaded"] for m in models), models
    return list(router.json("GET", "/history")[1])


def check_autoload_retry(binary, tmp):
    """A failed --autoload keeps the router up: the model is listed as
    failed, other models and pages answer, and a later retry loads it."""
    work = os.path.join(tmp, "autoload")
    pages = os.path.join(work, "pages", "status")
    os.makedirs(pages)
    with open(os.path.join(pages, "index.html"), "w") as handle:
        handle.write("<p>status</p>")
    fail_dir = os.path.join(work, "fail")
    os.makedirs(fail_dir)
    open(os.path.join(fail_dir, "flaky.fail"), "w").close()
    python = os.environ.get("COMFY_TEST_PYTHON") or sys.executable
    preset = os.path.join(work, "autoload.preset")
    with open(preset, "w") as handle:
        for model in ("flaky", "steady"):
            handle.write("[external/%s]\ncommand = %s\nargs = %s --fake-worker\n"
                         "served-model-name = %s\n\n" % (model, python, THIS, model))
    port = free_port()
    env = {k: v for k, v in os.environ.items()
           if k not in {"HOST", "PORT", "GUFO_HOST", "GUFO_PORT"}}
    env["COMFY_FAKE_LOG"] = os.path.join(work, "fake.jsonl")
    env["COMFY_FAKE_FAIL_DIR"] = fail_dir
    log = open(os.path.join(work, "router.log"), "ab")
    proc = subprocess.Popen(
        [binary, "router", "--models-preset", preset, "--host", "127.0.0.1",
         "--port", str(port), "--models-max", "0", "--autoload",
         "--autoload-retry-seconds", "3,60", "--pages-dir",
         os.path.join(work, "pages")],
        env=env, stdout=log, stderr=subprocess.STDOUT)

    def get(path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        if body is None:
            conn.request("GET", path)
        else:
            conn.request("POST", path, body=json.dumps(body).encode(),
                         headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, data

    def tail():
        with open(os.path.join(work, "router.log"), "rb") as handle:
            return handle.read()[-3000:].decode(errors="replace")

    def status_of(model):
        listing = json.loads(get("/v1/models")[1])["data"]
        return [m for m in listing if m["id"] == model][0]["status"]

    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                if get("/health")[0] == 200:
                    break
            except OSError:
                time.sleep(0.05)
        # The first start of "flaky" exits with status 1.
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not status_of("flaky").get("failed"):
            assert proc.poll() is None, "router exited: " + tail()
            time.sleep(0.1)
        status = status_of("flaky")
        assert status["value"] == "unloaded" and status["failed"] is True, status
        assert status["attempts"] == 1 and status["retry_in_s"] <= 3, status
        status, body = get("/router/status")
        assert status == 200 and "flaky" in json.loads(body)["autoload_failed"], body
        # Meanwhile other models and pages answer.
        status, body = get("/v1/audio/speech", {"model": "steady", "input": "Hallo",
                                                 "voice": "ryan"})
        assert status == 200 and body[:4] == b"RIFF", (status, body[:200])
        status, body = get("/status/")
        assert status == 200 and b"status" in body, status
        # The retry after 3 s loads it; the failure mark goes away.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and status_of("flaky")["value"] != "loaded":
            assert proc.poll() is None, "router exited: " + tail()
            time.sleep(0.2)
        status = status_of("flaky")
        assert status["value"] == "loaded" and "failed" not in status, status
        assert json.loads(get("/router/status")[1])["autoload_failed"] == {}
        assert proc.poll() is None
    except BaseException:
        print(tail())
        raise
    finally:
        proc.terminate()
        proc.wait(10)
        log.close()


def main():
    if "--fake-worker" in sys.argv:
        return serve_fake_worker(sys.argv)
    binary = sys.argv[1]
    ffmpeg = os.environ.get("COMFY_TEST_FFMPEG") or shutil.which("ffmpeg")
    tmp = tempfile.mkdtemp(prefix="gufo_comfy_")
    comfy_dir = os.path.join(tmp, "comfy")
    router = Router(binary, tmp, comfy_dir, ffmpeg)
    try:
        check_front(router)
        check_uploads(router)
        check_qwen(router)
        check_qwen_references(router)
        check_speech(router, "flac" if ffmpeg else "wav")
        check_video(router, bool(ffmpeg))
        check_music(router, "flac" if ffmpeg else "wav")
        check_camera(router)
        check_delivery(router)
        check_models_control(router)
        check_errors(router)
        check_websocket(router)
        remaining = check_queue_history_free(router)
    except BaseException:
        print(router.tail())
        raise
    finally:
        router.stop()
    # History survives a restart.
    router = Router(binary, tmp, comfy_dir, ffmpeg)
    try:
        assert list(router.json("GET", "/history")[1]) == remaining
        entry = router.run(qwen_graph())
        assert entry["outputs"]["8"]["images"][0]["filename"] == "demo_00003_.png"
    finally:
        router.stop()
    if not os.environ.get("COMFY_SKIP_AUTOLOAD"):
        check_autoload_retry(binary, tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    print("router_comfy_test: all passed (ffmpeg: %s)" % (ffmpeg or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
