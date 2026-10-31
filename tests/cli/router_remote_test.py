"""Contract tests for [remote/<id>] preset sections over real HTTP.

Run through ctest as router_remote_contract_test with the gufo binary as
argv[1]. A fake remote gufo router (in-process, 127.0.0.1 only) plays the
routes the router relays: model list, images (JSON and multipart), speech,
async music and video jobs, live previews (SSE), /models/load|unload. A
fake ComfyUI answers /queue for the busy check. One local model (this file
with --fake-worker) shows that local routing keeps working beside them.

Cases: /v1/models with an unreachable remote; byte-identical JSON body and
the remote Authorization; multipart and 413; the video chain; preview SSE
streamed; async music; load/unload without blocking the router; outage ->
503 within the connect timeout while local keeps working, and recovery;
the GPU slot runs requests one at a time and answers remote_busy after the
queue timeout; busy-check; a ComfyUI graph and a job-API job on a remote
model; exit 2 for unsafe presets; WebSocket 501; autoload, /router/status
and memory.json leave remote models alone.
"""

import base64
import http.client
import json
import os
import random
import shutil
import socket
import string
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

THIS = os.path.abspath(__file__)
KEY = "remote-test-secret"
KEY_ENV = "GUFO_REMOTE_TEST_KEY"


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def make_png(width, height):
    raw = (b"\x00" + b"\x10\x50\xa0" * width) * height

    def chunk(kind, data):
        body = kind + data
        return (struct.pack(">I", len(data)) + body +
                struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


def make_wav(samples=2400):
    pcm = b"\x01\x00" * samples
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16) +
            b"data" + struct.pack("<I", len(pcm)) + pcm)


def serve_fake_worker(argv):
    """The local model: health plus an echo of model and path."""
    port = int(argv[argv.index("--port") + 1])
    model = argv[argv.index("--served-model-name") + 1]

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def send(self, status, body):
            body = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send(200, {"status": "ok", "model": model, "path": self.path})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            self.send(200, {"model": model, "path": self.path})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.serve_forever()


class FakeRemote:
    """A gufo router on another machine, as far as the relay can tell."""

    def __init__(self, port=None, models=()):
        self.lock = threading.Lock()
        self.log = []  # dicts: path, method, auth, body, start, end
        self.states = {m: "unloaded" for m in models}
        self.videos = {}  # id -> status
        self.music = {}  # id -> [status, hold]
        self.load_delay = 0.0
        self.server = None
        self.port = port or free_port()
        self.start()

    def start(self):
        remote = self

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

            def record(self, raw=b""):
                entry = {"path": self.path, "method": self.command,
                         "auth": self.headers.get("Authorization"),
                         "body": raw, "start": time.monotonic(), "end": None}
                with remote.lock:
                    remote.log.append(entry)
                return entry

            def authorized(self):
                if self.headers.get("Authorization") == "Bearer " + KEY:
                    return True
                self.send(401, {"error": {"message": "invalid api key",
                                          "code": "invalid_api_key"}})
                return False

            def do_GET(self):
                if not self.authorized():
                    return
                path, _, query = self.path.partition("?")
                if path == "/v1/models":
                    with remote.lock:
                        data = [{"id": m, "object": "model", "loaded": s == "loaded",
                                 "status": {"value": s, "args": []}}
                                for m, s in remote.states.items()]
                    return self.send(200, {"object": "list", "data": data})
                self.record()
                if path.startswith("/v1/previews/") and path.endswith("/events"):
                    # SSE: three events, a pause before each later one.
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for seq in range(1, 4):
                        self.wfile.write(("data: {\"seq\": %d, \"query\": \"%s\"}\n\n"
                                          % (seq, query)).encode())
                        self.wfile.flush()
                        time.sleep(0.4)
                    self.close_connection = True
                    return
                if path.startswith("/v1/videos/"):
                    rest = path[len("/v1/videos/"):]
                    content = rest.endswith("/content")
                    job = rest[:-len("/content")] if content else rest
                    with remote.lock:
                        status = remote.videos.get(job)
                    if status is None:
                        return self.send(404, {"error": {"code": "video_not_found"}})
                    if content:
                        return self.send(200, b"FAKEMP4-" + job.encode(), "video/mp4")
                    return self.send(200, {"id": job, "object": "video",
                                           "status": status})
                if path.startswith("/v1/audio/music/"):
                    rest = path[len("/v1/audio/music/"):]
                    content = rest.endswith("/content")
                    job = rest[:-len("/content")] if content else rest
                    with remote.lock:
                        entry = remote.music.get(job)
                        if entry is not None and not content and not entry[1]:
                            entry[0] = {"queued": "in_progress",
                                        "in_progress": "completed"}.get(entry[0], entry[0])
                        status = None if entry is None else entry[0]
                    if status is None:
                        return self.send(404, {"error": {"code": "not_found"}})
                    if content:
                        return self.send(200, make_wav(), "audio/wav")
                    return self.send(200, {"id": job, "object": "audio.music",
                                           "status": status})
                self.send(404, {"error": {"code": "not_found"}})

            def do_DELETE(self):
                if not self.authorized():
                    return
                self.record()
                path = self.path.partition("?")[0]
                job = path[len("/v1/videos/"):]
                with remote.lock:
                    known = remote.videos.pop(job, None) is not None
                self.send(200 if known else 404, {"id": job, "deleted": known})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                if not self.authorized():
                    return
                entry = self.record(raw)
                path = self.path.partition("?")[0]
                try:
                    if path in ("/models/load", "/models/unload"):
                        model = json.loads(raw)["model"]
                        time.sleep(remote.load_delay)
                        with remote.lock:
                            remote.states[model] = ("loaded" if path.endswith("load")
                                                    and not path.endswith("unload")
                                                    else "unloaded")
                        return self.send(200, {"success": True})
                    if path == "/v1/images/edits":
                        return self.send(200, {"data": [{"b64_json": base64.b64encode(
                            make_png(8, 8)).decode()}]})
                    fields = json.loads(raw)
                    prompt = fields.get("prompt", "") or fields.get("input", "")
                    if "SLEEP" in prompt:
                        time.sleep(float(prompt.split("SLEEP")[1].split()[0]))
                    if path == "/v1/images/generations":
                        width, height = (int(v) for v in
                                         fields.get("size", "16x16").split("x"))
                        count = int(fields.get("n", 1))
                        return self.send(200, {"created": 1, "data": [
                            {"b64_json": base64.b64encode(make_png(width, height)).decode()}
                            for _ in range(count)]})
                    if path == "/v1/audio/speech":
                        return self.send(200, make_wav(), "audio/wav")
                    if path == "/v1/audio/music":
                        if not fields.get("async"):
                            return self.send(200, make_wav(), "audio/wav")
                        with remote.lock:
                            job = "music_%d" % (len(remote.music) + 1)
                            remote.music[job] = ["queued", "HOLD" in prompt]
                        return self.send(202, {"id": job, "object": "audio.music",
                                               "status": "queued"})
                    if path == "/v1/videos":
                        job = "video_" + "".join(random.choice(string.ascii_lowercase)
                                                 for _ in range(24))
                        with remote.lock:
                            remote.videos[job] = ("in_progress" if "HOLD" in prompt
                                                  else "completed")
                        return self.send(202, {"id": job, "object": "video",
                                               "status": "queued"})
                    if path == "/v1/chat/completions":
                        return self.send(200, {"object": "chat.completion",
                                               "model": fields.get("model")})
                    self.send(404, {"error": {"code": "not_found"}})
                finally:
                    entry["end"] = time.monotonic()

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def entries(self, path=None):
        with self.lock:
            return [e for e in self.log
                    if path is None or e["path"].partition("?")[0] == path]

    def clear(self):
        with self.lock:
            self.log.clear()


class FakeComfy:
    """ComfyUI's /queue, busy until `busy_until`."""

    def __init__(self):
        self.busy_until = 0.0
        self.port = free_port()
        comfy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                running = [[1, "prompt-x", {}, {}, []]] if time.monotonic() < comfy.busy_until else []
                body = json.dumps({"queue_running": running, "queue_pending": []}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def write_preset(path, remote_port, comfy_port, dead_port, python):
    url = "http://127.0.0.1:%d" % remote_port
    sections = [
        "[external/local]\ncommand = %s\nargs = %s --fake-worker\n"
        "served-model-name = local\n" % (python, THIS),
        "[remote/qwen-image-2.1]\nurl = %s\nmodality = image\napi-key-env = %s\n"
        "gpu-slot = studio\nqueue-timeout-seconds = 30\nmax-request-bytes = 200000\n"
        % (url, KEY_ENV),
        "[remote/img2]\nurl = %s\nmodality = image\napi-key-env = %s\n"
        "gpu-slot = studio\n" % (url, KEY_ENV),
        "[remote/wan]\nurl = %s\nmodality = video\napi-key-env = %s\n"
        "gpu-slot = studio\n" % (url, KEY_ENV),
        "[remote/ace]\nurl = %s\nmodality = tts\napi-key-env = %s\n"
        "gpu-slot = studio\n" % (url, KEY_ENV),
        "[remote/tts-quick]\nurl = %s\nmodality = tts\napi-key-env = %s\n"
        "gpu-slot = studio\nqueue-timeout-seconds = 1\n" % (url, KEY_ENV),
        "[remote/checked]\nurl = %s\nmodality = image\napi-key-env = %s\n"
        "gpu-slot = other\nbusy-check-url = http://127.0.0.1:%d/queue\n"
        % (url, KEY_ENV, comfy_port),
        "[remote/gone]\nurl = http://127.0.0.1:%d\nmodality = llm\n"
        "connect-timeout-seconds = 1\n" % dead_port,
    ]
    with open(path, "w") as handle:
        handle.write("\n".join(sections))


class Router:
    def __init__(self, binary, tmp, preset, extra=(), port=None, log_name="router.log"):
        self.port = port or free_port()
        self.tmp = tmp
        env = {k: v for k, v in os.environ.items()
               if k not in {"HOST", "PORT", "GUFO_HOST", "GUFO_PORT"}}
        env[KEY_ENV] = KEY
        args = [binary, "router", "--models-preset", preset, "--host", "127.0.0.1",
                "--port", str(self.port), "--models-max", "0",
                "--remote-allow", "127.0.0.1"] + list(extra)
        self.log_path = os.path.join(tmp, log_name)
        self.log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(args, env=env, stdout=self.log,
                                     stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                if self.request("GET", "/health")[0] == 200:
                    return
            except OSError:
                time.sleep(0.05)
        raise AssertionError("router did not start: " + self.tail())

    def tail(self):
        with open(self.log_path, "rb") as handle:
            return handle.read()[-4000:].decode(errors="replace")

    def stop(self):
        self.proc.terminate()
        self.proc.wait(30)
        self.log.close()

    def request(self, method, path, body=None, headers=None, timeout=60):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        headers = dict(headers or {})
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
            headers.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        payload = response.read()
        self.last_headers = dict(response.getheaders())
        conn.close()
        return response.status, payload

    def json(self, method, path, body=None, headers=None, timeout=60):
        status, payload = self.request(method, path, body, headers, timeout)
        try:
            return status, json.loads(payload)
        except ValueError:
            return status, payload

    def models(self):
        _, listing = self.json("GET", "/v1/models")
        return {m["id"]: m for m in listing["data"]}

    def status(self):
        return self.json("GET", "/router/status")[1]


def code_of(answer):
    return answer["error"]["code"] if isinstance(answer, dict) and "error" in answer else None


def wait_for(predicate, timeout, what):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError("timed out waiting for " + what)


# ------------------------------------------------------------------ cases

def check_models_listing(router, remote):
    with remote.lock:
        remote.states["qwen-image-2.1"] = "loaded"
    wait_for(lambda: router.models()["qwen-image-2.1"]["status"]["value"] == "loaded",
             8, "the remote model state in /v1/models")
    models = router.models()
    image = models["qwen-image-2.1"]
    assert image["modality"] == "image" and image["loaded"] is True, image
    assert image["remote"]["url"] == "http://127.0.0.1:%d" % remote.port, image
    assert image["remote"]["reachable"] is True, image
    assert image["status"]["args"] == [], image
    assert models["local"].get("remote") is None, models["local"]
    # The dead remote turns unreachable after the first probe.
    wait_for(lambda: router.models()["gone"]["remote"]["reachable"] is False, 8,
             "the dead remote to show as unreachable")
    gone = router.models()["gone"]
    assert gone["loaded"] is False and gone["status"]["value"] == "unloaded", gone
    assert gone["remote"]["error"], gone


def check_json_bytes_and_key(router, remote):
    remote.clear()
    body = ('{"model":"qwen-image-2.1",  "prompt":"ä 猫"  ,'
            '"size":"32x16","n":2}').encode("utf-8")
    code, answer = router.json("POST", "/v1/images/generations", body,
                               {"Content-Type": "application/json",
                                "Authorization": "Bearer client-key"})
    assert code == 200 and len(answer["data"]) == 2, answer
    seen = remote.entries("/v1/images/generations")
    assert len(seen) == 1 and seen[0]["body"] == body, seen
    assert seen[0]["auth"] == "Bearer " + KEY, seen[0]["auth"]


def multipart(fields, files):
    boundary = "remotetest%d" % time.monotonic_ns()
    parts = []
    for name, value in fields:
        parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                      % (boundary, name, value)).encode())
    for name, data in files:
        parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"; "
                      "filename=\"x.png\"\r\nContent-Type: image/png\r\n\r\n"
                      % (boundary, name)).encode() + data + b"\r\n")
    return (b"".join(parts) + ("--%s--\r\n" % boundary).encode(),
            "multipart/form-data; boundary=" + boundary)


def check_multipart_and_413(router, remote):
    remote.clear()
    body, ctype = multipart([("model", "qwen-image-2.1"), ("prompt", "edit"),
                             ("size", "8x8")], [("image[]", make_png(40, 40))])
    code, answer = router.json("POST", "/v1/images/edits", body, {"Content-Type": ctype})
    assert code == 200 and answer["data"], answer
    seen = remote.entries("/v1/images/edits")
    assert len(seen) == 1 and seen[0]["body"] == body, len(seen)
    big, ctype = multipart([("model", "qwen-image-2.1"), ("prompt", "edit")],
                           [("image[]", os.urandom(250000))])
    code, answer = router.json("POST", "/v1/images/edits", big, {"Content-Type": ctype})
    assert code == 413 and code_of(answer) == "request_too_large", (code, answer)
    assert len(remote.entries("/v1/images/edits")) == 1


def check_video_chain(router, remote):
    remote.clear()
    code, job = router.json("POST", "/v1/videos",
                            {"model": "wan", "prompt": "a wave HOLD", "seconds": "1"})
    assert code == 202 and job["id"].startswith("video_"), job
    video = job["id"]
    status = router.status()["remote"]
    assert status["jobs"][video]["model"] == "wan", status
    assert status["gpu_slots"]["studio"]["held"] is True, status
    # The job holds the studio slot: an image on the same slot waits.
    done = {}

    def image():
        done["answer"] = router.json("POST", "/v1/images/generations",
                                     {"model": "img2", "prompt": "after", "size": "8x8"})
        done["at"] = time.monotonic()

    worker = threading.Thread(target=image)
    worker.start()
    time.sleep(1.0)
    assert "answer" not in done and not remote.entries("/v1/images/generations")
    code, answer = router.json("GET", "/v1/videos/" + video)
    assert code == 200 and answer["status"] == "in_progress", answer
    with remote.lock:
        remote.videos[video] = "completed"
    released = time.monotonic()
    code, answer = router.json("GET", "/v1/videos/" + video)
    assert code == 200 and answer["status"] == "completed", answer
    worker.join(30)
    assert done["answer"][0] == 200, done
    assert done["at"] - released < 3, done["at"] - released
    code, mp4 = router.request("GET", "/v1/videos/%s/content" % video)
    assert code == 200 and mp4 == b"FAKEMP4-" + video.encode(), mp4
    code, answer = router.json("DELETE", "/v1/videos/" + video)
    assert code == 200 and answer["deleted"] is True, answer
    assert video not in router.status()["remote"]["jobs"]


def check_preview_sse(router, remote):
    conn = http.client.HTTPConnection("127.0.0.1", router.port, timeout=30)
    began = time.monotonic()
    conn.request("GET", "/v1/previews/img_7/events?model=qwen-image-2.1&x=1")
    response = conn.getresponse()
    assert response.status == 200, response.status
    assert response.getheader("Content-Type") == "text/event-stream"
    first = response.readline()
    first_at = time.monotonic() - began
    rest = response.read()
    total = time.monotonic() - began
    conn.close()
    assert first.startswith(b"data: {\"seq\": 1"), first
    assert b"model=qwen-image-2.1&x=1" in first, first
    assert rest.count(b"data: ") == 2, rest
    # Streamed, not buffered: the first event arrives long before the end.
    assert first_at < 0.6 and total > 0.9, (first_at, total)
    code, answer = router.json("GET", "/v1/previews/img_7?model=nope")
    assert code == 404, answer


def check_async_music(router, remote):
    code, job = router.json("POST", "/v1/audio/music",
                            {"model": "ace", "prompt": "lofi", "async": True})
    assert code == 202 and job["id"].startswith("music_"), job
    music = job["id"]
    assert router.status()["remote"]["jobs"][music]["kind"] == "music"
    seen = []
    for _ in range(4):
        code, status = router.json("GET", "/v1/audio/music/" + music)
        assert code == 200, status
        seen.append(status["status"])
        if status["status"] == "completed":
            break
    assert seen[-1] == "completed", seen
    wait_for(lambda: music not in router.status()["remote"]["jobs"], 8,
             "the music job to release its slot")
    code, wav = router.request("GET", "/v1/audio/music/%s/content" % music)
    assert code == 200 and wav[:4] == b"RIFF", wav[:16]
    code, wav = router.request("POST", "/v1/audio/music", {"model": "ace", "prompt": "sync"})
    assert code == 200 and wav[:4] == b"RIFF", wav[:16]
    assert router.status()["remote"]["gpu_slots"]["studio"]["held"] is False


def check_load_unload(router, remote):
    remote.load_delay = 2.0
    answers = {}

    def load():
        answers["load"] = router.json("POST", "/models/load", {"model": "img2"})

    worker = threading.Thread(target=load)
    worker.start()
    time.sleep(0.3)
    began = time.monotonic()
    code, _ = router.json("GET", "/v1/models")
    assert code == 200
    code, answer = router.json("POST", "/v1/chat/completions", {"model": "local"})
    assert code == 200 and answer["model"] == "local", answer
    code, _ = router.json("GET", "/router/status")
    assert code == 200
    assert time.monotonic() - began < 1.5, time.monotonic() - began
    worker.join(30)
    remote.load_delay = 0.0
    assert answers["load"] == (200, {"success": True}), answers
    assert remote.entries("/models/load")[-1]["auth"] == "Bearer " + KEY
    code, answer = router.json("POST", "/models/unload", {"model": "img2"})
    assert code == 200 and answer == {"success": True}, answer
    with remote.lock:
        assert remote.states["img2"] == "unloaded"


def check_outage(router):
    began = time.monotonic()
    code, answer = router.json("POST", "/v1/chat/completions", {"model": "gone"})
    took = time.monotonic() - began
    assert code == 503 and code_of(answer) == "remote_unavailable", (code, answer)
    assert router.last_headers.get("Retry-After") == "30", router.last_headers
    assert took < 2.0, took  # connect-timeout-seconds = 1, plus one
    code, answer = router.json("POST", "/v1/chat/completions", {"model": "local"})
    assert code == 200 and answer["model"] == "local", answer


def check_gpu_slot_serial(router, remote):
    remote.clear()
    answers = []

    def image(model):
        answers.append(router.json("POST", "/v1/images/generations",
                                   {"model": model, "prompt": "SLEEP0.8", "size": "8x8"}))

    workers = [threading.Thread(target=image, args=(m,))
               for m in ("qwen-image-2.1", "img2", "qwen-image-2.1")]
    for worker in workers:
        worker.start()
        time.sleep(0.05)
    time.sleep(0.3)
    slot = router.status()["remote"]["gpu_slots"]["studio"]
    assert slot["held"] is True and slot["waiting"] == 2, slot
    # tts-quick shares the slot but waits at most one second.
    began = time.monotonic()
    code, answer = router.json("POST", "/v1/audio/speech", {"model": "tts-quick", "input": "x"})
    assert code == 503 and code_of(answer) == "remote_busy", (code, answer)
    assert 0.9 < time.monotonic() - began < 2.5
    for worker in workers:
        worker.join(30)
    assert [a[0] for a in answers] == [200, 200, 200], answers
    runs = sorted((e["start"], e["end"]) for e in remote.entries("/v1/images/generations"))
    assert len(runs) == 3, runs
    for (_, end), (start, _) in zip(runs, runs[1:]):
        assert start >= end - 0.01, runs  # one at a time
    assert not remote.entries("/v1/audio/speech")


def check_busy_check(router, remote, comfy):
    comfy.busy_until = time.monotonic() + 3.0
    began = time.monotonic()
    code, answer = router.json("POST", "/v1/images/generations",
                               {"model": "checked", "prompt": "x", "size": "8x8"})
    took = time.monotonic() - began
    assert code == 200, answer
    assert took > 2.5, took
    checks = router.status()["remote"]["busy_checks"]
    url = "http://127.0.0.1:%d/queue" % comfy.port
    assert checks[url]["result"] == "free", checks


def qwen_graph():
    return {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
        "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["1", 0], "prompt": "an office", "negative_prompt": "", "vae": ["2", 0], "resolution": 1024}},
        "6": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["5", 2], "seed": 3, "steps": 25, "cfg": 2.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "remote/test"}},
    }


def check_comfy_graph(router, remote, comfy_dir):
    remote.clear()
    code, answer = router.json("POST", "/prompt", {"prompt": qwen_graph(), "client_id": "t"})
    assert code == 200 and answer["node_errors"] == {}, answer
    prompt_id = answer["prompt_id"]
    entry = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        code, history = router.json("GET", "/history/" + prompt_id)
        if prompt_id in history:
            entry = history[prompt_id]
            break
        time.sleep(0.1)
    assert entry is not None and entry["status"]["status_str"] == "success", entry
    images = entry["outputs"]["8"]["images"]
    saved = os.path.join(comfy_dir, "output", images[0]["subfolder"], images[0]["filename"])
    with open(saved, "rb") as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
    sent = remote.entries("/v1/images/generations")
    assert sent and json.loads(sent[-1]["body"])["model"] == "qwen-image-2.1", sent


def wait_job(router, job_id):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        code, job = router.json("GET", "/v1/jobs/" + job_id)
        assert code == 200, job
        if job["status"] not in ("queued", "in_progress"):
            return job
        time.sleep(0.1)
    raise AssertionError("job %s did not end" % job_id)


def check_job_api(router, remote):
    code, job = router.json("POST", "/v1/jobs", {"request": {
        "path": "/v1/images/generations",
        "body": {"model": "qwen-image-2.1", "prompt": "fox", "size": "16x8",
                 "response_format": "b64_json"}}})
    assert code == 202, job
    done = wait_job(router, job["id"])
    assert done["status"] == "completed", done
    code, png = router.request("GET", done["result"]["files"][0]["url"])
    assert code == 200 and png[:8] == b"\x89PNG\r\n\x1a\n", png[:16]
    code, job = router.json("POST", "/v1/jobs", {"request": {
        "path": "/v1/videos", "body": {"model": "wan", "prompt": "a wave", "seconds": "1"}}})
    assert code == 202, job
    done = wait_job(router, job["id"])
    assert done["status"] == "completed", done
    code, mp4 = router.request("GET", "/v1/jobs/%s/files/video.mp4" % job["id"])
    assert code == 200 and mp4.startswith(b"FAKEMP4-video_"), mp4


def check_websocket_501(router):
    conn = http.client.HTTPConnection("127.0.0.1", router.port, timeout=10)
    conn.request("GET", "/v1/realtime?model=qwen-image-2.1", headers={
        "Connection": "Upgrade", "Upgrade": "websocket", "Sec-WebSocket-Version": "13",
        "Sec-WebSocket-Key": base64.b64encode(os.urandom(16)).decode()})
    response = conn.getresponse()
    body = json.loads(response.read())
    conn.close()
    assert response.status == 501, response.status
    assert code_of(body) == "remote_websocket_unsupported", body


def check_status_and_memory(router, remote, preset):
    status = router.status()
    block = status["remote"]
    assert set(block["models"]) == {"qwen-image-2.1", "img2", "wan", "ace",
                                     "tts-quick", "checked", "gone"}, block["models"]
    key = "127.0.0.1:%d" % remote.port
    assert block["routers"][key]["reachable"] is True, block["routers"]
    assert block["models"]["wan"]["gpu_slot"] == "studio"
    # The memory part (residents, learned peaks, events) never names them.
    local_part = json.dumps({k: v for k, v in status.items() if k != "remote"})
    for remote_id in ("qwen-image-2.1", "img2", "wan", "gone"):
        assert '"%s"' % remote_id not in local_part, local_part
    # --autoload loaded the local model only; nothing asked the remote to load.
    assert all(json.loads(e["body"])["model"] != "qwen-image-2.1"
               for e in remote.entries("/models/load")), remote.entries("/models/load")
    wait_for(lambda: router.models()["local"]["loaded"], 20, "autoload of the local model")
    memory = os.path.splitext(preset)[0] + ".memory.json"
    if os.path.exists(memory):
        with open(memory, encoding="utf-8") as handle:
            text = handle.read()
        for remote_id in ("qwen-image-2.1", "img2", "wan", "gone"):
            assert '"%s"' % remote_id not in text, text


def check_recovery(router, remote):
    remote.stop()
    code, answer = router.json("POST", "/v1/images/generations",
                               {"model": "img2", "prompt": "x", "size": "8x8"})
    assert code == 503 and code_of(answer) == "remote_unavailable", (code, answer)
    began = time.monotonic()
    code, answer = router.json("POST", "/v1/images/generations",
                               {"model": "img2", "prompt": "x", "size": "8x8"})
    assert code == 503 and time.monotonic() - began < 0.5, (code, answer)
    code, answer = router.json("POST", "/v1/chat/completions", {"model": "local"})
    assert code == 200, answer
    remote.start()
    wait_for(lambda: router.json("POST", "/v1/images/generations",
                                 {"model": "img2", "prompt": "x", "size": "8x8"})[0] == 200,
             12, "the remote to come back")


def check_unsafe_presets(binary, tmp, remote_port):
    python = os.environ.get("COMFY_TEST_PYTHON") or sys.executable
    cases = [
        ([], "[remote/a]\nurl = http://127.0.0.1:%d\nmodality = image\n" % remote_port,
         "--remote-allow"),
        (["--remote-allow", "10.0.0.0/8"],
         "[remote/a]\nurl = http://127.0.0.1:%d\nmodality = image\n" % remote_port,
         "not in --remote-allow"),
        (["--remote-allow", "127.0.0.1:1"],
         "[remote/a]\nurl = http://127.0.0.1:%d\nmodality = image\n" % remote_port,
         "not in --remote-allow"),
        (["--remote-allow", "127.0.0.1"],
         "[remote/a]\nurl = http://localhost:%d\nmodality = image\n" % remote_port,
         "IPv4"),
        (["--remote-allow", "127.0.0.1"],
         "[remote/a]\nurl = http://127.0.0.1:%d\nmodality = image\n"
         "api-key-env = GUFO_REMOTE_TEST_UNSET\n" % remote_port,
         "GUFO_REMOTE_TEST_UNSET"),
        (["--remote-allow", "127.0.0.1"],
         "[remote/a]\nurl = http://127.0.0.1:%d\nmodality = image\n"
         "busy-check-url = http://10.1.1.1:8188/queue\n" % remote_port,
         "busy-check-url"),
    ]
    env = {k: v for k, v in os.environ.items()
           if k not in {"HOST", "PORT", "GUFO_HOST", "GUFO_PORT", "GUFO_REMOTE_TEST_UNSET"}}
    for index, (flags, body, token) in enumerate(cases):
        preset = os.path.join(tmp, "unsafe%d.preset" % index)
        with open(preset, "w") as handle:
            handle.write("[external/l]\ncommand = %s\nargs = %s --fake-worker\n\n%s"
                         % (python, THIS, body))
        result = subprocess.run(
            [binary, "router", "--models-preset", preset, "--host", "127.0.0.1",
             "--port", str(free_port())] + flags,
            env=env, capture_output=True, timeout=30)
        output = (result.stdout + result.stderr).decode(errors="replace")
        assert result.returncode == 2, (index, result.returncode, output)
        assert token in output, (index, output)


def check_two_way_loop(binary, tmp):
    """Routers A and B name each other: a model remote on both answers 508
    remote_loop instead of bouncing; A -> B-local works."""
    python = os.environ.get("COMFY_TEST_PYTHON") or sys.executable
    port_a, port_b = free_port(), free_port()
    preset_a = os.path.join(tmp, "a.preset")
    preset_b = os.path.join(tmp, "b.preset")
    with open(preset_a, "w") as handle:
        handle.write("[remote/shared]\nurl = http://127.0.0.1:%d\nmodality = llm\n\n"
                     "[remote/blocal]\nurl = http://127.0.0.1:%d\nmodality = llm\n"
                     % (port_b, port_b))
    with open(preset_b, "w") as handle:
        handle.write("[remote/shared]\nurl = http://127.0.0.1:%d\nmodality = llm\n\n"
                     "[external/blocal]\ncommand = %s\nargs = %s --fake-worker\n"
                     "served-model-name = blocal\n" % (port_a, python, THIS))
    router_b = Router(binary, tmp, preset_b, port=port_b, log_name="b.log")
    router_a = None
    try:
        router_a = Router(binary, tmp, preset_a, port=port_a, log_name="a.log")
        # A -> B, local there.
        code, answer = router_a.json("POST", "/v1/chat/completions", {"model": "blocal"})
        assert code == 200 and answer["model"] == "blocal", (code, answer)
        # Remote on both: one relay, then 508 from the other side.
        for router in (router_a, router_b):
            began = time.monotonic()
            code, answer = router.json("POST", "/v1/chat/completions", {"model": "shared"})
            assert code == 508 and code_of(answer) == "remote_loop", (code, answer)
            assert time.monotonic() - began < 5
        code, answer = router_a.json("POST", "/models/load", {"model": "shared"})
        assert code == 508 and code_of(answer) == "remote_loop", (code, answer)
        code, answer = router_a.json("GET", "/v1/previews/img_1/events?model=shared")
        assert code == 508 and code_of(answer) == "remote_loop", (code, answer)
        # A client header only refuses the client's own relay.
        code, answer = router_a.json("POST", "/v1/chat/completions", {"model": "blocal"},
                                     {"X-Gufo-Forwarded": "client"})
        assert code == 508 and code_of(answer) == "remote_loop", (code, answer)
        code, answer = router_b.json("POST", "/v1/chat/completions", {"model": "blocal"},
                                     {"X-Gufo-Forwarded": "client"})
        assert code == 200 and answer["model"] == "blocal", (code, answer)
        # The probes see that the other side relays "shared" as well.
        wait_for(lambda: router_a.models()["shared"]["remote"].get("loop") is True, 10,
                 "A to mark shared as a loop")
        shared = router_a.models()["shared"]
        assert shared["loaded"] is False, shared
        status = router_a.status()["remote"]
        key = "127.0.0.1:%d" % port_b
        assert status["routers"][key]["models"]["shared"] == "remote", status
        assert status["router_id"].startswith("gufo-"), status
        assert status["router_id"] != router_b.status()["remote"]["router_id"]
    except BaseException:
        print(router_b.tail())
        if router_a is not None:
            print(router_a.tail())
        raise
    finally:
        if router_a is not None:
            router_a.stop()
        router_b.stop()


def main():
    if "--fake-worker" in sys.argv:
        return serve_fake_worker(sys.argv)
    binary = sys.argv[1]
    tmp = tempfile.mkdtemp(prefix="gufo_remote_")
    remote = FakeRemote(models=("qwen-image-2.1", "img2", "wan", "ace", "tts-quick",
                                "checked"))
    comfy = FakeComfy()
    check_unsafe_presets(binary, tmp, remote.port)
    check_two_way_loop(binary, tmp)
    print("router_remote_test: two-way loop ok", flush=True)
    preset = os.path.join(tmp, "remote.preset")
    write_preset(preset, remote.port, comfy.port, free_port(),
                 os.environ.get("COMFY_TEST_PYTHON") or sys.executable)
    comfy_dir = os.path.join(tmp, "comfy")
    jobs_dir = os.path.join(tmp, "jobs")
    os.makedirs(comfy_dir)
    os.makedirs(jobs_dir)
    router = Router(binary, tmp, preset,
                    ["--autoload", "--comfy-dir", comfy_dir, "--jobs-dir", jobs_dir])
    try:
        for name, case in (
                ("models listing", lambda: check_models_listing(router, remote)),
                ("json bytes + key", lambda: check_json_bytes_and_key(router, remote)),
                ("multipart + 413", lambda: check_multipart_and_413(router, remote)),
                ("video chain", lambda: check_video_chain(router, remote)),
                ("preview sse", lambda: check_preview_sse(router, remote)),
                ("async music", lambda: check_async_music(router, remote)),
                ("load/unload", lambda: check_load_unload(router, remote)),
                ("outage", lambda: check_outage(router)),
                ("gpu slot", lambda: check_gpu_slot_serial(router, remote)),
                ("busy check", lambda: check_busy_check(router, remote, comfy)),
                ("comfy graph", lambda: check_comfy_graph(router, remote, comfy_dir)),
                ("job api", lambda: check_job_api(router, remote)),
                ("websocket 501", lambda: check_websocket_501(router)),
                ("status + memory", lambda: check_status_and_memory(router, remote, preset)),
                ("recovery", lambda: check_recovery(router, remote))):
            case()
            print("router_remote_test: %s ok" % name, flush=True)
    except BaseException:
        print(router.tail())
        raise
    finally:
        router.stop()
    shutil.rmtree(tmp, ignore_errors=True)
    print("router_remote_test: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
