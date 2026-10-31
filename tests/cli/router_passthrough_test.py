"""Contract tests for router pass-through routes over real HTTP.

Run through ctest as router_passthrough_contract_test with the gufo binary
as argv[1]. The router starts this file with --fake-worker for every preset
model; the fake echoes its model id and the path it was asked for, and plays
the async music job API of music_api.cpp (queued -> in_progress ->
completed, WAV under /content).

Covers the worker routes the router used to answer with 404 (embeddings,
count_tokens, llama-server /completion, /apply-template, /props, voices,
image LoRAs), async music status and content reads, and the ComfyUI
/embeddings split under --comfy-dir.
"""

import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

THIS = os.path.abspath(__file__)
MODELS = ("llm", "emb", "tts", "img", "ace", "ace2")


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def serve_fake_worker(argv):
    port = int(argv[argv.index("--port") + 1])
    model = argv[argv.index("--served-model-name") + 1]
    lock = threading.Lock()
    jobs = {}  # id -> number of status reads so far
    counter = [0]

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def send(self, status, body, content_type="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def echo(self, body=b""):
            path, _, query = self.path.partition("?")
            payload = {"model": model, "path": path, "method": self.command,
                       "query": urllib.parse.parse_qs(query)}
            if body:
                payload["body"] = json.loads(body)
            self.send(200, payload)

        def music_status(self, job_id, content):
            with lock:
                if job_id not in jobs:
                    self.send(404, {"error": {"message": "unknown music job " + job_id,
                                              "code": "not_found"}})
                    return
                if content:
                    if jobs[job_id] < 3:
                        self.send(409, {"error": {"code": "not_ready"}})
                        return
                    self.send(200, b"RIFF" + model.encode(), "audio/wav")
                    return
                reads = jobs[job_id]
                jobs[job_id] += 1
            status = ("queued", "in_progress", "completed")[min(reads, 2)]
            self.send(200 if status != "queued" else 202,
                      {"id": job_id, "object": "audio.music", "model": model,
                       "status": status})

        def do_GET(self):
            path = self.path.partition("?")[0]
            if path == "/health":
                self.send(200, {"status": "ok"})
                return
            prefix = "/v1/audio/music/"
            if path.startswith(prefix):
                rest = path[len(prefix):]
                content = rest.endswith("/content")
                self.music_status(rest[:-len("/content")] if content else rest, content)
                return
            self.echo()

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            path = self.path.partition("?")[0]
            if path == "/v1/audio/music":
                body = json.loads(raw)
                if not body.get("async"):
                    self.send(200, b"RIFF-sync-" + model.encode(), "audio/wav")
                    return
                with lock:
                    counter[0] += 1
                    job_id = "music_%d" % counter[0]
                    jobs[job_id] = 0
                self.send(202, {"id": job_id, "object": "audio.music",
                                "model": model, "status": "queued"})
                return
            self.echo(raw)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.serve_forever()


class Router:
    def __init__(self, binary, tmp, extra=()):
        self.port = free_port()
        self.tmp = tmp
        preset = os.path.join(tmp, "pass.preset")
        python = os.environ.get("COMFY_TEST_PYTHON") or sys.executable
        with open(preset, "w") as handle:
            for model in MODELS:
                handle.write("[external/%s]\ncommand = %s\nargs = %s --fake-worker\n"
                             "served-model-name = %s\n\n" % (model, python, THIS, model))
        env = {k: v for k, v in os.environ.items()
               if k not in {"HOST", "PORT", "GUFO_HOST", "GUFO_PORT"}}
        args = [binary, "router", "--models-preset", preset, "--host", "127.0.0.1",
                "--port", str(self.port), "--models-max", "0"] + list(extra)
        self.log_path = os.path.join(tmp, "router.log")
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
            return handle.read()[-3000:].decode(errors="replace")

    def stop(self):
        self.proc.terminate()
        self.proc.wait(30)
        self.log.close()

    def request(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        headers = {}
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        payload = response.read()
        conn.close()
        return response.status, payload

    def json(self, method, path, body=None):
        status, payload = self.request(method, path, body)
        try:
            return status, json.loads(payload)
        except ValueError:
            return status, payload

    def loaded(self, model):
        _, listing = self.json("GET", "/v1/models")
        return [m["loaded"] for m in listing["data"] if m["id"] == model][0]


def error_code(answer):
    return answer["error"]["code"] if isinstance(answer, dict) else None


def check_post_routes(router):
    routes = (("/v1/embeddings", "emb"), ("/embeddings", "emb"), ("/embedding", "emb"),
              ("/v1/messages/count_tokens", "llm"), ("/completion", "llm"),
              ("/apply-template", "llm"))
    for path, model in routes:
        code, answer = router.json("POST", path, {"model": model, "input": "x"})
        assert code == 200, (path, code, answer)
        assert answer["model"] == model and answer["path"] == path, (path, answer)
        assert answer["body"]["input"] == "x", (path, answer)
        code, answer = router.json("POST", path, {"input": "x"})
        assert code == 400 and error_code(answer) == "missing_model", (path, answer)
        code, answer = router.json("POST", path, {"model": "nope"})
        assert code == 404 and error_code(answer) == "model_not_found", (path, answer)


def check_get_routes(router):
    for path, model in (("/v1/audio/voices", "tts"), ("/props", "llm"),
                        ("/sdapi/v1/loras", "img")):
        code, answer = router.json("GET", path + "?model=" + model)
        assert code == 200, (path, code, answer)
        assert answer["model"] == model and answer["path"] == path, (path, answer)
        assert answer["query"] == {"model": [model]}, (path, answer)
        code, answer = router.json("GET", path)
        assert code == 400 and error_code(answer) == "missing_model", (path, answer)
        code, answer = router.json("GET", path + "?model=nope")
        assert code == 404 and error_code(answer) == "model_not_found", (path, answer)


def check_unchanged(router):
    # Existing routes keep their answers.
    code, answer = router.json("POST", "/v1/chat/completions", {"model": "llm"})
    assert code == 200 and answer["model"] == "llm", answer
    code, answer = router.json("POST", "/v1/rerank", {"model": "llm"})
    assert code == 404 and error_code(answer) == "not_found", answer
    code, answer = router.json("GET", "/v1/slots?model=emb")
    assert code == 200 and answer["model"] == "emb", answer
    code, answer = router.json("GET", "/v1/slots")
    assert code == 400 and error_code(answer) == "missing_model", answer
    code, answer = router.json("GET", "/v1/videos/video_unknown")
    assert code == 404 and error_code(answer) == "video_not_found", answer
    # Synchronous music still answers the WAV directly and records no job.
    code, wav = router.request("POST", "/v1/audio/music",
                               {"model": "ace", "prompt": "lofi"})
    assert code == 200 and wav == b"RIFF-sync-ace", wav


def check_async_music(router):
    code, job = router.json("POST", "/v1/audio/music",
                            {"model": "ace", "prompt": "lofi", "async": True})
    assert code == 202 and job["id"] == "music_1", job
    job_id = job["id"]
    # An open job keeps the worker busy.
    code, answer = router.json("POST", "/models/unload", {"model": "ace"})
    assert code == 409, answer
    code, answer = router.json("GET", "/v1/audio/music/%s/content" % job_id)
    assert code == 409 and error_code(answer) == "not_ready", answer
    seen = []
    for _ in range(5):
        code, status = router.json("GET", "/v1/audio/music/" + job_id)
        assert code in (200, 202) and status["model"] == "ace", status
        seen.append(status["status"])
        if status["status"] == "completed":
            break
    assert seen == ["queued", "in_progress", "completed"], seen
    code, wav = router.request("GET", "/v1/audio/music/%s/content" % job_id)
    assert code == 200 and wav == b"RIFFace", wav
    # The second music model numbers its jobs from 1 as well: the newest
    # create owns the bare id, ?model= picks the worker explicitly.
    code, other = router.json("POST", "/v1/audio/music",
                              {"model": "ace2", "prompt": "jazz", "async": True})
    assert code == 202 and other["id"] == job_id, other
    code, status = router.json("GET", "/v1/audio/music/" + job_id)
    assert status["model"] == "ace2", status
    code, status = router.json("GET", "/v1/audio/music/%s?model=ace" % job_id)
    assert code == 200 and status["model"] == "ace", status
    code, answer = router.json("GET", "/v1/audio/music/music_99")
    assert code == 404 and error_code(answer) == "not_found", answer
    code, answer = router.json("GET", "/v1/audio/music/music_1?model=nope")
    assert code == 404 and error_code(answer) == "model_not_found", answer
    code, answer = router.json("POST", "/v1/audio/music/" + job_id)
    assert code == 405, answer
    code, answer = router.json("DELETE", "/v1/audio/music/" + job_id)
    assert code == 405, answer
    # ace's job finished: it may unload, and a later read must not start it
    # again (music jobs live in the worker's memory only).
    code, answer = router.json("POST", "/models/unload", {"model": "ace"})
    assert code == 200, answer
    code, answer = router.json("GET", "/v1/audio/music/%s?model=ace" % job_id)
    assert code == 404 and error_code(answer) == "not_found", answer
    assert router.loaded("ace") is False
    # ace2's job is still open, so ace2 stays busy until it completes.
    code, answer = router.json("POST", "/models/unload", {"model": "ace2"})
    assert code == 409, answer
    for _ in range(3):
        code, status = router.json("GET", "/v1/audio/music/" + job_id)
    assert status["status"] == "completed" and status["model"] == "ace2", status
    code, answer = router.json("POST", "/models/unload", {"model": "ace2"})
    assert code == 200, answer


def check_comfy_embeddings(binary, tmp):
    comfy_dir = os.path.join(tmp, "comfy")
    os.makedirs(comfy_dir, exist_ok=True)
    router = Router(binary, tmp, ["--comfy-dir", comfy_dir])
    try:
        for path in ("/embeddings", "/api/embeddings"):
            code, answer = router.json("GET", path)
            assert code == 200 and answer == [], (path, answer)
        code, answer = router.json("POST", "/embeddings", {"model": "emb", "input": "x"})
        assert code == 200 and answer["model"] == "emb", answer
        code, answer = router.json("POST", "/v1/embeddings", {"model": "emb", "input": "x"})
        assert code == 200 and answer["path"] == "/v1/embeddings", answer
    except BaseException:
        print(router.tail())
        raise
    finally:
        router.stop()


def main():
    if "--fake-worker" in sys.argv:
        return serve_fake_worker(sys.argv)
    binary = sys.argv[1]
    tmp = tempfile.mkdtemp(prefix="gufo_pass_")
    router = Router(binary, tmp)
    try:
        check_post_routes(router)
        check_get_routes(router)
        check_unchanged(router)
        check_async_music(router)
    except BaseException:
        print(router.tail())
        raise
    finally:
        router.stop()
    check_comfy_embeddings(binary, os.path.join(tmp, "c"))
    shutil.rmtree(tmp, ignore_errors=True)
    print("router_passthrough_test: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
