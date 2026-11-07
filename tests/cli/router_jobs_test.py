"""Contract tests for the router's job API (--jobs-dir) over real HTTP.

Run through ctest as router_jobs_contract_test with the gufo binary as
argv[1]. The router starts the fake workers of router_comfy_test.py (image,
video, speech) and posts webhooks to a local test receiver, which fails the
first delivery so the retry is exercised as well.
"""

import base64
import hashlib
import hmac
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, "router_comfy_test.py")
SECRET = "whsec_" + base64.b64encode(b"jobs-test-secret").decode()


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Receiver:
    """Webhook receiver: answers `fail_first` deliveries with 500."""

    def __init__(self, fail_first=1):
        self.lock = threading.Lock()
        self.deliveries = []
        self.fail_first = fail_first
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                with receiver.lock:
                    receiver.deliveries.append(
                        {"path": self.path, "headers": dict(self.headers.items()),
                         "body": body})
                    status = 500 if len(receiver.deliveries) <= receiver.fail_first else 200
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def wait(self, count, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if len(self.deliveries) >= count:
                    return list(self.deliveries)
            time.sleep(0.05)
        raise AssertionError("got %d webhook deliveries, wanted %d" % (len(self.deliveries), count))


class Router:
    def __init__(self, binary, tmp, receiver_port):
        self.port = free_port()
        self.tmp = tmp
        preset = os.path.join(tmp, "jobs.preset")
        python = os.environ.get("COMFY_TEST_PYTHON") or sys.executable
        sections = []
        for model in ("qwen-image-2.1", "minimax-h3", "qwen3-tts"):
            sections.append("[external/%s]\ncommand = %s\nargs = %s --fake-worker\n"
                            "served-model-name = %s\n" % (model, python, FAKE, model))
        with open(preset, "w") as handle:
            handle.write("\n".join(sections))
        env = {k: v for k, v in os.environ.items()
               if k not in {"HOST", "PORT", "GUFO_HOST", "GUFO_PORT"}}
        env["COMFY_FAKE_LOG"] = os.path.join(tmp, "fake.jsonl")
        args = [binary, "router", "--models-preset", preset, "--host", "127.0.0.1",
                "--port", str(self.port), "--models-max", "1",
                "--jobs-dir", os.path.join(tmp, "jobs"),
                "--jobs-webhook-allow", "127.0.0.1:%d" % receiver_port,
                "--jobs-webhook-secret", SECRET,
                "--jobs-webhook-retry-seconds", "1,1"]
        self.log = open(os.path.join(tmp, "router.log"), "ab")
        self.proc = subprocess.Popen(args, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                if self.request("GET", "/health")[0] == 200:
                    return
            except OSError:
                time.sleep(0.05)
        raise AssertionError("router did not start: " + self.tail())

    def tail(self):
        with open(os.path.join(self.tmp, "router.log"), "rb") as handle:
            return handle.read()[-3000:].decode(errors="replace")

    def stop(self):
        self.proc.terminate()
        self.proc.wait(20)
        self.log.close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
            headers = dict(headers or {}, **{"Content-Type": "application/json"})
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        payload = response.read()
        conn.close()
        return response.status, payload

    def json(self, method, path, body=None, headers=None):
        status, payload = self.request(method, path, body, headers)
        return status, (json.loads(payload) if payload else None)

    def wait(self, job_id, status, timeout=60):
        deadline = time.monotonic() + timeout
        job = None
        while time.monotonic() < deadline:
            code, job = self.json("GET", "/v1/jobs/" + job_id)
            assert code == 200, job
            if job["status"] == status:
                return job
            assert job["status"] in ("queued", "in_progress"), job
            time.sleep(0.1)
        raise AssertionError("job %s did not reach %s: %s" % (job_id, status, job))


def image_job(prompt, **extra):
    body = {"request": {"path": "/v1/images/generations",
                        "body": {"model": "qwen-image-2.1", "prompt": prompt,
                                 "size": "64x48", "response_format": "b64_json"}}}
    body.update(extra)
    return body


def check_image_with_webhook(router, receiver):
    hook = "http://127.0.0.1:%d/hook?shot=A1" % receiver.port
    code, job = router.json("POST", "/v1/jobs",
                            image_job("a red fox", webhook_url=hook, metadata={"shot": "A1"}))
    assert code == 202, job
    done = router.wait(job["id"], "completed")
    assert done["progress"] == 100, done
    assert done["metadata"] == {"shot": "A1"}, done
    names = [f["name"] for f in done["result"]["files"]]
    assert names == ["image-1.png", "response.json"], names
    code, png = router.request("GET", done["result"]["files"][0]["url"])
    assert code == 200 and png[:8] == b"\x89PNG\r\n\x1a\n", png[:16]
    assert done["result"]["response"]["data"][0]["url"].endswith("/files/image-1.png")
    # First delivery answered 500, the retry one second later 200.
    deliveries = receiver.wait(2)
    for delivery in deliveries:
        assert delivery["path"] == "/hook?shot=A1", delivery["path"]
        headers = {k.lower(): v for k, v in delivery["headers"].items()}
        message = "%s.%s.%s" % (headers["webhook-id"], headers["webhook-timestamp"],
                                delivery["body"].decode())
        key = base64.b64decode(SECRET[len("whsec_"):])
        expected = "v1," + base64.b64encode(
            hmac.new(key, message.encode(), hashlib.sha256).digest()).decode()
        assert headers["webhook-signature"] == expected, headers
        assert headers["webhook-id"] == "msg_%s_completed" % job["id"], headers
        payload = json.loads(delivery["body"])
        assert payload["type"] == "job.completed", payload
        assert payload["data"]["id"] == job["id"], payload
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = router.json("GET", "/v1/jobs/" + job["id"])[1]["webhook"]
        if state["state"] == "delivered":
            break
        time.sleep(0.1)
    assert state["state"] == "delivered" and state["attempts"] == 2, state
    return job["id"]


def check_video(router):
    code, job = router.json("POST", "/v1/jobs", {"request": {
        "path": "/v1/videos",
        "body": {"model": "minimax-h3", "prompt": "a wave", "size": "64x64", "seconds": "1"}}})
    assert code == 202, job
    done = router.wait(job["id"], "completed")
    assert done["video_id"].startswith("video_minimax"), done
    code, mp4 = router.request("GET", "/v1/jobs/%s/files/video.mp4" % job["id"])
    assert code == 200 and mp4 == b"FAKEMP4-minimax-h3", mp4


def check_speech(router):
    code, job = router.json("POST", "/v1/jobs", {"request": {
        "path": "/v1/audio/speech", "body": {"model": "qwen3-tts", "input": "Hallo"}}})
    assert code == 202, job
    done = router.wait(job["id"], "completed")
    code, wav = router.request("GET", done["result"]["files"][0]["url"])
    assert code == 200 and wav[:4] == b"RIFF", wav[:8]


def check_cancel(router):
    # The fake worker sleeps 2 s for SLOW prompts.
    code, job = router.json("POST", "/v1/jobs", image_job("SLOW cat"))
    assert code == 202, job
    router.wait(job["id"], "in_progress")
    code, queued = router.json("POST", "/v1/jobs", image_job("never runs"))
    code, cancelled = router.json("DELETE", "/v1/jobs/" + queued["id"])
    assert code == 200 and cancelled["status"] == "cancelled", cancelled
    code, answer = router.json("POST", "/v1/jobs/%s/cancel" % job["id"])
    assert code == 200, answer
    router.wait(job["id"], "cancelled")
    code, again = router.json("DELETE", "/v1/jobs/" + job["id"])
    assert code == 409, again


def check_rejections(router, receiver):
    for url in ("http://127.0.0.1:%d/hook" % (receiver.port + 1),
                "http://169.254.169.254/latest/meta-data",
                "https://127.0.0.1:%d/hook" % receiver.port,
                "http://user:pw@127.0.0.1:%d/hook" % receiver.port,
                "file:///C:/Windows/win.ini"):
        code, answer = router.json("POST", "/v1/jobs", image_job("x", webhook_url=url))
        assert code == 400 and answer["error"]["code"] == "webhook_not_allowed", (url, answer)
    code, answer = router.json("POST", "/v1/jobs", {"request": {
        "path": "/v1/models", "body": {"model": "qwen-image-2.1"}}})
    assert code == 400, answer
    code, answer = router.json("POST", "/v1/jobs", {"request": {
        "path": "/v1/images/generations", "body": {"model": "nope", "prompt": "x"}}})
    assert code == 404, answer
    code, answer = router.json("GET", "/v1/jobs/job_00000000000000000000")
    assert code == 404, answer


def check_idempotency(router):
    headers = {"Idempotency-Key": "shot-7"}
    code, first = router.json("POST", "/v1/jobs", image_job("same"), headers)
    assert code == 202, first
    code, again = router.json("POST", "/v1/jobs", image_job("same"), headers)
    assert code == 200 and again["id"] == first["id"], again
    code, other = router.json("POST", "/v1/jobs", image_job("different"), headers)
    assert code == 409, other
    router.wait(first["id"], "completed")


def main():
    binary = sys.argv[1]
    tmp = tempfile.mkdtemp(prefix="gufo_jobs_")
    receiver = Receiver(fail_first=1)
    router = Router(binary, tmp, receiver.port)
    try:
        first = check_image_with_webhook(router, receiver)
        check_video(router)
        check_speech(router)
        check_cancel(router)
        check_rejections(router, receiver)
        check_idempotency(router)
        code, listing = router.json("GET", "/v1/jobs?limit=100")
        total = len(listing["data"])
    except BaseException:
        print(router.tail())
        raise
    finally:
        router.stop()
    # Jobs and their files survive a restart.
    router = Router(binary, tmp, receiver.port)
    try:
        code, listing = router.json("GET", "/v1/jobs?limit=100")
        assert len(listing["data"]) == total, listing
        job = router.json("GET", "/v1/jobs/" + first)[1]
        assert job["status"] == "completed", job
        code, png = router.request("GET", job["result"]["files"][0]["url"])
        assert code == 200 and png[:4] == b"\x89PNG", png[:8]
    except BaseException:
        print(router.tail())
        raise
    finally:
        router.stop()
    receiver.server.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)
    print("router_jobs_test: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
