"""Contract tests for switchable models (local section + [remote/<id>] twin)
and POST /router/placement, over real HTTP.

Run through ctest as router_placement_contract_test with the gufo binary as
argv[1]. Reuses the fake remote router of router_remote_test.py; the local
models are this file with --fake-worker (chat with an optional "SLEEP n"
prompt, and video jobs), so no model and no GPU is involved.

Cases: start placement from the preset (/v1/models, /router/status,
GET /router/placement, --autoload only for twins that start local); routing
by placement; request errors; switching to remote while a local request
runs (it finishes locally, new requests go remote, the idle local worker
unloads even while relays are open); switching back while a relay runs;
{"all": ...}; video job reads follow the job across a switch; the gpu slot
of a twin placed remote; 508 remote_loop for a relayed request.
"""

import json
import os
import random
import shutil
import string
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import router_remote_test as rr  # noqa: E402

THIS = os.path.abspath(__file__)


def serve_fake_worker(argv):
    """A local model: chat (sleeps on "SLEEP n") and video jobs."""
    port = int(argv[argv.index("--port") + 1])
    model = argv[argv.index("--served-model-name") + 1]
    jobs = set()  # video jobs stay in_progress until DELETE

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
            path = self.path.partition("?")[0]
            if path.startswith("/v1/videos/"):
                job = path[len("/v1/videos/"):]
                if job not in jobs:
                    return self.send(404, {"error": {"code": "video_not_found"},
                                           "where": "local"})
                return self.send(200, {"id": job, "object": "video",
                                       "status": "in_progress", "where": "local"})
            self.send(200, {"status": "ok", "model": model, "where": "local"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            fields = json.loads(self.rfile.read(length) or b"{}")
            prompt = fields.get("prompt", "")
            if "SLEEP" in prompt:
                time.sleep(float(prompt.split("SLEEP")[1].split()[0]))
            if self.path == "/v1/videos":
                job = "video_" + "".join(random.choice(string.ascii_lowercase)
                                         for _ in range(20))
                jobs.add(job)
                return self.send(202, {"id": job, "object": "video",
                                       "status": "queued", "where": "local"})
            self.send(200, {"model": model, "where": "local"})

        def do_DELETE(self):
            job = self.path.partition("?")[0][len("/v1/videos/"):]
            known = job in jobs
            jobs.discard(job)
            self.send(200 if known else 404, {"id": job, "deleted": known,
                                              "where": "local"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.serve_forever()


def write_preset(path, remote_port, python):
    url = "http://127.0.0.1:%d" % remote_port
    local = ("[external/%s]\ncommand = %s\nargs = %s --fake-worker\n"
             "served-model-name = %s\n")
    sections = [
        local % ("twin", python, THIS, "twin"),
        "[remote/twin]\nurl = %s\napi-key-env = %s\ngpu-slot = studio\n"
        % (url, rr.KEY_ENV),
        # Remote section first, start placement remote.
        "[remote/twin2]\nurl = %s\napi-key-env = %s\ngpu-slot = studio\n"
        "placement = remote\n" % (url, rr.KEY_ENV),
        local % ("twin2", python, THIS, "twin2"),
        local % ("plain", python, THIS, "plain"),
        "[remote/far]\nurl = %s\nmodality = llm\napi-key-env = %s\n"
        % (url, rr.KEY_ENV),
    ]
    with open(path, "w") as handle:
        handle.write("\n".join(sections))


def chat(router, model, prompt="", headers=None):
    return router.json("POST", "/v1/chat/completions",
                       {"model": model, "prompt": prompt,
                        "messages": [{"role": "user", "content": "x"}]},
                       headers)


def where(answer):
    return answer.get("where", "remote") if isinstance(answer, dict) else None


def place(router, body):
    return router.json("POST", "/router/placement", body)


def placement(router):
    code, answer = router.json("GET", "/router/placement")
    assert code == 200, (code, answer)
    return answer["models"]


def in_background(fn):
    result = {}

    def run():
        try:
            result["value"] = fn()
        except BaseException as error:  # reported by the caller
            result["error"] = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, result


def check_start_state(router, remote):
    models = router.models()
    twin, twin2 = models["twin"], models["twin2"]
    assert twin["switchable"] is True and twin["placement"] == "local", twin
    assert "remote" not in twin, twin  # other routers must not see a relay
    assert twin["remote_url"] == "http://127.0.0.1:%d" % remote.port, twin
    assert twin2["placement"] == "remote" and "remote" in twin2, twin2
    assert "switchable" not in models["plain"] and "switchable" not in models["far"]
    assert models["far"]["remote"]["url"].endswith(":%d" % remote.port)
    state = placement(router)
    assert set(state) == {"twin", "twin2"}, state
    assert state["twin2"]["remote_reachable"] is True, state
    status = router.status()["remote"]["models"]
    assert status["twin"]["placement"] == "local" and status["twin"]["switchable"]
    assert status["twin2"]["placement"] == "remote"
    assert status["far"]["placement"] == "remote" and "switchable" not in status["far"]
    # --autoload: the twin that starts local loads here, twin2 does not.
    rr.wait_for(lambda: router.models()["twin"]["loaded"], 20, "autoload of twin")
    assert placement(router)["twin2"]["local_status"] == "unloaded"
    assert not any(json.loads(e["body"])["model"] == "twin2"
                   for e in remote.entries("/models/load"))


def check_routing(router, remote):
    code, answer = chat(router, "twin")
    assert code == 200 and where(answer) == "local", (code, answer)
    remote.clear()
    code, answer = chat(router, "twin2")
    assert code == 200 and where(answer) == "remote", (code, answer)
    assert len(remote.entries("/v1/chat/completions")) == 1


def check_errors(router):
    cases = [
        (b"not json", 400, "invalid_json"),
        ({"model": "twin"}, 400, "invalid_placement"),
        ({"model": "twin", "placement": "elsewhere"}, 400, "invalid_placement"),
        ({"model": "twin", "all": "local", "placement": "local"}, 400,
         "invalid_placement"),
        ({}, 400, "invalid_placement"),
        ({"all": "sideways"}, 400, "invalid_placement"),
        ({"model": "nope", "placement": "local"}, 404, "model_not_found"),
        ({"model": "plain", "placement": "remote"}, 409, "not_switchable"),
        ({"model": "far", "placement": "local"}, 409, "not_switchable"),
    ]
    for body, status, code in cases:
        got, answer = router.json("POST", "/router/placement", body,
                                  {"Content-Type": "application/json"})
        assert got == status and rr.code_of(answer) == code, (body, got, answer)
    got, _ = router.json("DELETE", "/router/placement")
    assert got == 405, got
    # Nothing moved.
    state = placement(router)
    assert state["twin"]["placement"] == "local"
    assert state["twin2"]["placement"] == "remote"


def check_switch_to_remote(router, remote):
    assert router.models()["twin"]["loaded"]
    began = time.monotonic()
    running, result = in_background(lambda: chat(router, "twin", "SLEEP 3"))
    time.sleep(0.6)
    code, answer = place(router, {"model": "twin", "placement": "remote"})
    assert code == 200 and answer["changed"] == ["twin"], (code, answer)
    assert answer["models"]["twin"]["placement"] == "remote", answer
    assert answer["models"]["twin"]["local_busy"] is True, answer
    # Same placement again changes nothing.
    code, answer = place(router, {"model": "twin", "placement": "remote"})
    assert code == 200 and answer["changed"] == [], answer
    # New requests go remote at once, while the local one still runs.
    remote.clear()
    code, answer = chat(router, "twin")
    assert code == 200 and where(answer) == "remote", (code, answer)
    assert time.monotonic() - began < 2.5
    assert "remote" in router.models()["twin"]
    # A long relay is open while the local request ends: it must not keep
    # the local worker loaded.
    relay, relay_result = in_background(lambda: chat(router, "twin", "SLEEP 8"))
    running.join(10)
    assert "error" not in result, result
    code, answer = result["value"]
    assert code == 200 and where(answer) == "local", (code, answer)
    assert time.monotonic() - began >= 3
    rr.wait_for(lambda: placement(router)["twin"]["local_status"] == "unloaded",
                6, "the local worker of twin to unload")
    state = placement(router)["twin"]
    assert state["remote_busy"] is True and relay.is_alive(), state
    with open(router.log_path, "rb") as handle:
        assert b"event=placement_unload model=twin" in handle.read()
    relay.join(15)
    assert relay_result["value"][0] == 200 and where(relay_result["value"][1]) == "remote"
    assert placement(router)["twin"]["remote_busy"] is False


def check_switch_to_local(router, remote):
    relay, relay_result = in_background(lambda: chat(router, "twin", "SLEEP 2"))
    time.sleep(0.5)
    code, answer = place(router, {"model": "twin", "placement": "local"})
    assert code == 200 and answer["changed"] == ["twin"], answer
    code, answer = chat(router, "twin")  # loads the local worker again
    assert code == 200 and where(answer) == "local", (code, answer)
    relay.join(10)
    code, answer = relay_result["value"]
    assert code == 200 and where(answer) == "remote", (code, answer)
    assert "remote" not in router.models()["twin"]


def check_all(router, remote):
    code, answer = place(router, {"all": "remote"})
    assert code == 200 and answer["changed"] == ["twin"], answer
    assert {m["placement"] for m in answer["models"].values()} == {"remote"}
    code, answer = place(router, {"all": "local"})
    assert code == 200 and sorted(answer["changed"]) == ["twin", "twin2"], answer
    code, answer = chat(router, "twin2")  # first local load of twin2
    assert code == 200 and where(answer) == "local", (code, answer)
    assert router.models()["twin2"]["loaded"]


def check_job_reads_follow_the_job(router, remote):
    # A local job, then the model moves away: its reads still go local.
    code, created = router.json("POST", "/v1/videos", {"model": "twin", "prompt": "x"})
    assert code == 202 and where(created) == "local", (code, created)
    local_job = created["id"]
    code, answer = place(router, {"model": "twin", "placement": "remote"})
    assert code == 200
    time.sleep(3)  # the open job keeps the local worker loaded (grace 2 s)
    assert placement(router)["twin"]["local_status"] == "loaded"
    # A job created now runs remote.
    code, created = router.json("POST", "/v1/videos", {"model": "twin", "prompt": "HOLD"})
    assert code == 202 and where(created) == "remote", (code, created)
    remote_job = created["id"]
    code, answer = router.json("GET", "/v1/videos/" + local_job)
    assert code == 200 and where(answer) == "local", (code, answer)
    assert placement(router)["twin"]["local_status"] == "loaded"
    code, answer = router.json("DELETE", "/v1/videos/" + local_job)
    assert code == 200 and where(answer) == "local" and answer["deleted"], answer
    # The DELETE released the local job: the worker unloads.
    rr.wait_for(lambda: placement(router)["twin"]["local_status"] == "unloaded",
                6, "the local worker to unload after its job")
    # Back to local: the remote job's reads still go remote.
    code, answer = place(router, {"model": "twin", "placement": "local"})
    assert code == 200
    code, answer = router.json("GET", "/v1/videos/" + remote_job)
    assert code == 200 and answer["status"] == "in_progress", (code, answer)
    assert where(answer) == "remote", answer
    code, answer = router.json("DELETE", "/v1/videos/" + remote_job)
    assert code == 200 and answer["deleted"] is True, (code, answer)
    rr.wait_for(lambda: not placement(router)["twin"]["remote_busy"], 5,
                "the remote job to be released")
    # Wait for the job's gpu slot to be free again.
    rr.wait_for(lambda: not router.status()["remote"]["gpu_slots"]
                .get("studio", {}).get("held"), 10, "the studio slot")


def check_gpu_slot(router, remote):
    place(router, {"all": "remote"})
    remote.clear()
    threads = [in_background(lambda m=m: chat(router, m, "SLEEP 1"))
               for m in ("twin", "twin2")]
    for thread, result in threads:
        thread.join(15)
        assert result["value"][0] == 200, result
    entries = sorted(remote.entries("/v1/chat/completions"), key=lambda e: e["start"])
    assert len(entries) == 2, entries
    assert entries[1]["start"] >= entries[0]["end"] - 0.05, entries


def check_loop_guard(router):
    # Placed remote here: a relayed request is not relayed again.
    code, answer = chat(router, "twin", headers={"X-Gufo-Forwarded": "gufo-other"})
    assert code == 508 and rr.code_of(answer) == "remote_loop", (code, answer)
    place(router, {"model": "twin", "placement": "local"})
    code, answer = chat(router, "twin", headers={"X-Gufo-Forwarded": "gufo-other"})
    assert code == 200 and where(answer) == "local", (code, answer)


def main():
    if "--fake-worker" in sys.argv:
        return serve_fake_worker(sys.argv)
    binary = sys.argv[1]
    tmp = tempfile.mkdtemp(prefix="gufo_placement_")
    remote = rr.FakeRemote(models=("twin", "twin2", "far"))
    preset = os.path.join(tmp, "placement.preset")
    write_preset(preset, remote.port, os.environ.get("COMFY_TEST_PYTHON") or sys.executable)
    router = rr.Router(binary, tmp, preset, ["--autoload"])
    try:
        for name, case in (
                ("start state", lambda: check_start_state(router, remote)),
                ("routing", lambda: check_routing(router, remote)),
                ("errors", lambda: check_errors(router)),
                ("switch to remote", lambda: check_switch_to_remote(router, remote)),
                ("switch to local", lambda: check_switch_to_local(router, remote)),
                ("all", lambda: check_all(router, remote)),
                ("job reads", lambda: check_job_reads_follow_the_job(router, remote)),
                ("gpu slot", lambda: check_gpu_slot(router, remote)),
                ("loop guard", lambda: check_loop_guard(router))):
            case()
            print("router_placement_test: %s ok" % name, flush=True)
    except BaseException:
        print(router.tail())
        raise
    finally:
        router.stop()
        remote.stop()
    shutil.rmtree(tmp, ignore_errors=True)
    print("router_placement_test: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
