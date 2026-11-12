"""Contract tests for placement = pool (a switchable model served by this
router and the remote router per request), over real HTTP.

Run through ctest as router_pool_contract_test with the gufo binary as
argv[1]. Reuses the fake remote router of router_remote_test.py and the fake
local worker of router_placement_test.py (chat with an optional "SLEEP n"
prompt), so no model and no GPU is involved.

Cases: status fields (/v1/models placement "pool", "inflight", "activity",
no "remote" block; /router/status and /router/placement "pool"); idle ->
local; local busy -> remote; both busy -> wait for the first free; a
relayed request (X-Gufo-Forwarded) stays local; the remote's reported
activity (busy, gpu_busy) keeps requests off it; remote gone -> local only,
including a request that was already on its way there.
"""

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import router_placement_test as rp  # noqa: E402
import router_remote_test as rr  # noqa: E402


def write_preset(path, remote_port, python):
    with open(path, "w") as handle:
        handle.write(
            "[external/twin]\ncommand = %s\nargs = %s --fake-worker\n"
            "served-model-name = twin\n"
            "lora = a=/l/a.safetensors;b=/l/b.safetensors\n\n"
            "[remote/twin]\nurl = http://127.0.0.1:%d\napi-key-env = %s\n"
            "queue-timeout-seconds = 20\nplacement = pool\n"
            % (python, rp.THIS, remote_port, rr.KEY_ENV))


def remote_chats(remote):
    return len(remote.entries("/v1/chat/completions"))


def router_entry(router, remote):
    return router.status()["remote"]["routers"]["127.0.0.1:%d" % remote.port]


def check_status(router, remote):
    rr.wait_for(lambda: router.models()["twin"]["loaded"], 20, "autoload of twin")
    # The pool uses the remote once a probe listed the model there.
    rr.wait_for(lambda: router_entry(router, remote)["models"]["twin"] == "loaded",
                15, "the probe to list twin on the remote")
    twin = router.models()["twin"]
    assert twin["switchable"] is True and twin["placement"] == "pool", twin
    assert "remote" not in twin, twin  # the partner sees it served here
    assert twin["inflight"] == {"local": 0, "remote": 0}, twin
    activity = twin["activity"]
    assert activity["busy"] is False and activity["gpu_busy"] is False, activity
    assert activity["others_busy"] is False, activity
    status = router.status()["remote"]["models"]["twin"]
    assert status["placement"] == "pool" and status["switchable"], status
    pool = status["pool"]
    for key in ("local_inflight", "remote_inflight", "waiting", "local_picks",
                "remote_picks", "local_gpu_busy"):
        assert key in pool, (key, pool)
    state = rp.placement(router)["twin"]
    assert state["placement"] == "pool" and "pool" in state, state
    # pool is a valid target for POST /router/placement as well.
    code, answer = rp.place(router, {"model": "twin", "placement": "pool"})
    assert code == 200 and answer["changed"] == [], (code, answer)


def check_idle_goes_local(router, remote):
    remote.clear()
    code, answer = rp.chat(router, "twin")
    assert code == 200 and rp.where(answer) == "local", (code, answer)
    assert remote_chats(remote) == 0


def check_busy_goes_remote(router, remote):
    remote.clear()
    running, result = rp.in_background(lambda: rp.chat(router, "twin", "SLEEP 3"))
    time.sleep(0.6)
    began = time.monotonic()
    code, answer = rp.chat(router, "twin")
    assert code == 200 and rp.where(answer) == "remote", (code, answer)
    assert time.monotonic() - began < 2, "the remote request waited"
    assert router.models()["twin"]["inflight"]["local"] == 1
    running.join(10)
    assert result["value"][0] == 200 and rp.where(result["value"][1]) == "local"
    assert router.status()["remote"]["models"]["twin"]["pool"]["remote_picks"] >= 1


def check_both_busy_waits(router, remote):
    remote.clear()
    first, first_result = rp.in_background(lambda: rp.chat(router, "twin", "SLEEP 3"))
    time.sleep(0.5)
    second, second_result = rp.in_background(lambda: rp.chat(router, "twin", "SLEEP 3"))
    time.sleep(0.5)
    began = time.monotonic()
    waiting, waiting_result = rp.in_background(lambda: rp.chat(router, "twin"))
    time.sleep(0.5)
    assert router.status()["remote"]["models"]["twin"]["pool"]["waiting"] == 1
    waiting.join(15)
    code, answer = waiting_result["value"]
    assert code == 200, (code, answer)
    assert time.monotonic() - began >= 1.5, "did not wait for a free machine"
    for thread, result in ((first, first_result), (second, second_result)):
        thread.join(10)
        assert result["value"][0] == 200, result
    assert rp.where(first_result["value"][1]) == "local"
    assert rp.where(second_result["value"][1]) == "remote"
    assert router.status()["remote"]["models"]["twin"]["pool"]["waiting"] == 0


def check_forwarded_stays_local(router, remote):
    remote.clear()
    running, result = rp.in_background(lambda: rp.chat(router, "twin", "SLEEP 2"))
    time.sleep(0.5)
    code, answer = rp.chat(router, "twin", headers={"X-Gufo-Forwarded": "gufo-other"})
    assert code == 200 and rp.where(answer) == "local", (code, answer)
    assert remote_chats(remote) == 0
    running.join(10)


def check_remote_activity(router, remote):
    for flags in ({"busy": True, "gpu_busy": False, "others_busy": False},
                  {"busy": False, "gpu_busy": True, "others_busy": True}):
        remote.activity["twin"] = flags
        rr.wait_for(lambda: router_entry(router, remote)["activity"].get("twin") == flags,
                    15, "the probe to read %s" % flags)
        remote.clear()
        running, result = rp.in_background(lambda: rp.chat(router, "twin", "SLEEP 2"))
        time.sleep(0.5)
        began = time.monotonic()
        code, answer = rp.chat(router, "twin")
        assert code == 200 and rp.where(answer) == "local", (flags, code, answer)
        assert time.monotonic() - began >= 1, "did not wait for the local worker"
        assert remote_chats(remote) == 0, flags
        running.join(10)
    remote.activity.clear()
    rr.wait_for(lambda: router_entry(router, remote)["activity"] == {}, 15,
                "the probe to clear the activity")


def lora_chat(router, lora, headers=None):
    return router.json("POST", "/v1/chat/completions",
                       {"model": "twin", "prompt": "", "loras": [{"name": lora}],
                        "messages": [{"role": "user", "content": "x"}]},
                       headers)


def check_lora_side(router, remote):
    # One "lora = a=..;b=.." line registers both names (gufo 2.0 form).
    code, caps = router.json("GET", "/router/capabilities")
    assert code == 200 and caps["models"]["twin"]["loras"] == ["a", "b"], caps
    remote.clear()
    code, answer = lora_chat(router, "b")
    assert code == 200 and rp.where(answer) == "local", (code, answer)
    code, answer = lora_chat(router, "b", {"X-Gufo-Pool": "local"})
    assert code == 200 and rp.where(answer) == "local", (code, answer)
    # Not here, partner unknown (no capabilities route): goes remote ...
    code, answer = lora_chat(router, "zzz")
    assert code == 200 and rp.where(answer) == "remote", (code, answer)
    # ... unless the client pins this side: then 400, never silently remote.
    remote.clear()
    code, answer = lora_chat(router, "zzz", {"X-Gufo-Pool": "local"})
    assert code == 400, (code, answer)
    assert answer["error"]["code"] == "pool_side_cannot_run", answer
    assert "zzz" in answer["error"]["message"], answer
    assert remote_chats(remote) == 0


def check_remote_failure_retries_local(router, remote):
    # The partner answers 502 (its worker did not start) while the local
    # worker is busy: the request is served here once the local worker is
    # free instead of handing the 502 to the client, and the partner gets
    # no request of this model for 60 s.
    remote.clear()
    remote.chat_fail = 502
    try:
        running, result = rp.in_background(lambda: rp.chat(router, "twin", "SLEEP 2"))
        time.sleep(0.5)
        code, answer = rp.chat(router, "twin")
        assert code == 200 and rp.where(answer) == "local", (code, answer)
        assert remote_chats(remote) == 1, remote.entries()
        running.join(10)
        pool = router.status()["remote"]["models"]["twin"]["pool"]
        assert "remote_failed_s_ago" in pool and pool["retried_local"] >= 1, pool
    finally:
        remote.chat_fail = None
    # Held: local busy again, the next request waits for the local worker.
    remote.clear()
    running, result = rp.in_background(lambda: rp.chat(router, "twin", "SLEEP 2"))
    time.sleep(0.5)
    code, answer = rp.chat(router, "twin")
    assert code == 200 and rp.where(answer) == "local", (code, answer)
    assert remote_chats(remote) == 0, remote.entries()
    running.join(10)


def check_remote_gone(router, remote):
    # The remote goes away while the router still lists it as reachable:
    # a request sent there is decided again and served here.
    running, result = rp.in_background(lambda: rp.chat(router, "twin", "SLEEP 2"))
    time.sleep(0.5)
    remote.stop()
    code, answer = rp.chat(router, "twin")
    assert code == 200 and rp.where(answer) == "local", (code, answer)
    running.join(10)
    rr.wait_for(lambda: router_entry(router, remote)["reachable"] is False, 15,
                "the remote to count as unreachable")
    code, answer = rp.chat(router, "twin")
    assert code == 200 and rp.where(answer) == "local", (code, answer)
    assert router.models()["twin"]["inflight"]["remote"] == 0


def main():
    if "--fake-worker" in sys.argv:
        return rp.serve_fake_worker(sys.argv)
    binary = sys.argv[1]
    tmp = tempfile.mkdtemp(prefix="gufo_pool_")
    remote = rr.FakeRemote(models=("twin",))
    remote.states["twin"] = "loaded"
    preset = os.path.join(tmp, "pool.preset")
    write_preset(preset, remote.port,
                 os.environ.get("COMFY_TEST_PYTHON") or sys.executable)
    router = rr.Router(binary, tmp, preset, ["--autoload"])
    try:
        for name, case in (
                ("status", lambda: check_status(router, remote)),
                ("idle goes local", lambda: check_idle_goes_local(router, remote)),
                ("busy goes remote", lambda: check_busy_goes_remote(router, remote)),
                ("both busy waits", lambda: check_both_busy_waits(router, remote)),
                ("forwarded stays local",
                 lambda: check_forwarded_stays_local(router, remote)),
                ("remote activity", lambda: check_remote_activity(router, remote)),
                ("lora side", lambda: check_lora_side(router, remote)),
                ("remote failure retries local",
                 lambda: check_remote_failure_retries_local(router, remote)),
                ("remote gone", lambda: check_remote_gone(router, remote))):
            case()
            print("router_pool_test: %s ok" % name, flush=True)
    except BaseException:
        print(router.tail())
        raise
    finally:
        router.stop()
        try:
            remote.stop()
        except Exception:
            pass
    shutil.rmtree(tmp, ignore_errors=True)
    print("router_pool_test: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
