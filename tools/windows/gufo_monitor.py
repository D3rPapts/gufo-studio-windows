r"""gufo_monitor.py: a local web page that shows what the gufo servers are doing.

    python tools\windows\gufo_monitor.py [--port 8090] [--servers 8080,8081] [--logs DIR]
    then open http://127.0.0.1:8090

Per gufo server: each session's phase from /slots
(idle, prefill with progress, decode with tokens), throughput from /metrics, and the last
requests from its log when --logs names a folder of gufo-PORT-*.log files (cache hit or
full recompute, time to first token). System: GPU
utilisation (busiest engine, usually Compute), dedicated/shared GPU memory, free RAM, commit,
page-file use and pages written, sampled every second by one long-running PowerShell.
Read-only: it never starts, stops or sends requests to a model.
"""
import argparse
import json
import re
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOGS = None
PORTS = {8080: "", 8081: "", 8082: ""}
PAGE = Path(__file__).resolve().with_name("gufo_monitor.html")

# One PowerShell samples the counters each second and prints one JSON line.
SAMPLER = r"""
$ErrorActionPreference = 'SilentlyContinue'
$names = '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage',
  '\GPU Engine(*)\Utilization Percentage','\Memory\Available MBytes','\Memory\Committed Bytes',
  '\Memory\Commit Limit','\Memory\Pages Output/sec','\Paging File(_Total)\% Usage'
while ($true) {
  $s = (Get-Counter $names).CounterSamples
  $eng = @{}
  foreach ($x in ($s | Where-Object Path -like '*utilization percentage')) {
    $t = $x.InstanceName -replace '.*engtype_', ''
    $eng[$t] = $eng[$t] + $x.CookedValue
  }
  $o = [ordered]@{
    t = [DateTimeOffset]::Now.ToUnixTimeMilliseconds()
    gpu_ded = ($s | Where-Object Path -like '*dedicated usage' | Measure-Object CookedValue -Maximum).Maximum
    gpu_shared = ($s | Where-Object Path -like '*shared usage' | Measure-Object CookedValue -Maximum).Maximum
    engines = $eng
    ram_avail_mb = ($s | Where-Object Path -like '*available mbytes').CookedValue
    commit = ($s | Where-Object Path -like '*committed bytes').CookedValue
    commit_limit = ($s | Where-Object Path -like '*commit limit').CookedValue
    pages_out = ($s | Where-Object Path -like '*pages output/sec').CookedValue
    pagefile_pct = ($s | Where-Object Path -like '*% usage').CookedValue
  }
  $o | ConvertTo-Json -Compress
  [Console]::Out.Flush()
}
"""

latest = {"system": None}
lock = threading.Lock()


def sampler():
    while True:
        proc = subprocess.Popen(["powershell", "-NoProfile", "-Command", SAMPLER], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, creationflags=0x08000000)
        for line in proc.stdout:
            try:
                value = json.loads(line)
            except ValueError:
                continue
            with lock:
                latest["system"] = value
        time.sleep(2)  # PowerShell exited; start it again


def get(url, timeout=1.5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def metrics(text):
    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.rpartition(" ")
            try:
                out[name.split(":")[-1]] = float(value)
            except ValueError:
                pass
    return out


FIELDS = re.compile(r"(\w+)=(\S+)")


def recent_requests(port, count=8):
    """The last completed chat requests in the newest log of this port."""
    if LOGS is None:
        return []
    logs = sorted(LOGS.glob("gufo-%d-*.log" % port))
    if not logs:
        return []
    with logs[-1].open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 400_000))
        lines = f.read().decode("utf-8", "replace").splitlines()
    rows = []
    for line in reversed(lines):
        if "event=completed method=POST" not in line or "prompt_tokens=" not in line:
            continue
        r = dict(FIELDS.findall(line))
        rows.append({k: r.get(k) for k in (
            "request", "prompt_tokens", "prefill_tokens", "cached_tokens", "generated_tokens", "cache",
            "cache_miss_reason", "queue_ms", "ttft_ms", "duration_ms", "prefill_tps", "decode_tps",
            "acceptance_pct", "plan", "finish")} | {"time": line[11:19]})
        if len(rows) == count:
            break
    return rows


def server(port):
    base = "http://127.0.0.1:%d" % port
    try:
        slots = json.loads(get(base + "/slots"))
    except Exception:
        return {"port": port, "name": PORTS[port], "up": False}
    out = {"port": port, "name": PORTS[port], "up": True, "slots": slots, "recent": recent_requests(port)}
    try:
        out["metrics"] = metrics(get(base + "/metrics"))
    except Exception:
        out["metrics"] = {}
    out["model"] = slots[0].get("model") if slots else None
    return out


def snapshot():
    with lock:
        system = latest["system"]
    return {"t": time.time(), "system": system, "servers": [server(p) for p in PORTS]}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api"):
            body, kind = json.dumps(snapshot()).encode(), "application/json"
        elif self.path in ("/", "/index.html"):
            body, kind = PAGE.read_bytes(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open it from other devices on the LAN")
    ap.add_argument("--servers", default="8080,8081,8082",
                    help="gufo ports, each optionally PORT=label (8080=Qwen,8081=Gemma 26B)")
    ap.add_argument("--logs", type=Path, help="folder of the servers' gufo-PORT-*.log files")
    a = ap.parse_args()
    global LOGS
    LOGS = a.logs
    PORTS.clear()
    for item in a.servers.split(","):
        port, _, label = item.partition("=")
        PORTS[int(port)] = label.strip()
    threading.Thread(target=sampler, daemon=True).start()
    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    print("gufo monitor on http://127.0.0.1:%d  (Ctrl+C to stop)" % a.port, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
