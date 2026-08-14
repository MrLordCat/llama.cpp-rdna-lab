#!/usr/bin/env python3
"""Direct short-prompt coherence smoke for llama-server.

Usage:
  python scripts/research/coherence_smoke.py --server-bin <path> --model <gguf> \
      --dev <dev list> --cache-type <t> [--spec-type draft-mtp] [--prompt "..."]
Starts the server, sends one /completion request, prints the answer, stops
the server gracefully (CTRL_BREAK_EVENT), then exits 0.
ASCII-only logs on stderr; the model answer goes to stdout as UTF-8.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_PROMPT = (
    "\u041a\u0430\u043a\u0430\u044f \u0442\u044b \u043c\u043e\u0434\u0435\u043b\u044c? "
    "\u041e\u0442\u0432\u0435\u0442\u044c \u043e\u0434\u043d\u0438\u043c \u043f\u0440\u0435\u0434\u043b\u043e\u0436\u0435\u043d\u0438\u0435\u043c."
)

def wait_ready(base: str, timeout_s: float = 180.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.URLError:
            pass
        except OSError:
            pass
        time.sleep(0.5)
    return False

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--cache-type", default=None)
    ap.add_argument("--cache-type-k", default=None)
    ap.add_argument("--cache-type-v", default=None)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--spec-type", default="none")
    ap.add_argument("--spec-draft-n-max", type=int, default=2)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--n-predict", type=int, default=48)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--log", default=None, help="server log path (optional)")
    args = ap.parse_args()

    port = args.port or (58000 + (os.getpid() % 1000))
    cmd = [
        args.server_bin,
        "-m", args.model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--flash-attn", "on",
        "-np", "1",
        "-c", str(args.ctx),
        "-b", "512",
        "-ub", "256",
        "--cache-type-k", args.cache_type_k or args.cache_type,
        "--cache-type-v", args.cache_type_v or args.cache_type,
        "-ngl", "999",
        "--seed", "42",
        "--no-warmup",
        "--cache-ram", "0",
        "--ctx-checkpoints", "0",
        "-dev", args.dev,
        "-sm", "layer",
        "-ts", "1,1",
        "--spec-type", args.spec_type,
        "--spec-draft-n-max", str(args.spec_draft_n_max),
        "-fit", "off",
    ]
    log_fh = open(args.log, "wb") if args.log else subprocess.DEVNULL
    print(f"[smoke] starting server: {' '.join(cmd)}", file=sys.stderr)
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                            creationflags=creationflags)
    base = f"http://127.0.0.1:{port}"
    try:
        if not wait_ready(base):
            print("[smoke] ERROR: server not ready", file=sys.stderr)
            return 2
        body = json.dumps({
            "prompt": args.prompt,
            "n_predict": args.n_predict,
            "temperature": 0.0,
            "seed": 42,
        }).encode("utf-8")
        req = urllib.request.Request(base + "/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        print("[smoke] response:", file=sys.stderr)
        content = data.get("content", "")
        print(content)
        if not content.strip():
            print(f"[smoke] WARNING: empty content; raw json = {json.dumps(data)[:800]}", file=sys.stderr)
        print("[smoke] timings:", file=sys.stderr)
        timings = data.get("timings", {})
        for key in ("prompt_n", "predicted_n", "prompt_ms", "predicted_ms"):
            if key in timings:
                print(f"[smoke]   {key} = {timings[key]}", file=sys.stderr)
        return 0
    finally:
        print("[smoke] stopping server (CTRL_BREAK)", file=sys.stderr)
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
            proc.wait(timeout=90)
        except Exception:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
        if log_fh is not subprocess.DEVNULL:
            log_fh.close()
        print("[smoke] server exited", file=sys.stderr)

if __name__ == "__main__":
    sys.exit(main())
