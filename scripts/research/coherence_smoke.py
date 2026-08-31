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
import threading
import time
import urllib.error
import urllib.request

DEFAULT_PROMPT = (
    "\u041a\u0430\u043a\u0430\u044f \u0442\u044b \u043c\u043e\u0434\u0435\u043b\u044c? "
    "\u041e\u0442\u0432\u0435\u0442\u044c \u043e\u0434\u043d\u0438\u043c \u043f\u0440\u0435\u0434\u043b\u043e\u0436\u0435\u043d\u0438\u0435\u043c."
)

def wait_ready(base: str, proc: subprocess.Popen[bytes], timeout_s: float = 180.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
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
    ap.add_argument("--rpc", default=None,
                    help="optional comma-separated ggml-rpc endpoint list")
    ap.add_argument("--tensor-split", default=None,
                    help="per-device tensor split; defaults to equal weights")
    ap.add_argument("--cache-type", default=None)
    ap.add_argument("--cache-type-k", default=None)
    ap.add_argument("--cache-type-v", default=None)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--ubatch", type=int, default=256)
    ap.add_argument("--gpu-layers", default="999",
                    help="value passed to -ngl (integer, auto, or all)")
    ap.add_argument("--fit", choices=("on", "off"), default="off")
    ap.add_argument("--fit-target", default=None,
                    help="optional per-device MiB margin passed to -fitt")
    ap.add_argument("--n-cpu-moe", type=int, default=None,
                    help="keep routed MoE weights of the first N layers on CPU")
    ap.add_argument("--spec-type", default="none")
    ap.add_argument("--spec-draft-n-max", type=int, default=2)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--n-predict", type=int, default=48)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--log", default=None, help="server log path (optional)")
    ap.add_argument("--live-log", action="store_true",
                    help="tee llama-server output to stderr while retaining --log")
    ap.add_argument("--startup-timeout", type=float, default=180.0)
    ap.add_argument("--request-timeout", type=float, default=120.0)
    args = ap.parse_args()

    port = args.port or (58000 + (os.getpid() % 1000))
    tensor_split = args.tensor_split or ",".join("1" for _ in args.dev.split(","))
    cmd = [
        args.server_bin,
        "-m", args.model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--flash-attn", "on",
        "-np", "1",
        "-c", str(args.ctx),
        "-b", str(args.batch),
        "-ub", str(args.ubatch),
        "--cache-type-k", args.cache_type_k or args.cache_type,
        "--cache-type-v", args.cache_type_v or args.cache_type,
        "-ngl", str(args.gpu_layers),
        "--seed", "42",
        "--no-warmup",
        "--cache-ram", "0",
        "--ctx-checkpoints", "0",
        "-dev", args.dev,
        "-sm", "layer",
        "-ts", tensor_split,
        "--spec-type", args.spec_type,
        "--spec-draft-n-max", str(args.spec_draft_n_max),
        "-fit", args.fit,
    ]
    if args.rpc:
        dev_arg = cmd.index("-dev")
        cmd[dev_arg:dev_arg] = ["--rpc", args.rpc]
    if args.fit_target:
        cmd.extend(["-fitt", args.fit_target])
    if args.n_cpu_moe is not None:
        cmd.extend(["-ncmoe", str(args.n_cpu_moe)])
    log_fh = open(args.log, "wb") if args.log else None
    print(f"[smoke] starting server: {' '.join(cmd)}", file=sys.stderr)
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    capture_output = args.live_log
    server_output = subprocess.PIPE if capture_output else (log_fh or subprocess.DEVNULL)
    proc = subprocess.Popen(cmd, stdout=server_output, stderr=subprocess.STDOUT,
                            creationflags=creationflags)
    tee_thread = None
    if capture_output:
        def tee_server_output() -> None:
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                if log_fh:
                    log_fh.write(chunk)
                    log_fh.flush()
                sys.stderr.buffer.write(chunk)
                sys.stderr.buffer.flush()

        tee_thread = threading.Thread(target=tee_server_output, daemon=True)
        tee_thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        if not wait_ready(base, proc, args.startup_timeout):
            print(f"[smoke] ERROR: server not ready (exit={proc.poll()})", file=sys.stderr)
            return 2
        body = json.dumps({
            "prompt": args.prompt,
            "n_predict": args.n_predict,
            "temperature": 0.0,
            "seed": 42,
        }).encode("utf-8")
        req = urllib.request.Request(base + "/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=args.request_timeout) as resp:
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
        if tee_thread:
            tee_thread.join(timeout=5)
        if log_fh:
            log_fh.close()
        print("[smoke] server exited", file=sys.stderr)

if __name__ == "__main__":
    sys.exit(main())
