#!/usr/bin/env python3
"""Short agent-workload benchmark for local llama-server builds.

The benchmark is intentionally small: it runs a few coding-agent style prompts
against an OpenAI-compatible llama-server endpoint and writes CSV/JSONL results.
It can either start a local ROCm build of llama-server or use an already running
server.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


TASKS_QUICK = [
    {
        "id": "triage_diff",
        "title": "Agent triage from git status and diff notes",
        "prompt": """You are maintaining a local llama.cpp fork focused on ROCm.
Current status:
- gui/llama_gui.py changed: multimodal controls, build log export
- ggml/src/ggml-cpu/ggml-cpu.c changed: local fallback tweak
- docs were replaced with local fork docs
- models/*.gguf are local and must not be committed

Task: give a compact risk triage for the next implementation step. Return:
1. files to avoid touching,
2. files safe to edit,
3. two tests to run.
Keep it under 140 words.""",
    },
    {
        "id": "review_bug",
        "title": "Code review of a small server-launch bug",
        "prompt": """Review this Python snippet from a GUI server launcher:

def build_cmd(server, model, extra):
    cmd = [server, "-m", model, "--port", "8080"]
    if extra:
        cmd.extend(extra.split(" "))
    return cmd

The GUI runs on Windows and Linux. Users may pass quoted args like:
--chat-template-kwargs "{\\"preserve_thinking\\": true}"

Task: identify the bug, give a minimal fix, and mention one edge case. Keep it concise.""",
    },
    {
        "id": "rocm_log_plan",
        "title": "ROCm build-log diagnosis",
        "prompt": """A Windows ROCm llama.cpp build for RX 9070 XT fails:

lld-link: error: undefined symbol: __kmpc_fork_call
CMakeCache.txt contains GGML_OPENMP:BOOL=ON
HIP SDK path is C:\\Program Files\\AMD\\ROCm\\7.1
GPU target should be gfx1201

Task: propose the next commands/settings for a GUI build manager. Keep it as a short numbered list.""",
    },
    {
        "id": "patch_sim",
        "title": "Small patch simulation",
        "prompt": """Write a tiny unified diff for this function so MTP is disabled when vision is enabled:

def server_extra_args(vision_enabled, mtp_enabled):
    args = []
    if mtp_enabled:
        args += ["--spec-type", "mtp", "--spec-draft-n-max", "3"]
    if vision_enabled:
        args += ["--mmproj", "mmproj.gguf"]
    return args

Expected behavior: if both are true, raise ValueError. Return only the diff.""",
    },
]


TASKS_FULL = TASKS_QUICK + [
    {
        "id": "implementation_plan",
        "title": "Implementation planning",
        "prompt": """We want to port llama.cpp PR #22673 MTP support into a fork with TurboQuant and a PyQt GUI.
Constraints:
- focus on ROCm builds
- keep GUI stable
- MTP is text-only initially
- benchmark baseline before and after

Task: provide a concise phase plan with rollback points. Keep it under 180 words.""",
    },
    {
        "id": "config_compare",
        "title": "Compare two launch configs",
        "prompt": """Compare these two configs for Qwen3.6 on a 16 GB AMD GPU:

A: --flash-attn on -np 1 -c 32768 --cache-type-k q8_0 --cache-type-v q8_0
B: --flash-attn on -np 1 -c 65536 --cache-type-k q4_0 --cache-type-v q4_0 --spec-type ngram-mod

Task: say which is safer for baseline measurement and why. Keep it brief.""",
    },
]


def default_server_bin() -> Path | None:
    candidates = [
        ROOT / "build-rocm" / "bin" / "llama-server.exe",
        ROOT / "build-rocm" / "bin" / "Release" / "llama-server.exe",
        ROOT / "build" / "bin" / "llama-server.exe",
        ROOT / "build" / "bin" / "Release" / "llama-server.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def default_model() -> Path | None:
    candidates = [
        ROOT / "models" / "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
        ROOT / "models" / "Qwen3.6-27B-Q3_K_S.gguf",
        ROOT / "models" / "Qwen3.5-9B-Q6_K.gguf",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def rocm_env() -> dict[str, str]:
    env = os.environ.copy()
    rocm_bin = Path(r"C:\Program Files\AMD\ROCm\7.1\bin")
    if rocm_bin.exists():
        env["PATH"] = str(rocm_bin) + os.pathsep + env.get("PATH", "")
        env.setdefault("HIP_PATH", str(rocm_bin.parent))
    return env


def http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    if not raw:
        return None
    return json.loads(raw)


def wait_for_server(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        for endpoint in ("/health", "/v1/models"):
            try:
                http_json("GET", base_url + endpoint, timeout=2.0)
                return
            except Exception as exc:  # noqa: BLE001 - status probing is best-effort
                last_error = str(exc)
        time.sleep(1.0)
    raise TimeoutError(f"server did not become ready within {timeout_s:.0f}s; last error: {last_error}")


def terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def start_server(args: argparse.Namespace) -> subprocess.Popen[str]:
    server_bin = Path(args.server_bin) if args.server_bin else default_server_bin()
    model = Path(args.model) if args.model else default_model()
    if server_bin is None or not server_bin.exists():
        raise FileNotFoundError("llama-server.exe not found; pass --server-bin or build ROCm first")
    if model is None or not model.exists():
        raise FileNotFoundError("model GGUF not found; pass --model")
    if args.port == 0:
        args.port = find_free_port(args.host)

    cmd = [
        str(server_bin),
        "-m", str(model),
        "--host", args.host,
        "--port", str(args.port),
        "-ngl", str(args.gpu_layers),
        "--flash-attn", "on" if args.flash_attn else "off",
        "-np", str(args.parallel),
        "-c", str(args.ctx_size),
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "--cache-type-k", args.cache_type_k,
        "--cache-type-v", args.cache_type_v,
    ]
    if args.no_warmup:
        cmd.append("--no-warmup")
    if args.disable_thinking and "--chat-template-kwargs" not in args.server_extra:
        cmd.extend([
            "--chat-template-kwargs",
            json.dumps({"enable_thinking": False, "preserve_thinking": False}, separators=(",", ":")),
        ])
    if args.server_extra:
        cmd.extend(shlex.split(args.server_extra, posix=(os.name != "nt")))

    log_dir = Path(args.out_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = log_dir / f"{args.label}.server.log"
    log_file = server_log.open("w", encoding="utf-8")
    print("Starting server:", " ".join(cmd))
    print("Server log:", server_log)
    return subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=rocm_env(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )


def run_task(base_url: str, task: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "model": args.api_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise coding agent. Answer directly and avoid long preambles.",
            },
            {"role": "user", "content": task["prompt"]},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": False,
    }

    started = time.perf_counter()
    error = ""
    response: dict[str, Any] | None = None
    content = ""
    try:
        response = http_json("POST", base_url + "/v1/chat/completions", payload, timeout=args.request_timeout)
        message = response["choices"][0].get("message", {})
        content = message.get("content") or message.get("reasoning_content") or ""
    except Exception as exc:  # noqa: BLE001 - benchmark records failures as rows
        error = repr(exc)
    wall_s = time.perf_counter() - started

    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    timings = response.get("timings", {}) if isinstance(response, dict) else {}
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    total_tokens = usage.get("total_tokens")
    completion_tps = None
    if isinstance(completion_tokens, int) and wall_s > 0:
        completion_tps = completion_tokens / wall_s

    return {
        "label": args.label,
        "task_id": task["id"],
        "title": task["title"],
        "wall_s": round(wall_s, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tps_wall": round(completion_tps, 4) if completion_tps is not None else None,
        "response_chars": len(content),
        "error": error,
        "timings": timings,
        "response_preview": content[:500],
    }


def write_results(rows: list[dict[str, Any]], out_dir: Path, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{label}.jsonl"
    csv_path = out_dir / f"{label}.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = [
        "label", "task_id", "title", "wall_s", "prompt_tokens",
        "completion_tokens", "total_tokens", "completion_tps_wall",
        "response_chars", "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    total_completion = sum(row.get("completion_tokens") or 0 for row in rows)
    total_wall = sum(row.get("wall_s") or 0.0 for row in rows)
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    if total_wall > 0 and total_completion > 0:
        print(f"Aggregate completion TPS by wall time: {total_completion / total_wall:.2f}")


def parse_args() -> argparse.Namespace:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="Short coding-agent benchmark for llama-server")
    parser.add_argument("--label", default=f"rocm-baseline-{timestamp}", help="result file label")
    parser.add_argument("--out-dir", default=str(ROOT / "build_logs" / "agent-workload"), help="output directory")
    parser.add_argument("--tasks", choices=["quick", "full"], default="quick", help="prompt set")
    parser.add_argument("--runs", type=int, default=1, help="repeat each task N times")

    parser.add_argument("--no-start", action="store_true", help="use an already running server")
    parser.add_argument("--server-bin", default=None, help="path to llama-server executable")
    parser.add_argument("--model", default=None, help="path to GGUF model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="server port; 0 picks a free port when starting a server")
    parser.add_argument("--api-model", default="local-model")
    parser.add_argument("--server-extra", default="", help="extra llama-server args, e.g. '--spec-type mtp --spec-draft-n-max 3'")

    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--ctx-size", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=2048)
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q8_0")
    parser.add_argument("--flash-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True,
                        help="add chat-template kwargs to keep Qwen thinking off for short benchmark answers")

    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument("--keep-server", action="store_true", help="do not stop server started by this script")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    tasks = TASKS_QUICK if args.tasks == "quick" else TASKS_FULL
    proc: subprocess.Popen[str] | None = None

    try:
        if not args.no_start:
            proc = start_server(args)
            base_url = f"http://{args.host}:{args.port}"
            wait_for_server(base_url, args.startup_timeout)
        else:
            if args.port == 0:
                args.port = 8080
            base_url = f"http://{args.host}:{args.port}"
            wait_for_server(base_url, 10.0)

        rows: list[dict[str, Any]] = []
        for run_idx in range(args.runs):
            for task in tasks:
                print(f"[{run_idx + 1}/{args.runs}] {task['id']} ...", flush=True)
                row = run_task(base_url, task, args)
                row["run"] = run_idx + 1
                rows.append(row)
                if row["error"]:
                    print(f"  error: {row['error']}")
                else:
                    print(f"  {row['wall_s']:.2f}s, completion_tokens={row['completion_tokens']}")
        write_results(rows, out_dir, args.label)
        return 0 if not any(row["error"] for row in rows) else 2
    finally:
        if proc is not None and not args.keep_server:
            terminate_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
