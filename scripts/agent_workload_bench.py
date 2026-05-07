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
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from itertools import product


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


def find_background_llama_servers() -> list[str]:
    """Return a list of running llama-server process ids as strings."""
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return []
            pids: list[str] = []
            for raw in result.stdout.splitlines():
                line = raw.strip()
                if not line or "No tasks are running" in line:
                    continue
                cols = [c.strip().strip('"') for c in line.split('","')]
                if len(cols) >= 2 and cols[0].lower() == "llama-server.exe":
                    pids.append(cols[1])
            return pids

        result = subprocess.run(
            ["pgrep", "-x", "llama-server"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        return [pid.strip() for pid in result.stdout.splitlines() if re.fullmatch(r"\d+", pid.strip())]
    except Exception:
        return []


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

    tps_values = [float(row["completion_tps_wall"]) for row in rows if row.get("completion_tps_wall") is not None]
    if tps_values:
        print(f"Mean task TPS: {statistics.mean(tps_values):.2f}")
        print(f"Median task TPS: {statistics.median(tps_values):.2f}")
        if len(tps_values) > 1:
            print(f"Task TPS stdev: {statistics.pstdev(tps_values):.4f}")


def aggregate_completion_tps(rows: list[dict[str, Any]]) -> float:
    total_completion = sum(row.get("completion_tokens") or 0 for row in rows)
    total_wall = sum(row.get("wall_s") or 0.0 for row in rows)
    return (total_completion / total_wall) if total_wall > 0 and total_completion > 0 else 0.0


def parse_int_csv(values: str) -> list[int]:
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def parse_text_csv(values: str) -> list[str]:
    return [v.strip() for v in values.split(",") if v.strip()]


def update_model_preset_file(
    preset_file: Path,
    model_path: Path,
    best: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if not preset_file.exists():
        print(f"WARNING: preset file not found: {preset_file}")
        return

    try:
        data = json.loads(preset_file.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: failed to read preset file: {exc}")
        return

    presets = data.get("presets")
    if not isinstance(presets, list):
        print("WARNING: preset file has invalid format (missing presets array)")
        return

    model_name = model_path.name
    escaped = re.escape(model_name)
    name = f"AutoTune 32k+ {model_name}"
    kv_map = {
        "f32": 0,
        "f16": 1,
        "bf16": 2,
        "q8_0": 3,
        "q4_0": 7,
    }
    preset = {
        "pattern": escaped,
        "name": name,
        "ctx": int(best["ctx_size"]),
        "batch_size": int(best["batch_size"]),
        "ubatch_size": int(best["ubatch_size"]),
        "gpu_layers": int(args.gpu_layers),
        "parallel": int(args.parallel),
        "flash_attn": bool(args.flash_attn),
        "kv_cache": kv_map.get(str(best["kv"]).lower(), 3),
        "notes": (
            "Auto-generated by agent_workload_bench.py autotune for large context "
            f"(>= {args.autotune_min_ctx}). spec={best['spec_mode']}"
        ),
    }

    updated = False
    for idx, item in enumerate(presets):
        if isinstance(item, dict) and item.get("name") == name:
            presets[idx] = preset
            updated = True
            break
    if not updated:
        presets.insert(0, preset)

    preset_file.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    print(f"Updated preset: {name} in {preset_file}")


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
    parser.add_argument(
        "--background-server-policy",
        choices=["warn", "fail", "ignore"],
        default="warn",
        help="what to do if llama-server is already running before benchmark start",
    )

    parser.add_argument("--autotune", action="store_true", help="run large-context parameter sweep")
    parser.add_argument("--autotune-min-ctx", type=int, default=32768, help="minimum context for autotune")
    parser.add_argument("--autotune-ctx-values", default="32768,49152,65536", help="comma-separated ctx values")
    parser.add_argument("--autotune-batch-values", default="1024,2048,4096", help="comma-separated batch values")
    parser.add_argument("--autotune-ubatch-values", default="1024,2048,4096", help="comma-separated ubatch values")
    parser.add_argument("--autotune-kv-values", default="q8_0,q4_0", help="comma-separated kv cache values")
    parser.add_argument("--autotune-spec-values", default="none,ngram-mod", help="comma-separated speculative modes")
    parser.add_argument("--autotune-ngram-min", type=int, default=48)
    parser.add_argument("--autotune-ngram-match", type=int, default=24)
    parser.add_argument("--autotune-ngram-max", type=int, default=64)
    parser.add_argument("--autotune-max-configs", type=int, default=48, help="safety cap for sweep size")
    parser.add_argument("--autotune-update-preset", action="store_true", help="write best config into model presets file")
    parser.add_argument(
        "--autotune-preset-file",
        default=str(ROOT / "gui" / "model_presets.json"),
        help="preset JSON file path for --autotune-update-preset",
    )
    return parser.parse_args()


def run_suite(args: argparse.Namespace, tasks: list[dict[str, str]]) -> list[dict[str, Any]]:
    proc: subprocess.Popen[str] | None = None
    try:
        if not args.no_start:
            existing = find_background_llama_servers()
            if existing:
                msg = f"Detected already running llama-server process(es): {', '.join(existing)}"
                if args.background_server_policy == "fail":
                    raise RuntimeError(msg)
                if args.background_server_policy == "warn":
                    print(f"WARNING: {msg}")
                    print("Results may be skewed by shared GPU/CPU load.")

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
        return rows
    finally:
        if proc is not None and not args.keep_server:
            terminate_process(proc)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    tasks = TASKS_QUICK if args.tasks == "quick" else TASKS_FULL
    if not args.autotune:
        try:
            rows = run_suite(args, tasks)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            print("Stop background server(s) or rerun with --background-server-policy warn/ignore")
            return 3
        write_results(rows, out_dir, args.label)
        return 0 if not any(row["error"] for row in rows) else 2

    ctx_values = [v for v in parse_int_csv(args.autotune_ctx_values) if v >= args.autotune_min_ctx]
    batch_values = parse_int_csv(args.autotune_batch_values)
    ubatch_values = parse_int_csv(args.autotune_ubatch_values)
    kv_values = parse_text_csv(args.autotune_kv_values)
    spec_values = [v.lower() for v in parse_text_csv(args.autotune_spec_values)]

    configs = list(product(ctx_values, batch_values, ubatch_values, kv_values, spec_values))
    if not configs:
        print("ERROR: empty autotune config list")
        return 4
    if len(configs) > args.autotune_max_configs:
        print(f"Autotune config count {len(configs)} exceeds --autotune-max-configs {args.autotune_max_configs}")
        return 4

    base_server_extra = args.server_extra.strip()
    summaries: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for idx, (ctx_size, batch_size, ubatch_size, kv_type, spec_mode) in enumerate(configs, start=1):
        run_args = argparse.Namespace(**vars(args))
        run_args.ctx_size = int(ctx_size)
        run_args.batch_size = int(batch_size)
        run_args.ubatch_size = int(ubatch_size)
        run_args.cache_type_k = kv_type
        run_args.cache_type_v = kv_type
        run_args.label = f"{args.label}-cfg{idx:02d}"

        extra_bits: list[str] = []
        if base_server_extra:
            extra_bits.append(base_server_extra)
        if spec_mode == "ngram-mod":
            extra_bits.append("--spec-type ngram-mod")
            extra_bits.append(f"--spec-ngram-mod-n-min {args.autotune_ngram_min}")
            extra_bits.append(f"--spec-ngram-mod-n-match {args.autotune_ngram_match}")
            extra_bits.append(f"--spec-ngram-mod-n-max {args.autotune_ngram_max}")
        run_args.server_extra = " ".join(extra_bits)

        print(
            f"Autotune [{idx}/{len(configs)}]: ctx={ctx_size}, b={batch_size}, ub={ubatch_size}, "
            f"kv={kv_type}, spec={spec_mode}"
        )
        try:
            rows = run_suite(run_args, tasks)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return 3

        write_results(rows, out_dir, run_args.label)
        agg_tps = aggregate_completion_tps(rows)
        has_error = any(row.get("error") for row in rows)
        summary = {
            "label": run_args.label,
            "ctx_size": ctx_size,
            "batch_size": batch_size,
            "ubatch_size": ubatch_size,
            "kv": kv_type,
            "spec_mode": spec_mode,
            "aggregate_tps": round(agg_tps, 4),
            "mean_task_tps": round(statistics.mean([float(r["completion_tps_wall"]) for r in rows if r.get("completion_tps_wall") is not None]), 4),
            "errors": int(has_error),
        }
        summaries.append(summary)

        if not has_error:
            if best is None or summary["aggregate_tps"] > best["aggregate_tps"]:
                best = summary

    summary_json = out_dir / f"{args.label}-autotune-summary.json"
    summary_csv = out_dir / f"{args.label}-autotune-summary.csv"
    summary_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fields = ["label", "ctx_size", "batch_size", "ubatch_size", "kv", "spec_mode", "aggregate_tps", "mean_task_tps", "errors"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")
    if best:
        print(
            "BEST: "
            f"ctx={best['ctx_size']} b={best['batch_size']} ub={best['ubatch_size']} "
            f"kv={best['kv']} spec={best['spec_mode']} aggregate_tps={best['aggregate_tps']:.2f}"
        )
        if args.autotune_update_preset:
            model_path = Path(args.model) if args.model else default_model()
            if model_path:
                update_model_preset_file(Path(args.autotune_preset_file), model_path, best, args)
    else:
        print("No successful autotune runs.")

    return 0 if best else 2


if __name__ == "__main__":
    raise SystemExit(main())
