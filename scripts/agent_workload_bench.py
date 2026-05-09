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
import hashlib
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
HISTORY_DIR = ROOT / "build_logs" / "agent-workload"
HISTORY_CSV_NAME = "BENCH_HISTORY.csv"
HISTORY_MD_NAME = "BENCH_HISTORY.md"

HISTORY_FIELDS = [
    "timestamp",
    "run_id",
    "build_id",
    "build_name",
    "build_backend",
    "mode",
    "label",
    "model",
    "is_mtp_model",
    "tasks",
    "runs",
    "ctx",
    "batch",
    "ubatch",
    "kv_k",
    "kv_v",
    "spec_mode",
    "gpu_layers",
    "parallel",
    "flash_attn",
    "max_tokens",
    "temperature",
    "top_p",
    "aggregate_tps",
    "mean_task_tps",
    "errors",
    "best_config",
    "jsonl_file",
    "csv_file",
    "summary_file",
    "server_log_file",
    "is_group_best",
]


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

# ---------------------------------------------------------------------------
# V2 task set — more realistic agentic-flow prompts.
# Designed to produce longer, varied responses (target ~300-600 output tokens)
# so benchmark TPS is comparable to real assistant usage rather than
# micro-bursts of <=160 tokens.
# ---------------------------------------------------------------------------
TASKS_V2 = [
    {
        "id": "v2_code_review",
        "title": "Full code review of a build manager module",
        "prompt": """You are a senior engineer reviewing a pull request for a PyQt6 GUI application that manages
CMake builds of llama.cpp on Windows with ROCm/CUDA/Vulkan backends.

Here is a simplified extract from build_manager.py:

```python
class BuildManager(QObject):
    build_started = pyqtSignal(str)
    build_finished = pyqtSignal(int, str)
    build_log = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.build_dir = None

    def start_build(self, backend, extra_cmake_args=None):
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self.build_log.emit("Build already running")
            return
        self.build_dir = Path("build-" + backend.lower())
        cmake_args = [
            "cmake", "-B", str(self.build_dir),
            "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        if extra_cmake_args:
            cmake_args.extend(extra_cmake_args)
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.finished.connect(self._on_finished)
        self.process.start(cmake_args[0], cmake_args[1:])
        self.build_started.emit(backend)

    def _on_stdout(self):
        data = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self.build_log.emit(data)

    def _on_finished(self, exit_code, _exit_status):
        self.build_finished.emit(exit_code, str(self.build_dir))

    def cancel(self):
        if self.process:
            self.process.kill()
```

Review this code. Cover:
1. Correctness issues (race conditions, missing error handling, resource leaks)
2. Windows-specific concerns for ROCm builds (paths, process environment, Ninja availability)
3. Qt signal/slot lifecycle pitfalls
4. At least two concrete improvement suggestions with code examples

Be thorough but focused.""",
    },
    {
        "id": "v2_write_function",
        "title": "Write a build version registry helper",
        "prompt": """You are implementing a build registry for a PyQt6 GUI that tracks multiple llama.cpp builds
(ROCm, CUDA, Vulkan, CPU). The registry is stored in a JSON file: build_versions.json.

Each entry looks like:
{
  "id": "bld-20260508-abc123",
  "label": "ROCm gfx1201",
  "backend": "rocm",
  "bin_dir": "C:/repos/llama.cpp-with-GUI/build-rocm/bin",
  "created_at": "2026-05-08T11:38:03",
  "updated_at": "2026-05-08T11:38:03",
  "runnable": true,
  "notes": ""
}

Write a Python class `BuildRegistry` with these methods:
- `__init__(registry_path: Path)` — loads JSON from disk, validates, handles missing file
- `all_builds() -> list[dict]` — returns all entries sorted newest-first by updated_at
- `runnable_builds(backend: str | None = None) -> list[dict]` — only entries where runnable=True, optionally filtered by backend
- `register(entry: dict) -> str` — upserts by id (generates one if missing), saves, returns id
- `mark_runnable(build_id: str, runnable: bool) -> None` — updates flag and updated_at, saves
- `_save()` — atomic write (write to temp, rename) so partial writes don't corrupt registry

Include type hints and basic docstrings. Handle edge cases: corrupt JSON, missing keys, concurrent read/write on Windows.""",
    },
    {
        "id": "v2_debug_trace",
        "title": "Diagnose ROCm server crash from log",
        "prompt": """Analyze this llama-server crash log from a Windows ROCm build targeting gfx1201 (RX 9070 XT).
Identify the root cause and propose a fix or workaround:

```
[2026-05-08 14:21:03] INFO: llama_new_context_with_model: n_ctx = 65536
[2026-05-08 14:21:03] INFO: llama_new_context_with_model: flash_attn = 1
[2026-05-08 14:21:03] INFO: ggml_cuda_init: GGML_CUDA_FORCE_MMQ = 0
[2026-05-08 14:21:03] INFO: ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS = 0
[2026-05-08 14:21:03] INFO: ggml_cuda_init: found 1 ROCm device(s):
[2026-05-08 14:21:03] INFO:   Device 0: AMD Radeon RX 9070 XT, compute capability 12.0, VMM: no
[2026-05-08 14:21:03] INFO: llama_kv_cache_unified_init: kv_size = 3600, type_k = q4_0, type_v = q4_0
[2026-05-08 14:21:04] ERROR: HIP error: out of memory at ggml-cuda.cu:1847
[2026-05-08 14:21:04] FATAL: ggml_cuda_op_mul_mat: src1->extra == NULL
Aborted (core dumped)

Build flags: GGML_HIP=ON AMDGPU_TARGETS=gfx1201 GGML_HIP_NO_VMM=ON
Model: Qwen3.6-27B-Q3_K_S.gguf (~18 GB)
GPU VRAM: 16 GB
Server args: --n-gpu-layers 999 --flash-attn --ctx-size 65536 --cache-type-k q4_0 --cache-type-v q4_0 -np 1
```

Provide:
1. Root cause analysis (what ran out, why)
2. Quick fix: minimal server arg changes to fit within 16 GB
3. Formula or estimation to calculate safe ctx-size for this model/VRAM
4. Long-term suggestion for the GUI (auto-detect safe context)""",
    },
    {
        "id": "v2_refactor_plan",
        "title": "Refactor plan for monolithic GUI file",
        "prompt": """A PyQt6 GUI application lives in a single 2150-line file `gui/llama_gui.py` with 6 tabs:
Launch Server, Inference, Download Models, Build & Setup, Installed Builds, System Info.

The team wants to modularize it. Current structure:
- LlamaCppGUI(QMainWindow) — all 6 tabs inline as methods
- ServerThread(QThread), InferenceThread(QThread), UpdateForkThread(QThread) — nested inside main module
- QSettings persistence scattered across 40+ methods
- ~300 lines of build detection logic duplicated between Build tab and Launch tab

The modularization must:
1. Keep backward compatibility with existing QSettings keys (no user setting loss)
2. Not break ROCm build detection on Windows (path parsing is Windows-specific)
3. Allow incremental migration (can't rewrite everything at once)
4. Make it easy to write unit tests for server command generation

Provide a concrete refactoring plan:
- Target file/class structure (show the module tree)
- Migration order (which tabs/classes to extract first and why)
- How to handle QSettings backward compat
- One example of how ServerThread should be decoupled from the main window
- Risk areas to be careful about""",
    },
    {
        "id": "v2_perf_analysis",
        "title": "Analyze inference performance bottleneck",
        "prompt": """You are analyzing performance of a Qwen3.6-27B model running on AMD Radeon RX 9070 XT (gfx1201, 16 GB VRAM)
via llama.cpp ROCm build. The model is Q3_K_S quantized (~18 GB file, ~12 GB loaded).

Benchmark results across configs (all with flash-attn, ctx=65536, q4_0 KV cache, seed=42):

| ubatch | spec       | runs | agg TPS | mean TPS | stdev |
|--------|------------|------|---------|----------|-------|
| 128    | none       | 5    | 22.1    | 21.8     | 1.2   |
| 256    | none       | 5    | 26.3    | 26.1     | 1.8   |
| 512    | none       | 5    | 28.5    | 28.0     | 3.4   |
| 512    | ngram-mod  | 5    | 33.8    | 33.2     | 4.1   |
| 512    | ngram-mod  | 1    | 37.6    | 37.6     | 0.0   |
| 1024   | ngram-mod  | 5    | 23.6    | 23.1     | 2.9   |

Observations from profiling:
- ubatch=512 has FATTN switching between VEC and TILE kernels depending on sequence length
- ngram acceptance rate varies 0.08–0.45 per request depending on prompt pattern
- stdev increases with ubatch, especially with spec active

Analyze:
1. Why does ubatch=512 single-run show 37.6 TPS but 5-run average drops to 33.8?
2. What causes the sharp performance cliff at ubatch=1024?
3. Why does speculative decoding help more with some prompts than others, and what does 0.08 acceptance rate mean for throughput?
4. What would you investigate next to push stable 5-run average above 35 TPS?""",
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
    if args.server_seed is not None:
        cmd.extend(["--seed", str(args.server_seed)])
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


def write_results(
    rows: list[dict[str, Any]],
    out_dir: Path,
    label: str,
    artifact_mode: str,
    stats_ignore_first_run: bool = False,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{label}.jsonl"
    csv_path = out_dir / f"{label}.csv"
    artifacts = {"jsonl_file": "", "csv_file": ""}

    if artifact_mode == "full":
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        artifacts["jsonl_file"] = jsonl_path.name

    fieldnames = [
        "label", "task_id", "title", "wall_s", "prompt_tokens",
        "completion_tokens", "total_tokens", "completion_tps_wall",
        "response_chars", "error",
    ]
    if artifact_mode == "full":
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        artifacts["csv_file"] = csv_path.name

    total_completion = sum(row.get("completion_tokens") or 0 for row in rows)
    total_wall = sum(row.get("wall_s") or 0.0 for row in rows)
    if artifact_mode == "full":
        print(f"Wrote {jsonl_path}")
        print(f"Wrote {csv_path}")
    else:
        print("Artifact mode: unified (per-run CSV/JSONL skipped)")
    if total_wall > 0 and total_completion > 0:
        print(f"Aggregate completion TPS by wall time: {total_completion / total_wall:.2f}")

    tps_values = [float(row["completion_tps_wall"]) for row in rows if row.get("completion_tps_wall") is not None]
    if tps_values:
        print(f"Mean task TPS: {statistics.mean(tps_values):.2f}")
        print(f"Median task TPS: {statistics.median(tps_values):.2f}")
        if len(tps_values) > 1:
            print(f"Task TPS stdev: {statistics.pstdev(tps_values):.4f}")

    if stats_ignore_first_run:
        warm_rows = [row for row in rows if int(row.get("run") or 0) > 1]
        if warm_rows:
            warm_completion = sum(row.get("completion_tokens") or 0 for row in warm_rows)
            warm_wall = sum(row.get("wall_s") or 0.0 for row in warm_rows)
            warm_tps_values = [float(row["completion_tps_wall"]) for row in warm_rows if row.get("completion_tps_wall") is not None]

            print("Warm-only stats (excluding run #1):")
            if warm_wall > 0 and warm_completion > 0:
                print(f"Warm-only aggregate completion TPS: {warm_completion / warm_wall:.2f}")
            if warm_tps_values:
                print(f"Warm-only mean task TPS: {statistics.mean(warm_tps_values):.2f}")
                print(f"Warm-only median task TPS: {statistics.median(warm_tps_values):.2f}")
                if len(warm_tps_values) > 1:
                    print(f"Warm-only task TPS stdev: {statistics.pstdev(warm_tps_values):.4f}")
    return artifacts


def aggregate_completion_tps(rows: list[dict[str, Any]]) -> float:
    total_completion = sum(row.get("completion_tokens") or 0 for row in rows)
    total_wall = sum(row.get("wall_s") or 0.0 for row in rows)
    return (total_completion / total_wall) if total_wall > 0 and total_completion > 0 else 0.0


def infer_spec_mode(server_extra: str) -> str:
    text = (server_extra or "").lower()
    if "--spec-type mtp" in text:
        return "mtp"
    if "--spec-type ngram-mod" in text:
        return "ngram-mod"
    if "--spec-type draft" in text:
        return "draft"
    if "--spec-type eagle" in text:
        return "eagle"
    if "--spec-type none" in text:
        return "none"
    if "--spec-type" in text:
        return "other"
    return "none"


def is_mtp_model_name(model_path: str) -> bool:
    name = Path(model_path).name.lower()
    return "-mtp" in name or name.endswith("mtp.gguf")


def _model_display_name(model_path: str) -> str:
    normalized = str(model_path).replace("\\", "/").rstrip("/")
    if not normalized:
        return "-"
    return normalized.rsplit("/", 1)[-1]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_build_registry_records() -> list[dict[str, str]]:
    registry_path = ROOT / "gui" / "build_versions.json"
    if not registry_path.exists():
        return []
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    raw_records = payload.get("builds") if isinstance(payload, dict) else None
    if not isinstance(raw_records, list):
        return []

    records: list[dict[str, str]] = []
    for item in raw_records:
        if not isinstance(item, dict):
            continue
        build_dir = str(item.get("build_dir", "")).strip()
        records.append(
            {
                "id": str(item.get("id", "")).strip(),
                "name": str(item.get("name", "")).strip(),
                "backend": str(item.get("backend", "")).strip(),
                "build_dir": str(Path(build_dir)) if build_dir else "",
            }
        )
    return records


def resolve_build_metadata(build_id: str, server_bin: str | None) -> dict[str, str]:
    records = _load_build_registry_records()
    selected: dict[str, str] | None = None

    wanted_id = str(build_id or "").strip()
    if wanted_id:
        selected = next((r for r in records if r.get("id", "") == wanted_id), None)

    if selected is None and server_bin:
        server_path = Path(server_bin).resolve()
        for record in records:
            build_dir = str(record.get("build_dir", "")).strip()
            if not build_dir:
                continue
            try:
                if server_path.is_relative_to(Path(build_dir).resolve()):
                    selected = record
                    break
            except Exception:
                if str(server_path).lower().startswith(str(Path(build_dir).resolve()).lower()):
                    selected = record
                    break

    if selected is None:
        return {
            "build_id": wanted_id,
            "build_name": "",
            "build_backend": "",
        }

    return {
        "build_id": str(selected.get("id", wanted_id)).strip(),
        "build_name": str(selected.get("name", "")).strip(),
        "build_backend": str(selected.get("backend", "")).strip(),
    }


def _best_rows_by_group(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        if _to_int(row.get("errors", 0)) != 0:
            continue
        key = "MTP" if str(row.get("is_mtp_model", "0")) == "1" else "Non-MTP"
        if key not in best:
            best[key] = row
            continue
        if _to_float(row.get("aggregate_tps", 0.0)) > _to_float(best[key].get("aggregate_tps", 0.0)):
            best[key] = row
    return best


def _best_rows_by_model(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    best: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if _to_int(row.get("errors", 0)) != 0:
            continue
        model_name = _model_display_name(row.get("model", "-"))
        group = "MTP" if str(row.get("is_mtp_model", "0")) == "1" else "Non-MTP"
        key = (group, model_name)
        if key not in best:
            best[key] = row
            continue
        if _to_float(row.get("aggregate_tps", 0.0)) > _to_float(best[key].get("aggregate_tps", 0.0)):
            best[key] = row
    return best


def _best_rows_by_build(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        if _to_int(row.get("errors", 0)) != 0:
            continue

        build_id = str(row.get("build_id", "")).strip()
        build_name = str(row.get("build_name", "")).strip()
        build_backend = str(row.get("build_backend", "")).strip()
        if not build_id and not build_name:
            continue

        key = build_id or f"name:{build_name.lower()}::{build_backend.lower()}"
        current_best = best.get(key)
        if current_best is None or _to_float(row.get("aggregate_tps", 0.0)) > _to_float(current_best.get("aggregate_tps", 0.0)):
            best[key] = row

    return best


def _make_run_id(row: dict[str, str]) -> str:
    ts = str(row.get("timestamp", "")).strip()
    label = str(row.get("label", "")).strip()
    mode = str(row.get("mode", "")).strip()
    model = str(row.get("model", "")).strip()
    seed = f"{ts}|{label}|{mode}|{model}"
    digest = hashlib.sha1(seed.encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"run-{digest}"


def _render_best_row(group: str, row: dict[str, str] | None) -> str:
    if row is None:
        return f"| {group} | - | - | - | - | - | - | - | - | - | - |"
    return (
        f"| {group} | {row.get('run_id', '-')} | {row.get('build_id', '-')} | {row.get('aggregate_tps', '-')} | {row.get('label', '-')} | "
        f"{row.get('timestamp', '-')} | {row.get('mode', '-')} | {_model_display_name(row.get('model', '-'))} | "
        f"{row.get('spec_mode', '-')} | {row.get('ctx', '-')} | {row.get('batch', '-')}/{row.get('ubatch', '-')} |"
    )


def _render_model_best_row(group: str, model_name: str, row: dict[str, str]) -> str:
    return (
        f"| {group} | {model_name} | {row.get('run_id', '-')} | {row.get('build_id', '-') or '-'} | {row.get('aggregate_tps', '-')} | "
        f"{row.get('label', '-')} | {row.get('timestamp', '-')} | {row.get('mode', '-')} | {row.get('spec_mode', '-')} | "
        f"{row.get('ctx', '-')} | {row.get('batch', '-')}/{row.get('ubatch', '-')} |"
    )


def _render_build_best_row(row: dict[str, str]) -> str:
    build_id = str(row.get("build_id", "")).strip() or "-"
    build_name = str(row.get("build_name", "")).strip() or "-"
    build_backend = str(row.get("build_backend", "")).strip() or "-"
    return (
        f"| {build_id} | {build_name} | {build_backend} | {row.get('run_id', '-')} | {row.get('aggregate_tps', '-')} | {row.get('label', '-')} | "
        f"{row.get('timestamp', '-')} | {row.get('mode', '-')} | {_model_display_name(row.get('model', '-'))} | "
        f"{row.get('spec_mode', '-')} | {row.get('ctx', '-')} | {row.get('batch', '-')}/{row.get('ubatch', '-')} |"
    )


def write_history_md(history_md: Path, rows: list[dict[str, str]]) -> None:
    best = _best_rows_by_group(rows)
    best_by_model = _best_rows_by_model(rows)
    best_by_build = _best_rows_by_build(rows)
    lines: list[str] = [
        "# Agent Workload Benchmark History",
        "",
        "Автоматически обновляется скриптом scripts/agent_workload_bench.py.",
        "",
        "## Locked Best Overall Results",
        "",
        "| Group | Run ID | Build ID | Aggregate TPS | Label | Timestamp | Mode | Model | Spec | Ctx | Batch/UBatch |",
        "|---|---|---|---:|---|---|---|---|---|---:|---:|",
        _render_best_row("MTP", best.get("MTP")),
        _render_best_row("Non-MTP", best.get("Non-MTP")),
        "",
        "## Locked Best Per Model",
        "",
        "| Group | Model | Run ID | Build ID | Aggregate TPS | Label | Timestamp | Mode | Spec | Ctx | Batch/UBatch |",
        "|---|---|---|---|---:|---|---|---|---|---:|---:|",
    ]

    for (group, model_name), row in sorted(best_by_model.items(), key=lambda item: (item[0][0], item[0][1].lower())):
        lines.append(_render_model_best_row(group, model_name, row))

    lines += [
        "",
        "## Locked Best Per Build",
        "",
        "| Build ID | Build Name | Backend | Run ID | Aggregate TPS | Label | Timestamp | Mode | Model | Spec | Ctx | Batch/UBatch |",
        "|---|---|---|---|---:|---|---|---|---|---|---:|---:|",
    ]

    for row in sorted(
        best_by_build.values(),
        key=lambda item: (
            str(item.get("build_backend", "")).lower(),
            str(item.get("build_name", "")).lower(),
            str(item.get("build_id", "")).lower(),
        ),
    ):
        lines.append(_render_build_best_row(row))

    lines += [
        "",
        "## Full History",
        "",
        "| Timestamp | Run ID | Build ID | Build Name | Backend | Label | Mode | Model | Spec | Ctx | Batch/UBatch | TPS | Errors | Artifacts |",
        "|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]

    for row in sorted(rows, key=lambda r: r.get("timestamp", ""), reverse=True):
        model_name = _model_display_name(row.get("model", "-"))
        artifacts = ", ".join(
            p for p in [row.get("jsonl_file", ""), row.get("csv_file", ""), row.get("summary_file", ""), row.get("server_log_file", "")]
            if p
        )
        lines.append(
            f"| {row.get('timestamp', '-')} | {row.get('run_id', '-')} | {row.get('build_id', '-')} | "
            f"{row.get('build_name', '-') or '-'} | {row.get('build_backend', '-') or '-'} | {row.get('label', '-')} | {row.get('mode', '-')} | "
            f"{model_name} | {row.get('spec_mode', '-')} | {row.get('ctx', '-')} | "
            f"{row.get('batch', '-')}/{row.get('ubatch', '-')} | {row.get('aggregate_tps', '-')} | "
            f"{row.get('errors', '-')} | {artifacts or '-'} |"
        )

    history_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_history_entry(
    entry: dict[str, Any],
    out_dir: Path,
    history_csv_name: str = HISTORY_CSV_NAME,
    history_md_name: str = HISTORY_MD_NAME,
) -> None:
    history_dir = out_dir
    history_csv = history_dir / history_csv_name
    history_md = history_dir / history_md_name

    rows: list[dict[str, str]] = []
    if history_csv.exists():
        with history_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = {k: str(row.get(k, "")) for k in HISTORY_FIELDS}
                if not normalized["run_id"]:
                    normalized["run_id"] = _make_run_id(normalized)
                # Keep old history useful when prior schema did not include csv_file/summary_file.
                if not normalized["jsonl_file"] and normalized["label"] and normalized["mode"] == "single-run":
                    normalized["jsonl_file"] = f"{normalized['label']}.jsonl"
                if not normalized["csv_file"] and normalized["label"] and normalized["mode"] == "single-run":
                    normalized["csv_file"] = f"{normalized['label']}.csv"
                if not normalized["server_log_file"] and normalized["label"]:
                    normalized["server_log_file"] = f"{normalized['label']}.server.log"
                rows.append(normalized)

    normalized_entry = {k: "" for k in HISTORY_FIELDS}
    for key in HISTORY_FIELDS:
        if key in entry:
            normalized_entry[key] = str(entry[key])
    if not normalized_entry["run_id"]:
        normalized_entry["run_id"] = _make_run_id(normalized_entry)
    rows.append(normalized_entry)

    best = _best_rows_by_model(rows)
    for row in rows:
        model_name = _model_display_name(row.get("model", "-"))
        group = "MTP" if str(row.get("is_mtp_model", "0")) == "1" else "Non-MTP"
        row["is_group_best"] = "1" if best.get((group, model_name)) is row else "0"

    history_dir.mkdir(parents=True, exist_ok=True)
    with history_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in HISTORY_FIELDS})

    write_history_md(history_md, rows)
    print(f"Wrote {history_csv}")
    print(f"Wrote {history_md}")


def cleanup_legacy_artifacts(
    out_dir: Path,
    apply: bool,
    keep_pattern_expr: str,
    protected_names: set[str],
) -> tuple[int, list[str]]:
    """Delete legacy per-run benchmark artifacts while preserving unified history and protected files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[Path] = []
    for path in sorted(out_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        if name in protected_names:
            continue
        if keep_pattern_expr and re.fullmatch(keep_pattern_expr, name, flags=re.IGNORECASE):
            continue
        if path.suffix.lower() in {".jsonl", ".csv", ".log", ".md"}:
            candidates.append(path)

    removed: list[str] = []
    for path in candidates:
        if apply:
            try:
                path.unlink()
                removed.append(path.name)
            except Exception:
                continue
        else:
            removed.append(path.name)

    return len(removed), removed


def parse_int_csv(values: str) -> list[int]:
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def parse_text_csv(values: str) -> list[str]:
    return [v.strip() for v in values.split(",") if v.strip()]


def _is_drop_significant(prev_tps: float, curr_tps: float, drop_pct: float) -> bool:
    if prev_tps <= 0.0:
        return False
    ratio = (prev_tps - curr_tps) / prev_tps
    return ratio >= max(0.0, drop_pct)


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


def resolve_history_names(version: str) -> tuple[str, str]:
    ver = (version or "").strip().lower()
    if ver in ("", "v1", "1", "legacy"):
        return HISTORY_CSV_NAME, HISTORY_MD_NAME

    safe = re.sub(r"[^a-z0-9_\-]", "_", ver, flags=re.IGNORECASE).strip("_")
    if not safe:
        return HISTORY_CSV_NAME, HISTORY_MD_NAME

    suffix = safe.upper()
    return f"BENCH_HISTORY_{suffix}.csv", f"BENCH_HISTORY_{suffix}.md"


def parse_args() -> argparse.Namespace:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="Short coding-agent benchmark for llama-server")
    parser.add_argument("--label", default=f"rocm-baseline-{timestamp}", help="result file label")
    parser.add_argument("--out-dir", default=str(ROOT / "build_logs" / "agent-workload"), help="output directory")
    parser.add_argument("--tasks", choices=["quick", "full", "v2"], default="quick",
                        help="prompt set: quick (v1 short tasks), full (v1 extended), v2 (realistic agentic-flow tasks)")
    parser.add_argument("--runs", type=int, default=1, help="repeat each task N times")

    parser.add_argument("--no-start", action="store_true", help="use an already running server")
    parser.add_argument("--server-bin", default=None, help="path to llama-server executable")
    parser.add_argument("--model", default=None, help="path to GGUF model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="server port; 0 picks a free port when starting a server")
    parser.add_argument("--api-model", default="local-model")
    parser.add_argument("--server-extra", default="", help="extra llama-server args, e.g. '--spec-type mtp --spec-draft-n-max 3'")
    parser.add_argument("--build-id", default="", help="build registry ID linked to this benchmark run")
    parser.add_argument("--artifact-mode", choices=["full", "unified"], default="full",
                        help="artifact mode: full writes per-run CSV/JSONL, unified writes only history")
    parser.add_argument("--history-version", default="v1",
                        help="history namespace/version, e.g. v1 (default), v2 -> BENCH_HISTORY_V2.csv/.md")
    parser.add_argument("--cleanup-legacy-artifacts", action="store_true",
                        help="cleanup old per-run benchmark artifacts in out-dir")
    parser.add_argument("--cleanup-apply", action="store_true",
                        help="apply deletion for --cleanup-legacy-artifacts (default is dry-run)")
    parser.add_argument(
        "--cleanup-keep-patterns",
        default=r"BENCH_HISTORY\.(csv|md)|BASELINE_.*|.*LOCK.*|.*autotune-summary\.(md|csv|json)|ROCM_BENCH_COMPARISON\.md",
        help="regex patterns (pipe-separated) for artifact names to keep during cleanup",
    )

    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--ctx-size", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=2048)
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q8_0")
    parser.add_argument("--flash-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=False,
                        help="disable model thinking by forcing chat-template kwargs; default keeps thinking enabled")

    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument("--server-seed", type=int, default=42,
                        help="llama-server seed for deterministic decoding; set to -1 to disable fixed seed")
    parser.add_argument("--stats-ignore-first-run", action=argparse.BooleanOptionalAction, default=False,
                        help="print additional warm-only metrics that exclude run #1 (cold/probing phase)")
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
    parser.add_argument("--autotune-mtp-draft-n-max", type=int, default=3)
    parser.add_argument("--autotune-max-configs", type=int, default=48, help="safety cap for sweep size")
    parser.add_argument(
        "--autotune-smart-prune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="early-stop batch/ubatch branches when TPS consistently degrades",
    )
    parser.add_argument(
        "--autotune-prune-drop-pct",
        type=float,
        default=0.07,
        help="relative TPS drop threshold for smart prune (e.g. 0.07 = 7%%)",
    )
    parser.add_argument(
        "--autotune-prune-patience",
        type=int,
        default=2,
        help="how many consecutive significant drops trigger branch prune",
    )
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
    if args.server_seed is not None and args.server_seed < 0:
        args.server_seed = None
    out_dir = Path(args.out_dir)
    history_csv_name, history_md_name = resolve_history_names(args.history_version)
    build_meta = resolve_build_metadata(args.build_id, args.server_bin)
    args.build_id = build_meta["build_id"]

    if args.cleanup_legacy_artifacts:
        protected_history = {HISTORY_CSV_NAME, HISTORY_MD_NAME, history_csv_name, history_md_name}
        count, items = cleanup_legacy_artifacts(
            out_dir,
            apply=args.cleanup_apply,
            keep_pattern_expr=args.cleanup_keep_patterns,
            protected_names=protected_history,
        )
        mode = "APPLY" if args.cleanup_apply else "DRY-RUN"
        print(f"Legacy artifact cleanup ({mode}): {count} file(s)")
        for name in items[:200]:
            print(f"  - {name}")
        if count > 200:
            print(f"  ... and {count - 200} more")
        return 0

    if args.tasks == "v2":
        tasks = TASKS_V2
        # v2 tasks produce longer responses; bump max_tokens unless user set it explicitly
        if args.max_tokens == 160:
            args.max_tokens = 500
        # auto-select v2 history unless caller overrode to something other than v1
        if args.history_version in ("v1", "1", "legacy", ""):
            args.history_version = "v2"
            history_csv_name, history_md_name = resolve_history_names(args.history_version)
    elif args.tasks == "full":
        tasks = TASKS_FULL
    else:
        tasks = TASKS_QUICK
    if not args.autotune:
        try:
            rows = run_suite(args, tasks)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            print("Stop background server(s) or rerun with --background-server-policy warn/ignore")
            return 3
        artifacts = write_results(
            rows,
            out_dir,
            args.label,
            args.artifact_mode,
            stats_ignore_first_run=args.stats_ignore_first_run,
        )
        aggregate_tps = aggregate_completion_tps(rows)
        tps_values = [float(r["completion_tps_wall"]) for r in rows if r.get("completion_tps_wall") is not None]
        model_path = str(Path(args.model) if args.model else default_model() or "")
        append_history_entry(
            {
                "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_id": "",
                "build_id": args.build_id,
                "build_name": build_meta["build_name"],
                "build_backend": build_meta["build_backend"],
                "mode": "single-run",
                "label": args.label,
                "model": model_path,
                "is_mtp_model": 1 if is_mtp_model_name(model_path) else 0,
                "tasks": args.tasks,
                "runs": args.runs,
                "ctx": args.ctx_size,
                "batch": args.batch_size,
                "ubatch": args.ubatch_size,
                "kv_k": args.cache_type_k,
                "kv_v": args.cache_type_v,
                "spec_mode": infer_spec_mode(args.server_extra),
                "gpu_layers": args.gpu_layers,
                "parallel": args.parallel,
                "flash_attn": "on" if args.flash_attn else "off",
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "aggregate_tps": f"{aggregate_tps:.4f}",
                "mean_task_tps": f"{statistics.mean(tps_values):.4f}" if tps_values else "0.0000",
                "errors": sum(1 for row in rows if row.get("error")),
                "best_config": "",
                "jsonl_file": artifacts.get("jsonl_file", ""),
                "csv_file": artifacts.get("csv_file", ""),
                "summary_file": "",
                "server_log_file": f"{args.label}.server.log",
            },
            out_dir,
            history_csv_name=history_csv_name,
            history_md_name=history_md_name,
        )
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
        if args.autotune_smart_prune:
            print(
                "WARNING: "
                f"Autotune config count {len(configs)} exceeds --autotune-max-configs {args.autotune_max_configs}, "
                "continuing because --autotune-smart-prune is enabled"
            )
        else:
            print(f"Autotune config count {len(configs)} exceeds --autotune-max-configs {args.autotune_max_configs}")
            return 4

    base_server_extra = args.server_extra.strip()
    summaries: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    total_configs = len(configs)
    executed = 0
    skipped_by_prune = 0

    ctx_values_sorted = sorted(set(int(v) for v in ctx_values))
    kv_values_sorted = list(dict.fromkeys(kv_values))
    spec_values_sorted = list(dict.fromkeys(spec_values))
    ubatch_values_sorted = sorted(set(int(v) for v in ubatch_values))
    batch_values_sorted = sorted(set(int(v) for v in batch_values))

    for spec_mode in spec_values_sorted:
        for ctx_size in ctx_values_sorted:
            for kv_type in kv_values_sorted:
                ubatch_drop_streak = 0
                prev_ubatch_best_tps = 0.0

                for ubatch_size in ubatch_values_sorted:
                    prev_batch_tps = 0.0
                    batch_drop_streak = 0
                    ubatch_best_tps = 0.0

                    for batch_size in batch_values_sorted:
                        run_idx = executed + 1
                        run_args = argparse.Namespace(**vars(args))
                        run_args.ctx_size = int(ctx_size)
                        run_args.batch_size = int(batch_size)
                        run_args.ubatch_size = int(ubatch_size)
                        run_args.cache_type_k = kv_type
                        run_args.cache_type_v = kv_type
                        run_args.label = f"{args.label}-cfg{run_idx:02d}"

                        extra_bits: list[str] = []
                        if base_server_extra:
                            extra_bits.append(base_server_extra)
                        if spec_mode == "ngram-mod":
                            extra_bits.append("--spec-type ngram-mod")
                            extra_bits.append(f"--spec-ngram-mod-n-min {args.autotune_ngram_min}")
                            extra_bits.append(f"--spec-ngram-mod-n-match {args.autotune_ngram_match}")
                            extra_bits.append(f"--spec-ngram-mod-n-max {args.autotune_ngram_max}")
                        elif spec_mode == "mtp":
                            extra_bits.append("--spec-type mtp")
                            extra_bits.append(f"--spec-draft-n-max {args.autotune_mtp_draft_n_max}")
                        run_args.server_extra = " ".join(extra_bits)

                        print(
                            f"Autotune [{run_idx}/{total_configs}]: ctx={ctx_size}, b={batch_size}, ub={ubatch_size}, "
                            f"kv={kv_type}, spec={spec_mode}"
                        )
                        try:
                            rows = run_suite(run_args, tasks)
                        except RuntimeError as exc:
                            print(f"ERROR: {exc}")
                            return 3

                        executed += 1
                        write_results(
                            rows,
                            out_dir,
                            run_args.label,
                            args.artifact_mode,
                            stats_ignore_first_run=args.stats_ignore_first_run,
                        )
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

                            if best is not None:
                                print(
                                    "CURRENT BEST: "
                                    f"ctx={best['ctx_size']} b={best['batch_size']} ub={best['ubatch_size']} "
                                    f"kv={best['kv']} spec={best['spec_mode']} aggregate_tps={best['aggregate_tps']:.2f}"
                                )

                            curr_tps = float(summary["aggregate_tps"])
                            ubatch_best_tps = max(ubatch_best_tps, curr_tps)

                            if args.autotune_smart_prune and _is_drop_significant(prev_batch_tps, curr_tps, args.autotune_prune_drop_pct):
                                batch_drop_streak += 1
                            else:
                                batch_drop_streak = 0

                            prev_batch_tps = curr_tps

                            if args.autotune_smart_prune and batch_drop_streak >= max(1, args.autotune_prune_patience):
                                remaining = len(batch_values_sorted) - (batch_values_sorted.index(batch_size) + 1)
                                if remaining > 0:
                                    skipped_by_prune += remaining
                                    print(
                                        "Autotune prune: stop larger batch values for "
                                        f"ctx={ctx_size}, ub={ubatch_size}, kv={kv_type}, spec={spec_mode} "
                                        f"after {batch_drop_streak} drop(s)"
                                    )
                                break

                    if args.autotune_smart_prune and _is_drop_significant(prev_ubatch_best_tps, ubatch_best_tps, args.autotune_prune_drop_pct):
                        ubatch_drop_streak += 1
                    else:
                        ubatch_drop_streak = 0

                    prev_ubatch_best_tps = max(prev_ubatch_best_tps, ubatch_best_tps)

                    if args.autotune_smart_prune and ubatch_drop_streak >= max(1, args.autotune_prune_patience):
                        remaining_ub = len(ubatch_values_sorted) - (ubatch_values_sorted.index(ubatch_size) + 1)
                        if remaining_ub > 0:
                            est_skip = remaining_ub * len(batch_values_sorted)
                            skipped_by_prune += est_skip
                            print(
                                "Autotune prune: stop larger ubatch values for "
                                f"ctx={ctx_size}, kv={kv_type}, spec={spec_mode} "
                                f"after {ubatch_drop_streak} drop(s)"
                            )
                        break

    if args.autotune_smart_prune:
        print(f"Autotune smart-prune summary: executed={executed}, skipped~={skipped_by_prune}, total-grid={total_configs}")

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
    model_path = str(Path(args.model) if args.model else default_model() or "")
    best_cfg = ""
    aggregate_tps = 0.0
    mean_tps = 0.0
    spec_mode = "mixed"
    if best:
        best_cfg = (
            f"ctx={best['ctx_size']} b={best['batch_size']} ub={best['ubatch_size']} "
            f"kv={best['kv']} spec={best['spec_mode']}"
        )
        aggregate_tps = float(best.get("aggregate_tps", 0.0))
        mean_tps = float(best.get("mean_task_tps", 0.0))
        spec_mode = str(best.get("spec_mode", "mixed"))

    append_history_entry(
        {
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": "",
            "build_id": args.build_id,
            "build_name": build_meta["build_name"],
            "build_backend": build_meta["build_backend"],
            "mode": "autotune",
            "label": args.label,
            "model": model_path,
            "is_mtp_model": 1 if is_mtp_model_name(model_path) else 0,
            "tasks": args.tasks,
            "runs": args.runs,
            "ctx": args.autotune_min_ctx,
            "batch": "sweep",
            "ubatch": "sweep",
            "kv_k": "sweep",
            "kv_v": "sweep",
            "spec_mode": spec_mode,
            "gpu_layers": args.gpu_layers,
            "parallel": args.parallel,
            "flash_attn": "on" if args.flash_attn else "off",
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "aggregate_tps": f"{aggregate_tps:.4f}",
            "mean_task_tps": f"{mean_tps:.4f}",
            "errors": sum(int(r.get("errors", 0)) for r in summaries),
            "best_config": best_cfg,
            "jsonl_file": "",
            "csv_file": "",
            "summary_file": summary_csv.name,
            "server_log_file": "",
        },
        out_dir,
        history_csv_name=history_csv_name,
        history_md_name=history_md_name,
    )

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
