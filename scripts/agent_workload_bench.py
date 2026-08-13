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
import threading
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
CANONICAL_BENCH_RUNS_CSV = "BENCH_RUNS.csv"
CANONICAL_BENCH_RECENT_MD = "BENCH_RECENT.md"
CANONICAL_BENCH_LANES_MD = "BENCH_LANES.md"
CANONICAL_RECENT_LIMIT = 80
CANONICAL_HISTORY_START = "2026-07-01"
PRIMARY_MAX_CTX = 131072
PRIMARY_CTX_LABEL = "130k"

KERNEL_FULL_TRACE_ENV = {
    "GGML_TRACE_FATTN_SELECTED": "1",
    "GGML_TRACE_FATTN_TIMING": "1",
    "GGML_TRACE_FATTN_TIMING_SYNC": "1",
    "GGML_TRACE_FATTN_TIMING_PRE_SYNC": "1",
    "GGML_TRACE_FATTN_WMMA_CONFIG": "1",
    "GGML_TRACE_GDN_PATH": "1",
    "GGML_TRACE_GDN_TIMING": "1",
    "GGML_TRACE_GDN_TIMING_SYNC_HIP": "1",
    "GGML_TRACE_GDN_TIMING_PRE_SYNC_HIP": "1",
    "GGML_TRACE_MMQ_PATH": "1",
    "GGML_TRACE_MMQ_TIMING": "1",
    "GGML_TRACE_MMQ_TIMING_SYNC": "1",
    "GGML_TRACE_MMQ_TIMING_PRE_SYNC": "1",
    "GGML_TRACE_MMVQ_PATH": "1",
    "GGML_TRACE_MMVQ_SMALL_K": "1",
    "GGML_TRACE_MMVQ_TIMING": "1",
    "GGML_TRACE_MMVQ_TIMING_SYNC": "1",
    "GGML_TRACE_MMVQ_TIMING_PRE_SYNC": "1",
    "GGML_TRACE_CUDA_NODE_TIMING": "1",
    "GGML_TRACE_CUDA_NODE_TIMING_SYNC": "1",
    "GGML_TRACE_CUDA_MUL_MAT_ROUTE": "1",
    "LLAMA_UBATCH_TIMING": "1",
    "LLAMA_UBATCH_TIMING_SYNC": "1",
}

VULKAN_ROUTE_TRACE_ENV = {
    "GGML_VK_MATMUL_ROUTE_TRACE": "1",
    "GGML_VK_FA_ROUTE_TRACE": "1",
    "GGML_VK_FFN_ROUTE_TRACE": "1",
}

VULKAN_PERF_TRACE_ENV = {
    **VULKAN_ROUTE_TRACE_ENV,
    "GGML_VK_PERF_LOGGER": "1",
    "GGML_VK_PERF_LOGGER_FREQUENCY": "1",
}

VULKAN_Q3_STATS_ENV = {
    **VULKAN_ROUTE_TRACE_ENV,
    "GGML_VK_PIPELINE_STATS": "matmul_q3_k",
}

BENCHMARK_ENV_PREFIXES = ("GGML_", "LLAMA_", "HSA_", "ROCBLAS_", "HIPBLASLT_")
SENSITIVE_ENV_FRAGMENTS = (
    "KEY", "TOKEN", "SECRET", "PASSWORD", "ENDPOINT", "ACCOUNT",
    "URL", "URI", "HOST", "USER", "ORG", "PROJECT",
)


def collect_active_benchmark_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return inherited performance knobs without exposing credentials."""
    source = os.environ if env is None else env
    active: dict[str, str] = {}
    for key in sorted(source):
        value = source[key]
        if not value or not key.startswith(BENCHMARK_ENV_PREFIXES):
            continue
        active[key] = "<redacted>" if any(fragment in key.upper() for fragment in SENSITIVE_ENV_FRAGMENTS) else value
    return active


def install_graceful_stop_handlers() -> None:
    """Turn console stop events into stack unwinding so server cleanup runs."""
    def handle_stop(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_stop)
    if os.name == "nt" and hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, handle_stop)

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
    "task_ids",
    "runs",
    "ctx",
    "batch",
    "ubatch",
    "kv_k",
    "kv_v",
    "spec_mode",
    "extra_preset",
    "extra_args",
    "no_reuse",
    "gpu_layers",
    "parallel",
    "flash_attn",
    "max_tokens",
    "real_context_mode",
    "real_context_chars",
    "real_context_safe_fill",
    "no_v2_prime_pass",
    "temperature",
    "top_p",
    "aggregate_tps",
    "mean_task_tps",
    "prompt_eval_tps",
    "decode_eval_tps",
    "prompt_eval_ms",
    "decode_eval_ms",
    "errors",
    "metric_scope",
    "lane_key",
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
]


TASKS_FULL = TASKS_QUICK + [
    {
        "id": "implementation_plan",
        "title": "Implementation planning",
        "prompt": """We want to port llama.cpp PR #22673 MTP support into a Windows ROCm/Vulkan fork with a PyQt GUI.
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
    "bin_dir": "C:/repos/llama.cpp-rdna-lab/build-rocm/bin",
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
    # Ordered by preference: the current dual-GPU ROCm build first, then the
    # generic ROCm/Vulkan/CPU build dirs that actually exist in this repo.
    # NOTE: the historical default `build-rocm/bin` does not exist on this rig;
    # the live ROCm build lives in `build-rocm-vec/bin`.
    build_dirs = [
        "build-rocm-vec",
        "build-rocm",
        "build-vulkan",
        "build-rocm-upstream-stock",
        "build",
        "build-cpu",
    ]
    candidates: list[Path] = []
    for d in build_dirs:
        candidates.append(ROOT / d / "bin" / "llama-server.exe")
        candidates.append(ROOT / d / "bin" / "Release" / "llama-server.exe")
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


def choose_repo_snapshot_files(root: Path) -> list[Path]:
    seed_files = [
        root / "AGENTS.md",
        root / "BENCHMARKS.md",
        root / "PROJECT_PROFILE.md",
        root / "QWEN_SPEED_RESEARCH.md",
        root / "CMakeLists.txt",
    ]
    scan_roots = [
        root / "gui",
        root / "scripts",
        root / "src",
        root / "include",
        root / "common",
        root / "ggml" / "src",
        root / "ggml" / "include",
        root / "tests",
    ]
    exts = {".py", ".md", ".txt", ".json", ".cmake", ".cpp", ".c", ".h", ".hpp", ".cu", ".cuh"}

    selected: list[Path] = []
    seen: set[Path] = set()

    for path in seed_files:
        if path.exists() and path not in seen:
            selected.append(path)
            seen.add(path)

    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in sorted(scan_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            rel = path.relative_to(root).as_posix()
            if rel.startswith("build/") or "/build/" in rel:
                continue
            if path in seen:
                continue
            selected.append(path)
            seen.add(path)

    return selected


def build_repo_snapshot_prefix(root: Path, char_budget: int) -> tuple[str, int, int]:
    if char_budget <= 0:
        return "", 0, 0

    files = choose_repo_snapshot_files(root)
    chunks: list[str] = []
    char_count = 0
    file_count = 0

    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if not text.strip():
            continue

        rel = path.relative_to(root).as_posix()
        block = f"\n\n### FILE: {rel}\n{text}"
        if char_count + len(block) > char_budget:
            remaining = char_budget - char_count
            if remaining > 4096:
                block = block[:remaining]
                chunks.append(block)
                char_count += len(block)
                file_count += 1
            break

        chunks.append(block)
        char_count += len(block)
        file_count += 1

    prefix = (
        "Ниже входящий контекст из текущего репозитория llama.cpp-rdna-lab. "
        "Учитывай этот контекст при ответе на задачу.\n"
        "===== REPO SNAPSHOT BEGIN ====="
        + "".join(chunks)
        + "\n===== REPO SNAPSHOT END =====\n\n"
    )
    return prefix, char_count, file_count


def apply_real_context_prefix(tasks: list[dict[str, str]], prefix: str) -> list[dict[str, str]]:
    if not prefix:
        return tasks

    out: list[dict[str, str]] = []
    for task in tasks:
        task_copy = dict(task)
        task_copy["prompt"] = prefix + task["prompt"]
        out.append(task_copy)
    return out


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


def apply_trace_preset(env: dict[str, str], preset: str) -> None:
    if preset == "none":
        return
    if preset == "kernel-full":
        env.update(KERNEL_FULL_TRACE_ENV)
        return
    if preset == "vulkan-routes":
        env.update(VULKAN_ROUTE_TRACE_ENV)
        return
    if preset == "vulkan-perf":
        env.update(VULKAN_PERF_TRACE_ENV)
        return
    if preset == "vulkan-q3-stats":
        env.update(VULKAN_Q3_STATS_ENV)
        return
    raise ValueError(f"unknown trace preset: {preset}")


def is_timeout_like_exception(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError):
        return True
    text = repr(exc).lower()
    return "timed out" in text or "timeout" in text


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


def wait_for_server(base_url: str, timeout_s: float, proc: subprocess.Popen[str] | None = None) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited before becoming ready (exit code {proc.returncode})")
        for endpoint in ("/health", "/v1/models"):
            try:
                http_json("GET", base_url + endpoint, timeout=2.0)
                return
            except Exception as exc:  # noqa: BLE001 - status probing is best-effort
                last_error = str(exc)
        time.sleep(1.0)
    raise TimeoutError(f"server did not become ready within {timeout_s:.0f}s; last error: {last_error}")


def _close_proc_log(proc: subprocess.Popen[str]) -> None:
    log_file = getattr(proc, "_llama_log_file", None)
    if log_file is not None:
        try:
            log_file.close()
        except Exception:
            pass
        try:
            delattr(proc, "_llama_log_file")
        except Exception:
            pass


def terminate_process(proc: subprocess.Popen[str]) -> bool:
    """Ask llama-server to exit gently.

    Returns true if the process exited. By default this function deliberately
    avoids hard-killing ROCm servers: terminating HIP teardown mid-flight has
    correlated with Windows driver/device loss on this rig. Set
    LLAMA_BENCH_ALLOW_HARD_KILL=1 only when an operator explicitly wants the
    old behavior.
    """
    if proc.poll() is not None:
        _close_proc_log(proc)
        return True

    soft_timeout = float(os.environ.get("LLAMA_BENCH_SOFT_STOP_TIMEOUT", "180"))
    try:
        if os.name == "nt":
            # The server is a CREATE_NEW_PROCESS_GROUP child. CTRL_BREAK is
            # targetable to that group and server.cpp handles it gracefully.
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
    except Exception as exc:
        print(f"WARNING: failed to send graceful stop to llama-server pid={proc.pid}: {exc}")

    try:
        # llama-server teardown at large ctx legitimately takes tens of seconds
        # (freeing >10GB of VRAM + HIP contexts). Hard-killing mid-teardown leaves
        # the ROCm driver with orphaned allocations and correlates with the GPU
        # dropping off the bus (Windows code 43), so wait generously.
        proc.wait(timeout=soft_timeout)
        _close_proc_log(proc)
        return True
    except subprocess.TimeoutExpired:
        pass

    if os.environ.get("LLAMA_BENCH_ALLOW_HARD_KILL", "").strip() not in ("1", "true", "TRUE", "yes", "on"):
        print(
            f"WARNING: llama-server pid={proc.pid} did not exit after {soft_timeout:.0f}s; "
            "leaving it alive to avoid a hard ROCm teardown. Stop it manually after GPU activity settles, "
            "or set LLAMA_BENCH_ALLOW_HARD_KILL=1 for the old force-kill behavior."
        )
        return False

    print(f"WARNING: force-killing llama-server pid={proc.pid} because LLAMA_BENCH_ALLOW_HARD_KILL=1")
    proc.kill()
    try:
        proc.wait(timeout=30)
        _close_proc_log(proc)
        return True
    except subprocess.TimeoutExpired:
        return False


def split_server_extra(server_extra: str) -> list[str]:
    if not server_extra:
        return []
    return shlex.split(server_extra, posix=(os.name != "nt"))


def server_extra_has_flag(tokens: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(flag + "=") for token in tokens)


def bench_auto_fit_allowed() -> bool:
    return os.environ.get("LLAMA_BENCH_ALLOW_AUTO_FIT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def enforce_bench_safe_fit_args(tokens: list[str]) -> tuple[list[str], bool]:
    """Disable llama-server auto-fit during bench runs.

    ROCm/Windows can crash before real workload starts while auto-fit queries
    memory through hipMemGetInfo. Bench/autotune should use explicit params.
    """
    if bench_auto_fit_allowed():
        return tokens, False

    filtered: list[str] = []
    removed_fit = False
    skip_next = False
    fit_flags = {"-fit", "--fit"}

    for idx, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue

        if token in fit_flags:
            removed_fit = True
            if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("-"):
                skip_next = True
            continue

        if any(token.startswith(flag + "=") for flag in fit_flags):
            removed_fit = True
            continue

        filtered.append(token)

    filtered.extend(["-fit", "off"])
    return filtered, removed_fit


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
        "--flash-attn", "on" if args.flash_attn else "off",
        "-np", str(args.parallel),
        "-c", str(args.ctx_size),
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "--cache-type-k", args.cache_type_k,
        "--cache-type-v", args.cache_type_v,
    ]
    if args.gpu_layers >= 0:
        cmd.extend(["-ngl", str(args.gpu_layers)])
    if args.server_seed is not None:
        cmd.extend(["--seed", str(args.server_seed)])
    if args.no_warmup:
        cmd.append("--no-warmup")
    if args.disable_thinking and "--chat-template-kwargs" not in args.server_extra:
        cmd.extend([
            "--chat-template-kwargs",
            json.dumps({"enable_thinking": False, "preserve_thinking": False}, separators=(",", ":")),
        ])
    extra_tokens = split_server_extra(args.server_extra)
    if args.no_reuse:
        if not server_extra_has_flag(extra_tokens, "--cache-ram"):
            cmd.extend(["--cache-ram", "0"])
        if not server_extra_has_flag(extra_tokens, "--ctx-checkpoints"):
            cmd.extend(["--ctx-checkpoints", "0"])
    if extra_tokens:
        cmd.extend(extra_tokens)

    cmd, removed_fit = enforce_bench_safe_fit_args(cmd)
    if removed_fit:
        print(
            "WARNING: bench/autotune disabled auto-fit (-fit off) to avoid the ROCm "
            "hipMemGetInfo startup path. Set LLAMA_BENCH_ALLOW_AUTO_FIT=1 to allow it."
        )

    env = rocm_env()
    apply_trace_preset(env, args.trace_preset)
    active_benchmark_env = collect_active_benchmark_env(env)
    args.active_benchmark_env = active_benchmark_env
    if active_benchmark_env:
        print("Active benchmark environment:", json.dumps(active_benchmark_env, sort_keys=True))

    # Known issue: on some ROCm/Windows paths, forcing RDNA4 graph-opt can hang.
    # Current backend guard disables RDNA4 graph-opt by default. Treat only explicit
    # override mode as unsafe.
    server_bin_l = str(server_bin).lower()
    if (
        os.name == "nt"
        and "rocm" in server_bin_l
        and env.get("GGML_CUDA_GRAPH_OPT") == "1"
        and env.get("GGML_CUDA_ALLOW_RDNA4_GRAPH_OPT") == "1"
        and not args.allow_unsafe_graph_opt
    ):
        raise RuntimeError(
            "Unsafe override detected: GGML_CUDA_ALLOW_RDNA4_GRAPH_OPT=1 on ROCm/Windows. "
            "This mode can hang at request start. "
            "If you still want to test this unstable mode, pass --allow-unsafe-graph-opt."
        )

    log_dir = Path(args.out_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = log_dir / f"{args.label}.server.log"
    log_file = server_log.open("w", encoding="utf-8")
    print("Starting server:", " ".join(cmd))
    print("Server log:", server_log)
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    setattr(proc, "_llama_log_file", log_file)
    return proc


def run_task(
    base_url: str,
    task: dict[str, str],
    args: argparse.Namespace,
    proc: subprocess.Popen[str] | None = None,
) -> dict[str, Any]:
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
    hard_timeout = False
    terminated_server = False
    response: dict[str, Any] | None = None
    caught_exc: BaseException | None = None
    content = ""
    request_timeout = float(args.request_timeout)
    if args.task_hard_timeout > 0:
        request_timeout = min(request_timeout, float(args.task_hard_timeout))
    progress_stop = threading.Event()
    progress_thread: threading.Thread | None = None
    if not args.no_start:
        server_log = Path(args.out_dir) / f"{args.label}.server.log"
        progress_start_offset = server_log.stat().st_size if server_log.exists() else 0
        progress_thread = threading.Thread(
            target=tail_prompt_progress,
            args=(server_log, progress_stop, str(task["id"]), progress_start_offset),
            daemon=True,
        )
        progress_thread.start()
    try:
        response = http_json("POST", base_url + "/v1/chat/completions", payload, timeout=request_timeout)
        message = response["choices"][0].get("message", {})
        content = message.get("content") or message.get("reasoning_content") or ""
    except Exception as exc:  # noqa: BLE001 - benchmark records failures as rows
        caught_exc = exc
        error = repr(exc)
    finally:
        progress_stop.set()
        if progress_thread is not None:
            progress_thread.join(timeout=1.0)
    wall_s = time.perf_counter() - started

    if args.task_hard_timeout > 0 and (
        wall_s > args.task_hard_timeout or (caught_exc is not None and is_timeout_like_exception(caught_exc))
    ):
        hard_timeout = True
        timeout_error = f"TaskHardTimeoutExceeded(wall_s={wall_s:.2f}s, limit={args.task_hard_timeout:.2f}s)"
        error = timeout_error if not error else f"{error}; {timeout_error}"
        if proc is not None and proc.poll() is None:
            terminated_server = terminate_process(proc)

    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    timings = response.get("timings", {}) if isinstance(response, dict) else {}
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    total_tokens = usage.get("total_tokens")

    # Hard fail-slow guard for autotune and other scripted runs.
    # This catches cases where request-timeout is larger than desired wall SLA.
    if args.task_fail_timeout > 0 and wall_s > args.task_fail_timeout:
        timeout_error = (
            f"TaskTimeoutExceeded(wall_s={wall_s:.2f}s, limit={args.task_fail_timeout:.2f}s)"
        )
        error = timeout_error if not error else f"{error}; {timeout_error}"
        completion_tokens = None
        prompt_tokens = None
        total_tokens = None

    if hard_timeout:
        completion_tokens = None
        prompt_tokens = None
        total_tokens = None

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
        "hard_timeout": hard_timeout,
        "terminated_server": terminated_server,
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
        "label", "run", "task_id", "title", "wall_s", "prompt_tokens",
        "completion_tokens", "total_tokens", "completion_tps_wall",
        "response_chars", "error", "hard_timeout", "terminated_server",
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
        cold_rows = [row for row in rows if int(row.get("run") or 0) == 1]
        if cold_rows:
            cold_completion = sum(row.get("completion_tokens") or 0 for row in cold_rows)
            cold_wall = sum(row.get("wall_s") or 0.0 for row in cold_rows)
            cold_tps_values = [float(row["completion_tps_wall"]) for row in cold_rows if row.get("completion_tps_wall") is not None]

            print("Cold-only stats (run #1):")
            if cold_wall > 0 and cold_completion > 0:
                print(f"Cold-only aggregate completion TPS: {cold_completion / cold_wall:.2f}")
            if cold_tps_values:
                print(f"Cold-only mean task TPS: {statistics.mean(cold_tps_values):.2f}")
                print(f"Cold-only median task TPS: {statistics.median(cold_tps_values):.2f}")
                if len(cold_tps_values) > 1:
                    print(f"Cold-only task TPS stdev: {statistics.pstdev(cold_tps_values):.4f}")

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


def tail_prompt_progress(
    server_log: Path,
    stop_event: threading.Event,
    task_id: str,
    start_offset: int = 0,
) -> None:
    """Emit compact prompt-prefill progress lines while a blocking HTTP request runs."""
    offset = max(0, int(start_offset))
    buffer = ""
    total_tokens: int | None = None
    last_pct = -1.0

    progress_re = re.compile(
        r"prompt processing progress,\s*n_tokens =\s*(\d+),\s*"
        r"batch\.n_tokens =\s*(\d+),\s*progress =\s*([0-9.]+)"
    )
    done_re = re.compile(r"prompt processing done,\s*n_tokens =\s*(\d+),\s*batch\.n_tokens =\s*(\d+)")
    task_tokens_re = re.compile(r"task\.n_tokens =\s*(\d+)")

    while not stop_event.wait(0.25):
        try:
            if not server_log.exists():
                continue
            with server_log.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                data = f.read()
                offset = f.tell()
        except Exception:
            continue

        if not data:
            continue

        buffer += data
        lines = buffer.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            buffer = lines.pop()
        else:
            buffer = ""

        for raw_line in lines:
            line = raw_line.strip()
            task_match = task_tokens_re.search(line)
            if task_match:
                total_tokens = int(task_match.group(1))

            progress_match = progress_re.search(line)
            if progress_match:
                processed = int(progress_match.group(1))
                progress = float(progress_match.group(3))
                pct = max(0.0, min(100.0, progress * 100.0))
                if total_tokens is None and progress > 0:
                    total_tokens = max(processed, int(round(processed / progress)))
                if pct - last_pct >= 0.5:
                    total_text = str(total_tokens) if total_tokens else "?"
                    print(
                        f"PROMPT PROGRESS: task={task_id} pct={pct:.1f} "
                        f"tokens={processed}/{total_text} status=running",
                        flush=True,
                    )
                    last_pct = pct
                continue

            done_match = done_re.search(line)
            if done_match:
                processed = int(done_match.group(1))
                total_tokens = total_tokens or processed
                print(
                    f"PROMPT PROGRESS: task={task_id} pct=100.0 "
                    f"tokens={processed}/{total_tokens} status=done",
                    flush=True,
                )
                return


def parse_server_log_diagnostics(server_log: Path) -> dict[str, Any]:
    if not server_log.exists():
        return {
            "available": False,
            "error": f"server log not found: {server_log}",
        }

    text = server_log.read_text(encoding="utf-8", errors="replace")

    prompt_matches = re.findall(
        r"prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens \([^)]*,\s*([0-9.]+) tokens per second\)",
        text,
    )
    eval_matches = re.findall(
        r"(?m)^(?![^\n]*prompt eval time)[^\n]*\beval time =\s*([0-9.]+) ms /\s*(\d+) tokens "
        r"\([^)]*,\s*([0-9.]+) tokens per second\)",
        text,
    )
    total_matches = re.findall(r"total time =\s*([0-9.]+) ms /\s*(\d+) tokens", text)
    task_prompt_tokens = [int(x) for x in re.findall(r"task\.n_tokens =\s*(\d+)", text)]
    batch_chunks = [int(x) for x in re.findall(r"batch\.n_tokens =\s*(\d+)", text)]

    # Speculative (MTP/draft) acceptance: sum the cumulative per-task stats from
    # "statistics mtp: ... #gen tokens = G, #acc tokens = A". Acceptance =
    # accepted / generated draft tokens. A healthy MTP head is ~0.6-0.8; values
    # near 0 mean the draft head is fed the wrong hidden state or is otherwise
    # broken, so MTP is pure overhead.
    mtp_gen_tokens = sum(int(x) for x in re.findall(r"#gen tokens =\s*(\d+)", text))
    mtp_acc_tokens = sum(int(x) for x in re.findall(r"#acc tokens =\s*(\d+)", text))

    # Upstream-style speculative stats print one compact line per request:
    #   draft acceptance [rate] = 0.57895 ( 209 accepted / 361 generated)
    # Keep this separate from the older "#gen/#acc tokens" format to avoid
    # double-counting logs that may contain both in the future.
    draft_accept_matches = re.findall(
        r"draft acceptance(?: rate)? =\s*([0-9.]+)\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated\)",
        text,
    )
    if draft_accept_matches and mtp_gen_tokens == 0 and mtp_acc_tokens == 0:
        mtp_acc_tokens = sum(int(m[1]) for m in draft_accept_matches)
        mtp_gen_tokens = sum(int(m[2]) for m in draft_accept_matches)

    mtp_present = (
        mtp_gen_tokens > 0
        or bool(draft_accept_matches)
        or "statistics mtp" in text
        or "statistics dflash" in text
        or "speculative decoding context initialized" in text
    )
    mtp_acceptance = (mtp_acc_tokens / mtp_gen_tokens) if mtp_gen_tokens > 0 else 0.0

    # Newer stats also print "#verify tokens = N, eff acceptance = X%" — the honest
    # acceptance over actually-verified tokens (gen-based acceptance can read 100%
    # when rejected rounds re-draft the same token). Prefer the LAST occurrence
    # (stats lines are cumulative per request).
    eff_matches = re.findall(r"eff acceptance =\s*([0-9.]+)%", text)
    if eff_matches:
        mtp_acceptance = float(eff_matches[-1]) / 100.0

    prompt_tps = [float(m[2]) for m in prompt_matches]
    prompt_ms = [float(m[0]) for m in prompt_matches]
    total_ms = [float(m[0]) for m in total_matches]

    # llama.cpp prints "1000000.00 tokens per second" when the decode loop ran
    # for ~0 ms (1 token, eval time = 0.00). That sentinel is meaningless and,
    # if averaged in, makes decode_eval_tps useless for baseline-vs-spec
    # comparison. Keep only samples that actually decoded >1 token, and count
    # the degenerate ones separately so a broken spec run (e.g. MTP emitting a
    # single token) is visible instead of silently inflating the metric.
    decode_tps = []
    decode_ms = []
    decode_degenerate = 0
    for m in eval_matches:
        eval_ms = float(m[0])
        eval_tokens = int(m[1])
        eval_tps = float(m[2])
        if eval_tokens <= 1 or eval_ms <= 0.0 or eval_tps >= 100000.0:
            decode_degenerate += 1
            continue
        decode_tps.append(eval_tps)
        decode_ms.append(eval_ms)

    def _series_stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
        return {
            "count": float(len(values)),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "mean": round(statistics.mean(values), 4),
            "median": round(statistics.median(values), 4),
        }

    warning_lines = [ln.strip() for ln in text.splitlines() if re.search(r"\b(warning|failed|error)\b", ln, re.IGNORECASE)]

    return {
        "available": True,
        "path": str(server_log),
        "prompt_eval_tps": _series_stats(prompt_tps),
        "decode_eval_tps": _series_stats(decode_tps),
        "decode_degenerate_count": decode_degenerate,
        "decode_sample_count": len(decode_tps),
        "mtp_present": mtp_present,
        "mtp_gen_tokens": mtp_gen_tokens,
        "mtp_acc_tokens": mtp_acc_tokens,
        "mtp_acceptance": round(mtp_acceptance, 4),
        "prompt_eval_ms": _series_stats(prompt_ms),
        "decode_eval_ms": _series_stats(decode_ms),
        "total_ms": _series_stats(total_ms),
        "task_prompt_tokens": {
            "count": len(task_prompt_tokens),
            "mean": round(statistics.mean(task_prompt_tokens), 2) if task_prompt_tokens else 0.0,
            "max": max(task_prompt_tokens) if task_prompt_tokens else 0,
            "min": min(task_prompt_tokens) if task_prompt_tokens else 0,
        },
        "batch_chunks": {
            "count": len(batch_chunks),
            "mean": round(statistics.mean(batch_chunks), 2) if batch_chunks else 0.0,
            "max": max(batch_chunks) if batch_chunks else 0,
            "min": min(batch_chunks) if batch_chunks else 0,
        },
        "warning_lines": warning_lines[:30],
    }


def build_bottleneck_hints(rows: list[dict[str, Any]], server_diag: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    if not server_diag.get("available"):
        return ["server log diagnostics unavailable; check server startup and log path"]

    prompt_tps_mean = float(server_diag.get("prompt_eval_tps", {}).get("mean", 0.0) or 0.0)
    decode_tps_mean = float(server_diag.get("decode_eval_tps", {}).get("mean", 0.0) or 0.0)
    prompt_ms_mean = float(server_diag.get("prompt_eval_ms", {}).get("mean", 0.0) or 0.0)
    decode_ms_mean = float(server_diag.get("decode_eval_ms", {}).get("mean", 0.0) or 0.0)
    total_ms_mean = float(server_diag.get("total_ms", {}).get("mean", 0.0) or 0.0)

    degenerate = int(server_diag.get("decode_degenerate_count", 0) or 0)
    valid_decode = int(server_diag.get("decode_sample_count", 0) or 0)
    if degenerate > 0:
        hints.append(
            f"{degenerate} task(s) decoded <=1 token (sentinel ~1e6 tok/s) and were "
            "excluded from decode_eval_tps; likely an immediate stop or a broken "
            "spec/MTP loop (#gen drafts=0) -- raise --max-tokens and check the spec head"
        )
    if valid_decode == 0:
        hints.append("no valid decode samples; decode_eval_tps is unreliable for this run")

    if server_diag.get("mtp_present"):
        gen = int(server_diag.get("mtp_gen_tokens", 0) or 0)
        acc = float(server_diag.get("mtp_acceptance", 0.0) or 0.0)
        if gen == 0:
            hints.append("MTP active but 0 draft tokens generated; draft head never ran (check spec gating / max-tokens)")
        elif acc < 0.30:
            hints.append(
                f"MTP draft acceptance is low ({acc:.2%}); drafts are mostly rejected so "
                "MTP is net overhead -- check the hidden-state fed to the draft head "
                "(pre- vs post-norm) and the conditioning token/position"
            )

    if prompt_tps_mean > 0 and decode_tps_mean > 0:
        ratio = prompt_tps_mean / decode_tps_mean
        if ratio > 20:
            hints.append("prefill much faster than decode; prioritize decode/MMQ/FATTN path")
        elif ratio < 8:
            hints.append("prefill not much faster than decode; investigate prompt processing path")

    if total_ms_mean > 0:
        prefill_share = prompt_ms_mean / total_ms_mean
        decode_share = decode_ms_mean / total_ms_mean
        if prefill_share >= 0.70:
            hints.append("wall time dominated by prefill; focus on batch/ubatch chunking and prefill kernels")
        elif decode_share >= 0.30:
            hints.append("decode share is significant; test MMQ/FATTN routing hypotheses")

    row_tps = [float(r.get("completion_tps_wall") or 0.0) for r in rows if r.get("completion_tps_wall")]
    if len(row_tps) > 1:
        stdev = statistics.pstdev(row_tps)
        if stdev > 0.5:
            hints.append("high per-task TPS variance; check speculative stability and thermal/load interference")

    if server_diag.get("warning_lines"):
        hints.append("server log contains warning/error lines; inspect diagnostics markdown section")

    if not hints:
        hints.append("no obvious red flags from diagnostics; continue targeted kernel A/B experiments")

    return hints


def write_diagnostics_report(
    out_dir: Path,
    label: str,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    server_log = out_dir / f"{label}.server.log"
    diag_json = out_dir / f"{label}.diagnostics.json"
    diag_md = out_dir / f"{label}.diagnostics.md"

    server_diag = parse_server_log_diagnostics(server_log)
    hints = build_bottleneck_hints(rows, server_diag)
    aggregate_tps = aggregate_completion_tps(rows)
    active_benchmark_env = getattr(args, "active_benchmark_env", collect_active_benchmark_env())

    payload: dict[str, Any] = {
        "label": label,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "ctx": args.ctx_size,
            "batch": args.batch_size,
            "ubatch": args.ubatch_size,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
            "flash_attn": args.flash_attn,
            "spec_mode": infer_spec_mode(args.server_extra),
            "server_extra": args.server_extra,
            "no_reuse": args.no_reuse,
            "ubatch_split_policy": os.environ.get("LLAMA_UBATCH_SPLIT_POLICY", ""),
            "ubatch_shape_preferred": os.environ.get("LLAMA_UBATCH_SHAPE_PREFERRED", ""),
            "ubatch_shape_min_tail": os.environ.get("LLAMA_UBATCH_SHAPE_MIN_TAIL", ""),
            "active_benchmark_env": active_benchmark_env,
            "tasks": args.tasks,
            "runs": args.runs,
        },
        "run_metrics": {
            "aggregate_completion_tps": round(aggregate_tps, 4),
            "task_count": len(rows),
            "errors": sum(1 for row in rows if row.get("error")),
        },
        "server_log_diagnostics": server_diag,
        "hints": hints,
    }
    diag_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Diagnostics: {label}",
        "",
        "## Config",
        "",
        f"- ctx: {args.ctx_size}",
        f"- batch/ubatch: {args.batch_size}/{args.ubatch_size}",
        f"- kv: {args.cache_type_k}/{args.cache_type_v}",
        f"- flash_attn: {'on' if args.flash_attn else 'off'}",
        f"- spec_mode: {infer_spec_mode(args.server_extra)}",
        f"- no_reuse: {args.no_reuse}",
        f"- server_extra: {args.server_extra or '-'}",
        f"- active_benchmark_env: {json.dumps(active_benchmark_env, sort_keys=True) if active_benchmark_env else '-'}",
    ]

    split_policy = os.environ.get("LLAMA_UBATCH_SPLIT_POLICY", "")
    if split_policy:
        lines += [
            f"- ubatch_split_policy: {split_policy}",
            f"- ubatch_shape_preferred: {os.environ.get('LLAMA_UBATCH_SHAPE_PREFERRED', '-') or '-'}",
            f"- ubatch_shape_min_tail: {os.environ.get('LLAMA_UBATCH_SHAPE_MIN_TAIL', '-') or '-'}",
        ]

    lines += [
        "",
        "## Metrics",
        "",
        f"- aggregate_completion_tps: {aggregate_tps:.4f}",
        f"- task_count: {len(rows)}",
        f"- errors: {sum(1 for row in rows if row.get('error'))}",
        "",
        "## Server Log Summary",
        "",
    ]

    if server_diag.get("available"):
        p = server_diag.get("prompt_eval_tps", {})
        d = server_diag.get("decode_eval_tps", {})
        pm = server_diag.get("prompt_eval_ms", {})
        dm = server_diag.get("decode_eval_ms", {})
        lines += [
            f"- prompt_eval_tps mean/min/max: {p.get('mean', 0.0)}/{p.get('min', 0.0)}/{p.get('max', 0.0)}",
            f"- decode_eval_tps mean/min/max: {d.get('mean', 0.0)}/{d.get('min', 0.0)}/{d.get('max', 0.0)}",
            f"- prompt_eval_ms mean: {pm.get('mean', 0.0)}",
            f"- decode_eval_ms mean: {dm.get('mean', 0.0)}",
            f"- task_prompt_tokens mean: {server_diag.get('task_prompt_tokens', {}).get('mean', 0.0)}",
            f"- batch_chunks mean/max: {server_diag.get('batch_chunks', {}).get('mean', 0.0)}/{server_diag.get('batch_chunks', {}).get('max', 0)}",
        ]
        if server_diag.get("mtp_present"):
            lines.append(
                f"- mtp acceptance: {server_diag.get('mtp_acceptance', 0.0):.2%} "
                f"({server_diag.get('mtp_acc_tokens', 0)} acc / {server_diag.get('mtp_gen_tokens', 0)} gen draft tokens)"
            )
    else:
        lines.append(f"- unavailable: {server_diag.get('error', 'unknown error')}")

    lines += ["", "## Bottleneck Hints", ""]
    for hint in hints:
        lines.append(f"- {hint}")

    warnings = server_diag.get("warning_lines", []) if isinstance(server_diag, dict) else []
    if warnings:
        lines += ["", "## Warning/Error Lines (tail)", ""]
        for line in warnings[:20]:
            lines.append(f"- {line}")

    diag_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {diag_json}")
    print(f"Wrote {diag_md}")
    return {
        "diagnostics_json": diag_json.name,
        "diagnostics_md": diag_md.name,
    }


def infer_spec_mode(server_extra: str) -> str:
    tokens = [token.lower() for token in split_server_extra(server_extra)]
    for i, token in enumerate(tokens):
        if token == "--spec-type" and i + 1 < len(tokens):
            return normalize_spec_mode(tokens[i + 1])
        if token.startswith("--spec-type="):
            return normalize_spec_mode(token.split("=", 1)[1])
    return "none"


def required_local_speedup(target_share: float, target_total_speedup: float) -> float | None:
    denom = (1.0 / target_total_speedup) - (1.0 - target_share)
    if denom <= 0.0:
        return None
    return target_share / denom


def run_preflight_gate(args: argparse.Namespace) -> tuple[bool, str]:
    share = float(getattr(args, "preflight_share", 0.0))
    if share <= 0.0:
        return True, ""

    if share >= 1.0:
        return False, "preflight-share must be in (0,1)"

    goal = float(getattr(args, "preflight_goal_total_speedup", 1.01))
    if goal <= 1.0:
        return False, "preflight-goal-total-speedup must be > 1.0"

    candidate = float(getattr(args, "preflight_candidate_local_speedup", 1.0))
    if candidate <= 0.0:
        return False, "preflight-candidate-local-speedup must be > 0"

    req = required_local_speedup(share, goal)
    if req is None:
        return False, "preflight goal is impossible for the provided share"

    req_gain_pct = (req - 1.0) * 100.0
    cand_gain_pct = (candidate - 1.0) * 100.0
    passed = candidate >= req

    print(
        "[preflight] "
        f"share={share:.6f} goal_total={goal:.4f} "
        f"required_local={req:.4f} ({req_gain_pct:.2f}%) "
        f"candidate_local={candidate:.4f} ({cand_gain_pct:.2f}%) "
        f"verdict={'PASS' if passed else 'FAIL'}"
    )

    if not passed and bool(getattr(args, "preflight_enforce", False)):
        return False, "preflight ceiling gate failed and --preflight-enforce is set"

    return True, ""


def normalize_spec_mode(value: str) -> str:
    value = value.strip().lower()
    if value in {"draft-mtp", "draft_mtp"}:
        return "mtp"
    if value in {"mtp", "ngram-mtp", "ngram-mod", "draft", "none", "eagle", "eagle3"}:
        return value
    if value.startswith("eagle3"):
        return "eagle3"
    if value.startswith("eagle"):
        return "eagle"
    if value.startswith("ngram"):
        return value
    return "other" if value else "none"


def is_mtp_model_name(model_path: str) -> bool:
    name = Path(model_path).name.lower()
    if "-mtp" in name or name.endswith("mtp.gguf") or "nextn" in name:
        return True

    model_file = Path(model_path)
    if not model_file.exists() or not model_file.is_file():
        return False

    marker = b"nextn"
    chunk_size = 4 * 1024 * 1024
    overlap = len(marker) - 1
    tail = b""

    try:
        with model_file.open("rb") as file_obj:
            while True:
                chunk = file_obj.read(chunk_size)
                if not chunk:
                    break
                data = (tail + chunk).lower()
                if marker in data:
                    return True
                tail = data[-overlap:] if len(data) > overlap else data
    except Exception:
        return False

    return False


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


def _history_truthy(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _history_float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def _history_lane_key(row: dict[str, str]) -> str:
    backend = str(row.get("build_backend", "") or "-").strip() or "-"
    model = _model_display_name(row.get("model", "-"))
    ctx = str(row.get("ctx", "") or "-").strip() or "-"
    batch = str(row.get("batch", "") or "-").strip() or "-"
    ubatch = str(row.get("ubatch", "") or "-").strip() or "-"
    kv = f"{row.get('kv_k', '-') or '-'}/{row.get('kv_v', '-') or '-'}"
    spec = str(row.get("spec_mode", "") or "-").strip() or "-"
    reuse = "cold" if _history_truthy(row.get("no_reuse", "")) else "reuse"
    tasks = str(row.get("tasks", "") or "-").strip() or "-"
    task_ids = str(row.get("task_ids", "") or "").strip()
    task_part = f"{tasks}:{task_ids}" if task_ids else tasks
    max_tokens = str(row.get("max_tokens", "") or "-").strip() or "-"
    real_context = str(row.get("real_context_mode", "") or "-").strip() or "-"
    return f"{backend}|{model}|ctx{ctx}|b{batch}/ub{ubatch}|kv{kv}|spec={spec}|{reuse}|tasks={task_part}|max={max_tokens}|ctxsrc={real_context}"


def _history_metric_scope(row: dict[str, str]) -> str:
    mode = str(row.get("mode", "") or "").strip().lower()
    if mode == "autotune":
        return "autotune"
    if _history_truthy(row.get("no_reuse", "")):
        if _history_truthy(row.get("no_v2_prime_pass", "")):
            return "cold-first"
        return "cold/no-reuse"
    return "repeated/steady"


def _history_diag_mean(diag: dict[str, Any], key: str) -> str:
    value = diag.get(key, {}) if isinstance(diag, dict) else {}
    if not isinstance(value, dict):
        return ""
    mean = float(value.get("mean", 0.0) or 0.0)
    return f"{mean:.4f}" if mean > 0 else ""


def _fill_history_diagnostics(row: dict[str, str], out_dir: Path) -> None:
    if row.get("prompt_eval_tps") and row.get("decode_eval_tps"):
        return
    server_log_name = str(row.get("server_log_file", "") or "").strip()
    if not server_log_name:
        return
    diag = parse_server_log_diagnostics(out_dir / server_log_name)
    if not diag.get("available"):
        return
    row["prompt_eval_tps"] = row.get("prompt_eval_tps") or _history_diag_mean(diag, "prompt_eval_tps")
    row["decode_eval_tps"] = row.get("decode_eval_tps") or _history_diag_mean(diag, "decode_eval_tps")
    row["prompt_eval_ms"] = row.get("prompt_eval_ms") or _history_diag_mean(diag, "prompt_eval_ms")
    row["decode_eval_ms"] = row.get("decode_eval_ms") or _history_diag_mean(diag, "decode_eval_ms")


def _normalize_history_row(row: dict[str, Any], out_dir: Path | None = None) -> dict[str, str]:
    normalized = {k: str(row.get(k, "")) for k in HISTORY_FIELDS}
    if not normalized["run_id"]:
        normalized["run_id"] = _make_run_id(normalized)
    if not normalized["jsonl_file"] and normalized["label"] and normalized["mode"] == "single-run":
        normalized["jsonl_file"] = f"{normalized['label']}.jsonl"
    if not normalized["csv_file"] and normalized["label"] and normalized["mode"] == "single-run":
        normalized["csv_file"] = f"{normalized['label']}.csv"
    if not normalized["server_log_file"] and normalized["label"]:
        normalized["server_log_file"] = f"{normalized['label']}.server.log"
    if out_dir is not None:
        _fill_history_diagnostics(normalized, out_dir)
    if not normalized["metric_scope"]:
        normalized["metric_scope"] = _history_metric_scope(normalized)
    if not normalized["lane_key"]:
        normalized["lane_key"] = _history_lane_key(normalized)
    return normalized


def _read_history_csv(path: Path, out_dir: Path | None = None, *, enrich_diagnostics: bool = False) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(_normalize_history_row(row, out_dir=out_dir if enrich_diagnostics else None))
    return rows


def _dedupe_history_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for row in rows:
        key = row.get("run_id") or _make_run_id(row)
        if key not in deduped:
            order.append(key)
        deduped[key] = row
    return [deduped[key] for key in order]


def _canonical_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = [_normalize_history_row(row) for row in rows]
    retained = [row for row in normalized if row.get("timestamp", "") >= CANONICAL_HISTORY_START]
    return _dedupe_history_rows(retained)


def _best_rows_by_lane(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        if int(float(row.get("errors", "0") or 0)) != 0:
            continue
        tps = _history_float(row.get("aggregate_tps", "0"))
        if tps <= 0:
            continue
        lane = row.get("lane_key") or _history_lane_key(row)
        if lane not in best or tps > _history_float(best[lane].get("aggregate_tps", "0")):
            best[lane] = row
    return best


def write_canonical_bench_files(out_dir: Path, rows: list[dict[str, str]]) -> None:
    rows = _canonical_history_rows(rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_csv = out_dir / CANONICAL_BENCH_RUNS_CSV
    with runs_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in HISTORY_FIELDS})

    recent_rows = sorted(rows, key=lambda r: r.get("timestamp", ""), reverse=True)[:CANONICAL_RECENT_LIMIT]
    recent_lines = [
        "# Canonical Benchmark Recent Runs",
        "",
        "Автоматически обновляется `scripts/agent_workload_bench.py`.",
        "Содержит последние прогоны с метриками, полезными для быстрого сравнения.",
        "",
        f"Retention: `{CANONICAL_HISTORY_START}` and newer; latest {CANONICAL_RECENT_LIMIT} rows from `{CANONICAL_BENCH_RUNS_CSV}`.",
        "",
        "| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |",
        "|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in recent_rows:
        recent_lines.append(
            f"| {row.get('timestamp', '-')} | {row.get('metric_scope', '-')} | {row.get('build_backend', '-') or '-'} | "
            f"{row.get('label', '-')} | {_model_display_name(row.get('model', '-'))} | {row.get('ctx', '-')} | "
            f"{row.get('batch', '-')}/{row.get('ubatch', '-')} | {row.get('kv_k', '-')}/{row.get('kv_v', '-')} | "
            f"{row.get('spec_mode', '-')} | {row.get('aggregate_tps', '-')} | {row.get('prompt_eval_tps', '-') or '-'} | "
            f"{row.get('decode_eval_tps', '-') or '-'} | {row.get('errors', '-')} |"
        )
    (out_dir / CANONICAL_BENCH_RECENT_MD).write_text("\n".join(recent_lines) + "\n", encoding="utf-8")

    lane_lines = [
        "# Canonical Benchmark Best By Lane",
        "",
        "Автоматически обновляется `scripts/agent_workload_bench.py`.",
        "Lane key включает backend, модель, ctx, batch/ubatch, KV, spec, reuse/task/max-token context.",
        "",
        "| Lane | Timestamp | Label | TPS | Prompt tok/s | Decode tok/s | Build | Artifacts |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for lane, row in sorted(_best_rows_by_lane(rows).items(), key=lambda item: item[0].lower()):
        artifacts = ", ".join(
            p for p in [row.get("summary_file", ""), row.get("csv_file", ""), row.get("jsonl_file", ""), row.get("server_log_file", "")]
            if p
        )
        lane_lines.append(
            f"| {lane} | {row.get('timestamp', '-')} | {row.get('label', '-')} | {row.get('aggregate_tps', '-')} | "
            f"{row.get('prompt_eval_tps', '-') or '-'} | {row.get('decode_eval_tps', '-') or '-'} | "
            f"{row.get('build_name', '-') or '-'} | {artifacts or '-'} |"
        )
    (out_dir / CANONICAL_BENCH_LANES_MD).write_text("\n".join(lane_lines) + "\n", encoding="utf-8")

    print(f"Wrote {runs_csv}")
    print(f"Wrote {out_dir / CANONICAL_BENCH_RECENT_MD}")
    print(f"Wrote {out_dir / CANONICAL_BENCH_LANES_MD}")


def refresh_canonical_history(out_dir: Path) -> int:
    candidates = [out_dir / CANONICAL_BENCH_RUNS_CSV]
    candidates.extend(sorted(out_dir.glob("BENCH_HISTORY*.csv")))
    rows: list[dict[str, str]] = []
    for path in candidates:
        rows.extend(_read_history_csv(path, out_dir=out_dir))
    rows = _canonical_history_rows(rows)
    write_canonical_bench_files(out_dir, rows)
    return len(rows)


def append_canonical_history_entry(entry: dict[str, str], out_dir: Path) -> None:
    existing = _read_history_csv(out_dir / CANONICAL_BENCH_RUNS_CSV)
    existing.append(_normalize_history_row(entry, out_dir=out_dir))
    write_canonical_bench_files(out_dir, existing)


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
    rows.extend(_read_history_csv(history_csv, out_dir=history_dir))

    normalized_entry = _normalize_history_row(entry, out_dir=history_dir)
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
    append_canonical_history_entry(normalized_entry, history_dir)


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


def parse_autotune_extra_presets(values: str) -> list[tuple[str, str]]:
    raw_parts = [v.strip() for v in values.split("||") if v.strip()]
    if not raw_parts:
        raw_parts = ["base"]

    presets: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for idx, token in enumerate(raw_parts, start=1):
        name: str
        extra_args: str

        if "::" in token:
            left, right = token.split("::", 1)
            name = left.strip() or f"preset{idx}"
            extra_args = right.strip()
        else:
            lowered = token.lower()
            if lowered in {"base", "default", "none", "off", "-"}:
                name = "base"
                extra_args = ""
            else:
                name = f"extra{idx}"
                extra_args = token

        name = re.sub(r"\s+", "_", name).strip() or f"preset{idx}"

        key = (name, extra_args)
        if key in seen:
            continue
        seen.add(key)
        presets.append(key)

    return presets


def _is_drop_significant(prev_tps: float, curr_tps: float, drop_pct: float) -> bool:
    if prev_tps <= 0.0:
        return False
    ratio = (prev_tps - curr_tps) / prev_tps
    return ratio >= max(0.0, drop_pct)


def _autotune_config_key(
    ctx_size: int,
    batch_size: int,
    ubatch_size: int,
    kv_type: str,
    spec_mode: str,
    extra_name: str,
    extra_args: str,
) -> str:
    payload = {
        "ctx_size": int(ctx_size),
        "batch_size": int(batch_size),
        "ubatch_size": int(ubatch_size),
        "kv": str(kv_type),
        "spec_mode": str(spec_mode),
        "extra_preset": str(extra_name),
        "extra_args": str(extra_args),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _autotune_session_fingerprint(
    args: argparse.Namespace,
    ctx_values: list[int],
    batch_values: list[int],
    ubatch_values: list[int],
    kv_values: list[str],
    spec_values: list[str],
    extra_presets: list[tuple[str, str]],
) -> str:
    payload = {
        "version": 1,
        "model": str(args.model or ""),
        "server_bin": str(args.server_bin or ""),
        "tasks": str(args.tasks),
        "runs": int(args.runs),
        "max_tokens": int(args.max_tokens),
        "startup_timeout": float(args.startup_timeout),
        "request_timeout": float(args.request_timeout),
        "task_hard_timeout": float(args.task_hard_timeout),
        "task_fail_timeout": float(args.task_fail_timeout),
        "trace_preset": str(args.trace_preset),
        "no_reuse": bool(args.no_reuse),
        "real_context_mode": str(args.real_context_mode),
        "real_context_chars": int(args.real_context_chars),
        "ctx_values": [int(v) for v in ctx_values],
        "batch_values": [int(v) for v in batch_values],
        "ubatch_values": [int(v) for v in ubatch_values],
        "kv_values": [str(v) for v in kv_values],
        "spec_values": [str(v) for v in spec_values],
        "extra_presets": [{"name": n, "args": a} for n, a in extra_presets],
        "base_server_extra": str(args.server_extra or "").strip(),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8", errors="replace")).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_autotune_session(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: failed to load autotune session file {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        print(f"WARNING: invalid autotune session file format: {path}")
        return None
    return payload


def _best_from_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for row in summaries:
        try:
            has_error = int(row.get("errors", 0)) != 0
            agg_tps = float(row.get("aggregate_tps", 0.0))
        except Exception:
            continue
        if has_error:
            continue
        if best is None or agg_tps > float(best.get("aggregate_tps", 0.0)):
            best = row
    return best


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
    name = f"AutoTune 130k {model_name}"
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

    extra_bits = []
    base_server_extra = str(getattr(args, "server_extra", "") or "").strip()
    best_extra_args = str(best.get("extra_args", "") or "").strip()
    if base_server_extra:
        extra_bits.append(base_server_extra)
    if best_extra_args:
        extra_bits.append(best_extra_args)

    current_extra = " ".join(extra_bits).strip()
    if not server_extra_has_flag(split_server_extra(current_extra), "--spec-type"):
        spec_mode = str(best.get("spec_mode", "") or "").strip()
        if spec_mode == "none":
            extra_bits.append("--spec-type none")
        elif spec_mode == "ngram-mod":
            extra_bits.append("--spec-type ngram-mod")
            extra_bits.append(f"--spec-ngram-mod-n-min {args.autotune_ngram_min}")
            extra_bits.append(f"--spec-ngram-mod-n-match {args.autotune_ngram_match}")
            extra_bits.append(f"--spec-ngram-mod-n-max {args.autotune_ngram_max}")
        elif spec_mode == "ngram-mtp":
            extra_bits.append("--spec-type ngram-mtp")
            extra_bits.append(f"--spec-ngram-mod-n-min {args.autotune_ngram_min}")
            extra_bits.append(f"--spec-ngram-mod-n-match {args.autotune_ngram_match}")
            extra_bits.append(f"--spec-ngram-mod-n-max {args.autotune_ngram_max}")
            extra_bits.append(f"--spec-draft-n-max {args.autotune_mtp_draft_n_max}")
        elif spec_mode == "mtp":
            extra_bits.append("--spec-type draft-mtp")
            extra_bits.append(f"--spec-draft-n-max {args.autotune_mtp_draft_n_max}")
        elif spec_mode:
            extra_bits.append(f"--spec-type {spec_mode}")

    extra_tokens = split_server_extra(" ".join(extra_bits).strip())
    if server_extra_has_flag(extra_tokens, "--no-mmap"):
        preset["no_mmap"] = True
        extra_tokens = [token for token in extra_tokens if token != "--no-mmap"]
    if extra_tokens:
        preset["extra_args"] = " ".join(extra_tokens)

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
    parser.add_argument("--tasks", choices=["quick", "full", "v2", "v2-mini", "v2-review"], default="quick",
                        help="prompt set: quick (v1 short tasks), full (v1 extended), v2 (realistic agentic-flow tasks), v2-mini (v2_code_review + v2_write_function), v2-review (only v2_code_review)")
    parser.add_argument(
        "--task-ids",
        default="",
        help="comma-separated task IDs to run from the selected task set (e.g. 'review_bug,patch_sim')",
    )
    parser.add_argument("--runs", type=int, default=1, help="repeat each task N times")

    parser.add_argument("--no-start", action="store_true", help="use an already running server")
    parser.add_argument("--server-bin", default=None, help="path to llama-server executable")
    parser.add_argument("--model", default=None, help="path to GGUF model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="server port; 0 picks a free port when starting a server")
    parser.add_argument("--api-model", default="local-model")
    parser.add_argument("--server-extra", default="", help="extra llama-server args, e.g. '--spec-type draft-mtp --spec-draft-n-max 3'")
    parser.add_argument("--build-id", default="", help="build registry ID linked to this benchmark run")
    parser.add_argument("--artifact-mode", choices=["full", "unified"], default="full",
                        help="artifact mode: full writes per-run CSV/JSONL, unified writes only history")
    parser.add_argument("--history-version", default="v1",
                        help="history namespace/version, e.g. v1 (default), v2 -> BENCH_HISTORY_V2.csv/.md")
    parser.add_argument("--cleanup-legacy-artifacts", action="store_true",
                        help="cleanup old per-run benchmark artifacts in out-dir")
    parser.add_argument("--refresh-canonical-history", action="store_true",
                        help="rebuild BENCH_RUNS.csv, BENCH_RECENT.md, and BENCH_LANES.md from existing history CSV files")
    parser.add_argument("--cleanup-apply", action="store_true",
                        help="apply deletion for --cleanup-legacy-artifacts (default is dry-run)")
    parser.add_argument(
        "--cleanup-keep-patterns",
        default=r"BENCH_(HISTORY(_V[0-9]+)?|RUNS|RECENT|LANES)\.(csv|md)|BASELINE_.*|.*LOCK.*|.*autotune-summary\.(md|csv|json)|ROCM_BENCH_COMPARISON\.md",
        help="regex patterns (pipe-separated) for artifact names to keep during cleanup",
    )

    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--ctx-size", type=int, default=131072)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ubatch-size", type=int, default=128)
    parser.add_argument("--cache-type-k", default="q4_0")
    parser.add_argument("--cache-type-v", default="q4_0")
    parser.add_argument("--flash-attn", action=argparse.BooleanOptionalAction, default=True)
    warmup_group = parser.add_mutually_exclusive_group()
    warmup_group.add_argument(
        "--no-warmup",
        dest="no_warmup",
        action="store_true",
        default=True,
        help="disable llama-server startup warmup (default)",
    )
    warmup_group.add_argument(
        "--warmup",
        "--no-no-warmup",
        dest="no_warmup",
        action="store_false",
        help="enable llama-server startup warmup (--no-no-warmup is a compatibility alias)",
    )
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=False,
                        help="disable model thinking by forcing chat-template kwargs; default keeps thinking enabled")

    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--real-context-mode",
        choices=["off", "repo-snapshot"],
        default="repo-snapshot",
        help="inject large incoming context into each task prompt",
    )
    parser.add_argument(
        "--real-context-chars",
        type=int,
        default=24576,
        help="target character budget for injected incoming context; 0 uses ctx-aware safe cap when repo-snapshot mode is enabled",
    )
    parser.add_argument(
        "--real-context-safe-fill",
        type=float,
        default=0.88,
        help="target fraction of ctx budget allocated to incoming context tokens (0..1)",
    )
    parser.add_argument(
        "--real-context-reserve-tokens",
        type=int,
        default=2048,
        help="extra token reserve to avoid overflow from chat template/system overhead",
    )
    parser.add_argument(
        "--real-context-chars-per-token",
        type=float,
        default=2.6,
        help="conservative conversion from token budget to char budget for snapshot injection",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument(
        "--task-hard-timeout",
        type=float,
        default=45.0,
        help="abort a task request after this many seconds and terminate a server started by this script; 0 disables",
    )
    parser.add_argument(
        "--task-fail-timeout",
        type=float,
        default=0.0,
        help="mark task as failed if wall time exceeds this threshold in seconds; 0 disables",
    )
    parser.add_argument(
        "--preflight-share",
        type=float,
        default=0.0,
        help="target center wall share in (0,1); 0 disables preflight ceiling gate",
    )
    parser.add_argument(
        "--preflight-goal-total-speedup",
        type=float,
        default=1.02,
        help="desired end-to-end speedup for preflight gate, e.g. 1.02 for +2%%",
    )
    parser.add_argument(
        "--preflight-candidate-local-speedup",
        type=float,
        default=1.0,
        help="estimated candidate local speedup for the target center, e.g. 1.06",
    )
    parser.add_argument(
        "--preflight-enforce",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="abort before benchmark run when preflight gate fails",
    )
    parser.add_argument(
        "--no-reuse",
        dest="no_reuse",
        action="store_true",
        default=True,
        help="disable llama-server prompt cache and context checkpoints for cold prompt-heavy measurements (default for the active 130k lane)",
    )
    parser.add_argument(
        "--reuse",
        dest="no_reuse",
        action="store_false",
        help="allow llama-server prompt cache/checkpoints for explicit repeated-session probes",
    )
    parser.add_argument(
        "--allow-ctx-above-16k",
        action="store_true",
        help="legacy compatibility flag; the active policy now allows ctx up to 131072 by default",
    )
    parser.add_argument("--server-seed", type=int, default=42,
                        help="llama-server seed for deterministic decoding; set to -1 to disable fixed seed")
    parser.add_argument("--stats-ignore-first-run", action=argparse.BooleanOptionalAction, default=False,
                        help="print additional warm-only metrics that exclude run #1 (cold/probing phase)")
    parser.add_argument("--write-diagnostics", action=argparse.BooleanOptionalAction, default=True,
                        help="write per-run diagnostics json/md parsed from server log")
    parser.add_argument("--v2-prime-pass", action=argparse.BooleanOptionalAction, default=False,
                        help="opt into one unmeasured priming pass for v2/v2-mini when speculative ngram-mod and runs=1")
    parser.add_argument("--keep-server", action="store_true", help="do not stop server started by this script")
    parser.add_argument(
        "--background-server-policy",
        choices=["warn", "fail", "ignore"],
        default="fail",
        help="what to do if llama-server is already running before benchmark start",
    )
    parser.add_argument(
        "--allow-unsafe-graph-opt",
        action="store_true",
        help="allow running with GGML_CUDA_GRAPH_OPT=1 on ROCm/Windows despite known hang risk",
    )
    parser.add_argument(
        "--trace-preset",
        choices=["none", "kernel-full", "vulkan-routes", "vulkan-perf", "vulkan-q3-stats"],
        default="none",
        help="enable a built-in trace environment preset for the started server",
    )

    parser.add_argument("--autotune", action="store_true", help="run parameter sweep")
    parser.add_argument("--autotune-min-ctx", type=int, default=131072, help="minimum context for autotune")
    parser.add_argument("--autotune-ctx-values", default="131072", help="comma-separated ctx values")
    parser.add_argument("--autotune-batch-values", default="256,512,768,1024", help="comma-separated batch values")
    parser.add_argument("--autotune-ubatch-values", default="64,128,256", help="comma-separated ubatch values")
    parser.add_argument("--autotune-kv-values", default="q4_0", help="comma-separated kv cache values")
    parser.add_argument("--autotune-spec-values", default="none", help="comma-separated speculative modes")
    parser.add_argument(
        "--autotune-extra-presets",
        default="base",
        help=(
            "server extra presets separated by '||'; use 'base' for no extra args; "
            "format 'name::args' or plain args token"
        ),
    )
    parser.add_argument("--autotune-ngram-min", type=int, default=12)
    parser.add_argument("--autotune-ngram-match", type=int, default=16)
    parser.add_argument("--autotune-ngram-max", type=int, default=32)
    parser.add_argument("--autotune-mtp-draft-n-max", type=int, default=8)
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
    parser.add_argument(
        "--autotune-session-file",
        default="",
        help="path to autotune session checkpoint file (default: <out-dir>/<label>-autotune-session.json)",
    )
    parser.add_argument(
        "--autotune-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume unfinished autotune session when checkpoint exists",
    )
    parser.add_argument(
        "--autotune-reset-session",
        action="store_true",
        help="discard previous session checkpoint before autotune run",
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
            wait_for_server(base_url, args.startup_timeout, proc=proc)
        else:
            if args.port == 0:
                args.port = 8080
            base_url = f"http://{args.host}:{args.port}"
            wait_for_server(base_url, 10.0)

        rows: list[dict[str, Any]] = []

        # Optional steady-state probe: this is disabled by default because the cold-first
        # prompt-heavy lane must not pre-fill speculative state with an unmeasured request.
        should_prime_v2 = (
            args.v2_prime_pass
            and args.tasks in ("v2", "v2-mini", "v2-review")
            and infer_spec_mode(args.server_extra) == "ngram-mod"
            and args.runs == 1
        )
        if should_prime_v2:
            print("[prime] running unmeasured v2 priming pass ...", flush=True)
            for task in tasks:
                prime_row = run_task(base_url, task, args, proc=proc)
                if prime_row["error"]:
                    raise RuntimeError(f"v2 prime pass failed on {task['id']}: {prime_row['error']}")

        for run_idx in range(args.runs):
            for task in tasks:
                print(f"[{run_idx + 1}/{args.runs}] {task['id']} ...", flush=True)
                row = run_task(base_url, task, args, proc=proc)
                row["run"] = run_idx + 1
                rows.append(row)
                if row["error"]:
                    print(f"  error: {row['error']}")
                else:
                    print(f"  {row['wall_s']:.2f}s, completion_tokens={row['completion_tokens']}")
                if row.get("hard_timeout"):
                    print("  aborting suite after hard task timeout")
                    return rows
        return rows
    finally:
        if proc is not None and not args.keep_server:
            terminate_process(proc)


def main() -> int:
    install_graceful_stop_handlers()
    args = parse_args()
    if args.ctx_size > PRIMARY_MAX_CTX and not args.allow_ctx_above_16k:
        print(
            f"ERROR: ctx-size > {PRIMARY_MAX_CTX} is above the active {PRIMARY_CTX_LABEL} benchmark policy. "
            "Use --allow-ctx-above-16k only for explicit over-130k exploratory runs."
        )
        return 4

    if args.server_seed is not None and args.server_seed < 0:
        args.server_seed = None

    spec_mode = infer_spec_mode(args.server_extra)
    if spec_mode not in ("", "none") and args.max_tokens < 64:
        print(
            f"WARNING: spec_mode='{spec_mode}' with --max-tokens {args.max_tokens}. "
            "Short generations are prefill-dominated and won't exercise the draft/accept "
            "loop, so baseline-vs-spec decode comparison is unreliable. "
            "Use --max-tokens 128+ for honest spec benchmarking."
        )

    out_dir = Path(args.out_dir)
    history_csv_name, history_md_name = resolve_history_names(args.history_version)
    build_meta = resolve_build_metadata(args.build_id, args.server_bin)
    args.build_id = build_meta["build_id"]

    if args.cleanup_legacy_artifacts:
        protected_history = {
            HISTORY_CSV_NAME,
            HISTORY_MD_NAME,
            history_csv_name,
            history_md_name,
            CANONICAL_BENCH_RUNS_CSV,
            CANONICAL_BENCH_RECENT_MD,
            CANONICAL_BENCH_LANES_MD,
        }
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

    if args.refresh_canonical_history:
        count = refresh_canonical_history(out_dir)
        print(f"Canonical history refreshed from {count} row(s)")
        return 0

    if args.tasks in ("v2", "v2-mini", "v2-review"):
        tasks = TASKS_V2
        if args.tasks == "v2-mini":
            selected_ids = {"v2_write_function"}
            tasks = [task for task in TASKS_V2 if task["id"] in selected_ids]
        elif args.tasks == "v2-review":
            selected_ids = {"v2_code_review"}
            tasks = [task for task in TASKS_V2 if task["id"] in selected_ids]
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

    if args.task_ids.strip():
        requested_ids = [x.strip() for x in args.task_ids.split(",") if x.strip()]
        requested_set = set(requested_ids)
        available_ids = {task["id"] for task in tasks}
        unknown_ids = [task_id for task_id in requested_ids if task_id not in available_ids]
        if unknown_ids:
            print(
                "ERROR: unknown --task-ids for selected task set: "
                + ", ".join(unknown_ids)
                + ". Available: "
                + ", ".join(sorted(available_ids))
            )
            return 5
        tasks = [task for task in tasks if task["id"] in requested_set]
        if not tasks:
            print("ERROR: --task-ids filter left no tasks to run")
            return 5

    real_context_ctx_size = int(args.ctx_size)
    if args.autotune:
        try:
            autotune_ctx_candidates = parse_int_csv(args.autotune_ctx_values)
        except ValueError as exc:
            print(f"ERROR: invalid --autotune-ctx-values: {exc}")
            return 4
        autotune_ctx_candidates = [v for v in autotune_ctx_candidates if v >= args.autotune_min_ctx]
        if autotune_ctx_candidates:
            real_context_ctx_size = min(autotune_ctx_candidates)

    if args.real_context_mode == "repo-snapshot":
        safe_fill = min(max(float(args.real_context_safe_fill), 0.05), 0.95)
        usable_tokens = int(real_context_ctx_size * safe_fill) - int(args.max_tokens) - int(args.real_context_reserve_tokens)
        usable_tokens = max(1024, usable_tokens)
        safe_char_cap = int(usable_tokens * float(args.real_context_chars_per_token))

        requested_chars = int(args.real_context_chars)
        effective_chars = safe_char_cap if requested_chars <= 0 else min(requested_chars, safe_char_cap)

        prefix, chars, files = build_repo_snapshot_prefix(ROOT, effective_chars)
        tasks = apply_real_context_prefix(tasks, prefix)
        print(
            f"Real context injection: mode=repo-snapshot chars={chars} files={files} "
            f"requested={requested_chars} safe_cap={safe_char_cap} effective={effective_chars} "
            f"ctx_cap={real_context_ctx_size}"
        )

    preflight_ok, preflight_error = run_preflight_gate(args)
    if not preflight_ok:
        print(f"ERROR: {preflight_error}")
        return 6

    if not args.autotune:
        try:
            rows = run_suite(args, tasks)
        except Exception as exc:  # noqa: BLE001 - print clean error instead of traceback for operator workflows
            print(f"ERROR: {exc}")
            if "background-server" in str(exc) or "already running llama-server" in str(exc):
                print("Stop background server(s) or rerun with --background-server-policy warn/ignore")
            return 3
        artifacts = write_results(
            rows,
            out_dir,
            args.label,
            args.artifact_mode,
            stats_ignore_first_run=args.stats_ignore_first_run,
        )
        diagnostics_artifacts = {"diagnostics_json": "", "diagnostics_md": ""}
        if args.write_diagnostics:
            diagnostics_artifacts = write_diagnostics_report(out_dir, args.label, args, rows)
        server_diag = parse_server_log_diagnostics(out_dir / f"{args.label}.server.log")

        aggregate_tps = aggregate_completion_tps(rows)
        if server_diag.get("available"):
            print(
                f"RUN RESULT: aggregate_tps={aggregate_tps:.2f} "
                f"prompt_tps={float(server_diag.get('prompt_eval_tps', {}).get('mean', 0.0) or 0.0):.2f} "
                f"decode_tps={float(server_diag.get('decode_eval_tps', {}).get('mean', 0.0) or 0.0):.2f}"
            )
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
                "task_ids": args.task_ids,
                "runs": args.runs,
                "ctx": args.ctx_size,
                "batch": args.batch_size,
                "ubatch": args.ubatch_size,
                "kv_k": args.cache_type_k,
                "kv_v": args.cache_type_v,
                "spec_mode": infer_spec_mode(args.server_extra),
                "extra_preset": "base",
                "extra_args": args.server_extra,
                "no_reuse": 1 if args.no_reuse else 0,
                "gpu_layers": args.gpu_layers,
                "parallel": args.parallel,
                "flash_attn": "on" if args.flash_attn else "off",
                "max_tokens": args.max_tokens,
                "real_context_mode": args.real_context_mode,
                "real_context_chars": args.real_context_chars,
                "real_context_safe_fill": args.real_context_safe_fill,
                "no_v2_prime_pass": 0 if args.v2_prime_pass else 1,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "aggregate_tps": f"{aggregate_tps:.4f}",
                "mean_task_tps": f"{statistics.mean(tps_values):.4f}" if tps_values else "0.0000",
                "prompt_eval_tps": _history_diag_mean(server_diag, "prompt_eval_tps"),
                "decode_eval_tps": _history_diag_mean(server_diag, "decode_eval_tps"),
                "prompt_eval_ms": _history_diag_mean(server_diag, "prompt_eval_ms"),
                "decode_eval_ms": _history_diag_mean(server_diag, "decode_eval_ms"),
                "errors": sum(1 for row in rows if row.get("error")),
                "best_config": "",
                "jsonl_file": artifacts.get("jsonl_file", ""),
                "csv_file": artifacts.get("csv_file", ""),
                "summary_file": diagnostics_artifacts.get("diagnostics_md", ""),
                "server_log_file": f"{args.label}.server.log",
            },
            out_dir,
            history_csv_name=history_csv_name,
            history_md_name=history_md_name,
        )
        return 0 if not any(row["error"] for row in rows) else 2

    ctx_values = [v for v in parse_int_csv(args.autotune_ctx_values) if v >= args.autotune_min_ctx]
    if not args.allow_ctx_above_16k:
        over_limit = [v for v in ctx_values if v > PRIMARY_MAX_CTX]
        if over_limit or args.autotune_min_ctx > PRIMARY_MAX_CTX:
            print(
                f"ERROR: autotune ctx values above {PRIMARY_MAX_CTX} are above the active {PRIMARY_CTX_LABEL} policy. "
                "Use --allow-ctx-above-16k only for explicit over-130k exploratory runs."
            )
            return 4
    batch_values = parse_int_csv(args.autotune_batch_values)
    ubatch_values = parse_int_csv(args.autotune_ubatch_values)
    kv_values = parse_text_csv(args.autotune_kv_values)
    spec_values = [v.lower() for v in parse_text_csv(args.autotune_spec_values)]
    extra_presets = parse_autotune_extra_presets(args.autotune_extra_presets)

    configs = list(product(ctx_values, batch_values, ubatch_values, kv_values, spec_values, extra_presets))
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
    completed_keys: set[str] = set()

    ctx_values_sorted = sorted(set(int(v) for v in ctx_values))
    kv_values_sorted = list(dict.fromkeys(kv_values))
    spec_values_sorted = list(dict.fromkeys(spec_values))
    extra_presets_sorted = extra_presets
    ubatch_values_sorted = sorted(set(int(v) for v in ubatch_values))
    batch_values_sorted = sorted(set(int(v) for v in batch_values))

    session_file = Path(str(args.autotune_session_file).strip()) if str(args.autotune_session_file).strip() else (out_dir / f"{args.label}-autotune-session.json")
    session_fingerprint = _autotune_session_fingerprint(
        args,
        ctx_values_sorted,
        batch_values_sorted,
        ubatch_values_sorted,
        kv_values_sorted,
        spec_values_sorted,
        extra_presets_sorted,
    )
    print(f"Autotune session file: {session_file}")

    if args.autotune_reset_session and session_file.exists():
        try:
            session_file.unlink()
            print("Autotune session reset: removed previous checkpoint")
        except Exception as exc:
            print(f"WARNING: failed to remove previous autotune session: {exc}")

    if args.autotune_resume and not args.autotune_reset_session:
        session_payload = _load_autotune_session(session_file)
        if session_payload is not None:
            old_fingerprint = str(session_payload.get("fingerprint", "")).strip()
            completed_flag = bool(session_payload.get("completed", False))

            if completed_flag:
                print("Autotune resume: existing session is already completed; starting fresh run")
            elif old_fingerprint and old_fingerprint != session_fingerprint:
                print("Autotune resume: session fingerprint mismatch; starting fresh run")
            else:
                loaded_summaries = session_payload.get("summaries", [])
                if isinstance(loaded_summaries, list):
                    summaries = [row for row in loaded_summaries if isinstance(row, dict)]

                loaded_best = session_payload.get("best")
                if isinstance(loaded_best, dict):
                    best = loaded_best

                loaded_keys = session_payload.get("completed_keys", [])
                if isinstance(loaded_keys, list):
                    completed_keys = {str(v) for v in loaded_keys if str(v).strip()}

                if not completed_keys and summaries:
                    for row in summaries:
                        try:
                            key = _autotune_config_key(
                                int(row.get("ctx_size", 0)),
                                int(row.get("batch_size", 0)),
                                int(row.get("ubatch_size", 0)),
                                str(row.get("kv", "")),
                                str(row.get("spec_mode", "")),
                                str(row.get("extra_preset", "base")),
                                str(row.get("extra_args", "")),
                            )
                            completed_keys.add(key)
                        except Exception:
                            continue

                executed = max(
                    executed,
                    int(session_payload.get("executed", 0) or 0),
                    len(completed_keys),
                    len(summaries),
                )
                skipped_by_prune = max(
                    skipped_by_prune,
                    int(session_payload.get("skipped_by_prune", 0) or 0),
                )

                if best is None:
                    best = _best_from_summaries(summaries)

                print(
                    "Autotune resume: loaded previous progress "
                    f"({len(completed_keys)} done / {total_configs} total)"
                )

    def save_autotune_session(completed: bool = False) -> None:
        payload = {
            "version": 1,
            "label": args.label,
            "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "completed": bool(completed),
            "fingerprint": session_fingerprint,
            "total_configs": total_configs,
            "executed": executed,
            "skipped_by_prune": skipped_by_prune,
            "completed_keys": sorted(completed_keys),
            "summaries": summaries,
            "best": best,
            "grid": {
                "ctx_values": ctx_values_sorted,
                "batch_values": batch_values_sorted,
                "ubatch_values": ubatch_values_sorted,
                "kv_values": kv_values_sorted,
                "spec_values": spec_values_sorted,
                "extra_presets": [{"name": n, "args": a} for n, a in extra_presets_sorted],
            },
        }
        _write_json_atomic(session_file, payload)

    save_autotune_session(completed=False)

    for spec_mode in spec_values_sorted:
        for extra_name, extra_args in extra_presets_sorted:
            for ctx_size in ctx_values_sorted:
                for kv_type in kv_values_sorted:
                    ubatch_drop_streak = 0
                    prev_ubatch_best_tps = 0.0

                    for ubatch_size in ubatch_values_sorted:
                        prev_batch_tps = 0.0
                        batch_drop_streak = 0
                        ubatch_best_tps = 0.0
                        ran_any_batch_for_ubatch = False

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
                            if extra_args:
                                extra_bits.append(extra_args)

                            # an extra preset may pin its own draft budget
                            # (e.g. "mtp-n2::--spec-draft-n-max 2"); the sweep
                            # default must not override it
                            has_draft_n_override = server_extra_has_flag(
                                split_server_extra(" ".join(extra_bits)), "--spec-draft-n-max"
                            )

                            if spec_mode == "ngram-mod":
                                extra_bits.append("--spec-type ngram-mod")
                                extra_bits.append(f"--spec-ngram-mod-n-min {args.autotune_ngram_min}")
                                extra_bits.append(f"--spec-ngram-mod-n-match {args.autotune_ngram_match}")
                                extra_bits.append(f"--spec-ngram-mod-n-max {args.autotune_ngram_max}")
                            elif spec_mode in {"none", ""}:
                                extra_bits.append("--spec-type none")
                            elif spec_mode == "ngram-mtp":
                                extra_bits.append("--spec-type ngram-mtp")
                                extra_bits.append(f"--spec-ngram-mod-n-min {args.autotune_ngram_min}")
                                extra_bits.append(f"--spec-ngram-mod-n-match {args.autotune_ngram_match}")
                                extra_bits.append(f"--spec-ngram-mod-n-max {args.autotune_ngram_max}")
                                if not has_draft_n_override:
                                    extra_bits.append(f"--spec-draft-n-max {args.autotune_mtp_draft_n_max}")
                            elif spec_mode == "mtp":
                                extra_bits.append("--spec-type draft-mtp")
                                if not has_draft_n_override:
                                    extra_bits.append(f"--spec-draft-n-max {args.autotune_mtp_draft_n_max}")
                            elif spec_mode not in {"none", ""}:
                                extra_bits.append(f"--spec-type {spec_mode}")

                            run_args.server_extra = " ".join(bit for bit in extra_bits if bit).strip()
                            config_key = _autotune_config_key(
                                int(ctx_size),
                                int(batch_size),
                                int(ubatch_size),
                                str(kv_type),
                                str(spec_mode),
                                str(extra_name),
                                str(extra_args),
                            )

                            if config_key in completed_keys:
                                continue

                            mtp_draft_n = ""
                            if spec_mode in {"mtp", "ngram-mtp"}:
                                extra_token_list = split_server_extra(run_args.server_extra)
                                for token_idx, token in enumerate(extra_token_list[:-1]):
                                    if token == "--spec-draft-n-max":
                                        mtp_draft_n = extra_token_list[token_idx + 1]

                            print(
                                f"Autotune [{run_idx}/{total_configs}]: ctx={ctx_size}, b={batch_size}, ub={ubatch_size}, "
                                f"kv={kv_type}, spec={spec_mode}, extra={extra_name}, draftn={mtp_draft_n or '-'}"
                            )
                            startup_error = ""
                            try:
                                rows = run_suite(run_args, tasks)
                            except TimeoutError as exc:
                                startup_error = str(exc)
                                rows = []
                                print(f"CONFIG FAILED (startup timeout): {startup_error}")
                            except RuntimeError as exc:
                                msg = str(exc)
                                if msg.startswith("server exited before becoming ready"):
                                    startup_error = msg
                                    rows = []
                                    print(f"CONFIG FAILED (startup/runtime): {startup_error}")
                                else:
                                    # Most RuntimeError failures here are operator/environment-level issues.
                                    # Keep existing fail-fast behavior for them.
                                    print(f"ERROR: {msg}")
                                    return 3

                            ran_any_batch_for_ubatch = True
                            executed += 1
                            write_results(
                                rows,
                                out_dir,
                                run_args.label,
                                args.artifact_mode,
                                stats_ignore_first_run=args.stats_ignore_first_run,
                            )
                            agg_tps = aggregate_completion_tps(rows)
                            has_error = bool(startup_error) or any(row.get("error") for row in rows)
                            task_tps_values = [
                                float(r["completion_tps_wall"])
                                for r in rows
                                if r.get("completion_tps_wall") is not None
                            ]
                            cfg_diag = parse_server_log_diagnostics(out_dir / f"{run_args.label}.server.log")
                            if cfg_diag.get("available"):
                                cfg_prompt_tps = float(cfg_diag.get("prompt_eval_tps", {}).get("mean", 0.0) or 0.0)
                                cfg_decode_tps = float(cfg_diag.get("decode_eval_tps", {}).get("mean", 0.0) or 0.0)
                            else:
                                cfg_prompt_tps = 0.0
                                cfg_decode_tps = 0.0
                            summary = {
                                "label": run_args.label,
                                "ctx_size": ctx_size,
                                "batch_size": batch_size,
                                "ubatch_size": ubatch_size,
                                "kv": kv_type,
                                "spec_mode": spec_mode,
                                "extra_preset": extra_name,
                                "extra_args": extra_args,
                                "aggregate_tps": round(agg_tps, 4),
                                "mean_task_tps": round(statistics.mean(task_tps_values), 4) if task_tps_values else 0.0,
                                "prompt_eval_tps": round(cfg_prompt_tps, 2),
                                "decode_eval_tps": round(cfg_decode_tps, 2),
                                "mtp_draft_n": mtp_draft_n,
                                "errors": int(has_error),
                            }
                            summaries.append(summary)
                            print(
                                f"CONFIG RESULT: aggregate_tps={summary['aggregate_tps']:.2f} "
                                f"prompt_tps={cfg_prompt_tps:.2f} decode_tps={cfg_decode_tps:.2f}"
                            )
                            completed_keys.add(config_key)
                            save_autotune_session(completed=False)

                            if not has_error:
                                if best is None or summary["aggregate_tps"] > best["aggregate_tps"]:
                                    best = summary

                                if best is not None:
                                    best_extra_args = str(best.get("extra_args", "")).strip()
                                    best_extra_repr = best_extra_args if best_extra_args else "<none>"
                                    print(
                                        "CURRENT BEST: "
                                        f"ctx={best['ctx_size']} b={best['batch_size']} ub={best['ubatch_size']} "
                                        f"kv={best['kv']} spec={best['spec_mode']} extra={best.get('extra_preset', 'base')} "
                                        f"aggregate_tps={best['aggregate_tps']:.2f} "
                                        f"extra_args={best_extra_repr}"
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
                                        save_autotune_session(completed=False)
                                        print(
                                            "Autotune prune: stop larger batch values for "
                                            f"ctx={ctx_size}, ub={ubatch_size}, kv={kv_type}, spec={spec_mode}, extra={extra_name} "
                                            f"after {batch_drop_streak} drop(s)"
                                        )
                                    break

                        if (
                            ran_any_batch_for_ubatch
                            and args.autotune_smart_prune
                            and _is_drop_significant(prev_ubatch_best_tps, ubatch_best_tps, args.autotune_prune_drop_pct)
                        ):
                            ubatch_drop_streak += 1
                        else:
                            ubatch_drop_streak = 0

                        prev_ubatch_best_tps = max(prev_ubatch_best_tps, ubatch_best_tps)

                        if (
                            ran_any_batch_for_ubatch
                            and args.autotune_smart_prune
                            and ubatch_drop_streak >= max(1, args.autotune_prune_patience)
                        ):
                            remaining_ub = len(ubatch_values_sorted) - (ubatch_values_sorted.index(ubatch_size) + 1)
                            if remaining_ub > 0:
                                est_skip = remaining_ub * len(batch_values_sorted)
                                skipped_by_prune += est_skip
                                save_autotune_session(completed=False)
                                print(
                                    "Autotune prune: stop larger ubatch values for "
                                    f"ctx={ctx_size}, kv={kv_type}, spec={spec_mode}, extra={extra_name} "
                                    f"after {ubatch_drop_streak} drop(s)"
                                )
                            break

    if args.autotune_smart_prune:
        print(f"Autotune smart-prune summary: executed={executed}, skipped~={skipped_by_prune}, total-grid={total_configs}")

    summary_json = out_dir / f"{args.label}-autotune-summary.json"
    summary_csv = out_dir / f"{args.label}-autotune-summary.csv"
    summary_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "label",
            "ctx_size",
            "batch_size",
            "ubatch_size",
            "kv",
            "spec_mode",
            "extra_preset",
            "extra_args",
            "aggregate_tps",
            "mean_task_tps",
            "prompt_eval_tps",
            "decode_eval_tps",
            "mtp_draft_n",
            "errors",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")
    save_autotune_session(completed=True)
    model_path = str(Path(args.model) if args.model else default_model() or "")
    best_cfg = ""
    aggregate_tps = 0.0
    mean_tps = 0.0
    spec_mode = "mixed"
    if best:
        best_extra_args = str(best.get("extra_args", "")).strip()
        best_extra_repr = best_extra_args if best_extra_args else "<none>"
        best_cfg = (
            f"ctx={best['ctx_size']} b={best['batch_size']} ub={best['ubatch_size']} "
            f"kv={best['kv']} spec={best['spec_mode']} extra={best.get('extra_preset', 'base')} "
            f"extra_args={best_extra_repr}"
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
            "task_ids": args.task_ids,
            "runs": args.runs,
            "ctx": args.autotune_min_ctx,
            "batch": "sweep",
            "ubatch": "sweep",
            "kv_k": "sweep",
            "kv_v": "sweep",
            "spec_mode": spec_mode,
            "extra_preset": str(best.get("extra_preset", "mixed") if best else "mixed"),
            "extra_args": str(best.get("extra_args", "") if best else ""),
            "no_reuse": 1 if args.no_reuse else 0,
            "gpu_layers": args.gpu_layers,
            "parallel": args.parallel,
            "flash_attn": "on" if args.flash_attn else "off",
            "max_tokens": args.max_tokens,
            "real_context_mode": args.real_context_mode,
            "real_context_chars": args.real_context_chars,
            "real_context_safe_fill": args.real_context_safe_fill,
            "no_v2_prime_pass": 0 if args.v2_prime_pass else 1,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "aggregate_tps": f"{aggregate_tps:.4f}",
            "mean_task_tps": f"{mean_tps:.4f}",
            "prompt_eval_tps": f"{float(best.get('prompt_eval_tps', 0.0) or 0.0):.2f}" if best else "",
            "decode_eval_tps": f"{float(best.get('decode_eval_tps', 0.0) or 0.0):.2f}" if best else "",
            "errors": sum(int(r.get("errors", 0)) for r in summaries),
            "best_config": best_cfg,
            "jsonl_file": "",
            "csv_file": "",
            "summary_file": summary_csv.name,
            "server_log_file": f"{best['label']}.server.log" if best and best.get("label") else "",
        },
        out_dir,
        history_csv_name=history_csv_name,
        history_md_name=history_md_name,
    )

    if best:
        best_extra_args = str(best.get("extra_args", "")).strip()
        best_extra_repr = best_extra_args if best_extra_args else "<none>"
        print(
            "BEST: "
            f"ctx={best['ctx_size']} b={best['batch_size']} ub={best['ubatch_size']} "
            f"kv={best['kv']} spec={best['spec_mode']} extra={best.get('extra_preset', 'base')} "
            f"aggregate_tps={best['aggregate_tps']:.2f} "
            f"extra_args={best_extra_repr}"
        )
        if args.autotune_update_preset:
            model_path = Path(args.model) if args.model else default_model()
            if model_path:
                update_model_preset_file(Path(args.autotune_preset_file), model_path, best, args)
    else:
        print("No successful autotune runs.")

    return 0 if best else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Benchmark stop requested; graceful server cleanup completed.", flush=True)
        raise SystemExit(130)
