#!/usr/bin/env python3
"""bench2 - universal llama-server benchmark (single levels + agent sessions).

Backend-agnostic: works with any llama-server build (ROCm, Vulkan, CPU, RPC)
either started by the script (--server-bin) or attached (--attach URL).

Run:
    python scripts/bench2.py run --run-name q38-rocm-l2 --level 2
    python scripts/bench2.py run --run-name d094-vk-session --session-level 2
    python scripts/bench2.py run --run-name recheck --level 0,2 --runs 3
    python scripts/bench2.py find --name "l2"
    python scripts/bench2.py list --recent 10

Output layout per run (default build_logs/bench/<RUN_NAME>/):
    run.json            effective config + metadata
    <RUN_NAME>.jsonl    event log (server, per-request, summary)
    metrics.csv         one row per measurement (single level or session)
    summary.md          human-readable report
    server.log          raw server output
Global indexes: build_logs/bench/index.csv + index.md
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import re
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

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(os.environ.get("BENCH2_CONFIG_DIR", ROOT / "configs" / "bench"))
DEFAULT_RESULTS = Path(os.environ.get("BENCH2_RESULTS_DIR", ROOT / "build_logs" / "bench"))

LOAD_GGUF_FALLBACK = [ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"]

METRICS_COLUMNS = [
    "run_name", "type", "level", "run_idx", "timestamp", "backend", "profile",
    "model", "commit", "ctx", "prompt_tokens", "decoded_tokens", "prefill_tps",
    "decode_tps", "ttft_ms", "total_ms", "aggregate_tps", "mtp_draft_n",
    "mtp_accepted", "eff_decode_tps", "session_turns", "decode_slope",
    "status", "path",
]


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_json(path: Path, required: bool = True) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        if required:
            raise SystemExit(f"bench2: missing config file: {path}")
        return {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"bench2: invalid JSON in {path}: {exc}")


class Config:
    """Merged view of configs + CLI overrides (CLI wins, then profile, then defaults)."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.levels = load_json(CONFIG_DIR / "levels.json")
        self.sessions = load_json(CONFIG_DIR / "sessions.json")
        self.hw = load_json(CONFIG_DIR / "hardware.profiles.json")
        self.defaults = load_json(CONFIG_DIR / "server.defaults.json")
        self.args = args
        prof_name = args.profile or self.hw.get("default_profile", "rdna-lab")
        self.profile = self.hw.get("profiles", {}).get(
            prof_name, self.hw.get("profiles", {}).get("generic", {})
        )
        self.profile_name = prof_name
        self.backend = self._resolve_backend()
        bd = self.profile.get("backend_defaults", {}).get(self.backend, {})
        self.bd = bd

    def _resolve_backend(self) -> str:
        if self.args.backend not in ("auto", None):
            return self.args.backend
        if self.args.server_bin:
            low = str(self.args.server_bin).lower()
            if "rocm" in low:
                return "rocm"
            if "vulkan" in low:
                return "vk"
        return "rocm"

    # server-side settings -------------------------------------------------- #
    def server(self) -> dict[str, Any]:
        s = dict(self.defaults.get("server", {}))
        a = self.args
        bd = self.bd
        s["n_gpu_layers"] = a.gpu_layers if a.gpu_layers is not None else s.get("n_gpu_layers", 999)
        s["parallel"] = a.parallel if a.parallel is not None else s.get("parallel", 1)
        s["flash_attn"] = a.flash_attn if a.flash_attn is not None else s.get("flash_attn", True)
        s["kv_k"] = a.kv_k or s.get("kv_k", "q8_0")
        s["kv_v"] = a.kv_v or s.get("kv_v", "q8_0")
        s["seed"] = a.seed if a.seed is not None else s.get("seed", 42)
        s["temperature"] = a.temperature if a.temperature is not None else s.get("temperature", 0.2)
        s["top_p"] = a.top_p if a.top_p is not None else s.get("top_p", 0.9)
        s["fit"] = a.fit if a.fit is not None else s.get("fit", "off")
        s["spec"] = a.spec or s.get("spec", "none")
        s["dev"] = a.dev if a.dev is not None else bd.get("dev", "")
        s["sm"] = a.sm if a.sm is not None else bd.get("sm", "layer")
        s["ts"] = a.ts if a.ts is not None else bd.get("ts", "")
        s["batch"] = a.batch_size if a.batch_size is not None else self.profile.get("default_batch", 8192)
        s["ubatch"] = a.ubatch_size if a.ubatch_size is not None else self.profile.get("default_ubatch", 1024)
        s["no_warmup"] = a.no_warmup if a.no_warmup is not None else s.get("no_warmup", True)
        s["warmup_shot"] = a.warmup_shot if a.warmup_shot is not None else s.get("warmup_shot", True)
        s["warmup_tokens"] = a.warmup_tokens if a.warmup_tokens is not None else s.get("warmup_tokens", 512)
        s["warmup_decode"] = a.warmup_decode if a.warmup_decode is not None else s.get("warmup_decode", 16)
        return s

    def git_commit(self) -> str:
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                capture_output=True, text=True, timeout=10,
            )
            return out.stdout.strip() or ""
        except Exception:
            return ""

    def results_dir(self) -> Path:
        return Path(self.args.results_dir) if self.args.results_dir else DEFAULT_RESULTS


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def parse_levels(spec: str | None) -> list[int]:
    if not spec:
        return []
    result: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)(?:-(\d+))?$", part)
        if not m:
            raise SystemExit(f"bench2: bad --level part: {part!r}")
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        if lo > hi:
            lo, hi = hi, lo
        for val in range(lo, hi + 1):
            if val not in result:
                result.append(val)
    return sorted(result)


def parse_text_csv(values: str | None) -> list[str]:
    return [v.strip() for v in (values or "").split(",") if v.strip()]


def syn_token_estimate(target_tokens: int, chars_per_token: float) -> str:
    """Deterministic pseudo-code-like text of approximately target_tokens."""
    rng = random.Random(20260828)
    words = [
        "context", "buffer", "schedule", "attention", "kernel", "weight",
        "tensor", "layer", "revision", "pipeline", "dispatch", "memory",
        "cache", "quant", "normalize", "segment", "token", "logits",
        "gradient", "address", "stack", "flush", "prefetch", "payload",
        "selector", "factor", "channel", "stride", "packet", "window",
    ]
    lines: list[str] = []
    total_chars = 0
    target_chars = int(target_tokens * chars_per_token)
    idx = 0
    while total_chars < target_chars:
        words_sel = " ".join(rng.choice(words) for _ in range(24))
        line = (
            f"section {idx:06d} | {words_sel} | "
            f"offset={idx * 128} mask=0x{idx * 7919 & 0xFFFF:04x} "
            f"group={(idx % 7) + 1} ref=entry-{idx * 13:06d}\n"
        )
        lines.append(line)
        total_chars += len(line)
        idx += 1
    return "".join(lines)


def repo_snapshot_context(max_tokens: int, chars_per_token: float) -> str:
    """Text snapshot of the repository (markdown/py/c/cpp), deterministic order."""
    from collections import deque

    exts = {".md", ".py", ".txt", ".cpp", ".c", ".h", ".hpp", ".cmake", ".json", ".yaml", ".yml"}
    skip_dirs = {"build", "build-rocm", "build-vulkan", "build-cpu", "models", "vendor",
                 "third_party", ".git", "node_modules", "build_logs", "kanon-cleanup"}
    files: list[Path] = []
    q: deque[Path] = deque([ROOT])
    while q and len(files) < 4000:
        d = q.popleft()
        try:
            for entry in sorted(d.iterdir(), key=lambda p: p.name.lower()):
                if entry.is_dir():
                    if entry.name.lower() not in skip_dirs and not entry.name.startswith("."):
                        q.append(entry)
                elif entry.suffix.lower() in exts and entry.stat().st_size < 2_000_000:
                    files.append(entry)
        except OSError:
            continue
    out: list[str] = []
    total = 0
    cap = int(max_tokens * chars_per_token * 0.88)
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = re.sub(r"\n{3,}", "\n\n", text[:2_000_000])
        if total + len(text) > cap:
            text = text[: cap - total]
        if not text:
            continue
        out.append(f"### FILE {path.as_posix()}\n\n{text}\n")
        total += len(text)
        if total >= cap:
            break
    return "".join(out)


def build_prompt_text(cfg: Config, level: dict[str, Any], max_tokens: int) -> str:
    source = cfg.args.context_source or cfg.levels.get("default_context_source", "synthetic")
    src_cfg = cfg.levels.get("context_sources", {}).get(source, {})
    cpt = float(src_cfg.get("chars_per_token", 3.9))
    if source == "synthetic":
        body = syn_token_estimate(max_tokens, cpt)
    elif source == "repo-snapshot":
        body = repo_snapshot_context(max_tokens, cpt)
    elif source == "file":
        if not cfg.args.context_file:
            raise SystemExit("bench2: --context-source file requires --context-file")
        body = Path(cfg.args.context_file).read_text(encoding="utf-8", errors="replace")
    else:
        raise SystemExit(f"bench2: unknown context source {source}")
    return (
        "## TASK\n"
        "Summarize the provided repository context and produce a concise "
        "implementation plan.\n\n"
        "## CONTEXT\n" + body +
        "\n## RESPONSE\n"
    )


def session_turn_text(turn: int, input_tokens: int) -> str:
    """Deterministic agent turn request (grows the context by ~input_tokens)."""
    header = (
        f"[turn {turn:02d}] agent request: examine the following result slice, "
        f"verify hypothesis and draft a one-paragraph summary.\n"
    )
    return header + syn_token_estimate(input_tokens, 3.9)


# --------------------------------------------------------------------------- #
# server lifecycle
# --------------------------------------------------------------------------- #
def build_server_cmd(cfg: Config, host: str, port: int, ctx: int, model: Path,
                     spec_n: int | None = None) -> list[str]:
    s = cfg.server()
    server_bin = resolve_server_bin(cfg)
    if server_bin is None:
        raise SystemExit("bench2: server binary not resolved")
    if cfg.args.server_extra:
        extra = list(filter(None, re.split(r"\s+", cfg.args.server_extra.strip())))
    else:
        extra = []
    cmd = [
        str(server_bin),
        "-m", str(model),
        "--host", host,
        "--port", str(port),
        "--flash-attn", "on" if s["flash_attn"] else "off",
        "-np", str(s["parallel"]),
        "-c", str(ctx),
        "-b", str(s["batch"]),
        "-ub", str(s["ubatch"]),
        "--cache-type-k", s["kv_k"],
        "--cache-type-v", s["kv_v"],
        "-ngl", str(s["n_gpu_layers"]),
        "--seed", str(s["seed"]),
    ]
    if s["no_warmup"]:
        cmd.append("--no-warmup")
    if s["spec"] and s["spec"] != "none":
        cmd.extend(["--spec-type", "draft-mtp"])
        n = spec_n or 2
        cmd.extend(["--spec-draft-n-max", str(n)])
    else:
        cmd.extend(["--spec-type", "none"])
    if s["dev"]:
        cmd.extend(["-dev", s["dev"]])
    if s["sm"]:
        cmd.extend(["-sm", s["sm"]])
    if s["ts"]:
        cmd.extend(["-ts", s["ts"]])
    if s["fit"]:
        cmd.extend(["-fit", s["fit"]])
    if s.get("cache_ram"):
        cmd.extend(["--cache-ram", str(s["cache_ram"])])
    if s.get("ctx_checkpoints"):
        cmd.extend(["--ctx-checkpoints", str(s["ctx_checkpoints"])])
    cmd.extend(extra)
    return cmd


def runtime_path_prepend(server_bin: str | Path) -> list[str]:
    """Extra bin dirs needed by the chosen build (ROCm runtime / MinGW DLLs)."""
    low = str(server_bin).lower()
    out: list[str] = []
    if "rocm" in low or "hip" in low:
        out.append(r"C:\Program Files\AMD\ROCm\7.1\bin")
    if "vulkan" in low:
        out.append(r"C:\Strawberry\c\bin")
    return out


def start_server(cmd: list[str], log_path: Path, out_stream: Any) -> subprocess.Popen[str]:
    log_file = open(log_path, "w", encoding="utf-8", errors="replace")
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    env = os.environ.copy()
    extra = runtime_path_prepend(cmd[0] if cmd else "")
    if extra:
        old_path = env.get("PATH", "")
        parts = [p for p in old_path.split(os.pathsep) if p]
        parts = [p for p in parts if p not in extra]
        env["PATH"] = os.pathsep.join(extra + parts)
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        env=env,
    )
    return proc


def wait_health(host: str, port: int, timeout: float = 180.0, poll: float = 1.0,
                proc: subprocess.Popen[str] | None = None) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(poll)
    return False


def stop_server(proc: subprocess.Popen[str], timeout: float = 180.0) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        except (OSError, ValueError):
            try:
                proc.terminate()
            except OSError:
                pass
    else:
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=60)


# --------------------------------------------------------------------------- #
# HTTP completion
# --------------------------------------------------------------------------- #
def post_completion(host: str, port: int, prompt: str, n_predict: int,
                    temperature: float, top_p: float, seed: int,
                    cache_prompt: bool, timeout: float,
                    payload_extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Chat-completions request (needed for thinking models: raw /completion
    without chat template makes Qwen3.x emit EOG as the first token)."""
    payload = {
        "model": "local-model",
        "messages": [
            {
                "role": "system",
                "content": "You are a concise coding agent. Answer directly and avoid long preambles.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": n_predict,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "cache_prompt": cache_prompt,
        "stream": False,
    }
    if payload_extra:
        payload.update(payload_extra)
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data


# --------------------------------------------------------------------------- #
# metric extraction
# --------------------------------------------------------------------------- #
def extract_timings(resp: dict[str, Any]) -> dict[str, Any]:
    t = resp.get("timings") or {}
    prompt_n = int(t.get("prompt_n") or 0)
    predicted_n = int(t.get("predicted_n") or 0)
    prompt_ms = float(t.get("prompt_ms") or 0.0)
    predicted_ms = float(t.get("predicted_ms") or 0.0)
    total_ms = float(t.get("total_ms") or 0.0)
    if total_ms <= 0 and (prompt_ms > 0 or predicted_ms > 0):
        total_ms = prompt_ms + predicted_ms
    cache_n = int(t.get("cache_n") or 0)
    # server timing prompt_n counts the whole prompt even when KV-reuse
    # restored the prefix; use the newly processed tokens for prefill TPS.
    new_n = max(0, prompt_n - cache_n)
    prefill_tps = (new_n / (prompt_ms / 1000.0)) if prompt_ms > 0 else 0.0
    decode_tps = (predicted_n / (predicted_ms / 1000.0)) if predicted_ms > 0 else 0.0
    aggregate_tps = (predicted_n / (total_ms / 1000.0)) if total_ms > 0 else 0.0
    return {
        "prompt_n": prompt_n,
        "cache_n": cache_n,
        "prompt_ms": prompt_ms,
        "prefill_tps": round(prefill_tps, 4),
        "predicted_n": predicted_n,
        "predicted_ms": predicted_ms,
        "decode_tps": round(decode_tps, 4),
        "total_ms": round(total_ms, 4),
        "aggregate_tps": round(aggregate_tps, 4),
        "ttft_ms": round(prompt_ms, 4),
    }


def parse_mtp_counts(server_log_text: str) -> tuple[int, int]:
    """Return (draft_n, accepted_n); best-effort from server log / repeated calls."""
    draft_n = 0
    accepted_n = 0
    for m in re.finditer(r"draft_n\s*=\s*(\d+)", server_log_text):
        draft_n += int(m.group(1))
    for m in re.finditer(r"accepted\s*=\s*(\d+)", server_log_text):
        accepted_n += int(m.group(1))
    for m in re.finditer(r"draft accepted\s*[:=]\s*(\d+)", server_log_text):
        accepted_n += int(m.group(1))
    return draft_n, accepted_n


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class RunWriter:
    def __init__(self, run_dir: Path, run_name: str) -> None:
        ensure_dir(run_dir)
        self.run_dir = run_dir
        self.run_name = run_name
        self.events: list[dict[str, Any]] = []
        self.rows: list[dict[str, Any]] = []

    def event(self, kind: str, **kw: Any) -> None:
        ev = {"ts": now_iso(), "event": kind}
        ev.update(kw)
        self.events.append(ev)
        print(f"[bench2] {kind}: {kw}", file=sys.stderr)

    def add_row(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def write_jsonl(self) -> None:
        with open(self.run_dir / f"{self.run_name}.jsonl", "w", encoding="utf-8") as fh:
            for ev in self.events:
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def write_run_json(self, meta: dict[str, Any]) -> None:
        with open(self.run_dir / "run.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

    def write_csv(self) -> None:
        rows = self.rows
        with open(self.run_dir / "metrics.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=METRICS_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in METRICS_COLUMNS})

    def write_summary(self, title: str, sections: list[tuple[str, str]]) -> None:
        lines = [f"# {title}", "", f"Run: `{self.run_name}` — {now_iso()}", ""]
        for heading, body in sections:
            lines += [f"## {heading}", "", body, ""]
        with open(self.run_dir / "summary.md", "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))


def update_index(results_dir: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(results_dir)
    path = results_dir / "index.csv"
    existing: list[dict[str, Any]] = []
    if path.exists():
        with open(path, "r", encoding="utf-8", newline="") as fh:
            existing = list(csv.DictReader(fh))
    existing = [r for r in existing if r.get("run_name") != rows[0]["run_name"]]
    existing.extend(rows)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=METRICS_COLUMNS)
        w.writeheader()
        for r in existing:
            w.writerow({k: r.get(k, "") for k in METRICS_COLUMNS})
    # index.md
    md_lines = ["# Bench2 index", "", "| run_name | type | level | timestamp | backend | model | prefill_tps | decode_tps | total_ms | status |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in existing:
        md_lines.append(
            f"| {r.get('run_name','')} | {r.get('type','')} | {r.get('level','')} | "
            f"{r.get('timestamp','')} | {r.get('backend','')} | {r.get('model','')} | "
            f"{r.get('prefill_tps','')} | {r.get('decode_tps','')} | "
            f"{r.get('total_ms','')} | {r.get('status','')} |"
        )
    with open(results_dir / "index.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")


def table_md(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("-" * len(h) for h in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
def resolve_server_bin(cfg: Config) -> Path | None:
    """Auto-select llama-server by backend if --server-bin was not given."""
    if cfg.args.server_bin:
        p = Path(cfg.args.server_bin)
        return p if p.exists() else None
    bd = cfg.backend
    candidates: list[Path] = []
    if bd == "rocm":
        candidates = [ROOT / "build-rocm" / "bin" / "llama-server.exe"]
    elif bd == "vk":
        candidates = [ROOT / "build-vulkan" / "bin" / "llama-server.exe"]
    elif bd == "cpu":
        candidates = [ROOT / "build-cpu" / "bin" / "llama-server.exe"]
    else:
        candidates = [
            ROOT / "build-rocm" / "bin" / "llama-server.exe",
            ROOT / "build-vulkan" / "bin" / "llama-server.exe",
            ROOT / "build-cpu" / "bin" / "llama-server.exe",
        ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def preflight(cfg: Config) -> tuple[bool, str]:
    problems: list[str] = []
    if cfg.args.attach:
        return True, ""
    if not resolve_server_bin(cfg):
        problems.append(
            "no matching llama-server build found for backend "
            f"{cfg.backend!r} (expects build-rocm|build-vulkan|build-cpu/bin/llama-server.exe)"
        )
    if os.name == "nt":
        try:
            out = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=30).stdout
            live = [ln for ln in out.splitlines() if "llama-server" in ln]
            if live:
                problems.append(f"active llama-server process found: {live[0].strip()}")
        except Exception:
            pass
    if problems:
        return False, "; ".join(problems)
    return True, ""


# --------------------------------------------------------------------------- #
# scenarios
# --------------------------------------------------------------------------- #
def resolve_model(cfg: Config) -> Path:
    if cfg.args.model:
        p = Path(cfg.args.model)
        if not p.exists():
            raise SystemExit(f"bench2: model not found: {p}")
        return p
    for cand in LOAD_GGUF_FALLBACK:
        if cand.exists():
            return cand
    # fallback: first *.gguf in models/
    models_dir = ROOT / "models"
    if models_dir.exists():
        ggu = sorted(models_dir.glob("*.gguf"))
        if ggu:
            return ggu[0]
    raise SystemExit("bench2: --model is required (no default GGUF found)")


# --------------------------------------------------------------------------- #
# live progress (reads server.log while the HTTP request is blocking)
# --------------------------------------------------------------------------- #
PROGRESS_RE = re.compile(
    r"prompt processing progress,\s*n_tokens =\s*(\d+),\s*"
    r"batch\.n_tokens =\s*(\d+),\s*progress =\s*([0-9.]+)"
)
DONE_RE = re.compile(
    r"prompt processing done,\s*n_tokens =\s*(\d+),\s*batch\.n_tokens =\s*(\d+)"
)


def watch_progress(server_log: Path | None, label: str) -> threading.Thread | None:
    """Background thread: print prefill progress from server.log to stdout.

    Starts at the current log size so it only reports the new request
    (not earlier warmup/server messages)."""
    if server_log is None or not server_log.exists():
        return None
    stop = threading.Event()
    start_offset = server_log.stat().st_size if server_log.exists() else 0

    def _worker() -> None:
        offset = start_offset
        buffer = ""
        last_pct = -1.0
        total_hint: int | None = None
        while not stop.is_set():
            try:
                with server_log.open("r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(offset)
                    data = fh.read()
                    offset = fh.tell()
            except OSError:
                time.sleep(0.25)
                continue
            if data:
                buffer += data
                lines = buffer.splitlines(keepends=True)
                if lines and not lines[-1].endswith(("\n", "\r")):
                    buffer = lines.pop()
                else:
                    buffer = ""
                for raw in lines:
                    line = raw.strip()
                    m = PROGRESS_RE.search(line)
                    if m:
                        processed = int(m.group(1))
                        progress = float(m.group(3))
                        pct = max(0.0, min(100.0, progress * 100.0))
                        if total_hint is None and progress > 0:
                            total_hint = max(processed, int(round(processed / progress)))
                        if pct - last_pct >= 1.0:
                            total = str(total_hint) if total_hint else "?"
                            print(f"[bench2] {label}: prefill {pct:.0f}% ({processed}/{total} tok)",
                                  flush=True)
                            last_pct = pct
                    elif DONE_RE.search(line):
                        m2 = DONE_RE.search(line)
                        print(f"[bench2] {label}: prefill done ({m2.group(1)} tok)", flush=True)
                        return
            time.sleep(0.25)

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    return th



def run_warmup(cfg: Config, writer: RunWriter, host: str, port: int,
               opts: dict[str, Any]) -> dict[str, Any]:
    """One short request right after server start: compiles ROCm/Vulkan
    kernels and initializes buffers so the measured run is a warm,
    real-usage prefill (cold-start fixed overhead otherwise inflates
    small L0/L1 times). Results are recorded as an event, not a metric."""
    tokens = int(opts.get("warmup_tokens", 512))
    n_predict = int(opts.get("warmup_decode", 16))
    prompt = ("Warmup probe. Reply with exactly: OK\n\n"
              + syn_token_estimate(tokens, 3.9))
    s = cfg.server()
    print(f"[bench2] warmup: request ~{tokens} tok (decode {n_predict})...", flush=True)
    t0 = time.monotonic()
    try:
        resp = post_completion(
            host, port, prompt, n_predict, s["temperature"], s["top_p"],
            s["seed"], cache_prompt=False, timeout=600,
            payload_extra=getattr(cfg.args, "api_extra", None),
        )
        wall = time.monotonic() - t0
    except Exception as exc:
        print(f"[bench2] warmup: ERROR {exc}", file=sys.stderr, flush=True)
        info = {"enabled": True, "ok": False, "error": str(exc)}
        writer.event("warmup_done", **info)
        return info
    tim = extract_timings(resp)
    info = {
        "enabled": True,
        "ok": True,
        "wall_s": round(wall, 3),
        "prompt_n": tim["prompt_n"],
        "predicted_n": tim["predicted_n"],
        "prompt_ms": tim["prompt_ms"],
        "prefill_tps": tim["prefill_tps"],
        "decode_tps": tim["decode_tps"],
    }
    writer.event("warmup_done", **info)
    print(f"[bench2] warmup: DONE prefill={tim['prefill_tps']} tok/s "
          f"({tim['prompt_n']} tok, {tim['prompt_ms']} ms) decode={tim['decode_tps']} tok/s",
          flush=True)
    return info


def run_single_level(cfg: Config, writer: RunWriter, backend: str, host: str,
                     port: int, level_idx: int, shot: int,
                     model_name: str) -> dict[str, Any]:
    level = cfg.levels["levels"][str(level_idx)]
    ctx = int(level["ctx"])
    prompt_tokens = int(level["prompt_tokens"])
    decode_tokens = int(level["decode_tokens"])
    s = cfg.server()
    prompt = build_prompt_text(cfg, level, prompt_tokens)
    writer.event(
        "level_start", level=level_idx, shot=shot, ctx=ctx,
        prompt_target=prompt_tokens, decode_target=decode_tokens,
        prompt_chars=len(prompt),
    )
    print(f"[bench2] level L{level_idx} shot {shot}: start "
          f"(ctx={ctx}, prompt~{prompt_tokens}, decode={decode_tokens})", flush=True)
    t0 = time.monotonic()
    prog = watch_progress(writer.run_dir / "server.log", f"L{level_idx}/{shot}")
    try:
        resp = post_completion(
            host, port, prompt, decode_tokens, s["temperature"], s["top_p"],
            s["seed"] + shot, cache_prompt=False,
            timeout=3600 if ctx >= 98304 else 900,
            payload_extra=getattr(cfg.args, "api_extra", None),
        )
        wall = time.monotonic() - t0
    except Exception as exc:
        writer.event("level_error", level=level_idx, shot=shot, error=str(exc))
        print(f"[bench2] level L{level_idx} shot {shot}: ERROR {exc}", flush=True)
        return {"status": "error", "error": str(exc), "level": level_idx, "shot": shot}
    finally:
        if prog is not None:
            prog.join(timeout=2)
    tim = extract_timings(resp)
    writer.event(
        "level_done", level=level_idx, shot=shot, wall_s=round(wall, 3), **tim,
    )
    print(f"[bench2] level L{level_idx} shot {shot}: DONE prefill={tim['prefill_tps']} tok/s "
          f"decode={tim['decode_tps']} tok/s ttft={tim['ttft_ms']} ms total={tim['total_ms']} ms",
          flush=True)
    row = {
        "run_name": writer.run_name,
        "type": "single",
        "level": level_idx,
        "run_idx": shot,
        "timestamp": now_iso(),
        "backend": backend,
        "profile": cfg.profile_name,
        "model": model_name,
        "commit": cfg.git_commit(),
        "ctx": ctx,
        "prompt_tokens": tim["prompt_n"],
        "decoded_tokens": tim["predicted_n"],
        "prefill_tps": tim["prefill_tps"],
        "decode_tps": tim["decode_tps"],
        "ttft_ms": tim["ttft_ms"],
        "total_ms": tim["total_ms"],
        "aggregate_tps": tim["aggregate_tps"],
        "status": "ok",
        "path": str(writer.run_dir),
    }
    writer.add_row(row)
    return row


def run_session(cfg: Config, writer: RunWriter, backend: str, host: str,
                port: int, session_level: int, shot: int,
                model_name: str) -> dict[str, Any]:
    sess = cfg.sessions["sessions"][str(session_level)]
    ctx = int(sess["ctx"])
    turns = int(sess["turns"])
    input_tokens = int(sess["input_tokens"])
    decode_tokens = int(sess["decode_tokens"])
    s = cfg.server()
    writer.event(
        "session_start", session_level=session_level, shot=shot, ctx=ctx,
        turns=turns, input_tokens=input_tokens, decode_tokens=decode_tokens,
    )
    print(f"[bench2] session SL{session_level} shot {shot}: start "
          f"(ctx={ctx}, turns={turns}, input~{input_tokens}, decode={decode_tokens})", flush=True)
    turn_rows: list[dict[str, Any]] = []
    prompt = ""
    wall_total = 0.0
    est_ctx = 0
    for turn in range(1, turns + 1):
        turn_text = session_turn_text(turn, input_tokens)
        prompt = turn_text if turn == 1 else prompt + "\n\n" + turn_text
        est_ctx += input_tokens + decode_tokens
        t0 = time.monotonic()
        prog = watch_progress(writer.run_dir / "server.log", f"SL{session_level}/{shot} turn{turn}")
        try:
            resp = post_completion(
                host, port, prompt, decode_tokens, s["temperature"], s["top_p"],
                s["seed"] + shot + turn, cache_prompt=True,
                timeout=1800 if ctx >= 98304 else 600,
                payload_extra=getattr(cfg.args, "api_extra", None),
            )
        except Exception as exc:
            writer.event("turn_error", turn=turn, error=str(exc))
            print(f"[bench2] session SL{session_level} shot {shot} turn {turn}: ERROR {exc}", flush=True)
            return {"status": "error", "error": str(exc)}
        finally:
            if prog is not None:
                prog.join(timeout=2)
        wall = time.monotonic() - t0
        wall_total += wall
        tim = extract_timings(resp)
        turn_rows.append({
            "turn": turn,
            "wall_s": round(wall, 3),
            "ctx_est_tokens": est_ctx,
            **tim,
        })
        writer.event("turn_done", turn=turn, shot=shot, **tim)
        print(f"[bench2] session SL{session_level} shot {shot} turn {turn}: DONE "
              f"prefill={tim['prefill_tps']} tok/s decode={tim['decode_tps']} tok/s "
              f"ttft={tim['ttft_ms']} ms wall={wall:.1f} s", flush=True)

    decode_tps_list = [float(t["decode_tps"]) for t in turn_rows if t["decode_tps"] > 0]
    ttft_list = [float(t["ttft_ms"]) for t in turn_rows if t["ttft_ms"] > 0]
    mean_decode = statistics.fmean(decode_tps_list) if decode_tps_list else 0.0
    mean_ttft = statistics.fmean(ttft_list) if ttft_list else 0.0
    slope = 0.0
    if len(decode_tps_list) >= 3:
        xs = list(range(len(decode_tps_list)))
        mean_x = statistics.fmean(xs)
        mean_y = statistics.fmean(decode_tps_list)
        denom = sum((x - mean_x) ** 2 for x in xs)
        slope = (sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, decode_tps_list)) / denom) if denom else 0.0
    total_tokens = sum(int(t["predicted_n"]) for t in turn_rows)
    session_tps = (total_tokens / wall_total) if wall_total > 0 else 0.0
    writer.event(
        "session_done", session_level=session_level, shot=shot,
        mean_decode_tps=round(mean_decode, 4), mean_ttft_ms=round(mean_ttft, 2),
        decode_slope=round(slope, 5), session_agg_tps=round(session_tps, 4),
        wall_s=round(wall_total, 3),
    )
    # turns csv
    with open(writer.run_dir / "session_turns.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["turn", "ctx_est_tokens", "prompt_n", "cache_n", "prompt_ms", "prefill_tps",
                    "predicted_n", "predicted_ms", "decode_tps", "ttft_ms", "wall_s"])
        for t in turn_rows:
            w.writerow([t["turn"], t["ctx_est_tokens"], t["prompt_n"], t.get("cache_n", 0),
                        t["prompt_ms"], t["prefill_tps"], t["predicted_n"], t["predicted_ms"],
                        t["decode_tps"], t["ttft_ms"], t["wall_s"]])
    row = {
        "run_name": writer.run_name,
        "type": "session",
        "level": session_level,
        "run_idx": shot,
        "timestamp": now_iso(),
        "backend": backend,
        "profile": cfg.profile_name,
        "model": model_name,
        "commit": cfg.git_commit(),
        "ctx": ctx,
        "prompt_tokens": sum(int(t["prompt_n"]) for t in turn_rows),
        "decoded_tokens": total_tokens,
        "prefill_tps": statistics.fmean([float(t["prefill_tps"]) for t in turn_rows if t["prefill_tps"] > 0]) if turn_rows else 0.0,
        "decode_tps": round(mean_decode, 4),
        "ttft_ms": round(mean_ttft, 2),
        "total_ms": round(wall_total * 1000.0, 2),
        "aggregate_tps": round(session_tps, 4),
        "session_turns": turns,
        "decode_slope": round(slope, 5),
        "status": "ok",
        "path": str(writer.run_dir),
    }
    writer.add_row(row)
    return row


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace) -> int:
    cfg = Config(args)
    if args.api_extra:
        try:
            extra = json.loads(args.api_extra)
            if not isinstance(extra, dict):
                raise ValueError("expected a JSON object")
        except Exception as exc:
            print(f"bench2: --api-extra invalid: {exc}", file=sys.stderr)
            return 2
        args.api_extra = extra
    else:
        args.api_extra = None
    ok, reason = preflight(cfg)
    if not ok:
        print(f"bench2: preflight failed: {reason}", file=sys.stderr)
        return 2
    backend = cfg.backend
    model = resolve_model(cfg)
    opts = cfg.server()

    levels = parse_levels(args.level)
    session_levels = parse_levels(args.session_level) if args.session_level else []
    if not levels and not session_levels:
        levels = [1]
        print("bench2: no level given, defaulting to --level 1 (use --level 0..5 / --session-level 1..3)",
              file=sys.stderr)
    if not args.run_name:
        args.run_name = auto_run_name(backend, levels, session_levels)
        print(f"bench2: --run-name omitted, generated: {args.run_name}", flush=True)

    run_name = args.run_name
    results_dir = cfg.results_dir()
    writer = RunWriter(results_dir / run_name, run_name)

    host = args.host
    port = args.port or 0
    attached = bool(args.attach)

    proc: subprocess.Popen[str] | None = None
    server_cmd: list[str] | None = None
    ctx_targets: list[int] = []
    ctx_targets += [int(cfg.levels["levels"][str(i)]["ctx"]) for i in levels]
    ctx_targets += [int(cfg.sessions["sessions"][str(i)]["ctx"]) for i in session_levels]
    max_ctx = max(ctx_targets) if ctx_targets else 4096
    if attached:
        if not port:
            m = re.match(r"^(https?://[^/:]+)(?::(\d+))?/?(.*)$", args.attach)
            if m:
                host = m.group(1).split("://")[-1] if m.group(1) else host
                port = int(m.group(2)) if m.group(2) else port
        ctx_size = max_ctx
        print(f"[bench2] attach: {host}:{port}", flush=True)
    else:
        if port == 0:
            port = find_free_port()
        ctx_size = max_ctx
        server_cmd = build_server_cmd(cfg, host, port, ctx_size, model, spec_n=args.spec_n)
        writer.event("server_start", cmd=" ".join(server_cmd), host=host, port=port)
        proc = start_server(server_cmd, writer.run_dir / "server.log", sys.stderr)
        writer.event("server_spawned", pid=proc.pid)
        if not wait_health(host, port, timeout=args.health_timeout, proc=proc):
            tail = ""
            log_path = writer.run_dir / "server.log"
            if log_path.exists():
                tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
            returncode = proc.poll()
            if returncode is not None:
                print(f"[bench2] server exited during startup (code {returncode}); log tail:\n{tail}",
                      file=sys.stderr, flush=True)
                writer.event("server_exited", returncode=returncode, tail=tail)
            else:
                print(f"[bench2] server health timeout after {args.health_timeout}s; log tail:\n{tail}",
                      file=sys.stderr, flush=True)
                writer.event("server_health_timeout", tail=tail)
                stop_server(proc)
            return 3
        print(f"[bench2] server ready: {server_cmd[0]} (pid {proc.pid}, port {port})", flush=True)
    print(f"[bench2] run {run_name} | backend={backend} | model={model.name} | "
          f"levels={levels or '-'} | sessions={session_levels or '-'} | runs={args.runs}", flush=True)

    warmup_info: dict[str, Any] | None = None
    if opts.get("warmup_shot", True):
        warmup_info = run_warmup(cfg, writer, host, port, opts)
    elif opts.get("warmup_tokens", 0) > 0:
        # warmup disabled but tokens requested via config: nothing to do
        warmup_info = {"enabled": False}

    try:
        writer.run_json_meta = {
            "run_name": run_name,
            "timestamp": now_iso(),
            "type": "mixed" if levels and session_levels else ("single" if levels else "session"),
            "backend": backend,
            "profile": cfg.profile_name,
            "model": str(model),
            "commit": cfg.git_commit(),
            "server": opts,
            "warmup": warmup_info or {"enabled": opts.get("warmup_shot", True)},
            "levels": [levels, session_levels],
            "attach": args.attach or None,
            "results_dir": str(writer.run_dir),
            "bench2_version": "0.1.0",
        }
        writer.write_run_json(writer.run_json_meta)
        model_name = model.name

        for level_idx in levels:
            for shot in range(1, args.runs + 1):
                row_out = run_single_level(cfg, writer, backend, host, port, level_idx, shot, model_name)
                if row_out.get("status") == "error" and args.fail_fast:
                    raise RuntimeError(f"level {level_idx} failed: {row_out.get('error')}")
        for sl in session_levels:
            for shot in range(1, args.runs + 1):
                row_out = run_session(cfg, writer, backend, host, port, sl, shot, model_name)
                if row_out.get("status") == "error" and args.fail_fast:
                    raise RuntimeError(f"session {sl} failed: {row_out.get('error')}")
    finally:
        writer.write_jsonl()
        writer.write_csv()
        # summary
        sections: list[tuple[str, str]] = []
        single_rows = [r for r in writer.rows if r["type"] == "single"]
        session_rows = [r for r in writer.rows if r["type"] == "session"]
        if single_rows:
            h = ["level", "run", "prompt_tps", "decode_tps", "ttft_ms", "total_ms", "agg_tps"]
            rows = [[r["level"], r["run_idx"], r["prefill_tps"], r["decode_tps"],
                     r["ttft_ms"], r["total_ms"], r["aggregate_tps"]] for r in single_rows]
            sections.append(("Single levels", table_md(h, rows)))
        if session_rows:
            h = ["slevel", "run", "decode_tps", "ttft_ms", "total_ms", "agg_tps", "slope"]
            rows = [[r["level"], r["run_idx"], r["decode_tps"], r["ttft_ms"],
                     r["total_ms"], r["aggregate_tps"], r["decode_slope"]] for r in session_rows]
            sections.append(("Sessions", table_md(h, rows)))
        writer.write_summary(f"bench2 results: {run_name}", sections)
        update_index(results_dir, writer.rows)
        if proc is not None:
            stop_server(proc)
            writer.event("server_stopped")

    print(f"bench2: DONE {run_name} -> {writer.run_dir}", file=sys.stderr)
    return 0


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def auto_run_name(backend: str, levels: list[int], session_levels: list[int]) -> str:
    """Generate a readable run name when --run-name is omitted."""
    tag = ""
    if levels:
        tag += "l" + "".join(str(i) for i in levels)
    if session_levels:
        tag += "s" + "".join(str(i) for i in session_levels)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    return f"{backend}-{tag or 'l1'}-{stamp}"


def cmd_find(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS
    path = results_dir / "index.csv"
    if not path.exists():
        print("bench2: index not found", file=sys.stderr)
        return 1
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    name_pat = (args.name or "").lower()
    filt = [r for r in rows if name_pat in (r.get("run_name", "") or "").lower()]
    if args.type:
        filt = [r for r in filt if (r.get("type") or "") == args.type]
    if args.backend:
        filt = [r for r in filt if (r.get("backend") or "") == args.backend]
    if args.filters:
        # key=value[,key=value]
        for pair in args.filters.split(","):
            if "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            filt = [r for r in filt if str(r.get(k, "")).lower() == v.lower()]
    print(table_md(["run_name", "type", "level", "timestamp", "backend", "model",
                    "prefill_tps", "decode_tps", "agg_tps", "status"],
                   [[r.get(c, "") for c in ["run_name", "type", "level", "timestamp",
                                            "backend", "model", "prefill_tps",
                                            "decode_tps", "aggregate_tps", "status"]] for r in filt]))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS
    path = results_dir / "index.csv"
    if not path.exists():
        print("bench2: index not found", file=sys.stderr)
        return 1
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows = rows[-args.recent:]
    print(table_md(["run_name", "type", "level", "timestamp", "backend",
                    "prefill_tps", "decode_tps", "agg_tps"],
                   [[r.get(c, "") for c in ["run_name", "type", "level", "timestamp",
                                            "backend", "prefill_tps", "decode_tps",
                                            "aggregate_tps"]] for r in rows]))
    return 0


# --------------------------------------------------------------------------- #
# argparse
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench2", description="Universal llama-server benchmark 2.0")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run benchmark scenario(s)")
    run.add_argument("--run-name", default=None, help="unique run name (auto-generated if omitted)")
    run.add_argument("--level", default="", help="single levels: '2', '0,2', '1-3' (default: 1)")
    run.add_argument("--session-level", default="", help="session levels: '1', '2', '3'")
    run.add_argument("--runs", type=int, default=1, help="repeat each scenario N times")
    run.add_argument("--server-bin", default=None, help="path to llama-server executable")
    run.add_argument("--attach", default=None, help="attach to running server URL (http://host:port)")
    run.add_argument("--model", default=None, help="path to GGUF model")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=0, help="0 = auto")
    run.add_argument("--backend", choices=["auto", "rocm", "vk", "cpu"], default="auto")
    run.add_argument("--profile", default=None, help="hardware profile name")
    run.add_argument("--context-source", choices=["synthetic", "repo-snapshot", "file"], default=None)
    run.add_argument("--context-file", default=None)
    run.add_argument("--batch-size", type=int, default=None)
    run.add_argument("--ubatch-size", type=int, default=None)
    run.add_argument("--kv-k", default=None)
    run.add_argument("--kv-v", default=None)
    run.add_argument("--spec", choices=["none", "mtp"], default=None)
    run.add_argument("--spec-n", type=int, default=None, help="MTP draft max N")
    run.add_argument("--flash-attn", dest="flash_attn", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--gpu-layers", type=int, default=None)
    run.add_argument("--parallel", type=int, default=None)
    run.add_argument("--dev", default=None, help="override -dev list")
    run.add_argument("--sm", default=None, help="override -sm")
    run.add_argument("--ts", default=None, help="override -ts")
    run.add_argument("--fit", default=None)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--temperature", type=float, default=None)
    run.add_argument("--top-p", type=float, default=None)
    run.add_argument("--no-warmup", dest="no_warmup", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--warmup-shot", dest="warmup_shot", action=argparse.BooleanOptionalAction,
                     default=None, help="bench2 warmup request before measuring (default on)")
    run.add_argument("--warmup-tokens", type=int, default=None, help="warmup prompt tokens (default 512)")
    run.add_argument("--warmup-decode", type=int, default=None, help="warmup decode tokens (default 16)")
    run.add_argument("--server-extra", default="", help="raw extra server args")
    run.add_argument("--api-extra", default="",
                     help="extra JSON object merged into every completion payload "
                          "(e.g. '{\"chat_format\": 0}')")
    run.add_argument("--results-dir", default=None)
    run.add_argument("--health-timeout", type=int, default=300,
                     help="seconds to wait for server /health (default 300)")
    run.add_argument("--fail-fast", action="store_true")

    find = sub.add_parser("find", help="search runs in index")
    find.add_argument("--name", default="", help="substring of run_name")
    find.add_argument("--type", choices=["single", "session"], default=None)
    find.add_argument("--backend", default=None)
    find.add_argument("--filters", default="", help="key=value[,key=value]")
    find.add_argument("--results-dir", default=None)

    lst = sub.add_parser("list", help="recent runs")
    lst.add_argument("--recent", type=int, default=10)
    lst.add_argument("--results-dir", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "find":
        return cmd_find(args)
    if args.command == "list":
        return cmd_list(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
