"""Benchmark / autotune invocation, derived from the same RunSpec.

`scripts/agent_workload_bench.py` owns a few llama-server knobs natively; the
rest of the generated server command is forwarded verbatim via --server-extra,
so server launches and benchmark runs cannot drift apart.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from gui2.core.params import aliases_of
from gui2.core.runspec import RunSpec, to_argv

TASK_SETS = ("quick", "full", "v2", "v2-mini", "v2-review")
REAL_CONTEXT_MODES = ("off", "repo-snapshot")
BACKGROUND_POLICIES = ("fail", "warn", "ignore")
TRACE_PRESETS = ("none", "kernel-full", "vulkan-routes", "vulkan-perf", "vulkan-q3-stats")

# flags the bench script passes to llama-server itself
BENCH_OWNED: frozenset[str] = frozenset().union(*(
    aliases_of(flag) for flag in (
        "-m", "--host", "--port", "-c", "--batch-size", "--ubatch-size",
        "-ngl", "--parallel", "--cache-type-k", "--cache-type-v", "--flash-attn",
    )
)) | {"--no-warmup"}


@dataclass(frozen=True, slots=True)
class BenchSpec:
    label: str = ""
    tasks: str = "quick"
    task_ids: str = ""
    runs: int = 1
    max_tokens: int = 16
    real_context_mode: str = "repo-snapshot"
    real_context_chars: int = 24576
    no_reuse: bool = True
    v2_prime_pass: bool = False
    disable_thinking: bool = False
    request_timeout: float = 180.0
    startup_timeout: float = 900.0
    task_hard_timeout: float = 45.0
    background_server_policy: str = "fail"
    write_diagnostics: bool = True
    trace_preset: str = "none"
    autotune: bool = False


def server_extra_tokens(spec: RunSpec) -> list[str]:
    """Generated server flags minus the ones the bench script sets itself."""
    tokens = to_argv(spec)[1:]
    kept: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.split("=", 1)[0] in BENCH_OWNED:
            index += 1
            if "=" not in token and index < len(tokens) and not tokens[index].startswith("-"):
                index += 1
            continue
        kept.append(token)
        index += 1
    return kept


def _flag(value: bool, on: str, off: str) -> list[str]:
    return [on] if value else [off]


def to_bench_argv(
    spec: RunSpec,
    bench: BenchSpec,
    script: str | Path,
    server_bin: str | Path,
    python: str = "python",
) -> list[str]:
    """Full `python scripts/agent_workload_bench.py ...` command line."""
    argv = [
        python, str(script),
        "--server-bin", str(server_bin),
        "--model", spec.model,
        "--ctx-size", str(spec.ctx_size),
        "--batch-size", str(spec.batch_size),
        "--ubatch-size", str(spec.ubatch_size),
        "--gpu-layers", str(spec.gpu_layers),
        "--parallel", str(spec.parallel),
        "--cache-type-k", spec.cache_type_k,
        "--cache-type-v", spec.cache_type_v,
    ]
    argv += _flag(spec.flash_attn != "off", "--flash-attn", "--no-flash-attn")

    if bench.label:
        argv += ["--label", bench.label]
    argv += ["--tasks", bench.tasks]
    if bench.task_ids:
        argv += ["--task-ids", bench.task_ids]
    argv += [
        "--runs", str(bench.runs),
        "--max-tokens", str(bench.max_tokens),
        "--real-context-mode", bench.real_context_mode,
        "--real-context-chars", str(bench.real_context_chars),
    ]
    argv += _flag(bench.no_reuse, "--no-reuse", "--reuse")
    argv += _flag(bench.disable_thinking, "--disable-thinking", "--no-disable-thinking")
    argv += _flag(bench.v2_prime_pass, "--v2-prime-pass", "--no-v2-prime-pass")
    argv += _flag(bench.write_diagnostics, "--write-diagnostics", "--no-write-diagnostics")
    argv += [
        "--request-timeout", f"{bench.request_timeout:g}",
        "--startup-timeout", f"{bench.startup_timeout:g}",
        "--task-hard-timeout", f"{bench.task_hard_timeout:g}",
        "--background-server-policy", bench.background_server_policy,
    ]
    if bench.trace_preset != "none":
        argv += ["--trace-preset", bench.trace_preset]
    if bench.autotune:
        argv.append("--autotune")

    extra = server_extra_tokens(spec)
    if extra:
        # --flag=value form: a separate value starting with '-' would be read as a new option
        argv.append("--server-extra=" + " ".join(shlex.quote(token) for token in extra))
    return argv
