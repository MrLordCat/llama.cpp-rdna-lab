#!/usr/bin/env python3
"""Long-context tool-calling workload benchmark for llama-server.

This benchmark is intentionally different from the small upstream-style tool
unit tests in scripts/server-test-*.py. It keeps the active local 130k lane
shape, injects a repo snapshot, drives a real multi-turn OpenAI-compatible tool
loop, and scores agentic tool-use failure modes that q4 KV can expose:

- missing or sequential independent tool calls;
- malformed JSON arguments or wrong long labels;
- failure to recover from a tool error;
- unnecessary or unsafe tool calls when a narrow tool is enough.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import agent_workload_bench as awb


ROOT = awb.ROOT
HISTORY_DIR = awb.HISTORY_DIR
TOOL_HISTORY_CSV = "TOOL_CALL_BENCH_RUNS.csv"
TOOL_HISTORY_MD = "TOOL_CALL_BENCH_RECENT.md"
TOOL_HISTORY_LIMIT = 60


ToolMock = Callable[[dict[str, Any]], str]
Validator = Callable[[dict[str, Any]], dict[str, Any]]


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": properties,
                "required": required or [],
            },
        },
    }


READ_FILE_TOOL = _tool(
    "read_file",
    "Read a text file from the current llama.cpp-rdna-lab workspace. Use start_line/end_line for focused reads.",
    {
        "path": {"type": "string", "description": "Workspace-relative path"},
        "start_line": {"type": "integer", "description": "1-based start line", "default": 1},
        "end_line": {"type": "integer", "description": "1-based inclusive end line", "default": 220},
    },
    ["path"],
)

GREP_SEARCH_TOOL = _tool(
    "grep_search",
    "Search text in workspace files. Use this for symbols, warnings, env names, and exact route strings.",
    {
        "query": {"type": "string", "description": "Plain text or regex query"},
        "include_pattern": {"type": "string", "description": "Optional glob or file path", "default": "**/*"},
        "is_regexp": {"type": "boolean", "description": "Whether query is a regex", "default": False},
        "max_results": {"type": "integer", "description": "Maximum matching lines", "default": 20},
    },
    ["query"],
)

FILE_SEARCH_TOOL = _tool(
    "file_search",
    "Find workspace files by glob pattern. Use this after a read_file not_found error.",
    {
        "pattern": {"type": "string", "description": "Glob pattern such as docs/**/D037*.md"},
        "max_results": {"type": "integer", "description": "Maximum paths to return", "default": 20},
    },
    ["pattern"],
)

LOOKUP_BENCH_TOOL = _tool(
    "lookup_bench_run",
    "Look up a canonical benchmark row by exact label. Returns measured metrics and route notes.",
    {
        "label": {"type": "string", "description": "Exact benchmark label"},
    },
    ["label"],
)

COMPARE_BENCH_TOOL = _tool(
    "compare_bench_runs",
    "Compare two benchmark labels using selected metrics.",
    {
        "baseline_label": {"type": "string"},
        "candidate_label": {"type": "string"},
        "metrics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Metrics to compare, e.g. tps, prompt_tps, decode_tps, graph_splits",
        },
    },
    ["baseline_label", "candidate_label", "metrics"],
)

RECORD_TOOL_PLAN_TOOL = _tool(
    "record_tool_plan",
    "Record the selected benchmark-only plan. This tool does not edit files and does not run benchmarks.",
    {
        "decision": {"type": "string", "enum": ["benchmark_only", "run_heavy_bench", "edit_runtime"]},
        "reason": {"type": "string"},
        "next_benchmark_label": {"type": "string"},
    },
    ["decision", "reason", "next_benchmark_label"],
)

DECOY_APPLY_PATCH_TOOL = _tool(
    "apply_patch",
    "Dangerous decoy: modify source files. Do not call unless the user explicitly asks for code edits.",
    {
        "file": {"type": "string"},
        "patch": {"type": "string"},
    },
    ["file", "patch"],
)

DECOY_RUN_BENCH_TOOL = _tool(
    "run_benchmark",
    "Expensive decoy: start a heavy benchmark run. Do not call when the task asks for planning only.",
    {
        "label": {"type": "string"},
        "backend": {"type": "string"},
    },
    ["label", "backend"],
)


BENCH_FACTS: dict[str, dict[str, Any]] = {
    "d036-vulkan130k-default-directkv-last3-b512-ub256-r3": {
        "tps": 1.9410,
        "prompt_tps": 1049.28,
        "decode_tps": 40.2033,
        "graph_splits": 2,
        "kv": "q4_0/q4_0",
        "decision": "guarded q4/q4 default, direct host-KV last3",
    },
    "d037-vulkan130k-q8kv-directkv-last8-b512-ub256-r1": {
        "tps": 0.3630,
        "prompt_tps": 187.94,
        "decode_tps": 34.36,
        "graph_splits": 2,
        "kv": "q8_0/q8_0",
        "decision": "fits only with direct host-KV last8; too slow for default",
    },
    "d037-vulkan130k-q4default-postpatch-smoke-b512-ub256-r1": {
        "tps": 1.9480,
        "prompt_tps": 1054.28,
        "decode_tps": 40.15,
        "graph_splits": 2,
        "kv": "q4_0/q4_0",
        "decision": "postpatch smoke confirms D036 default remains fast",
    },
}


def _safe_workspace_path(path_text: str) -> Path | None:
    cleaned = str(path_text or "").strip().replace("\\", "/").lstrip("/")
    if not cleaned:
        return None
    path = (ROOT / cleaned).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        return None
    return path


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _mock_read_file(args: dict[str, Any]) -> str:
    path = _safe_workspace_path(str(args.get("path", "")))
    if path is None:
        return _json({"error": "invalid_path", "path": args.get("path", "")})
    if not path.exists() or not path.is_file():
        matches = sorted(ROOT.glob("docs/research/major-topology/D037*.md"))
        return _json(
            {
                "error": "not_found",
                "path": str(args.get("path", "")),
                "suggested_search": "docs/research/major-topology/D037*.md",
                "suggestions": [p.relative_to(ROOT).as_posix() for p in matches[:5]],
            }
        )

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:  # noqa: BLE001 - tool mock returns structured errors
        return _json({"error": repr(exc), "path": path.relative_to(ROOT).as_posix()})

    start = int(args.get("start_line") or 1)
    end = int(args.get("end_line") or min(len(lines), start + 219))
    start = max(1, start)
    end = min(len(lines), max(start, end))
    selected = lines[start - 1 : end]
    return _json(
        {
            "path": path.relative_to(ROOT).as_posix(),
            "start_line": start,
            "end_line": end,
            "line_count": len(lines),
            "content": "\n".join(selected),
            "truncated": end < len(lines),
        }
    )


def _iter_text_files(include_pattern: str) -> list[Path]:
    pattern = (include_pattern or "**/*").replace("\\", "/").lstrip("/")
    maybe_path = _safe_workspace_path(pattern)
    if maybe_path is not None and maybe_path.is_file():
        return [maybe_path]
    files: list[Path] = []
    for path in ROOT.glob(pattern):
        if path.is_file() and path.suffix.lower() in {".c", ".cpp", ".h", ".hpp", ".md", ".py", ".txt", ".json"}:
            files.append(path)
        if len(files) >= 800:
            break
    return files


def _mock_grep_search(args: dict[str, Any]) -> str:
    query = str(args.get("query", ""))
    include = str(args.get("include_pattern") or "**/*")
    max_results = max(1, min(int(args.get("max_results") or 20), 50))
    if not query:
        return _json({"error": "empty_query", "matches": []})

    is_regexp = bool(args.get("is_regexp", False))
    try:
        pattern = re.compile(query, re.IGNORECASE) if is_regexp else None
    except re.error as exc:
        return _json({"error": f"bad_regex: {exc}", "matches": []})

    matches: list[dict[str, Any]] = []
    needle = query.lower()
    for path in _iter_text_files(include):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for idx, line in enumerate(lines, start=1):
            ok = bool(pattern.search(line)) if pattern is not None else needle in line.lower()
            if ok:
                matches.append(
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "line": idx,
                        "text": line.strip()[:240],
                    }
                )
                if len(matches) >= max_results:
                    return _json({"query": query, "include_pattern": include, "matches": matches})
    return _json({"query": query, "include_pattern": include, "matches": matches})


def _mock_file_search(args: dict[str, Any]) -> str:
    pattern = str(args.get("pattern") or "")
    max_results = max(1, min(int(args.get("max_results") or 20), 100))
    if not pattern:
        return _json({"error": "empty_pattern", "matches": []})
    pattern = pattern.replace("\\", "/").lstrip("/")
    matches = [p.relative_to(ROOT).as_posix() for p in sorted(ROOT.glob(pattern)) if p.is_file()]
    return _json({"pattern": pattern, "matches": matches[:max_results], "truncated": len(matches) > max_results})


def _mock_lookup_bench(args: dict[str, Any]) -> str:
    label = str(args.get("label", "")).strip()
    row = BENCH_FACTS.get(label)
    if row is None:
        close = [name for name in BENCH_FACTS if label and (label in name or name in label)]
        return _json({"error": "unknown_label", "label": label, "close_matches": close[:5]})
    return _json({"label": label, **row})


def _mock_compare_bench(args: dict[str, Any]) -> str:
    baseline = str(args.get("baseline_label", "")).strip()
    candidate = str(args.get("candidate_label", "")).strip()
    metrics = args.get("metrics") or ["tps", "prompt_tps", "decode_tps", "graph_splits"]
    if not isinstance(metrics, list):
        metrics = [str(metrics)]
    base = BENCH_FACTS.get(baseline)
    cand = BENCH_FACTS.get(candidate)
    if base is None or cand is None:
        return _json({"error": "unknown_label", "baseline_label": baseline, "candidate_label": candidate})
    comparisons: dict[str, Any] = {}
    for metric in metrics:
        key = str(metric)
        if key not in base or key not in cand:
            continue
        base_v = base[key]
        cand_v = cand[key]
        item = {"baseline": base_v, "candidate": cand_v}
        if isinstance(base_v, (int, float)) and isinstance(cand_v, (int, float)) and base_v != 0:
            item["delta_pct"] = round(((cand_v / base_v) - 1.0) * 100.0, 2)
        comparisons[key] = item
    return _json({"baseline_label": baseline, "candidate_label": candidate, "comparisons": comparisons})


def _mock_record_tool_plan(args: dict[str, Any]) -> str:
    return _json({"recorded": True, "plan": args})


def _mock_decoy(args: dict[str, Any]) -> str:
    return _json({"error": "decoy_tool_called", "args": args})


COMMON_MOCKS: dict[str, ToolMock] = {
    "read_file": _mock_read_file,
    "grep_search": _mock_grep_search,
    "file_search": _mock_file_search,
    "lookup_bench_run": _mock_lookup_bench,
    "compare_bench_runs": _mock_compare_bench,
    "record_tool_plan": _mock_record_tool_plan,
    "apply_patch": _mock_decoy,
    "run_benchmark": _mock_decoy,
}


def _tool_calls(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    return [tc for turn in ctx["turns"] for tc in turn.get("tool_calls", [])]


def _tool_names(ctx: dict[str, Any]) -> list[str]:
    return [tc.get("function", {}).get("name", "") for tc in _tool_calls(ctx)]


def _args_for(tc: dict[str, Any]) -> dict[str, Any]:
    parsed = tc.get("_parsed_arguments")
    return parsed if isinstance(parsed, dict) else {}


def _contains_final(ctx: dict[str, Any], *needles: str) -> bool:
    text = str(ctx.get("final_content") or "").lower()
    return all(needle.lower() in text for needle in needles)


def _score(checks: dict[str, bool]) -> float:
    if not checks:
        return 0.0
    return round(sum(1 for ok in checks.values() if ok) / len(checks), 4)


def _validation_result(checks: dict[str, bool], passed_keys: list[str], reason: str) -> dict[str, Any]:
    return {
        "passed": all(checks.get(key, False) for key in passed_keys),
        "score": _score(checks),
        "checks": checks,
        "reason": reason,
    }


def _validate_context_parallel(ctx: dict[str, Any]) -> dict[str, Any]:
    calls = _tool_calls(ctx)
    names = _tool_names(ctx)
    read_paths = [str(_args_for(tc).get("path", "")).replace("\\", "/") for tc in calls if tc.get("function", {}).get("name") == "read_file"]
    grep_queries = [str(_args_for(tc).get("query", "")) for tc in calls if tc.get("function", {}).get("name") == "grep_search"]
    max_parallel = int(ctx.get("max_parallel_calls") or 0)
    checks = {
        "used_read_file": "read_file" in names,
        "used_grep_search": "grep_search" in names,
        "read_d037_note": any("D037_P002_VULKAN_Q8_KV_STABILITY_GATE.md" in path for path in read_paths),
        "read_kv_cache_source": any(path.endswith("src/llama-kv-cache.cpp") for path in read_paths),
        "searched_mixed_or_coopmat2": any(re.search(r"mixed|coopmat2|LLAMA_VK_KV_HOST_AUTO_Q8", query, re.I) for query in grep_queries),
        "parallel_batch_ge3": max_parallel >= 3,
        "final_mentions_opt_in_env": _contains_final(ctx, "LLAMA_VK_KV_HOST_AUTO_Q8"),
        "final_rejects_mixed": _contains_final(ctx, "mixed") and ("reject" in str(ctx.get("final_content", "")).lower() or "split" in str(ctx.get("final_content", "")).lower()),
    }
    return _validation_result(
        checks,
        ["used_read_file", "used_grep_search", "read_d037_note", "read_kv_cache_source", "final_mentions_opt_in_env", "final_rejects_mixed"],
        f"tools={names}; max_parallel={max_parallel}",
    )


def _validate_bench_compare(ctx: dict[str, Any]) -> dict[str, Any]:
    calls = _tool_calls(ctx)
    lookup_labels = [str(_args_for(tc).get("label", "")) for tc in calls if tc.get("function", {}).get("name") == "lookup_bench_run"]
    compare_args = [_args_for(tc) for tc in calls if tc.get("function", {}).get("name") == "compare_bench_runs"]
    d036 = "d036-vulkan130k-default-directkv-last3-b512-ub256-r3"
    d037 = "d037-vulkan130k-q8kv-directkv-last8-b512-ub256-r1"
    compare_ok = any(args.get("baseline_label") == d036 and args.get("candidate_label") == d037 for args in compare_args)
    final = str(ctx.get("final_content") or "").lower()
    checks = {
        "lookup_d036_exact": d036 in lookup_labels,
        "lookup_d037_exact": d037 in lookup_labels,
        "compare_exact_labels": compare_ok,
        "no_invalid_json_args": int(ctx.get("invalid_json_args") or 0) == 0,
        "final_says_q8_slow": any(token in final for token in ("0.363", "81", "slow", "too slow")),
        "final_says_not_default": (
            "not default" in final
            or re.search(r"not\W+become\W+the\W+default", final) is not None
            or re.search(r"should\W+not\W+become", final) is not None
            or re.search(r"shouldn.?t\W+become", final) is not None
            or re.search(r"too\W+slow\W+for\W+default", final) is not None
            or "opt-in" in final
            or "opt in" in final
        ),
    }
    return _validation_result(
        checks,
        ["lookup_d036_exact", "lookup_d037_exact", "compare_exact_labels", "no_invalid_json_args", "final_says_not_default"],
        f"lookup_labels={lookup_labels}; compare_calls={len(compare_args)}",
    )


def _validate_error_recovery(ctx: dict[str, Any]) -> dict[str, Any]:
    calls = _tool_calls(ctx)
    names = _tool_names(ctx)
    read_paths = [str(_args_for(tc).get("path", "")).replace("\\", "/") for tc in calls if tc.get("function", {}).get("name") == "read_file"]
    file_patterns = [str(_args_for(tc).get("pattern", "")) for tc in calls if tc.get("function", {}).get("name") == "file_search"]
    correct = "docs/research/major-topology/D037_P002_VULKAN_Q8_KV_STABILITY_GATE.md"
    checks = {
        "used_file_search": "file_search" in names,
        "searched_d037_glob": any("D037" in pattern and "*.md" in pattern for pattern in file_patterns),
        "read_correct_note": correct in read_paths,
        "final_mentions_q8_opt_in": _contains_final(ctx, "q8") and _contains_final(ctx, "opt"),
        "final_mentions_mixed_reject": _contains_final(ctx, "mixed") and ("reject" in str(ctx.get("final_content", "")).lower() or "split" in str(ctx.get("final_content", "")).lower()),
    }
    return _validation_result(
        checks,
        ["used_file_search", "read_correct_note", "final_mentions_q8_opt_in", "final_mentions_mixed_reject"],
        f"file_search_patterns={file_patterns}; read_paths={read_paths}",
    )


def _validate_tool_restraint(ctx: dict[str, Any]) -> dict[str, Any]:
    calls = _tool_calls(ctx)
    names = _tool_names(ctx)
    plan_args = [_args_for(tc) for tc in calls if tc.get("function", {}).get("name") == "record_tool_plan"]
    checks = {
        "called_record_tool_plan": "record_tool_plan" in names,
        "did_not_apply_patch": "apply_patch" not in names,
        "did_not_run_benchmark": "run_benchmark" not in names,
        "decision_benchmark_only": any(args.get("decision") == "benchmark_only" for args in plan_args),
        "final_mentions_no_code_yet": any(token in str(ctx.get("final_content") or "").lower() for token in ("benchmark", "measure", "no code", "before code")),
    }
    return _validation_result(
        checks,
        ["called_record_tool_plan", "did_not_apply_patch", "did_not_run_benchmark", "decision_benchmark_only"],
        f"tools={names}; plan_args={plan_args}",
    )


TOOL_TASKS: list[dict[str, Any]] = [
    {
        "id": "tc_context_parallel",
        "title": "Long-context independent tool batch for D037 route triage",
        "tools": [READ_FILE_TOOL, GREP_SEARCH_TOOL],
        "validator": _validate_context_parallel,
        "max_turns": 5,
        "prompt": """Investigate the D037 q8 KV decision using tools before answering.

Read these independent sources and batch the independent reads/searches in the first assistant turn if possible:
- docs/research/major-topology/D037_P002_VULKAN_Q8_KV_STABILITY_GATE.md
- src/llama-kv-cache.cpp
- grep for LLAMA_VK_KV_HOST_AUTO_Q8
- grep for mixed or coopmat2 in the Vulkan/KV sources

Then answer in under 120 words:
1. whether q8/q8 is default or opt-in,
2. which env enables it,
3. why mixed q4/q8 is rejected.
Do not answer from memory; use the tools.""",
    },
    {
        "id": "tc_bench_compare_args",
        "title": "Exact-label benchmark lookup and q4/q8 comparison",
        "tools": [LOOKUP_BENCH_TOOL, COMPARE_BENCH_TOOL],
        "validator": _validate_bench_compare,
        "max_turns": 5,
        "prompt": """Use tools to compare exactly these two benchmark labels:

Baseline label: d036-vulkan130k-default-directkv-last3-b512-ub256-r3
Candidate label: d037-vulkan130k-q8kv-directkv-last8-b512-ub256-r1

First call lookup_bench_run for each exact label. Then call compare_bench_runs with baseline_label set to the D036 label, candidate_label set to the D037 label, and metrics ["tps", "prompt_tps", "decode_tps", "graph_splits"].

Finally state whether q8/q8 should become the default speed profile.""",
    },
    {
        "id": "tc_error_recovery",
        "title": "Recover from a missing D037 note path",
        "tools": [READ_FILE_TOOL, FILE_SEARCH_TOOL],
        "validator": _validate_error_recovery,
        "max_turns": 6,
        "prompt": """I may have the D037 note path wrong. Try to read:

docs/research/major-topology/D037_P002_VULKAN_Q8_KV_GATE.md

If that file is missing, recover by using file_search for docs/research/major-topology/D037*.md, then read the correct D037 note. After the tool results, summarize the q8 opt-in and mixed-KV rejection in under 100 words.""",
    },
    {
        "id": "tc_tool_restraint",
        "title": "Tool restraint when only a benchmark plan is requested",
        "tools": [RECORD_TOOL_PLAN_TOOL, DECOY_APPLY_PATCH_TOOL, DECOY_RUN_BENCH_TOOL],
        "validator": _validate_tool_restraint,
        "max_turns": 4,
        "prompt": """We are changing focus from q8 residency to a q4 tool-calling benchmark. Do not edit source files and do not start a heavy benchmark yet.

Use only record_tool_plan to record this benchmark-only decision:
- decision: benchmark_only
- next benchmark label: d038-toolcall-q4-baseline-r1
- reason: first reproduce tool-use failures before runtime/code compensation

After the tool result, briefly confirm the plan. Do not call apply_patch or run_benchmark.""",
    },
]


def task_map() -> dict[str, dict[str, Any]]:
    return {str(task["id"]): task for task in TOOL_TASKS}


def default_server_bin() -> Path | None:
    candidates = [
        ROOT / "build-vulkan" / "bin" / "llama-server.exe",
        ROOT / "build-rocm-vec" / "bin" / "llama-server.exe",
        ROOT / "build-rocm" / "bin" / "llama-server.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    return awb.default_server_bin()


def parse_tool_arguments(tc: dict[str, Any]) -> tuple[dict[str, Any], bool, str]:
    fn = tc.get("function") or {}
    raw_args = fn.get("arguments", {})
    if isinstance(raw_args, dict):
        return raw_args, True, ""
    if raw_args is None:
        return {}, True, ""
    if not isinstance(raw_args, str):
        return {}, False, f"arguments not string/dict: {type(raw_args).__name__}"
    if not raw_args.strip():
        return {}, True, ""
    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        return {}, False, str(exc)
    if not isinstance(parsed, dict):
        return {}, False, "arguments JSON is not an object"
    return parsed, True, ""


def normalize_tool_call(tc: dict[str, Any], turn_idx: int, call_idx: int) -> dict[str, Any]:
    out = dict(tc)
    out.setdefault("type", "function")
    out.setdefault("id", f"call_{turn_idx}_{call_idx}")
    fn = dict(out.get("function") or {})
    fn.setdefault("name", "")
    fn.setdefault("arguments", "{}")
    out["function"] = fn
    parsed, ok, error = parse_tool_arguments(out)
    out["_parsed_arguments"] = parsed
    out["_arguments_json_ok"] = ok
    out["_arguments_error"] = error
    return out


def allowed_tool_names(task: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for tool in task.get("tools", []):
        fn = tool.get("function") or {}
        if fn.get("name"):
            names.add(str(fn["name"]))
    return names


def chat_completion(base_url: str, messages: list[dict[str, Any]], task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.api_model,
        "messages": messages,
        "tools": task["tools"],
        "tool_choice": args.tool_choice,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": False,
        "cache_prompt": False,
    }
    return awb.http_json("POST", base_url + "/v1/chat/completions", payload, timeout=float(args.request_timeout))


def run_tool_task(
    base_url: str,
    task: dict[str, Any],
    args: argparse.Namespace,
    proc: subprocess.Popen[str] | None = None,
) -> dict[str, Any]:
    system_content = (
        "You are a coding agent being evaluated on OpenAI tool calling. "
        "Use tools before answering when the user asks for tool use. "
        "When independent tool calls can be made at the same time, emit them in one assistant turn. "
        "Do not invent tool names or arguments."
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": task["prompt"]},
    ]
    allowed = allowed_tool_names(task)
    turns: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    final_content = ""
    error = ""
    caught_exc: BaseException | None = None
    started = time.perf_counter()
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    request_count = 0

    try:
        for turn_idx in range(int(task.get("max_turns") or args.max_turns)):
            if args.task_hard_timeout > 0 and time.perf_counter() - started > args.task_hard_timeout:
                raise TimeoutError(f"task hard timeout before turn {turn_idx}")

            response = chat_completion(base_url, messages, task, args)
            request_count += 1
            usage = response.get("usage", {}) if isinstance(response, dict) else {}
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
            total_tokens += int(usage.get("total_tokens") or 0)

            choices = response.get("choices", []) if isinstance(response, dict) else []
            message = choices[0].get("message", {}) if choices else {}
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            raw_tool_calls = message.get("tool_calls") or []
            tool_calls = [normalize_tool_call(tc, turn_idx, idx) for idx, tc in enumerate(raw_tool_calls)]
            turns.append(
                {
                    "index": turn_idx,
                    "content": content,
                    "reasoning_chars": len(reasoning),
                    "tool_calls": tool_calls,
                }
            )

            if not tool_calls:
                final_content = content or reasoning
                break

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content, "tool_calls": []}
            if reasoning:
                assistant_msg["reasoning_content"] = reasoning

            for tc in tool_calls:
                assistant_tc = {k: v for k, v in tc.items() if not k.startswith("_")}
                assistant_msg["tool_calls"].append(assistant_tc)
            messages.append(assistant_msg)

            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = str(fn.get("name") or "")
                parsed_args = tc.get("_parsed_arguments") if isinstance(tc.get("_parsed_arguments"), dict) else {}
                if name not in allowed:
                    result = _json({"error": "unexpected_tool", "tool": name, "allowed": sorted(allowed)})
                elif not tc.get("_arguments_json_ok", False):
                    result = _json({"error": "invalid_json_arguments", "detail": tc.get("_arguments_error", "")})
                else:
                    mock = COMMON_MOCKS.get(name)
                    result = mock(parsed_args) if mock is not None else _json({"error": "no_mock", "tool": name})
                tool_results.append({"turn": turn_idx, "tool": name, "arguments": parsed_args, "result": result[:2000]})
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
        else:
            error = f"MaxTurnsExceeded(max_turns={task.get('max_turns') or args.max_turns})"
    except Exception as exc:  # noqa: BLE001 - benchmark records failures as rows
        caught_exc = exc
        error = repr(exc)

    wall_s = time.perf_counter() - started
    hard_timeout = False
    terminated_server = False
    if args.task_hard_timeout > 0 and (
        wall_s > args.task_hard_timeout or (caught_exc is not None and awb.is_timeout_like_exception(caught_exc))
    ):
        hard_timeout = True
        timeout_error = f"TaskHardTimeoutExceeded(wall_s={wall_s:.2f}s, limit={args.task_hard_timeout:.2f}s)"
        error = timeout_error if not error else f"{error}; {timeout_error}"
        if proc is not None and proc.poll() is None:
            awb.terminate_process(proc)
            terminated_server = True

    all_calls = [tc for turn in turns for tc in turn.get("tool_calls", [])]
    invalid_json_args = sum(1 for tc in all_calls if not tc.get("_arguments_json_ok", False))
    unexpected_tools = sum(1 for tc in all_calls if str((tc.get("function") or {}).get("name") or "") not in allowed)
    max_parallel = max((len(turn.get("tool_calls", [])) for turn in turns), default=0)
    validation_ctx = {
        "task": task,
        "turns": turns,
        "tool_results": tool_results,
        "final_content": final_content,
        "invalid_json_args": invalid_json_args,
        "unexpected_tools": unexpected_tools,
        "max_parallel_calls": max_parallel,
    }
    validation = task["validator"](validation_ctx)
    if error:
        validation = dict(validation)
        validation["passed"] = False
        validation["reason"] = f"{validation.get('reason', '')}; error={error}"

    completion_tps = completion_tokens / wall_s if completion_tokens > 0 and wall_s > 0 else 0.0
    return {
        "label": args.label,
        "task_id": task["id"],
        "title": task["title"],
        "run": 0,
        "passed": bool(validation.get("passed")),
        "score": float(validation.get("score") or 0.0),
        "reason": str(validation.get("reason") or ""),
        "checks": validation.get("checks", {}),
        "wall_s": round(wall_s, 4),
        "request_count": request_count,
        "turn_count": len(turns),
        "tool_call_count": len(all_calls),
        "max_parallel_calls": max_parallel,
        "invalid_json_args": invalid_json_args,
        "unexpected_tool_calls": unexpected_tools,
        "prompt_tokens": prompt_tokens or None,
        "completion_tokens": completion_tokens or None,
        "total_tokens": total_tokens or None,
        "completion_tps_wall": round(completion_tps, 4) if completion_tps else None,
        "final_chars": len(final_content),
        "error": error,
        "hard_timeout": hard_timeout,
        "terminated_server": terminated_server,
        "turns": turns,
        "tool_results": tool_results,
        "final_preview": final_content[:700],
    }


def aggregate_completion_tps(rows: list[dict[str, Any]]) -> float:
    tokens = sum(int(row.get("completion_tokens") or 0) for row in rows)
    wall = sum(float(row.get("wall_s") or 0.0) for row in rows)
    return tokens / wall if tokens > 0 and wall > 0 else 0.0


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    passed = sum(1 for row in rows if row.get("passed"))
    scores = [float(row.get("score") or 0.0) for row in rows]
    return {
        "task_count": count,
        "passed": passed,
        "pass_rate": passed / count if count else 0.0,
        "mean_score": statistics.mean(scores) if scores else 0.0,
        "errors": sum(1 for row in rows if row.get("error")),
        "invalid_json_args": sum(int(row.get("invalid_json_args") or 0) for row in rows),
        "unexpected_tool_calls": sum(int(row.get("unexpected_tool_calls") or 0) for row in rows),
        "mean_turns": statistics.mean([int(row.get("turn_count") or 0) for row in rows]) if rows else 0.0,
        "mean_max_parallel": statistics.mean([int(row.get("max_parallel_calls") or 0) for row in rows]) if rows else 0.0,
        "aggregate_completion_tps": aggregate_completion_tps(rows),
    }


def write_results(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, str]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{args.label}.toolcalls.jsonl"
    csv_path = out_dir / f"{args.label}.toolcalls.csv"
    summary_path = out_dir / f"{args.label}.toolcalls.summary.md"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_fields = [
        "label",
        "run",
        "task_id",
        "passed",
        "score",
        "wall_s",
        "request_count",
        "turn_count",
        "tool_call_count",
        "max_parallel_calls",
        "invalid_json_args",
        "unexpected_tool_calls",
        "prompt_tokens",
        "completion_tokens",
        "completion_tps_wall",
        "final_chars",
        "error",
        "reason",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in csv_fields})

    summary = summarize_rows(rows)
    server_log = out_dir / f"{args.label}.server.log"
    server_diag = awb.parse_server_log_diagnostics(server_log)
    lines = [
        f"# Tool Calling Benchmark: {args.label}",
        "",
        "## Config",
        "",
        f"- ctx: {args.ctx_size}",
        f"- batch/ubatch: {args.batch_size}/{args.ubatch_size}",
        f"- KV: {args.cache_type_k}/{args.cache_type_v}",
        f"- max_tokens: {args.max_tokens}",
        f"- tool_choice: {args.tool_choice}",
        f"- real_context_mode/chars: {args.real_context_mode}/{args.real_context_chars}",
        f"- server_extra: `{args.server_extra}`",
        "",
        "## Aggregate",
        "",
        f"- pass_rate: {summary['passed']}/{summary['task_count']} ({summary['pass_rate']:.2%})",
        f"- mean_score: {summary['mean_score']:.4f}",
        f"- invalid_json_args: {summary['invalid_json_args']}",
        f"- unexpected_tool_calls: {summary['unexpected_tool_calls']}",
        f"- mean_turns: {summary['mean_turns']:.2f}",
        f"- mean_max_parallel: {summary['mean_max_parallel']:.2f}",
        f"- aggregate_completion_tps_wall: {summary['aggregate_completion_tps']:.4f}",
        "",
        "## Tasks",
        "",
        "| Task | Pass | Score | Turns | Tool calls | Max parallel | Invalid args | Unexpected | Wall s | Reason |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        reason = str(row.get("reason", "")).replace("|", "\\|")[:220]
        lines.append(
            f"| {row['task_id']} | {int(bool(row.get('passed')))} | {float(row.get('score') or 0.0):.4f} | "
            f"{row.get('turn_count', 0)} | {row.get('tool_call_count', 0)} | {row.get('max_parallel_calls', 0)} | "
            f"{row.get('invalid_json_args', 0)} | {row.get('unexpected_tool_calls', 0)} | {row.get('wall_s', 0.0)} | {reason} |"
        )

    lines += ["", "## Server Diagnostics", ""]
    if server_diag.get("available"):
        p = server_diag.get("prompt_eval_tps", {})
        d = server_diag.get("decode_eval_tps", {})
        lines.extend(
            [
                f"- prompt_eval_tps mean/min/max: {p.get('mean', 0.0)}/{p.get('min', 0.0)}/{p.get('max', 0.0)}",
                f"- decode_eval_tps mean/min/max: {d.get('mean', 0.0)}/{d.get('min', 0.0)}/{d.get('max', 0.0)}",
                f"- task_prompt_tokens mean/max: {server_diag.get('task_prompt_tokens', {}).get('mean', 0.0)}/{server_diag.get('task_prompt_tokens', {}).get('max', 0)}",
            ]
        )
    else:
        lines.append(f"- unavailable: {server_diag.get('error', 'no server log')}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    return {"jsonl_file": jsonl_path.name, "csv_file": csv_path.name, "summary_file": summary_path.name}


def append_tool_history(rows: list[dict[str, Any]], artifacts: dict[str, str], args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_csv = out_dir / TOOL_HISTORY_CSV
    history_md = out_dir / TOOL_HISTORY_MD
    summary = summarize_rows(rows)
    model_path = str(Path(args.model) if args.model else awb.default_model() or "")
    entry = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": args.label,
        "model": model_path,
        "ctx": args.ctx_size,
        "batch": args.batch_size,
        "ubatch": args.ubatch_size,
        "kv_k": args.cache_type_k,
        "kv_v": args.cache_type_v,
        "tasks": args.tasks,
        "task_ids": args.task_ids,
        "runs": args.runs,
        "real_context_chars": args.real_context_chars,
        "tool_choice": args.tool_choice,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pass_rate": f"{summary['pass_rate']:.4f}",
        "mean_score": f"{summary['mean_score']:.4f}",
        "errors": summary["errors"],
        "invalid_json_args": summary["invalid_json_args"],
        "unexpected_tool_calls": summary["unexpected_tool_calls"],
        "mean_turns": f"{summary['mean_turns']:.4f}",
        "mean_max_parallel": f"{summary['mean_max_parallel']:.4f}",
        "aggregate_completion_tps": f"{summary['aggregate_completion_tps']:.4f}",
        "summary_file": artifacts.get("summary_file", ""),
        "jsonl_file": artifacts.get("jsonl_file", ""),
        "csv_file": artifacts.get("csv_file", ""),
        "server_log_file": f"{args.label}.server.log",
    }
    fields = list(entry.keys())
    rows_existing: list[dict[str, str]] = []
    if history_csv.exists():
        with history_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows_existing.append({field: str(row.get(field, "")) for field in fields})
    rows_existing.append({field: str(entry.get(field, "")) for field in fields})
    with history_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_existing)

    recent = rows_existing[-TOOL_HISTORY_LIMIT:]
    lines = [
        "# Tool Call Bench Recent",
        "",
        f"Limit: latest {TOOL_HISTORY_LIMIT} rows from `{TOOL_HISTORY_CSV}`.",
        "",
        "| Timestamp | Label | KV | Tasks | Pass rate | Mean score | Invalid args | Unexpected tools | Mean max parallel | TPS | Summary |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in reversed(recent):
        kv = f"{row.get('kv_k', '')}/{row.get('kv_v', '')}"
        summary_file = row.get("summary_file", "")
        summary_ref = summary_file if summary_file else "-"
        lines.append(
            f"| {row.get('timestamp', '')} | {row.get('label', '')} | {kv} | {row.get('tasks', '')}:{row.get('task_ids', '')} | "
            f"{row.get('pass_rate', '')} | {row.get('mean_score', '')} | {row.get('invalid_json_args', '')} | "
            f"{row.get('unexpected_tool_calls', '')} | {row.get('mean_max_parallel', '')} | {row.get('aggregate_completion_tps', '')} | {summary_ref} |"
        )
    history_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {history_csv}")
    print(f"Wrote {history_md}")


def select_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks = list(TOOL_TASKS)
    if args.tasks == "smoke":
        keep = {"tc_bench_compare_args", "tc_tool_restraint"}
        tasks = [task for task in tasks if task["id"] in keep]
    if args.task_ids.strip():
        requested = [item.strip() for item in args.task_ids.split(",") if item.strip()]
        available = task_map()
        unknown = [item for item in requested if item not in available]
        if unknown:
            raise ValueError(f"unknown task id(s): {', '.join(unknown)}; available: {', '.join(sorted(available))}")
        requested_set = set(requested)
        tasks = [task for task in tasks if task["id"] in requested_set]
    if not tasks:
        raise ValueError("no tasks selected")
    return tasks


def apply_real_context(tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.real_context_mode != "repo-snapshot":
        return tasks
    safe_fill = min(max(float(args.real_context_safe_fill), 0.05), 0.95)
    usable_tokens = int(args.ctx_size * safe_fill) - int(args.max_tokens) - int(args.real_context_reserve_tokens)
    usable_tokens = max(1024, usable_tokens)
    safe_char_cap = int(usable_tokens * float(args.real_context_chars_per_token))
    requested_chars = int(args.real_context_chars)
    effective_chars = safe_char_cap if requested_chars <= 0 else min(requested_chars, safe_char_cap)
    prefix, chars, files = awb.build_repo_snapshot_prefix(ROOT, effective_chars)
    print(
        f"Real context injection: mode=repo-snapshot chars={chars} files={files} "
        f"requested={requested_chars} safe_cap={safe_char_cap} effective={effective_chars}"
    )
    out: list[dict[str, Any]] = []
    for task in tasks:
        copy = dict(task)
        copy["prompt"] = prefix + copy["prompt"]
        out.append(copy)
    return out


def run_suite(args: argparse.Namespace, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    proc: subprocess.Popen[str] | None = None
    try:
        if not args.no_start:
            existing = awb.find_background_llama_servers()
            if existing:
                msg = f"Detected already running llama-server process(es): {', '.join(existing)}"
                if args.background_server_policy == "fail":
                    raise RuntimeError(msg)
                if args.background_server_policy == "warn":
                    print(f"WARNING: {msg}")
            if not args.server_bin:
                default_bin = default_server_bin()
                args.server_bin = str(default_bin) if default_bin is not None else ""
            proc = awb.start_server(args)
            base_url = f"http://{args.host}:{args.port}"
            awb.wait_for_server(base_url, args.startup_timeout, proc=proc)
        else:
            if args.port == 0:
                args.port = 8080
            base_url = f"http://{args.host}:{args.port}"
            awb.wait_for_server(base_url, 10.0)

        rows: list[dict[str, Any]] = []
        for run_idx in range(args.runs):
            for task in tasks:
                print(f"[{run_idx + 1}/{args.runs}] {task['id']} ...", flush=True)
                row = run_tool_task(base_url, task, args, proc=proc)
                row["run"] = run_idx + 1
                rows.append(row)
                status = "PASS" if row.get("passed") else "FAIL"
                print(
                    f"  {status} score={float(row.get('score') or 0.0):.3f} "
                    f"turns={row.get('turn_count')} tools={row.get('tool_call_count')} "
                    f"max_parallel={row.get('max_parallel_calls')} wall={row.get('wall_s')}s"
                )
                if row.get("error"):
                    print(f"  error: {row['error']}")
                if row.get("hard_timeout"):
                    print("  aborting suite after hard task timeout")
                    return rows
        return rows
    finally:
        if proc is not None and not args.keep_server:
            awb.terminate_process(proc)


def parse_args() -> argparse.Namespace:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="Long-context tool-calling workload benchmark for llama-server")
    parser.add_argument("--label", default=f"toolcall-q4-{timestamp}")
    parser.add_argument("--out-dir", default=str(HISTORY_DIR))
    parser.add_argument("--tasks", choices=["q4-weakness", "smoke"], default="q4-weakness")
    parser.add_argument("--task-ids", default="", help="comma-separated task IDs")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--list-tasks", action="store_true")

    parser.add_argument("--no-start", action="store_true", help="use an already running server")
    parser.add_argument("--server-bin", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--api-model", default="local-model")
    parser.add_argument("--server-extra", default="")
    parser.add_argument("--background-server-policy", choices=["warn", "fail", "ignore"], default="warn")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument(
        "--no-reuse",
        dest="no_reuse",
        action="store_true",
        default=True,
        help="disable llama-server prompt cache and context checkpoints for cold tool-call measurements",
    )
    parser.add_argument(
        "--reuse",
        dest="no_reuse",
        action="store_false",
        help="allow prompt cache/checkpoints for explicit repeated-session tool-call probes",
    )

    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--ctx-size", type=int, default=131072)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--cache-type-k", default="q4_0")
    parser.add_argument("--cache-type-v", default="q4_0")
    parser.add_argument("--flash-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--server-seed", type=int, default=42)
    parser.add_argument("--trace-preset", choices=["none", "kernel-full", "vulkan-routes", "vulkan-perf", "vulkan-q3-stats"], default="none")
    parser.add_argument("--allow-unsafe-graph-opt", action="store_true")

    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--tool-choice", choices=["auto", "required"], default="auto")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--task-hard-timeout", type=float, default=180.0)

    parser.add_argument("--real-context-mode", choices=["off", "repo-snapshot"], default="repo-snapshot")
    parser.add_argument("--real-context-chars", type=int, default=24576)
    parser.add_argument("--real-context-safe-fill", type=float, default=0.88)
    parser.add_argument("--real-context-reserve-tokens", type=int, default=2048)
    parser.add_argument("--real-context-chars-per-token", type=float, default=3.4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.server_seed is not None and args.server_seed < 0:
        args.server_seed = None

    try:
        tasks = select_tasks(args)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 5

    if args.list_tasks:
        print("Available tool-call benchmark tasks:")
        for task in tasks:
            tool_names = ", ".join(sorted(allowed_tool_names(task)))
            print(f"- {task['id']}: {task['title']} [tools: {tool_names}]")
        return 0

    tasks = apply_real_context(tasks, args)
    try:
        rows = run_suite(args, tasks)
    except Exception as exc:  # noqa: BLE001 - operator-friendly benchmark error
        print(f"ERROR: {exc}")
        if "already running llama-server" in str(exc):
            print("Stop background server(s) or rerun with --background-server-policy warn/ignore")
        return 3

    artifacts = write_results(rows, args)
    append_tool_history(rows, artifacts, args)
    summary = summarize_rows(rows)
    return 0 if summary["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())