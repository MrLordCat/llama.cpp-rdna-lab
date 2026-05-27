#!/usr/bin/env python3
"""Small BFCL v4 pilot runner for local OpenAI-compatible llama-server.

This is not a replacement for the official Berkeley Function Calling
Leaderboard runner. It is a narrow pilot harness for local q3/q4 research when
the official package cannot be installed in the active Python environment. It
uses the public BFCL JSONL questions and possible-answer files and applies a
strict subset of the official AST matching rules.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = ROOT / "build_logs" / "agent-workload"
RUNS_CSV = "BFCL_LITE_RUNS.csv"
RECENT_MD = "BFCL_LITE_RECENT.md"
RECENT_LIMIT = 40

TYPE_MAP = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}

PY_TYPE_MAP = {
    "string": str,
    "integer": int,
    "float": float,
    "number": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "object": dict,
    "any": str,
}

DEFAULT_CASE_IDS = [
    "simple_python_0",
    "simple_python_1",
    "multiple_0",
    "multiple_2",
    "parallel_0",
    "parallel_3",
    "irrelevance_0",
    "irrelevance_1",
]


def _category_from_case_id(case_id: str) -> str:
    match = re.match(r"^(.*)_\d+$", case_id)
    if not match:
        raise ValueError(f"Cannot infer BFCL category from id: {case_id}")
    return match.group(1)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_cases(data_dir: Path, case_ids: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    cases: dict[str, dict[str, Any]] = {}
    answers: dict[str, dict[str, Any]] = {}
    categories = sorted({_category_from_case_id(case_id) for case_id in case_ids})

    for category in categories:
        question_path = data_dir / f"BFCL_v4_{category}.json"
        if not question_path.exists():
            raise FileNotFoundError(f"Missing BFCL question file: {question_path}")
        for row in _read_jsonl(question_path):
            if row.get("id") in case_ids:
                cases[row["id"]] = row

        answer_path = data_dir / "possible_answer" / f"BFCL_v4_{category}.json"
        if answer_path.exists():
            for row in _read_jsonl(answer_path):
                if row.get("id") in case_ids:
                    answers[row["id"]] = row

    missing = [case_id for case_id in case_ids if case_id not in cases]
    if missing:
        raise KeyError(f"Missing BFCL cases: {', '.join(missing)}")
    return cases, answers


def _convert_type(value: Any) -> Any:
    if isinstance(value, dict):
        converted = dict(value)
        if "type" in converted:
            raw_type = str(converted["type"])
            converted["type"] = TYPE_MAP.get(raw_type, "string")
            if raw_type == "float":
                converted.setdefault("description", "")
                converted["description"] += " This is a float type value."
                converted["format"] = "float"
        if "properties" in converted and isinstance(converted["properties"], dict):
            converted["properties"] = {key: _convert_type(val) for key, val in converted["properties"].items()}
        if "items" in converted and isinstance(converted["items"], dict):
            converted["items"] = _convert_type(converted["items"])
        converted.pop("optional", None)
        return converted
    return value


def _openai_name(function_name: str) -> str:
    return re.sub(r"\.", "_", function_name)


def _tools_from_functions(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for function in functions:
        converted = dict(function)
        converted["name"] = _openai_name(str(converted["name"]))
        converted["parameters"] = _convert_type(converted.get("parameters", {}))
        converted["parameters"]["type"] = "object"
        tools.append({"type": "function", "function": converted})
    return tools


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_body = response.read().decode("utf-8", errors="replace")
    return json.loads(response_body), time.perf_counter() - start


def _get_json(url: str, timeout: float) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def _wait_ready(base_url: str, timeout_s: float) -> None:
    root_url = base_url.rstrip("/")
    if root_url.endswith("/v1"):
        root_url = root_url[:-3]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        health = _get_json(f"{root_url}/health", timeout=3)
        if health is not None:
            return
        time.sleep(1)
    raise TimeoutError(f"Server did not become ready before {timeout_s:.0f}s: {root_url}/health")


def _standardize_string(value: str) -> str:
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", value).lower().replace("'", '"')


def _coerce_float(value: Any) -> Any:
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _matches_possible_value(actual: Any, accepted: list[Any], expected_type: str) -> bool:
    if actual is None:
        return "" in accepted
    if expected_type == "float":
        actual = _coerce_float(actual)
    if isinstance(actual, str):
        standardized = _standardize_string(actual)
        return any(isinstance(item, str) and standardized == _standardize_string(item) for item in accepted)
    if isinstance(actual, list):
        for candidate in accepted:
            if not isinstance(candidate, list) or len(actual) != len(candidate):
                continue
            matched = True
            for actual_item, expected_item in zip(actual, candidate):
                if isinstance(actual_item, str) and isinstance(expected_item, str):
                    if _standardize_string(actual_item) != _standardize_string(expected_item):
                        matched = False
                        break
                elif actual_item != expected_item:
                    matched = False
                    break
            if matched:
                return True
        return False
    return actual in accepted


def _find_function_description(functions: list[dict[str, Any]], expected_name: str) -> dict[str, Any] | None:
    for function in functions:
        if function.get("name") == expected_name:
            return function
    return None


def _check_single_call(
    functions: list[dict[str, Any]],
    actual_call: dict[str, Any],
    expected_call: dict[str, Any],
) -> tuple[bool, str]:
    expected_name = next(iter(expected_call.keys()))
    expected_params = expected_call[expected_name]
    actual_name = next(iter(actual_call.keys()), "")
    actual_params = actual_call.get(actual_name, {})
    expected_openai_name = _openai_name(expected_name)

    if actual_name != expected_openai_name:
        return False, f"wrong_func_name expected={expected_openai_name} actual={actual_name}"

    if not isinstance(actual_params, dict):
        return False, "args_not_object"

    description = _find_function_description(functions, expected_name)
    if description is None:
        return False, f"missing_function_description {expected_name}"
    param_schema = description.get("parameters", {}).get("properties", {})
    required_params = description.get("parameters", {}).get("required", [])

    for param in required_params:
        if param not in actual_params:
            return False, f"missing_required {param}"

    for param, actual_value in actual_params.items():
        if param not in expected_params or param not in param_schema:
            return False, f"unexpected_param {param}"
        raw_type = str(param_schema[param].get("type", "string"))
        expected_type = TYPE_MAP.get(raw_type, raw_type)
        py_type = PY_TYPE_MAP.get(expected_type)
        if expected_type == "number":
            py_type = (int, float)
        if py_type is not None and not isinstance(actual_value, py_type):
            return False, f"wrong_type {param} expected={expected_type} actual={type(actual_value).__name__}"
        bfcl_expected_type = "float" if raw_type == "float" else expected_type
        if not _matches_possible_value(actual_value, list(expected_params[param]), bfcl_expected_type):
            return False, f"wrong_value {param} actual={actual_value!r} expected={expected_params[param]!r}"

    for param, accepted_values in expected_params.items():
        if param not in actual_params and "" not in accepted_values:
            return False, f"missing_optional {param}"

    return True, "ok"


def _score_calls(category: str, functions: list[dict[str, Any]], actual_calls: list[dict[str, Any]], answer: dict[str, Any] | None) -> dict[str, Any]:
    if category == "irrelevance":
        return {
            "pass": len(actual_calls) == 0,
            "error_type": "ok" if len(actual_calls) == 0 else "unexpected_tool_call",
            "expected_calls": 0,
            "actual_calls": len(actual_calls),
        }

    if answer is None:
        return {"pass": False, "error_type": "missing_possible_answer", "expected_calls": None, "actual_calls": len(actual_calls)}

    expected_calls = list(answer.get("ground_truth", []))
    if len(actual_calls) != len(expected_calls):
        return {
            "pass": False,
            "error_type": "wrong_call_count",
            "expected_calls": len(expected_calls),
            "actual_calls": len(actual_calls),
        }

    matched: set[int] = set()
    first_error = "no_match"
    for expected_call in expected_calls:
        matched_this = False
        errors: list[str] = []
        for index, actual_call in enumerate(actual_calls):
            if index in matched:
                continue
            ok, error = _check_single_call(functions, actual_call, expected_call)
            if ok:
                matched.add(index)
                matched_this = True
                break
            errors.append(error)
        if not matched_this:
            first_error = errors[0] if errors else "no_unmatched_actual_call"
            break

    return {
        "pass": len(matched) == len(expected_calls),
        "error_type": "ok" if len(matched) == len(expected_calls) else first_error,
        "expected_calls": len(expected_calls),
        "actual_calls": len(actual_calls),
    }


def _extract_tool_calls(response: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    message = response.get("choices", [{}])[0].get("message", {})
    tool_calls = message.get("tool_calls") or []
    parsed: list[dict[str, Any]] = []
    parse_error = ""
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        name = function.get("name", "")
        raw_args = function.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError as exc:
            args = {}
            parse_error = f"invalid_json_args:{exc.msg}"
        parsed.append({name: args})
    return parsed, parse_error


def _make_payload(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    messages = list(case["question"][0])
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "tools": _tools_from_functions(case.get("function", [])),
        "tool_choice": args.tool_choice,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    if args.chat_template_enable_thinking != "unset":
        payload["chat_template_kwargs"] = {"enable_thinking": args.chat_template_enable_thinking == "true"}
    return payload


def _append_run(summary: dict[str, Any], summary_path: Path) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = HISTORY_DIR / RUNS_CSV
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "label",
                "passed",
                "total",
                "pass_rate",
                "categories",
                "mode",
                "summary_path",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": summary["timestamp"],
                "label": summary["label"],
                "passed": summary["passed"],
                "total": summary["total"],
                "pass_rate": f"{summary['pass_rate']:.4f}",
                "categories": ";".join(summary["categories"]),
                "mode": summary["mode"],
                "summary_path": summary_path.relative_to(ROOT).as_posix(),
            }
        )

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))[-RECENT_LIMIT:]
    recent_path = HISTORY_DIR / RECENT_MD
    lines = ["# BFCL Lite Recent Runs", "", "| Timestamp | Label | Pass | Categories | Mode |", "| --- | --- | ---: | --- | --- |"]
    for row in reversed(rows):
        lines.append(
            f"| {row['timestamp']} | {row['label']} | {row['passed']}/{row['total']} ({float(row['pass_rate']):.2%}) | {row['categories']} | {row['mode']} |"
        )
    recent_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_outputs(label: str, args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[Path, Path, Path, dict[str, Any]]:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = HISTORY_DIR / f"{label}.bfcl_lite.jsonl"
    csv_path = HISTORY_DIR / f"{label}.bfcl_lite.csv"
    summary_path = HISTORY_DIR / f"{label}.bfcl_lite.summary.md"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "category",
                "id",
                "pass",
                "error_type",
                "expected_calls",
                "actual_calls",
                "latency_s",
                "input_tokens",
                "output_tokens",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    total = len(rows)
    passed = sum(1 for row in rows if row.get("pass"))
    categories = sorted({str(row["category"]) for row in rows})
    by_category: dict[str, tuple[int, int]] = {}
    for category in categories:
        category_rows = [row for row in rows if row["category"] == category]
        by_category[category] = (sum(1 for row in category_rows if row.get("pass")), len(category_rows))

    mode = f"tool_choice={args.tool_choice},chat_template_enable_thinking={args.chat_template_enable_thinking}"
    summary = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "label": label,
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total else 0.0,
        "categories": categories,
        "mode": mode,
    }

    lines = [
        f"# BFCL Lite Pilot: {label}",
        "",
        "This is a small BFCL v4 pilot over public JSONL cases, not an official leaderboard score.",
        "",
        f"- BFCL data: `{args.bfcl_data_dir}`",
        f"- Endpoint: `{args.base_url}`",
        f"- Model field: `{args.model}`",
        f"- Mode: `{mode}`",
        f"- Temperature: `{args.temperature}`; max tokens: `{args.max_tokens}`; seed: `{args.seed}`",
        f"- Overall: `{passed}/{total}` (`{summary['pass_rate']:.2%}`)",
        "",
        "| Category | Pass |",
        "| --- | ---: |",
    ]
    for category, (category_passed, category_total) in by_category.items():
        lines.append(f"| `{category}` | `{category_passed}/{category_total}` |")
    failures = [row for row in rows if not row.get("pass")]
    if failures:
        lines.extend(["", "| ID | Error | Actual Calls |", "| --- | --- | ---: |"])
        for row in failures:
            lines.append(f"| `{row['id']}` | `{row['error_type']}` | `{row['actual_calls']}` |")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _append_run(summary, summary_path)
    return jsonl_path, csv_path, summary_path, summary


def run(args: argparse.Namespace) -> int:
    label = args.label or f"bfcl-lite-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    case_ids = args.case_id or DEFAULT_CASE_IDS
    data_dir = Path(args.bfcl_data_dir).resolve()
    cases, answers = _load_cases(data_dir, case_ids)
    if args.wait_ready:
        _wait_ready(args.base_url, args.startup_timeout)

    rows: list[dict[str, Any]] = []
    post_url = args.base_url.rstrip("/") + "/chat/completions"
    for case_id in case_ids:
        category = _category_from_case_id(case_id)
        case = cases[case_id]
        payload = _make_payload(args, case)
        error = ""
        response: dict[str, Any] = {}
        latency_s = 0.0
        try:
            response, latency_s = _post_json(post_url, payload, args.request_timeout)
            actual_calls, parse_error = _extract_tool_calls(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            actual_calls = []
            parse_error = ""
            error = f"request_error:{type(exc).__name__}:{exc}"

        score = _score_calls(category, case.get("function", []), actual_calls, answers.get(case_id))
        if parse_error:
            score["pass"] = False
            score["error_type"] = parse_error
        if error:
            score["pass"] = False
            score["error_type"] = error

        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        rows.append(
            {
                "label": label,
                "category": category,
                "id": case_id,
                "pass": bool(score["pass"]),
                "error_type": score["error_type"],
                "expected_calls": score.get("expected_calls"),
                "actual_calls": score.get("actual_calls"),
                "latency_s": f"{latency_s:.3f}",
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "actual": actual_calls,
                "response_content": response.get("choices", [{}])[0].get("message", {}).get("content") if response else None,
            }
        )
        print(f"{case_id}: {'PASS' if rows[-1]['pass'] else 'FAIL'} ({rows[-1]['error_type']})", flush=True)

    jsonl_path, csv_path, summary_path, summary = _write_outputs(label, args, rows)
    print(f"Summary: {summary['passed']}/{summary['total']} ({summary['pass_rate']:.2%})")
    print(f"Wrote {jsonl_path.relative_to(ROOT).as_posix()}")
    print(f"Wrote {csv_path.relative_to(ROOT).as_posix()}")
    print(f"Wrote {summary_path.relative_to(ROOT).as_posix()}")
    return 0 if summary["passed"] == summary["total"] else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfcl-data-dir", default=str((ROOT / ".." / "bfcl-eval-work" / "wheel" / "bfcl_eval" / "data").resolve()))
    parser.add_argument("--base-url", default="http://127.0.0.1:8088/v1")
    parser.add_argument("--model", default="qwen3.6-27b-q3ks-local")
    parser.add_argument("--label", default=None)
    parser.add_argument("--case-id", action="append", help="BFCL case id. Repeat to select multiple cases.")
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tool-choice", default="auto")
    parser.add_argument("--chat-template-enable-thinking", choices=["unset", "true", "false"], default="unset")
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--wait-ready", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args(sys.argv[1:])))