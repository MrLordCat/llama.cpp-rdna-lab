#!/usr/bin/env python3
"""Run a real-scenario context benchmark using a large repository snapshot prompt.

Unlike agent_workload_bench tasks (short prompts), this script builds a large prompt
from real repository files and scales prompt size with context. The active default
is now the 130k lane; older 12k/16k/64k profiles are historical probes.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import signal
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_DEFAULT = ROOT / "build_logs" / "agent-workload"
PRIMARY_MAX_CTX = 131072


def parse_args() -> argparse.Namespace:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(
        description="Real-scenario benchmark: repository snapshot prompt scaled by context",
    )
    parser.add_argument("--label-prefix", default=f"repo-snapshot-{stamp}")
    parser.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    parser.add_argument("--server-bin", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ctx-values", default="131072")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--cache-type-k", default="q4_0")
    parser.add_argument("--cache-type-v", default="q4_0")
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--threads-http", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--server-extra", default="", help="extra llama-server args appended as-is")
    parser.add_argument("--base-char-budget", type=int, default=386000)
    parser.add_argument("--min-char-budget", type=int, default=120000)
    parser.add_argument("--base-ctx", type=int, default=131072)
    parser.add_argument(
        "--allow-ctx-above-16k",
        action="store_true",
        help="legacy compatibility flag; active policy allows ctx up to 131072 and uses this only for explicit over-130k probes",
    )
    parser.add_argument("--spec-type", choices=["none", "ngram-mod"], default="none")
    parser.add_argument(
        "--quick-profile",
        choices=["off", "64k-smoke"],
        default="off",
        help="optional faster screening preset for real-scenario comparisons",
    )
    parser.add_argument(
        "--compare-to",
        default="",
        help="path to a baseline result.json from a previous run; prints a quick verdict vs baseline",
    )
    parser.add_argument(
        "--compare-threshold-pct",
        type=float,
        default=2.0,
        help="minimum percent delta required before classifying a run as better/worse vs baseline",
    )
    parser.add_argument("--no-warmup", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def apply_quick_profile(args: argparse.Namespace) -> None:
    if args.quick_profile != "64k-smoke":
        return

    # Keep the quick profile materially cheaper than the full 64k run,
    # but large enough to preserve prompt-heavy behavior and ranking.
    args.base_char_budget = 240000
    args.min_char_budget = 120000
    args.max_tokens = min(args.max_tokens, 24)
    args.request_timeout = min(args.request_timeout, 900.0)


def parse_ctx_values(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--ctx-values must contain at least one value, e.g. 131072")
    if len(set(values)) != len(values):
        raise ValueError("--ctx-values must not contain duplicates")
    if any(v < 4096 for v in values):
        raise ValueError("all --ctx-values must be >= 4096")
    return values


def wait_ready(base_url: str, timeout_s: float) -> None:
    probes = [f"{base_url}/v1/models", f"{base_url}/health"]
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        for url in probes:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        return
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"server did not become ready within {timeout_s}s; last error: {last_error}")


def choose_prompt_files(root: Path) -> list[Path]:
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


def build_prompt(root: Path, ctx: int, base_ctx: int, base_char_budget: int, min_char_budget: int) -> tuple[str, int, int]:
    files = choose_prompt_files(root)
    scaled_budget = max(min_char_budget, int(base_char_budget * (ctx / float(base_ctx))))

    chunks: list[str] = []
    char_count = 0
    file_count = 0

    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue

        if not text.strip():
            continue

        rel = path.relative_to(root).as_posix()
        block = f"\n\n### FILE: {rel}\n{text}"

        if char_count + len(block) > scaled_budget:
            remaining = scaled_budget - char_count
            if remaining > 4096:
                block = block[:remaining]
                chunks.append(block)
                char_count += len(block)
                file_count += 1
            break

        chunks.append(block)
        char_count += len(block)
        file_count += 1

    prompt = (
        "Ниже большой срез текущего репозитория llama.cpp-rdna-lab. "
        "Проанализируй структуру, long-context/performance риски и предложи 2 практичных шага оптимизации. "
        "Дай компактный ответ по пунктам.\n"
        "===== REPO SNAPSHOT BEGIN ====="
        + "".join(chunks)
        + "\n===== REPO SNAPSHOT END =====\n"
    )

    return prompt, char_count, file_count


def parse_server_timings(server_log_path: Path) -> dict[str, float]:
    text = server_log_path.read_text(encoding="utf-8", errors="ignore")
    prompt_match = re.search(r"prompt eval time =\s+[0-9.]+ ms /\s+\d+ tokens \([^)]*,\s+([0-9.]+) tokens per second\)", text)
    eval_match = re.search(r"\n\s*eval time =\s+[0-9.]+ ms /\s+\d+ tokens \([^)]*,\s+([0-9.]+) tokens per second\)", text)
    total_match = re.search(r"total time =\s+([0-9.]+) ms /\s+\d+ tokens", text)

    return {
        "prompt_eval_tps": round(float(prompt_match.group(1)), 2) if prompt_match else 0.0,
        "decode_eval_tps": round(float(eval_match.group(1)), 2) if eval_match else 0.0,
        "server_total_ms": round(float(total_match.group(1)), 2) if total_match else 0.0,
    }


def load_baseline_result(compare_to: str) -> dict[str, Any] | None:
    if not compare_to:
        return None

    path = Path(compare_to)
    if not path.exists():
        raise FileNotFoundError(f"baseline result file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    baseline: dict[str, Any] = dict(data)

    server_log_name = str(data.get("server_log") or "")
    if server_log_name:
        server_log_path = path.parent / server_log_name
        if server_log_path.exists():
            baseline.update(parse_server_timings(server_log_path))

    return baseline


def compare_against_baseline(current: dict[str, Any], baseline: dict[str, Any], threshold_pct: float) -> dict[str, Any]:
    metrics: list[tuple[str, str]] = [
        ("completion_tps_wall", "wall_tps_delta_pct"),
        ("prompt_eval_tps", "prompt_eval_delta_pct"),
        ("decode_eval_tps", "decode_eval_delta_pct"),
    ]
    result: dict[str, Any] = {
        "baseline_label": baseline.get("label", "baseline"),
        "threshold_pct": threshold_pct,
    }

    prompt_tokens_baseline = float(baseline.get("prompt_tokens") or 0.0)
    prompt_tokens_current = float(current.get("prompt_tokens") or 0.0)
    completion_tokens_baseline = float(baseline.get("completion_tokens") or 0.0)
    completion_tokens_current = float(current.get("completion_tokens") or 0.0)

    prompt_shape_ratio = (prompt_tokens_current / prompt_tokens_baseline) if prompt_tokens_baseline > 0 else 0.0
    completion_shape_ratio = (
        completion_tokens_current / completion_tokens_baseline if completion_tokens_baseline > 0 else 0.0
    )
    result["prompt_shape_ratio_pct"] = round(prompt_shape_ratio * 100.0, 2) if prompt_shape_ratio > 0 else 0.0
    result["completion_shape_ratio_pct"] = (
        round(completion_shape_ratio * 100.0, 2) if completion_shape_ratio > 0 else 0.0
    )

    if not (0.90 <= prompt_shape_ratio <= 1.10 and 0.90 <= completion_shape_ratio <= 1.10):
        result["verdict"] = "incompatible-baseline"
        result["note"] = "compare-to should use the same prompt/completion shape, e.g. quick-to-quick or full-to-full"
        return result

    positives = 0
    negatives = 0
    for key, out_key in metrics:
        base_value = float(baseline.get(key) or 0.0)
        cur_value = float(current.get(key) or 0.0)
        if base_value <= 0:
            continue
        delta_pct = ((cur_value / base_value) - 1.0) * 100.0
        result[out_key] = round(delta_pct, 2)
        if delta_pct >= threshold_pct:
            positives += 1
        elif delta_pct <= -threshold_pct:
            negatives += 1

    if negatives >= 2 or float(result.get("wall_tps_delta_pct", 0.0)) <= -threshold_pct:
        result["verdict"] = "regression"
    elif positives >= 2 or float(result.get("wall_tps_delta_pct", 0.0)) >= threshold_pct:
        result["verdict"] = "promising"
    else:
        result["verdict"] = "inconclusive"

    return result


def run_chat(base_url: str, prompt: str, max_tokens: int, temperature: float, top_p: float, timeout_s: float) -> dict[str, Any]:
    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": "Ты опытный инженер по inference/runtime и GUI."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1200]}") from exc

    elapsed = time.time() - start
    obj = json.loads(raw)
    usage = obj.get("usage", {}) if isinstance(obj, dict) else {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tps_wall = (completion_tokens / elapsed) if completion_tokens and elapsed > 0 else 0.0

    return {
        "elapsed_s": round(elapsed, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completion_tps_wall": round(completion_tps_wall, 4),
        "response_preview": obj.get("choices", [{}])[0].get("message", {}).get("content", "")[:1000],
    }


def start_server(args: argparse.Namespace, ctx: int, port: int, server_log_path: Path) -> subprocess.Popen[str]:
    cmd = [
        str(Path(args.server_bin)),
        "-m",
        str(Path(args.model)),
        "--host",
        args.host,
        "--port",
        str(port),
        "-c",
        str(ctx),
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.ubatch_size),
        "-ngl",
        str(args.gpu_layers),
        "-np",
        str(args.parallel),
        "-t",
        str(args.threads),
        "--threads-http",
        str(args.threads_http),
        "--cache-type-k",
        args.cache_type_k,
        "--cache-type-v",
        args.cache_type_v,
        "--flash-attn",
        "on",
        "--seed",
        "42",
        "--spec-type",
        args.spec_type,
    ]
    if args.server_extra:
        cmd.extend(shlex.split(args.server_extra))
    if args.no_warmup:
        cmd.append("--no-warmup")

    handle = server_log_path.open("w", encoding="utf-8", newline="\n")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    setattr(proc, "_log_handle", handle)
    return proc


def stop_server(proc: subprocess.Popen[str]) -> None:
    log_handle = getattr(proc, "_log_handle", None)
    try:
        if proc.poll() is None:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=float(os.environ.get("LLAMA_BENCH_SOFT_STOP_TIMEOUT", "180")))
            except subprocess.TimeoutExpired:
                if os.environ.get("LLAMA_BENCH_ALLOW_HARD_KILL", "").strip() in ("1", "true", "TRUE", "yes", "on"):
                    proc.kill()
                    proc.wait(timeout=10)
                else:
                    print(
                        f"WARNING: llama-server pid={proc.pid} did not exit after soft stop; "
                        "leaving it alive to avoid hard ROCm teardown."
                    )
                    return
    finally:
        if log_handle is not None and proc.poll() is not None:
            log_handle.close()


def write_summary(out_dir: Path, label_prefix: str, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = out_dir / f"{label_prefix}-repo-summary.csv"
    md_path = out_dir / f"{label_prefix}-repo-summary.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "context",
            "label",
            "prompt_chars",
            "selected_files",
            "prompt_tokens",
            "completion_tokens",
            "elapsed_s",
            "completion_tps_wall",
            "prompt_eval_tps",
            "decode_eval_tps",
            "comparison_verdict",
            "comparison_wall_tps_delta_pct",
            "comparison_prompt_eval_delta_pct",
            "comparison_decode_eval_delta_pct",
            "result_json",
            "server_log",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    lines = [
        "# Repo Snapshot Context Benchmark Summary",
        "",
        f"Label prefix: {label_prefix}",
        "",
        "| Context | Label | Prompt Chars | Files | Prompt Tokens | Completion Tokens | Elapsed (s) | Wall TPS | Prompt Eval TPS | Decode Eval TPS | Result | Server Log |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['context']} | {row['label']} | {row['prompt_chars']} | {row['selected_files']} | "
            f"{row['prompt_tokens']} | {row['completion_tokens']} | {row['elapsed_s']:.4f} | {row['completion_tps_wall']:.4f} | "
            f"{row['prompt_eval_tps']:.2f} | {row['decode_eval_tps']:.2f} | "
            f"{row['result_json']} | {row['server_log']} |"
        )

        comparison = row.get("comparison")
        if comparison:
            lines.append(
                f"  Verdict vs {comparison['baseline_label']}: {comparison['verdict']} "
                f"(wall {comparison.get('wall_tps_delta_pct', 0.0):+.2f}%, "
                f"prompt {comparison.get('prompt_eval_delta_pct', 0.0):+.2f}%, "
                f"decode {comparison.get('decode_eval_delta_pct', 0.0):+.2f}%)"
            )
            if comparison.get("note"):
                lines.append(f"  Note: {comparison['note']}")

    if len(rows) >= 2:
        base = rows[0]
        compare = rows[1]
        base_tps = float(base["completion_tps_wall"])
        cmp_tps = float(compare["completion_tps_wall"])
        delta = cmp_tps - base_tps
        ratio = (cmp_tps / base_tps * 100.0) if base_tps > 0 else 0.0
        lines.append("")
        lines.append(
            f"Comparison: {compare['context']} vs {base['context']} -> delta {delta:+.4f} TPS, ratio {ratio:.2f}%"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    args = parse_args()
    apply_quick_profile(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    server_bin = Path(args.server_bin)
    model_path = Path(args.model)
    if not server_bin.exists():
        print(f"ERROR: server binary not found: {server_bin}")
        return 2
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        return 2

    try:
        ctx_values = parse_ctx_values(args.ctx_values)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    if not args.allow_ctx_above_16k:
        over_limit = [ctx for ctx in ctx_values if ctx > PRIMARY_MAX_CTX]
        if over_limit:
            print(
                f"ERROR: ctx-values above {PRIMARY_MAX_CTX} are disabled by current 130k benchmark policy. "
                "Use --allow-ctx-above-16k only for explicit over-130k exploratory runs."
            )
            return 2

    try:
        baseline = load_baseline_result(args.compare_to)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2

    rows: list[dict[str, Any]] = []
    for idx, ctx in enumerate(ctx_values):
        label = f"{args.label_prefix}-ctx{ctx // 1024}k"
        port = args.port if args.port > 0 else (58000 + idx * 37)
        server_log = out_dir / f"{label}.server.log"
        result_json = out_dir / f"{label}.result.json"

        prompt, prompt_chars, selected_files = build_prompt(
            ROOT,
            ctx=ctx,
            base_ctx=args.base_ctx,
            base_char_budget=args.base_char_budget,
            min_char_budget=args.min_char_budget,
        )

        print(
            f"Running ctx={ctx} label={label} prompt_chars={prompt_chars} files={selected_files} "
            f"quick_profile={args.quick_profile} max_tokens={args.max_tokens}"
        )
        proc = start_server(args, ctx=ctx, port=port, server_log_path=server_log)
        base_url = f"http://{args.host}:{port}"

        try:
            wait_ready(base_url, timeout_s=args.startup_timeout)
            result = run_chat(
                base_url,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout_s=args.request_timeout,
            )
        finally:
            stop_server(proc)

        timings = parse_server_timings(server_log)

        output = {
            "label": label,
            "context": f"ctx{ctx // 1024}k",
            "prompt_chars": prompt_chars,
            "selected_files": selected_files,
            **result,
            **timings,
            "server_log": server_log.name,
        }

        if baseline is not None:
            output["comparison"] = compare_against_baseline(output, baseline, args.compare_threshold_pct)

        result_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        rows.append(
            {
                "context": output["context"],
                "label": label,
                "prompt_chars": output["prompt_chars"],
                "selected_files": output["selected_files"],
                "prompt_tokens": output["prompt_tokens"],
                "completion_tokens": output["completion_tokens"],
                "elapsed_s": output["elapsed_s"],
                "completion_tps_wall": output["completion_tps_wall"],
                "prompt_eval_tps": output["prompt_eval_tps"],
                "decode_eval_tps": output["decode_eval_tps"],
                "comparison_verdict": (output.get("comparison") or {}).get("verdict", ""),
                "comparison_wall_tps_delta_pct": (output.get("comparison") or {}).get("wall_tps_delta_pct", ""),
                "comparison_prompt_eval_delta_pct": (output.get("comparison") or {}).get("prompt_eval_delta_pct", ""),
                "comparison_decode_eval_delta_pct": (output.get("comparison") or {}).get("decode_eval_delta_pct", ""),
                "result_json": result_json.name,
                "server_log": output["server_log"],
                "comparison": output.get("comparison"),
            }
        )

        if output.get("comparison"):
            comparison = output["comparison"]
            print(
                f"Quick verdict vs {comparison['baseline_label']}: {comparison['verdict']} "
                f"(wall {comparison.get('wall_tps_delta_pct', 0.0):+.2f}%, "
                f"prompt {comparison.get('prompt_eval_delta_pct', 0.0):+.2f}%, "
                f"decode {comparison.get('decode_eval_delta_pct', 0.0):+.2f}%)"
            )
            note = comparison.get("note")
            if note:
                print(f"Compare note: {note}")

    summary_csv, summary_md = write_summary(out_dir, args.label_prefix, rows)
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
