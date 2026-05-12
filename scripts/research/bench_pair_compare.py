#!/usr/bin/env python3
"""Compare two benchmark CSV artifacts from build_logs/agent-workload."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Any


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        return int(float(text))
    except Exception:
        return default


def read_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def aggregate_completion_tps(rows: list[dict[str, Any]]) -> float:
    total_completion = sum(_int(r.get("completion_tokens"), 0) for r in rows)
    total_wall = sum(_float(r.get("wall_s"), 0.0) for r in rows)
    if total_wall <= 0.0 or total_completion <= 0:
        return 0.0
    return total_completion / total_wall


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tps_values = [_float(r.get("completion_tps_wall"), 0.0) for r in rows if _float(r.get("completion_tps_wall"), 0.0) > 0.0]
    wall_values = [_float(r.get("wall_s"), 0.0) for r in rows if _float(r.get("wall_s"), 0.0) > 0.0]
    completion_values = [_int(r.get("completion_tokens"), 0) for r in rows if _int(r.get("completion_tokens"), 0) > 0]

    errors = sum(1 for r in rows if str(r.get("error", "")).strip())
    return {
        "task_count": len(rows),
        "errors": errors,
        "total_wall_s": sum(wall_values),
        "total_completion_tokens": sum(completion_values),
        "aggregate_tps": aggregate_completion_tps(rows),
        "mean_task_tps": statistics.mean(tps_values) if tps_values else 0.0,
        "median_task_tps": statistics.median(tps_values) if tps_values else 0.0,
        "task_tps_stdev": statistics.pstdev(tps_values) if len(tps_values) > 1 else 0.0,
    }


def parse_server_log(log_path: Path) -> dict[str, float]:
    text = log_path.read_text(encoding="utf-8", errors="replace")

    prompt_matches = re.findall(
        r"prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens \([^)]*,\s*([0-9.]+) tokens per second\)",
        text,
    )
    decode_matches = re.findall(
        r"(?:^|\n)\s*eval time =\s*([0-9.]+) ms /\s*(\d+) tokens \([^)]*,\s*([0-9.]+) tokens per second\)",
        text,
    )

    prompt_tps = [float(m[2]) for m in prompt_matches]
    decode_tps = [float(m[2]) for m in decode_matches]

    return {
        "prompt_eval_tps_mean": statistics.mean(prompt_tps) if prompt_tps else 0.0,
        "decode_eval_tps_mean": statistics.mean(decode_tps) if decode_tps else 0.0,
    }


def print_summary_block(name: str, stats: dict[str, Any], log_stats: dict[str, float] | None) -> None:
    print(f"[{name}]")
    print(f"tasks={stats['task_count']} errors={stats['errors']}")
    print(f"aggregate_tps={stats['aggregate_tps']:.4f}")
    print(f"mean_task_tps={stats['mean_task_tps']:.4f} median_task_tps={stats['median_task_tps']:.4f} stdev={stats['task_tps_stdev']:.4f}")
    print(f"total_wall_s={stats['total_wall_s']:.4f} total_completion_tokens={stats['total_completion_tokens']}")
    if log_stats is not None:
        print(
            "server_log_means: "
            f"prompt_eval_tps={log_stats['prompt_eval_tps_mean']:.4f}, "
            f"decode_eval_tps={log_stats['decode_eval_tps_mean']:.4f}"
        )
    print("")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two benchmark CSV artifacts")
    parser.add_argument("--baseline-csv", required=True, help="path to baseline CSV")
    parser.add_argument("--candidate-csv", required=True, help="path to candidate CSV")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--baseline-log", default="", help="optional baseline server log")
    parser.add_argument("--candidate-log", default="", help="optional candidate server log")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    baseline_csv = Path(args.baseline_csv)
    candidate_csv = Path(args.candidate_csv)
    if not baseline_csv.exists():
        raise SystemExit(f"ERROR: baseline CSV not found: {baseline_csv}")
    if not candidate_csv.exists():
        raise SystemExit(f"ERROR: candidate CSV not found: {candidate_csv}")

    baseline_rows = read_rows(baseline_csv)
    candidate_rows = read_rows(candidate_csv)

    baseline_stats = summarize(baseline_rows)
    candidate_stats = summarize(candidate_rows)

    baseline_log_stats = None
    candidate_log_stats = None

    if args.baseline_log.strip():
        baseline_log = Path(args.baseline_log)
        if baseline_log.exists():
            baseline_log_stats = parse_server_log(baseline_log)
    if args.candidate_log.strip():
        candidate_log = Path(args.candidate_log)
        if candidate_log.exists():
            candidate_log_stats = parse_server_log(candidate_log)

    print_summary_block(args.baseline_name, baseline_stats, baseline_log_stats)
    print_summary_block(args.candidate_name, candidate_stats, candidate_log_stats)

    base_tps = float(baseline_stats["aggregate_tps"])
    cand_tps = float(candidate_stats["aggregate_tps"])
    speedup = (cand_tps / base_tps) if base_tps > 0 else 0.0
    delta = cand_tps - base_tps

    print("[delta]")
    print(f"aggregate_tps_delta={delta:.4f}")
    print(f"aggregate_tps_speedup={speedup:.4f}x")

    if baseline_log_stats is not None and candidate_log_stats is not None:
        b_pref = baseline_log_stats["prompt_eval_tps_mean"]
        c_pref = candidate_log_stats["prompt_eval_tps_mean"]
        b_dec = baseline_log_stats["decode_eval_tps_mean"]
        c_dec = candidate_log_stats["decode_eval_tps_mean"]

        pref_speedup = (c_pref / b_pref) if b_pref > 0 else 0.0
        dec_speedup = (c_dec / b_dec) if b_dec > 0 else 0.0
        print(f"prompt_eval_speedup={pref_speedup:.4f}x")
        print(f"decode_eval_speedup={dec_speedup:.4f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
