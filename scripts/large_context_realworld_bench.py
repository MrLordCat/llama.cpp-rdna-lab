#!/usr/bin/env python3
"""Run a realistic two-point large-context benchmark (32K + 64K).

This script wraps scripts/agent_workload_bench.py and executes two scenarios
with matching settings so context length is the only intentional variable.
It writes a compact summary CSV/Markdown for quick before/after comparisons.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "build_logs" / "agent-workload"
DEFAULT_RUNNER = ROOT / "scripts" / "agent_workload_bench.py"
DEFAULT_CTX_FAST64 = "32768,65536"
DEFAULT_CTX_SENTINEL128 = "65536,131072"
NGRAM_MOD_N_MIN = 12
NGRAM_MOD_N_MATCH = 16
NGRAM_MOD_N_MAX = 32


def parse_args() -> argparse.Namespace:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(
        description="Large-context realistic benchmark runner (32K and 64K for faster agentic sessions)",
    )
    parser.add_argument("--label-prefix", default=f"largectx-real-{stamp}")
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER), help="path to agent_workload_bench.py")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--profile",
        choices=["fast64", "sentinel128", "custom"],
        default="fast64",
        help="fast64=32K/64K quick loop, sentinel128=64K/128K periodic check, custom=use --ctx-values",
    )

    parser.add_argument("--tasks", choices=["v2", "v2-mini"], default="v2")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--startup-timeout", type=float, default=300.0)

    parser.add_argument(
        "--ctx-values",
        default="",
        help="optional comma-separated context sizes override; when omitted, selected from --profile",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--cache-type-k", default="q4_0")
    parser.add_argument("--cache-type-v", default="q4_0")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--flash-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to keep llama-server warmup enabled (default: disabled)",
    )
    parser.add_argument("--server-seed", type=int, default=42)
    parser.add_argument("--stats-ignore-first-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--spec-profile",
        choices=["none", "ngram-mod", "custom"],
        default="none",
        help="none=disable speculative; ngram-mod=repo default tuned profile; custom=use --server-extra as-is",
    )
    parser.add_argument("--server-extra", default="", help="extra llama-server args")

    parser.add_argument("--server-bin", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--build-id", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--api-model", default="local-model")
    parser.add_argument("--history-version", default="v2")
    parser.add_argument(
        "--background-server-policy",
        choices=["warn", "fail", "ignore"],
        default="fail",
    )
    return parser.parse_args()


def parse_ctx_values(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) < 2:
        raise ValueError("--ctx-values must contain at least two values, for example 32768,65536")
    if len(set(values)) != len(values):
        raise ValueError("--ctx-values must not contain duplicates")
    for value in values:
        if value < 32768:
            raise ValueError("all --ctx-values must be >= 32768 for the fast large-context comparison")
    return values


def resolve_ctx_values(args: argparse.Namespace) -> str:
    if args.ctx_values.strip():
        return args.ctx_values.strip()
    if args.profile == "sentinel128":
        return DEFAULT_CTX_SENTINEL128
    return DEFAULT_CTX_FAST64


def build_server_extra(spec_profile: str, server_extra: str) -> str:
    custom = server_extra.strip()
    if spec_profile == "custom":
        return custom
    if spec_profile == "none":
        parts = [custom] if custom else []
        parts.append("--spec-type none")
        return " ".join(parts).strip()

    # ngram-mod profile tuned for this repository workflow.
    parts = [custom] if custom else []
    parts.extend(
        [
            "--spec-type ngram-mod",
            f"--spec-ngram-mod-n-min {NGRAM_MOD_N_MIN}",
            f"--spec-ngram-mod-n-match {NGRAM_MOD_N_MATCH}",
            f"--spec-ngram-mod-n-max {NGRAM_MOD_N_MAX}",
        ]
    )
    return " ".join(parts).strip()


def run_single_case(args: argparse.Namespace, runner: Path, ctx_size: int, label: str, server_extra: str) -> None:
    cmd = [
        sys.executable,
        str(runner),
        "--label",
        label,
        "--out-dir",
        args.out_dir,
        "--tasks",
        args.tasks,
        "--runs",
        str(args.runs),
        "--ctx-size",
        str(ctx_size),
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--cache-type-k",
        args.cache_type_k,
        "--cache-type-v",
        args.cache_type_v,
        "--parallel",
        str(args.parallel),
        "--gpu-layers",
        str(args.gpu_layers),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--api-model",
        args.api_model,
        "--history-version",
        args.history_version,
        "--background-server-policy",
        args.background_server_policy,
        "--request-timeout",
        str(args.request_timeout),
        "--startup-timeout",
        str(args.startup_timeout),
        "--max-tokens",
        str(args.max_tokens),
        "--server-seed",
        str(args.server_seed),
    ]

    if args.server_bin:
        cmd.extend(["--server-bin", args.server_bin])
    if args.model:
        cmd.extend(["--model", args.model])
    if args.build_id:
        cmd.extend(["--build-id", args.build_id])
    if server_extra:
        cmd.extend(["--server-extra", server_extra])

    cmd.append("--flash-attn" if args.flash_attn else "--no-flash-attn")
    cmd.append("--warmup" if args.warmup else "--no-warmup")
    cmd.append("--stats-ignore-first-run" if args.stats_ignore_first_run else "--no-stats-ignore-first-run")
    cmd.append("--disable-thinking" if args.disable_thinking else "--no-disable-thinking")

    print(f"Running ctx={ctx_size} label={label}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def read_case_metrics(out_dir: Path, label: str) -> dict[str, Any]:
    csv_path = out_dir / f"{label}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected {csv_path.name} not found. Use default artifact mode (full) in wrapped runner."
        )

    completion_sum = 0
    wall_sum = 0.0
    tps_values: list[float] = []
    errors = 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("error"):
                errors += 1
            completion = row.get("completion_tokens")
            wall = row.get("wall_s")
            tps = row.get("completion_tps_wall")

            if completion:
                completion_sum += int(float(completion))
            if wall:
                wall_sum += float(wall)
            if tps:
                tps_values.append(float(tps))

    aggregate_tps = (completion_sum / wall_sum) if completion_sum > 0 and wall_sum > 0 else 0.0
    mean_task_tps = (sum(tps_values) / len(tps_values)) if tps_values else 0.0
    return {
        "label": label,
        "csv_file": csv_path.name,
        "completion_tokens": completion_sum,
        "wall_s": round(wall_sum, 4),
        "aggregate_tps": round(aggregate_tps, 4),
        "mean_task_tps": round(mean_task_tps, 4),
        "errors": errors,
    }


def write_summary(out_dir: Path, label_prefix: str, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = out_dir / f"{label_prefix}-largectx-summary.csv"
    md_path = out_dir / f"{label_prefix}-largectx-summary.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "context",
            "label",
            "aggregate_tps",
            "mean_task_tps",
            "completion_tokens",
            "wall_s",
            "errors",
            "csv_file",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    baseline = rows[0]
    lines = [
        "# Large Context Real-World Benchmark Summary",
        "",
        f"Label prefix: {label_prefix}",
        "",
        "| Context | Label | Aggregate TPS | Mean Task TPS | Completion Tokens | Wall Time (s) | Errors | CSV |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['context']} | {row['label']} | {row['aggregate_tps']:.4f} | {row['mean_task_tps']:.4f} | "
            f"{row['completion_tokens']} | {row['wall_s']:.4f} | {row['errors']} | {row['csv_file']} |"
        )

    lines.append("")
    if len(rows) >= 2:
        compare = rows[1]
        base_tps = float(baseline["aggregate_tps"])
        cmp_tps = float(compare["aggregate_tps"])
        delta = cmp_tps - base_tps
        ratio = (cmp_tps / base_tps * 100.0) if base_tps > 0 else 0.0
        lines.append(
            "Comparison: "
            f"{compare['context']} vs {baseline['context']} -> "
            f"delta {delta:+.4f} TPS, ratio {ratio:.2f}%"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = Path(args.runner)
    if not runner.exists():
        print(f"ERROR: runner not found: {runner}")
        return 2

    try:
        ctx_values = parse_ctx_values(resolve_ctx_values(args))
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    server_extra = build_server_extra(args.spec_profile, args.server_extra)

    scenario_rows: list[dict[str, Any]] = []
    for ctx_size in ctx_values:
        ctx_label = f"ctx{ctx_size // 1024}k"
        label = f"{args.label_prefix}-{ctx_label}"
        run_single_case(args, runner, ctx_size, label, server_extra)
        metrics = read_case_metrics(out_dir, label)
        metrics["context"] = ctx_label
        scenario_rows.append(metrics)

    summary_csv, summary_md = write_summary(out_dir, args.label_prefix, scenario_rows)
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
