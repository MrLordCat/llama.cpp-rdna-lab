#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "build_logs" / "agent-workload"
DEFAULT_SPIRV = [
    ROOT / "build-vulkan/ggml/src/ggml-vulkan/vulkan-shaders.spv/matmul_q3_k_f32_aligned_f16acc_cm1.spv",
    ROOT / "build-vulkan/ggml/src/ggml-vulkan/vulkan-shaders.spv/flash_attn_f32_f16_q4_0_f16acc_cm1.spv",
]

NUM = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
MAT_RE = re.compile(
    rf"^(?P<op>MUL_MAT_ADD MUL_MAT_VEC|MUL_MAT_VEC|MUL_MAT)\s+"
    rf"(?P<typ>\S+)\s+m=(?P<m>\d+)\s+n=(?P<n>\d+)\s+k=(?P<k>\d+):\s+"
    rf"(?P<count>\d+)\s+x\s+(?P<avg>{NUM})\s+us\s+=\s+(?P<total>{NUM})\s+us"
)
FA_RE = re.compile(
    rf"^FLASH_ATTN_EXT\s+dst\([^)]*\),\s+q\(256,(?P<n>\d+),24,1\),\s+"
    rf"k\(256,(?P<kv>\d+),4,1\),\s+v\(256,(?P=kv),4,1\),.*:\s+"
    rf"(?P<count>\d+)\s+x\s+(?P<avg>{NUM})\s+us\s+=\s+(?P<total>{NUM})\s+us"
)
GLU_RE = re.compile(rf"^GLU:\s+(?P<count>\d+)\s+x\s+(?P<avg>{NUM})\s+us\s+=\s+(?P<total>{NUM})\s+us")
OP_RE = re.compile(r"\b(Op[A-Za-z0-9_]+)\b")
FOCUS_OPS = (
    "OpCooperativeMatrixLoadKHR",
    "OpCooperativeMatrixMulAddKHR",
    "OpCooperativeMatrixStoreKHR",
    "OpControlBarrier",
    "OpLoad",
    "OpStore",
    "OpFConvert",
    "OpFMul",
    "OpIAdd",
    "OpIMul",
    "OpShiftRightLogical",
    "OpShiftLeftLogical",
    "OpBitwiseAnd",
    "OpBitwiseOr",
    "OpBitwiseXor",
)


@dataclass(frozen=True)
class PerfRow:
    bucket: str
    shape: str
    calls: int
    total_us: float


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def stats_mean(diag: dict[str, Any], key: str) -> str:
    value = diag.get("server_log_diagnostics", {}).get(key, {}).get("mean")
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def stats_max(diag: dict[str, Any], key: str) -> str:
    value = diag.get("server_log_diagnostics", {}).get(key, {}).get("max")
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def parse_float_csv(raw: str) -> list[float]:
    values: list[float] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if text:
            values.append(float(text))
    return values


def local_route_speedup(route_share: float, local_speedup: float) -> float:
    return 1.0 / ((1.0 - route_share) + route_share / local_speedup)


def parse_perf_rows(text: str) -> list[PerfRow]:
    rows: list[PerfRow] = []
    for line in text.splitlines():
        if match := MAT_RE.search(line):
            bucket = f"{match.group('op')} {match.group('typ')}"
            shape = f"m={match.group('m')} n={match.group('n')} k={match.group('k')}"
            rows.append(PerfRow(bucket, shape, int(match.group("count")), float(match.group("total"))))
            continue
        if match := FA_RE.search(line):
            rows.append(PerfRow("FLASH_ATTN_EXT", f"N={match.group('n')} KV={match.group('kv')}", int(match.group("count")), float(match.group("total"))))
            continue
        if match := GLU_RE.search(line):
            rows.append(PerfRow("GLU", "all", int(match.group("count")), float(match.group("total"))))
    return rows


def aggregate_perf(rows: list[PerfRow], by_shape: bool) -> list[tuple[str, int, float]]:
    calls: collections.defaultdict[str, int] = collections.defaultdict(int)
    total: collections.defaultdict[str, float] = collections.defaultdict(float)
    for row in rows:
        key = f"{row.bucket} {row.shape}" if by_shape else row.bucket
        calls[key] += row.calls
        total[key] += row.total_us
    return sorted(((key, calls[key], total[key]) for key in total), key=lambda item: item[2], reverse=True)


def parse_key_values(raw: str) -> dict[str, str]:
    parts = raw.split("|")
    values = {"pipeline": parts[0].strip()}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def parse_route_traces(text: str) -> dict[str, list[dict[str, str]]]:
    traces: dict[str, list[dict[str, str]]] = {"matmul": [], "flash_attn": [], "ffn": []}
    for line in text.splitlines():
        stripped = line.strip()
        if "ggml_vulkan: matmul route:" in stripped:
            traces["matmul"].append(parse_key_values(stripped.split("ggml_vulkan: matmul route:", 1)[1].strip()))
        elif "ggml_vulkan: flash_attn route:" in stripped:
            traces["flash_attn"].append(parse_key_values(stripped.split("ggml_vulkan: flash_attn route:", 1)[1].strip()))
        elif "ggml_vulkan: ffn_route_trace" in stripped:
            traces["ffn"].append({"line": stripped})
    return traces


def collect_server_signals(text: str) -> dict[str, list[str]]:
    signals: dict[str, list[str]] = {
        "memory": [],
        "context": [],
        "buffers": [],
        "pipeline_stats": [],
        "warnings": [],
    }
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "common_memory_breakdown_print:" in line or "projected to use" in line or "will leave" in line:
            signals["memory"].append(line)
        elif line.startswith("llama_context:"):
            if any(key in line for key in ("n_ctx", "n_batch", "n_ubatch", "flash_attn", "kv_unified")):
                signals["context"].append(line)
        elif "buffer size" in line or "graph nodes" in line or "graph splits" in line or "PP reserve outputs" in line:
            if line.startswith(("load_tensors:", "llama_kv_cache:", "llama_memory_recurrent:", "sched_reserve:")):
                signals["buffers"].append(line)
        elif "pipeline stats for" in line or "VGPR" in line or "SGPR" in line or "scratch" in line:
            signals["pipeline_stats"].append(line)
        elif "warning" in line.lower() or "error" in line.lower() or "fallback" in line.lower():
            signals["warnings"].append(line)
    return signals


def run_spirv_dis(spirv_dis: str, path: Path) -> collections.Counter[str] | None:
    if not path.exists():
        return None
    completed = subprocess.run(
        [spirv_dis, str(path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if completed.returncode != 0:
        return None
    counts: collections.Counter[str] = collections.Counter()
    for line in completed.stdout.splitlines():
        if match := OP_RE.search(line):
            counts[match.group(1)] += 1
    return counts


def append_table(lines: list[str], headers: list[str], rows: list[list[str]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(" --- " for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def format_lines(items: list[str], limit: int = 24) -> list[str]:
    if not items:
        return ["- missing"]
    out = [f"- `{item}`" for item in items[:limit]]
    if len(items) > limit:
        out.append(f"- ... {len(items) - limit} more")
    return out


def append_ceiling_table(
    lines: list[str],
    title: str,
    rows: list[tuple[str, int, float]],
    parsed_total_us: float,
    local_speedups: list[float],
    baseline_tps: float | None,
    limit: int,
) -> None:
    if not rows or parsed_total_us <= 0.0:
        return
    header = ["Route", "Calls", "Total ms", "Parsed share"] + [f"{value:.2f}x local" for value in local_speedups]
    table_rows: list[list[str]] = []
    for key, calls, total_us in rows[:limit]:
        share = total_us / parsed_total_us
        out_row = [f"`{key}`", str(calls), f"{total_us / 1000.0:.2f}", f"{share * 100.0:.2f}%"]
        for local in local_speedups:
            speedup = local_route_speedup(share, local)
            if baseline_tps is None:
                out_row.append(f"{speedup:.4f}x")
            else:
                out_row.append(f"{speedup:.4f}x / {baseline_tps * speedup:.4f} TPS")
        table_rows.append(out_row)
    lines += [f"### {title}", ""]
    append_table(lines, header, table_rows)


def build_report(args: argparse.Namespace, diag: dict[str, Any], server_text: str, perf_text: str, extra_text: str, spirv_files: list[Path]) -> str:
    label = args.label or diag.get("label") or "manual"
    config = diag.get("config", {})
    run_metrics = diag.get("run_metrics", {})
    combined_text = server_text + "\n" + perf_text + "\n" + extra_text
    server_signals = collect_server_signals(combined_text)
    route_traces = parse_route_traces(combined_text)
    perf_rows = parse_perf_rows(combined_text)
    parsed_total_us = sum(row.total_us for row in perf_rows)
    ffn_counts = collections.Counter(item["line"] for item in route_traces["ffn"])
    local_speedups = parse_float_csv(args.ceiling_speedups)

    lines: list[str] = [
        f"# Vulkan Evidence Pack: {label}",
        "",
        "## Baseline Metrics",
        "",
    ]
    append_table(
        lines,
        ["Metric", "Value"],
        [
            ["aggregate_completion_tps", f"{run_metrics.get('aggregate_completion_tps', '-')}"] ,
            ["errors", f"{run_metrics.get('errors', '-')}"] ,
            ["prompt_eval_tps mean", stats_mean(diag, "prompt_eval_tps")],
            ["decode_eval_tps mean", stats_mean(diag, "decode_eval_tps")],
            ["prompt_eval_ms mean", stats_mean(diag, "prompt_eval_ms")],
            ["decode_eval_ms mean", stats_mean(diag, "decode_eval_ms")],
            ["total_ms mean", stats_mean(diag, "total_ms")],
            ["task_prompt_tokens mean", stats_mean(diag, "task_prompt_tokens")],
            ["batch_chunks mean/max", f"{stats_mean(diag, 'batch_chunks')}/{stats_max(diag, 'batch_chunks')}"],
        ],
    )

    lines += ["## Lane Contract", ""]
    append_table(
        lines,
        ["Field", "Value"],
        [
            ["ctx", str(config.get("ctx", "-"))],
            ["batch/ubatch", f"{config.get('batch', '-')}/{config.get('ubatch', '-')}"] ,
            ["KV", f"{config.get('cache_type_k', '-')}/{config.get('cache_type_v', '-')}"] ,
            ["flash_attn", str(config.get("flash_attn", "-"))],
            ["spec_mode", str(config.get("spec_mode", "-"))],
            ["no_reuse", str(config.get("no_reuse", "-"))],
            ["tasks/runs", f"{config.get('tasks', '-')}/{config.get('runs', '-')}"] ,
            ["server_extra", str(config.get("server_extra", "-"))],
        ],
    )

    lines += ["## Residency And Startup Signals", ""]
    lines += ["### Memory Fit", ""] + format_lines(server_signals["memory"]) + [""]
    lines += ["### Context", ""] + format_lines(server_signals["context"]) + [""]
    lines += ["### Buffers", ""] + format_lines(server_signals["buffers"], limit=32) + [""]

    lines += ["## Vulkan Route Trace", ""]
    append_table(
        lines,
        ["Trace", "Unique rows"],
        [
            ["matmul", str(len(route_traces["matmul"]))],
            ["flash_attn", str(len(route_traces["flash_attn"]))],
                ["ffn", str(len(ffn_counts))],
        ],
    )
    for name in ("matmul", "flash_attn"):
        if route_traces[name]:
            rows = []
            for item in route_traces[name][: args.top]:
                if name == "matmul":
                    rows.append([
                        item.get("pipeline", "-"),
                        item.get("src0", "-"),
                        item.get("src1", "-"),
                        f"m={item.get('m', '-')} n={item.get('n', '-')} k={item.get('k', '-')}",
                    ])
                else:
                    rows.append([
                        item.get("pipeline", "-"),
                        item.get("k", "-"),
                        item.get("v", "-"),
                        f"N={item.get('N', '-')} KV={item.get('KV', '-')} Br={item.get('Br', '-')} Bc={item.get('Bc', '-')} split_k={item.get('split_k', '-')}",
                    ])
            lines += [f"### {name}", ""]
            append_table(lines, ["Pipeline", "A/K", "B/V", "Shape"], rows)
    if ffn_counts:
        lines += ["### ffn", ""]
        rows = [[f"`{line}`", str(count)] for line, count in ffn_counts.most_common(args.top)]
        append_table(lines, ["Line", "Occurrences"], rows)

    lines += ["## Vulkan Perf Timing", ""]
    if perf_rows:
        lines.append(f"- parsed_rows: `{len(perf_rows)}`")
        lines.append(f"- parsed_total_ms: `{parsed_total_us / 1000.0:.2f}`")
        lines.append("")
        for title, by_shape in (("By bucket", False), ("By shape", True)):
            rows = []
            for key, calls, total_us in aggregate_perf(perf_rows, by_shape=by_shape)[: args.top]:
                share = 100.0 * total_us / parsed_total_us if parsed_total_us else 0.0
                rows.append([f"`{key}`", str(calls), f"{total_us / 1000.0:.2f}", f"{share:.2f}%"])
            lines += [f"### {title}", ""]
            append_table(lines, ["Key", "Calls", "Total ms", "Parsed share"], rows)
    else:
        lines += [
            "- missing perf rows. Re-run a diagnostic pass with `--trace-preset vulkan-perf` to populate Q3_K/FA timing buckets.",
            "",
        ]

    if perf_rows:
        lines += [
            "## Ceiling Sketch",
            "",
            "Diagnostic-only Amdahl estimate from parsed Vulkan perf shares; use paired cold A/B before claiming speed.",
        ]
        if args.baseline_tps is not None:
            lines.append(f"- baseline_tps_for_projection: `{args.baseline_tps:.4f}`")
        lines.append("")
        append_ceiling_table(
            lines,
            "Top buckets",
            aggregate_perf(perf_rows, by_shape=False),
            parsed_total_us,
            local_speedups,
            args.baseline_tps,
            args.ceiling_top,
        )
        append_ceiling_table(
            lines,
            "Top shapes",
            aggregate_perf(perf_rows, by_shape=True),
            parsed_total_us,
            local_speedups,
            args.baseline_tps,
            args.ceiling_top,
        )

    lines += ["## Pipeline Resource Signals", ""]
    lines += format_lines(server_signals["pipeline_stats"], limit=32) + [""]

    lines += ["## SPIR-V Fingerprints", ""]
    spirv_dis = shutil.which("spirv-dis")
    if spirv_dis is None:
        lines += ["- missing: `spirv-dis` not found on PATH", ""]
    else:
        for path in spirv_files:
            counts = run_spirv_dis(spirv_dis, path)
            lines += [f"### {repo_rel(path)}", ""]
            if counts is None:
                lines += ["- missing or failed to disassemble", ""]
                continue
            rows = [[op, str(counts[op])] for op in FOCUS_OPS if counts.get(op, 0)]
            append_table(lines, ["Focus op", "Count"], rows or [["-", "0"]])

    missing: list[str] = []
    if not route_traces["matmul"]:
        missing.append("matmul route trace (`--trace-preset vulkan-routes` or `vulkan-perf`)")
    if not route_traces["flash_attn"]:
        missing.append("FlashAttention route trace (`--trace-preset vulkan-routes` or `vulkan-perf`)")
    if not perf_rows:
        missing.append("Vulkan perf timing rows (`--trace-preset vulkan-perf`)")
    if not server_signals["pipeline_stats"]:
        missing.append("pipeline resource stats (`GGML_VK_PIPELINE_STATS=matmul_q3_k` or a focused FA filter)")

    lines += ["## Rewrite Gate", ""]
    if missing:
        lines.append("Do not start a large Vulkan route rewrite from this pack alone; missing evidence:")
        lines += [f"- {item}" for item in missing]
    else:
        lines.append("Evidence pack is complete enough to rank a design before source edits.")
    lines += [
        "",
        "Recommended next gate order:",
        "- route share and hot shape table",
        "- local ceiling model for the touched route",
        "- SPIR-V/resource delta before server A/B",
        "- correctness scout for prompt and decode paths",
        "- paired cold r1 A/B on the E265 lane",
        "",
    ]

    lines += ["## Source Artifacts", ""]
    for path in (args.diagnostics, args.server_log, args.perf_log):
        if path is not None:
            lines.append(f"- `{repo_rel(path)}`")
    for path in args.extra_log:
        lines.append(f"- `{repo_rel(path)}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact Vulkan route/residency evidence pack from benchmark artifacts")
    parser.add_argument("--label", default="scout-vulkan130k-quick-c24k-b512-ub128-r1")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--diagnostics", type=Path, default=None)
    parser.add_argument("--server-log", type=Path, default=None)
    parser.add_argument("--perf-log", type=Path, default=None, help="optional additional perf/trace log; defaults to server log only")
    parser.add_argument("--extra-log", type=Path, nargs="*", default=[], help="additional server/perf logs to merge into the pack")
    parser.add_argument("--spirv", type=Path, nargs="*", default=DEFAULT_SPIRV)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top", type=int, default=16)
    parser.add_argument("--baseline-tps", type=float, default=None, help="cold baseline TPS used only for ceiling projections")
    parser.add_argument("--ceiling-speedups", default="1.10,1.20,1.35,1.50,2.00", help="comma-separated local route speedups for Amdahl projections")
    parser.add_argument("--ceiling-top", type=int, default=8, help="number of top buckets/shapes to include in the ceiling sketch")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.diagnostics is None:
        args.diagnostics = args.out_dir / f"{args.label}.diagnostics.json"
    if args.server_log is None:
        args.server_log = args.out_dir / f"{args.label}.server.log"
    if args.out is None:
        args.out = args.out_dir / f"{args.label}.vulkan-evidence.md"

    diag = load_json(args.diagnostics)
    server_text = read_text(args.server_log)
    perf_text = read_text(args.perf_log)
    extra_text = "\n".join(read_text(path) for path in args.extra_log)
    report = build_report(args, diag, server_text, perf_text, extra_text, list(args.spirv))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())