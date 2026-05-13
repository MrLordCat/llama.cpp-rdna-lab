#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


TOTAL_MS_RE = re.compile(r"total_ms=([0-9.]+)")
PRE_SYNC_MS_RE = re.compile(r"pre_sync_ms=([0-9.]+)")
SYNC_RE = re.compile(r"sync_applied=(\d+)")


def field(line: str, name: str, default: str = "?") -> str:
    match = re.search(rf"\b{re.escape(name)}=([^\s]+)", line)
    return match.group(1).rstrip(",") if match else default


def total_ms(line: str) -> float | None:
    match = TOTAL_MS_RE.search(line)
    return float(match.group(1)) if match else None


def pre_sync_ms(line: str) -> float:
    match = PRE_SYNC_MS_RE.search(line)
    return float(match.group(1)) if match else 0.0


def sync_applied(line: str) -> bool:
    match = SYNC_RE.search(line)
    return bool(match and match.group(1) == "1")


def add(groups: dict[str, list[float]], key: str, value: float) -> None:
    groups.setdefault(key, []).append(value)


def parse_log(path: Path, sync_only: bool) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = {}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = total_ms(line)
            if value is None:
                continue
            if sync_only and "sync_applied=" in line and not sync_applied(line):
                continue

            if "GGML_TRACE_CUDA_NODE_TIMING:" in line:
                op = field(line, "op")
                kind = field(line, "kind")
                name = field(line, "name")
                ne = field(line, "ne")
                add(groups, "CUDA_NODE", value)
                add(groups, f"CUDA_NODE op={op}", value)
                add(groups, f"CUDA_NODE op={op} kind={kind}", value)
                add(groups, f"CUDA_NODE op={op} name={name}", value)
                add(groups, f"CUDA_NODE op={op} ne={ne}", value)
                add(groups, f"CUDA_NODE op={op} name={name} ne={ne}", value)
                add(groups, "PRE_SYNC before CUDA_NODE", pre_sync_ms(line))
                add(groups, f"PRE_SYNC before CUDA_NODE op={op}", pre_sync_ms(line))
                continue

            if "ggml_cuda_flash_attn_ext: timing" in line:
                selected = field(line, "selected")
                q1 = field(line, "Q1")
                k1 = field(line, "K1")
                add(groups, f"FATTN selected={selected}", value)
                add(groups, f"PRE_SYNC before FATTN selected={selected}", pre_sync_ms(line))
                add(groups, f"FATTN selected={selected} Q1={q1}", value)
                if q1 not in ("?", "1", "2"):
                    add(groups, f"FATTN_PREFILL selected={selected}", value)
                    add(groups, f"PRE_SYNC before FATTN_PREFILL selected={selected}", pre_sync_ms(line))
                    add(groups, f"FATTN_PREFILL selected={selected} K1={k1}", value)
                continue

            if "operator(): timing token_offset=" in line:
                chunk = field(line, "n_tokens_chunk")
                fast_exp = field(line, "fast_exp")
                add(groups, "GDN", value)
                add(groups, "PRE_SYNC before GDN", pre_sync_ms(line))
                add(groups, f"GDN chunk={chunk}", value)
                add(groups, f"PRE_SYNC before GDN chunk={chunk}", pre_sync_ms(line))
                add(groups, f"GDN chunk={chunk} fast_exp={fast_exp}", value)
                continue

            if "operator(): timing type=" in line:
                qtype = field(line, "type")
                ncols_dst = field(line, "ncols_dst")
                small_k = field(line, "small_k")
                fusion = field(line, "fusion")
                add(groups, "MMVQ", value)
                add(groups, "PRE_SYNC before MMVQ", pre_sync_ms(line))
                add(groups, f"MMVQ type={qtype} ncols_dst={ncols_dst}", value)
                add(groups, f"MMVQ type={qtype} ncols_dst={ncols_dst} small_k={small_k} fusion={fusion}", value)
                continue

            if "mul_mat_q_case: timing" in line:
                qtype = field(line, "type")
                ncols_max = field(line, "ncols_max")
                mmq_x_best = field(line, "mmq_x_best")
                mmq_y = field(line, "mmq_y")
                occupancy_pct = field(line, "occupancy_pct")
                waves_per_sm = field(line, "waves_per_sm")
                regs = field(line, "regs")
                shared_pct = field(line, "shared_pct")

                add(groups, "MMQ", value)
                add(groups, "PRE_SYNC before MMQ", pre_sync_ms(line))
                add(groups, f"MMQ type={qtype} ncols_max={ncols_max}", value)
                add(groups, f"MMQ type={qtype} ncols_max={ncols_max} mmq_x_best={mmq_x_best}", value)
                add(groups, f"MMQ type={qtype} mmq_x_best={mmq_x_best} mmq_y={mmq_y}", value)
                add(groups, f"MMQ_RES type={qtype} regs={regs} shared_pct={shared_pct}", value)
                add(groups, f"MMQ_RES type={qtype} occupancy_pct={occupancy_pct} waves_per_sm={waves_per_sm}", value)
                continue

            if "process_ubatch: ubatch timing" in line:
                n_tokens = field(line, "n_tokens")
                add(groups, "UBATCH", value)
                add(groups, f"UBATCH n_tokens={n_tokens}", value)

    return groups


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "sum": 0.0, "avg": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "sum": sum(values),
        "avg": sum(values) / len(values),
        "max": max(values),
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare kernel full-trace llama-server logs")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--all", action="store_true", help="include capture-only enqueue timings where sync_applied=0")
    args = parser.parse_args()

    base = parse_log(args.baseline, sync_only=not args.all)
    cand = parse_log(args.candidate, sync_only=not args.all)
    keys = sorted(set(base) | set(cand))

    rows = []
    for key in keys:
        b = summarize(base.get(key, []))
        c = summarize(cand.get(key, []))
        delta_sum = c["sum"] - b["sum"]
        delta_avg = c["avg"] - b["avg"]
        ratio = c["avg"] / b["avg"] if b["avg"] > 0 else 0.0
        rows.append((abs(delta_sum), key, b, c, delta_sum, delta_avg, ratio))

    rows.sort(reverse=True, key=lambda item: item[0])

    print(f"# Kernel Trace Compare: {args.baseline_name} -> {args.candidate_name}")
    print()
    print(f"- baseline: {args.baseline}")
    print(f"- candidate: {args.candidate}")
    print(f"- timing mode: {'all enqueue timings' if args.all else 'sync_applied=1 only'}")
    print()
    print("| key | base count | cand count | base sum ms | cand sum ms | delta sum ms | base avg ms | cand avg ms | avg ratio | cand max ms |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for _, key, b, c, delta_sum, _delta_avg, ratio in rows[: args.top]:
        print(
            f"| `{key}` | {int(b['count'])} | {int(c['count'])} | "
            f"{fmt(b['sum'])} | {fmt(c['sum'])} | {fmt(delta_sum)} | "
            f"{fmt(b['avg'])} | {fmt(c['avg'])} | {fmt(ratio)} | {fmt(c['max'])} |"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())