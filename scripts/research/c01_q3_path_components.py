#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


TOTAL_MS_RE = re.compile(r"total_ms=([0-9.]+)")


def field(line: str, name: str, default: str = "?") -> str:
    m = re.search(rf"\b{re.escape(name)}=([^\s]+)", line)
    return m.group(1).rstrip(",") if m else default


def total_ms(line: str) -> float | None:
    m = TOTAL_MS_RE.search(line)
    return float(m.group(1)) if m else None


def component_from_route(route: str) -> str:
    if route == "mul_mat_q_direct":
        return "compute_core_q3"
    if route == "mul_mat_vec_q_direct":
        return "dequant_load_vec_q3"
    if route == "cublas_backend":
        return "fallback_cublas"
    return "other"


def main() -> int:
    p = argparse.ArgumentParser(description="Coarse q3 path component split from existing trace lines")
    p.add_argument("log", type=Path)
    p.add_argument("--kind", default="forward", help="MUL_MAT kind filter")
    p.add_argument("--steady-max-ms", type=float, default=5.0)
    args = p.parse_args()

    if not args.log.exists():
        raise SystemExit(f"ERROR: log not found: {args.log}")

    route_by_dst: dict[str, tuple[str, str]] = {}
    rows: list[tuple[str, float]] = []
    rows_steady: list[tuple[str, float]] = []

    with args.log.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "GGML_TRACE_CUDA_MUL_MAT_ROUTE:" in line:
                dst = field(line, "dst")
                route = field(line, "route")
                src0_type = field(line, "src0_type")
                route_by_dst[dst] = (route, src0_type)
                continue

            if "GGML_TRACE_CUDA_NODE_TIMING:" not in line:
                continue
            if field(line, "op") != "MUL_MAT":
                continue
            if field(line, "kind") != args.kind:
                continue

            v = total_ms(line)
            if v is None:
                continue
            name = field(line, "name")
            route, src0_type = route_by_dst.get(name, ("unknown", "?"))
            if src0_type != "q3_K" and route != "cublas_backend":
                continue

            comp = component_from_route(route)
            rows.append((comp, v))
            if v <= args.steady_max_ms:
                rows_steady.append((comp, v))

    def print_table(title: str, data: list[tuple[str, float]]) -> None:
        agg: dict[str, float] = defaultdict(float)
        cnt: dict[str, int] = defaultdict(int)
        total = 0.0
        for comp, v in data:
            agg[comp] += v
            cnt[comp] += 1
            total += v

        print(f"\n## {title} (sum_ms={total:.3f}, count={len(data)})")
        print("| component | sum_ms | share | count | avg_ms |")
        print("|---|---:|---:|---:|---:|")
        for comp, s in sorted(agg.items(), key=lambda x: x[1], reverse=True):
            c = cnt[comp]
            share = 100.0 * s / total if total > 0 else 0.0
            avg = s / c if c > 0 else 0.0
            print(f"| {comp} | {s:.3f} | {share:.2f}% | {c} | {avg:.3f} |")

    print("# C01 q3 Path Component Split")
    print()
    print("- Components are coarse proxies derived from route lines, not direct kernel-internal cycle counters.")
    print(f"- steady_max_ms: {args.steady_max_ms}")

    if not rows:
        print("\nNo matching q3_K MUL_MAT rows found.")
        return 2

    print_table("all", rows)
    print_table("steady", rows_steady)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
