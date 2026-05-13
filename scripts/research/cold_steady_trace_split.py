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


def main() -> int:
    p = argparse.ArgumentParser(description="Split CUDA_NODE center into cold spikes vs steady window")
    p.add_argument("log", type=Path)
    p.add_argument("--op", default="MUL_MAT")
    p.add_argument("--kind", default="forward")
    p.add_argument("--steady-max-ms", type=float, default=5.0)
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()

    if not args.log.exists():
        raise SystemExit(f"ERROR: log not found: {args.log}")

    route_by_dst: dict[str, str] = {}
    buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)

    with args.log.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "GGML_TRACE_CUDA_MUL_MAT_ROUTE:" in line:
                dst = field(line, "dst")
                route = field(line, "route")
                src0_type = field(line, "src0_type")
                route_by_dst[dst] = f"{route}|{src0_type}"
                continue

            if "GGML_TRACE_CUDA_NODE_TIMING:" not in line:
                continue
            if field(line, "op") != args.op or field(line, "kind") != args.kind:
                continue

            v = total_ms(line)
            if v is None:
                continue
            name = field(line, "name")
            route_key = route_by_dst.get(name, "unknown")

            mode = "steady" if v <= args.steady_max_ms else "cold"
            buckets[mode].append((route_key, v))
            buckets["all"].append((route_key, v))

    print("# Cold-vs-Steady Split")
    print()
    print(f"- log: {args.log}")
    print(f"- center: op={args.op}, kind={args.kind}")
    print(f"- steady_max_ms: {args.steady_max_ms}")

    if not buckets["all"]:
        print("\nNo matching CUDA_NODE lines found.")
        return 2

    for mode in ("all", "cold", "steady"):
        entries = buckets[mode]
        total = sum(v for _, v in entries)
        print(f"\n## {mode} (count={len(entries)}, sum_ms={total:.3f})")

        route_sum: dict[str, float] = defaultdict(float)
        route_count: dict[str, int] = defaultdict(int)
        for route, v in entries:
            route_sum[route] += v
            route_count[route] += 1

        print("| route|type | sum_ms | share | count | avg_ms |")
        print("|---|---:|---:|---:|---:|")
        rows = sorted(route_sum.items(), key=lambda x: x[1], reverse=True)[: args.top]
        for route, s in rows:
            c = route_count[route]
            share = 100.0 * s / total if total > 0 else 0.0
            avg = s / c if c > 0 else 0.0
            print(f"| {route} | {s:.3f} | {share:.2f}% | {c} | {avg:.3f} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
