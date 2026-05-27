#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


NUM = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"

MAT_RE = re.compile(
    rf"^(?P<op>MUL_MAT_ADD MUL_MAT_VEC|MUL_MAT_VEC|MUL_MAT)\s+"
    rf"(?P<typ>\S+)\s+m=(?P<m>\d+)\s+n=(?P<n>\d+)\s+k=(?P<k>\d+):\s+"
    rf"(?P<count>\d+)\s+x\s+(?P<avg>{NUM})\s+us\s+=\s+(?P<total>{NUM})\s+us",
)

FA_RE = re.compile(
    rf"^FLASH_ATTN_EXT\s+dst\([^)]*\),\s+q\(256,(?P<n>\d+),24,1\),\s+"
    rf"k\(256,(?P<kv>\d+),4,1\),\s+v\(256,(?P=kv),4,1\),.*:\s+"
    rf"(?P<count>\d+)\s+x\s+(?P<avg>{NUM})\s+us\s+=\s+(?P<total>{NUM})\s+us",
)

GLU_RE = re.compile(
    rf"^GLU:\s+(?P<count>\d+)\s+x\s+(?P<avg>{NUM})\s+us\s+=\s+(?P<total>{NUM})\s+us",
)


@dataclass(frozen=True)
class PerfRow:
    bucket: str
    shape: str
    calls: int
    total_us: float


@dataclass(frozen=True)
class Route:
    key: str
    label: str
    total_us: float
    note: str


def parse_float_csv(raw: str) -> list[float]:
    out: list[float] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if text:
            out.append(float(text))
    return out


def parse_rows(path: Path) -> list[PerfRow]:
    rows: list[PerfRow] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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


def aggregate_us(rows: list[PerfRow], predicate) -> float:
    return sum(row.total_us for row in rows if predicate(row))


def has_mat_shape(row: PerfRow, m: int, k: int) -> bool:
    return row.shape.startswith(f"m={m} ") and row.shape.endswith(f" k={k}")


def route_speedup(total_share: float, local_speedup: float) -> float:
    return 1.0 / ((1.0 - total_share) + total_share / local_speedup)


def required_local_speedup(route_share: float, target_speedup: float) -> float | None:
    denom = (1.0 / target_speedup) - (1.0 - route_share)
    if denom <= 0.0:
        return None
    return route_share / denom


def build_routes(rows: list[PerfRow]) -> list[Route]:
    q3_all = aggregate_us(rows, lambda row: row.bucket == "MUL_MAT q3_K")
    q3_gate_up = aggregate_us(rows, lambda row: row.bucket == "MUL_MAT q3_K" and has_mat_shape(row, 17408, 5120))
    q3_down = aggregate_us(rows, lambda row: row.bucket == "MUL_MAT q3_K" and has_mat_shape(row, 5120, 17408))
    q3_other_prefill = aggregate_us(
        rows,
        lambda row: row.bucket == "MUL_MAT q3_K"
        and not has_mat_shape(row, 17408, 5120)
        and not has_mat_shape(row, 5120, 17408),
    )
    fa_all = aggregate_us(rows, lambda row: row.bucket == "FLASH_ATTN_EXT")
    glu_all = aggregate_us(rows, lambda row: row.bucket == "GLU")

    return [
        Route("q3_gate_up", "Dense FFN gate/up Q3_K", q3_gate_up, "all n columns for two sibling 17408x5120 projections per layer; fusion must reuse B/activation tile to matter"),
        Route("q3_down", "Dense FFN down Q3_K", q3_down, "all n columns for 5120x17408 projection after SwiGLU; not helped by gate/up-only fusion"),
        Route("q3_top2", "Dense FFN gate/up + down Q3_K", q3_gate_up + q3_down, "main dense FFN Q3_K route"),
        Route("q3_other", "Other Q3_K prefill shapes", q3_other_prefill, "GDN/SSM/attention-side Q3_K shapes outside top FFN pair"),
        Route("q3_all", "All Q3_K MUL_MAT", q3_all, "whole Q3_K large-prefill route"),
        Route("fa_all", "All FLASH_ATTN_EXT", fa_all, "q4/q4 long-KV FlashAttention route"),
        Route("q3_plus_fa", "All Q3_K MUL_MAT + FA", q3_all + fa_all, "combined prefill core"),
        Route("glu_all", "All GLU", glu_all, "post-op only; useful to show launch/post-op fusion ceiling"),
    ]


def print_routes(routes: list[Route], parsed_total_us: float, target_speedup: float | None) -> None:
    print("## Route Shares")
    print()
    print("| Route | Total ms | Parsed share | Required local speedup to target | Note |")
    print("|---|---:|---:|---:|---|")
    for route in routes:
        share = route.total_us / parsed_total_us if parsed_total_us > 0 else 0.0
        if target_speedup is None:
            required = "-"
        else:
            required_value = required_local_speedup(share, target_speedup)
            required = "unreachable" if required_value is None else f"{required_value:.3f}x"
        print(f"| `{route.label}` | {route.total_us / 1000.0:.2f} | {share * 100.0:.2f}% | {required} | {route.note} |")
    print()


def print_scenarios(routes: list[Route], parsed_total_us: float, local_speedups: list[float], baseline_tps: float | None) -> None:
    print("## Local-Speedup Scenarios")
    print()
    header = ["Route"] + [f"{value:.2f}x local" for value in local_speedups]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for route in routes:
        share = route.total_us / parsed_total_us if parsed_total_us > 0 else 0.0
        row = [f"`{route.label}`"]
        for local in local_speedups:
            speedup = route_speedup(share, local)
            if baseline_tps is None:
                row.append(f"{speedup:.4f}x")
            else:
                row.append(f"{speedup:.4f}x / {baseline_tps * speedup:.4f} TPS")
        print("| " + " | ".join(row) + " |")
    print()


def print_shape_counts(rows: list[PerfRow]) -> None:
    calls: defaultdict[str, int] = defaultdict(int)
    total_us: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        if row.bucket not in {"MUL_MAT q3_K", "FLASH_ATTN_EXT", "GLU"}:
            continue
        key = f"{row.bucket} {row.shape}"
        calls[key] += row.calls
        total_us[key] += row.total_us

    print("## Top Parsed Shapes")
    print()
    print("| Shape | Calls | Total ms |")
    print("|---|---:|---:|")
    for key, total in sorted(total_us.items(), key=lambda item: item[1], reverse=True)[:16]:
        print(f"| `{key}` | {calls[key]} | {total / 1000.0:.2f} |")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate route-level speedup ceilings from a Vulkan perf log")
    parser.add_argument("log", type=Path)
    parser.add_argument("--baseline-tps", type=float, default=None)
    parser.add_argument("--target-tps", type=float, default=None)
    parser.add_argument("--local-speedups", default="1.05,1.10,1.20,1.35,1.50,2.00")
    args = parser.parse_args()

    rows = parse_rows(args.log)
    if not rows:
        raise SystemExit("no Vulkan perf rows parsed")

    parsed_total_us = sum(row.total_us for row in rows)
    target_speedup = None
    if args.baseline_tps and args.target_tps:
        target_speedup = args.target_tps / args.baseline_tps

    print("# Vulkan Route Ceiling")
    print()
    print(f"- log: `{args.log}`")
    print(f"- parsed_rows: {len(rows)}")
    print(f"- parsed_total_ms: {parsed_total_us / 1000.0:.2f}")
    if args.baseline_tps:
        print(f"- baseline_tps: {args.baseline_tps:.4f}")
    if args.target_tps:
        print(f"- target_tps: {args.target_tps:.4f}")
    if target_speedup is not None:
        print(f"- target_speedup: {target_speedup:.4f}x")
    print()

    routes = build_routes(rows)
    print_routes(routes, parsed_total_us, target_speedup)
    print_scenarios(routes, parsed_total_us, parse_float_csv(args.local_speedups), args.baseline_tps)
    print_shape_counts(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
