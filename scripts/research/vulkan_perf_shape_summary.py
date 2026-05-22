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


@dataclass
class Row:
    bucket: str
    shape: str
    calls: int
    total_us: float


def parse_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := MAT_RE.search(line):
            bucket = f"{match.group('op')} {match.group('typ')}"
            shape = f"m={match.group('m')} n={match.group('n')} k={match.group('k')}"
            rows.append(Row(bucket, shape, int(match.group("count")), float(match.group("total"))))
            continue
        if match := FA_RE.search(line):
            rows.append(Row("FLASH_ATTN_EXT", f"N={match.group('n')} KV={match.group('kv')}", int(match.group("count")), float(match.group("total"))))
    return rows


def aggregate(rows: list[Row], key_fn) -> list[tuple[str, int, float]]:
    calls: defaultdict[str, int] = defaultdict(int)
    total_us: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        key = key_fn(row)
        calls[key] += row.calls
        total_us[key] += row.total_us
    return sorted(((key, calls[key], total_us[key]) for key in total_us), key=lambda item: item[2], reverse=True)


def print_table(title: str, rows: list[tuple[str, int, float]], parsed_total_us: float, limit: int) -> None:
    print(f"## {title}")
    print()
    print("| Key | Calls | Total ms | Parsed share |")
    print("|---|---:|---:|---:|")
    for key, calls, total_us in rows[:limit]:
        share = 100.0 * total_us / parsed_total_us if parsed_total_us > 0 else 0.0
        print(f"| `{key}` | {calls} | {total_us / 1000.0:.2f} | {share:.2f}% |")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Vulkan perf logger matmul/FA timings by shape")
    parser.add_argument("log", type=Path)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = parse_rows(args.log)
    if not rows:
        raise SystemExit("no Vulkan perf rows parsed")

    parsed_total_us = sum(row.total_us for row in rows)
    print("# Vulkan Perf Shape Summary")
    print()
    print(f"- log: `{args.log}`")
    print(f"- parsed_rows: {len(rows)}")
    print(f"- parsed_total_ms: {parsed_total_us / 1000.0:.2f}")
    print()

    print_table("By Bucket", aggregate(rows, lambda row: row.bucket), parsed_total_us, args.top)
    print_table("By Shape", aggregate(rows, lambda row: f"{row.bucket} {row.shape}"), parsed_total_us, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
