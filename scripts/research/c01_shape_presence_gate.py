#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


def field(line: str, name: str, default: str = "?") -> str:
    m = re.search(rf"\b{re.escape(name)}=([^\s]+)", line)
    return m.group(1).rstrip(",") if m else default


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shape-presence gate for C01/MMQ lane traces")
    p.add_argument("log", type=Path, help="server log with GGML trace lines")
    p.add_argument("--qtype", default="11", help="MMQ qtype to check (default: 11 for q3_K)")
    p.add_argument("--ncols", default="", help="comma-separated ncols_max values that must be present, e.g. 139,140")
    p.add_argument("--min-count", type=int, default=1, help="minimum count per required ncols bucket")
    p.add_argument("--top", type=int, default=10, help="top histogram rows to print")
    p.add_argument("--strict", action="store_true", help="return non-zero when required buckets are missing")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.log.exists():
        raise SystemExit(f"ERROR: log not found: {args.log}")

    required = [x.strip() for x in args.ncols.split(",") if x.strip()]

    hist = Counter()
    with args.log.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "mul_mat_q_case: timing" not in line:
                continue
            if field(line, "type") != args.qtype:
                continue
            ncols_max = field(line, "ncols_max")
            hist[ncols_max] += 1

    print(f"# Shape Presence Gate\n")
    print(f"- log: {args.log}")
    print(f"- qtype: {args.qtype}")
    print(f"- required ncols_max: {required if required else '[none]'}")
    print(f"- min_count: {args.min_count}")
    print("")

    if not hist:
        print("No MMQ timing lines found for selected qtype.")
        return 2 if args.strict else 0

    print("## Observed ncols_max histogram")
    print("| ncols_max | count |")
    print("|---:|---:|")
    for key, count in hist.most_common(args.top):
        print(f"| {key} | {count} |")

    if not required:
        return 0

    missing = []
    print("\n## Required bucket check")
    print("| ncols_max | observed | pass |")
    print("|---:|---:|:---:|")
    for ncols in required:
        observed = hist.get(ncols, 0)
        ok = observed >= args.min_count
        print(f"| {ncols} | {observed} | {'yes' if ok else 'no'} |")
        if not ok:
            missing.append(ncols)

    if missing:
        print("\nGate verdict: FAIL")
        print(f"Missing/underfilled buckets: {', '.join(missing)}")
        return 3 if args.strict else 0

    print("\nGate verdict: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
