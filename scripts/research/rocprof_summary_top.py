#!/usr/bin/env python3
"""Parse rocprofv3 --summary output and rank kernels by GPU time.

W17 integration: rocprofv3 (ROCm 10.0) `--kernel-trace --stats --summary -D`
prints a table per tracing domain. This script extracts the KERNEL_DISPATCH
section and prints the dominant kernels by total/percent/calls, so future
weight-stream / GDN device-trace candidate measurement can start from a
single source of truth.

Usage:
    python3 scripts/research/rocprof_summary_top.py summary.txt [--top 25]
    python3 scripts/research/rocprof_summary_top.py /path/to/dir  # finds *summary*.txt
    ... --contains mmvq|mul_mat_vec_q|gated_delta_net
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def find_summary(path: Path) -> Path:
    if path.is_file():
        return path
    # rocprofv3 names the summary "<out>/<summary-output-file>.txt"
    for p in sorted(path.rglob("*.txt")):
        if "summary" in p.name.lower():
            return p
    raise FileNotFoundError(f"no summary .txt under {path}")


def parse_kernel_rows(text: str) -> list[tuple[str, int, float, float, float, float, float]]:
    """Return (kernel_name, calls, sum_ms, avg_ms, pct, min_ms, max_ms) rows."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if "KERNEL_DISPATCH SUMMARY" in l), None)
    if start is None:
        return []
    # stop at the next section header (the tool also prints an all-domain SUMMARY
    # after KERNEL_DISPATCH; without this the kernel rows are double-counted).
    end = next((i for i, l in enumerate(lines[start + 1:], start + 1)
                if " SUMMARY:" in l), len(lines))
    kernel_lines = lines[start:end]

    rows: list[tuple[str, int, float, float, float, float, float]] = []
    for line in kernel_lines:
        if "| KERNEL_DISPATCH |" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 10:
            continue
        # parts[1] = demangled kernel name (single line), parts[2] domain
        # parts[3..] = calls, dur, avg, pct, min, max, std
        try:
            calls = int(parts[3])
            dur = float(parts[4])
            avg = float(parts[5])
            pct = float(parts[6])
            mn = float(parts[7])
            mx = float(parts[8])
        except ValueError:
            continue
        rows.append((parts[1], calls, dur, avg, pct, mn, mx))
    return rows


def short_name(full: str, width: int = 88) -> str:
    name = full.strip().replace("void ", "")
    name = re.sub(r"<.*?>$", "", name, count=1)  # drop template args tail
    return name[:width]


def parse_any_section(text: str, header: str) -> list[tuple[str, int, float, float, float, float, float]]:
    """Parse a generic rocprofv3 summary table identified by `header`."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if f"{header} SUMMARY" in l), None)
    if start is None:
        return []
    end = next((i for i, l in enumerate(lines[start + 1:], start + 1)
                if " SUMMARY:" in l), len(lines))
    rows: list[tuple[str, int, float, float, float, float, float]] = []
    for line in lines[start:end]:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 10 or "DISPATCH" not in str(parts[2]) and "GRAPH" not in str(parts[2]) \
                and "API" not in str(parts[2]):
            # accept DOMAIN cells that name the traced domain
            if len(parts) >= 4 and not parts[1] and not parts[2]:
                continue
        if len(parts) < 10:
            continue
        try:
            calls = int(parts[3])
            dur = float(parts[4])
            avg = float(parts[5])
            pct = float(parts[6])
            mn = float(parts[7])
            mx = float(parts[8])
        except (ValueError, IndexError):
            continue
        rows.append((parts[1], calls, dur, avg, pct, mn, mx))
    return rows


def print_rows(rows: list[tuple[str, int, float, float, float, float, float]], top: int) -> None:
    print(f"{'%':>7} {'total_ms':>10} {'calls':>7} {'avg_ms':>9} {'min':>8} {'max':>8}  kernel")
    for name, calls, dur, avg, pct, mn, mx in rows[:top]:
        print(f"{pct:6.2f}% {dur:10.2f} {calls:7d} {avg:9.3f} {mn:8.4f} {mx:8.4f}  {short_name(name)}")
    total = sum(r[2] for r in rows)
    print(f"\ntotal time: {total:.2f} ms over {len(rows)} distinct entries")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="summary .txt or directory containing it")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--contains", help="case-insensitive substring filter on kernel name")
    ap.add_argument("--graph", action="store_true", help="also print the HIP_GRAPH summary table")
    args = ap.parse_args()

    text = find_summary(args.path).read_text(encoding="utf-8", errors="replace")
    rows = parse_kernel_rows(text)
    if not rows:
        print("no KERNEL_DISPATCH section found", file=sys.stderr)
        return 1

    if args.contains:
        rows = [r for r in rows if args.contains.lower() in r[0].lower()]
    rows.sort(key=lambda r: r[2], reverse=True)

    print("=== KERNEL_DISPATCH ===")
    print_rows(rows, args.top)

    if args.graph:
        g = parse_any_section(text, "HIP_GRAPH")
        if g:
            print("\n=== HIP_GRAPH ===")
            print_rows(g, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
