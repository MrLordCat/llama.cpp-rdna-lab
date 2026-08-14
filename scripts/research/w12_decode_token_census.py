#!/usr/bin/env python3
"""W12 decode-token census: aggregate GGML_TRACE_CUDA_NODE_TIMING lines.

Usage: python scripts/research/w12_decode_token_census.py <server.log>

Nodes that repeat >=REPEAT_MIN times across the run are treated as the
per-token decode graph replay (prefill nodes occur once per ubatch step).
The per-token cost of an op = median total_ms over its decode occurrences.
ASCII-only output (Windows git-bash cp1252).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

NODE_RE = re.compile(
    r"GGML_TRACE_CUDA_NODE_TIMING: idx=(?P<idx>\d+) kind=(?P<kind>\S+) "
    r"skip=(?P<skip>\d+) op=(?P<op>\S+) name=(?P<name>\S+) stream=(?P<stream>\d+) "
    r"type=(?P<type>\S+) src0_type=(?P<src0_type>\S+) src1_type=(?P<src1_type>\S+) "
    r"buf=(?P<buf>\S+) src0_buf=(?P<src0_buf>\S+) src1_buf=(?P<src1_buf>\S+) "
    r"ne=\((?P<ne>[^)]*)\) .*? total_ms=(?P<total_ms>[0-9.]+)"
)

REPEAT_MIN = 20
MEDIAN_LIMIT = 200  # cap occurrences kept per (op,name) for median memory

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="server log with GGML_TRACE_CUDA_NODE_TIMING lines")
    parser.add_argument("--repeat-min", type=int, default=REPEAT_MIN)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--all", action="store_true", help="print every (op,name), not just top")
    args = parser.parse_args()

    occurrences: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for line in Path(args.log).open(encoding="utf-8", errors="replace"):
        m = NODE_RE.search(line)
        if not m:
            continue
        key = (m["kind"], m["op"], m["name"])
        if m["skip"] == "1":
            continue
        occurrences[key].append(float(m["total_ms"]))

    decode: list[tuple[tuple[str, str, str], float, int]] = []
    prefill: list[tuple[tuple[str, str, str], float, int]] = []
    for key, values in occurrences.items():
        values = sorted(values)[:MEDIAN_LIMIT]
        med = values[len(values) // 2]
        row = (key, med, len(values))
        if len(values) >= args.repeat_min:
            decode.append(row)
        else:
            prefill.append(row)

    decode.sort(key=lambda r: -r[1])
    prefill.sort(key=lambda r: -r[1])

    total_decode = sum(r[1] for r in decode)
    total_prefill = sum(r[1] for r in prefill)

    print(f"decode nodes (repeat>={args.repeat_min}): {len(decode)} unique, per-token total {total_decode:.3f} ms")
    print(f"prefill/unrepeated nodes: {len(prefill)} unique, sum of medians {total_prefill:.3f} ms")
    print()
    print("=== per-token decode breakdown (median ms per node) ===")
    for (kind, op, name), med, cnt in decode[: args.top]:
        share = 100.0 * med / total_decode if total_decode > 0 else 0.0
        print(f"{med:9.3f} ms {share:6.2f}%  n={cnt:<4} {kind:<6} {op:<16} {name}")
    if args.all:
        for (kind, op, name), med, cnt in decode[args.top :]:
            print(f"{med:9.3f} ms   {cnt}  {kind} {op} {name}")

    print()
    print("=== prefill/unrepeated (median ms, informative) ===")
    for (kind, op, name), med, cnt in prefill[: args.top]:
        print(f"{med:9.3f} ms  n={cnt:<4} {kind:<6} {op:<16} {name}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
