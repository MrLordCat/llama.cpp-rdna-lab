#!/usr/bin/env python3
"""Summarize ROCm MMQ timing/resource trace rows from llama-server logs."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


TYPE_NAMES = {
    10: "q2_K",
    11: "q3_K",
    12: "q4_K",
    13: "q5_K",
    14: "q6_K",
}

TIMING_RE = re.compile(
    r"mul_mat_q_case: timing type=(?P<type>\d+).*?"
    r"nrows_x=(?P<nrows>\d+) ncols_max=(?P<ncols_max>\d+) ncols_dst=(?P<ncols_dst>\d+).*?"
    r"mmq_x_best=(?P<mmq_x>\d+).*?mmq_y=(?P<mmq_y>\d+) q3k_padded=(?P<q3k_padded>\d+).*?"
    r"nbytes_shared=(?P<nbytes_shared>\d+).*?regs=(?P<regs>-?\d+).*?"
    r"occupancy_pct=(?P<occupancy>[0-9.\-]+).*?total_ms=(?P<total_ms>[0-9.]+)"
)


@dataclass(frozen=True)
class ShapeKey:
    qtype: int
    nrows: int
    ncols_max: int
    ncols_dst: int


@dataclass
class Aggregate:
    count: int = 0
    total_ms: float = 0.0
    mmq_x: int = 0
    mmq_y: int = 0
    q3k_padded: int = 0
    regs: int = 0
    nbytes_shared: int = 0
    occupancy: float = 0.0

    def add(self, match: re.Match[str]) -> None:
        self.count += 1
        self.total_ms += float(match.group("total_ms"))
        self.mmq_x = int(match.group("mmq_x"))
        self.mmq_y = int(match.group("mmq_y"))
        self.q3k_padded = int(match.group("q3k_padded"))
        self.regs = int(match.group("regs"))
        self.nbytes_shared = int(match.group("nbytes_shared"))
        self.occupancy = float(match.group("occupancy"))


def parse_log(path: Path) -> tuple[dict[int, Aggregate], dict[ShapeKey, Aggregate]]:
    by_type: dict[int, Aggregate] = defaultdict(Aggregate)
    by_shape: dict[ShapeKey, Aggregate] = defaultdict(Aggregate)

    for line in path.read_text(errors="replace").splitlines():
        match = TIMING_RE.search(line)
        if match is None:
            continue

        qtype = int(match.group("type"))
        by_type[qtype].add(match)

        key = ShapeKey(
            qtype=qtype,
            nrows=int(match.group("nrows")),
            ncols_max=int(match.group("ncols_max")),
            ncols_dst=int(match.group("ncols_dst")),
        )
        by_shape[key].add(match)

    return by_type, by_shape


def type_name(qtype: int) -> str:
    return TYPE_NAMES.get(qtype, f"type_{qtype}")


def print_summary(path: Path, top: int, wall_ms: float | None) -> None:
    by_type, by_shape = parse_log(path)
    total_ms = sum(item.total_ms for item in by_type.values())
    total_count = sum(item.count for item in by_type.values())

    print(f"# ROCm MMQ Trace Summary\n")
    print(f"Log: `{path}`")
    print(f"Timing rows: `{total_count}`")
    print(f"MMQ total ms: `{total_ms:.3f}`")
    if wall_ms is not None and wall_ms > 0.0:
        print(f"MMQ / wall: `{100.0 * total_ms / wall_ms:.2f}%`")
    print()

    print("## By Type\n")
    print("| Type | Name | Count | Total ms | MMQ share | Wall share |")
    print("| ---: | --- | ---: | ---: | ---: | ---: |")
    for qtype, aggregate in sorted(by_type.items(), key=lambda item: item[1].total_ms, reverse=True):
        mmq_share = 100.0 * aggregate.total_ms / total_ms if total_ms > 0.0 else 0.0
        wall_share = 100.0 * aggregate.total_ms / wall_ms if wall_ms else 0.0
        print(
            f"| {qtype} | {type_name(qtype)} | {aggregate.count} | "
            f"{aggregate.total_ms:.3f} | {mmq_share:.2f}% | {wall_share:.2f}% |"
        )
    print()

    print("## Top Shapes\n")
    print("| Type | Name | nrows | ncols_max | ncols_dst | Count | Total ms | Avg ms | MMQ share | mmq | padded | regs | LDS | occ |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for key, aggregate in sorted(by_shape.items(), key=lambda item: item[1].total_ms, reverse=True)[:top]:
        avg_ms = aggregate.total_ms / aggregate.count if aggregate.count else 0.0
        mmq_share = 100.0 * aggregate.total_ms / total_ms if total_ms > 0.0 else 0.0
        print(
            f"| {key.qtype} | {type_name(key.qtype)} | {key.nrows} | {key.ncols_max} | {key.ncols_dst} | "
            f"{aggregate.count} | {aggregate.total_ms:.3f} | {avg_ms:.4f} | {mmq_share:.2f}% | "
            f"{aggregate.mmq_x}x{aggregate.mmq_y} | {aggregate.q3k_padded} | {aggregate.regs} | "
            f"{aggregate.nbytes_shared} | {aggregate.occupancy:.2f}% |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="llama-server log containing GGML_TRACE_MMQ_TIMING rows")
    parser.add_argument("--top", type=int, default=20, help="number of hot shapes to print")
    parser.add_argument("--wall-ms", type=float, default=None, help="optional task wall time in ms for wall-share estimates")
    args = parser.parse_args()

    print_summary(args.log, args.top, args.wall_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())