#!/usr/bin/env python3
"""Amdahl gate for ROCm low-level Q3_K route-body scouts.

This script does not benchmark. It answers the cheap question before writing a
HIP/GCN prototype: how much local speedup must the touched route deliver to move
the active 130k ROCm lane, and what would candidate local wins project to?
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Projection:
    share: float
    local_speedup: float
    projected_tps: float
    total_speedup: float
    delta_percent: float


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value <= 0.0:
            raise argparse.ArgumentTypeError("values must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one value")
    return values


def project_tps(baseline_tps: float, share: float, local_speedup: float) -> Projection:
    if not 0.0 < share < 1.0:
        raise ValueError("share must be between 0 and 1")
    if local_speedup <= 0.0:
        raise ValueError("local_speedup must be positive")
    total_speedup = 1.0 / ((1.0 - share) + share / local_speedup)
    projected_tps = baseline_tps * total_speedup
    return Projection(
        share=share,
        local_speedup=local_speedup,
        projected_tps=projected_tps,
        total_speedup=total_speedup,
        delta_percent=(total_speedup - 1.0) * 100.0,
    )


def required_local_speedup(baseline_tps: float, target_tps: float, share: float) -> float | None:
    if target_tps <= baseline_tps:
        return 1.0
    required_total = target_tps / baseline_tps
    required_time = 1.0 / required_total
    denominator = required_time - (1.0 - share)
    if denominator <= 0.0:
        return None
    return share / denominator


def print_required_table(baseline_tps: float, target_tps: float, shares: list[float]) -> None:
    print("## Required Local Speedup")
    print()
    print(f"Baseline TPS: `{baseline_tps:.4f}`")
    print(f"Target TPS: `{target_tps:.4f}`")
    print()
    print("| Touched share | Required local speedup |")
    print("| ---: | ---: |")
    for share in shares:
        required = required_local_speedup(baseline_tps, target_tps, share)
        if required is None:
            rendered = "unreachable"
        else:
            rendered = f"{required:.4f}x"
        print(f"| {share:.2f} | {rendered} |")
    print()


def print_projection_table(baseline_tps: float, shares: list[float], local_speedups: list[float]) -> None:
    print("## Candidate Projections")
    print()
    print("| Touched share | Local speedup | Projected TPS | Total speedup | Delta |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for share in shares:
        for local_speedup in local_speedups:
            projection = project_tps(baseline_tps, share, local_speedup)
            print(
                f"| {share:.2f} | {local_speedup:.4f}x | "
                f"{projection.projected_tps:.4f} | {projection.total_speedup:.4f}x | "
                f"{projection.delta_percent:+.2f}% |"
            )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-tps", type=float, default=1.3984)
    parser.add_argument("--target-tps", type=float, default=1.6249)
    parser.add_argument("--shares", type=parse_float_list, default=parse_float_list("0.6,0.7,0.8"))
    parser.add_argument(
        "--local-speedups",
        type=parse_float_list,
        default=parse_float_list("1.05,1.10,1.15,1.20,1.30,1.50"),
    )
    args = parser.parse_args()

    if args.baseline_tps <= 0.0 or args.target_tps <= 0.0:
        parser.error("TPS values must be positive")
    for share in args.shares:
        if not 0.0 < share < 1.0:
            parser.error("all shares must be between 0 and 1")

    print_required_table(args.baseline_tps, args.target_tps, args.shares)
    print_projection_table(args.baseline_tps, args.shares, args.local_speedups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())