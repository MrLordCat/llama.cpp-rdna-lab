#!/usr/bin/env python3
from __future__ import annotations

import argparse


def required_local_speedup(target_share: float, target_total_speedup: float) -> float | None:
    # Amdahl form: S_total = 1 / ((1 - s) + s / S_local)
    # Solve for S_local.
    denom = (1.0 / target_total_speedup) - (1.0 - target_share)
    if denom <= 0.0:
        return None
    return target_share / denom


def max_possible_total_speedup(target_share: float) -> float:
    # S_local -> inf
    return 1.0 / (1.0 - target_share) if target_share < 1.0 else float("inf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute required local speedup from target share and desired wall speedup")
    p.add_argument("--share", type=float, required=True, help="target center wall share in [0,1], e.g. 0.39")
    p.add_argument(
        "--goals",
        default="1.01,1.02,1.03",
        help="comma-separated total speedup goals, e.g. 1.01,1.02,1.05",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    s = args.share
    if s <= 0.0 or s >= 1.0:
        raise SystemExit("ERROR: --share must be in (0,1)")

    goals = [float(x.strip()) for x in args.goals.split(",") if x.strip()]

    print("# Required Local Speedup")
    print()
    print(f"- target_share: {s:.6f}")
    print(f"- max_possible_total_speedup: {max_possible_total_speedup(s):.4f}")
    print()
    print("| target_total_speedup | required_local_speedup | feasible |")
    print("|---:|---:|:---:|")

    for g in goals:
        req = required_local_speedup(s, g)
        if req is None:
            print(f"| {g:.4f} | - | no |")
        else:
            print(f"| {g:.4f} | {req:.4f} | yes |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
