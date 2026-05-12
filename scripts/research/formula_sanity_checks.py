#!/usr/bin/env python3
"""Run internal consistency checks for research formulas."""

from __future__ import annotations

import argparse
import random

from speedup_math import combined_wall_speedup, required_acceptance_for_target_wall, speculative_speedup


def check_monotonic_spec(samples: int, rng: random.Random) -> None:
    for _ in range(samples):
        d = rng.randint(2, 128)
        o = rng.uniform(0.0, 0.5)
        a1 = rng.uniform(0.0, 1.0)
        a2 = rng.uniform(a1, 1.0)
        s1 = speculative_speedup(d, a1, o)
        s2 = speculative_speedup(d, a2, o)
        if s2 + 1e-12 < s1:
            raise AssertionError("spec speedup must be non-decreasing with acceptance")

        d2 = rng.randint(d, 256)
        sd1 = speculative_speedup(d, a1, o)
        sd2 = speculative_speedup(d2, a1, o)
        if sd2 + 1e-12 < sd1:
            raise AssertionError("spec speedup must be non-decreasing with draft length")

        o2 = rng.uniform(o, 1.0)
        so1 = speculative_speedup(d, a1, o)
        so2 = speculative_speedup(d, a1, o2)
        if so2 - 1e-12 > so1:
            raise AssertionError("spec speedup must be non-increasing with overhead")


def check_wall_boundaries(samples: int, rng: random.Random) -> None:
    for _ in range(samples):
        s_prefill = rng.uniform(1.0, 3.0)
        s_spec = rng.uniform(1.0, 40.0)
        s_decode = rng.uniform(1.0, 2.0)

        s_pure_prefill = combined_wall_speedup(1.0, s_prefill, s_spec, s_decode)
        if abs(s_pure_prefill - s_prefill) > 1e-9:
            raise AssertionError("with p=1, wall speedup must equal prefill speedup")

        s_pure_decode = combined_wall_speedup(0.0, s_prefill, s_spec, s_decode)
        expected_decode = s_spec * s_decode
        if abs(s_pure_decode - expected_decode) > 1e-9:
            raise AssertionError("with p=0, wall speedup must equal decode product")


def check_inverse_solver(samples: int, rng: random.Random) -> None:
    checked = 0
    for _ in range(samples * 3):
        d = rng.randint(2, 96)
        a = rng.uniform(0.1, 0.95)
        o = rng.uniform(0.0, 0.2)
        p = rng.uniform(0.3, 0.9)
        s_prefill = rng.uniform(1.0, 1.6)
        s_decode = rng.uniform(1.0, 1.15)

        s_spec = speculative_speedup(d, a, o)
        s_wall = combined_wall_speedup(p, s_prefill, s_spec, s_decode)

        solved = required_acceptance_for_target_wall(
            target_wall_speedup=s_wall,
            prefill_share=p,
            prefill_speedup=s_prefill,
            decode_kernel_speedup=s_decode,
            draft_len=d,
            overhead=o,
        )
        if solved is None:
            continue

        if abs(solved - a) > 1e-7:
            raise AssertionError("inverse solver failed to recover original acceptance")

        checked += 1
        if checked >= samples:
            break

    if checked < max(5, samples // 2):
        raise AssertionError("insufficient inverse-solver checks completed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run consistency checks for speculative/prefill speedup formulas")
    parser.add_argument("--samples", type=int, default=2000, help="number of random checks per category")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    check_monotonic_spec(args.samples, rng)
    check_wall_boundaries(args.samples, rng)
    check_inverse_solver(args.samples, rng)

    print(
        "OK: formula sanity checks passed "
        f"(samples={args.samples}, seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
