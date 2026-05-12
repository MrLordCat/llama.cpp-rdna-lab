#!/usr/bin/env python3
"""Compare observed wall speedup against analytic model assumptions."""

from __future__ import annotations

import argparse

from speedup_math import combined_wall_speedup, required_acceptance_for_target_wall, speculative_speedup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-check observed speedup against analytic assumptions")
    parser.add_argument("--observed-baseline-tps", type=float, required=True)
    parser.add_argument("--observed-candidate-tps", type=float, required=True)
    parser.add_argument("--prefill-share", type=float, default=0.70)
    parser.add_argument("--prefill-speedup", type=float, default=1.00)
    parser.add_argument("--decode-kernel-speedup", type=float, default=1.00)
    parser.add_argument("--draft-len", type=int, default=24)
    parser.add_argument("--accept-rate", type=float, default=0.0)
    parser.add_argument("--spec-overhead", type=float, default=0.08)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    base = float(args.observed_baseline_tps)
    cand = float(args.observed_candidate_tps)
    if base <= 0.0:
        raise SystemExit("ERROR: observed baseline TPS must be > 0")

    observed_speedup = cand / base
    s_spec_assumed = speculative_speedup(args.draft_len, args.accept_rate, args.spec_overhead)
    projected_speedup = combined_wall_speedup(
        prefill_share=args.prefill_share,
        prefill_speedup=args.prefill_speedup,
        spec_speedup=s_spec_assumed,
        decode_kernel_speedup=args.decode_kernel_speedup,
    )

    implied_acceptance = required_acceptance_for_target_wall(
        target_wall_speedup=observed_speedup,
        prefill_share=args.prefill_share,
        prefill_speedup=args.prefill_speedup,
        decode_kernel_speedup=args.decode_kernel_speedup,
        draft_len=args.draft_len,
        overhead=args.spec_overhead,
    )

    print("=== Formula vs Observed ===")
    print(f"observed_baseline_tps        : {base:.6f}")
    print(f"observed_candidate_tps       : {cand:.6f}")
    print(f"observed_wall_speedup        : {observed_speedup:.6f}x")
    print("")
    print("Assumptions:")
    print(f"prefill_share               : {float(args.prefill_share):.6f}")
    print(f"prefill_speedup             : {float(args.prefill_speedup):.6f}x")
    print(f"decode_kernel_speedup       : {float(args.decode_kernel_speedup):.6f}x")
    print(f"draft_len                   : {int(args.draft_len)}")
    print(f"accept_rate_assumed         : {float(args.accept_rate):.6f}")
    print(f"spec_overhead               : {float(args.spec_overhead):.6f}")
    print("")
    print(f"projected_wall_speedup      : {projected_speedup:.6f}x")
    print(f"projection_gap              : {(observed_speedup - projected_speedup):.6f}x")

    if implied_acceptance is None:
        print("implied_acceptance_for_observed: unreachable under current assumptions")
    else:
        print(f"implied_acceptance_for_observed: {implied_acceptance:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
