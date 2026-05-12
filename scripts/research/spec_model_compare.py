#!/usr/bin/env python3
"""Compare naive and coverage-aware speculative formula projections."""

from __future__ import annotations

import argparse

from speedup_math import combined_wall_speedup, required_acceptance_for_target_wall, speculative_speedup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare naive vs coverage-aware speculative projections")
    parser.add_argument("--observed-baseline-tps", type=float, required=True)
    parser.add_argument("--observed-candidate-tps", type=float, required=True)
    parser.add_argument("--prefill-share", type=float, required=True)
    parser.add_argument("--prefill-speedup", type=float, required=True)
    parser.add_argument("--decode-kernel-speedup", type=float, required=True)
    parser.add_argument("--draft-len", type=int, required=True)
    parser.add_argument("--local-acceptance", type=float, required=True)
    parser.add_argument("--coverage", type=float, required=True)
    parser.add_argument("--spec-overhead", type=float, default=0.08)
    return parser


def project_wall_speedup(
    prefill_share: float,
    prefill_speedup: float,
    decode_kernel_speedup: float,
    draft_len: int,
    acceptance: float,
    spec_overhead: float,
) -> float:
    s_spec = speculative_speedup(draft_len, acceptance, spec_overhead)
    return combined_wall_speedup(
        prefill_share=prefill_share,
        prefill_speedup=prefill_speedup,
        spec_speedup=s_spec,
        decode_kernel_speedup=decode_kernel_speedup,
    )


def main() -> int:
    args = build_parser().parse_args()

    base = float(args.observed_baseline_tps)
    cand = float(args.observed_candidate_tps)
    if base <= 0.0:
        raise SystemExit("ERROR: baseline TPS must be > 0")

    observed = cand / base
    local = float(args.local_acceptance)
    coverage = float(args.coverage)
    effective = local * coverage

    naive_proj = project_wall_speedup(
        prefill_share=args.prefill_share,
        prefill_speedup=args.prefill_speedup,
        decode_kernel_speedup=args.decode_kernel_speedup,
        draft_len=args.draft_len,
        acceptance=local,
        spec_overhead=args.spec_overhead,
    )

    cov_proj = project_wall_speedup(
        prefill_share=args.prefill_share,
        prefill_speedup=args.prefill_speedup,
        decode_kernel_speedup=args.decode_kernel_speedup,
        draft_len=args.draft_len,
        acceptance=effective,
        spec_overhead=args.spec_overhead,
    )

    implied = required_acceptance_for_target_wall(
        target_wall_speedup=observed,
        prefill_share=args.prefill_share,
        prefill_speedup=args.prefill_speedup,
        decode_kernel_speedup=args.decode_kernel_speedup,
        draft_len=args.draft_len,
        overhead=args.spec_overhead,
    )

    naive_err = abs(observed - naive_proj)
    cov_err = abs(observed - cov_proj)

    print("=== Spec Model Compare ===")
    print(f"observed_wall_speedup          : {observed:.6f}x")
    print(f"local_acceptance              : {local:.6f}")
    print(f"coverage                      : {coverage:.6f}")
    print(f"effective_acceptance          : {effective:.6f}")
    print("")
    print(f"naive_projection              : {naive_proj:.6f}x")
    print(f"coverage_aware_projection     : {cov_proj:.6f}x")
    print(f"naive_abs_error               : {naive_err:.6f}")
    print(f"coverage_aware_abs_error      : {cov_err:.6f}")
    if implied is None:
        print("implied_acceptance_for_observed: unreachable")
    else:
        print(f"implied_acceptance_for_observed: {implied:.6f}")

    winner = "coverage-aware" if cov_err < naive_err else "naive"
    print(f"better_fit_model              : {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
