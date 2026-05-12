#!/usr/bin/env python3
"""Estimate required acceptance to hit target wall speedup."""

from __future__ import annotations

import argparse

from speedup_math import required_acceptance_for_target_wall


def parse_float_csv(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        out.append(float(text))
    return out


def parse_int_csv(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        out.append(int(text))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve required acceptance ratio for target wall speedup")
    parser.add_argument("--target-wall", required=True, help="target wall speedup(s), comma-separated")
    parser.add_argument("--draft-len", required=True, help="draft length value(s), comma-separated")
    parser.add_argument("--prefill-share", type=float, default=0.70, help="baseline prefill wall-share")
    parser.add_argument("--prefill-speedup", type=float, default=1.20, help="prefill speedup from kernel changes")
    parser.add_argument("--decode-kernel-speedup", type=float, default=1.00, help="decode speedup outside speculation")
    parser.add_argument("--spec-overhead", type=float, default=0.08, help="relative speculative overhead")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    targets = parse_float_csv(args.target_wall)
    drafts = parse_int_csv(args.draft_len)

    if not targets:
        raise SystemExit("ERROR: --target-wall has no values")
    if not drafts:
        raise SystemExit("ERROR: --draft-len has no values")

    print("| target_wall | draft_len | required_acceptance | note |")
    print("|---:|---:|---:|---|")

    for target in targets:
        for draft in drafts:
            required = required_acceptance_for_target_wall(
                target_wall_speedup=target,
                prefill_share=args.prefill_share,
                prefill_speedup=args.prefill_speedup,
                decode_kernel_speedup=args.decode_kernel_speedup,
                draft_len=draft,
                overhead=args.spec_overhead,
            )
            if required is None:
                print(f"| {target:.3f} | {draft} | - | unreachable under current assumptions |")
            else:
                print(f"| {target:.3f} | {draft} | {required:.4f} | feasible |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
