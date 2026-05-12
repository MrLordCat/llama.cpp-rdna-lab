#!/usr/bin/env python3
"""Analytic speedup estimator for speculative decoding + prefill improvements.

This is a planning tool, not a replacement for real benchmarks.
"""

from __future__ import annotations

import argparse
from typing import Iterable

from speedup_math import clamp, combined_wall_speedup, speculative_speedup


def parse_float_csv(raw: str) -> list[float]:
    values: list[float] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        values.append(float(text))
    return values


def print_single_report(
    baseline_tps: float,
    prefill_share: float,
    draft_len: int,
    accept_rate: float,
    spec_overhead: float,
    flash_prefill_speedup: float,
    decode_kernel_speedup: float,
) -> None:
    spec = speculative_speedup(draft_len, accept_rate, spec_overhead)
    wall = combined_wall_speedup(prefill_share, flash_prefill_speedup, spec, decode_kernel_speedup)
    projected_tps = baseline_tps * wall

    print("=== Speedup Estimate ===")
    print(f"Baseline TPS               : {baseline_tps:.4f}")
    print(f"Prefill share (baseline)   : {prefill_share:.4f}")
    print(f"Spec decode speedup        : {spec:.4f}x")
    print(f"Prefill speedup            : {flash_prefill_speedup:.4f}x")
    print(f"Decode kernel speedup      : {decode_kernel_speedup:.4f}x")
    print(f"Combined wall speedup      : {wall:.4f}x")
    print(f"Projected TPS              : {projected_tps:.4f}")


def iter_sweep_values(single: float, sweep: list[float]) -> Iterable[float]:
    if sweep:
        for item in sweep:
            yield item
    else:
        yield single


def print_sweep_table(
    baseline_tps: float,
    prefill_share: float,
    draft_len: int,
    spec_overhead: float,
    decode_kernel_speedup: float,
    accept_values: list[float],
    prefill_values: list[float],
) -> None:
    if not accept_values:
        accept_values = [0.50, 0.60, 0.70]
    if not prefill_values:
        prefill_values = [1.10, 1.20, 1.30]

    print("\n=== Sensitivity Grid (cell = speedup x / projected TPS) ===")
    header = ["prefill\\accept"] + [f"a={a:.2f}" for a in accept_values]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))

    for s_prefill in prefill_values:
        row = [f"{s_prefill:.2f}x"]
        for a in accept_values:
            s_spec = speculative_speedup(draft_len, a, spec_overhead)
            s_wall = combined_wall_speedup(prefill_share, s_prefill, s_spec, decode_kernel_speedup)
            tps = baseline_tps * s_wall
            row.append(f"{s_wall:.2f}x / {tps:.2f}")
        print("| " + " | ".join(row) + " |")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate combined speedups for speculative decode and prefill improvements")
    parser.add_argument("--baseline-tps", type=float, default=9.85, help="baseline aggregate TPS")
    parser.add_argument("--prefill-share", type=float, default=0.70, help="baseline wall-time share spent in prefill")
    parser.add_argument("--draft-len", type=int, default=48, help="drafted tokens per speculative verification")
    parser.add_argument("--accept-rate", type=float, default=0.60, help="accepted draft token ratio (0..1)")
    parser.add_argument("--spec-overhead", type=float, default=0.08, help="relative speculative overhead")
    parser.add_argument("--flash-prefill-speedup", type=float, default=1.30, help="prefill speedup from attention/kernel changes")
    parser.add_argument("--decode-kernel-speedup", type=float, default=1.00, help="decode kernel speedup outside speculation")
    parser.add_argument("--sweep-accept", default="", help="comma-separated acceptance values for sensitivity grid")
    parser.add_argument("--sweep-flash", default="", help="comma-separated prefill speedups for sensitivity grid")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    baseline_tps = float(args.baseline_tps)
    prefill_share = clamp(float(args.prefill_share), 0.0, 1.0)
    draft_len = max(1, int(args.draft_len))
    accept_rate = clamp(float(args.accept_rate), 0.0, 1.0)
    spec_overhead = max(0.0, float(args.spec_overhead))
    flash_prefill_speedup = float(args.flash_prefill_speedup)
    decode_kernel_speedup = float(args.decode_kernel_speedup)

    accept_sweep = parse_float_csv(args.sweep_accept) if args.sweep_accept.strip() else []
    flash_sweep = parse_float_csv(args.sweep_flash) if args.sweep_flash.strip() else []

    print_single_report(
        baseline_tps=baseline_tps,
        prefill_share=prefill_share,
        draft_len=draft_len,
        accept_rate=accept_rate,
        spec_overhead=spec_overhead,
        flash_prefill_speedup=flash_prefill_speedup,
        decode_kernel_speedup=decode_kernel_speedup,
    )

    if accept_sweep or flash_sweep:
        print_sweep_table(
            baseline_tps=baseline_tps,
            prefill_share=prefill_share,
            draft_len=draft_len,
            spec_overhead=spec_overhead,
            decode_kernel_speedup=decode_kernel_speedup,
            accept_values=[clamp(v, 0.0, 1.0) for v in accept_sweep],
            prefill_values=[max(1e-9, v) for v in flash_sweep],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
