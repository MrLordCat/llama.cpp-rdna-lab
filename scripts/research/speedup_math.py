#!/usr/bin/env python3
"""Shared math helpers for research speedup estimation."""

from __future__ import annotations


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def speculative_speedup(draft_len: int, accept_rate: float, overhead: float) -> float:
    """Approximate decode speedup from speculative decoding.

    S_spec ~= (1 + a * (D - 1)) / (1 + o)
    """
    d = max(1, int(draft_len))
    a = clamp(float(accept_rate), 0.0, 1.0)
    o = max(0.0, float(overhead))

    effective_tokens_per_step = 1.0 + a * (d - 1.0)
    return effective_tokens_per_step / (1.0 + o)


def combined_wall_speedup(
    prefill_share: float,
    prefill_speedup: float,
    spec_speedup: float,
    decode_kernel_speedup: float,
) -> float:
    """Combine prefill/decode improvements into one wall-time speedup."""
    p = clamp(float(prefill_share), 0.0, 1.0)
    s_prefill = max(1e-9, float(prefill_speedup))
    s_spec = max(1e-9, float(spec_speedup))
    s_decode = max(1e-9, float(decode_kernel_speedup))

    denom = p / s_prefill + (1.0 - p) / (s_spec * s_decode)
    if denom <= 0.0:
        raise ValueError("invalid denominator in combined speedup computation")
    return 1.0 / denom


def required_acceptance_for_target_wall(
    target_wall_speedup: float,
    prefill_share: float,
    prefill_speedup: float,
    decode_kernel_speedup: float,
    draft_len: int,
    overhead: float,
) -> float | None:
    """Solve required acceptance ratio to reach target wall speedup.

    Returns None when target is mathematically unreachable with given inputs.
    """
    d = max(1, int(draft_len))
    p = clamp(float(prefill_share), 0.0, 1.0)
    s_prefill = max(1e-9, float(prefill_speedup))
    s_decode = max(1e-9, float(decode_kernel_speedup))
    o = max(0.0, float(overhead))
    target = max(1e-9, float(target_wall_speedup))

    # Acceptance has no effect when only 1 token is drafted.
    if d == 1:
        s_spec_fixed = speculative_speedup(draft_len=1, accept_rate=0.0, overhead=o)
        achievable = combined_wall_speedup(
            prefill_share=p,
            prefill_speedup=s_prefill,
            spec_speedup=s_spec_fixed,
            decode_kernel_speedup=s_decode,
        )
        if achievable + 1e-12 >= target:
            return 0.0
        return None

    wall_at_a0 = combined_wall_speedup(
        prefill_share=p,
        prefill_speedup=s_prefill,
        spec_speedup=speculative_speedup(draft_len=d, accept_rate=0.0, overhead=o),
        decode_kernel_speedup=s_decode,
    )
    if wall_at_a0 + 1e-12 >= target:
        return 0.0

    wall_at_a1 = combined_wall_speedup(
        prefill_share=p,
        prefill_speedup=s_prefill,
        spec_speedup=speculative_speedup(draft_len=d, accept_rate=1.0, overhead=o),
        decode_kernel_speedup=s_decode,
    )
    if wall_at_a1 + 1e-12 < target:
        return None

    inv_target = 1.0 / target
    rhs = inv_target - (p / s_prefill)
    if rhs <= 0.0:
        return None

    s_spec_required = (1.0 - p) / (s_decode * rhs)
    if s_spec_required <= 0.0:
        return None

    acceptance = (s_spec_required * (1.0 + o) - 1.0) / (d - 1.0)
    if acceptance < 0.0:
        return 0.0
    if acceptance > 1.0:
        return None
    return acceptance
