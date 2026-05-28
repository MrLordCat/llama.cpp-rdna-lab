#!/usr/bin/env python3
"""H53 analytical gate: can nibble reordering ever reach P003 corridor?

Analytical (no GGUF loading) check of whether the reordering approach can
theoretically reach the corridor, given:
- Current H1 entropy floor: 3.864885 bpw (from D055)
- Corridor: 3.57-3.77 bpw
- Max headroom: 4.0 - 3.864885 = 0.135115 bpw

For reordering to help:
- Sorting creates runs → reduces conditional entropy
- But permutation overhead must be paid
- Net benefit = H1 - Hcond_sorted - perm_overhead

Key insight: for uniform-ish nibble distribution (H1≈3.865), sorting within
small blocks gives minimal Hcond reduction (most bigrams are still random at
block boundaries). For large superblocks, Hcond drops more but perm overhead
grows.

We model:
- H1 = 3.864885 (measured)
- Hcond_sorted ≈ H1 * (1 - run_fraction) + 0 * run_fraction
  where run_fraction = fraction of bigrams that are (a,a) self-transitions
- For sorted data: run_fraction depends on value distribution
- Perm overhead = f(block_size, encoding_method)

We sweep block_size and encoding methods to find if any combination
can reach corridor.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        default=f"q4c2-reorder-analytical-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "build_logs" / "agent-workload"),
    )
    parser.add_argument(
        "--h1",
        type=float,
        default=3.864885,
        help="Source unigram entropy bpw (from D055)",
    )
    parser.add_argument(
        "--corridor-min",
        type=float,
        default=3.57,
    )
    parser.add_argument(
        "--corridor-max",
        type=float,
        default=3.77,
    )
    return parser.parse_args(argv)


def model_sorted_cond_ent(h1: float, block_size: int, distribution: str = "uniform_approx") -> float:
    """Model conditional entropy of sorted nibble stream.

    For sorted data within blocks of size N:
    - Within-block: most transitions are (a, a+1) or (a, a) — very predictable
    - At block boundaries: random transitions (full H1 entropy)

    For a block of N nibbles:
    - N-1 within-block bigrams
    - 1 block-boundary bigram per block

    If we have S symbols (16 for nibbles), and data is roughly uniform:
    - Each value appears ~N/S times per block
    - Within-block entropy ≈ log2(log2(N)) for sorted uniform data
      (each run is ~N/S long, so we just predict 'same or next')

    For block boundaries: entropy ≈ H1 (full)

    Overall Hcond ≈ [(N-1) * H_within + 1 * H1] / N
    """
    N = block_size
    S = 16  # nibble alphabet

    # Within-block conditional entropy for sorted uniform data
    # For sorted data, each value v appears ~N/S times
    # Transitions: (v, v) with probability ~(N/S-1)/(N/S) ≈ 1 - S/N
    #              (v, v+1) with probability ~1/(N/S-1) ≈ S/N
    # H_within ≈ -( (1-S/N)*log2(1-S/N) + (S/N)*log2(S/N) )
    # For large N: ≈ -(1*log2(1) + 0*log2(0)) → small

    if N <= 1:
        return h1

    run_len = N / S  # average run length
    if run_len < 1:
        # Small blocks: no runs, H_within ≈ H1
        h_within = h1
    else:
        # Probability of self-transition (a,a)
        p_same = (run_len - 1) / run_len if run_len > 0 else 0
        # Probability of next-value transition
        p_next = 1.0 - p_same

        # Within-block H(X_i | X_{i-1})
        h_within = 0.0
        if p_same > 0:
            h_within -= p_same * math.log2(p_same)
        if p_next > 0:
            # When transitioning to next, it's one of ~1 values (deterministic for sorted)
            # So conditional entropy for next is ~0
            # But we model as log2(2) = 1 bit for (stay or advance)
            h_within -= p_next * 1.0

    # Block boundary entropy: full H1
    h_boundary = h1

    # Overall: weighted average
    # (N-1) within-block bigrams + 1 boundary bigram per N symbols
    hcond = ((N - 1) * h_within + h_boundary) / N

    return hcond


def permutation_overhead_bpw(block_size: int, method: str) -> float:
    """Model permutation encoding overhead in bpw.

    Methods:
    - "naive_index": log2(N) bits per symbol
    - "run_length": encode sorted data as runs
    - "delta": encode differences between consecutive values
    """
    N = block_size

    if method == "naive_index":
        return math.log2(N)

    elif method == "run_length":
        # For sorted uniform data with S=16 values:
        # ~S runs per block, each run encoded as (value, length)
        # Value: log2(S) = 4 bits
        # Length: log2(N) bits
        # Total: S * (4 + log2(N)) bits for N symbols
        # Per symbol: S * (4 + log2(N)) / N bpw
        S = 16
        return S * (4 + math.log2(N)) / N

    elif method == "delta":
        # For sorted data, deltas are small (mostly 0 or 1)
        # Encode as: first value (4 bits) + N-1 deltas
        # Each delta: for sorted uniform, mostly 0/1 → ~1 bit average
        # Plus block header: 4 bits for first value
        # Per symbol: (4 + (N-1)*1) / N ≈ 1 bpw for large N
        return (4 + (N - 1) * 1.0) / N

    elif method == "ecs":
        # Entropy Coded Symbols: encode sorted nibbles with arithmetic/rANS
        # The sorted sequence has low entropy (mostly runs)
        # H_sorted ≈ h_within from model
        # ECS can approach H_sorted with small overhead
        # We model: H_sorted + 0.05 overhead
        return 0.05  # Just the coding overhead, H_sorted is separate

    return math.log2(N)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h1 = args.h1
    corridor_min = args.corridor_min
    corridor_max = args.corridor_max

    # Sweep block sizes
    block_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    methods = ["naive_index", "run_length", "delta", "ecs"]

    results = []
    feasible_configs = []

    for N in block_sizes:
        hcond = model_sorted_cond_ent(h1, N)

        for method in methods:
            perm_overhead = permutation_overhead_bpw(N, method)
            net_bpw = hcond + perm_overhead

            result = {
                "block_size": N,
                "method": method,
                "h1": h1,
                "hcond_sorted": round(hcond, 6),
                "hcond_reduction": round(h1 - hcond, 6),
                "perm_overhead_bpw": round(perm_overhead, 6),
                "net_bpw": round(net_bpw, 6),
                "in_corridor": corridor_min <= net_bpw <= corridor_max,
            }
            results.append(result)

            if corridor_min <= net_bpw <= corridor_max:
                feasible_configs.append(result)

            print(f"N={N:>6} {method:<12} Hcond={hcond:.4f} Red={h1-hcond:+.4f} "
                  f"Perm={perm_overhead:.4f} Net={net_bpw:.4f} "
                  f"{'FEASIBLE' if result['in_corridor'] else ''}")

    # Summary
    print(f"\n{'='*60}")
    print(f"H53 Analytical Gate: {args.label}")
    print(f"{'='*60}")
    print(f"Source H1: {h1} bpw")
    print(f"Corridor: {corridor_min}-{corridor_max} bpw")
    print(f"Max headroom: {4.0 - h1:.6f} bpw")
    print(f"Configs tested: {len(results)}")
    print(f"Feasible configs: {len(feasible_configs)}")

    if feasible_configs:
        best = min(feasible_configs, key=lambda x: x["net_bpw"])
        print(f"Best feasible: N={best['block_size']}, {best['method']}, "
              f"net={best['net_bpw']:.6f} bpw")
    else:
        best_overall = min(results, key=lambda x: x["net_bpw"])
        print(f"Best overall (not feasible): N={best_overall['block_size']}, "
              f"{best_overall['method']}, net={best_overall['net_bpw']:.6f} bpw")
        print(f"Gap to corridor max: {best_overall['net_bpw'] - corridor_max:+.6f} bpw")

    # Write JSON
    output = {
        "label": args.label,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "h1": h1,
        "corridor": [corridor_min, corridor_max],
        "results": results,
        "feasible_configs": feasible_configs,
    }

    json_path = out_dir / f"{args.label}.q4_c2_reorder_analytical_gate.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nJSON: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
