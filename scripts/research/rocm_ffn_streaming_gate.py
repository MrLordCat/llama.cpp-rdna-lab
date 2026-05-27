#!/usr/bin/env python3
"""Analytical gate for ROCm full-FFN streaming designs on the P002 lane."""

from __future__ import annotations

import argparse
import math


def mib(bytes_count: float) -> float:
    return bytes_count / (1024.0 * 1024.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=17408)
    parser.add_argument("--model", type=int, default=5120)
    parser.add_argument("--ncols", type=int, default=128)
    parser.add_argument("--down-row-tile", type=int, default=64)
    parser.add_argument("--hidden-tile", type=int, default=128)
    parser.add_argument("--dtype-bytes", type=int, default=4)
    parser.add_argument("--gate-up-pair-ms", type=float, default=3.6452)
    parser.add_argument("--down-ms", type=float, default=1.4914)
    parser.add_argument("--pair-only-ms", type=float, default=6.8151)
    args = parser.parse_args()

    baseline_ms = args.gate_up_pair_ms + args.down_ms
    pair_only_total_ms = args.pair_only_ms + args.down_ms

    down_row_tiles = math.ceil(args.model / args.down_row_tile)
    hidden_tiles = math.ceil(args.hidden / args.hidden_tile)

    hidden_elems = args.hidden * args.ncols
    output_elems = args.model * args.ncols
    hidden_bytes = hidden_elems * args.dtype_bytes
    gate_up_glu_bytes = hidden_bytes * 3
    output_bytes = output_elems * args.dtype_bytes

    recompute_lower_bound_ms = args.gate_up_pair_ms * down_row_tiles + args.down_ms
    partial_output_rw_bytes = hidden_tiles * 2 * output_bytes

    print("# ROCm FFN Streaming Gate")
    print()
    print("Inputs:")
    print()
    print(f"- hidden: `{args.hidden}`")
    print(f"- model width: `{args.model}`")
    print(f"- ncols: `{args.ncols}`")
    print(f"- down row tile: `{args.down_row_tile}`")
    print(f"- hidden tile: `{args.hidden_tile}`")
    print(f"- D024 paired rocBLAS+SwiGLU gate/up point: `{args.gate_up_pair_ms:.4f} ms`")
    print(f"- D023/D024 down-like point reference: `{args.down_ms:.4f} ms`")
    print()
    print("| Route model | Lower-bound local time | Local speedup vs materialized baseline | Main blocker |")
    print("| --- | ---: | ---: | --- |")
    print(
        f"| Current materialized gate/up+SwiGLU then down | `{baseline_ms:.4f} ms` | `1.0000x` | Writes/reads hidden, but computes gate/up once |"
    )
    print(
        f"| D024 pair-only fused gate/up+SwiGLU plus unchanged down | `{pair_only_total_ms:.4f} ms` | `{baseline_ms / pair_only_total_ms:.4f}x` | Pair body already slower than separate rocBLAS pair |"
    )
    print(
        f"| Full streaming without hidden materialization, recompute per `{args.down_row_tile}` down rows | `{recompute_lower_bound_ms:.4f} ms` | `{baseline_ms / recompute_lower_bound_ms:.4f}x` | Requires `{down_row_tiles}` gate/up recomputes |"
    )
    print(
        f"| Hidden-tile output accumulation with global partial output R/W | bandwidth-only lower bound | n/a | `{mib(partial_output_rw_bytes):.1f} MiB` partial output traffic per layer at hidden tile `{args.hidden_tile}` |"
    )
    print()
    print("Memory traffic sketch:")
    print()
    print(f"- Minimum SwiGLU hidden materialization: `{mib(hidden_bytes):.2f} MiB` per layer.")
    print(f"- Separate gate/up/SwiGLU intermediates: `{mib(gate_up_glu_bytes):.2f} MiB` per layer.")
    print(f"- One down output tensor: `{mib(output_bytes):.2f} MiB`.")
    print(f"- Hidden-tile partial output read+write at `{args.hidden_tile}` hidden rows: `{mib(partial_output_rw_bytes):.2f} MiB` per layer.")
    print()
    print("Decision signal:")
    print()
    print(
        "A full FFN design that avoids hidden materialization needs cross-down-row sharing of the hidden tile. "
        "Without grid-wide sharing it either recomputes gate/up for every down-row tile or writes partial outputs many times. "
        "The recompute lower bound is already orders of magnitude slower than the materialized point baseline, while the "
        "partial-output path adds hundreds of MiB to GiB of traffic per layer before counting Q3_K work."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())