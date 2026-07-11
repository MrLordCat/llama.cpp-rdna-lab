#!/usr/bin/env python3
"""Model the RDNA4 MTP small-N WMMA padding and a DP4A MMQ alternative."""

from __future__ import annotations

import argparse
import math


MMQ_Y = 64
MMQ_TILE_NE_K = 32
WMMA_X = 16
BLOCK_Q8_1_MMQ_BYTES = 144
WORKGROUP_ALIGNMENT_BYTES = 4 * 32 * 4


def align_up(value: int, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def wmma_lds_bytes(mmq_x: int) -> int:
    ids = mmq_x * 4
    q3_mma_tile_x_k = 2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K // 2 + 4
    x_tile = MMQ_Y * q3_mma_tile_x_k * 4
    y_tile = align_up(mmq_x * BLOCK_Q8_1_MMQ_BYTES, WORKGROUP_ALIGNMENT_BYTES)
    return ids + x_tile + y_tile


def dp4a_lds_bytes(mmq_x: int) -> int:
    ids = mmq_x * 4
    qs_ints = MMQ_Y * MMQ_TILE_NE_K * 2 + MMQ_Y
    dm_half2 = MMQ_Y
    sc_ints = MMQ_Y * MMQ_TILE_NE_K // 8 + MMQ_Y // 8
    x_tile = qs_ints * 4 + dm_half2 * 4 + sc_ints * 4
    y_tile = align_up(mmq_x * BLOCK_Q8_1_MMQ_BYTES, WORKGROUP_ALIGNMENT_BYTES)
    return ids + x_tile + y_tile


def candidate_width(ncols: int) -> int:
    return 4 if ncols <= 4 else 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncols", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument(
        "--dp4a-throughput-ratio",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75, 1.00],
        help="DP4A effective per-computed-column throughput relative to WMMA",
    )
    args = parser.parse_args()

    print("n  dp4a_x  wmma_use  dp4a_use  break_even  lds_wmma  lds_dp4a")
    for ncols in args.ncols:
        width = candidate_width(ncols)
        break_even = width / WMMA_X
        print(
            f"{ncols:>1}  {width:>7}  {ncols / WMMA_X:>8.2%}  "
            f"{ncols / width:>8.2%}  {break_even:>10.2%}  "
            f"{wmma_lds_bytes(WMMA_X) / 1024:>7.2f}K  "
            f"{dp4a_lds_bytes(width) / 1024:>7.2f}K"
        )

    print("\nProjected local body speedup (16 * throughput_ratio / dp4a_x):")
    header = "n " + " ".join(f"r={ratio:.2f}" for ratio in args.dp4a_throughput_ratio)
    print(header)
    for ncols in args.ncols:
        width = candidate_width(ncols)
        values = " ".join(
            f"{WMMA_X * ratio / width:>6.2f}x" for ratio in args.dp4a_throughput_ratio
        )
        print(f"{ncols} {values}")


if __name__ == "__main__":
    main()
