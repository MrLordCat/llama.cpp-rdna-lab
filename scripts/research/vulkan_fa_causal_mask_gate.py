#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


NUM = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"

FA_RE = re.compile(
    rf"^FLASH_ATTN_EXT\s+dst\([^)]*\),\s+q\(256,(?P<n>\d+),24,1\),\s+"
    rf"k\(256,(?P<kv>\d+),4,1\),\s+v\(256,(?P=kv),4,1\),.*:\s+"
    rf"(?P<count>\d+)\s+x\s+(?P<avg>{NUM})\s+us\s+=\s+(?P<total>{NUM})\s+us",
)


@dataclass(frozen=True)
class FaRow:
    n: int
    kv: int
    calls: int
    total_us: float


@dataclass
class TileTotals:
    rows: int = 0
    calls: int = 0
    total_us: float = 0.0
    eligible_us: float = 0.0
    ineligible_us: float = 0.0
    fa_tiles: int = 0
    all_zero_tiles: int = 0
    all_neg_tiles: int = 0
    mixed_tiles: int = 0
    mask_prepass_groups: int = 0
    mask_prepass_cells: int = 0
    mixed_mask_cells: int = 0


def parse_rows(path: Path) -> list[FaRow]:
    rows: list[FaRow] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = FA_RE.search(line)
        if match:
            rows.append(
                FaRow(
                    n=int(match.group("n")),
                    kv=int(match.group("kv")),
                    calls=int(match.group("count")),
                    total_us=float(match.group("total")),
                )
            )
    return rows


def is_full_chunk(row: FaRow, chunk: int) -> bool:
    return row.n == chunk and row.kv >= row.n and row.kv % chunk == 0


def classify_tiles(row: FaRow, br: int, bc: int, chunk: int) -> tuple[int, int, int, int, int, int]:
    tr = (row.n + br - 1) // br
    tc = (row.kv + bc - 1) // bc

    q_global_base = row.kv - row.n

    fa_tiles = 0
    all_zero = 0
    all_neg = 0
    mixed = 0

    for qi in range(tr):
        q_min = q_global_base + qi * br
        q_max = min(q_min + br - 1, q_global_base + row.n - 1)

        for kj in range(tc):
            k_min = kj * bc
            k_max = min(k_min + bc - 1, row.kv - 1)
            fa_tiles += 1
            if k_min > q_max:
                all_neg += 1
            elif k_max <= q_min:
                all_zero += 1
            else:
                mixed += 1

    prepass_groups = ((row.kv + 16 * bc - 1) // (16 * bc)) * tr
    prepass_cells = prepass_groups * 16 * bc * br
    mixed_cells = mixed * br * bc

    return fa_tiles, all_zero, all_neg, mixed, prepass_groups, prepass_cells, mixed_cells


def route_speedup(total_share: float, local_speedup: float) -> float:
    return 1.0 / ((1.0 - total_share) + total_share / local_speedup)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate an analytic-causal-mask Vulkan FA route against a perf log")
    parser.add_argument("log", type=Path)
    parser.add_argument("--baseline-tps", type=float, default=1.3406)
    parser.add_argument("--parsed-total-ms", type=float, default=80382.35)
    parser.add_argument("--br", type=int, default=16)
    parser.add_argument("--bc", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=1024)
    args = parser.parse_args()

    rows = parse_rows(args.log)
    if not rows:
        raise SystemExit("no FLASH_ATTN_EXT rows parsed")

    totals = TileTotals(rows=len(rows))
    by_shape: list[tuple[FaRow, bool, tuple[int, int, int, int, int, int, int]]] = []

    for row in rows:
        eligible = is_full_chunk(row, args.chunk)
        counts = classify_tiles(row, args.br, args.bc, args.chunk)
        by_shape.append((row, eligible, counts))

        multiplier = row.calls
        totals.calls += row.calls
        totals.total_us += row.total_us
        if eligible:
            totals.eligible_us += row.total_us
            totals.fa_tiles += counts[0] * multiplier
            totals.all_zero_tiles += counts[1] * multiplier
            totals.all_neg_tiles += counts[2] * multiplier
            totals.mixed_tiles += counts[3] * multiplier
            totals.mask_prepass_groups += counts[4] * multiplier
            totals.mask_prepass_cells += counts[5] * multiplier
            totals.mixed_mask_cells += counts[6] * multiplier
        else:
            totals.ineligible_us += row.total_us

    parsed_total_us = args.parsed_total_ms * 1000.0
    fa_share = totals.total_us / parsed_total_us
    eligible_share = totals.eligible_us / parsed_total_us
    eligible_fa_share = totals.eligible_us / totals.total_us if totals.total_us else 0.0
    mask_prepass_mib = totals.mask_prepass_cells * 2 / (1024 * 1024)
    mixed_mask_mib = totals.mixed_mask_cells * 2 / (1024 * 1024)

    print("# Vulkan FA Analytic Causal Mask Gate")
    print()
    print(f"- log: `{args.log}`")
    print(f"- parsed_fa_rows: {len(rows)}")
    print(f"- fa_total_ms: {totals.total_us / 1000.0:.2f}")
    print(f"- fa_parsed_share: {fa_share * 100.0:.2f}%")
    print(f"- eligible_full_chunk_fa_ms: {totals.eligible_us / 1000.0:.2f}")
    print(f"- eligible_full_chunk_parsed_share: {eligible_share * 100.0:.2f}%")
    print(f"- eligible_share_of_fa: {eligible_fa_share * 100.0:.2f}%")
    print(f"- ineligible_tail_or_warmup_fa_ms: {totals.ineligible_us / 1000.0:.2f}")
    print()

    print("## Tile Classification For Eligible Full Chunks")
    print()
    print("| Class | Tiles | Share |")
    print("|---|---:|---:|")
    for label, value in [
        ("all zero", totals.all_zero_tiles),
        ("all -inf / skipped", totals.all_neg_tiles),
        ("mixed boundary", totals.mixed_tiles),
    ]:
        share = value / totals.fa_tiles if totals.fa_tiles else 0.0
        print(f"| {label} | {value} | {share * 100.0:.2f}% |")
    print()
    print(f"- eligible_fa_tiles: {totals.fa_tiles}")
    print(f"- mask_opt_prepass_groups: {totals.mask_prepass_groups}")
    print(f"- mask_opt_prepass_read_proxy: {mask_prepass_mib:.2f} MiB of fp16 mask cells")
    print(f"- mixed_boundary_mask_read_proxy_inside_fa: {mixed_mask_mib:.2f} MiB of fp16 mask cells")
    print()

    print("## Wall-Speedup Corridors")
    print()
    print("| Local speedup on eligible FA chunks | Wall speedup | Projected TPS |")
    print("|---:|---:|---:|")
    for local in [1.02, 1.03, 1.05, 1.08, 1.10, 1.15, 1.20]:
        wall = route_speedup(eligible_share, local)
        print(f"| {local:.2f}x | {wall:.4f}x | {args.baseline_tps * wall:.4f} |")
    print()

    print("## Top FA Rows")
    print()
    print("| N | KV | Calls | Total ms | Eligible | all-zero | all--inf | mixed |")
    print("|---:|---:|---:|---:|:---:|---:|---:|---:|")
    for row, eligible, counts in sorted(by_shape, key=lambda item: item[0].total_us, reverse=True)[:12]:
        print(
            f"| {row.n} | {row.kv} | {row.calls} | {row.total_us / 1000.0:.2f} | "
            f"{'yes' if eligible else 'no'} | {counts[1]} | {counts[2]} | {counts[3]} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
