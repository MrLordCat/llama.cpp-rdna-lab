#!/usr/bin/env python3
"""Analytic gate for C01 Q3_K MMQ ideas.

This is intentionally small and deterministic: it does not prove speedups, but
it catches candidates whose geometry cannot plausibly help before we spend a
ROCm rebuild/benchmark cycle.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


QK8_1 = 32
MMQ_TILE_NE_K = 32
QR3_K = 4
QI8_1 = QK8_1 // 4
Q3_MMA_STRIDE_INT = 2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K // 2 + 4
Q8_1_MMQ_BYTES = 4 * QK8_1 + 4 * 4
SMEM_HALF = 32768


RESOURCE_RE = re.compile(
    r"mul_mat_q_case: timing type=11 .*?"
    r"ncols_max=(?P<ncols>\d+).*?"
    r"mmq_x_best=(?P<mmq_x>\d+).*?"
    r"mmq_y=(?P<mmq_y>\d+).*?"
    r"block_threads=(?P<threads>\d+).*?"
    r"nbytes_shared=(?P<shared>\d+).*?"
    r"shared_pct=(?P<shared_pct>[0-9.]+).*?"
    r"regs=(?P<regs>\d+).*?"
    r"occupancy_pct=(?P<occupancy>[0-9.]+).*?"
    r"waves_per_sm=(?P<waves>[0-9.]+)",
    re.S,
)


@dataclass(frozen=True)
class Resource:
    mmq_x: int
    mmq_y: int
    shared: int
    regs: int
    occupancy: float
    waves: float
    threads: int


def parse_resource(log_path: Path, ncols: int) -> Resource:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for match in RESOURCE_RE.finditer(text):
        if int(match.group("ncols")) == ncols:
            return Resource(
                mmq_x=int(match.group("mmq_x")),
                mmq_y=int(match.group("mmq_y")),
                shared=int(match.group("shared")),
                regs=int(match.group("regs")),
                occupancy=float(match.group("occupancy")),
                waves=float(match.group("waves")),
                threads=int(match.group("threads")),
            )
    raise SystemExit(f"no type=11 resource line found for ncols_max={ncols} in {log_path}")


def q3_x_bytes(mmq_y: int, stride_int: int = Q3_MMA_STRIDE_INT) -> int:
    return mmq_y * stride_int * 4


def q8_y_bytes(mmq_x: int) -> int:
    return mmq_x * Q8_1_MMQ_BYTES


def inferred_misc_bytes(res: Resource) -> int:
    return res.shared - q3_x_bytes(res.mmq_y) - q8_y_bytes(res.mmq_x)


def ntiles(ncols: int, mmq_x: int) -> int:
    return (ncols + mmq_x - 1) // mmq_x


def print_row(name: str, shared: int, tile_ratio: float, block_unlock: bool, verdict: str) -> None:
    print(f"| {name} | {shared} | {shared / 65536 * 100:.2f}% | {tile_ratio:.3f}x | {block_unlock} | {verdict} |")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_log", type=Path)
    parser.add_argument("--ncols", type=int, default=192)
    args = parser.parse_args()

    res = parse_resource(args.trace_log, args.ncols)
    misc = inferred_misc_bytes(res)
    base_tiles = ntiles(args.ncols, res.mmq_x)

    print("# C01 Q3_K MMQ Theory Gate")
    print()
    print(f"- trace: `{args.trace_log}`")
    print(f"- active bucket: `type=11 ncols_max={args.ncols}`")
    print(f"- current geometry: `mmq_x={res.mmq_x}`, `mmq_y={res.mmq_y}`, `threads={res.threads}`")
    print(f"- resource: shared `{res.shared}` bytes ({res.shared / 65536 * 100:.2f}%), regs `{res.regs}`, waves `{res.waves}`")
    print(f"- inferred shared split: x_tile `{q3_x_bytes(res.mmq_y)}`, y_tile `{q8_y_bytes(res.mmq_x)}`, misc `{misc}`")
    print(f"- current x tile count for ncols={args.ncols}: `{base_tiles}`")
    print()
    print("| candidate | projected shared | shared pct | tile-count ratio | unlock <=32KiB | verdict |")
    print("| --- | ---: | ---: | ---: | --- | --- |")

    # Pack 16 float scales per row into half. Keep the +4 int padding so the MMA
    # stride still has the required `% 8 == 4` bank-conflict padding.
    packed_scale_stride = 2 * MMQ_TILE_NE_K + (MMQ_TILE_NE_K // 2) // 2 + 4
    x96_half_shared = q3_x_bytes(res.mmq_y, packed_scale_stride) + q8_y_bytes(res.mmq_x) + misc
    print_row(
        "pack q3 scales to half at x96",
        x96_half_shared,
        ntiles(args.ncols, res.mmq_x) / base_tiles,
        x96_half_shared <= SMEM_HALF,
        "weak: saves shared but still one block/SM",
    )

    # RDNA4-only compact layout: keep 64 int quant payload, store the 16
    # precomputed scales as half, and drop the old +4 int Q3 padding. This
    # changes the row stride from 84 to 72 ints.
    x96_half_compact_stride = 2 * MMQ_TILE_NE_K + (MMQ_TILE_NE_K // 2) // 2
    x96_half_compact_shared = q3_x_bytes(res.mmq_y, x96_half_compact_stride) + q8_y_bytes(res.mmq_x) + misc
    print_row(
        "pack q3 scales to half compact x96",
        x96_half_compact_shared,
        ntiles(args.ncols, res.mmq_x) / base_tiles,
        x96_half_compact_shared <= SMEM_HALF,
        "proceed: can unlock 2 blocks/SM; risk scale conversion + padding change",
    )

    x80_half_shared = q3_x_bytes(res.mmq_y, packed_scale_stride) + q8_y_bytes(80) + misc
    print_row(
        "pack q3 scales + force x80",
        x80_half_shared,
        ntiles(args.ncols, 80) / base_tiles,
        x80_half_shared <= SMEM_HALF,
        "risky: 2-block possible but 50% more x tiles",
    )

    print_row(
        "pair two MMA k steps (k4 -> k8)",
        res.shared,
        1.0,
        res.shared <= SMEM_HALF,
        "proceed: same geometry, halves outer k loop/dB loads",
    )

    print()
    print("Decision:")
    print("- Do not test x96 half-scale first: it cannot unlock a second block and has conversion risk.")
    print("- Do not test x80 packing first unless compact x96 fails: it needs a large occupancy win to offset 3 vs 2 x tiles.")
    print("- After k-step pairing failed at runtime, compact x96 is the next plausible Q3_K shared-footprint probe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
