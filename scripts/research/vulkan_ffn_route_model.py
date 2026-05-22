#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace


BYTES_F16 = 2
BYTES_F32 = 4


@dataclass(frozen=True)
class Tile:
    name: str
    bm: int = 128
    bn: int = 128
    bk: int = 32
    wm: int = 64
    wn: int = 64
    tm: int = 16
    tn: int = 16
    warp: int = 64

    def valid(self) -> bool:
        if min(self.bm, self.bn, self.bk, self.wm, self.wn, self.tm, self.tn, self.warp) <= 0:
            return False
        if self.bm % self.wm or self.bn % self.wn:
            return False
        if self.wm % self.tm or self.wn % self.tn:
            return False
        storestride = self.warp // self.tm
        return self.warp % self.tm == 0 and storestride <= self.tn and self.tn % storestride == 0

    def prepared_block(self) -> int | None:
        if not self.valid():
            return None
        return self.warp * (self.bm // self.wm) * (self.bn // self.wn)

    def wg_count(self, m: int, n: int) -> int:
        return math.ceil(m / self.bm) * math.ceil(n / self.bn)

    def q3_stride(self) -> int:
        return self.bk // 2 + 2

    def coopmat_stage_bytes(self) -> int:
        block = self.prepared_block() or self.warp
        warps = max(block // self.warp, 1)
        return self.tm * self.tn * warps * BYTES_F16

    def single_q3_lds(self) -> int:
        return (self.bm + self.bn) * self.q3_stride() * 4 + self.coopmat_stage_bytes()

    def dual_a_q3_lds(self) -> int:
        return (2 * self.bm + self.bn) * self.q3_stride() * 4 + self.coopmat_stage_bytes()

    def accumulator_fragments(self) -> int:
        return (self.wm // self.tm) * (self.wn // self.tn)

    def b_reload_bytes(self, m: int, n: int, k: int) -> int:
        return self.wg_count(m, n) * self.bn * k * BYTES_F16

    def a_pair_dequants(self, m: int, n: int, k: int) -> int:
        return self.wg_count(m, n) * self.bm * k // 2

    def output_bytes(self, m: int, n: int) -> int:
        return m * n * BYTES_F32


VARIANTS: dict[str, dict[str, int]] = {
    "base": {},
    "wn32": {"wn": 32},
    "bk16": {"bk": 16},
    "bk16-wn32": {"bk": 16, "wn": 32},
    "bn64": {"bn": 64},
    "bn256": {"bn": 256},
    "bm64": {"bm": 64},
    "wm128-wn32": {"wm": 128, "wn": 32},
}

MEASURED_NOTES: dict[str, str] = {
    "base": "current accepted Q3_K route: 113 VGPR / 45 SGPR / 20480 B LDS / 0 scratch",
    "wn32": "single-matmul WN32 was rejected in E085/E091; use only if dual accumulators require smaller WN",
    "bk16": "static-only; halves LDS pressure but doubles K blocks/barriers",
    "bk16-wn32": "static-only resource-relief profile; likely slower unless dual accumulator pressure dominates",
    "bn64": "single-route BN64 rejected; doubles workgroups and A dequant proxy",
    "bn256": "single-route BN256 rejected in E098 despite halving A dequant proxy; dual-A LDS exceeds 32 KiB",
    "bm64": "single-route BM64 rejected; doubles workgroups/B reload",
    "wm128-wn32": "single-route rejected in E085; accumulator relief without work reduction",
}


def apply_variant(name: str) -> Tile:
    if name not in VARIANTS:
        raise SystemExit(f"ERROR: unknown variant {name!r}")
    return replace(Tile(name=name), **VARIANTS[name])


def parse_shape(text: str) -> tuple[int, int, int]:
    parts = [int(p.strip()) for p in text.lower().replace("x", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise SystemExit("ERROR: --shape must be MxNxK")
    return parts[0], parts[1], parts[2]


def mib(value: int | float) -> float:
    return value / (1024 * 1024)


def speedup_from_fraction_removed(fraction_removed: float) -> float:
    if fraction_removed >= 1.0:
        return float("inf")
    return 1.0 / (1.0 - fraction_removed)


def route_speedup(wall_share: float, local_speedup: float) -> float:
    return 1.0 / ((1.0 - wall_share) + wall_share / local_speedup)


def required_local_speedup(wall_share: float, target_total_speedup: float) -> float | None:
    denom = 1.0 / target_total_speedup - (1.0 - wall_share)
    if denom <= 0.0:
        return None
    return wall_share / denom


def row(tile: Tile, m: int, n: int, k: int, wall_share: float) -> str:
    block = tile.prepared_block()
    valid = block is not None and block <= 1024
    wg = tile.wg_count(m, n)

    current_b = 2 * tile.b_reload_bytes(m, n, k)
    fused_b = tile.b_reload_bytes(m, n, k)
    current_a_pairs = 2 * tile.a_pair_dequants(m, n, k)
    fused_a_pairs = current_a_pairs
    current_a_lds = current_a_pairs * 4
    current_intermediate = 2 * tile.output_bytes(m, n)
    fused_intermediate = 0
    glu_read_write = 3 * tile.output_bytes(m, n)

    removable_bytes = (current_b - fused_b) + current_intermediate + glu_read_write
    removable_only_proxy = removable_bytes / (current_b + current_intermediate + glu_read_write)
    optimistic_mem_ceiling = speedup_from_fraction_removed(removable_only_proxy)
    current_proxy_with_a = current_b + current_a_lds + current_intermediate + glu_read_write
    removable_with_a_proxy = removable_bytes / current_proxy_with_a if current_proxy_with_a else 0.0
    local_proxy_ceiling = speedup_from_fraction_removed(removable_with_a_proxy)
    wall_from_proxy_ceiling = route_speedup(wall_share, local_proxy_ceiling)

    dual_frag = 2 * tile.accumulator_fragments()
    lds = tile.dual_a_q3_lds()
    resource = "ok"
    if not valid:
        resource = "invalid"
    elif lds > 32 * 1024:
        resource = "lds>32k"
    elif lds >= 30 * 1024:
        resource = "near-32k"
    if dual_frag > 24:
        resource += "+accum-high"

    return (
        f"| {tile.name} | {valid} | {block or 0} | {tile.bm}x{tile.bn}x{tile.bk} | {tile.wm}x{tile.wn} | "
        f"{tile.single_q3_lds()} | {lds} | {tile.accumulator_fragments()} -> {dual_frag} | {wg} | "
        f"{mib(current_b):.1f} -> {mib(fused_b):.1f} | {current_a_pairs / 1_000_000:.1f} -> {fused_a_pairs / 1_000_000:.1f} | "
        f"{mib(current_a_lds):.1f} | {mib(removable_bytes):.1f} | {optimistic_mem_ceiling:.3f}x | {local_proxy_ceiling:.3f}x | {wall_from_proxy_ceiling:.3f}x | {resource} | {MEASURED_NOTES.get(tile.name, '')} |"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model Vulkan dense FFN Q3_K dual-A/same-B route ceiling and resource risk")
    parser.add_argument("--shape", default="17408x1024x5120", help="gate/up FFN shape as MxNxK")
    parser.add_argument("--variants", default="base,wn32,bk16,bk16-wn32,bn64,bn256,bm64,wm128-wn32")
    parser.add_argument("--baseline-tps", type=float, default=1.3406)
    parser.add_argument("--target-tps", type=float, default=1.5545)
    parser.add_argument("--wall-share", type=float, default=0.2491, help="wall share for dense FFN gate/up Q3_K route")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.wall_share < 1.0:
        raise SystemExit("ERROR: --wall-share must be in (0,1)")
    m, n, k = parse_shape(args.shape)
    target_total = args.target_tps / args.baseline_tps
    required = required_local_speedup(args.wall_share, target_total)

    print("# Vulkan Dense FFN Q3_K Route Model")
    print()
    print(f"- shape: M={m}, N={n}, K={k}")
    print(f"- baseline_tps: {args.baseline_tps:.4f}")
    print(f"- target_tps: {args.target_tps:.4f}")
    print(f"- target_total_speedup: {target_total:.4f}x")
    if required is None:
        print("- required_local_speedup_for_this_route_alone: unreachable")
    else:
        print(f"- required_local_speedup_for_this_route_alone: {required:.3f}x")
    print("- model scope: gate/up pair only; current route is two Q3_K matmuls plus GLU intermediates")
    print("- fused route assumption: one B/activation tile feeds gate and up accumulators, two A/Q3 tiles are still dequantized, and GLU output is written directly")
    print()
    print("| variant | valid | block | BMxBNxBK | WMxWN | single_lds_B | dual_a_lds_B | acc_fragments | wg | B_reload_MiB current->fused | A_pair_dequants_M current->fused | unchanged_A_LDS_MiB | removable_proxy_MiB | optimistic_mem_only_ceiling | local_ceiling_with_A_proxy | wall_speedup_with_A_proxy | resource_gate | notes |")
    print("|---|:---:|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        print(row(apply_variant(name), m, n, k, args.wall_share))

    print()
    print("## Interpretation")
    print()
    print("- `optimistic_mem_only_ceiling` intentionally ignores unchanged A dequant and coopmat work; it is a hard upper bound, not a speed prediction.")
    print("- `local_ceiling_with_A_proxy` adds unchanged A LDS writes as a simple fixed-work proxy. Real speed can be lower because coopmat arithmetic and Q3 decode ALU are also unchanged.")
    print("- Dual-A/same-B fusion does not remove Q3_K A dequant repetition; it mainly removes one B reload, two intermediate writes, GLU reads, and the GLU write.")
    print("- A useful full-lane result therefore needs either resource-safe dual accumulators plus measurable memory relief, or a separate Q3_K layout/repack route that reduces repeated A dequant across N-blocks.")
    print("- Variants marked `lds>32k` should not be built without a new shader layout. Variants marked `accum-high` need driver pipeline stats before any server benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
