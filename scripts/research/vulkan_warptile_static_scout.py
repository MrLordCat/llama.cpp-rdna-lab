#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Warptile:
    label: str
    block_size: int = 256
    bm: int = 128
    bn: int = 128
    bk: int = 32
    wm: int = 64
    wn: int = 64
    wmiter: int = 2
    tm: int = 16
    tn: int = 16
    tk: int = 16
    warp: int = 64

    def prepared_block_size(self) -> int | None:
        if not self.layout_valid():
            return None
        return self.warp * (self.bm // self.wm) * (self.bn // self.wn)

    def prepared(self) -> Warptile | None:
        block = self.prepared_block_size()
        if block is None or block > 1024:
            return None
        return replace(self, block_size=block)

    def signature(self) -> tuple[int, int, int, int, int, int, int, int, int, int, int]:
        return (
            self.block_size,
            self.bm,
            self.bn,
            self.bk,
            self.wm,
            self.wn,
            self.wmiter,
            self.tm,
            self.tn,
            self.tk,
            self.warp,
        )

    def layout_valid(self) -> bool:
        if min(self.bm, self.bn, self.wm, self.wn, self.tm, self.tn, self.warp) <= 0:
            return False
        if self.bm % self.wm != 0 or self.bn % self.wn != 0:
            return False
        if self.wm % self.tm != 0 or self.wn % self.tn != 0:
            return False
        storestride = self.warp // self.tm
        if storestride == 0 or self.warp % self.tm != 0:
            return False
        if storestride > self.tn or self.tn % storestride != 0:
            return False
        return True

    def coopmat_stage_bytes(self) -> int:
        warps = max(self.block_size // self.warp, 1)
        return self.tm * self.tn * warps * 2

    def shmem_bytes_generic(self) -> int:
        # Mirrors ggml_vk_matmul_shmem_support for fp16 + coopmat, excluding tiny rounded effects.
        bank_conflict_offset = 8
        type_size = 2
        warps = max(self.block_size // self.warp, 1)
        load_bufs = (self.bm + self.bn) * (self.bk + bank_conflict_offset) * type_size
        coopmat_stage = max((self.tm * self.tn // warps) * 4, 0)
        return load_bufs + coopmat_stage

    def shmem_bytes_q3_stride(self) -> int:
        # Actual E082+ Q3 shader buffer: shared f16vec2 buf_a/b with stride BK/2+2.
        stride = self.bk // 2 + 2
        load_bufs = (self.bm + self.bn) * stride * 4
        return load_bufs + self.coopmat_stage_bytes()

    def wg_count(self, m: int, n: int) -> int:
        return math.ceil(m / self.bm) * math.ceil(n / self.bn)

    def b_reload_bytes(self, m: int, n: int) -> int:
        # B tile is f16 and is loaded per M-block/N-block workgroup for one BK slice.
        return self.wg_count(m, n) * self.bn * self.bk * 2

    def b_reload_bytes_full_k(self, m: int, n: int, k: int) -> int:
        # Full-K B traffic is mostly independent of BK; BK changes loop/barrier cadence.
        return self.wg_count(m, n) * self.bn * k * 2

    def a_pair_dequants(self, m: int, n: int) -> int:
        # A dequant pairs per workgroup tile for one BK slice, multiplied by workgroups.
        return self.wg_count(m, n) * self.bm * self.bk // 2

    def a_pair_dequants_full_k(self, m: int, n: int, k: int) -> int:
        return self.wg_count(m, n) * self.bm * k // 2

    def k_blocks(self, k: int) -> int:
        return math.ceil(k / self.bk)

    def load_mapping_valid(self, load_vec_a: int = 4, load_vec_b: int = 8) -> bool:
        if self.bk % load_vec_a != 0 or self.bk % load_vec_b != 0:
            return False
        a_load_cols = self.bk // load_vec_a
        b_load_cols = self.bk // load_vec_b
        if a_load_cols == 0 or b_load_cols == 0:
            return False
        if self.block_size % a_load_cols != 0 or self.block_size % b_load_cols != 0:
            return False
        loadstride_a = self.block_size * load_vec_a // self.bk
        loadstride_b = self.block_size * load_vec_b // self.bk
        if loadstride_a == 0 or loadstride_b == 0:
            return False
        return self.bm % loadstride_a == 0 and self.bn % loadstride_b == 0


VARIANT_PATCHES = {
    "base": {},
    "wn16": {"wn": 16},
    "wn32": {"wn": 32},
    "wn48": {"wn": 48},
    "wn64": {"wn": 64},
    "wn96": {"wn": 96},
    "wn128": {"wn": 128},
    "bk16": {"bk": 16},
    "bk64": {"bk": 64},
    "wm128": {"wm": 128},
    "wm128-wn32": {"wm": 128, "wn": 32},
    "bm64": {"bm": 64},
    "bn64": {"bn": 64},
    "bm64-bn64": {"bm": 64, "bn": 64},
    "bm256": {"bm": 256},
    "bn192": {"bn": 192},
    "bn192-wn96": {"bn": 192, "wn": 96},
    "bn192-wm128-wn96": {"bn": 192, "wm": 128, "wn": 96},
    "bn256": {"bn": 256},
    "bn512": {"bn": 512},
    "bn256-wn128": {"bn": 256, "wn": 128},
    "bn256-wm128": {"bn": 256, "wm": 128},
    "bm256-bn256": {"bm": 256, "bn": 256},
    "block128": {"block_size": 128},
    "block128-bn64": {"block_size": 128, "bn": 64},
    "block128-wm128": {"block_size": 128, "wm": 128},
    "wm32-wn32": {"bm": 64, "bn": 64, "wm": 32, "wn": 32},
}

MEASURED = {
    "base": "E086 pp r3 961.82; E091 workload base reference 6.6277",
    "wn16": "E085/E091 rejected; E085 771.75",
    "wn32": "E085 rejected; 915.98",
    "wn48": "E091 pp r3 972.31, workload r3 6.7981; E093 marks invalid for BN=128, do not promote without proving active layout",
    "wn96": "E091 r1 975.72; E093 marks invalid for BN=128, do not promote without proving active layout",
    "bk16": "E144 rejected; 587.52 vs 972.77 base despite 70 VGPR / 12288 B LDS",
    "bk64": "E144 static reject for current device; Q3 shader LDS 36864 B exceeds 32 KiB limit",
    "wm128-wn32": "E085 rejected; 892.75",
    "bm64": "E091 rejected; 797.92",
    "bn64": "E091 rejected; 720.07/737.21",
    "bm64-bn64": "E091 rejected; 646.10",
    "bm256": "E146 rejected; 916.62 vs 972.84 base, 94 VGPR / 31744 B LDS despite halved B reload proxy",
    "bn192": "static-only: reduces A dequant proxy by about 25% for N=1024; requires LDS/resource proof",
    "bn192-wn96": "E143 rejected; 760.78 vs 974.19 base, 139 VGPR / 25088 B LDS",
    "bn192-wm128-wn96": "E143 rejected; 137.71, 171 VGPR plus scratch",
    "bn256": "E098 rejected; 947.12 vs 983.21 base, high LDS/register pressure",
    "bn512": "D082 rejected: projected 54272 B LDS exceeds the usable route limit; runtime stayed on medium Q3 and measured 950.35 tok/s",
    "bn256-wn128": "E098/E143 rejected; E143 659.02, 165 VGPR / 29696 B LDS",
    "bn256-wm128": "E098/E143 rejected; E143 660.97, 165 VGPR / 29696 B LDS",
    "bm256-bn256": "static-only: likely over LDS limit; included as negative resource bound",
    "block128": "E091 r1 970.12; close/no workload confirmation",
    "block128-bn64": "E091 rejected; 736.65",
    "block128-wm128": "E091 rejected; 903.45",
    "wm32-wn32": "E075 invalid/corrupt undercoverage; do not promote",
}


def apply_variant(name: str) -> Warptile:
    if name not in VARIANT_PATCHES:
        raise SystemExit(f"ERROR: unknown variant {name!r}")
    tile = Warptile(label=name)
    return replace(tile, **VARIANT_PATCHES[name])


def parse_shapes(text: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for chunk in text.split(","):
        if not chunk.strip():
            continue
        left, right = chunk.lower().split("x", 1)
        shapes.append((int(left.strip()), int(right.strip())))
    return shapes


def row_for_variant(name: str, shapes: list[tuple[int, int]], base_signature: tuple[int, ...], k_size: int) -> str:
    tile = apply_variant(name)
    base_prepared = apply_variant("base").prepared()
    if base_prepared is None:
        raise SystemExit("ERROR: base warptile is invalid")
    prepared = tile.prepared()
    valid = prepared is not None
    effective = prepared if prepared is not None else base_prepared
    load_valid = effective.load_mapping_valid() if valid else True
    runtime_effective = "variant" if valid and load_valid else "base-fallback" if not valid else "invalid-load-map"
    same_as_base = effective.signature() == base_signature
    wg = "/".join(str(effective.wg_count(m, n)) for m, n in shapes)
    k_blocks = effective.k_blocks(k_size)
    barriers_m = sum(effective.wg_count(m, n) * k_blocks * 2 for m, n in shapes) / 1_000_000
    b_mb = sum(effective.b_reload_bytes_full_k(m, n, k_size) for m, n in shapes) / (1024 * 1024)
    a_pairs_m = sum(effective.a_pair_dequants_full_k(m, n, k_size) for m, n in shapes) / 1_000_000
    measured = MEASURED.get(name, "-")
    return (
        f"| {name} | {'yes' if valid else 'no'} | {'yes' if load_valid else 'no'} | {runtime_effective} | {'yes' if same_as_base else 'no'} | {effective.block_size} | "
        f"{effective.bm}x{effective.bn}x{effective.bk} | {effective.wm}x{effective.wn} | "
        f"{effective.shmem_bytes_q3_stride()} | {effective.shmem_bytes_generic()} | {wg} | {k_blocks} | {barriers_m:.2f} | {b_mb:.2f} | {a_pairs_m:.2f} | {measured} |"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static scout for Vulkan coopmat warptile variants")
    parser.add_argument(
        "--variants",
        default=",".join(VARIANT_PATCHES.keys()),
        help="comma-separated variants to inspect",
    )
    parser.add_argument(
        "--shapes",
        default="17408x1024,5120x1024",
        help="comma-separated MxN matmul shapes for workgroup/reload proxy",
    )
    parser.add_argument("--k-size", type=int, default=5120, help="representative K dimension for full-K traffic and K-loop proxy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shapes = parse_shapes(args.shapes)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    base_prepared = apply_variant("base").prepared()
    if base_prepared is None:
        raise SystemExit("ERROR: base warptile is invalid")
    base_signature = base_prepared.signature()

    print("# Vulkan Warptile Static Scout")
    print()
    print(f"- shapes: {', '.join(f'{m}x{n}' for m, n in shapes)}")
    print(f"- k_size: {args.k_size}")
    print("- model: validates ggml_vk_matmul_prepare_variant_warptile layout and estimates workgroup/load proxies; no build or benchmark")
    print("- assumption: RX 9070 XT KHR coopmat runtime uses subgroup 64 and 16x16x16 cooperative matrix shape")
    print()
    print("| variant | valid_layout | valid_load_map | runtime_effective | same_as_base | prepared_block | BMxBNxBK | WMxWN | q3_shader_lds_bytes | backend_guard_lds_bytes | wg_counts | k_blocks | barrier_rounds_M | full_B_reload_MiB | full_A_pair_dequants_M | measured_note |")
    print("|---|:---:|:---:|---|:---:|---:|---|---|---:|---:|---|---:|---:|---:|---:|---|")
    for name in variants:
        print(row_for_variant(name, shapes, base_signature, args.k_size))

    print()
    print("## Interpretation")
    print()
    print("- Invalid layouts are modeled as `runtime_effective=base-fallback`, matching the backend restore-to-base path; one-off speedups there should be treated as noise unless logs prove otherwise.")
    print("- Effective variants marked `same_as_base=yes` should be treated as measurement noise unless a backend log proves a hidden route difference.")
    print("- BK variants do not reduce full-K dequant/B traffic in this model; they mainly trade K-loop/barrier cadence against LDS footprint and compiler resources.")
    print("- `valid_load_map=no` means the current `mul_mm.comp` load-loop mapping would overshoot the shared tile for Q3_K `LOAD_VEC_A=4` / aligned B `LOAD_VEC_B=8`; do not benchmark it without shader bounds/mapping changes.")
    print("- Variants that only reduce block size without reducing workgroup count are low-ceiling unless measured repeatedly positive.")
    print("- Large changes in B reload proxy or workgroup count are better candidates than helper-only shader rewrites, but still need output sanity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
