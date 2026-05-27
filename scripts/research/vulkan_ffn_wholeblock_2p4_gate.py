#!/usr/bin/env python3
from __future__ import annotations

import math


HIDDEN = 17408
MODEL = 5120
NCOLS = 256
PREFILL_LAYERS = 63
HIDDEN_TILE = 128
DOWN_ROW_TILE = 64
BYTES_F32 = 4

BASELINE_TPS = 2.0013
TARGET_TPS = 2.4

# D009/D012 point trace after bn256 + lowtile3 + q3quad, before final GLU wall
# confirmation. These are parsed route totals across the active prompt graph.
GATE_UP_MS = 2759.96
DOWN_MS = 1417.34
DENSE_FFN_MS = GATE_UP_MS + DOWN_MS


def mib(value: int | float) -> float:
    return float(value) / float(1 << 20)


def gib(value: int | float) -> float:
    return float(value) / float(1 << 30)


def local_speedup_needed(wall_speedup: float, share: float) -> float:
    return share / (1.0 / wall_speedup - (1.0 - share))


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def main() -> int:
    wall_speedup = TARGET_TPS / BASELINE_TPS
    dense_ffn_share = 0.5952
    required_local = local_speedup_needed(wall_speedup, dense_ffn_share)
    required_dense_time_ms = DENSE_FFN_MS / required_local
    required_savings_ms = DENSE_FFN_MS - required_dense_time_ms

    hidden_bytes = HIDDEN * NCOLS * BYTES_F32
    hidden_write_read = 2 * hidden_bytes
    hidden_write_read_all_layers = hidden_write_read * PREFILL_LAYERS

    output_bytes = MODEL * NCOLS * BYTES_F32
    hidden_tiles = math.ceil(HIDDEN / HIDDEN_TILE)
    down_row_tiles = math.ceil(MODEL / DOWN_ROW_TILE)
    partial_output_rw_per_layer = hidden_tiles * 2 * output_bytes
    partial_output_rw_all_layers = partial_output_rw_per_layer * PREFILL_LAYERS
    recompute_lower_bound_ms = GATE_UP_MS * down_row_tiles + DOWN_MS

    traffic_rows = [
        [
            "GLU hidden materialization write+read",
            f"{mib(hidden_write_read):.2f} MiB/layer",
            f"{gib(hidden_write_read_all_layers):.2f} GiB / {PREFILL_LAYERS} layers",
            "small unless traffic is pathologically slow",
        ],
        [
            f"Hidden-tile partial output R/W, tile={HIDDEN_TILE}",
            f"{gib(partial_output_rw_per_layer):.2f} GiB/layer",
            f"{gib(partial_output_rw_all_layers):.2f} GiB / {PREFILL_LAYERS} layers",
            "too much traffic for a speed route",
        ],
    ]

    route_rows = [
        [
            "Activation-only whole-FFN fusion",
            f"save <= {gib(hidden_write_read_all_layers):.2f} GiB traffic",
            f"needs {required_savings_ms:.1f} ms dense-FFN savings",
            "insufficient ceiling",
        ],
        [
            f"Full streaming with gate/up recompute per {DOWN_ROW_TILE} down rows",
            f"{recompute_lower_bound_ms:.1f} ms lower bound",
            f"{DENSE_FFN_MS / recompute_lower_bound_ms:.4f}x local",
            "blocked by recompute",
        ],
        [
            f"Full streaming with partial output per {HIDDEN_TILE} hidden rows",
            f"{gib(partial_output_rw_all_layers):.2f} GiB partial R/W",
            "bandwidth-only gate",
            "blocked by output traffic",
        ],
    ]

    print("# Vulkan Whole-FFN 2.4 TPS Gate")
    print()
    print("Inputs:")
    print()
    print(f"- baseline TPS: `{BASELINE_TPS:.4f}`")
    print(f"- target TPS: `{TARGET_TPS:.4f}`")
    print(f"- required wall speedup: `{wall_speedup:.4f}x`")
    print(f"- dense FFN share: `{dense_ffn_share * 100.0:.2f}%`")
    print(f"- required dense-FFN local speedup: `{required_local:.4f}x`")
    print(f"- D009/D012 gate/up Q3_K point: `{GATE_UP_MS:.2f} ms`")
    print(f"- D009/D012 down Q3_K point: `{DOWN_MS:.2f} ms`")
    print(f"- D009/D012 dense FFN point: `{DENSE_FFN_MS:.2f} ms`")
    print(f"- dense FFN target point time: `{required_dense_time_ms:.2f} ms`")
    print(f"- dense FFN savings needed: `{required_savings_ms:.2f} ms`")
    print()
    print(table(["Traffic item", "Per layer", "Active prefill graph", "Signal"], traffic_rows))
    print()
    print(table(["Route model", "Lower bound / saving", "Local signal", "Decision"], route_rows))
    print()
    print("Decision signal:")
    print()
    print(
        "A non-adjacent whole-FFN route is only worth implementing if it reduces Q3_K "
        "matmul work or creates a new all-Q3 dataflow. Merely avoiding the GLU hidden "
        "activation write/read cannot supply the roughly 1.39x dense-FFN local speedup "
        "needed for 2.4 TPS. Naive full streaming is blocked by either gate/up recompute "
        "or partial-output traffic."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())