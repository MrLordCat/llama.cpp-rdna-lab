#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass


BASELINE_TPS = 2.0013
TARGET_TPS = 2.4
ALL_Q3_SHARE = 0.8050
SELECTED_Q3_SHARE = 0.7765

# D008/D009 point-route totals from the D012-family route ceiling traces.
D008_ALL_Q3_MS = 5876.48
D009_ALL_Q3_MS = 5691.67
D009_SELECTED_Q3_MS = 5691.67

# Measured rejection facts from prior Vulkan layout/body probes.
SIGNEDNIB_CONTROL_TPS = 1.5798
SIGNEDNIB_HOT5_TPS = 1.5186


@dataclass(frozen=True)
class Candidate:
    name: str
    evidence: str
    signal: str
    decision: str


def local_speedup_needed(wall_speedup: float, share: float) -> float:
    return share / (1.0 / wall_speedup - (1.0 - share))


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def main() -> int:
    wall_speedup = TARGET_TPS / BASELINE_TPS
    all_q3_needed = local_speedup_needed(wall_speedup, ALL_Q3_SHARE)
    selected_q3_needed = local_speedup_needed(wall_speedup, SELECTED_Q3_SHARE)
    all_q3_target_ms = D009_ALL_Q3_MS / all_q3_needed
    all_q3_savings_needed_ms = D009_ALL_Q3_MS - all_q3_target_ms
    selected_q3_target_ms = D009_SELECTED_Q3_MS / selected_q3_needed
    selected_q3_savings_needed_ms = D009_SELECTED_Q3_MS - selected_q3_target_ms
    q3quad_local = D008_ALL_Q3_MS / D009_ALL_Q3_MS
    q3quad_savings_ms = D008_ALL_Q3_MS - D009_ALL_Q3_MS
    signednib_local = SIGNEDNIB_HOT5_TPS / SIGNEDNIB_CONTROL_TPS

    candidates = [
        Candidate(
            "Extend current q3quad/tile stack",
            f"D008 -> D009 all-Q3 point `{D008_ALL_Q3_MS:.2f} -> {D009_ALL_Q3_MS:.2f} ms` (`{q3quad_local:.4f}x`, `{q3quad_savings_ms:.2f} ms` saved)",
            f"already in D012; still needs about `{all_q3_savings_needed_ms:.0f} ms` more all-Q3 point savings",
            "reject as next route unless it changes the body beyond q3quad/bn256/lowtile",
        ),
        Candidate(
            "Scale-only metadata or helper reuse",
            "E088 pair-scale helper `-0.20%`; E080 unsigned scale `-0.44%`; E101 scale-int arithmetic negative",
            "even removing a large fraction of repeated scale decode did not improve the route",
            "reject scale-only or expression-only probes",
        ),
        Candidate(
            "Persistent signed-nibble Q3_K layout",
            f"S001 static SPIR-V improved, but runtime hot5 `{SIGNEDNIB_HOT5_TPS:.4f}` vs control `{SIGNEDNIB_CONTROL_TPS:.4f}` (`{signednib_local:.4f}x`); all-Q3 storage failed the 130k fit check",
            "removes some bit unpack but not scale/FMA/coopmat work, and adds residency/conversion pressure",
            "reject as-is; do not reopen without a different compute body",
        ),
        Candidate(
            "Q8_1 / integer-dot prefill route",
            "E099 forced `matmul_q3_k_q8_1_l`: pp256 `225.08`, `143 VGPR / 28672 B LDS`; D006 no-coopmat/Q8 prompt about `400 tok/s`",
            "route switch was real and strongly slower",
            "reject Q8_1/int-dot as a Vulkan P002 prompt speed path",
        ),
        Candidate(
            "Persistent fp16/int8 expanded layouts",
            "P002 layout gate: FFN fp16 `+25.03 GiB`, FFN int8 `+9.09 GiB`; D026 all-Q3 int8+fp16 expansion `+15.42 GiB`",
            "too much residency for the 16 GiB 130k spill-sensitive lane",
            "reject broad expanded layouts",
        ),
        Candidate(
            "Neighbor tile/resource tweaks",
            "D012 already includes the useful `bn256 + lowtile3 + q3quad` stack; lowtile2/4, down split-K 6, m10240 inclusion, vector-return q3quad, and MMVQ-disable were measured negative",
            "tile-only changes do not remove enough Q3_K work and often trade into LDS/VGPR/residency cliffs",
            "reject as first move toward 2.4 TPS",
        ),
    ]

    print("# Vulkan All-Q3 2.4 TPS Gate")
    print()
    print("Inputs:")
    print()
    print(f"- baseline TPS: `{BASELINE_TPS:.4f}`")
    print(f"- target TPS: `{TARGET_TPS:.4f}`")
    print(f"- required wall speedup: `{wall_speedup:.4f}x`")
    print(f"- all-Q3 wall share: `{ALL_Q3_SHARE * 100.0:.2f}%`")
    print(f"- D012 selected Q3 wall share: `{SELECTED_Q3_SHARE * 100.0:.2f}%`")
    print(f"- all-Q3 local speedup needed: `{all_q3_needed:.4f}x`")
    print(f"- D012 selected Q3 local speedup needed: `{selected_q3_needed:.4f}x`")
    print(f"- D009/D012 all-Q3 point: `{D009_ALL_Q3_MS:.2f} ms`")
    print(f"- all-Q3 target point time: `{all_q3_target_ms:.2f} ms`")
    print(f"- all-Q3 point savings needed: `{all_q3_savings_needed_ms:.2f} ms`")
    print(f"- selected-Q3 target point time: `{selected_q3_target_ms:.2f} ms`")
    print(f"- selected-Q3 point savings needed: `{selected_q3_savings_needed_ms:.2f} ms`")
    print()
    print(table(["Candidate family", "Evidence", "Signal", "Decision"], [[c.name, c.evidence, c.signal, c.decision] for c in candidates]))
    print()
    print("Survivor criteria:")
    print()
    print(
        "Any next Vulkan all-Q3 candidate must remove roughly one fifth of the D012 all-Q3 "
        "point time, while preserving the D012 `bn256 + lowtile3 + q3quad + GLU` stack and "
        "the 130k residency contract. Rejected families above are below that bar because they "
        "only rearrange metadata/helper expressions, add too much residency, or switch to a "
        "measured-slower route. A D031 candidate therefore needs a new Q3_K compute body or "
        "layout-body pair that reduces actual matmul/dequant work, not just a storage-only or "
        "tile-neighbor change."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())