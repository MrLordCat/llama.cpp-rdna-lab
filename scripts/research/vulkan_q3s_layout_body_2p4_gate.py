#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASELINE_TPS = 2.0013
TARGET_TPS = 2.4

ALL_Q3_MS = 5691.67
ALL_Q3_TARGET_MS = 4517.10
ALL_Q3_REQUIRED_SAVINGS_MS = ALL_Q3_MS - ALL_Q3_TARGET_MS

S001_BASE_OPS = 1491
S001_SIGNEDNIB_OPS = 1375
S001_BASE_BITWISE = 12
S001_SIGNEDNIB_BITWISE = 11
S001_BASE_SPV_BYTES = 25128
S001_SIGNEDNIB_SPV_BYTES = 23088

SIGNEDNIB_CONTROL_TPS = 1.5798
SIGNEDNIB_HOT5_TPS = 1.5186

CURRENT_ALL_Q3_GIB = 9.815
CURRENT_FFN_Q3_GIB = 6.973
Q3S_RAW_ALL_EXTRA_GIB = 2.980
Q3S_RAW_FFN_EXTRA_GIB = 2.117
Q3S_ALIGNED_ALL_EXTRA_GIB = 4.206
Q3S_ALIGNED_FFN_EXTRA_GIB = 2.988


@dataclass(frozen=True)
class Candidate:
    name: str
    evidence: str
    issue: str
    decision: str


def pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def write_report(path: Path) -> None:
    target_wall = TARGET_TPS / BASELINE_TPS
    op_drop = 1.0 - (S001_SIGNEDNIB_OPS / S001_BASE_OPS)
    bitwise_drop = 1.0 - (S001_SIGNEDNIB_BITWISE / S001_BASE_BITWISE)
    optimistic_savings_ms = ALL_Q3_MS * op_drop
    signednib_runtime = SIGNEDNIB_HOT5_TPS / SIGNEDNIB_CONTROL_TPS

    candidates = [
        Candidate(
            "Q3S compact signed-nibble + packed scales",
            f"S001 static ops `{S001_BASE_OPS} -> {S001_SIGNEDNIB_OPS}` (`{pct(op_drop)}` fewer); bitwise/shift `{S001_BASE_BITWISE} -> {S001_SIGNEDNIB_BITWISE}`; SPIR-V `{S001_BASE_SPV_BYTES} -> {S001_SIGNEDNIB_SPV_BYTES}` bytes",
            f"Even a linear opcode-to-time upper bound saves only `{optimistic_savings_ms:.2f} ms` of all-Q3 point time versus `{ALL_Q3_REQUIRED_SAVINGS_MS:.2f} ms` needed; it still leaves scale multiply/coopmat work intact",
            "reject as target-closing D031 route",
        ),
        Candidate(
            "Runtime hot5 signed-nibble selector",
            f"same-session control `{SIGNEDNIB_CONTROL_TPS:.4f} TPS`; hot5 `{SIGNEDNIB_HOT5_TPS:.4f} TPS` (`{signednib_runtime:.4f}x`)",
            "The only runtime evidence is negative before the stronger D012 q3quad/bn256/lowtile3 stack is considered",
            "reject and do not reopen without a different compute body",
        ),
        Candidate(
            "Raw compact Q3S persistent layout",
            f"D026 all-Q3 extra `+{Q3S_RAW_ALL_EXTRA_GIB:.3f} GiB`; FFN-only extra `+{Q3S_RAW_FFN_EXTRA_GIB:.3f} GiB` over runtime padded Q3_K",
            "Adds residency pressure on the 16 GiB 130k lane while the static ceiling is below the required point savings",
            "reject for 2.4 TPS route",
        ),
        Candidate(
            "Aligned compact Q3S persistent layout",
            f"D026 all-Q3 extra `+{Q3S_ALIGNED_ALL_EXTRA_GIB:.3f} GiB`; FFN-only extra `+{Q3S_ALIGNED_FFN_EXTRA_GIB:.3f} GiB`",
            "The alignment needed for convenient device layout consumes too much extra residency for this spill-sensitive lane",
            "reject for 2.4 TPS route",
        ),
    ]

    lines: list[str] = []
    lines.append("# D031 Vulkan Q3S Layout-Body 2.4 Gate")
    lines.append("")
    lines.append("## Lane")
    lines.append("")
    lines.append("- Model: `Qwen3.6-27B-Q3_K_S`.")
    lines.append("- Backend: Vulkan, cold-first 130k lane, D012 opt-in stack baseline.")
    lines.append(f"- Baseline: `{BASELINE_TPS:.4f} TPS`; target: `{TARGET_TPS:.4f} TPS` (`{target_wall:.4f}x` wall).")
    lines.append(f"- All-Q3 point: `{ALL_Q3_MS:.2f} ms`; target point time: `{ALL_Q3_TARGET_MS:.2f} ms`; savings needed: `{ALL_Q3_REQUIRED_SAVINGS_MS:.2f} ms`.")
    lines.append("")
    lines.append("## Candidate Fence")
    lines.append("")
    lines.append("| Candidate | Evidence | Blocking issue | Decision |")
    lines.append("| --- | --- | --- | --- |")
    for candidate in candidates:
        lines.append(f"| {candidate.name} | {candidate.evidence} | {candidate.issue} | {candidate.decision} |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("Reject compact Q3S/signed-nibble plus predecoded-scale layout-body work as the next Vulkan `2.4 TPS` route. It is the only moderately memory-plausible persistent Q3_K layout family left, but the static ceiling is far below the needed `1174.57 ms` all-Q3 point savings and existing runtime evidence is negative.")
    lines.append("")
    lines.append("D032 should move away from layout-only unpack simplification. The next candidate needs a true Q3_K compute body or compressed-dot route that reduces matrix work itself while preserving the D012 `bn256 + lowtile3 + q3quad + GLU` stack unless it replaces it with a measured point/resource proof.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    out = root / "build_logs/agent-workload/d031-vulkan-q3s-layout-body-2p4-gate.md"
    write_report(out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())