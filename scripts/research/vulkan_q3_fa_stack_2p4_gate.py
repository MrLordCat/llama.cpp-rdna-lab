#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASELINE_TPS = 2.0013
TARGET_TPS = 2.4

ALL_Q3_MS = 5691.67
FA_MS = 693.77
REQUIRED_SAVINGS_MS = 1174.57


@dataclass(frozen=True)
class StackCase:
    fa_speedup: float

    @property
    def fa_savings_ms(self) -> float:
        return FA_MS * (1.0 - 1.0 / self.fa_speedup)

    @property
    def q3_savings_needed_ms(self) -> float:
        return max(REQUIRED_SAVINGS_MS - self.fa_savings_ms, 0.0)

    @property
    def q3_local_needed(self) -> float:
        remaining = ALL_Q3_MS - self.q3_savings_needed_ms
        if remaining <= 0.0:
            return float("inf")
        return ALL_Q3_MS / remaining


def wall_tps_with_stack(q3_speedup: float, fa_speedup: float) -> float:
    q3_savings = ALL_Q3_MS * (1.0 - 1.0 / q3_speedup)
    fa_savings = FA_MS * (1.0 - 1.0 / fa_speedup)
    achieved = min(q3_savings + fa_savings, REQUIRED_SAVINGS_MS)
    speed_fraction = achieved / REQUIRED_SAVINGS_MS
    return BASELINE_TPS + (TARGET_TPS - BASELINE_TPS) * speed_fraction


def write_report(path: Path) -> None:
    cases = [StackCase(speedup) for speedup in (1.0, 1.1, 1.25, 1.5, 2.0, 3.0)]
    q3_speedups = (1.10, 1.15, 1.20, 1.25, 1.30)
    fa_speedups = (1.0, 1.25, 1.5, 2.0)

    lines: list[str] = []
    lines.append("# D032 Vulkan Q3+FA Stack 2.4 Gate")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Baseline: `{BASELINE_TPS:.4f} TPS`; target: `{TARGET_TPS:.4f} TPS`.")
    lines.append(f"- D030 all-Q3 point: `{ALL_Q3_MS:.2f} ms`.")
    lines.append(f"- D010 full-trace FlashAttention point: `{FA_MS:.2f} ms`.")
    lines.append(f"- Required point savings from the D030 gate: `{REQUIRED_SAVINGS_MS:.2f} ms`.")
    lines.append("")
    lines.append("## FA Helps, But Cannot Carry")
    lines.append("")
    lines.append("| FA local speedup | FA point savings | Q3 savings still needed | Q3 local speedup still needed |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for case in cases:
        lines.append(
            f"| `{case.fa_speedup:.2f}x` | `{case.fa_savings_ms:.2f} ms` | "
            f"`{case.q3_savings_needed_ms:.2f} ms` | `{case.q3_local_needed:.4f}x` |"
        )
    lines.append("")
    lines.append("## Stack Sensitivity")
    lines.append("")
    lines.append("Projected TPS if Q3 and FA improvements stack linearly on the D030 point budget:")
    lines.append("")
    header = "| Q3 local speedup | " + " | ".join(f"FA `{fa:.2f}x`" for fa in fa_speedups) + " |"
    lines.append(header)
    lines.append("| ---: |" + " ---: |" * len(fa_speedups))
    for q3 in q3_speedups:
        row = [f"`{q3:.2f}x`"]
        for fa in fa_speedups:
            row.append(f"`{wall_tps_with_stack(q3, fa):.4f}`")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("Do not pivot to FA-only work for the Vulkan `2.4 TPS` target. Even a `2.0x` FA shader-body win still leaves about `827.69 ms` of Q3 point savings, or `1.1700x` local on all-Q3. A plausible stack needs a true Q3_K body/compressed-dot route near `1.20x` local plus a substantial FA win around `1.5x`; smaller Q3 work leaves too much for FA to carry.")
    lines.append("")
    lines.append("D032 therefore keeps Q3_K as the first implementation gate, but allows a Q3+FA stack as the target-closing route once a real Q3 body candidate reaches at least the `~1.18-1.20x` local band in static or point evidence.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    out = root / "build_logs/agent-workload/d032-vulkan-q3-fa-stack-2p4-gate.md"
    write_report(out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())