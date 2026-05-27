#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Baseline:
    label: str
    tps: float
    prompt_tps: float
    decode_tps: float
    prompt_tokens: int
    gen_tokens: int


@dataclass(frozen=True)
class Share:
    route: str
    share: float
    note: str


BASELINE = Baseline(
    label="d012-vulkan-130k-glu-fast-q3quad-bn256-lowtile3-confirm3",
    tps=2.0013,
    prompt_tps=1053.1067,
    decode_tps=42.7233,
    prompt_tokens=7970,
    gen_tokens=16,
)

TARGET_TPS = 2.4

SHARES = [
    Share("dense FFN gate/up only", 0.3491, "D004 corrected dense-FFN route share"),
    Share("dense FFN down only", 0.2461, "D004 corrected dense-FFN route share"),
    Share("dense FFN gate/up + down", 0.5952, "D004 corrected dense-FFN route share"),
    Share("all Q3_K MUL_MAT", 0.8050, "D004 corrected all-Q3 route share"),
    Share("D012 selected q3quad point Q3_K", 0.7765, "D009/D012 point trace: 5691.67 / 7330.07 ms"),
]


def local_speedup_needed(wall_speedup: float, share: float) -> float | None:
    target_time_fraction = 1.0 / wall_speedup
    denominator = target_time_fraction - (1.0 - share)
    if denominator <= 0.0:
        return None
    return share / denominator


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def main() -> int:
    current_wall_s = BASELINE.gen_tokens / BASELINE.tps
    target_wall_s = BASELINE.gen_tokens / TARGET_TPS
    wall_speedup = TARGET_TPS / BASELINE.tps
    prompt_s = BASELINE.prompt_tokens / BASELINE.prompt_tps
    decode_s = BASELINE.gen_tokens / BASELINE.decode_tps
    other_s = current_wall_s - prompt_s - decode_s
    required_prompt_s = target_wall_s - decode_s - other_s
    required_prompt_tps = BASELINE.prompt_tokens / required_prompt_s
    required_prompt_speedup = required_prompt_tps / BASELINE.prompt_tps

    rows: list[list[str]] = []
    for share in SHARES:
        needed = local_speedup_needed(wall_speedup, share.share)
        rows.append([
            share.route,
            pct(share.share),
            "impossible" if needed is None else f"{needed:.3f}x",
            share.note,
        ])

    print("# Vulkan 130k 2.4 TPS Target Gate")
    print()
    print("Inputs:")
    print()
    print(f"- baseline label: `{BASELINE.label}`")
    print(f"- baseline TPS: `{BASELINE.tps:.4f}`")
    print(f"- target TPS: `{TARGET_TPS:.4f}`")
    print(f"- prompt tokens: `{BASELINE.prompt_tokens}`")
    print(f"- generated tokens: `{BASELINE.gen_tokens}`")
    print(f"- baseline prompt eval: `{BASELINE.prompt_tps:.4f} tok/s`")
    print(f"- baseline decode eval: `{BASELINE.decode_tps:.4f} tok/s`")
    print()
    print("Derived target:")
    print()
    print(f"- current wall from aggregate TPS: `{current_wall_s:.4f} s`")
    print(f"- target wall at 2.4 TPS: `{target_wall_s:.4f} s`")
    print(f"- required wall speedup: `{wall_speedup:.4f}x` (`+{(wall_speedup - 1.0) * 100.0:.2f}%`)")
    print(f"- current prompt time estimate: `{prompt_s:.4f} s`")
    print(f"- current decode time estimate: `{decode_s:.4f} s`")
    print(f"- residual overhead estimate: `{other_s:.4f} s`")
    print(f"- prompt eval needed if decode/overhead stay flat: `{required_prompt_tps:.2f} tok/s` (`{required_prompt_speedup:.4f}x`)")
    print()
    print(table(["Touched route", "Wall share", "Local speedup needed", "Source"], rows))
    print()
    print("Decision signal:")
    print()
    print(
        "The 2.4 TPS target is a new topology target, not a promotion-hardening target. "
        "A gate/up-only route now needs about 1.9x local speedup, while the whole dense "
        "FFN path needs about 1.39x and all-Q3 needs about 1.26x. The next Vulkan route "
        "should therefore reopen broad Q3_K/FFN dataflow work, with D012 as the baseline "
        "and the old 2 TPS target treated as already solved."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())