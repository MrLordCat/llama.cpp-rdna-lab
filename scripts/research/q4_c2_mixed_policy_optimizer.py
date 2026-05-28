#!/usr/bin/env python3
"""Ck-5 analytical mixed-policy optimizer for Q4 C2 candidates.

Explores grid mixtures over candidate payload bpw and complexity index to find
feasible policies under corridor and complexity constraints.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Route:
    id: str
    payload_bpw: float
    complexity_index: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        default=f"q4c2-mixed-policy-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Output label",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path("build_logs") / "agent-workload"),
        help="Output directory",
    )
    parser.add_argument(
        "--corridor-min-bpw",
        type=float,
        default=3.57,
        help="Required lower corridor bound from D052",
    )
    parser.add_argument(
        "--corridor-max-bpw",
        type=float,
        default=3.7701,
        help="Required upper corridor bound from D052",
    )
    parser.add_argument(
        "--max-complexity-index",
        type=float,
        default=1.35,
        help="Ck-4 hard complexity budget",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Share grid step (0,1], e.g. 0.05 for 5%",
    )
    return parser.parse_args()


def frange_01(step: float) -> list[float]:
    n = max(1, int(round(1.0 / step)))
    return [i / n for i in range(n + 1)]


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Payload and complexity values from D055/D057/D058 outputs.
    routes = [
        Route("H45", payload_bpw=3.864885, complexity_index=1.5125),
        Route("H46", payload_bpw=3.999999, complexity_index=1.2570),
        Route("H47", payload_bpw=4.500000, complexity_index=1.7040),
        Route("FALLBACK_Q4", payload_bpw=4.000000, complexity_index=1.0000),
    ]

    vals = frange_01(args.step)
    feasible = []
    best_bpw = None
    best_combo = None

    for s_h45, s_h46, s_h47 in itertools.product(vals, vals, vals):
        used = s_h45 + s_h46 + s_h47
        if used > 1.0 + 1e-9:
            continue
        s_fallback = max(0.0, 1.0 - used)

        shares = {
            "H45": s_h45,
            "H46": s_h46,
            "H47": s_h47,
            "FALLBACK_Q4": s_fallback,
        }

        payload = sum(r.payload_bpw * shares[r.id] for r in routes)
        complexity = sum(r.complexity_index * shares[r.id] for r in routes)

        in_corridor = (args.corridor_min_bpw <= payload <= args.corridor_max_bpw)
        in_budget = complexity <= args.max_complexity_index

        row = {
            "shares": shares,
            "payload_bpw": payload,
            "complexity_index": complexity,
            "in_corridor": in_corridor,
            "in_budget": in_budget,
        }

        if in_corridor and in_budget:
            feasible.append(row)

        if best_bpw is None or payload < best_bpw:
            best_bpw = payload
            best_combo = row

    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    summary = {
        "timestamp": ts,
        "label": args.label,
        "corridor_min_bpw": args.corridor_min_bpw,
        "corridor_max_bpw": args.corridor_max_bpw,
        "max_complexity_index": args.max_complexity_index,
        "step": args.step,
        "routes": [r.__dict__ for r in routes],
        "feasible_count": len(feasible),
        "best_payload_combo": best_combo,
        "feasible_examples": feasible[:10],
        "decision": "no_feasible_mixed_policy" if len(feasible) == 0 else "feasible_mixed_policy_found",
    }

    json_path = out_dir / f"{args.label}.q4_c2_mixed_policy.json"
    md_path = out_dir / f"{args.label}.q4_c2_mixed_policy.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Q4 C2 Mixed Policy Optimizer: {args.label}",
        "",
        f"- timestamp: {ts}",
        f"- corridor: [{args.corridor_min_bpw:.4f}, {args.corridor_max_bpw:.4f}] bpw",
        f"- max complexity index: {args.max_complexity_index:.4f}",
        f"- grid step: {args.step:.4f}",
        "",
        "## Outcome",
        "",
        f"- feasible policy count: {len(feasible)}",
    ]

    if best_combo is not None:
        b = best_combo
        lines.extend(
            [
                f"- best payload on searched grid: {b['payload_bpw']:.6f} bpw",
                f"- complexity at best payload: {b['complexity_index']:.6f}",
                "- best shares:",
                f"  - H45: {b['shares']['H45']:.2f}",
                f"  - H46: {b['shares']['H46']:.2f}",
                f"  - H47: {b['shares']['H47']:.2f}",
                f"  - FALLBACK_Q4: {b['shares']['FALLBACK_Q4']:.2f}",
            ]
        )

    if len(feasible) == 0:
        lines.extend(
            [
                "",
                "Decision: no feasible mixed policy found on current candidate set.",
                "",
                "Reason: every route has payload bpw above corridor upper bound, so convex mixtures",
                "cannot enter the required band.",
            ]
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path.as_posix()}")
    print(f"Wrote {md_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
