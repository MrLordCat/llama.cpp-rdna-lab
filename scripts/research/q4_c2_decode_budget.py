#!/usr/bin/env python3
"""Analytical Ck-4 decode-complexity budget model for Q4 C2 candidates.

This is a theory-only script. It does not benchmark runtime kernels. It
produces a comparable complexity index from event-rate proxies and combines it
with already measured compression-side evidence from Ck-1..Ck-3.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Candidate:
    id: str
    name: str
    modeled_bpw: float
    extra_bit_ops: float
    table_lookups: float
    branch_events: float
    random_reads: float
    state_updates: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        default=f"q4c2-decode-budget-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Output label",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path("build_logs") / "agent-workload"),
        help="Output directory",
    )
    parser.add_argument(
        "--required-payload-bpw-max",
        type=float,
        default=3.7701,
        help="Upper bound of required payload corridor (from D052)",
    )
    parser.add_argument(
        "--aux-gap-threshold-bpw",
        type=float,
        default=0.12,
        help="Max compression gap to still keep candidate as auxiliary-only",
    )
    parser.add_argument(
        "--hard-complexity-budget",
        type=float,
        default=1.35,
        help="Complexity index budget for primary candidates",
    )
    return parser.parse_args()


def complexity_index(c: Candidate) -> float:
    # Weighted additive proxy around legacy Q4 decode baseline (1.0).
    return (
        1.0
        + 0.18 * c.extra_bit_ops
        + 0.22 * c.table_lookups
        + 0.35 * c.branch_events
        + 0.30 * c.random_reads
        + 0.12 * c.state_updates
    )


def decide(
    modeled_bpw: float,
    required_max: float,
    cidx: float,
    hard_budget: float,
    aux_gap_threshold: float,
    allow_aux: bool,
) -> str:
    gap = modeled_bpw - required_max
    if gap <= 0.0 and cidx <= hard_budget:
        return "candidate_primary"
    if allow_aux and gap > 0.0 and gap <= aux_gap_threshold and cidx <= (hard_budget + 0.30):
        return "auxiliary_only"
    return "reject_primary"


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compression-side modeled bpw values are carried from prior gates:
    # - H45 from D055 global payload entropy floor
    # - H46 from D055 active-symbol fixed-bits expectation
    # - H47 from D057 best full tuple proxy
    candidates = [
        Candidate(
            id="H45",
            name="EBNS",
            modeled_bpw=3.864885,
            extra_bit_ops=0.80,
            table_lookups=0.60,
            branch_events=0.35,
            random_reads=0.20,
            state_updates=0.45,
        ),
        Candidate(
            id="H46",
            name="SPRE",
            modeled_bpw=3.999999,
            extra_bit_ops=0.25,
            table_lookups=0.40,
            branch_events=0.20,
            random_reads=0.08,
            state_updates=0.25,
        ),
        Candidate(
            id="H47",
            name="PDNT",
            modeled_bpw=4.500000,
            extra_bit_ops=0.55,
            table_lookups=0.95,
            branch_events=0.60,
            random_reads=0.40,
            state_updates=0.55,
        ),
    ]

    rows = []
    for c in candidates:
        cidx = complexity_index(c)
        gap = c.modeled_bpw - args.required_payload_bpw_max
        decision = decide(
            modeled_bpw=c.modeled_bpw,
            required_max=args.required_payload_bpw_max,
            cidx=cidx,
            hard_budget=args.hard_complexity_budget,
            aux_gap_threshold=args.aux_gap_threshold_bpw,
            allow_aux=(c.id == "H45"),
        )
        rows.append(
            {
                **asdict(c),
                "complexity_index": cidx,
                "compression_gap_bpw": gap,
                "decision": decision,
            }
        )

    ranked = sorted(rows, key=lambda r: (r["decision"], r["complexity_index"]))

    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    summary = {
        "timestamp": ts,
        "label": args.label,
        "required_payload_bpw_max": args.required_payload_bpw_max,
        "aux_gap_threshold_bpw": args.aux_gap_threshold_bpw,
        "hard_complexity_budget": args.hard_complexity_budget,
        "baseline_decode_complexity_index": 1.0,
        "weights": {
            "extra_bit_ops": 0.18,
            "table_lookups": 0.22,
            "branch_events": 0.35,
            "random_reads": 0.30,
            "state_updates": 0.12,
        },
        "candidates": rows,
        "recommended_keep": [
            r["id"] for r in rows if r["decision"] in {"candidate_primary", "auxiliary_only"}
        ],
        "recommended_reject": [r["id"] for r in rows if r["decision"] == "reject_primary"],
        "ranking_note": "Lower complexity index is better; primary eligibility also requires corridor fit.",
    }

    json_path = out_dir / f"{args.label}.q4_c2_decode_budget.json"
    md_path = out_dir / f"{args.label}.q4_c2_decode_budget.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Q4 C2 Decode Budget: {args.label}",
        "",
        f"- timestamp: {ts}",
        f"- required payload bpw max: {args.required_payload_bpw_max:.4f}",
        f"- hard complexity budget: {args.hard_complexity_budget:.2f}",
        "",
        "## Candidate Table",
        "",
        "| ID | Name | Modeled bpw | Compression gap bpw | Complexity index | Decision |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            "| "
            f"{r['id']} | {r['name']} | {r['modeled_bpw']:.6f} | {r['compression_gap_bpw']:.6f} | "
            f"{r['complexity_index']:.6f} | {r['decision']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `candidate_primary`: fits corridor and complexity budget.",
            "- `auxiliary_only`: corridor miss is small, complexity acceptable only as mixed-policy assist.",
            "- `reject_primary`: corridor/complexity profile does not justify primary use.",
            "",
            "This output is analytical and must not be interpreted as measured runtime speed.",
        ]
    )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path.as_posix()}")
    print(f"Wrote {md_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
