#!/usr/bin/env python3
"""Admission gate for post-D059 Q4 C2 hypothesis reboot.

Theory-only script. It screens new hypothesis candidates by projected payload
fit against the D052 corridor and projected decode complexity against D058
budget, then emits an admission table for the next analytical cycle.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Candidate:
    id: str
    name: str
    projected_payload_bpw: float
    projected_complexity_index: float
    maturity: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        default=f"q4c2-reopen-admission-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Output label",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path("build_logs") / "agent-workload"),
        help="Output directory",
    )
    parser.add_argument(
        "--corridor-max-bpw",
        type=float,
        default=3.7701,
        help="D052 required payload upper bound",
    )
    parser.add_argument(
        "--complexity-budget",
        type=float,
        default=1.35,
        help="D058 decode complexity hard budget",
    )
    return parser.parse_args()


def decision(c: Candidate, max_bpw: float, max_cidx: float) -> str:
    fits_bpw = c.projected_payload_bpw <= max_bpw
    fits_cidx = c.projected_complexity_index <= max_cidx
    if fits_bpw and fits_cidx:
        return "admit_research"
    if fits_bpw and not fits_cidx:
        return "park_complexity"
    if (not fits_bpw) and fits_cidx:
        return "park_compression"
    return "reject_pre_gate"


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # These are projected pre-gate targets for a new theory cycle, not measured outcomes.
    candidates = [
        Candidate("H49", "context-conditioned entropy pages", 3.7400, 1.33, "concept"),
        Candidate("H50", "bounded-rANS with deterministic micropages", 3.7600, 1.29, "concept"),
        Candidate("H51", "superblock graph remap + selective literal lanes", 3.7900, 1.27, "concept"),
        Candidate("H52", "hierarchical tuple-context dictionary", 3.7000, 1.48, "concept"),
    ]

    rows = []
    for c in candidates:
        d = decision(c, args.corridor_max_bpw, args.complexity_budget)
        rows.append(
            {
                **asdict(c),
                "compression_margin_bpw": args.corridor_max_bpw - c.projected_payload_bpw,
                "complexity_margin": args.complexity_budget - c.projected_complexity_index,
                "decision": d,
            }
        )

    admitted = [r["id"] for r in rows if r["decision"] == "admit_research"]

    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    summary = {
        "timestamp": ts,
        "label": args.label,
        "corridor_max_bpw": args.corridor_max_bpw,
        "complexity_budget": args.complexity_budget,
        "notes": [
            "Projected values are planning priors, not measured evidence.",
            "Admission means the hypothesis is allowed into the next analytical checkpoint queue.",
        ],
        "candidates": rows,
        "admitted": admitted,
        "parked_or_rejected": [r["id"] for r in rows if r["decision"] != "admit_research"],
    }

    json_path = out_dir / f"{args.label}.q4_c2_reopen_admission.json"
    md_path = out_dir / f"{args.label}.q4_c2_reopen_admission.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Q4 C2 Reopen Admission Gate: {args.label}",
        "",
        f"- timestamp: {ts}",
        f"- corridor max bpw: {args.corridor_max_bpw:.4f}",
        f"- complexity budget: {args.complexity_budget:.4f}",
        "",
        "Projected pre-gate screen (planning priors only):",
        "",
        "| ID | Name | projected bpw | bpw margin | projected complexity | complexity margin | decision |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for r in rows:
        lines.append(
            "| "
            f"{r['id']} | {r['name']} | {r['projected_payload_bpw']:.4f} | "
            f"{r['compression_margin_bpw']:.4f} | {r['projected_complexity_index']:.4f} | "
            f"{r['complexity_margin']:.4f} | {r['decision']} |"
        )

    lines.extend(
        [
            "",
            f"Admitted to next analytical cycle: {', '.join(admitted) if admitted else 'none'}.",
            "",
            "Warning: this file is not runtime proof and not a speed/compression claim.",
        ]
    )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path.as_posix()}")
    print(f"Wrote {md_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
