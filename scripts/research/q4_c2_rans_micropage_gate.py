#!/usr/bin/env python3
"""Fast analytical gate for H50 bounded-rANS micropage route.

Uses D055 full entropy as source floor and evaluates whether plausible
micropage/header/coder overhead settings can satisfy D052 corridor.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-symbol-atlas-json",
        required=True,
        help="Path to D055 full symbol atlas json",
    )
    parser.add_argument(
        "--label",
        default=f"q4c2-rans-micropage-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
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
        help="D052 corridor upper bound",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.source_symbol_atlas_json)
    if not src.exists():
        raise FileNotFoundError(f"Source json not found: {src}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(src.read_text(encoding="utf-8"))
    h1 = float(data["global_entropy_bpw"])

    # Planning priors only: per-page static header and coding inefficiency ranges.
    page_symbols_grid = [512, 1024, 2048, 4096, 8192]
    header_bits_grid = [64, 96, 128]
    coder_overhead_grid = [0.010, 0.020, 0.035]

    rows = []
    best = None
    for page_symbols in page_symbols_grid:
        for header_bits in header_bits_grid:
            for coder_overhead in coder_overhead_grid:
                header_bpw = header_bits / float(page_symbols)
                modeled_bpw = h1 + coder_overhead + header_bpw
                row = {
                    "page_symbols": page_symbols,
                    "header_bits": header_bits,
                    "coder_overhead_bpw": coder_overhead,
                    "header_bpw": header_bpw,
                    "modeled_bpw": modeled_bpw,
                    "margin_vs_corridor_max": args.corridor_max_bpw - modeled_bpw,
                    "decision": "pass_fast" if modeled_bpw <= args.corridor_max_bpw else "fail_fast",
                }
                rows.append(row)
                if best is None or modeled_bpw < best["modeled_bpw"]:
                    best = row

    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    pass_count = sum(1 for r in rows if r["decision"] == "pass_fast")

    summary = {
        "timestamp": ts,
        "label": args.label,
        "source_symbol_atlas_json": str(src),
        "source_global_entropy_bpw": h1,
        "corridor_max_bpw": args.corridor_max_bpw,
        "rows": rows,
        "pass_count": pass_count,
        "best_row": best,
        "decision": "no_feasible_micropage_config" if pass_count == 0 else "feasible_config_exists",
        "notes": [
            "This is a planning model, not measured runtime/compression evidence.",
            "Order-0/global-symbol floor from D055 is used as source entropy baseline.",
        ],
    }

    json_path = out_dir / f"{args.label}.q4_c2_rans_micropage_gate.json"
    md_path = out_dir / f"{args.label}.q4_c2_rans_micropage_gate.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Q4 C2 H50 rANS Micropage Fast Gate: {args.label}",
        "",
        f"- timestamp: {ts}",
        f"- source H1 entropy: {h1:.6f} bpw",
        f"- corridor max: {args.corridor_max_bpw:.4f} bpw",
        "",
        "| page_symbols | header_bits | coder_overhead | header_bpw | modeled_bpw | margin vs corridor max | decision |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            "| "
            f"{r['page_symbols']} | {r['header_bits']} | {r['coder_overhead_bpw']:.3f} | "
            f"{r['header_bpw']:.6f} | {r['modeled_bpw']:.6f} | {r['margin_vs_corridor_max']:.6f} | {r['decision']} |"
        )

    lines.extend(
        [
            "",
            f"Feasible config count: {pass_count}",
            f"Best modeled bpw: {best['modeled_bpw']:.6f}" if best else "Best modeled bpw: n/a",
        ]
    )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path.as_posix()}")
    print(f"Wrote {md_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
