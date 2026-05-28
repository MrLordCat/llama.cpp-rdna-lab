#!/usr/bin/env python3
"""Compute payload-side (C2) requirements across metadata compaction levels.

Uses Q4 model decomposition to report required effective payload bpw for target
size at each metadata-save fraction.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-json",
        required=True,
        help="Path to *.q4_metacomp_target.json produced by q4_metacomp_target_solver.py",
    )
    parser.add_argument(
        "--label",
        default=f"q4metacomp-c2req-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Output label",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path("build_logs") / "agent-workload"),
        help="Output directory",
    )
    parser.add_argument(
        "--meta-fracs",
        default="0.60,0.70,0.80,0.90,1.00",
        help="Comma-separated metadata save fractions",
    )
    return parser.parse_args()


def gib(n_bytes: int) -> float:
    return n_bytes / float(1024**3)


def parse_fracs(raw: str) -> list[float]:
    vals: list[float] = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        v = float(p)
        vals.append(max(0.0, min(1.0, v)))
    return vals or [0.6, 0.8, 1.0]


def main() -> int:
    args = parse_args()
    target_json = Path(args.target_json)
    if not target_json.exists():
        raise FileNotFoundError(f"Target json not found: {target_json}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(target_json.read_text(encoding="utf-8"))

    total_bytes = int(data["total_bytes"])
    target_total_bytes = int(data["target_total_bytes"])
    q4_metadata_bytes = int(data["q4_metadata_bytes"])
    q4_payload_floor_bytes = int(data["q4_payload_floor_bytes"])
    q4_elements = q4_payload_floor_bytes * 2

    required_savings = max(0, total_bytes - target_total_bytes)

    rows = []
    for f in parse_fracs(args.meta_fracs):
        meta_saved = int(round(q4_metadata_bytes * f))
        remaining = max(0, required_savings - meta_saved)
        payload_bpw_drop = (remaining * 8.0 / q4_elements) if q4_elements > 0 else 0.0
        required_payload_bpw = max(0.0, 4.0 - payload_bpw_drop)
        payload_comp_ratio = (required_payload_bpw / 4.0) if 4.0 > 0 else 1.0
        rows.append(
            {
                "meta_save_frac": f,
                "meta_saved_bytes": meta_saved,
                "meta_saved_gib": gib(meta_saved),
                "remaining_bytes": remaining,
                "remaining_gib": gib(remaining),
                "payload_bpw_drop": payload_bpw_drop,
                "required_payload_bpw": required_payload_bpw,
                "payload_comp_ratio": payload_comp_ratio,
            }
        )

    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    summary = {
        "timestamp": ts,
        "label": args.label,
        "source_target_json": str(target_json),
        "target_total_bytes": target_total_bytes,
        "target_total_gib": gib(target_total_bytes),
        "required_savings_bytes": required_savings,
        "required_savings_gib": gib(required_savings),
        "rows": rows,
    }

    json_path = out_dir / f"{args.label}.q4_metacomp_c2_requirements.json"
    md_path = out_dir / f"{args.label}.q4_metacomp_c2_requirements.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Q4 MetaComp C2 Requirements: {args.label}",
        "",
        f"- timestamp: {ts}",
        f"- source: {target_json.as_posix()}",
        "",
        "## Table",
        "",
        "| meta_save_frac | meta saved GiB | remaining GiB | payload bpw drop needed | required payload bpw | payload ratio vs 4.0bpw |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            "| "
            f"{r['meta_save_frac']:.2f} | "
            f"{r['meta_saved_gib']:.3f} | "
            f"{r['remaining_gib']:.3f} | "
            f"{r['payload_bpw_drop']:.4f} | "
            f"{r['required_payload_bpw']:.4f} | "
            f"{r['payload_comp_ratio']:.4f} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path.as_posix()}")
    print(f"Wrote {md_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
