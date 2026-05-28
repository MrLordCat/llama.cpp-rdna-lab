#!/usr/bin/env python3
"""Build a per-tensor Phase 1 Q4 MetaComp plan for a target model size.

The planner distributes required savings across Q4 tensors using metadata-first
allocation, then reports unresolved gap that must be covered by payload-side C2.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GGUF_PY = ROOT / "gguf-py"
if str(GGUF_PY) not in sys.path:
    sys.path.insert(0, str(GGUF_PY))

from gguf.gguf_reader import GGUFReader  # type: ignore


Q4_TYPES = {
    "Q4_0",
    "Q4_1",
    "Q4_K",
    "Q4_K_M",
    "Q4_K_S",
}


@dataclass
class TensorPlan:
    name: str
    ttype: str
    n_elements: int
    n_bytes: int
    payload_floor_bytes: int
    metadata_bytes: int
    requested_save_bytes: int
    meta_save_frac: float
    residual_save_bytes: int


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument(
        "--target-gib",
        type=float,
        default=13.0,
        help="Target total model size in GiB",
    )
    parser.add_argument(
        "--label",
        default=f"q4metacomp-phase1-plan-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Output label",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "build_logs" / "agent-workload"),
        help="Directory for outputs",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=32,
        help="How many tensors to include in markdown table",
    )
    return parser.parse_args(argv)


def gib(n_bytes: int) -> float:
    return n_bytes / float(1024**3)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = GGUFReader(str(model_path))

    q4_rows: list[TensorPlan] = []
    total_bytes = 0
    q4_total_elements = 0
    q4_total_metadata = 0
    for t in reader.tensors:
        n_elements = int(t.n_elements)
        n_bytes = int(t.n_bytes)
        total_bytes += n_bytes
        ttype = t.tensor_type.name
        if ttype not in Q4_TYPES:
            continue
        payload_floor = int(math.ceil(n_elements * 0.5))
        metadata = max(0, n_bytes - payload_floor)
        q4_total_elements += n_elements
        q4_total_metadata += metadata
        q4_rows.append(
            TensorPlan(
                name=str(t.name),
                ttype=ttype,
                n_elements=n_elements,
                n_bytes=n_bytes,
                payload_floor_bytes=payload_floor,
                metadata_bytes=metadata,
                requested_save_bytes=0,
                meta_save_frac=0.0,
                residual_save_bytes=0,
            )
        )

    target_total_bytes = int(args.target_gib * (1024**3))
    required_savings = max(0, total_bytes - target_total_bytes)

    unresolved_global = max(0, required_savings - q4_total_metadata)

    # Metadata-first distribution:
    # each tensor is asked to save proportionally to its metadata share.
    for row in q4_rows:
        if q4_total_metadata > 0 and required_savings > 0:
            proportional = int(round(required_savings * (row.metadata_bytes / q4_total_metadata)))
        else:
            proportional = 0
        row.requested_save_bytes = proportional
        row.meta_save_frac = min(1.0, (proportional / row.metadata_bytes) if row.metadata_bytes > 0 else 0.0)
        row.residual_save_bytes = max(0, proportional - row.metadata_bytes)

    # Correct small rounding drift by adjusting the largest metadata tensor.
    allocated = sum(r.requested_save_bytes for r in q4_rows)
    drift = required_savings - allocated
    if q4_rows and drift != 0:
        largest = max(q4_rows, key=lambda r: r.metadata_bytes)
        largest.requested_save_bytes = max(0, largest.requested_save_bytes + drift)
        largest.meta_save_frac = min(
            1.0,
            (largest.requested_save_bytes / largest.metadata_bytes) if largest.metadata_bytes > 0 else 0.0,
        )
        largest.residual_save_bytes = max(0, largest.requested_save_bytes - largest.metadata_bytes)

    unresolved_from_rows = sum(r.residual_save_bytes for r in q4_rows)
    payload_bpw_gap = (unresolved_global * 8.0 / q4_total_elements) if q4_total_elements > 0 else 0.0

    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    summary = {
        "timestamp": ts,
        "label": args.label,
        "model": str(model_path),
        "target_gib": args.target_gib,
        "total_bytes": total_bytes,
        "total_gib": gib(total_bytes),
        "target_total_bytes": target_total_bytes,
        "target_total_gib": gib(target_total_bytes),
        "required_savings_bytes": required_savings,
        "required_savings_gib": gib(required_savings),
        "q4_total_metadata_bytes": q4_total_metadata,
        "q4_total_metadata_gib": gib(q4_total_metadata),
        "unresolved_after_full_metadata_bytes": unresolved_global,
        "unresolved_after_full_metadata_gib": gib(unresolved_global),
        "payload_bpw_gap_if_meta_full": payload_bpw_gap,
        "unresolved_from_tensor_distribution_bytes": unresolved_from_rows,
        "tensor_plans": [
            {
                "name": r.name,
                "type": r.ttype,
                "metadata_bytes": r.metadata_bytes,
                "requested_save_bytes": r.requested_save_bytes,
                "meta_save_frac": r.meta_save_frac,
                "residual_save_bytes": r.residual_save_bytes,
            }
            for r in q4_rows
        ],
    }

    json_path = out_dir / f"{args.label}.q4_metacomp_phase1_plan.json"
    md_path = out_dir / f"{args.label}.q4_metacomp_phase1_plan.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    top_n = max(1, int(args.top_n))
    top_rows = sorted(q4_rows, key=lambda r: r.metadata_bytes, reverse=True)[:top_n]

    lines = [
        f"# Q4 MetaComp Phase1 Plan: {args.label}",
        "",
        f"- timestamp: {ts}",
        f"- model: {model_path.as_posix()}",
        f"- target: {args.target_gib:.3f} GiB",
        "",
        "## Global Gate",
        "",
        f"- current total: {total_bytes} bytes ({gib(total_bytes):.3f} GiB)",
        f"- required savings: {required_savings} bytes ({gib(required_savings):.3f} GiB)",
        f"- total Q4 metadata budget: {q4_total_metadata} bytes ({gib(q4_total_metadata):.3f} GiB)",
        f"- unresolved after full metadata removal: {unresolved_global} bytes ({gib(unresolved_global):.3f} GiB)",
        f"- payload-side gap if metadata is fully compacted: {payload_bpw_gap:.4f} bpw",
        "",
        "## Per-Tensor Metadata Plan (Top by metadata)",
        "",
        "| Tensor | Type | Metadata MiB | Requested save MiB | Meta save frac | Residual MiB |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in top_rows:
        lines.append(
            "| "
            f"{r.name} | {r.ttype} | "
            f"{r.metadata_bytes / (1024**2):.2f} | "
            f"{r.requested_save_bytes / (1024**2):.2f} | "
            f"{r.meta_save_frac:.4f} | "
            f"{r.residual_save_bytes / (1024**2):.2f} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path.as_posix()}")
    print(f"Wrote {md_path.as_posix()}")
    print(
        "Plan summary: "
        f"required={required_savings} bytes, "
        f"metadata_budget={q4_total_metadata} bytes, "
        f"unresolved={unresolved_global} bytes ({gib(unresolved_global):.3f} GiB)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
