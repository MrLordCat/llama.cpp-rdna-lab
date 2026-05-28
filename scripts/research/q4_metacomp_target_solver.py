#!/usr/bin/env python3
"""Solve Q4 MetaComp target feasibility from GGUF tensor stats.

This tool estimates whether a target model size (for example 13 GiB) is
reachable with metadata-only compaction while keeping 4-bit payload semantics.
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
class TensorStats:
    name: str
    ttype: str
    n_elements: int
    n_bytes: int
    payload_bytes_floor: int
    metadata_bytes: int


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
        default=f"q4metacomp-target-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Output label",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "build_logs" / "agent-workload"),
        help="Directory for outputs",
    )
    parser.add_argument(
        "--meta-fracs",
        default="0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated metadata save fractions for scenario table",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top tensors by metadata bytes",
    )
    return parser.parse_args(argv)


def gib(n_bytes: int) -> float:
    return n_bytes / float(1024**3)


def parse_meta_fracs(raw: str) -> list[float]:
    out: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        out.append(max(0.0, min(1.0, value)))
    return out or [0.6, 0.8, 1.0]


def build_tensor_stats(reader: GGUFReader) -> list[TensorStats]:
    rows: list[TensorStats] = []
    for t in reader.tensors:
        ttype = t.tensor_type.name
        n_elements = int(t.n_elements)
        n_bytes = int(t.n_bytes)
        payload_bytes_floor = 0
        metadata_bytes = 0
        if ttype in Q4_TYPES:
            # 4-bit payload lower bound: exactly 0.5 bytes per weight.
            payload_bytes_floor = int(math.ceil(n_elements * 0.5))
            metadata_bytes = max(0, n_bytes - payload_bytes_floor)
        rows.append(
            TensorStats(
                name=str(t.name),
                ttype=ttype,
                n_elements=n_elements,
                n_bytes=n_bytes,
                payload_bytes_floor=payload_bytes_floor,
                metadata_bytes=metadata_bytes,
            )
        )
    return rows


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = GGUFReader(str(model_path))
    rows = build_tensor_stats(reader)

    total_bytes = sum(r.n_bytes for r in rows)
    q4_rows = [r for r in rows if r.ttype in Q4_TYPES]
    q4_total_bytes = sum(r.n_bytes for r in q4_rows)
    q4_payload_floor_bytes = sum(r.payload_bytes_floor for r in q4_rows)
    q4_metadata_bytes = sum(r.metadata_bytes for r in q4_rows)
    q4_total_elements = sum(r.n_elements for r in q4_rows)

    target_total_bytes = int(args.target_gib * (1024**3))
    required_total_savings = max(0, total_bytes - target_total_bytes)

    max_metadata_only_savings = q4_metadata_bytes
    metadata_only_possible = required_total_savings <= max_metadata_only_savings

    required_meta_save_frac = (
        (required_total_savings / q4_metadata_bytes) if q4_metadata_bytes > 0 else float("inf")
    )

    remaining_after_full_metadata = max(0, required_total_savings - max_metadata_only_savings)

    extra_payload_bpw_drop_needed = (
        (remaining_after_full_metadata * 8.0 / q4_total_elements) if q4_total_elements > 0 else 0.0
    )

    meta_fracs = parse_meta_fracs(args.meta_fracs)
    scenarios = []
    for frac in meta_fracs:
        saved = int(round(q4_metadata_bytes * frac))
        predicted_total = total_bytes - saved
        scenarios.append(
            {
                "meta_save_frac": frac,
                "saved_bytes": saved,
                "saved_gib": gib(saved),
                "predicted_total_bytes": predicted_total,
                "predicted_total_gib": gib(predicted_total),
                "hits_target": predicted_total <= target_total_bytes,
            }
        )

    top_n = max(1, int(args.top_n))
    top_meta = sorted(q4_rows, key=lambda r: r.metadata_bytes, reverse=True)[:top_n]

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
        "required_total_savings_bytes": required_total_savings,
        "required_total_savings_gib": gib(required_total_savings),
        "q4_total_bytes": q4_total_bytes,
        "q4_total_gib": gib(q4_total_bytes),
        "q4_payload_floor_bytes": q4_payload_floor_bytes,
        "q4_payload_floor_gib": gib(q4_payload_floor_bytes),
        "q4_metadata_bytes": q4_metadata_bytes,
        "q4_metadata_gib": gib(q4_metadata_bytes),
        "max_metadata_only_savings_bytes": max_metadata_only_savings,
        "max_metadata_only_savings_gib": gib(max_metadata_only_savings),
        "metadata_only_possible": metadata_only_possible,
        "required_meta_save_frac": required_meta_save_frac,
        "remaining_after_full_metadata_bytes": remaining_after_full_metadata,
        "remaining_after_full_metadata_gib": gib(remaining_after_full_metadata),
        "extra_payload_bpw_drop_needed": extra_payload_bpw_drop_needed,
        "scenarios": scenarios,
        "top_q4_metadata_tensors": [
            {
                "name": r.name,
                "type": r.ttype,
                "metadata_bytes": r.metadata_bytes,
                "metadata_mib": r.metadata_bytes / (1024**2),
                "tensor_bytes": r.n_bytes,
                "tensor_mib": r.n_bytes / (1024**2),
            }
            for r in top_meta
        ],
    }

    json_path = out_dir / f"{args.label}.q4_metacomp_target.json"
    md_path = out_dir / f"{args.label}.q4_metacomp_target.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Q4 MetaComp Target Solver: {args.label}",
        "",
        f"- timestamp: {ts}",
        f"- model: {model_path.as_posix()}",
        f"- target: {args.target_gib:.3f} GiB",
        "",
        "## Core Numbers",
        "",
        f"- current total: {total_bytes} bytes ({gib(total_bytes):.3f} GiB)",
        f"- target total: {target_total_bytes} bytes ({gib(target_total_bytes):.3f} GiB)",
        f"- required savings: {required_total_savings} bytes ({gib(required_total_savings):.3f} GiB)",
        "",
        f"- Q4 total bytes: {q4_total_bytes} ({gib(q4_total_bytes):.3f} GiB)",
        f"- Q4 payload floor (4-bit only): {q4_payload_floor_bytes} ({gib(q4_payload_floor_bytes):.3f} GiB)",
        f"- Q4 metadata bytes (current overhead): {q4_metadata_bytes} ({gib(q4_metadata_bytes):.3f} GiB)",
        "",
        f"- max metadata-only savings: {max_metadata_only_savings} ({gib(max_metadata_only_savings):.3f} GiB)",
        f"- metadata-only reaches target: {str(metadata_only_possible).lower()}",
        f"- required metadata save fraction for target: {required_meta_save_frac:.4f}",
        f"- remaining gap after full metadata removal: {remaining_after_full_metadata} bytes ({gib(remaining_after_full_metadata):.3f} GiB)",
        f"- extra payload bpw drop still needed (if metadata already at zero): {extra_payload_bpw_drop_needed:.4f} bpw",
        "",
        "## Scenario Table",
        "",
        "| meta_save_frac | saved GiB | predicted total GiB | hits target |",
        "| ---: | ---: | ---: | --- |",
    ]
    for s in scenarios:
        lines.append(
            f"| {s['meta_save_frac']:.2f} | {s['saved_gib']:.3f} | {s['predicted_total_gib']:.3f} | {str(s['hits_target']).lower()} |"
        )

    lines.extend(
        [
            "",
            "## Top Q4 Metadata Tensors",
            "",
            "| Tensor | Type | Metadata MiB | Tensor MiB |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for r in top_meta:
        lines.append(
            f"| {r.name} | {r.ttype} | {r.metadata_bytes / (1024**2):.2f} | {r.n_bytes / (1024**2):.2f} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path.as_posix()}")
    print(f"Wrote {md_path.as_posix()}")
    if metadata_only_possible:
        print("Result: metadata-only target is feasible.")
    else:
        print(
            "Result: metadata-only target is NOT feasible; "
            f"remaining gap={remaining_after_full_metadata} bytes ({gib(remaining_after_full_metadata):.3f} GiB)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
