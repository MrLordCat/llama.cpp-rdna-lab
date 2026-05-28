#!/usr/bin/env python3
"""Build a guarded C1+C2 prototype plan for Q4 MetaComp.

This script does not modify GGUF payloads. It creates a prototype manifest
using:
- C1 metadata compaction budget (phase1 plan),
- C2 value-aware entropy deltas (H54-B artifact),
with explicit per-tensor fallback policy.
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


Q4_TYPES = {"Q4_0", "Q4_1", "Q4_K", "Q4_K_M", "Q4_K_S"}


@dataclass
class TensorDecision:
    name: str
    ttype: str
    elements_total: int
    entropy_delta_bpw: float
    nrmse: float
    projected_c2_saved_bytes: int
    selected: bool
    reason: str


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Path to GGUF model")
    p.add_argument("--phase1-json", required=True, help="Path to q4_metacomp_phase1_plan.json")
    p.add_argument("--value-aware-json", required=True, help="Path to q4_c2_value_aware_gate.json")
    p.add_argument("--target-gib", type=float, default=13.0, help="Target model size in GiB")
    p.add_argument("--nrmse-budget", type=float, default=0.115, help="Per-tensor NRMSE budget")
    p.add_argument(
        "--min-entropy-gain-bpw",
        type=float,
        default=0.45,
        help="Minimum per-tensor entropy gain for selecting C2 path",
    )
    p.add_argument(
        "--c2-safety-margin",
        type=float,
        default=0.90,
        help="Apply safety factor to projected C2 savings (0..1)",
    )
    p.add_argument(
        "--label",
        default=f"q4metacomp-guarded-prototype-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Output label",
    )
    p.add_argument(
        "--out-dir",
        default=str(ROOT / "build_logs" / "agent-workload"),
        help="Output directory",
    )
    return p.parse_args(argv)


def gib(n_bytes: int) -> float:
    return n_bytes / float(1024**3)


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    model_path = Path(args.model)
    phase1_path = Path(args.phase1_json)
    value_path = Path(args.value_aware_json)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not phase1_path.exists():
        raise FileNotFoundError(f"Phase1 artifact not found: {phase1_path}")
    if not value_path.exists():
        raise FileNotFoundError(f"Value-aware artifact not found: {value_path}")

    safety_margin = clamp01(float(args.c2_safety_margin))

    phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
    value = json.loads(value_path.read_text(encoding="utf-8"))

    reader = GGUFReader(str(model_path))
    total_bytes = sum(int(t.n_bytes) for t in reader.tensors)

    # Map tensor type from model for robust manifest rows.
    tensor_type_map = {str(t.name): t.tensor_type.name for t in reader.tensors}

    tensor_rows = value.get("tensors", [])
    decisions: list[TensorDecision] = []

    c2_saved_raw = 0
    for row in tensor_rows:
        name = str(row["name"])
        ttype = tensor_type_map.get(name, str(row.get("type", "UNKNOWN")))
        if ttype not in Q4_TYPES:
            continue

        n_total = int(row.get("elements_total", 0))
        delta = float(row.get("entropy_delta_bpw", 0.0))
        nrmse = float(row.get("nrmse", 1.0e9))

        gain_bpw = max(0.0, -delta)
        saved_bytes = int(math.floor(gain_bpw * n_total / 8.0))

        if gain_bpw < args.min_entropy_gain_bpw:
            selected = False
            reason = "entropy_gain_below_threshold"
        elif nrmse > args.nrmse_budget:
            selected = False
            reason = "nrmse_above_budget"
        else:
            selected = True
            reason = "selected"
            c2_saved_raw += saved_bytes

        decisions.append(
            TensorDecision(
                name=name,
                ttype=ttype,
                elements_total=n_total,
                entropy_delta_bpw=delta,
                nrmse=nrmse,
                projected_c2_saved_bytes=saved_bytes,
                selected=selected,
                reason=reason,
            )
        )

    c2_saved_safe = int(math.floor(c2_saved_raw * safety_margin))

    # C1 full metadata budget from phase1 gate.
    c1_saved = int(phase1.get("q4_total_metadata_bytes", 0))

    projected_bytes = max(0, total_bytes - c1_saved - c2_saved_safe)
    target_bytes = int(args.target_gib * (1024**3))
    headroom = target_bytes - projected_bytes

    selected_count = sum(1 for d in decisions if d.selected)

    payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "label": args.label,
        "model": str(model_path),
        "inputs": {
            "phase1_json": str(phase1_path),
            "value_aware_json": str(value_path),
        },
        "contracts": {
            "target_gib": args.target_gib,
            "nrmse_budget": args.nrmse_budget,
            "min_entropy_gain_bpw": args.min_entropy_gain_bpw,
            "c2_safety_margin": safety_margin,
        },
        "projection": {
            "total_bytes": total_bytes,
            "total_gib": gib(total_bytes),
            "c1_saved_bytes": c1_saved,
            "c1_saved_gib": gib(c1_saved),
            "c2_saved_raw_bytes": c2_saved_raw,
            "c2_saved_safe_bytes": c2_saved_safe,
            "c2_saved_safe_gib": gib(c2_saved_safe),
            "projected_bytes": projected_bytes,
            "projected_gib": gib(projected_bytes),
            "target_bytes": target_bytes,
            "target_gib": args.target_gib,
            "target_headroom_bytes": headroom,
            "target_headroom_gib": gib(headroom),
        },
        "selection": {
            "total_rows": len(decisions),
            "selected_rows": selected_count,
            "fallback_rows": len(decisions) - selected_count,
        },
        "prototype_mode": "guarded",
        "fallback_policy": "per-tensor fail-closed: if row not selected, keep legacy Q4 path",
        "tensor_decisions": [d.__dict__ for d in decisions],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.label}.q4_metacomp_guarded_prototype.json"
    md_path = out_dir / f"{args.label}.q4_metacomp_guarded_prototype.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Q4 MetaComp Guarded Prototype Plan: {args.label}",
        "",
        f"- model: {model_path.as_posix()}",
        f"- target_gib: {args.target_gib:.3f}",
        f"- nrmse_budget: {args.nrmse_budget:.6f}",
        f"- min_entropy_gain_bpw: {args.min_entropy_gain_bpw:.6f}",
        f"- c2_safety_margin: {safety_margin:.3f}",
        "",
        "## Projection",
        "",
        f"- total: {total_bytes} bytes ({gib(total_bytes):.3f} GiB)",
        f"- C1 saved: {c1_saved} bytes ({gib(c1_saved):.3f} GiB)",
        f"- C2 saved (raw): {c2_saved_raw} bytes ({gib(c2_saved_raw):.3f} GiB)",
        f"- C2 saved (safe): {c2_saved_safe} bytes ({gib(c2_saved_safe):.3f} GiB)",
        f"- projected: {projected_bytes} bytes ({gib(projected_bytes):.3f} GiB)",
        f"- target: {target_bytes} bytes ({args.target_gib:.3f} GiB)",
        f"- target headroom: {headroom} bytes ({gib(headroom):.3f} GiB)",
        "",
        "## Selection",
        "",
        f"- selected tensors: {selected_count}/{len(decisions)}",
        f"- fallback tensors: {len(decisions) - selected_count}/{len(decisions)}",
        "",
        "Per-tensor fallback policy is fail-closed.",
    ]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path.as_posix()}")
    print(f"Wrote {md_path.as_posix()}")
    print(
        "Guarded prototype projection: "
        f"projected={gib(projected_bytes):.3f} GiB, target={args.target_gib:.3f} GiB, "
        f"headroom={gib(headroom):.3f} GiB"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
