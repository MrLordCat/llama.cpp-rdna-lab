#!/usr/bin/env python3
"""Final analytical gate for H54-B value-aware quantization.

This gate consumes the wide H54-B artifact and applies explicit contracts:
1) payload entropy corridor,
2) quality budget (weighted NRMSE),
3) decode-complexity proxy budget.

It is theory-only and does not claim measured runtime speed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--value-aware-json",
        required=True,
        help="Path to *.q4_c2_value_aware_gate.json artifact",
    )
    p.add_argument(
        "--label",
        default=f"q4c2-value-aware-final-gate-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Output label",
    )
    p.add_argument(
        "--required-payload-bpw-max",
        type=float,
        default=3.7701,
        help="Upper payload corridor bound",
    )
    p.add_argument(
        "--nrmse-budget",
        type=float,
        default=0.115,
        help="Weighted NRMSE budget",
    )
    p.add_argument(
        "--hard-complexity-budget",
        type=float,
        default=1.35,
        help="Complexity index budget",
    )
    p.add_argument(
        "--out-dir",
        default=str(Path("build_logs") / "agent-workload"),
        help="Output directory",
    )
    return p.parse_args()


def weighted_complexity(extra_bit_ops: float, table_lookups: float, branch_events: float, random_reads: float, state_updates: float) -> float:
    # Keep weights aligned with Ck-4 script for consistency.
    return (
        1.0
        + 0.18 * extra_bit_ops
        + 0.22 * table_lookups
        + 0.35 * branch_events
        + 0.30 * random_reads
        + 0.12 * state_updates
    )


def main() -> int:
    args = parse_args()
    src = Path(args.value_aware_json)
    if not src.exists():
        raise FileNotFoundError(f"Missing input artifact: {src}")

    payload = json.loads(src.read_text(encoding="utf-8"))
    summary = payload["summary"]

    entropy_bpw = float(summary["new_entropy_bpw"])
    weighted_nrmse = float(summary.get("weighted_nrmse", 1.0e9))

    # Complexity proxy model:
    # - Baseline: current Q4 decode (unpack + scale/min arithmetic), normalized later.
    # - Candidate: value-aware decode (table lookup path with lower bit-unpack pressure).
    baseline_profile = {
        "extra_bit_ops": 1.00,
        "table_lookups": 0.20,
        "branch_events": 0.05,
        "random_reads": 0.15,
        "state_updates": 0.10,
    }
    candidate_profile = {
        "extra_bit_ops": 0.35,
        "table_lookups": 1.00,
        "branch_events": 0.02,
        "random_reads": 0.55,
        "state_updates": 0.08,
    }

    baseline_raw = weighted_complexity(**baseline_profile)
    candidate_raw = weighted_complexity(**candidate_profile)
    complexity_index = candidate_raw / baseline_raw

    entropy_pass = entropy_bpw <= args.required_payload_bpw_max
    quality_pass = weighted_nrmse <= args.nrmse_budget
    complexity_pass = complexity_index <= args.hard_complexity_budget
    gate_pass = entropy_pass and quality_pass and complexity_pass

    decision = "authorize_guarded_prototype" if gate_pass else "hold_theory_only"

    out = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "label": args.label,
        "input_artifact": src.as_posix(),
        "contracts": {
            "required_payload_bpw_max": args.required_payload_bpw_max,
            "nrmse_budget": args.nrmse_budget,
            "hard_complexity_budget": args.hard_complexity_budget,
        },
        "observed": {
            "new_entropy_bpw": entropy_bpw,
            "weighted_nrmse": weighted_nrmse,
            "complexity_index": complexity_index,
            "baseline_complexity_raw": baseline_raw,
            "candidate_complexity_raw": candidate_raw,
            "tensors_analyzed": int(summary.get("tensors_analyzed", 0)),
            "total_elements": int(summary.get("total_elements", 0)),
        },
        "checks": {
            "entropy_pass": entropy_pass,
            "quality_pass": quality_pass,
            "complexity_pass": complexity_pass,
            "gate_pass": gate_pass,
        },
        "profiles": {
            "baseline": baseline_profile,
            "candidate": candidate_profile,
        },
        "decision": decision,
        "note": "Analytical decision only. Runtime A/B and quality validation on task benchmarks are still required before default rollout.",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.label}.q4_c2_value_aware_final_gate.json"
    md_path = out_dir / f"{args.label}.q4_c2_value_aware_final_gate.md"

    json_path.write_text(json.dumps(out, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        f"# H54-B Final Analytical Gate: {args.label}",
        "",
        f"- input: `{src.as_posix()}`",
        f"- entropy: `{entropy_bpw:.6f}` (limit `{args.required_payload_bpw_max:.6f}`) -> `{'PASS' if entropy_pass else 'FAIL'}`",
        f"- weighted_nrmse: `{weighted_nrmse:.6f}` (limit `{args.nrmse_budget:.6f}`) -> `{'PASS' if quality_pass else 'FAIL'}`",
        f"- complexity_index: `{complexity_index:.6f}` (limit `{args.hard_complexity_budget:.6f}`) -> `{'PASS' if complexity_pass else 'FAIL'}`",
        f"- decision: `{decision}`",
        "",
        "This gate is analytical and must be followed by runtime/quality validation before default rollout.",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path.as_posix()}")
    print(f"Wrote {md_path.as_posix()}")
    print(f"Decision: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
