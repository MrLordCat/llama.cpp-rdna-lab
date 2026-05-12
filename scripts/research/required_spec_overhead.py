#!/usr/bin/env python3
"""Backsolve required speculative overhead for an observed wall speedup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def required_overhead_for_target_wall(
    target_wall_speedup: float,
    prefill_share: float,
    prefill_speedup: float,
    decode_kernel_speedup: float,
    draft_len: int,
    acceptance: float,
) -> float | None:
    """Solve overhead o from:

    S_spec = (1 + a * (D - 1)) / (1 + o)
    S_total = 1 / (p / S_prefill + (1 - p) / (S_spec * S_decode))
    """
    target = float(target_wall_speedup)
    p = float(prefill_share)
    s_pref = float(prefill_speedup)
    s_dec = float(decode_kernel_speedup)
    d = max(1, int(draft_len))
    a = max(0.0, min(1.0, float(acceptance)))

    if target <= 0.0 or s_pref <= 0.0 or s_dec <= 0.0:
        return None
    if p < 0.0 or p >= 1.0:
        return None

    rhs = (1.0 / target) - (p / s_pref)
    if rhs <= 0.0:
        return None

    numerator = rhs * s_dec * (1.0 + a * (d - 1.0))
    denom = (1.0 - p)
    if denom <= 0.0:
        return None

    one_plus_o = numerator / denom
    return one_plus_o - 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backsolve speculative overhead required for observed wall speedup")
    parser.add_argument("--target-wall", type=float, help="observed wall speedup")
    parser.add_argument("--prefill-share", type=float, default=0.70)
    parser.add_argument("--prefill-speedup", type=float)
    parser.add_argument("--decode-kernel-speedup", type=float)
    parser.add_argument("--draft-len", type=int)
    parser.add_argument("--acceptance", type=float)
    parser.add_argument("--batch-json", help="optional JSON file from spec_model_batch_compare --json")
    parser.add_argument("--acceptance-mode", choices=["effective", "local"], default="effective")
    parser.add_argument("--json", action="store_true")
    return parser


def run_single(args: argparse.Namespace) -> dict[str, float | None]:
    if args.prefill_speedup is None or args.decode_kernel_speedup is None or args.draft_len is None or args.acceptance is None:
        raise SystemExit("ERROR: single mode requires --prefill-speedup, --decode-kernel-speedup, --draft-len, --acceptance")

    required = required_overhead_for_target_wall(
        target_wall_speedup=float(args.target_wall),
        prefill_share=float(args.prefill_share),
        prefill_speedup=float(args.prefill_speedup),
        decode_kernel_speedup=float(args.decode_kernel_speedup),
        draft_len=int(args.draft_len),
        acceptance=float(args.acceptance),
    )

    return {
        "target_wall_speedup": float(args.target_wall),
        "prefill_share": float(args.prefill_share),
        "prefill_speedup": float(args.prefill_speedup),
        "decode_kernel_speedup": float(args.decode_kernel_speedup),
        "draft_len": int(args.draft_len),
        "acceptance": float(args.acceptance),
        "required_overhead": required,
    }


def run_batch(args: argparse.Namespace) -> dict[str, object]:
    path = Path(str(args.batch_json))
    if not path.exists():
        raise SystemExit(f"ERROR: batch JSON not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        raise SystemExit("ERROR: batch JSON must contain object with 'cases' array")

    mode = str(args.acceptance_mode)
    rows = []
    for c in cases:
        if not isinstance(c, dict):
            continue
        acceptance_key = "effective_acceptance" if mode == "effective" else "local_acceptance"
        acceptance = float(c.get(acceptance_key, 0.0))
        required = required_overhead_for_target_wall(
            target_wall_speedup=float(c.get("observed_wall_speedup", 0.0)),
            prefill_share=float(args.prefill_share),
            prefill_speedup=float(c.get("prefill_speedup", 0.0)),
            decode_kernel_speedup=float(c.get("decode_speedup", 0.0)),
            draft_len=int(c.get("draft_len_used", 1)),
            acceptance=acceptance,
        )
        rows.append(
            {
                "id": str(c.get("id", "case")),
                "acceptance_mode": mode,
                "acceptance": acceptance,
                "draft_len": int(c.get("draft_len_used", 1)),
                "observed_wall_speedup": float(c.get("observed_wall_speedup", 0.0)),
                "required_overhead": required,
            }
        )

    return {"prefill_share": float(args.prefill_share), "acceptance_mode": mode, "rows": rows}


def main() -> int:
    args = build_parser().parse_args()

    if args.batch_json:
        result = run_batch(args)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        print("=== Required Spec Overhead (batch) ===")
        print("| id | acceptance_mode | acceptance | D | observed | required_overhead |")
        print("| --- | --- | ---: | ---: | ---: | ---: |")
        for row in result["rows"]:
            val = row["required_overhead"]
            text = "unreachable" if val is None else f"{val:.6f}"
            print(
                f"| {row['id']} | {row['acceptance_mode']} | {row['acceptance']:.6f} | {row['draft_len']} | "
                f"{row['observed_wall_speedup']:.6f} | {text} |"
            )
        return 0

    if args.target_wall is None:
        raise SystemExit("ERROR: provide --target-wall for single mode or use --batch-json")

    result_single = run_single(args)
    if args.json:
        print(json.dumps(result_single, ensure_ascii=False, indent=2))
        return 0

    value = result_single["required_overhead"]
    if value is None:
        print("required_overhead=unreachable")
    else:
        print(f"required_overhead={value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
