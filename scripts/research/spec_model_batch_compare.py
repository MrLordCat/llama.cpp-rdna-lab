#!/usr/bin/env python3
"""Batch compare naive vs coverage-aware speculative projections."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

from speedup_math import combined_wall_speedup, required_acceptance_for_target_wall, speculative_speedup

PROMPT_RE = re.compile(
    r"prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens \([^)]*,\s*([0-9.]+) tokens per second\)"
)
DECODE_RE = re.compile(
    r"(?:^|\n)\s*eval time =\s*([0-9.]+) ms /\s*(\d+) tokens \([^)]*,\s*([0-9.]+) tokens per second\)"
)
STAT_RE = re.compile(
    r"statistics\s+(?P<impl>[a-zA-Z0-9_\-]+):\s*"
    r"#calls\(b,g,a\)\s*=\s*(?P<calls_b>\d+)\s+(?P<calls_g>\d+)\s+(?P<calls_a>\d+),\s*"
    r"#gen drafts\s*=\s*(?P<gen_drafts>\d+),\s*"
    r"#acc drafts\s*=\s*(?P<acc_drafts>\d+),\s*"
    r"#gen tokens\s*=\s*(?P<gen_tokens>\d+),\s*"
    r"#acc tokens\s*=\s*(?P<acc_tokens>\d+)"
)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        return int(float(text))
    except Exception:
        return default


def read_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    return rows


def aggregate_completion_tps(rows: list[dict[str, Any]]) -> float:
    total_completion = sum(_int(r.get("completion_tokens"), 0) for r in rows)
    total_wall = sum(_float(r.get("wall_s"), 0.0) for r in rows)
    if total_wall <= 0.0 or total_completion <= 0:
        return 0.0
    return total_completion / total_wall


def parse_eval_tps_means(log_path: Path) -> tuple[float, float]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    prompt = [float(m[2]) for m in PROMPT_RE.findall(text)]
    decode = [float(m[2]) for m in DECODE_RE.findall(text)]
    prompt_mean = statistics.mean(prompt) if prompt else 0.0
    decode_mean = statistics.mean(decode) if decode else 0.0
    return prompt_mean, decode_mean


def parse_spec_stats(log_path: Path) -> dict[str, float | int | str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = list(STAT_RE.finditer(text))
    if not matches:
        return {
            "impl": "none",
            "calls_generate": 0,
            "calls_accumulate": 0,
            "gen_drafts": 0,
            "gen_tokens": 0,
            "acc_tokens": 0,
            "local_acceptance": 0.0,
            "coverage": 0.0,
            "effective_acceptance": 0.0,
            "avg_tokens_per_draft": 1.0,
        }

    # Prefer the most informative speculative stats block.
    best = max(
        matches,
        key=lambda m: (
            int(m.group("gen_tokens")),
            int(m.group("calls_a")),
            int(m.group("calls_g")),
        ),
    )

    calls_g = int(best.group("calls_g"))
    calls_a = int(best.group("calls_a"))
    gen_drafts = int(best.group("gen_drafts"))
    gen_tokens = int(best.group("gen_tokens"))
    acc_tokens = int(best.group("acc_tokens"))

    local = (acc_tokens / gen_tokens) if gen_tokens > 0 else 0.0
    coverage = (calls_a / calls_g) if calls_g > 0 else 0.0
    effective = local * coverage
    avg_draft = (gen_tokens / gen_drafts) if gen_drafts > 0 else 1.0

    return {
        "impl": best.group("impl"),
        "calls_generate": calls_g,
        "calls_accumulate": calls_a,
        "gen_drafts": gen_drafts,
        "gen_tokens": gen_tokens,
        "acc_tokens": acc_tokens,
        "local_acceptance": local,
        "coverage": coverage,
        "effective_acceptance": effective,
        "avg_tokens_per_draft": avg_draft,
    }


def _project(
    prefill_share: float,
    prefill_speedup: float,
    decode_speedup: float,
    draft_len: int,
    acceptance: float,
    overhead: float,
) -> float:
    s_spec = speculative_speedup(draft_len=draft_len, accept_rate=acceptance, overhead=overhead)
    return combined_wall_speedup(
        prefill_share=prefill_share,
        prefill_speedup=prefill_speedup,
        spec_speedup=s_spec,
        decode_kernel_speedup=decode_speedup,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch compare naive vs coverage-aware speculative models")
    parser.add_argument("--cases-json", required=True, help="path to JSON array with case definitions")
    parser.add_argument("--default-prefill-share", type=float, default=0.70)
    parser.add_argument("--default-spec-overhead", type=float, default=0.08)
    parser.add_argument("--json", action="store_true", help="print JSON payload")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    cases_path = Path(args.cases_json)
    if not cases_path.exists():
        raise SystemExit(f"ERROR: cases file not found: {cases_path}")

    cases_obj = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases_obj, list):
        raise SystemExit("ERROR: cases JSON must be an array")

    results: list[dict[str, Any]] = []

    for case in cases_obj:
        if not isinstance(case, dict):
            raise SystemExit("ERROR: each case must be an object")

        cid = str(case.get("id", "case"))
        baseline_csv = Path(str(case["baseline_csv"]))
        candidate_csv = Path(str(case["candidate_csv"]))
        baseline_log = Path(str(case["baseline_log"]))
        candidate_log = Path(str(case["candidate_log"]))

        for p in [baseline_csv, candidate_csv, baseline_log, candidate_log]:
            if not p.exists():
                raise SystemExit(f"ERROR: missing file in case {cid}: {p}")

        base_rows = read_csv_rows(baseline_csv)
        cand_rows = read_csv_rows(candidate_csv)
        base_tps = aggregate_completion_tps(base_rows)
        cand_tps = aggregate_completion_tps(cand_rows)
        observed = (cand_tps / base_tps) if base_tps > 0.0 else 0.0

        base_prompt, base_decode = parse_eval_tps_means(baseline_log)
        cand_prompt, cand_decode = parse_eval_tps_means(candidate_log)

        prefill_speedup = (cand_prompt / base_prompt) if base_prompt > 0.0 else 1.0
        decode_speedup = (cand_decode / base_decode) if base_decode > 0.0 else 1.0

        spec = parse_spec_stats(candidate_log)

        draft_len_raw = case.get("draft_len")
        if draft_len_raw is None:
            draft_len = max(1, int(round(float(spec["avg_tokens_per_draft"]))))
        else:
            draft_len = max(1, int(draft_len_raw))

        prefill_share = float(case.get("prefill_share", args.default_prefill_share))
        overhead = float(case.get("spec_overhead", args.default_spec_overhead))

        local = float(spec["local_acceptance"])
        effective = float(spec["effective_acceptance"])

        naive_proj = _project(
            prefill_share=prefill_share,
            prefill_speedup=prefill_speedup,
            decode_speedup=decode_speedup,
            draft_len=draft_len,
            acceptance=local,
            overhead=overhead,
        )
        cov_proj = _project(
            prefill_share=prefill_share,
            prefill_speedup=prefill_speedup,
            decode_speedup=decode_speedup,
            draft_len=draft_len,
            acceptance=effective,
            overhead=overhead,
        )

        naive_err = abs(observed - naive_proj)
        cov_err = abs(observed - cov_proj)
        better = "coverage-aware" if cov_err < naive_err else ("naive" if naive_err < cov_err else "tie")

        implied = required_acceptance_for_target_wall(
            target_wall_speedup=observed,
            prefill_share=prefill_share,
            prefill_speedup=prefill_speedup,
            decode_kernel_speedup=decode_speedup,
            draft_len=draft_len,
            overhead=overhead,
        )

        results.append(
            {
                "id": cid,
                "observed_wall_speedup": observed,
                "baseline_tps": base_tps,
                "candidate_tps": cand_tps,
                "prefill_speedup": prefill_speedup,
                "decode_speedup": decode_speedup,
                "impl": spec["impl"],
                "draft_len_used": draft_len,
                "local_acceptance": local,
                "coverage": float(spec["coverage"]),
                "effective_acceptance": effective,
                "naive_projection": naive_proj,
                "coverage_aware_projection": cov_proj,
                "naive_abs_error": naive_err,
                "coverage_aware_abs_error": cov_err,
                "better_fit": better,
                "implied_acceptance_for_observed": implied,
                "candidate_log": str(candidate_log),
            }
        )

    if args.json:
        print(json.dumps({"cases": results}, ensure_ascii=False, indent=2))
        return 0

    print("=== Spec Model Batch Compare ===")
    print("| id | observed | naive | cov | err_naive | err_cov | better | D | local | covg | eff |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for r in results:
        print(
            "| {id} | {obs:.6f} | {naive:.6f} | {covp:.6f} | {en:.6f} | {ec:.6f} | {better} | {d} | {local:.6f} | {covg:.6f} | {eff:.6f} |".format(
                id=r["id"],
                obs=r["observed_wall_speedup"],
                naive=r["naive_projection"],
                covp=r["coverage_aware_projection"],
                en=r["naive_abs_error"],
                ec=r["coverage_aware_abs_error"],
                better=r["better_fit"],
                d=r["draft_len_used"],
                local=r["local_acceptance"],
                covg=r["coverage"],
                eff=r["effective_acceptance"],
            )
        )

    cov_wins = sum(1 for r in results if r["better_fit"] == "coverage-aware")
    naive_wins = sum(1 for r in results if r["better_fit"] == "naive")
    ties = sum(1 for r in results if r["better_fit"] == "tie")
    print("")
    print(f"summary: coverage-aware wins={cov_wins}, naive wins={naive_wins}, ties={ties}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
