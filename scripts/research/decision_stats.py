#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def f(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def completion_tps_aggregate(rows: list[dict[str, str]]) -> float:
    total_tokens = sum(int(f(r.get("completion_tokens", "0"))) for r in rows)
    total_wall = sum(f(r.get("wall_s", "0")) for r in rows)
    if total_tokens <= 0 or total_wall <= 0:
        return 0.0
    return total_tokens / total_wall


def per_task_tps(rows: list[dict[str, str]]) -> list[float]:
    values = [f(r.get("completion_tps_wall", "0")) for r in rows]
    return [v for v in values if v > 0.0]


def mean_ci_normal(values: list[float], z: float = 1.96) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    m = statistics.mean(values)
    if len(values) < 2:
        return m, m, m
    sd = statistics.stdev(values)
    se = sd / math.sqrt(len(values))
    return m, m - z * se, m + z * se


def cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma = statistics.mean(a)
    mb = statistics.mean(b)
    sda = statistics.stdev(a)
    sdb = statistics.stdev(b)
    na = len(a)
    nb = len(b)
    pooled = math.sqrt(((na - 1) * sda * sda + (nb - 1) * sdb * sdb) / max(1, (na + nb - 2)))
    if pooled == 0.0:
        return 0.0
    return (mb - ma) / pooled


def bootstrap_delta_ci(
    baseline: list[float], candidate: list[float], n_boot: int, seed: int
) -> tuple[float, float, float]:
    if not baseline or not candidate:
        return 0.0, 0.0, 0.0

    rng = random.Random(seed)
    diffs: list[float] = []

    for _ in range(n_boot):
        b = [baseline[rng.randrange(len(baseline))] for _ in range(len(baseline))]
        c = [candidate[rng.randrange(len(candidate))] for _ in range(len(candidate))]
        diffs.append(statistics.mean(c) - statistics.mean(b))

    diffs.sort()
    mid = statistics.mean(diffs)
    lo = diffs[int(0.025 * (len(diffs) - 1))]
    hi = diffs[int(0.975 * (len(diffs) - 1))]
    return mid, lo, hi


def main() -> int:
    p = argparse.ArgumentParser(description="Statistical verdict for lane CSV A/B")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--borderline-pct", type=float, default=1.0, help="threshold in percent for borderline region")
    args = p.parse_args()

    if not args.baseline.exists() or not args.candidate.exists():
        raise SystemExit("ERROR: baseline/candidate CSV not found")

    base_rows = read_rows(args.baseline)
    cand_rows = read_rows(args.candidate)

    base_agg = completion_tps_aggregate(base_rows)
    cand_agg = completion_tps_aggregate(cand_rows)
    agg_delta = cand_agg - base_agg
    agg_pct = 100.0 * agg_delta / base_agg if base_agg > 0 else 0.0

    base_task = per_task_tps(base_rows)
    cand_task = per_task_tps(cand_rows)

    base_mean, base_lo, base_hi = mean_ci_normal(base_task)
    cand_mean, cand_lo, cand_hi = mean_ci_normal(cand_task)

    boot_mid, boot_lo, boot_hi = bootstrap_delta_ci(base_task, cand_task, args.bootstrap, args.seed)
    d = cohens_d(base_task, cand_task)

    borderline = abs(agg_pct) <= args.borderline_pct

    print("# Decision Stats")
    print()
    print(f"- baseline_csv: {args.baseline}")
    print(f"- candidate_csv: {args.candidate}")
    print(f"- aggregate_tps: {base_agg:.4f} -> {cand_agg:.4f}")
    print(f"- aggregate_delta: {agg_delta:+.4f} ({agg_pct:+.2f}%)")
    print(f"- borderline_region: {'yes' if borderline else 'no'} (|delta| <= {args.borderline_pct:.2f}%)")
    print()

    print("## Per-task mean CI (normal approx)")
    print(f"- baseline mean: {base_mean:.4f} 95% CI [{base_lo:.4f}, {base_hi:.4f}] (n={len(base_task)})")
    print(f"- candidate mean: {cand_mean:.4f} 95% CI [{cand_lo:.4f}, {cand_hi:.4f}] (n={len(cand_task)})")
    print()

    print("## Delta statistics")
    print(f"- bootstrap mean delta (cand-base): {boot_mid:+.4f}")
    print(f"- bootstrap 95% CI: [{boot_lo:+.4f}, {boot_hi:+.4f}]")
    print(f"- effect size (Cohen d): {d:+.4f}")

    if boot_lo > 0:
        verdict = "positive"
    elif boot_hi < 0:
        verdict = "negative"
    else:
        verdict = "inconclusive"

    print(f"- statistical_verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
