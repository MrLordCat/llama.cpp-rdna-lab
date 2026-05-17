#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit BENCH_HISTORY comparability for A/B workflow")
    p.add_argument("--history", type=Path, default=Path("build_logs/agent-workload/BENCH_HISTORY.csv"))
    p.add_argument("--label-prefix", default="c01-", help="only rows with label starting with this prefix")
    p.add_argument(
        "--keys",
        default="tasks,ctx,batch,ubatch,kv_k,kv_v,max_tokens,no_reuse,spec_mode,extra_args",
        help="comma-separated fields to define comparability signature",
    )
    p.add_argument("--strict", action="store_true", help="return non-zero if multiple signatures exist")
    p.add_argument("--top", type=int, default=8)
    return p.parse_args()


def normalize(v: str) -> str:
    return (v or "").strip()


def main() -> int:
    args = parse_args()

    if not args.history.exists():
        raise SystemExit(f"ERROR: history file not found: {args.history}")

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]

    with args.history.open("r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if normalize(r.get("label", "")).startswith(args.label_prefix)]

    print("# Bench History Audit")
    print()
    print(f"- history: {args.history}")
    print(f"- label_prefix: {args.label_prefix}")
    print(f"- rows: {len(rows)}")
    print(f"- signature_keys: {keys}")

    if not rows:
        print("\nNo rows matched label_prefix.")
        return 2

    signatures: Counter[tuple[str, ...]] = Counter()
    for r in rows:
        signatures[tuple(normalize(r.get(k, "")) for k in keys)] += 1

    print(f"\n- unique_signatures: {len(signatures)}")

    top = signatures.most_common(args.top)
    print("\n## Top signatures")
    print("| rank | count | signature |")
    print("|---:|---:|---|")
    for i, (sig, cnt) in enumerate(top, start=1):
        pairs = ", ".join(f"{k}={v if v else '-'}" for k, v in zip(keys, sig))
        print(f"| {i} | {cnt} | {pairs} |")

    print("\n## Field heterogeneity")
    print("| field | unique_values | top_values |")
    print("|---|---:|---|")
    for k in keys:
        dist = Counter(normalize(r.get(k, "")) for r in rows)
        vals = "; ".join(f"{(v if v else '-')}: {n}" for v, n in dist.most_common(4))
        print(f"| {k} | {len(dist)} | {vals} |")

    dominant_sig, dominant_count = signatures.most_common(1)[0]
    dominant_pairs = ", ".join(f"{k}={v if v else '-'}" for k, v in zip(keys, dominant_sig))
    print("\n## Recommended comparable lane")
    print(f"- dominant_signature_count: {dominant_count}")
    print(f"- dominant_signature: {dominant_pairs}")

    if len(signatures) > 1:
        print("\nVerdict: MISMATCHED (multiple signatures in same label_prefix)")
        return 3 if args.strict else 0

    print("\nVerdict: CONSISTENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
