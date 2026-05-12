#!/usr/bin/env python3
"""Compute coverage-weighted effective acceptance from speculative log stats."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STAT_RE = re.compile(
    r"statistics\s+(?P<impl>[a-zA-Z0-9_\-]+):\s*"
    r"#calls\(b,g,a\)\s*=\s*(?P<calls_b>\d+)\s+(?P<calls_g>\d+)\s+(?P<calls_a>\d+),\s*"
    r"#gen drafts\s*=\s*(?P<gen_drafts>\d+),\s*"
    r"#acc drafts\s*=\s*(?P<acc_drafts>\d+),\s*"
    r"#gen tokens\s*=\s*(?P<gen_tokens>\d+),\s*"
    r"#acc tokens\s*=\s*(?P<acc_tokens>\d+)"
)


def parse_last_impl_stats(log_path: Path) -> dict[str, float | int | str] | None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = list(STAT_RE.finditer(text))
    if not matches:
        return None

    m = matches[-1]
    calls_g = int(m.group("calls_g"))
    calls_a = int(m.group("calls_a"))
    gen_tokens = int(m.group("gen_tokens"))
    acc_tokens = int(m.group("acc_tokens"))

    local_acceptance = (acc_tokens / gen_tokens) if gen_tokens > 0 else 0.0
    coverage = (calls_a / calls_g) if calls_g > 0 else 0.0
    effective_acceptance = coverage * local_acceptance

    return {
        "impl": m.group("impl"),
        "calls_generate": calls_g,
        "calls_accumulate": calls_a,
        "gen_tokens": gen_tokens,
        "acc_tokens": acc_tokens,
        "local_acceptance": local_acceptance,
        "coverage": coverage,
        "effective_acceptance": effective_acceptance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute effective acceptance from speculative log stats")
    parser.add_argument("--log", required=True, help="path to .server.log")
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"ERROR: log not found: {log_path}")

    stats = parse_last_impl_stats(log_path)
    if stats is None:
        raise SystemExit("ERROR: speculative statistics not found in log")

    if args.json:
        print(json.dumps({"log": str(log_path), "stats": stats}, ensure_ascii=False, indent=2))
        return 0

    print(f"log={log_path}")
    print(f"impl={stats['impl']}")
    print(f"local_acceptance={float(stats['local_acceptance']):.6f}")
    print(f"coverage={float(stats['coverage']):.6f}")
    print(f"effective_acceptance={float(stats['effective_acceptance']):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
