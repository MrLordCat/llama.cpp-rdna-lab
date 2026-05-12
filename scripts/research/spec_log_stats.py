#!/usr/bin/env python3
"""Extract speculative decoding stats from llama-server log."""

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

RATE_RE = re.compile(
    r"draft acceptance rate\s*=\s*(?P<rate>[0-9.]+)\s*\(\s*(?P<acc>\d+) accepted\s*/\s*(?P<gen>\d+) generated\)"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract speculative stats from llama-server log")
    parser.add_argument("--log", required=True, help="path to .server.log file")
    parser.add_argument("--json", action="store_true", help="print JSON output")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"ERROR: log not found: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="replace")

    stat_matches = list(STAT_RE.finditer(text))
    rate_matches = list(RATE_RE.finditer(text))

    by_impl: dict[str, dict[str, float | int | str]] = {}
    for match in stat_matches:
        impl = match.group("impl")
        gen_tokens = int(match.group("gen_tokens"))
        acc_tokens = int(match.group("acc_tokens"))
        gen_drafts = int(match.group("gen_drafts"))
        acc_drafts = int(match.group("acc_drafts"))

        by_impl[impl] = {
            "impl": impl,
            "calls_begin": int(match.group("calls_b")),
            "calls_generate": int(match.group("calls_g")),
            "calls_accumulate": int(match.group("calls_a")),
            "gen_drafts": gen_drafts,
            "acc_drafts": acc_drafts,
            "gen_tokens": gen_tokens,
            "acc_tokens": acc_tokens,
            "token_accept_ratio": (acc_tokens / gen_tokens) if gen_tokens > 0 else 0.0,
            "draft_accept_ratio": (acc_drafts / gen_drafts) if gen_drafts > 0 else 0.0,
            "avg_tokens_per_draft": (gen_tokens / gen_drafts) if gen_drafts > 0 else 0.0,
        }

    acceptance_line = None
    if rate_matches:
        last = rate_matches[-1]
        acceptance_line = {
            "rate": float(last.group("rate")),
            "acc_tokens": int(last.group("acc")),
            "gen_tokens": int(last.group("gen")),
        }

    payload = {
        "log": str(log_path),
        "impl_count": len(by_impl),
        "impl_stats": list(by_impl.values()),
        "draft_acceptance_line": acceptance_line,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"log={log_path}")
    if not by_impl:
        print("no speculative statistics found")
    else:
        for impl in sorted(by_impl.keys()):
            item = by_impl[impl]
            print(
                f"impl={impl} gen_drafts={item['gen_drafts']} acc_drafts={item['acc_drafts']} "
                f"gen_tokens={item['gen_tokens']} acc_tokens={item['acc_tokens']} "
                f"token_accept_ratio={float(item['token_accept_ratio']):.4f}"
            )

    if acceptance_line is not None:
        print(
            "draft_acceptance_line="
            f"{acceptance_line['rate']:.6f} ({acceptance_line['acc_tokens']}/{acceptance_line['gen_tokens']})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
