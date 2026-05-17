#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def required_local_speedup(target_share: float, target_total_speedup: float) -> float | None:
    denom = (1.0 / target_total_speedup) - (1.0 - target_share)
    if denom <= 0.0:
        return None
    return target_share / denom


def parse_hypotheses_table(md_path: Path) -> list[dict[str, str]]:
    lines = md_path.read_text(encoding="utf-8").splitlines()

    header_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("| ID | Idea | Why It Might Work | Expected Impact | Main Risk | First Check |"):
            header_idx = i
            break

    if header_idx < 0:
        raise SystemExit("ERROR: hypotheses table header not found")

    rows: list[dict[str, str]] = []
    for line in lines[header_idx + 2 :]:
        s = line.strip()
        if not s.startswith("|"):
            break
        cols = [c.strip() for c in s.strip("|").split("|")]
        if len(cols) < 6:
            continue
        rows.append(
            {
                "id": cols[0],
                "idea": cols[1],
                "expected": cols[3],
                "risk": cols[4],
            }
        )

    return rows


def expected_max_gain_percent(expected_text: str) -> float:
    values = []
    for m in re.finditer(r"([+-]?\d+(?:\.\d+)?)%", expected_text):
        try:
            values.append(float(m.group(1)))
        except ValueError:
            continue
    if not values:
        return 0.0
    return max(values)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shortlist hypotheses by required local speedup ceiling")
    p.add_argument(
        "--hypotheses",
        default="docs/research/HYPOTHESES.md",
        help="path to HYPOTHESES.md",
    )
    p.add_argument("--share", type=float, required=True, help="target center wall share in (0,1)")
    p.add_argument(
        "--goal-total-speedup",
        type=float,
        default=1.02,
        help="desired end-to-end speedup, e.g. 1.02 for +2%%",
    )
    p.add_argument(
        "--min-absolute-gain-pct",
        type=float,
        default=0.0,
        help="extra floor for expected gain, independent of Amdahl gate",
    )
    p.add_argument(
        "--focus-ids",
        default="",
        help="comma-separated IDs to include only these hypotheses",
    )
    p.add_argument(
        "--top",
        type=int,
        default=0,
        help="limit shortlist length after sorting by expected max gain (0 means all)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.share <= 0.0 or args.share >= 1.0:
        raise SystemExit("ERROR: --share must be in (0,1)")

    req_speedup = required_local_speedup(args.share, args.goal_total_speedup)
    if req_speedup is None:
        raise SystemExit("ERROR: impossible goal for this share")

    required_gain_pct = (req_speedup - 1.0) * 100.0
    required_gain_pct = max(required_gain_pct, float(args.min_absolute_gain_pct))

    md_path = Path(args.hypotheses)
    rows = parse_hypotheses_table(md_path)

    focus_set = {
        x.strip().upper()
        for x in str(args.focus_ids).split(",")
        if x.strip()
    }

    scored: list[dict[str, str | float | bool]] = []
    for row in rows:
        hid = str(row["id"]).strip().upper()
        if focus_set and hid not in focus_set:
            continue
        exp = str(row["expected"])
        max_gain = expected_max_gain_percent(exp)
        keep = max_gain >= required_gain_pct
        scored.append(
            {
                "id": hid,
                "idea": str(row["idea"]),
                "expected": exp,
                "max_gain_pct": max_gain,
                "keep": keep,
            }
        )

    keep_rows = [r for r in scored if bool(r["keep"])]
    drop_rows = [r for r in scored if not bool(r["keep"])]

    keep_rows.sort(key=lambda r: float(r["max_gain_pct"]), reverse=True)
    drop_rows.sort(key=lambda r: float(r["max_gain_pct"]), reverse=True)

    if args.top > 0:
        keep_rows = keep_rows[: args.top]

    print("# Hypothesis Ceiling Shortlist")
    print()
    print(f"- hypotheses_file: {md_path}")
    print(f"- target_share: {args.share:.6f}")
    print(f"- goal_total_speedup: {args.goal_total_speedup:.4f}")
    print(f"- required_local_speedup: {req_speedup:.4f}")
    print(f"- required_local_gain_pct: {required_gain_pct:.2f}")
    print()

    print("## Keep")
    print()
    print("| id | max_gain_pct | expected_impact | idea |")
    print("|---|---:|---|---|")
    for row in keep_rows:
        print(
            f"| {row['id']} | {float(row['max_gain_pct']):.2f} | {row['expected']} | {row['idea']} |"
        )

    print()
    print("## Drop")
    print()
    print("| id | max_gain_pct | expected_impact | idea |")
    print("|---|---:|---|---|")
    for row in drop_rows:
        print(
            f"| {row['id']} | {float(row['max_gain_pct']):.2f} | {row['expected']} | {row['idea']} |"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
