"""D138 (B): summary of ablation runs.

Collects every build_logs/bench/<prefix>* run whose run.json contains a shot
with decode_tps, and prints one row per run.
"""
from __future__ import annotations

import glob
import json
import os
import sys


def find_shots(obj, out):
    if isinstance(obj, dict):
        if "decode_tps" in obj and "prefill_tps" in obj:
            out.append(obj)
        for v in obj.values():
            find_shots(v, out)
    elif isinstance(obj, list):
        for v in obj:
            find_shots(v, out)


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "d138-abl"
    rows = []
    for d in sorted(glob.glob(os.path.join("build_logs", "bench", prefix + "*"))):
        rj = os.path.join(d, "run.json")
        if not os.path.exists(rj):
            continue
        try:
            with open(rj, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            print(f"{os.path.basename(d):44} LOAD FAIL {exc}")
            continue
        shots = []
        find_shots(data, shots)
        if not shots:
            print(f"{os.path.basename(d):44} no shots (type={data.get('type')})")
            continue
        s = shots[-1]
        name = os.path.basename(d).split("--")[0]
        rows.append((name, s.get("level"), s.get("decode_tps", 0.0), s.get("prefill_tps", 0.0),
                     s.get("aggregate_tps", 0.0), s.get("predicted_n")))
    print(f"{'run':44} {'lvl':>4} {'decode_tps':>10} {'prefill_tps':>11} {'agg_tps':>8} {'n_dec':>6}")
    for name, lvl, dec, pre, agg, nd in rows:
        print(f"{name:44} {str(lvl):>4} {dec:10.2f} {pre:11.1f} {agg:8.2f} {str(nd):>6}")


if __name__ == "__main__":
    main()
