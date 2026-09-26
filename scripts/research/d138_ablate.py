"""D138 (C): per-class GPU cost inside graphs via ablation + device timing.

Reads server.log files, extracts GGML_TRACE_CUDA_GRAPH_DEVICE_TIMING lines and
reports the median device_ms per (dev, nodes) for the tail of the run, plus the
decode_tps from the sibling metrics.csv.

Usage: python tmp_d138_abl_gpu.py <prefix> [tail_n]
"""
from __future__ import annotations

import glob
import os
import re
import statistics
import sys
from collections import defaultdict

DEV = re.compile(r"dev=(\d+).*?nodes=(\d+).*?use=(\d+) update=(\d+) device_ms=([\d.]+)")


def median_dev_ms(log_path: str, tail_n: int = 200) -> dict[int, float]:
    per_dev: dict[int, list[float]] = defaultdict(list)
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "GGML_TRACE_CUDA_GRAPH_DEVICE_TIMING:" not in line:
                continue
            m = DEV.search(line)
            if not m:
                continue
            dev, _, use, _, ms = m.groups()
            if use != "1":
                continue
            per_dev[int(dev)].append(float(ms))
    out = {}
    for dev, vals in per_dev.items():
        out[dev] = statistics.median(vals[-tail_n:])
    return out


def metrics_tail(path: str) -> dict[str, str]:
    mf = os.path.join(path, "metrics.csv")
    if not os.path.exists(mf):
        return {}
    lines = [ln for ln in open(mf, encoding="utf-8", errors="replace").read().splitlines() if ln and not ln.startswith("run_name")]
    if not lines:
        return {}
    hdr = open(mf, encoding="utf-8", errors="replace").read().splitlines()[0].split(",")
    last = lines[-1]
    # metrics.csv may contain multi-line fields; take the last line as data row
    parts = last.split(",")
    if len(parts) != len(hdr):
        return {}
    return dict(zip(hdr, parts))


def main() -> None:
    prefix = sys.argv[1]
    tail_n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    dirs = sorted(glob.glob(os.path.join("build_logs", "bench", prefix + "*")))
    print(f"{'run':44} {'lvl':>3} {'dec_tps':>8} {'dev0_ms':>8} {'dev1_ms':>8} {'sum_ms':>7}")
    for d in dirs:
        log = os.path.join(d, "server.log")
        if not os.path.exists(log):
            continue
        dev_ms = median_dev_ms(log, tail_n)
        m = metrics_tail(d)
        name = os.path.basename(d).split("--")[0]
        dec = m.get("decode_tps", "")
        lvl = m.get("level", "")
        s0 = f"{dev_ms.get(0, float('nan')):.2f}" if 0 in dev_ms else "-"
        s1 = f"{dev_ms.get(1, float('nan')):.2f}" if 1 in dev_ms else "-"
        tot = sum(dev_ms.values()) if dev_ms else float("nan")
        print(f"{name:44} {lvl:>3} {dec:>8} {s0:>8} {s1:>8} {tot:7.2f}")


if __name__ == "__main__":
    main()
