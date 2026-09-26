"""D138 step 1: per-shape MMVQ census from GGML_TRACE_MMVQ_TIMING[_SYNC] server logs.

Usage: python tmp_d138_census.py <server.log> [--top N]
Reads only; prints shape groups sorted by total measured kernel time (capture=0 rows).
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict

# bytes per 256 weights for the quant types seen on the Qwen3.8-27B UD-Q4_K_M lane
BPW256 = {
    "q2_K": 84, "q3_K": 110, "q4_K": 144, "q5_K": 176, "q6_K": 210,
    "iq4_nl": 144, "iq4_xs": 136, "q8_0": 272, "mxlf": 136,
}

ROW = re.compile(
    r"timing type=(?P<tnum>\d+)/(?P<type>\S+) ncols_dst=(?P<ncols>\d+) small_k=(?P<small_k>\d+) "
    r"fusion=(?P<fusion>\d+) ncols_x=(?P<ncols_x>\d+) grid=\((?P<gx>\d+),(?P<gy>\d+),(?P<gz>\d+)\) "
    r"block=\((?P<bx>\d+),(?P<by>\d+),(?P<bz>\d+)\).*?regs=(?P<regs>-?\d+) "
    r".*?occupancy_pct=(?P<occ>[\d.]+) waves_per_sm=(?P<waves>[\d.]+)"
    r".*?sync_applied=(?P<sync>\d+) capture=(?P<capture>\d+) "
    r"pre_sync_ms=(?P<pre_sync>[\d.]+) enqueue_ms=(?P<enqueue>[\d.]+) "
    r"sync_ms=(?P<sync_ms>[\d.]+) total_ms=(?P<total>[\d.]+)"
)

def main() -> None:
    path = sys.argv[1]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25

    groups: dict[tuple, list[float]] = defaultdict(list)
    meta: dict[tuple, dict] = {}
    seen = 0
    for line in open(path, encoding="utf-8", errors="replace"):
        if "timing type=" not in line:
            continue
        m = ROW.search(line)
        if not m:
            continue
        seen += 1
        d = m.groupdict()
        key = (d["type"], int(d["ncols"]), int(d["small_k"]), int(d["fusion"]),
               int(d["ncols_x"]), int(d["gx"]), int(d["by"]), int(d["capture"]))
        groups[key].append(float(d["total"]))
        meta[key] = d
    print(f"parsed {seen} trace rows, {len(groups)} groups\n")

    hdr = (f"{'type':7} {'ncols':>5} {'sk':>2} {'fus':>3} {'K':>6} {'grid.x':>7} {'b.y':>4} "
           f"{'cap':>3} {'n':>5} {'med_ms':>9} {'sum_ms':>10} {'GB/s':>8} {'waves':>7} {'occ%':>6} {'regs':>5}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for key, times in groups.items():
        t, ncols, small_k, fusion, ncols_x, gx, by, capture = key
        rows_per_block = by if (small_k and ncols == 1) else 1
        nrows = gx * rows_per_block
        bpb = BPW256.get(t, 0)
        nbytes = nrows * (ncols_x // 256) * bpb
        med = statistics.median(times)
        total = sum(times)
        gbps = nbytes / (med / 1e3) / 1e9 if med > 0 else 0.0
        rows.append((total, key, med, total, gbps, nrows, nbytes, len(times)))
    rows.sort(reverse=True)
    for _, key, med, total, gbps, nrows, nbytes, n in rows[:top]:
        t, ncols, small_k, fusion, ncols_x, gx, by, capture = key
        d = meta[key]
        print(f"{t:7} {ncols:5} {small_k:2} {fusion:3} {ncols_x:6} {gx:7} {by:4} "
              f"{capture:3} {n:5} {med:9.3f} {total:10.2f} {gbps:8.1f} "
              f"{float(d['waves']):7.2f} {float(d['occ']):6.1f} {int(d['regs']):5}")

    print("\n-- capture=0 only totals --")
    tot0 = sum(t for _, k, _, t, *_ in rows if k[7] == 0)
    tot1 = sum(t for _, k, _, t, *_ in rows if k[7] == 1)
    print(f"capture=0 sum: {tot0:.1f} ms; capture=1 sum: {tot1:.1f} ms")

    by_type: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])  # sum_ms, bytes, calls
    for _, key, med, total, gbps, nrows, nbytes, n in rows:
        t = key[0]
        by_type[t][0] += total
        by_type[t][1] += nbytes * n
        by_type[t][2] += n
    print(f"\n-- by type (share of measured MMVQ kernel time) --")
    print(f"{'type':8} {'sum_ms':>10} {'share%':>7} {'GB':>9} {'calls':>7} {'avg GB/s':>9}")
    grand = sum(v[0] for v in by_type.values())
    for t, (s, b, c) in sorted(by_type.items(), key=lambda kv: -kv[1][0]):
        share = 100.0 * s / grand if grand else 0.0
        agg = b / (s / 1e3) / 1e9 if s > 0 else 0.0
        print(f"{t:8} {s:10.1f} {share:7.1f} {b/1e9:9.2f} {int(c):7d} {agg:9.1f}")
    print(f"{'TOTAL':8} {grand:10.1f}")


if __name__ == "__main__":
    main()
