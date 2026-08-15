#!/usr/bin/env python3
"""W13 C1: aggregate GGML_TRACE_MMVQ_* lines from a server log.

Usage: python scripts/research/w13_mmvq_trace_stats.py <server.log>
Groups decode (ncols_dst=1) MMVQ launches by (type, ncols_x, small_k, fusion)
and prints median total_ms and mean occupancy/waves/regs/shared.
"""
import re
import sys
from collections import defaultdict
from statistics import median, mean


def main() -> int:
    path = sys.argv[1]
    pat = re.compile(
        r"timing type=(\d+)/(\w+) ncols_dst=(\d+) small_k=(\d+) fusion=(\d+) "
        r"ncols_x=(\d+) grid=\((\d+),(\d+),(\d+)\) block=\((\d+),(\d+),(\d+)\) "
        r"trace_resources=(\d+) block_threads=(\d+) nbytes_shared=(\d+) "
        r"static_shared=(\d+) shared_total=(\d+) smpbo=(\d+) shared_pct=([\d.]+) "
        r"regs=(\d+) max_dyn_shared=(-?\d+) max_blocks_per_sm=(\d+) "
        r"max_threads_per_sm=(\d+) occupancy_pct=([\d.]+) waves_per_sm=([\d.]+) "
        r"sync_req=(\d+) pre_sync_applied=(\d+) sync_applied=(\d+) capture=(\d+) "
        r"pre_sync_ms=([\d.]+) enqueue_ms=([\d.]+) sync_ms=([\d.]+) "
        r"total_ms=([\d.]+)"
    )
    groups = defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            (ttype, tname, ncols_dst, small_k, fusion, ncols_x, gx, gy, gz,
             bx, by, bz, tr_res, block_threads, nbytes_shared, static_shared,
             shared_total, smpbo, shared_pct, regs, max_dyn_shared,
             max_blocks_per_sm, max_threads_per_sm, occ_pct, waves,
             sync_req, pre_sync, sync_app, capture, pre_sync_ms, enqueue_ms,
             sync_ms, total_ms) = m.groups()
            if ncols_dst != "1":
                continue  # decode stream only
            key = (tname, int(ncols_x), int(small_k), int(fusion))
            groups[key].append(
                (int(regs), int(shared_total), float(occ_pct), float(waves),
                 float(total_ms), int(gx), int(max_blocks_per_sm),
                 int(max_dyn_shared), int(max_threads_per_sm))
            )

    print(f"{'type':8s} {'ncols_x':>7s} {'sk':>2s} {'fus':>3s} {'n':>6s} "
          f"{'med_ms':>8s} {'occ%':>6s} {'waves':>6s} {'regs':>4s} "
          f"{'shared':>7s} {'mBPSM':>5s} {'mDynSh':>7s} {'mTPSM':>6s} {'grid_x':>7s}")
    for key in sorted(groups.keys(), key=lambda k: (k[1], k[2])):
        vals = groups[key]
        tname, ncols_x, small_k, fusion = key
        ms = median(v[4] for v in vals)
        occ = mean(v[2] for v in vals)
        waves = mean(v[3] for v in vals)
        regs = mean(v[0] for v in vals)
        shared = mean(v[1] for v in vals)
        gx = mean(v[5] for v in vals)
        mbpsm = mean(v[6] for v in vals)
        mdynsh = mean(v[7] for v in vals)
        mtpsm = mean(v[8] for v in vals)
        print(f"{tname:8s} {ncols_x:7d} {small_k:2d} {fusion:3d} {len(vals):6d} "
              f"{ms:8.4f} {occ:6.1f} {waves:6.2f} {regs:4.0f} {shared:7.0f} "
              f"{mbpsm:5.0f} {mdynsh:7.0f} {mtpsm:6.0f} {gx:7.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
