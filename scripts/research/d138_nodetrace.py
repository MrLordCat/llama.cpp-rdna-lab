"""D138 (B): aggregate GGML_TRACE_CUDA_NODE_TIMING per op class.

Usage: python tmp_d138_nodetrace.py <server.log>
Rows are split into decode (ne[1] <= 2) and prefill (ne[1] > 2).
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict

ROW = re.compile(
    r"GGML_TRACE_CUDA_NODE_TIMING: idx=(?P<idx>\d+) kind=(?P<kind>\w+) skip=(?P<skip>\d+) "
    r"op=(?P<op>\w+) name=(?P<name>.*?) stream=(?P<stream>\d+) type=(?P<type>\S+) "
    r"src0_type=(?P<src0>\S+) src1_type=(?P<src1>\S+) .*?ne=\((?P<ne0>-?\d+),(?P<ne1>-?\d+),"
    r"(?P<ne2>-?\d+),(?P<ne3>-?\d+)\) .*?capture=(?P<cap>\d+) .*?enqueue_ms=(?P<enq>[\d.]+) "
    r"sync_ms=(?P<sync>[\d.]+) total_ms=(?P<total>[\d.]+)")


def main() -> None:
    path = sys.argv[1]
    detail = len(sys.argv) > 2 and sys.argv[2] == "detail"
    tail = None
    for arg in sys.argv[2:]:
        if arg.startswith("tail="):
            tail = int(arg.split("=", 1)[1])
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    if tail:
        lines = lines[-tail:]
    data: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"n": 0, "sync": 0.0, "total": 0.0, "enq": 0.0})
    detail_data: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(
        lambda: {"n": 0, "sync": 0.0, "total": 0.0})
    for line in lines:
        if "GGML_TRACE_CUDA_NODE_TIMING:" not in line:
            continue
        m = ROW.search(line)
        if not m:
            continue
        d = m.groupdict()
        phase = "decode" if int(d["ne1"]) <= 2 else "prefill"
        key = (phase, d["op"])
        e = data[key]
        e["n"] += 1
        e["sync"] += float(d["sync"])
        e["total"] += float(d["total"])
        e["enq"] += float(d["enq"])
        if detail:
            name = re.sub(r"blk\.\d+\.", "blk.*.", d["name"])
            name = re.sub(r"-\d+$", "-*", name)
            dkey = (phase, d["op"], name, f"{d['src0']}/{d['type']}")
            de = detail_data[dkey]
            de["n"] += 1
            de["sync"] += float(d["sync"])
            de["total"] += float(d["total"])

    if detail:
        for phase in ("decode", "prefill"):
            rows = [(k, v) for k, v in detail_data.items() if k[0] == phase]
            rows.sort(key=lambda kv: -kv[1]["sync"])
            print(f"\n== {phase} detail (top 20 by sync_ms) ==")
            for k, v in rows[:20]:
                print(f"  {k[1]:18} {k[2][:52]:52} {k[3]:16} n={v['n']:6d} sync_s={v['sync']/1000:8.3f} "
                      f"mean={v['sync']/max(1,v['n']):8.4f} ms")
        return

    for phase in ("decode", "prefill"):
        rows = [(k[1], v) for k, v in data.items() if k[0] == phase]
        rows.sort(key=lambda kv: -kv[1]["sync"])
        tot_sync = sum(v["sync"] for _, v in rows)
        tot_total = sum(v["total"] for _, v in rows)
        print(f"\n== {phase}: sum sync_ms = {tot_sync/1000:.2f} s, sum total_ms = {tot_total/1000:.2f} s ==")
        print(f"{'op':22} {'n':>8} {'sync_s':>9} {'sync_share%':>12} {'mean_sync_ms':>13} {'mean_total_ms':>14}")
        for op, v in rows[:22]:
            print(f"{op:22} {v['n']:8d} {v['sync']/1000:9.3f} {100.0*v['sync']/tot_sync:12.2f} "
                  f"{v['sync']/max(1,v['n']):13.3f} {v['total']/max(1,v['n']):14.3f}")


if __name__ == "__main__":
    main()
