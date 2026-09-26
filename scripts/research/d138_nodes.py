"""D138: summarise GGML_TRACE_CUDA_NODE_TIMING output per graph.

Usage: python tmp_d138_nodes.py <server.log> [graph_idx]

The trace is only emitted outside graph capture, and with _SYNC=1 each node's
total_ms includes a stream sync, so absolute values are inflated. What is
usable: the per-node share, the split between enqueue (host) and sync (device
wait), and the node count per op - i.e. where the decode steps are.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict

ROW = re.compile(
    r"NODE_TIMING: idx=(?P<idx>\d+) kind=(?P<kind>\w+) skip=(?P<skip>\d+) op=(?P<op>\S+) name=(?P<name>\S+) "
    r"stream=(?P<stream>\d+).*?nbytes=(?P<nbytes>\d+).*?"
    r"enqueue_ms=(?P<enq>[\d.]+) sync_ms=(?P<sync>[\d.]+) total_ms=(?P<total>[\d.]+)"
)
GRAPH = re.compile(r"NODE_TIMING_GRAPH: graph_idx=(?P<g>\d+) n_nodes=(?P<n>\d+) active=(?P<a>\d)")


def main() -> None:
    path = sys.argv[1]
    want = int(sys.argv[2]) if len(sys.argv) > 2 else -1

    graphs: dict[int, int] = {}
    rows: list[dict] = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if "NODE_TIMING_GRAPH:" in line:
            m = GRAPH.search(line)
            if m:
                graphs[int(m.group("g"))] = int(m.group("n"))
            continue
        if "NODE_TIMING: idx=" not in line:
            continue
        m = ROW.search(line)
        if not m:
            continue
        d = m.groupdict()
        d["total"] = float(d["total"])
        d["enq"] = float(d["enq"])
        d["sync"] = float(d["sync"])
        rows.append(d)

    if want < 0:
        print("graphs seen (idx -> n_nodes):", graphs)
        print("total node rows:", len(rows))
        return

    # the trace has no graph id per row; use the position of each graph header
    # instead: re-read and slice.
    per_graph: dict[int, list[dict]] = defaultdict(list)
    g = None
    for line in open(path, encoding="utf-8", errors="replace"):
        if "NODE_TIMING_GRAPH:" in line:
            m = GRAPH.search(line)
            if m:
                g = int(m.group("g"))
            continue
        if "NODE_TIMING: idx=" not in line or g is None:
            continue
        m = ROW.search(line)
        if not m:
            continue
        d = m.groupdict()
        d["total"] = float(d["total"])
        d["enq"] = float(d["enq"])
        d["sync"] = float(d["sync"])
        per_graph[g].append(d)

    if want not in per_graph:
        print("graph", want, "not found; have", sorted(per_graph))
        return

    nodes = per_graph[want]
    total = sum(n["total"] for n in nodes)
    enq = sum(n["enq"] for n in nodes)
    print(f"graph {want}: {len(nodes)} nodes  total_ms={total:.2f}  enqueue_ms={enq:.2f}  sync_ms={total-enq:.2f}")

    by_op: dict[str, list] = defaultdict(lambda: [0, 0.0, 0.0])
    for n in nodes:
        e = by_op[n["op"]]
        e[0] += 1
        e[1] += n["total"]
        e[2] = max(e[2], n["total"])
    print(f"\n{'op':16}{'n':>6}{'sum_ms':>10}{'share':>8}{'max_ms':>9}")
    for op, (cnt, s, mx) in sorted(by_op.items(), key=lambda kv: -kv[1][1])[:14]:
        print(f"{op:16}{cnt:6d}{s:10.2f}{100.0*s/total:7.1f}%{mx:9.3f}")

    # hottest individual nodes (by total_ms)
    print("\nhottest nodes:")
    for n in sorted(nodes, key=lambda x: -x["total"])[:10]:
        print(f"  {n['total']:8.3f} ms  enq={n['enq']:6.3f} sync={n['sync']:8.3f}  {n['op']:14} {n['name'][:56]}")

    # group by tensor-name prefix (layer-agnostic)
    by_pref: dict[str, list] = defaultdict(lambda: [0, 0.0])
    for n in nodes:
        pref = n["name"].split("-", 1)[-1] if n["name"].startswith("blk") else n["name"]
        pref = pref.split(".", 1)[-1] if pref.startswith("blk") else pref
        e = by_pref[pref[:28]]
        e[0] += 1
        e[1] += n["total"]
    print(f"\n{'name-prefix':30}{'n':>6}{'sum_ms':>10}{'share':>8}")
    for p, (cnt, s) in sorted(by_pref.items(), key=lambda kv: -kv[1][1])[:14]:
        print(f"{p:30}{cnt:6d}{s:10.2f}{100.0*s/total:7.1f}%")


if __name__ == "__main__":
    main()
