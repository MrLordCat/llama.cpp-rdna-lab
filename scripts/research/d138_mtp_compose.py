"""D138 7(c): composition of the MTP verify graph (spec=mtp, graphs disabled).

Usage: python build_logs/tmp_d138_mtpcompose.py <server.log> [graph_idx]
"""
import re
import sys
from collections import defaultdict

GRAPH = re.compile(r"NODE_TIMING_GRAPH: graph_idx=(\d+) n_nodes=(\d+)")
NODE = re.compile(
    r"NODE_TIMING: idx=(\d+) kind=(\S+) .*?op=(\S+) name=(\S+) stream=(\d+) "
    r"type=(\S+) src0_type=(\S+) .*?ne=\(([\d,\-]+)\).*?nbytes=(\d+) src0_nbytes=(\d+).*?"
    r"total_ms=([\d.]+)"
)


def main() -> None:
    path = sys.argv[1]
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    idx = None
    per_graph = defaultdict(lambda: [0, defaultdict(lambda: [0, 0, 0.0, 0])])
    fa = defaultdict(lambda: [0, 0, 0])
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "NODE_TIMING" not in line:
                continue
            mg = GRAPH.search(line)
            if mg:
                idx = int(mg.group(1))
                continue
            m = NODE.search(line)
            if not m or idx is None:
                continue
            op, ne, nb, s0nb, t = m.group(3), m.group(8), int(m.group(9)), int(m.group(10)), float(m.group(11))
            entry = per_graph[idx]
            entry[0] = int(GRAPH.search(line).group(0).split('n_nodes=')[1]) if False else entry[0]
            row = entry[1][op]
            row[0] += 1
            row[1] += nb
            row[2] += t
            row[3] += s0nb
            if op == "FLASH_ATTN_EXT":
                key = ne
                f = fa[key]
                f[0] += 1
                f[1] += nb
                f[2] += t

    for gidx, (_, ops) in sorted(per_graph.items()):
        if want and gidx != want:
            continue
        total_nb = sum(v[1] for v in ops.values())
        total_t = sum(v[2] for v in ops.values())
        print(f"\n=== graph {gidx}: ops={len(ops)} nodes={sum(v[0] for v in ops.values())} "
              f"bytes={total_nb/1e9:.2f} GB sync_sum={total_t:.0f} ms ===")
        for op, (cnt, nb, t, s0nb) in sorted(ops.items(), key=lambda kv: -kv[1][2])[:10]:
            print(f"  {op:16} n={cnt:5} bytes={nb/1e6:9.1f} MB src0={s0nb/1e6:9.1f} MB sync={t:8.1f} ms")
    if fa:
        print("\nFLASH_ATTN_EXT shapes (all logged graphs):")
        for ne, (cnt, nb, t) in sorted(fa.items(), key=lambda kv: -kv[1][0])[:8]:
            print(f"  ne=({ne}) n={cnt} qk_bytes={nb/1e6:.1f} MB sync={t:.1f} ms")


if __name__ == "__main__":
    main()
