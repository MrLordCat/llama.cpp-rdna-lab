#!/usr/bin/env python3
"""RPC timeline analyzer for GGML_RPC_TIMELINE=1 traces.

Usage:
    python rpc_timeline_analyze.py --client build_logs/.../diag.server.log \
        --server /tmp/rpc-srv-tl2.log [--top 10]

The client and server use their own steady clocks, so the tool reports each
side's aggregates separately plus command-level statistics. To isolate the
prefill loop, use the "ubatch cycle" view: time between consecutive
GRAPH_COMPUTE commands on the client, and consecutive GRAPH_COMPUTE_ASYNC
handling on the server.

Formats parsed:
    RPC_TL|cli|<cmd_id>|<name>|<bytes>|<send_ms>|<rsp_ms>|<gap_ms>|t=<wall_ms>
    RPC_TL|srv|<cmd_id>|<name>|<bytes>|<idle_ms>|<proc_ms>|<flush_ms>|t=<wall_ms>
"""

import argparse
import re
from collections import defaultdict

CLI_RE = re.compile(
    r"RPC_TL\|cli\|(\d+)\|(\S+)\|(\d+)\|([\d.]+)\|([\d.]+)\|([\d.]+)\|t=([\d.]+)")
SRV_RE = re.compile(
    r"RPC_TL\|srv\|(\d+)\|(\S+)\|(\d+)\|([\d.]+)\|([\d.]+)\|([\d.]+)\|t=([\d.]+)")


def load(fn, regex, kind, after_marker=None):
    rows = []
    if not fn:
        return rows
    active = after_marker is None
    for line in open(fn, encoding="utf-8", errors="replace"):
        if not active:
            active = after_marker in line
            continue
        m = regex.search(line)
        if m:
            g = m.groups()
            rows.append({
                "cmd": g[1],
                "bytes": int(g[2]),
                "a": float(g[3]),
                "b": float(g[4]),
                "c": float(g[5]),
                "t": float(g[6]),
            })
    return rows


def summarize(rows, kind, top_n, cycles):
    if not rows:
        print(f"[{kind}] no timeline rows found")
        return
    # aggregate by command
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for r in rows:
        e = agg[r["cmd"]]
        e[0] += r["a"]
        e[1] += r["b"]
        e[2] += r["c"]
        e[3] += 1
    print(f"[{kind}] {len(rows)} events, rows per command (sum a/b/c ms, n):")
    for name, (a, b, c, n) in sorted(agg.items(), key=lambda kv: -(kv[1][0] + kv[1][1] + kv[1][2])):
        print(f"  {name:24s} n={n:4d}  a={a:8.1f}  b={b:8.1f}  c={c:8.1f}")
    # top rows by the wait field (rsp for client, flush for server)
    wait_idx = 1 if kind == "cli" else 2
    top = sorted(rows, key=lambda r: -r["b" if wait_idx == 1 else "c"])[:top_n]
    wait_name = "rsp_ms" if kind == "cli" else "flush_ms"
    print(f"[{kind}] top waits by {wait_name}:")
    for r in top:
        print(f"  t={r['t']:9.1f} {r['cmd']:24s} {wait_name}={r['b' if wait_idx == 1 else 'c']:8.1f} a={r['a']:7.1f}")
    if cycles and kind == "cli":
        # gap between consecutive GRAPH_COMPUTE (send or recompute) starts
        gcs = [r for r in rows if "GRAPH_COMPUTE" in r["cmd"]]
        deltas = [gcs[i + 1]["t"] - gcs[i]["t"] for i in range(len(gcs) - 1)]
        if deltas:
            deltas.sort()
            from statistics import median
            print(f"[{kind}] graph-to-graph deltas: n={len(deltas)} median={median(deltas):.1f} "
                  f"p90={deltas[int(len(deltas)*0.9)]:.1f} max={deltas[-1]:.1f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client")
    ap.add_argument("--server")
    ap.add_argument("--client-after-marker")
    ap.add_argument("--server-after-marker")
    ap.add_argument("--client-tail-events", type=int)
    ap.add_argument("--server-tail-events", type=int)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--no-cycles", action="store_true")
    args = ap.parse_args()
    cli = load(args.client, CLI_RE, "cli", args.client_after_marker)
    srv = load(args.server, SRV_RE, "srv", args.server_after_marker)
    if args.client_tail_events:
        cli = cli[-args.client_tail_events:]
    if args.server_tail_events:
        srv = srv[-args.server_tail_events:]
    summarize(cli, "cli", args.top, not args.no_cycles)
    summarize(srv, "srv", args.top, not args.no_cycles)


if __name__ == "__main__":
    main()
