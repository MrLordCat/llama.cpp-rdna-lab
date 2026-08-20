#!/usr/bin/env python3
"""Fit a fixed-per-dispatch-overhead model to GGML_VK_PERF_LOGGER output.

The serialized perf logger inserts a full pipeline barrier after every node
(vk_backend_execution.inc: writeTimestamp + ggml_vk_sync_buffers when
GGML_VK_PERF_LOGGER_CONCURRENT is unset).  Every dispatch therefore pays a
full wave ramp-up and drain with no overlap against its neighbours, i.e. a
roughly constant additive cost.  Under that model

    t_measured = bytes / BW_true + t_overhead

so short kernels look bandwidth-starved and long ones look perfect, purely
as an artifact.  This script fits (BW_true, t_overhead) over all weight-
streaming MUL_MAT_VEC kernels in one logger block and reports the residuals.
"""

import re
import sys
from collections import OrderedDict

# bytes per element for the quant types we care about
BPW = {
    "q4_K": 144.0 / 256.0,
    "q6_K": 210.0 / 256.0,
    "q8_0": 34.0 / 32.0,
    "f16": 2.0,
    "f32": 4.0,
}

LINE = re.compile(
    r"^(?P<name>.*?MUL_MAT_VEC\s+(?P<type>\S+)\s+"
    r"m=(?P<m>\d+)\s+n=(?P<n>\d+)\s+k=(?P<k>\d+)"
    r"(?:\s+n_expert=\d+)?(?:\s+batch=(?P<batch>\d+))?)"
    r":\s+(?P<count>\d+)\s+x\s+(?P<us>[\d.eE+-]+)\s+us"
)


def parse_blocks(path):
    blocks, cur = [], None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "Vulkan Timings" in line:
                cur = OrderedDict()
                blocks.append(cur)
                continue
            if cur is None:
                continue
            m = LINE.match(line.strip())
            if not m:
                continue
            batch = int(m.group("batch") or 1)
            rec = {
                "type": m.group("type"),
                "m": int(m.group("m")),
                "k": int(m.group("k")),
                "n": int(m.group("n")),
                "batch": batch,
                "count": int(m.group("count")),
                "us": float(m.group("us")),
            }
            cur[m.group("name")] = rec
    return blocks


def weight_bytes(rec):
    bpw = BPW.get(rec["type"])
    if bpw is None:
        return None
    # weight matrix is m x k, replicated over batch for the attention matmuls
    return rec["m"] * rec["k"] * rec["batch"] * bpw


def fit(points):
    """Least squares on t = bytes * (1/BW) + ovh -> classic linear fit."""
    n = len(points)
    sx = sum(b for b, _ in points)
    sy = sum(t for _, t in points)
    sxx = sum(b * b for b, _ in points)
    sxy = sum(b * t for b, t in points)
    den = n * sxx - sx * sx
    if den == 0:
        return None
    slope = (n * sxy - sx * sy) / den          # us per byte
    inter = (sy - slope * sx) / n              # us
    return slope, inter


def main(path):
    blocks = parse_blocks(path)
    if not blocks:
        print("no logger blocks found")
        return

    # Pick the block with the most distinct weight-streaming kernels, and
    # among ties the fastest one (the GPU-resident run rather than the
    # CPU-spilling one).
    scored = []
    for i, b in enumerate(blocks):
        pts = [(weight_bytes(r), r["us"]) for r in b.values() if weight_bytes(r)]
        big = [p for p in pts if p[0] > 1e6]
        if len(big) >= 3:
            scored.append((len(big), -sum(t for _, t in big), i, b))
    if not scored:
        print("no usable block")
        return
    scored.sort(reverse=True)
    _, _, idx, block = scored[0]
    print(f"using logger block #{idx} of {len(blocks)}\n")

    pts = []
    rows = []
    for name, r in block.items():
        wb = weight_bytes(r)
        if not wb or wb < 1e6:      # skip tiny/attention-cache kernels
            continue
        pts.append((wb, r["us"]))
        rows.append((name, wb, r["us"]))

    res = fit(pts)
    if not res:
        print("degenerate fit")
        return
    slope, inter = res
    bw = 1e6 / slope / 1e9 * 1e3    # us/byte -> GB/s
    print(f"fitted true bandwidth : {bw:8.1f} GB/s")
    print(f"fitted fixed overhead : {inter:8.2f} us per dispatch\n")

    hdr = f"{'kernel':<52}{'MB':>9}{'t_us':>10}{'naive':>9}{'corrected':>11}"
    print(hdr)
    print("-" * len(hdr))
    for name, wb, us in sorted(rows, key=lambda r: r[2]):
        naive = wb / (us * 1e-6) / 1e9
        corr_t = us - inter
        corr = wb / (corr_t * 1e-6) / 1e9 if corr_t > 0 else float("nan")
        short = name.replace("MUL_MAT_ADD ", "").replace("MUL_MAT_VEC ", "")
        print(f"{short:<52}{wb/1e6:9.1f}{us:10.1f}{naive:9.0f}{corr:11.0f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build_logs/d105-p2-perflog-serial.txt")
