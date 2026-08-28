#!/usr/bin/env python3
"""D136: parse GGML_VK_PERF_LOGGER blocks, isolate decode graphs, compare fork vs stock."""
import re, sys, collections

def parse(path):
    blocks = []
    cur = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("----------------") or "Vulkan Timings:" in line:
                if cur is not None:
                    blocks.append(cur)
                cur = collections.OrderedDict()
                continue
            if cur is None:
                continue
            m = re.match(r"^([^:]+): (\d+) x ([\d.]+) us = ([\d.]+) us(?: \(([\d.]+) GFLOPS/s\))?\s*$", line)
            if m:
                cur[m.group(1)] = (int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)) if m.group(5) else None)
                continue
            m = re.match(r"^Total time: ([\d.]+) us\.\s*$", line)
            if m:
                cur["__total__"] = float(m.group(1))
    if cur is not None:
        blocks.append(cur)
    return blocks

def decode_blocks(blocks):
    out = []
    for b in blocks:
        if any(k.startswith("FLASH_ATTN_EXT") and ",  q(256,1," in k for k in b):
            out.append(b)
    return out

def summarize(name, blocks):
    print(f"== {name}: decode blocks={len(blocks)} ==")
    agg = collections.defaultdict(lambda: [0, 0.0])  # instances, total_us
    for b in blocks:
        for k, v in b.items():
            if k == "__total__":
                continue
            agg[k][0] += v[0]  # instances
            agg[k][1] += v[2]  # total us
    tot = sum(b.get("__total__", 0.0) for b in blocks)
    print(f"Total decode GPU time (sum of blocks): {tot/1000:.1f} ms over {len(blocks)} tokens")
    rows = sorted(agg.items(), key=lambda kv: -kv[1][1])
    for k, (n, t) in rows[:22]:
        print(f"  {t/1000:8.1f} ms  {n:4d} calls  {k}")
    return tot, agg

if __name__ == "__main__":
    for label, path in [("FORK", sys.argv[1]), ("STOCK", sys.argv[2])]:
        blocks = parse(path)
        db = decode_blocks(blocks)
        summarize(label, db)
        print()
