#!/usr/bin/env python3
"""Diff two GGML_VK_PERF_LOGGER captures kernel by kernel.

Each logger block prints "<name>: <count> x <mean> us = <total> us".  The same
name can appear in several blocks (one per device / per print), so we keep the
fastest observation of each name, which corresponds to the GPU-resident run.
"""

import re
import sys
from collections import defaultdict

LINE = re.compile(r"^(?P<name>.+?):\s+(?P<count>\d+)\s+x\s+(?P<us>[\d.eE+-]+)\s+us\s+=")

BPW = {"q4_K": 144 / 256, "q6_K": 210 / 256, "q8_0": 34 / 32, "f16": 2.0, "f32": 4.0}
SHAPE = re.compile(r"(?P<type>\S+)\s+m=(?P<m>\d+)\s+n=(?P<n>\d+)\s+k=(?P<k>\d+)")


def load(path, min_count=16):
    best = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = LINE.match(line.strip())
            if not m:
                continue
            if int(m.group("count")) < min_count:
                continue
            name, us = m.group("name"), float(m.group("us"))
            if name not in best or us < best[name]:
                best[name] = us
    return best


def bytes_of(name):
    m = SHAPE.search(name)
    if not m:
        return None
    bpw = BPW.get(m.group("type"))
    if bpw is None:
        return None
    return int(m.group("m")) * int(m.group("k")) * bpw


def main(a_path, b_path):
    a, b = load(a_path), load(b_path)
    common = [k for k in a if k in b]
    common.sort(key=lambda k: -(a[k] - b[k]))

    hdr = f"{'kernel':<50}{'base us':>10}{'large us':>10}{'delta':>9}{'base GB/s':>11}{'large GB/s':>11}"
    print(hdr)
    print("-" * len(hdr))
    tot_a = tot_b = 0.0
    for k in common:
        if a[k] < 1.0 and b[k] < 1.0:
            continue
        tot_a += a[k]
        tot_b += b[k]
        nb = bytes_of(k)
        ga = f"{nb / (a[k] * 1e-6) / 1e9:11.0f}" if nb else " " * 11
        gb = f"{nb / (b[k] * 1e-6) / 1e9:11.0f}" if nb else " " * 11
        short = k.replace("MUL_MAT_ADD ", "+").replace("MUL_MAT_VEC ", "")
        print(f"{short:<50}{a[k]:10.2f}{b[k]:10.2f}{(b[k]-a[k])/a[k]*100:8.1f}%{ga}{gb}")
    print("-" * len(hdr))
    print(f"{'sum of per-kernel means':<50}{tot_a:10.2f}{tot_b:10.2f}{(tot_b-tot_a)/tot_a*100:8.1f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
