#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


NUM = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"

ROCM_ROUTE_RE = re.compile(
    r"GGML_TRACE_CUDA_MUL_MAT_ROUTE: "
    r"route=(?P<route>\S+).*?"
    r"src0_type=(?P<src0_type>\S+).*?"
    r"src1_type=(?P<src1_type>\S+).*?"
    r"dst_type=(?P<dst_type>\S+).*?"
    r"src0_ne=(?P<src0_ne>\([^)]+\)).*?"
    r"src1_ne=(?P<src1_ne>\([^)]+\)).*?"
    r"dst_ne=(?P<dst_ne>\([^)]+\))"
)

ROCM_NODE_RE = re.compile(
    rf"GGML_TRACE_CUDA_NODE_TIMING: idx=(?P<idx>\d+) "
    rf"kind=(?P<kind>\S+) .*?op=(?P<op>\S+) .*?"
    rf"ne=(?P<ne>\([^)]+\)).*?total_ms=(?P<total>{NUM})"
)

ROCM_MMVQ_TIMING_RE = re.compile(
    rf"operator\(\): timing type=\d+/(?P<qtype>\S+) "
    rf"ncols_dst=(?P<ncols_dst>\d+) .*?"
    rf"small_k=(?P<small_k>[01]) fusion=(?P<fusion>[01]) ncols_x=(?P<k>\d+) "
    rf"grid=\((?P<m>\d+),(?P<n>\d+),(?P<z>\d+)\).*?"
    rf"block=\((?P<block_x>\d+),(?P<block_y>\d+),(?P<block_z>\d+)\).*?"
    rf"total_ms=(?P<total>{NUM})"
)

VULKAN_MAT_RE = re.compile(
    rf"^(?P<op>MUL_MAT_ADD MUL_MAT_VEC|MUL_MAT_VEC|MUL_MAT)\s+"
    rf"(?P<qtype>\S+)\s+m=(?P<m>\d+)\s+n=(?P<n>\d+)\s+k=(?P<k>\d+):\s+"
    rf"(?P<calls>\d+)\s+x\s+(?P<avg>{NUM})\s+us\s+=\s+(?P<total>{NUM})\s+us"
)

VULKAN_TOTAL_RE = re.compile(rf"^Total time:\s+(?P<total>{NUM})\s+us\.")


@dataclass
class RouteInfo:
    route: str
    src0_type: str
    src1_type: str
    dst_type: str
    src0_ne: tuple[int, ...]
    src1_ne: tuple[int, ...]
    dst_ne: tuple[int, ...]


@dataclass
class Row:
    backend: str
    bucket: str
    qtype: str
    shape: str
    calls: int
    total_ms: float


@dataclass
class MmvqTiming:
    qtype: str
    ncols_dst: int
    small_k: int
    fusion: int
    m: int
    n: int
    k: int
    block_y: int

    @property
    def semantic_m(self) -> int:
        # MMVQ timing logs the CUDA grid x dimension, not the logical output
        # rows. In the small-k ncols_dst=1 path rows_per_block matches block.y,
        # so the logical row count is grid.x * block.y.
        if self.ncols_dst == 1 and self.small_k:
            return self.m * self.block_y
        return self.m


def parse_ne(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.strip("()").split(","))


def rocm_shape(info: RouteInfo) -> str:
    k = info.src0_ne[0] if len(info.src0_ne) >= 1 else 0
    m = info.src0_ne[1] if len(info.src0_ne) >= 2 else 0
    n = info.dst_ne[1] if len(info.dst_ne) >= 2 else 1
    return f"m={m} n={n} k={k}"


def parse_rocm(path: Path, skip_sections: int) -> list[Row]:
    rows: list[Row] = []
    pending: RouteInfo | None = None
    pending_timing: MmvqTiming | None = None
    section = -1

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := ROCM_ROUTE_RE.search(line):
            pending = RouteInfo(
                route=match.group("route"),
                src0_type=match.group("src0_type"),
                src1_type=match.group("src1_type"),
                dst_type=match.group("dst_type"),
                src0_ne=parse_ne(match.group("src0_ne")),
                src1_ne=parse_ne(match.group("src1_ne")),
                dst_ne=parse_ne(match.group("dst_ne")),
            )
            continue

        if match := ROCM_MMVQ_TIMING_RE.search(line):
            pending_timing = MmvqTiming(
                qtype=match.group("qtype"),
                ncols_dst=int(match.group("ncols_dst")),
                small_k=int(match.group("small_k")),
                fusion=int(match.group("fusion")),
                m=int(match.group("m")),
                n=int(match.group("n")),
                k=int(match.group("k")),
                block_y=int(match.group("block_y")),
            )
            continue

        match = ROCM_NODE_RE.search(line)
        if not match:
            continue

        if int(match.group("idx")) == 0:
            section += 1

        if match.group("op") != "MUL_MAT" or pending is None:
            if match.group("op") == "MUL_MAT" and pending_timing is not None and pending_timing.fusion:
                if section >= skip_sections:
                    rows.append(
                        Row(
                            backend="ROCm",
                            bucket=f"mul_mat_vec_q_fused {pending_timing.qtype}->f32",
                            qtype=pending_timing.qtype,
                            shape=f"m={pending_timing.semantic_m} n={pending_timing.n} k={pending_timing.k}",
                            calls=1,
                            total_ms=float(match.group("total")),
                        )
                    )
                pending_timing = None
            continue

        if section >= skip_sections:
            rows.append(
                Row(
                    backend="ROCm",
                    bucket=f"{pending.route} {pending.src0_type}->{pending.dst_type}",
                    qtype=pending.src0_type,
                    shape=rocm_shape(pending),
                    calls=1,
                    total_ms=float(match.group("total")),
                )
            )
        pending = None
        pending_timing = None

    return rows


def parse_vulkan(path: Path, min_section_us: float | None, max_section_us: float | None) -> list[Row]:
    rows: list[Row] = []
    section_rows: list[Row] = []

    def keep_section(total_us: float) -> bool:
        if min_section_us is not None and total_us < min_section_us:
            return False
        if max_section_us is not None and total_us > max_section_us:
            return False
        return True

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := VULKAN_MAT_RE.search(line):
            op = match.group("op").replace("MUL_MAT_ADD MUL_MAT_VEC", "MUL_MAT_ADD_VEC")
            qtype = match.group("qtype")
            shape = f"m={match.group('m')} n={match.group('n')} k={match.group('k')}"
            section_rows.append(
                Row(
                    backend="Vulkan",
                    bucket=f"{op} {qtype}",
                    qtype=qtype,
                    shape=shape,
                    calls=int(match.group("calls")),
                    total_ms=float(match.group("total")) / 1000.0,
                )
            )
            continue

        if match := VULKAN_TOTAL_RE.search(line):
            total_us = float(match.group("total"))
            if keep_section(total_us):
                rows.extend(section_rows)
            section_rows = []

    return rows


def aggregate(rows: list[Row], key_fn) -> list[tuple[str, int, float]]:
    calls: defaultdict[str, int] = defaultdict(int)
    total_ms: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        key = key_fn(row)
        calls[key] += row.calls
        total_ms[key] += row.total_ms
    return sorted(((key, calls[key], total_ms[key]) for key in total_ms), key=lambda item: item[2], reverse=True)


def print_table(title: str, rows: list[tuple[str, int, float]], total_ms: float, top: int) -> None:
    print(f"## {title}")
    print()
    print("| Key | Calls | Total ms | Share |")
    print("| --- | ---: | ---: | ---: |")
    for key, calls, row_ms in rows[:top]:
        share = 100.0 * row_ms / total_ms if total_ms else 0.0
        print(f"| `{key}` | {calls} | {row_ms:.2f} | {share:.2f}% |")
    print()


def print_shape_delta(rocm_rows: list[Row], vulkan_rows: list[Row], top: int) -> None:
    rocm_total = sum(row.total_ms for row in rocm_rows)
    vulkan_total = sum(row.total_ms for row in vulkan_rows)
    rocm_by_shape = {key: (calls, total_ms) for key, calls, total_ms in aggregate(rocm_rows, lambda row: f"{row.qtype} {row.shape}")}
    vulkan_by_shape = {key: (calls, total_ms) for key, calls, total_ms in aggregate(vulkan_rows, lambda row: f"{row.qtype} {row.shape}")}
    keys = set(rocm_by_shape) | set(vulkan_by_shape)

    ranked: list[tuple[str, int, float, int, float, float]] = []
    for key in keys:
        rcalls, rms = rocm_by_shape.get(key, (0, 0.0))
        vcalls, vms = vulkan_by_shape.get(key, (0, 0.0))
        rshare = 100.0 * rms / rocm_total if rocm_total else 0.0
        vshare = 100.0 * vms / vulkan_total if vulkan_total else 0.0
        ranked.append((key, rcalls, rms, vcalls, vms, max(rshare, vshare)))
    ranked.sort(key=lambda item: item[5], reverse=True)

    print("## Normalized Shape Delta")
    print()
    print("| QType / shape | ROCm calls | ROCm ms | ROCm share | Vulkan calls | Vulkan ms | Vulkan share |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key, rcalls, rms, vcalls, vms, _ in ranked[:top]:
        rshare = 100.0 * rms / rocm_total if rocm_total else 0.0
        vshare = 100.0 * vms / vulkan_total if vulkan_total else 0.0
        print(f"| `{key}` | {rcalls} | {rms:.2f} | {rshare:.2f}% | {vcalls} | {vms:.2f} | {vshare:.2f}% |")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ROCm CUDA-node route trace against Vulkan perf logger decode sections.")
    parser.add_argument("--rocm-log", type=Path, required=True)
    parser.add_argument("--vulkan-log", type=Path, required=True)
    parser.add_argument("--rocm-skip-sections", type=int, default=0)
    parser.add_argument("--vulkan-min-section-us", type=float)
    parser.add_argument("--vulkan-max-section-us", type=float)
    parser.add_argument("--qtype", action="append", help="Filter to one or more quant/source types, e.g. q3_K")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rocm_rows = parse_rocm(args.rocm_log, args.rocm_skip_sections)
    vulkan_rows = parse_vulkan(args.vulkan_log, args.vulkan_min_section_us, args.vulkan_max_section_us)
    if args.qtype:
        qtypes = set(args.qtype)
        rocm_rows = [row for row in rocm_rows if row.qtype in qtypes]
        vulkan_rows = [row for row in vulkan_rows if row.qtype in qtypes]

    if not rocm_rows:
        raise SystemExit("no ROCm rows parsed")
    if not vulkan_rows:
        raise SystemExit("no Vulkan rows parsed")

    rocm_total = sum(row.total_ms for row in rocm_rows)
    vulkan_total = sum(row.total_ms for row in vulkan_rows)

    print("# ROCm / Vulkan Decode Route Delta")
    print()
    print(f"- ROCm log: `{args.rocm_log}`")
    print(f"- Vulkan log: `{args.vulkan_log}`")
    print(f"- ROCm parsed rows: `{len(rocm_rows)}`, total: `{rocm_total:.2f} ms`")
    print(f"- Vulkan parsed rows: `{len(vulkan_rows)}`, total: `{vulkan_total:.2f} ms`")
    print()

    print_table("ROCm By Bucket", aggregate(rocm_rows, lambda row: row.bucket), rocm_total, args.top)
    print_table("Vulkan By Bucket", aggregate(vulkan_rows, lambda row: row.bucket), vulkan_total, args.top)
    print_shape_delta(rocm_rows, vulkan_rows, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
