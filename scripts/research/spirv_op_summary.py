#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import re
import shutil
import subprocess
from pathlib import Path


OP_RE = re.compile(r"\b(Op[A-Za-z0-9_]+)\b")
FOCUS_OPS = (
    "OpCooperativeMatrixLoadKHR",
    "OpCooperativeMatrixMulAddKHR",
    "OpCooperativeMatrixStoreKHR",
    "OpCooperativeMatrixLengthKHR",
    "OpTypeCooperativeMatrixKHR",
    "OpControlBarrier",
    "OpLoad",
    "OpStore",
    "OpFConvert",
    "OpFMul",
    "OpIAdd",
    "OpIMul",
    "OpUDiv",
    "OpUMod",
    "OpShiftRightLogical",
    "OpShiftLeftLogical",
    "OpBitwiseAnd",
    "OpBitwiseOr",
    "OpBitwiseXor",
)


def run_spirv_dis(spirv_dis: str, path: Path) -> str:
    completed = subprocess.run(
        [spirv_dis, str(path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"spirv-dis failed for {path}")
    return completed.stdout


def opcode_counts(disassembly: str) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    for line in disassembly.splitlines():
        match = OP_RE.search(line)
        if match:
            counts[match.group(1)] += 1
    return counts


def print_summary(path: Path, counts: collections.Counter[str], top: int) -> None:
    print(f"## {path}")
    print()
    print("### Focus Ops")
    print()
    print("| op | count |")
    print("|---|---:|")
    for op in FOCUS_OPS:
        value = counts.get(op, 0)
        if value:
            print(f"| {op} | {value} |")
    print()

    print("### Top Ops")
    print()
    print("| op | count |")
    print("|---|---:|")
    for op, value in counts.most_common(top):
        print(f"| {op} | {value} |")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SPIR-V opcode counts for Vulkan shader research")
    parser.add_argument("files", nargs="+", help="SPIR-V files to disassemble")
    parser.add_argument("--top", type=int, default=40, help="number of top opcodes to print")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spirv_dis = shutil.which("spirv-dis")
    if spirv_dis is None:
        raise SystemExit("ERROR: spirv-dis not found")

    print("# SPIR-V Opcode Summary")
    print()
    print(f"- spirv-dis: {spirv_dis}")
    print()

    for raw in args.files:
        path = Path(raw)
        if not path.exists():
            print(f"## {path}")
            print()
            print("missing")
            print()
            continue
        disassembly = run_spirv_dis(spirv_dis, path)
        print_summary(path, opcode_counts(disassembly), args.top)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())