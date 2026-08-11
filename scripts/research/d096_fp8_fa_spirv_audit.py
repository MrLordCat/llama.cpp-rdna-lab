#!/usr/bin/env python3
"""Fail-closed structural audit for the canonical Vulkan cm1 FA SPIR-V.

The D096 fp8 transform is allowed to replace only the first cooperative matrix
stage (K * Q -> S). This audit identifies that stage without depending on
numeric result IDs and rejects modules whose cooperative-matrix structure has
drifted. Profiles cover the generated canonical shader, the build-only P2/P3
bases, and their transformed fp8 modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


_RESULT_PREFIX = r"^\s*(%\S+)\s*=\s*"
_FLOAT_TYPE_RE = re.compile(_RESULT_PREFIX + r"OpTypeFloat\s+(\d+)")
_UINT_CONSTANT_RE = re.compile(_RESULT_PREFIX + r"OpConstant\s+%\S+\s+(\d+)")
_COOP_TYPE_RE = re.compile(
    _RESULT_PREFIX
    + r"OpTypeCooperativeMatrixKHR\s+(%\S+)\s+(%\S+)\s+(%\S+)\s+(%\S+)\s+(%\S+)"
)
_COOP_LOAD_RE = re.compile(_RESULT_PREFIX + r"OpCooperativeMatrixLoadKHR\s+(%\S+)\b")
_COOP_MUL_RE = re.compile(_RESULT_PREFIX + r"OpCooperativeMatrixMulAddKHR\s+(%\S+)\b")
_COOP_STORE_RE = re.compile(r"^\s*OpCooperativeMatrixStoreKHR\b")


class AuditError(RuntimeError):
    """The canonical module no longer matches the transform contract."""


@dataclass(frozen=True)
class StageAnchors:
    first_load_line: int
    mul_add_line: int
    store_line: int
    load_count: int
    a_load_count: int
    b_load_count: int
    accumulator_width: int


@dataclass(frozen=True)
class AuditReport:
    source: str
    profile: str
    sha256: str
    line_count: int
    cooperative_type_count: int
    s_stage: StageAnchors
    pv_stage: StageAnchors


@dataclass(frozen=True)
class _CoopType:
    component_width: int
    rows: int
    cols: int
    use: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def audit_disassembly(text: str, source: str = "<memory>", profile: str = "canonical") -> AuditReport:
    lines = text.splitlines()
    float_widths: dict[str, int] = {}
    constants: dict[str, int] = {}
    raw_coop_types: list[tuple[str, str, str, str, str, str]] = []

    for line in lines:
        if match := _FLOAT_TYPE_RE.match(line):
            float_widths[match.group(1)] = int(match.group(2))
        if match := _UINT_CONSTANT_RE.match(line):
            constants[match.group(1)] = int(match.group(2))
        if match := _COOP_TYPE_RE.match(line):
            raw_coop_types.append(match.groups())

    coop_types: dict[str, _CoopType] = {}
    for result_id, component_id, scope_id, rows_id, cols_id, use_id in raw_coop_types:
        _require(component_id in float_widths, f"unknown cooperative component type {component_id}")
        for constant_id in (scope_id, rows_id, cols_id, use_id):
            _require(constant_id in constants, f"unknown cooperative constant {constant_id}")
        _require(constants[scope_id] == 3, "cooperative matrix scope is not Subgroup")
        coop_types[result_id] = _CoopType(
            component_width=float_widths[component_id],
            rows=constants[rows_id],
            cols=constants[cols_id],
            use=constants[use_id],
        )

    _require(
        profile in {"canonical", "p2-base", "fp8-p2", "p3-base", "fp8-p3", "p4-base", "fp8-p4", "p5-base", "fp8-p5"},
        f"unknown audit profile {profile}",
    )
    expected_types = {(32, 2), (16, 0), (16, 1), (16, 2)}
    if profile == "p5-base":
        # P5 already accumulates PV in f32: no f16 accumulator type exists.
        expected_types = {(32, 2), (16, 0), (16, 1)}
    if profile == "fp8-p5":
        # Same set as fp8-p3/p4 minus the f16 accumulator type (PV is f32).
        expected_types = {(32, 2), (16, 0), (16, 1), (8, 0), (8, 1)}
    if profile in {"fp8-p2", "fp8-p3", "fp8-p4"}:
        expected_types |= {(8, 0), (8, 1)}
    actual_types = {(item.component_width, item.use) for item in coop_types.values()}
    _require(len(coop_types) == len(expected_types),
             f"expected {len(expected_types)} cooperative types, found {len(coop_types)}")
    _require(actual_types == expected_types, f"unexpected cooperative types: {sorted(actual_types)}")
    _require(
        all(item.rows == 16 and item.cols == 16 for item in coop_types.values()),
        "cooperative matrices are no longer all 16x16",
    )

    loads: list[tuple[int, str]] = []
    muls: list[tuple[int, str]] = []
    stores: list[int] = []
    for line_number, line in enumerate(lines, start=1):
        if match := _COOP_LOAD_RE.match(line):
            loads.append((line_number, match.group(2)))
        if match := _COOP_MUL_RE.match(line):
            muls.append((line_number, match.group(2)))
        if _COOP_STORE_RE.match(line):
            stores.append(line_number)

    _require(len(muls) == 2, f"expected 2 cooperative MulAdd instructions, found {len(muls)}")
    _require(len(stores) == 2, f"expected 2 cooperative stores, found {len(stores)}")
    _require(muls[0][0] < stores[0] < muls[1][0] < stores[1], "S/PV stage ordering changed")

    def make_stage(
        lower_bound: int,
        mul: tuple[int, str],
        store_line: int,
        expected_a: int,
        expected_b: int,
        expected_acc_width: int,
        expected_load_width: int | set[int],
        name: str,
    ) -> StageAnchors:
        stage_loads = [(line_no, type_id) for line_no, type_id in loads if lower_bound < line_no < mul[0]]
        _require(stage_loads, f"{name} stage has no cooperative loads")
        for _, type_id in stage_loads:
            _require(type_id in coop_types, f"{name} load uses unknown type {type_id}")
        a_count = sum(coop_types[type_id].use == 0 for _, type_id in stage_loads)
        b_count = sum(coop_types[type_id].use == 1 for _, type_id in stage_loads)
        _require(a_count == expected_a, f"{name} expected {expected_a} A loads, found {a_count}")
        _require(b_count == expected_b, f"{name} expected {expected_b} B loads, found {b_count}")
        load_widths = {coop_types[type_id].component_width for _, type_id in stage_loads}
        expected_widths = {expected_load_width} if isinstance(expected_load_width, int) else expected_load_width
        _require(load_widths == expected_widths,
             f"{name} expected {expected_widths}-bit loads, found {sorted(load_widths)}")
        _require(mul[1] in coop_types, f"{name} MulAdd uses unknown accumulator type {mul[1]}")
        acc_width = coop_types[mul[1]].component_width
        _require(acc_width == expected_acc_width, f"{name} accumulator width changed to {acc_width}")
        return StageAnchors(
            first_load_line=stage_loads[0][0],
            mul_add_line=mul[0],
            store_line=store_line,
            load_count=len(stage_loads),
            a_load_count=a_count,
            b_load_count=b_count,
            accumulator_width=acc_width,
        )

    if profile == "canonical":
        # The canonical f8 variant has two shared-memory K alternatives.
        s_stage = make_stage(0, muls[0], stores[0], 2, 1, 32, 16, "S")
    elif profile in {"p2-base", "p3-base", "p4-base", "p5-base"}:
        s_stage = make_stage(0, muls[0], stores[0], 1, 1, 32, 16, "S")
    else:
        s_stage = make_stage(0, muls[0], stores[0], 1, 1, 32, 8, "S")
    pv_b = 3 if profile in {"p4-base", "fp8-p4"} else 2
    if profile in {"p5-base", "fp8-p5"}:
        # P5: PV accumulates in f32; the transformed variant keeps the unused
        # f16 staging/fallback V loads alongside the single fp8 direct load.
        pv_acc, pv_widths = 32, {8, 16} if profile == "fp8-p5" else 16
    else:
        pv_acc, pv_widths = 16, 16
    pv_stage = make_stage(stores[0], muls[1], stores[1], 1, pv_b, pv_acc, pv_widths, "PV")

    return AuditReport(
        source=source,
        profile=profile,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        line_count=len(lines),
        cooperative_type_count=len(coop_types),
        s_stage=s_stage,
        pv_stage=pv_stage,
    )


def _find_spirv_dis(explicit: str | None) -> str:
    if explicit:
        return explicit
    if found := shutil.which("spirv-dis"):
        return found
    if sdk := os.environ.get("VULKAN_SDK"):
        candidate = Path(sdk) / "Bin" / "spirv-dis.exe"
        if candidate.is_file():
            return str(candidate)
    raise AuditError("spirv-dis not found; pass --spirv-dis or set VULKAN_SDK")


def _load_disassembly(path: Path, spirv_dis: str | None) -> str:
    if path.suffix.lower() != ".spv":
        return path.read_text(encoding="utf-8")
    tool = _find_spirv_dis(spirv_dis)
    with tempfile.TemporaryDirectory(prefix="d096-fp8-fa-") as temp_dir:
        output = Path(temp_dir) / "canonical.spvasm"
        subprocess.run([tool, str(path), "-o", str(output)], check=True)
        return output.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="canonical cm1 .spv or disassembled .spvasm")
    parser.add_argument("--spirv-dis", help="path to spirv-dis when input is binary SPIR-V")
    parser.add_argument(
        "--profile",
        choices=("canonical", "p2-base", "fp8-p2", "p3-base", "fp8-p3", "p4-base", "fp8-p4", "p5-base", "fp8-p5"),
        default="canonical",
    )
    args = parser.parse_args()

    try:
        text = _load_disassembly(args.input, args.spirv_dis)
        report = audit_disassembly(text, str(args.input), args.profile)
    except (AuditError, OSError, subprocess.CalledProcessError) as exc:
        print(f"D096 SPIR-V audit failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())