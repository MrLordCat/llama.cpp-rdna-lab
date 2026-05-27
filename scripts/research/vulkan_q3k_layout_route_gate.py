#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path


Q3_K_TYPE_ID = 11
DEFAULT_MODEL = "models/Qwen3.6-27B-Q3_K_S.gguf"


@dataclass(frozen=True)
class TensorInfo:
    name: str
    qtype: int
    shape: tuple[int, ...]
    elements: int
    bytes: int


@dataclass(frozen=True)
class ShapeProxy:
    label: str
    m: int
    n: int
    k: int
    count: int


HOT_SHAPE_PROFILES: dict[str, tuple[ShapeProxy, ...]] = {
    "p002-130k": (
        ShapeProxy("ffn_gate_up", 17408, 128, 5120, 2),
        ShapeProxy("ffn_down", 5120, 128, 17408, 1),
    ),
    "p001-64k": (
        ShapeProxy("ffn_gate_up", 17408, 1024, 5120, 2),
        ShapeProxy("ffn_down", 5120, 1024, 17408, 1),
    ),
}


def fmt_bytes(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    out = float(value)
    for unit in units:
        if abs(out) < 1024.0 or unit == units[-1]:
            return f"{out:.2f} {unit}"
        out /= 1024.0
    return f"{out:.2f} TiB"


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def load_tensors(model: Path, repo_root: Path) -> list[TensorInfo]:
    sys.path.insert(0, str(repo_root / "gguf-py"))
    try:
        from gguf.gguf_reader import GGUFReader  # type: ignore
    except ImportError as exc:
        raise SystemExit("ERROR: could not import gguf reader; expected local gguf-py") from exc

    reader = GGUFReader(str(model))
    out: list[TensorInfo] = []
    for tensor in reader.tensors:
        out.append(
            TensorInfo(
                name=str(tensor.name),
                qtype=int(tensor.tensor_type),
                shape=tuple(int(x) for x in tensor.shape),
                elements=int(tensor.n_elements),
                bytes=int(tensor.n_bytes),
            )
        )
    return out


def select_tensors(tensors: list[TensorInfo], pattern: str) -> list[TensorInfo]:
    if pattern == "all-q3":
        return [t for t in tensors if t.qtype == Q3_K_TYPE_ID]
    if pattern == "ffn-all":
        needles = (".ffn_gate.weight", ".ffn_up.weight", ".ffn_down.weight")
    elif pattern == "ffn-gate-up":
        needles = (".ffn_gate.weight", ".ffn_up.weight")
    elif pattern == "ffn-down":
        needles = (".ffn_down.weight",)
    else:
        raise SystemExit(f"ERROR: unknown pattern {pattern!r}")
    return [t for t in tensors if t.qtype == Q3_K_TYPE_ID and any(needle in t.name for needle in needles)]


def q3_a_pair_proxy(m: int, n: int, k: int, bm: int, bn: int, n_reuse: int = 1) -> int:
    m_blocks = math.ceil(m / bm)
    n_groups = math.ceil(math.ceil(n / bn) / n_reuse)
    return m_blocks * n_groups * (bm * k // 2)


def q3_workgroups(m: int, n: int, bm: int, bn: int, n_reuse: int = 1) -> int:
    return math.ceil(m / bm) * math.ceil(math.ceil(n / bn) / n_reuse)


def print_hot_shape_proxy(profile: str, shapes: tuple[ShapeProxy, ...]) -> None:
    bm = 128
    bn = 128

    base_pairs = 0
    reuse2_pairs = 0
    reuse4_pairs = 0
    base_wgs = 0
    print("## Hot Shape Proxy")
    print()
    print(f"- shape_profile: `{profile}`")
    print(f"- base_tile: `BM={bm},BN={bn}`")
    print()
    print("| shape | count | base workgroups | base A pair-dequants | N-reuse2 A pairs | N-reuse4 A pairs |")
    print("|---|---:|---:|---:|---:|---:|")
    for shape in shapes:
        pairs = q3_a_pair_proxy(shape.m, shape.n, shape.k, bm, bn) * shape.count
        pairs2 = q3_a_pair_proxy(shape.m, shape.n, shape.k, bm, bn, 2) * shape.count
        pairs4 = q3_a_pair_proxy(shape.m, shape.n, shape.k, bm, bn, 4) * shape.count
        wgs = q3_workgroups(shape.m, shape.n, bm, bn) * shape.count
        base_pairs += pairs
        reuse2_pairs += pairs2
        reuse4_pairs += pairs4
        base_wgs += wgs
        print(f"| {shape.label} `{shape.m}x{shape.n}x{shape.k}` | {shape.count} | {wgs} | {pairs:,} | {pairs2:,} | {pairs4:,} |")
    print(f"| total hot FFN proxy |  | {base_wgs} | {base_pairs:,} | {reuse2_pairs:,} | {reuse4_pairs:,} |")
    print()
    print("- Important constraint: true A reuse across N-blocks requires either multiple accumulator sets alive across the full K-loop or global partial sums/reduce.")
    print("- A single-accumulator sequential N loop can only compute one N tile at a time; it then reloads/dequants A for the next tile and does not reduce this proxy.")
    if all(shape.n <= bn for shape in shapes):
        print("- For this profile, `n <= BN`, so N-reuse cannot reduce A-side Q3_K dequant work; FFN fusion must justify itself through B/activation reuse, launch reduction, or a new A layout.")
    print("- E137 already rejected the multiple-accumulator `niter2` implementation (`120 VGPR`, pp7488 `855.29` vs clean `974.92`).")
    print()


def print_memory_table(tensors: list[TensorInfo], pattern: str) -> None:
    selected = select_tensors(tensors, pattern)
    elems = sum(t.elements for t in selected)
    q3_bytes = sum(t.bytes for t in selected)
    f16_bytes = elems * 2
    int8_bytes = elems
    nibble_bytes = math.ceil(elems / 2)

    print(f"## Persistent Layout Memory: {pattern}")
    print()
    print(f"- tensor_count: {len(selected)}")
    print(f"- elements: {elems:,}")
    print(f"- current_q3_bytes: {fmt_bytes(q3_bytes)}")
    print()
    print("| layout | absolute bytes | delta vs current Q3 device copy | decision |")
    print("|---|---:|---:|---|")
    print(f"| persistent fp16 | {fmt_bytes(f16_bytes)} | {fmt_bytes(f16_bytes - q3_bytes)} | reject for 16 GiB long-context lane |")
    print(f"| persistent int8 expanded values | {fmt_bytes(int8_bytes)} | {fmt_bytes(int8_bytes - q3_bytes)} | reject unless very narrow tensor subset |")
    print(f"| persistent signed-nibble values | {fmt_bytes(nibble_bytes)} | {fmt_bytes(nibble_bytes - q3_bytes)} | memory-plausible, but only removes bit unpack, not scale multiply or coopmat work |")
    print()


def print_route_decision(profile: str) -> None:
    print("## Route Decision")
    print()
    print("| candidate | gate verdict | why |")
    print("|---|---|---|")
    print("| single-accumulator sequential N loop | reject analytically | no A reuse is possible without keeping per-N partial sums across the K-loop |")
    print("| dual-accumulator N reuse | measured reject | E137 got the intended route class but raised VGPR and regressed pp7488 |")
    print("| per-node fp16 predequant | measured reject | E139 lowered matmul resources but lost to fp16 temp traffic and sync |")
    print("| persistent fp16 FFN layout | reject analytically | FFN-only fp16 alternate copy is tens of GiB |")
    print("| persistent int8 FFN layout | reject for broad FFN | still multi-GiB extra memory on a 16 GiB 64k lane |")
    print("| persistent signed-nibble Q3 layout | possible research branch | memory can be near +16% for selected Q3_K tensors, but expected speed is uncertain after E088/E090 |")
    print()
    print("- Next buildable Q3_K branch should be a separate pipeline/layout, not a shared-source mutation of `mul_mm.comp`.")
    print("- The only memory-plausible persistent layout is a signed-nibble or similarly compact backend-private format; it needs an instruction-count/SPIR-V gate before code.")
    if profile == "p002-130k":
        print("- On P002 `ubatch=128`, broad FFN gate/up fusion is not the first code candidate unless its design proves B/activation reuse is large enough despite no N-tile A reuse.")
    print("- Because E088 scale reuse and E090 packed32 pair helpers were negative, this branch should not claim target-closing speed unless it removes a large share of Q3_K bit unpack work without raising registers.")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analytic gate for Vulkan Q3_K persistent layout / route branches")
    parser.add_argument("--repo-root", default=".", help="repository root")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GGUF model path")
    parser.add_argument("--shape-profile", choices=sorted(HOT_SHAPE_PROFILES), default="p002-130k", help="hot-shape proxy profile")
    parser.add_argument("--patterns", default="ffn-all,ffn-gate-up,ffn-down,all-q3", help="comma-separated tensor groups")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    model = Path(args.model)
    if not model.is_absolute():
        model = repo_root / model
    tensors = load_tensors(model, repo_root)

    print("# Vulkan Q3_K Layout Route Gate")
    print()
    print(f"- model: `{model}`")
    print(f"- tensors: {len(tensors)}")
    print()
    print_hot_shape_proxy(args.shape_profile, HOT_SHAPE_PROFILES[args.shape_profile])
    for pattern in [p.strip() for p in args.patterns.split(",") if p.strip()]:
        print_memory_table(tensors, pattern)
    print_route_decision(args.shape_profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
