#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


Q3_K_TYPE_ID = 11
Q3_K_HOST_BLOCK_BYTES = 110
Q3_K_PADDED_BLOCK_BYTES = 112
DEFAULT_MODEL = "models/Qwen3.6-27B-Q3_K_S.gguf"


@dataclass(frozen=True)
class TensorInfo:
    name: str
    qtype: int
    elements: int
    bytes: int


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


def required_local_speedup(target_share: float, target_total_speedup: float) -> float | None:
    denom = (1.0 / target_total_speedup) - (1.0 - target_share)
    if denom <= 0.0:
        return None
    return target_share / denom


def print_memory_table(tensors: list[TensorInfo], pattern: str) -> None:
    selected = select_tensors(tensors, pattern)
    q3_bytes = sum(t.bytes for t in selected)
    blocks = q3_bytes // Q3_K_HOST_BLOCK_BYTES
    remainder = q3_bytes % Q3_K_HOST_BLOCK_BYTES
    padded_bytes = blocks * Q3_K_PADDED_BLOCK_BYTES + remainder
    delta = padded_bytes - q3_bytes

    print(f"## Padded Q3_K Device Layout: {pattern}")
    print()
    print(f"- tensor_count: {len(selected)}")
    print(f"- current_q3_bytes: {fmt_bytes(q3_bytes)}")
    print(f"- padded112_bytes: {fmt_bytes(padded_bytes)}")
    print(f"- delta: {fmt_bytes(delta)} ({(delta / q3_bytes * 100.0) if q3_bytes else 0.0:.3f}%)")
    if remainder:
        print(f"- warning: q3 byte count has {remainder} trailing bytes outside 110-byte block accounting")
    print()


def print_amdahl_table() -> None:
    shares = {
        "optimistic parsed Q3_K/MMVQ share": 0.595,
        "conservative sync wall share proxy": 0.320,
    }
    goals = [1.02, 1.05, 1.10, 1.278]

    print("## Required Local Speedup")
    print()
    print("| share label | share | target wall | required local |")
    print("|---|---:|---:|---:|")
    for label, share in shares.items():
        for goal in goals:
            required = required_local_speedup(share, goal)
            text = "-" if required is None else f"{required:.4f}x"
            print(f"| {label} | {share:.3f} | {goal:.3f}x | {text} |")
    print()


def print_route_decision() -> None:
    print("## Route Decision")
    print()
    print("| candidate | verdict | why |")
    print("|---|---|---|")
    print("| per-node transient Q3_K 110->112 repack | reject analytically | hot decode would add a new copy/pack path over large immutable weights before every matvec family, replacing one bottleneck with memory traffic |")
    print("| duplicate persistent padded Q3_K copy beside current ROCm weights | reject for 16 GiB lane | duplicates about a full Q3_K model copy, not just the 1.8% padding delta |")
    print("| replace ROCm Q3_K storage with backend-private 112-byte blocks | possible large project | padding overhead is small, but all Q3_K CUDA/HIP kernels and buffer set/get/view offsets must honor a device block stride different from GGUF host stride |")
    print("| vecdot-only 32-bit load rewrite on 110-byte blocks | low confidence | current code already uses aligned 16-bit pair loads; unaligned 32-bit loads are unsafe/irregular for every other block and do not remove scale/dot work |")
    print()
    print("- Vulkan gets packed32 cheaply because its backend storage contract already pads Q3_K/Q6_K device blocks.")
    print("- ROCm currently has no equivalent device type-size layer; `block_q3_K *` pointer arithmetic assumes 110-byte stride.")
    print("- A useful implementation would need a planned backend-storage branch, not a local `vecdotq.cuh` patch.")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analytic gate for ROCm Q3_K padded 112-byte device-layout route")
    parser.add_argument("--repo-root", default=".", help="repository root")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GGUF model path")
    parser.add_argument("--patterns", default="ffn-all,ffn-gate-up,ffn-down,all-q3", help="comma-separated tensor groups")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    model = Path(args.model)
    if not model.is_absolute():
        model = repo_root / model

    tensors = load_tensors(model, repo_root)
    print("# ROCm Q3_K Padded Layout Gate")
    print()
    print(f"- model: `{model}`")
    print(f"- tensors: {len(tensors)}")
    print(f"- q3_host_block_bytes: {Q3_K_HOST_BLOCK_BYTES}")
    print(f"- q3_padded_block_bytes: {Q3_K_PADDED_BLOCK_BYTES}")
    print()
    for pattern in [p.strip() for p in args.patterns.split(",") if p.strip()]:
        print_memory_table(tensors, pattern)
    print_amdahl_table()
    print_route_decision()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
