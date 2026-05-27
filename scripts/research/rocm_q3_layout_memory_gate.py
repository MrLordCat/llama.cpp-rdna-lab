#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from gguf import GGUFReader  # noqa: E402


QK_K = 256
Q3K_GGUF_BLOCK_BYTES = 110
Q3K_RUNTIME_PADDED_BLOCK_BYTES = 112


@dataclass(frozen=True)
class Layout:
    name: str
    bytes_per_block: int
    note: str


LAYOUTS = [
    Layout(
        "compact signed-nibble + int8 scales raw",
        146,
        "128 B signed nibbles + 16 B int8 scales + 2 B d; still needs nibble unpack",
    ),
    Layout(
        "compact signed-nibble + int8 scales aligned",
        160,
        "same data rounded for simple aligned vector loads",
    ),
    Layout(
        "MMA-ready int8 values + fp16 scales",
        288,
        "256 B expanded int8 values + 32 B fp16 scales; removes most Q3 bit unpack",
    ),
    Layout(
        "MMA-ready int8 values + fp32 scales",
        320,
        "256 B expanded int8 values + 64 B fp32 scales; close to current shared tile format",
    ),
]


def gib(value: int | float) -> float:
    return float(value) / float(1 << 30)


def category(name: str) -> str:
    if ".ffn_gate." in name or ".ffn_up." in name or ".ffn_down." in name:
        return "ffn_q3"
    return "other_q3"


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate persistent Q3_K layout expansion for the active GGUF.")
    parser.add_argument("--model", default="models/Qwen3.6-27B-Q3_K_S.gguf")
    args = parser.parse_args()

    reader = GGUFReader(args.model)
    q3_tensors = [tensor for tensor in reader.tensors if getattr(tensor.tensor_type, "name", "") == "Q3_K"]
    if not q3_tensors:
        raise SystemExit("no Q3_K tensors found")

    groups: dict[str, dict[str, int]] = {
        "ffn_q3": {"tensors": 0, "elements": 0, "gguf_bytes": 0, "blocks": 0},
        "other_q3": {"tensors": 0, "elements": 0, "gguf_bytes": 0, "blocks": 0},
    }

    for tensor in q3_tensors:
        group = groups[category(tensor.name)]
        blocks = (tensor.n_elements + QK_K - 1) // QK_K
        group["tensors"] += 1
        group["elements"] += tensor.n_elements
        group["gguf_bytes"] += tensor.n_bytes
        group["blocks"] += blocks

    total = {
        "tensors": sum(group["tensors"] for group in groups.values()),
        "elements": sum(group["elements"] for group in groups.values()),
        "gguf_bytes": sum(group["gguf_bytes"] for group in groups.values()),
        "blocks": sum(group["blocks"] for group in groups.values()),
    }

    current_rows: list[list[str]] = []
    for label, group in [("FFN Q3_K", groups["ffn_q3"]), ("Other Q3_K", groups["other_q3"]), ("All Q3_K", total)]:
        padded = group["blocks"] * Q3K_RUNTIME_PADDED_BLOCK_BYTES
        current_rows.append([
            label,
            str(group["tensors"]),
            f"{group['elements']:,}",
            f"{gib(group['gguf_bytes']):.3f}",
            f"{gib(padded):.3f}",
        ])

    layout_rows: list[list[str]] = []
    for layout in LAYOUTS:
        for label, group in [("FFN Q3_K", groups["ffn_q3"]), ("All Q3_K", total)]:
            layout_bytes = group["blocks"] * layout.bytes_per_block
            current_padded = group["blocks"] * Q3K_RUNTIME_PADDED_BLOCK_BYTES
            extra = layout_bytes - current_padded
            layout_rows.append([
                layout.name if label == "FFN Q3_K" else "",
                label,
                str(layout.bytes_per_block),
                f"{gib(layout_bytes):.3f}",
                f"{gib(extra):+.3f}",
                f"{layout_bytes / current_padded:.3f}x",
            ])

    print("# ROCm Q3_K Layout Memory Gate")
    print()
    print("Inputs:")
    print()
    print(f"- model: `{args.model}`")
    print(f"- Q3_K GGUF block bytes: `{Q3K_GGUF_BLOCK_BYTES}`")
    print(f"- Q3_K runtime padded block bytes: `{Q3K_RUNTIME_PADDED_BLOCK_BYTES}`")
    print(f"- total tensors: `{len(reader.tensors)}`")
    print()
    print(table(["Group", "Tensors", "Elements", "GGUF GiB", "Runtime padded GiB"], current_rows))
    print()
    print(table(["Persistent layout", "Scope", "Bytes/block", "Layout GiB", "Extra vs padded GiB", "Expansion"], layout_rows))
    print()
    print("Decision signal:")
    print()
    print(
        "A persistent MMA-ready Q3_K layout is too large for the 130k ROCm lane: "
        "even FFN-only int8+fp16 expansion adds several GiB, while all-Q3 expansion "
        "adds far more than the current spill-sensitive budget can absorb. A compact "
        "signed-nibble layout has a smaller residency cost, but it still keeps nibble "
        "unpack work and only removes the hmask/scale bit packing; it should not be "
        "promoted without a point kernel proving enough local speedup to offset the "
        "extra residency pressure."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())