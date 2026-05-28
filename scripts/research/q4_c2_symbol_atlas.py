#!/usr/bin/env python3
"""Build Q4 C2 symbol/entropy atlas from GGUF payload bytes.

Theory-only research utility for P003 Ck-1/Ck-2 checkpoints.
No converter/runtime prototype behavior is included.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GGUF_PY = ROOT / "gguf-py"
if str(GGUF_PY) not in sys.path:
    sys.path.insert(0, str(GGUF_PY))

from gguf.gguf_reader import GGUFReader  # type: ignore


@dataclass
class Q4Layout:
    block_bytes: int
    payload_offset: int
    payload_bytes: int


Q4_LAYOUTS: dict[str, Q4Layout] = {
    # block_q4_0 = f16 d + 16 payload bytes
    "Q4_0": Q4Layout(block_bytes=18, payload_offset=2, payload_bytes=16),
    # block_q4_1 = f16 d + f16 m + 16 payload bytes
    "Q4_1": Q4Layout(block_bytes=20, payload_offset=4, payload_bytes=16),
    # block_q4_K = 2*f16 + 12 scales + 128 payload bytes
    "Q4_K": Q4Layout(block_bytes=144, payload_offset=16, payload_bytes=128),
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument(
        "--label",
        default=f"q4-c2-symbol-atlas-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Output label",
    )
    parser.add_argument(
        "--chunk-blocks",
        type=int,
        default=65536,
        help="Blocks processed per chunk",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "build_logs" / "agent-workload"),
        help="Output directory",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=24,
        help="Top tensors to include in markdown summaries",
    )
    parser.add_argument(
        "--max-tensors",
        type=int,
        default=0,
        help="Process at most this many largest Q4 tensors by n_bytes (0 = all)",
    )
    parser.add_argument(
        "--max-blocks-per-tensor",
        type=int,
        default=0,
        help="Cap processed blocks per tensor for fast exploratory runs (0 = all)",
    )
    return parser.parse_args(argv)


def entropy_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts.astype(np.float64) / float(total)
    nz = probs > 0
    return float(-(probs[nz] * np.log2(probs[nz])).sum())


def make_popcount16_lut() -> np.ndarray:
    values = np.arange(65536, dtype=np.uint16)
    bytes_view = values.view(np.uint8).reshape(-1, 2)
    return np.unpackbits(bytes_view, axis=1).sum(axis=1).astype(np.uint8)


def make_byte_symbol_bitset_lut() -> np.ndarray:
    lut = np.zeros(256, dtype=np.uint16)
    for b in range(256):
        lo = b & 0x0F
        hi = b >> 4
        lut[b] = np.uint16((1 << lo) | (1 << hi))
    return lut


def gib(n_bytes: int) -> float:
    return n_bytes / float(1024**3)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_blocks = max(1024, int(args.chunk_blocks))
    top_n = max(1, int(args.top_n))

    reader = GGUFReader(str(model_path))

    pc16 = make_popcount16_lut()
    byte_set_lut = make_byte_symbol_bitset_lut()

    global_symbol_counts = np.zeros(16, dtype=np.int64)
    global_active_hist = np.zeros(17, dtype=np.int64)
    global_payload_bytes = 0
    global_q4_bytes = 0

    tensors = []

    q4_tensors = [t for t in reader.tensors if t.tensor_type.name in Q4_LAYOUTS]
    q4_tensors.sort(key=lambda t: int(t.n_bytes), reverse=True)
    if int(args.max_tensors) > 0:
        q4_tensors = q4_tensors[: int(args.max_tensors)]

    for tensor in q4_tensors:
        ttype = tensor.tensor_type.name
        layout = Q4_LAYOUTS[ttype]

        raw = np.asarray(tensor.data, dtype=np.uint8).reshape(-1)
        if raw.size % layout.block_bytes != 0:
            continue

        n_blocks = raw.size // layout.block_bytes
        if n_blocks <= 0:
            continue

        if int(args.max_blocks_per_tensor) > 0:
            n_blocks = min(n_blocks, int(args.max_blocks_per_tensor))
            raw = raw[: n_blocks * layout.block_bytes]

        payload_bytes_total = n_blocks * layout.payload_bytes
        payload_symbols_total = payload_bytes_total * 2

        symbol_counts = np.zeros(16, dtype=np.int64)
        active_hist = np.zeros(17, dtype=np.int64)

        for b0 in range(0, n_blocks, chunk_blocks):
            b1 = min(n_blocks, b0 + chunk_blocks)
            off0 = b0 * layout.block_bytes
            off1 = b1 * layout.block_bytes

            blocks = raw[off0:off1].reshape(b1 - b0, layout.block_bytes)
            payload = blocks[:, layout.payload_offset : layout.payload_offset + layout.payload_bytes]

            lo = payload & 0x0F
            hi = payload >> 4

            symbol_counts += np.bincount(lo.ravel(), minlength=16).astype(np.int64)
            symbol_counts += np.bincount(hi.ravel(), minlength=16).astype(np.int64)

            # Ck-2 active-symbol count per block via bitset OR across payload bytes.
            byte_sets = byte_set_lut[payload]
            block_sets = np.bitwise_or.reduce(byte_sets, axis=1)
            active_counts = pc16[block_sets]
            active_hist += np.bincount(active_counts, minlength=17).astype(np.int64)

        ent = entropy_from_counts(symbol_counts)

        # naive fixed-width requirement from active-symbol expectation:
        # E[ceil(log2(K))], where K is active symbol count in a block.
        fixed_bits_by_k = np.array([
            0,
            0,
            1,
            2,
            2,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
        ], dtype=np.float64)
        block_count = int(active_hist.sum())
        if block_count > 0:
            expected_fixed_bits = float((active_hist * fixed_bits_by_k).sum() / block_count)
        else:
            expected_fixed_bits = 4.0

        tensors.append(
            {
                "name": tensor.name,
                "type": ttype,
                "n_bytes": int(tensor.n_bytes),
                "n_elements": int(tensor.n_elements),
                "n_blocks": int(n_blocks),
                "payload_bytes": int(payload_bytes_total),
                "payload_symbols": int(payload_symbols_total),
                "entropy_bpw": ent,
                "entropy_ratio_vs_4bit": (ent / 4.0) if 4.0 else 1.0,
                "expected_fixed_bits_from_activeK": expected_fixed_bits,
                "symbol_counts": symbol_counts.tolist(),
                "active_symbol_hist": active_hist.tolist(),
            }
        )

        global_symbol_counts += symbol_counts
        global_active_hist += active_hist
        global_payload_bytes += payload_bytes_total
        global_q4_bytes += int(tensor.n_bytes)

    global_entropy = entropy_from_counts(global_symbol_counts)
    g_block_count = int(global_active_hist.sum())
    fixed_bits_by_k = np.array([0, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4], dtype=np.float64)
    if g_block_count > 0:
        global_expected_fixed_bits = float((global_active_hist * fixed_bits_by_k).sum() / g_block_count)
    else:
        global_expected_fixed_bits = 4.0

    total_model_bytes = sum(int(t.n_bytes) for t in reader.tensors)

    # Ranking: most relevant large tensors by payload contribution.
    by_payload = sorted(tensors, key=lambda x: x["payload_bytes"], reverse=True)
    by_entropy_gain = sorted(tensors, key=lambda x: (4.0 - x["entropy_bpw"]) * x["payload_symbols"], reverse=True)

    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    summary = {
        "timestamp": ts,
        "label": args.label,
        "model": str(model_path),
        "total_model_bytes": total_model_bytes,
        "total_model_gib": gib(total_model_bytes),
        "q4_bytes": global_q4_bytes,
        "q4_gib": gib(global_q4_bytes),
        "q4_share_percent": (100.0 * global_q4_bytes / total_model_bytes) if total_model_bytes else 0.0,
        "payload_bytes": global_payload_bytes,
        "payload_gib": gib(global_payload_bytes),
        "global_entropy_bpw": global_entropy,
        "global_entropy_ratio_vs_4bit": (global_entropy / 4.0) if 4.0 else 1.0,
        "global_expected_fixed_bits_from_activeK": global_expected_fixed_bits,
        "global_symbol_counts": global_symbol_counts.tolist(),
        "global_active_symbol_hist": global_active_hist.tolist(),
        "tensor_count_q4": len(tensors),
        "max_tensors": int(args.max_tensors),
        "max_blocks_per_tensor": int(args.max_blocks_per_tensor),
        "top_by_payload": by_payload[:top_n],
        "top_by_entropy_gain": by_entropy_gain[:top_n],
    }

    json_path = out_dir / f"{args.label}.q4_c2_symbol_atlas.json"
    md_path = out_dir / f"{args.label}.q4_c2_symbol_atlas.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Q4 C2 Symbol Atlas: {args.label}",
        "",
        f"- timestamp: {ts}",
        f"- model: {model_path.as_posix()}",
        f"- total model: {total_model_bytes} bytes ({gib(total_model_bytes):.3f} GiB)",
        f"- q4 bytes: {global_q4_bytes} bytes ({gib(global_q4_bytes):.3f} GiB, {summary['q4_share_percent']:.2f}%)",
        f"- payload bytes only: {global_payload_bytes} bytes ({gib(global_payload_bytes):.3f} GiB)",
        "",
        "## Global Ck-1/Ck-2 metrics",
        "",
        f"- global payload symbol entropy: {global_entropy:.6f} bpw",
        f"- entropy ratio vs raw 4-bit payload: {summary['global_entropy_ratio_vs_4bit']:.6f}",
        f"- expected fixed bits from active-symbol count: {global_expected_fixed_bits:.6f}",
        "",
        "## Top tensors by payload bytes",
        "",
        "| Tensor | Type | Payload MiB | Entropy bpw | Entropy ratio | E[fixed bits from activeK] |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for row in by_payload[:top_n]:
        lines.append(
            "| "
            f"{row['name']} | {row['type']} | {row['payload_bytes'] / (1024**2):.2f} | "
            f"{row['entropy_bpw']:.6f} | {row['entropy_ratio_vs_4bit']:.6f} | "
            f"{row['expected_fixed_bits_from_activeK']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Top tensors by theoretical entropy gain contribution",
            "",
            "| Tensor | Type | Entropy bpw | Potential gain (4-ent) bpw | Payload MiB |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )

    for row in by_entropy_gain[:top_n]:
        lines.append(
            "| "
            f"{row['name']} | {row['type']} | {row['entropy_bpw']:.6f} | {4.0 - row['entropy_bpw']:.6f} | "
            f"{row['payload_bytes'] / (1024**2):.2f} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path.as_posix()}")
    print(f"Wrote {md_path.as_posix()}")
    print(
        "Global summary: "
        f"entropy={global_entropy:.6f} bpw, "
        f"expected_fixed_bits={global_expected_fixed_bits:.6f}, "
        f"q4_bytes={global_q4_bytes}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
