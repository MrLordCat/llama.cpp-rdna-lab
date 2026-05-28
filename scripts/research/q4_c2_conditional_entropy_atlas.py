#!/usr/bin/env python3
"""Build Q4 conditional-entropy atlas from GGUF payload symbols.

Theory-only utility for post-D060 H49 screening. Computes first-order
conditional entropy H(X_t | X_{t-1}) on nibble symbols.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
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
    "Q4_0": Q4Layout(block_bytes=18, payload_offset=2, payload_bytes=16),
    "Q4_1": Q4Layout(block_bytes=20, payload_offset=4, payload_bytes=16),
    "Q4_K": Q4Layout(block_bytes=144, payload_offset=16, payload_bytes=128),
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument(
        "--label",
        default=f"q4-c2-condent-atlas-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Output label",
    )
    parser.add_argument("--chunk-blocks", type=int, default=32768, help="Blocks per chunk")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "build_logs" / "agent-workload"),
        help="Output directory",
    )
    parser.add_argument("--top-n", type=int, default=24, help="Rows in markdown top lists")
    parser.add_argument("--max-tensors", type=int, default=0, help="0 means all")
    parser.add_argument("--max-blocks-per-tensor", type=int, default=0, help="0 means all")
    return parser.parse_args(argv)


def entropy_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts.astype(np.float64) / float(total)
    nz = probs > 0
    return float(-(probs[nz] * np.log2(probs[nz])).sum())


def conditional_entropy_from_bigram(bigram: np.ndarray) -> float:
    total = int(bigram.sum())
    if total <= 0:
        return 0.0
    row_sum = bigram.sum(axis=1).astype(np.float64)
    probs_row = np.zeros_like(bigram, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(bigram, row_sum[:, None], out=probs_row, where=row_sum[:, None] > 0)
        logp = np.zeros_like(probs_row, dtype=np.float64)
        nz = probs_row > 0
        logp[nz] = np.log2(probs_row[nz])
        h_rows = -(probs_row * logp).sum(axis=1)
    p_prev = row_sum / float(total)
    return float((p_prev * h_rows).sum())


def gib(n_bytes: int) -> float:
    return n_bytes / float(1024**3)


def payload_to_nibbles(payload: np.ndarray) -> np.ndarray:
    flat = payload.reshape(-1)
    out = np.empty(flat.size * 2, dtype=np.uint8)
    out[0::2] = flat & 0x0F
    out[1::2] = flat >> 4
    return out


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = GGUFReader(str(model_path))

    tensors = []
    g_uni = np.zeros(16, dtype=np.int64)
    g_bi = np.zeros((16, 16), dtype=np.int64)
    g_q4_bytes = 0
    g_payload_bytes = 0

    q4_tensors = [t for t in reader.tensors if t.tensor_type.name in Q4_LAYOUTS]
    q4_tensors.sort(key=lambda t: int(t.n_bytes), reverse=True)
    if int(args.max_tensors) > 0:
        q4_tensors = q4_tensors[: int(args.max_tensors)]

    chunk_blocks = max(1024, int(args.chunk_blocks))

    for tensor in q4_tensors:
        layout = Q4_LAYOUTS[tensor.tensor_type.name]
        raw = np.asarray(tensor.data, dtype=np.uint8).reshape(-1)
        if raw.size % layout.block_bytes != 0:
            continue

        n_blocks = raw.size // layout.block_bytes
        if n_blocks <= 0:
            continue

        if int(args.max_blocks_per_tensor) > 0:
            n_blocks = min(n_blocks, int(args.max_blocks_per_tensor))
            raw = raw[: n_blocks * layout.block_bytes]

        t_uni = np.zeros(16, dtype=np.int64)
        t_bi = np.zeros((16, 16), dtype=np.int64)
        carry_prev = None

        for b0 in range(0, n_blocks, chunk_blocks):
            b1 = min(n_blocks, b0 + chunk_blocks)
            sl = raw[b0 * layout.block_bytes : b1 * layout.block_bytes]
            blocks = sl.reshape(b1 - b0, layout.block_bytes)
            payload = blocks[:, layout.payload_offset : layout.payload_offset + layout.payload_bytes]
            nib = payload_to_nibbles(payload)

            t_uni += np.bincount(nib, minlength=16).astype(np.int64)

            if carry_prev is not None and nib.size > 0:
                t_bi[carry_prev, int(nib[0])] += 1

            if nib.size >= 2:
                pairs = nib[:-1].astype(np.int32) * 16 + nib[1:].astype(np.int32)
                cnt = np.bincount(pairs, minlength=256).astype(np.int64).reshape(16, 16)
                t_bi += cnt

            carry_prev = int(nib[-1]) if nib.size > 0 else carry_prev

        h_uni = entropy_from_counts(t_uni)
        h_cond = conditional_entropy_from_bigram(t_bi)

        payload_bytes = int(n_blocks * layout.payload_bytes)

        row = {
            "name": tensor.name,
            "type": tensor.tensor_type.name,
            "n_blocks": int(n_blocks),
            "payload_bytes": payload_bytes,
            "payload_symbols": int(payload_bytes * 2),
            "entropy_unigram_bpw": h_uni,
            "entropy_conditional_bpw": h_cond,
            "delta_unigram_minus_conditional": h_uni - h_cond,
        }
        tensors.append(row)

        g_uni += t_uni
        g_bi += t_bi
        g_q4_bytes += int(tensor.n_bytes)
        g_payload_bytes += payload_bytes

    g_h_uni = entropy_from_counts(g_uni)
    g_h_cond = conditional_entropy_from_bigram(g_bi)

    by_payload = sorted(tensors, key=lambda r: r["payload_bytes"], reverse=True)
    by_delta = sorted(tensors, key=lambda r: r["delta_unigram_minus_conditional"] * r["payload_symbols"], reverse=True)

    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    total_model_bytes = sum(int(t.n_bytes) for t in reader.tensors)

    summary = {
        "timestamp": ts,
        "label": args.label,
        "model": str(model_path),
        "max_tensors": int(args.max_tensors),
        "max_blocks_per_tensor": int(args.max_blocks_per_tensor),
        "total_model_bytes": total_model_bytes,
        "q4_bytes": g_q4_bytes,
        "payload_bytes": g_payload_bytes,
        "global_entropy_unigram_bpw": g_h_uni,
        "global_entropy_conditional_bpw": g_h_cond,
        "global_delta_unigram_minus_conditional": g_h_uni - g_h_cond,
        "tensor_count_q4": len(tensors),
        "top_by_payload": by_payload[: max(1, int(args.top_n))],
        "top_by_conditional_delta": by_delta[: max(1, int(args.top_n))],
    }

    json_path = out_dir / f"{args.label}.q4_c2_conditional_entropy_atlas.json"
    md_path = out_dir / f"{args.label}.q4_c2_conditional_entropy_atlas.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Q4 C2 Conditional Entropy Atlas: {args.label}",
        "",
        f"- timestamp: {ts}",
        f"- model: {model_path.as_posix()}",
        f"- total model: {total_model_bytes} bytes ({gib(total_model_bytes):.3f} GiB)",
        f"- q4 bytes: {g_q4_bytes} bytes ({gib(g_q4_bytes):.3f} GiB)",
        f"- payload bytes: {g_payload_bytes} bytes ({gib(g_payload_bytes):.3f} GiB)",
        "",
        "## Global metrics",
        "",
        f"- unigram entropy: {g_h_uni:.6f} bpw",
        f"- first-order conditional entropy: {g_h_cond:.6f} bpw",
        f"- unigram minus conditional delta: {g_h_uni - g_h_cond:.6f} bpw",
        "",
        "## Top tensors by payload",
        "",
        "| Tensor | Type | Payload MiB | H1 bpw | Hcond bpw | H1-Hcond |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for r in by_payload[: max(1, int(args.top_n))]:
        lines.append(
            "| "
            f"{r['name']} | {r['type']} | {r['payload_bytes'] / (1024**2):.2f} | "
            f"{r['entropy_unigram_bpw']:.6f} | {r['entropy_conditional_bpw']:.6f} | "
            f"{r['delta_unigram_minus_conditional']:.6f} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path.as_posix()}")
    print(f"Wrote {md_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
