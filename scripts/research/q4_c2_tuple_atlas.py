#!/usr/bin/env python3
"""Build Q4 C2 tuple-frequency atlas from GGUF payload nibbles.

Theory-only research utility for P003 Ck-3.
No converter/runtime prototype behavior is included.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
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


def parse_int_csv(value: str, name: str) -> list[int]:
    out = []
    for part in value.split(","):
        p = part.strip()
        if not p:
            continue
        n = int(p)
        if n <= 0:
            raise ValueError(f"{name} must contain positive integers")
        out.append(n)
    if not out:
        raise ValueError(f"{name} must not be empty")
    return sorted(set(out))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument(
        "--label",
        default=f"q4-c2-tuple-atlas-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Output label",
    )
    parser.add_argument(
        "--chunk-blocks",
        type=int,
        default=8192,
        help="Blocks processed per chunk",
    )
    parser.add_argument(
        "--tuple-lens",
        default="2,3,4",
        help="Comma-separated tuple lengths in symbols",
    )
    parser.add_argument(
        "--dict-sizes",
        default="16,64,256,1024",
        help="Comma-separated dictionary sizes K to evaluate",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=24,
        help="Top tuples and tensors to include in markdown",
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
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "build_logs" / "agent-workload"),
        help="Output directory",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=True,
        help="Enable periodic live progress output",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable periodic live progress output",
    )
    parser.add_argument(
        "--progress-interval-sec",
        type=float,
        default=5.0,
        help="Minimum interval between live progress messages",
    )
    return parser.parse_args(argv)


def gib(n_bytes: int) -> float:
    return n_bytes / float(1024**3)


def tuple_ids_for_len(symbols: np.ndarray, tuple_len: int) -> np.ndarray:
    n = symbols.size // tuple_len
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    arr = symbols[: n * tuple_len].reshape(n, tuple_len).astype(np.int64)
    shifts = (np.arange(tuple_len, dtype=np.int64) * 4).reshape(1, tuple_len)
    return (arr << shifts).sum(axis=1)


def fixed_code_escape_bpw(
    tuple_len: int,
    dict_size: int,
    coverage: float,
    total_symbols: int,
) -> float:
    # Model: fixed codebook symbol over (K + ESC), then literal tuple on ESC.
    # bits/token = ceil(log2(K+1)) + (1-coverage) * 4*L
    code_bits = int(math.ceil(math.log2(dict_size + 1)))
    literal_bits = 4 * tuple_len
    bits_per_token = code_bits + (1.0 - coverage) * literal_bits

    # Header proxy: dictionary literals + tiny framing margin.
    header_bits = dict_size * literal_bits + 64

    bits_per_symbol = bits_per_token / float(tuple_len)
    return bits_per_symbol + header_bits / float(max(1, total_symbols))


def topk_coverage(counts: np.ndarray, k: int) -> float:
    total = int(counts.sum())
    if total <= 0:
        return 0.0
    if k >= counts.size:
        return 1.0
    top_sum = int(np.partition(counts, -k)[-k:].sum())
    return top_sum / float(total)


def decode_tuple_id(value: int, tuple_len: int) -> str:
    parts = []
    v = int(value)
    for _ in range(tuple_len):
        parts.append(str(v & 0x0F))
        v >>= 4
    return "(" + ",".join(parts) + ")"


def progress(enabled: bool, message: str) -> None:
    if not enabled:
        return
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    tuple_lens = parse_int_csv(args.tuple_lens, "tuple-lens")
    if max(tuple_lens) > 6:
        raise ValueError("tuple-lens above 6 are not supported in this utility")
    dict_sizes = parse_int_csv(args.dict_sizes, "dict-sizes")
    top_n = max(1, int(args.top_n))
    chunk_blocks = max(1024, int(args.chunk_blocks))
    progress_interval = max(0.25, float(args.progress_interval_sec))
    started = time.monotonic()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress(args.progress, f"loading GGUF: {model_path}")

    reader = GGUFReader(str(model_path))
    progress(args.progress, f"GGUF loaded: tensors={len(reader.tensors)}")

    total_model_bytes = sum(int(t.n_bytes) for t in reader.tensors)
    q4_tensors = [t for t in reader.tensors if t.tensor_type.name in Q4_LAYOUTS]
    q4_tensors.sort(key=lambda t: int(t.n_bytes), reverse=True)
    if int(args.max_tensors) > 0:
        q4_tensors = q4_tensors[: int(args.max_tensors)]

    progress(
        args.progress,
        f"Q4 tensor scope: {len(q4_tensors)} tensors, chunk_blocks={chunk_blocks}, "
        f"max_blocks_per_tensor={int(args.max_blocks_per_tensor)}",
    )

    global_counts: dict[int, np.ndarray] = {
        l: np.zeros(16**l, dtype=np.int64) for l in tuple_lens
    }

    tensor_rows = []
    global_payload_bytes = 0
    global_q4_bytes = 0

    total_tensors = len(q4_tensors)
    processed_tensors = 0

    for tensor in q4_tensors:
        processed_tensors += 1
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

        progress(
            args.progress,
            f"tensor {processed_tensors}/{total_tensors}: {tensor.name} ({ttype}), blocks={n_blocks}",
        )

        payload_bytes_total = n_blocks * layout.payload_bytes
        payload_symbols_total = payload_bytes_total * 2

        per_tensor_counts = {l: np.zeros(16**l, dtype=np.int64) for l in tuple_lens}
        carry = {l: np.empty(0, dtype=np.uint8) for l in tuple_lens}

        chunk_count = (n_blocks + chunk_blocks - 1) // chunk_blocks
        chunks_done = 0
        last_progress = time.monotonic()

        for b0 in range(0, n_blocks, chunk_blocks):
            b1 = min(n_blocks, b0 + chunk_blocks)
            off0 = b0 * layout.block_bytes
            off1 = b1 * layout.block_bytes

            blocks = raw[off0:off1].reshape(b1 - b0, layout.block_bytes)
            payload = blocks[:, layout.payload_offset : layout.payload_offset + layout.payload_bytes].reshape(-1)

            nibs = np.empty(payload.size * 2, dtype=np.uint8)
            nibs[0::2] = payload & 0x0F
            nibs[1::2] = payload >> 4

            for l in tuple_lens:
                stream = np.concatenate((carry[l], nibs)) if carry[l].size else nibs
                ids = tuple_ids_for_len(stream, l)
                if ids.size:
                    binc = np.bincount(ids, minlength=16**l).astype(np.int64)
                    per_tensor_counts[l] += binc
                used = (stream.size // l) * l
                carry[l] = stream[used:]

            chunks_done += 1
            now = time.monotonic()
            if args.progress and (now - last_progress >= progress_interval or chunks_done == chunk_count):
                elapsed = now - started
                progress(
                    args.progress,
                    f"  chunks {chunks_done}/{chunk_count} for {tensor.name}; elapsed={elapsed:.1f}s",
                )
                last_progress = now

        tensor_entry = {
            "name": tensor.name,
            "type": ttype,
            "n_bytes": int(tensor.n_bytes),
            "payload_bytes": int(payload_bytes_total),
            "payload_symbols": int(payload_symbols_total),
            "tuple": {},
        }

        for l in tuple_lens:
            counts = per_tensor_counts[l]
            total_tokens = int(counts.sum())
            if total_tokens <= 0:
                continue
            global_counts[l] += counts

            coverage_rows = []
            best = None
            for k in dict_sizes:
                k_eff = min(k, counts.size)
                cov = topk_coverage(counts, k_eff)
                bpw = fixed_code_escape_bpw(
                    tuple_len=l,
                    dict_size=k_eff,
                    coverage=cov,
                    total_symbols=payload_symbols_total,
                )
                row = {
                    "dict_size": int(k_eff),
                    "coverage": float(cov),
                    "modeled_bpw": float(bpw),
                }
                coverage_rows.append(row)
                if best is None or row["modeled_bpw"] < best["modeled_bpw"]:
                    best = row

            top_idx = np.argpartition(counts, -min(top_n, counts.size))[-min(top_n, counts.size):]
            top_idx = top_idx[np.argsort(counts[top_idx])[::-1]]
            top_tuples = [
                {
                    "id": int(i),
                    "tuple": decode_tuple_id(int(i), l),
                    "count": int(counts[i]),
                    "share": float(counts[i] / total_tokens),
                }
                for i in top_idx
                if counts[i] > 0
            ]

            tensor_entry["tuple"][str(l)] = {
                "total_tokens": total_tokens,
                "coverage": coverage_rows,
                "best_modeled": best,
                "top_tuples": top_tuples,
            }

        tensor_rows.append(tensor_entry)
        global_payload_bytes += payload_bytes_total
        global_q4_bytes += int(tensor.n_bytes)

    global_summary = {}
    global_payload_symbols = global_payload_bytes * 2

    for l in tuple_lens:
        counts = global_counts[l]
        total_tokens = int(counts.sum())
        if total_tokens <= 0:
            continue

        coverage_rows = []
        best = None
        for k in dict_sizes:
            k_eff = min(k, counts.size)
            cov = topk_coverage(counts, k_eff)
            bpw = fixed_code_escape_bpw(
                tuple_len=l,
                dict_size=k_eff,
                coverage=cov,
                total_symbols=global_payload_symbols,
            )
            row = {
                "dict_size": int(k_eff),
                "coverage": float(cov),
                "modeled_bpw": float(bpw),
            }
            coverage_rows.append(row)
            if best is None or row["modeled_bpw"] < best["modeled_bpw"]:
                best = row

        top_idx = np.argpartition(counts, -min(top_n, counts.size))[-min(top_n, counts.size):]
        top_idx = top_idx[np.argsort(counts[top_idx])[::-1]]
        top_tuples = [
            {
                "id": int(i),
                "tuple": decode_tuple_id(int(i), l),
                "count": int(counts[i]),
                "share": float(counts[i] / total_tokens),
            }
            for i in top_idx
            if counts[i] > 0
        ]

        global_summary[str(l)] = {
            "total_tokens": total_tokens,
            "coverage": coverage_rows,
            "best_modeled": best,
            "top_tuples": top_tuples,
        }

    by_payload = sorted(tensor_rows, key=lambda x: x["payload_bytes"], reverse=True)

    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    summary = {
        "timestamp": ts,
        "label": args.label,
        "model": str(model_path),
        "tuple_lens": tuple_lens,
        "dict_sizes": dict_sizes,
        "total_model_bytes": int(total_model_bytes),
        "total_model_gib": gib(total_model_bytes),
        "q4_bytes": int(global_q4_bytes),
        "q4_gib": gib(global_q4_bytes),
        "q4_share_percent": (100.0 * global_q4_bytes / total_model_bytes) if total_model_bytes else 0.0,
        "payload_bytes": int(global_payload_bytes),
        "payload_gib": gib(global_payload_bytes),
        "payload_symbols": int(global_payload_symbols),
        "tensor_count_q4": len(tensor_rows),
        "max_tensors": int(args.max_tensors),
        "max_blocks_per_tensor": int(args.max_blocks_per_tensor),
        "global": global_summary,
        "top_tensors_by_payload": by_payload[:top_n],
    }

    json_path = out_dir / f"{args.label}.q4_c2_tuple_atlas.json"
    md_path = out_dir / f"{args.label}.q4_c2_tuple_atlas.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Q4 C2 Tuple Atlas: {args.label}",
        "",
        f"- timestamp: {ts}",
        f"- model: {model_path.as_posix()}",
        f"- total model: {total_model_bytes} bytes ({gib(total_model_bytes):.3f} GiB)",
        f"- q4 bytes: {global_q4_bytes} bytes ({gib(global_q4_bytes):.3f} GiB, {summary['q4_share_percent']:.2f}%)",
        f"- payload bytes: {global_payload_bytes} bytes ({gib(global_payload_bytes):.3f} GiB)",
        f"- payload symbols: {global_payload_symbols}",
        f"- tuple lengths: {', '.join(str(x) for x in tuple_lens)}",
        f"- dict sizes: {', '.join(str(x) for x in dict_sizes)}",
        f"- max_tensors: {int(args.max_tensors)}",
        f"- max_blocks_per_tensor: {int(args.max_blocks_per_tensor)}",
        "",
        "## Global Ck-3 metrics",
        "",
    ]

    for l in tuple_lens:
        key = str(l)
        if key not in global_summary:
            continue
        data = global_summary[key]
        best = data["best_modeled"]
        lines.extend(
            [
                f"### L={l}",
                "",
                f"- total tuple tokens: {data['total_tokens']}",
                f"- best modeled bpw: {best['modeled_bpw']:.6f} at K={best['dict_size']} (coverage={best['coverage']:.6f})",
                "",
                "| K | Coverage | Modeled bpw |",
                "| ---: | ---: | ---: |",
            ]
        )
        for row in data["coverage"]:
            lines.append(f"| {row['dict_size']} | {row['coverage']:.6f} | {row['modeled_bpw']:.6f} |")

        lines.extend(
            [
                "",
                "Top tuples:",
                "",
                "| Tuple | Share | Count |",
                "| --- | ---: | ---: |",
            ]
        )
        for t in data["top_tuples"][: min(12, len(data["top_tuples"]))]:
            lines.append(f"| {t['tuple']} | {t['share']:.6f} | {t['count']} |")
        lines.append("")

    lines.extend(
        [
            "## Top tensors by payload",
            "",
            "| Tensor | Type | Payload MiB | L | Best K | Coverage | Modeled bpw |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in by_payload[:top_n]:
        if not row.get("tuple"):
            continue
        for l in tuple_lens:
            t = row["tuple"].get(str(l))
            if not t:
                continue
            best = t["best_modeled"]
            lines.append(
                f"| {row['name']} | {row['type']} | {row['payload_bytes'] / (1024**2):.2f} | "
                f"{l} | {best['dict_size']} | {best['coverage']:.6f} | {best['modeled_bpw']:.6f} |"
            )

    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    elapsed = time.monotonic() - started
    progress(args.progress, f"done in {elapsed:.1f}s")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    for l in tuple_lens:
        key = str(l)
        if key in global_summary:
            best = global_summary[key]["best_modeled"]
            print(
                "Global summary "
                f"L={l}: best_bpw={best['modeled_bpw']:.6f} "
                f"K={best['dict_size']} cov={best['coverage']:.6f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
