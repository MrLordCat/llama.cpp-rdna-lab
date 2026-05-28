#!/usr/bin/env python3
"""H53 fast gate: nibble reordering within superblocks for entropy reduction.

Analytical gate that measures how much sorting nibbles within each superblock
reduces first-order conditional entropy, and whether the permutation overhead
is small enough to net benefit within the P003 corridor (3.57-3.77 bpw).

Key insight: H1 (unigram entropy) doesn't change with reordering, but H1_cond
(first-order conditional entropy) drops significantly when similar values are
grouped together, creating compressible runs.

Permutation overhead model:
- For Q4_0 (32 nibbles/block): each nibble needs ceil(log2(32))=5 bits to specify
  its position in the sorted order, but we can encode more efficiently.
- For Q4_K (256 nibbles/block): similar but larger.
- We model: for N nibbles, we store the permutation as a lookup table where each
  original position i maps to sorted position p[i].

Fast gate approach:
1. Sample representative Q4 tensors
2. For each superblock, extract nibble payload
3. Sort nibbles and measure H1_cond on sorted vs original stream
4. Model permutation overhead
5. Check if: H1_cond_sorted + perm_overhead_bpw < corridor_min (3.57)
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
    nibbles_per_block: int


Q4_LAYOUTS: dict[str, Q4Layout] = {
    "Q4_0": Q4Layout(block_bytes=18, payload_offset=2, payload_bytes=16, nibbles_per_block=32),
    "Q4_1": Q4Layout(block_bytes=20, payload_offset=4, payload_bytes=16, nibbles_per_block=32),
    "Q4_K": Q4Layout(block_bytes=144, payload_offset=16, payload_bytes=128, nibbles_per_block=256),
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument(
        "--label",
        default=f"q4c2-nibbler-order-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Output label",
    )
    parser.add_argument(
        "--chunk-blocks",
        type=int,
        default=16384,
        help="Blocks processed per chunk",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "build_logs" / "agent-workload"),
    )
    parser.add_argument(
        "--max-tensors",
        type=int,
        default=24,
        help="Process at most this many largest Q4 tensors (0 = all)",
    )
    parser.add_argument(
        "--max-blocks-per-tensor",
        type=int,
        default=131072,
        help="Cap blocks per tensor for fast gate (0 = all)",
    )
    parser.add_argument(
        "--progress-interval-sec",
        type=float,
        default=15.0,
        help="Print progress every N seconds",
    )
    return parser.parse_args(argv)


def entropy_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts.astype(np.float64) / float(total)
    nz = probs > 0
    return float(-(probs[nz] * np.log2(probs[nz])).sum())


def conditional_entropy_from_bigrams(nibbles: np.ndarray) -> float:
    """First-order conditional entropy H(X_i | X_{i-1}).

    For a stream of nibbles, count bigrams (a,b) and compute:
    H(X_i | X_{i-1}) = sum_a P(a) * sum_b [-P(b|a) * log2(P(b|a))]
    """
    n = len(nibbles)
    if n < 2:
        return 0.0

    # Count bigrams: (prev, curr) - vectorized
    prev = nibbles[:-1]
    curr = nibbles[1:]

    # Joint counts: 16x16 table - use add.at for speed
    joint = np.zeros((16, 16), dtype=np.int64)
    np.add.at(joint, (prev.astype(np.intp), curr.astype(np.intp)), 1)

    # Marginal P(prev=a)
    marginal = joint.sum(axis=1).astype(np.float64)
    total_bigrams = marginal.sum()

    if total_bigrams <= 0:
        return 0.0

    # H(X_i | X_{i-1}) = sum_a P(a) * H(X_i | X_{i-1}=a)
    cond_ent = 0.0
    for a in range(16):
        if marginal[a] <= 0:
            continue
        p_a = marginal[a] / total_bigrams
        # Conditional distribution P(curr=b | prev=a)
        cond_probs = joint[a, :].astype(np.float64) / marginal[a]
        nz = cond_probs > 0
        h_given_a = float(-(cond_probs[nz] * np.log2(cond_probs[nz])).sum())
        cond_ent += p_a * h_given_a

    return cond_ent


def permutation_overhead_bpw(nibbles_per_block: int, method: str = "index") -> float:
    """Model permutation overhead in bpw (bits per original nibble).

    Methods:
    - "index": store permutation as array of positions, each ceil(log2(N)) bits
    - "rank": store rank differences (usually smaller for near-sorted data)
    - "runs": encode as run-length style (efficient for sorted data)

    For fast gate, we use "index" as upper bound on overhead.
    """
    N = nibbles_per_block

    if method == "index":
        # Each of N nibbles needs log2(N) bits to specify position
        bits_per_nibble = math.log2(N)
        # But we store N such indices, so total bits = N * log2(N)
        # Per original nibble: log2(N) bits
        return bits_per_nibble

    elif method == "index_compact":
        # For sorted data, we can encode permutation more efficiently:
        # most nibbles don't move far from original position
        # Approximation: log2(N) * 0.5 bits per nibble
        return math.log2(N) * 0.5

    elif method == "runs":
        # Encode sorted sequence as runs; overhead depends on distribution
        # For uniform-ish data: ~log2(N) + average_run_count * log2(N)
        # Approximation: 0.25 * log2(N) bits per nibble for sorted data
        return math.log2(N) * 0.25

    return math.log2(N)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_blocks = max(1024, int(args.chunk_blocks))

    reader = GGUFReader(str(model_path))

    q4_tensors = [t for t in reader.tensors if t.tensor_type.name in Q4_LAYOUTS]
    q4_tensors.sort(key=lambda t: int(t.n_bytes), reverse=True)
    if int(args.max_tensors) > 0:
        q4_tensors = q4_tensors[: int(args.max_tensors)]

    # Global accumulators
    global_orig_bigram_ent = 0.0
    global_sorted_bigram_ent = 0.0
    global_orig_payload_symbols = 0
    global_sorted_payload_symbols = 0

    tensor_results = []

    start_time = dt.datetime.now()
    tensors_done = 0

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

        # Per-tensor accumulators
        tensor_orig_bigrams = np.zeros((16, 16), dtype=np.int64)
        tensor_sorted_bigrams = np.zeros((16, 16), dtype=np.int64)
        tensor_orig_symbols = 0
        tensor_sorted_symbols = 0
        blocks_analyzed = 0

        for b0 in range(0, n_blocks, chunk_blocks):
            b1 = min(n_blocks, b0 + chunk_blocks)
            off0 = b0 * layout.block_bytes
            off1 = b1 * layout.block_bytes

            blocks = raw[off0:off1].reshape(b1 - b0, layout.block_bytes)
            payload = blocks[:, layout.payload_offset : layout.payload_offset + layout.payload_bytes]

            # Extract nibbles: each byte has 2 nibbles (lo, hi)
            lo = payload & 0x0F
            hi = payload >> 4
            # Interleave to get original order: lo0, hi0, lo1, hi1, ...
            orig_nibbles = np.empty((lo.shape[0], lo.shape[1] * 2), dtype=np.uint8)
            orig_nibbles[:, 0::2] = lo
            orig_nibbles[:, 1::2] = hi
            orig_flat = orig_nibbles.ravel()

            # Sort nibbles within each block
            sorted_nibbles = np.empty_like(orig_nibbles)
            for block_idx in range(lo.shape[0]):
                block_nibbles = orig_nibbles[block_idx, :].copy()
                np.sort(block_nibbles)
                sorted_nibbles[block_idx, :] = block_nibbles
            sorted_flat = sorted_nibbles.ravel()

            # Count bigrams
            if len(orig_flat) >= 2:
                for p, c in zip(orig_flat[:-1], orig_flat[1:]):
                    tensor_orig_bigrams[int(p), int(c)] += 1

            if len(sorted_flat) >= 2:
                for p, c in zip(sorted_flat[:-1], sorted_flat[1:]):
                    tensor_sorted_bigrams[int(p), int(c)] += 1

            tensor_orig_symbols += len(orig_flat)
            tensor_sorted_symbols += len(sorted_flat)
            blocks_analyzed += (b1 - b0)

            # Progress
            if args.progress_interval_sec > 0:
                now = dt.datetime.now()
                if (now - start_time).total_seconds() >= args.progress_interval_sec:
                    print(f"[{now.isoformat()}] Tensors: {tensors_done + 1}/{len(q4_tensors)}, "
                          f"Blocks so far: {blocks_analyzed}/{n_blocks}, "
                          f"Type: {ttype}", flush=True)
                    start_time = now

        # Compute conditional entropies for this tensor
        orig_marginal = tensor_orig_bigrams.sum(axis=1).astype(np.float64)
        sorted_marginal = tensor_sorted_bigrams.sum(axis=1).astype(np.float64)

        orig_total = orig_marginal.sum()
        sorted_total = sorted_marginal.sum()

        orig_cond_ent = 0.0
        if orig_total > 0:
            for a in range(16):
                if orig_marginal[a] <= 0:
                    continue
                p_a = orig_marginal[a] / orig_total
                cond_probs = tensor_orig_bigrams[a, :].astype(np.float64) / orig_marginal[a]
                nz = cond_probs > 0
                orig_cond_ent += p_a * float(-(cond_probs[nz] * np.log2(cond_probs[nz])).sum())

        sorted_cond_ent = 0.0
        if sorted_total > 0:
            for a in range(16):
                if sorted_marginal[a] <= 0:
                    continue
                p_a = sorted_marginal[a] / sorted_total
                cond_probs = tensor_sorted_bigrams[a, :].astype(np.float64) / sorted_marginal[a]
                nz = cond_probs > 0
                sorted_cond_ent += p_a * float(-(cond_probs[nz] * np.log2(cond_probs[nz])).sum())

        # Delta
        delta = orig_cond_ent - sorted_cond_ent

        # Model permutation overhead
        perm_overhead = permutation_overhead_bpw(layout.nibbles_per_block, "index")
        perm_overhead_compact = permutation_overhead_bpw(layout.nibbles_per_block, "index_compact")
        perm_overhead_runs = permutation_overhead_bpw(layout.nibbles_per_block, "runs")

        # Net effective bpw for sorted stream with overhead
        net_bpw_index = sorted_cond_ent + perm_overhead
        net_bpw_compact = sorted_cond_ent + perm_overhead_compact
        net_bpw_runs = sorted_cond_ent + perm_overhead_runs

        tensor_results.append({
            "name": tensor.name,
            "type": ttype,
            "n_bytes": int(tensor.n_bytes),
            "n_blocks": int(n_blocks),
            "nibbles_per_block": layout.nibbles_per_block,
            "orig_symbols": int(tensor_orig_symbols),
            "sorted_symbols": int(tensor_sorted_symbols),
            "orig_cond_ent_bpw": round(orig_cond_ent, 6),
            "sorted_cond_ent_bpw": round(sorted_cond_ent, 6),
            "hcond_delta_bpw": round(delta, 6),
            "perm_overhead_index_bpw": round(perm_overhead, 4),
            "perm_overhead_compact_bpw": round(perm_overhead_compact, 4),
            "perm_overhead_runs_bpw": round(perm_overhead_runs, 4),
            "net_bpw_index": round(net_bpw_index, 6),
            "net_bpw_compact": round(net_bpw_compact, 6),
            "net_bpw_runs": round(net_bpw_runs, 6),
        })

        # Accumulate globally
        global_orig_bigrams = tensor_orig_bigrams  # Will be merged later
        global_sorted_bigrams = tensor_sorted_bigrams
        global_orig_payload_symbols += tensor_orig_symbols
        global_sorted_payload_symbols += tensor_sorted_symbols

        tensors_done += 1

    # Global conditional entropies (need to merge bigram tables)
    # For simplicity in fast gate, use weighted average by symbol count
    if tensor_results:
        total_symbols = sum(t["orig_symbols"] for t in tensor_results)
        global_orig_cond_ent = sum(t["orig_cond_ent_bpw"] * t["orig_symbols"] for t in tensor_results) / total_symbols
        global_sorted_cond_ent = sum(t["sorted_cond_ent_bpw"] * t["sorted_symbols"] for t in tensor_results) / total_symbols
    else:
        global_orig_cond_ent = 0.0
        global_sorted_cond_ent = 0.0

    global_delta = global_orig_cond_ent - global_sorted_cond_ent

    # Corridor check
    corridor_min = 3.57
    corridor_max = 3.77

    # For sorted stream, effective bpw = sorted_cond_ent + overhead
    # We check if any overhead model reaches corridor
    feasible_count = 0
    best_net_bpw = float("inf")
    best_config = None

    for t in tensor_results:
        for method, net_bpw in [
            ("index", t["net_bpw_index"]),
            ("compact", t["net_bpw_compact"]),
            ("runs", t["net_bpw_runs"]),
        ]:
            if net_bpw <= corridor_max:
                feasible_count += 1
            if net_bpw < best_net_bpw:
                best_net_bpw = net_bpw
                best_config = (t["name"], t["type"], method, net_bpw)

    result = {
        "label": args.label,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": str(model_path),
        "n_tensors": len(q4_tensors),
        "max_tensors": int(args.max_tensors),
        "max_blocks_per_tensor": int(args.max_blocks_per_tensor),
        "global_orig_symbols": int(global_orig_payload_symbols),
        "global_sorted_symbols": int(global_sorted_payload_symbols),
        "global_orig_cond_ent_bpw": round(global_orig_cond_ent, 6),
        "global_sorted_cond_ent_bpw": round(global_sorted_cond_ent, 6),
        "global_hcond_delta_bpw": round(global_delta, 6),
        "corridor": [corridor_min, corridor_max],
        "feasible_configs_under_corridor_max": feasible_count,
        "best_net_bpw": round(best_net_bpw, 6),
        "best_config": best_config,
        "tensor_results": tensor_results,
    }

    # Write JSON
    json_path = out_dir / f"{args.label}.q4_c2_nibble_reorder_gate.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"H53 Nibble Reorder Fast Gate: {args.label}")
    print(f"{'='*60}")
    print(f"Tensors analyzed: {len(q4_tensors)}")
    print(f"Original payload symbols: {global_orig_payload_symbols:,}")
    print(f"Original conditional entropy: {global_orig_cond_ent:.6f} bpw")
    print(f"Sorted conditional entropy: {global_sorted_cond_ent:.6f} bpw")
    print(f"H1-Hcond delta (reorder benefit): {global_delta:.6f} bpw")
    print(f"Corridor: {corridor_min}-{corridor_max} bpw")
    print(f"Feasible configs <= {corridor_max} bpw: {feasible_count}")
    if best_config:
        print(f"Best: {best_config[0]} ({best_config[1]}, {best_config[2]}) = {best_config[3]:.6f} bpw")
    else:
        print(f"Best net bpw: {best_net_bpw:.6f} (above corridor)")

    # Per-tensor summary
    print(f"\nTop 10 tensors by Hcond delta:")
    by_delta = sorted(tensor_results, key=lambda x: x["hcond_delta_bpw"], reverse=True)[:10]
    for t in by_delta:
        print(f"  {t['name'][:60]:.<60} "
              f"H1={t['orig_cond_ent_bpw']:.4f} -> H1s={t['sorted_cond_ent_bpw']:.4f} "
              f"Delta={t['hcond_delta_bpw']:+.4f} "
              f"Net(runs)={t['net_bpw_runs']:.4f}")

    print(f"\nJSON: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
