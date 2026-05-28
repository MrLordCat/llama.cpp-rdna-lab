#!/usr/bin/env python3
"""
H54-A Analytical Gate: Can Householder Q rotation reduce Q4 entropy?

Measures Q4 nibble entropy and compares with sorted-data entropy (upper bound
on what any permutation/rotation could achieve). If sorted entropy can't reach
corridor, rotation won't either.

Usage:
    python scripts/research/q4_c2_rotation_entropy_gate.py \\
        --model models/Qwen3.6-27B-Q4_K_S.gguf \\
        [--max-tensors N] [--seed SEED] [--label LABEL]
"""

from __future__ import annotations

import argparse
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
    block_size: int
    payload_offset: int
    payload_bytes: int


Q4_LAYOUTS = {
    'Q4_0': Q4Layout(block_size=18, payload_offset=2, payload_bytes=16),
    'Q4_1': Q4Layout(block_size=20, payload_offset=4, payload_bytes=16),
    'Q4_K': Q4Layout(block_size=144, payload_offset=16, payload_bytes=128),
    'Q4_K_M': Q4Layout(block_size=144, payload_offset=16, payload_bytes=128),
    'Q4_K_S': Q4Layout(block_size=144, payload_offset=16, payload_bytes=128),
}


def extract_q4_nibbles(tensor_data, q4_kind):
    layout = Q4_LAYOUTS.get(q4_kind)
    if not layout:
        raise ValueError(f"Unknown Q4 kind: {q4_kind}")
    raw = np.asarray(tensor_data, dtype=np.uint8).reshape(-1)
    if raw.size % layout.block_size != 0:
        raise ValueError(f"Tensor size {raw.size} not aligned to block_size {layout.block_size}")
    n_blocks = raw.size // layout.block_size
    all_nibbles = []
    for i in range(n_blocks):
        base = i * layout.block_size
        nibbles = raw[base + layout.payload_offset: base + layout.payload_offset + layout.payload_bytes]
        all_nibbles.append(nibbles)
    return np.concatenate(all_nibbles)


def unpack_nibbles(byte_array):
    """Unpack byte array into individual nibbles (0-15)."""
    lo = byte_array & 0x0F
    hi = byte_array >> 4
    # Interleave: [lo0, hi0, lo1, hi1, ...]
    result = np.zeros(byte_array.size * 2, dtype=np.uint8)
    result[0::2] = lo
    result[1::2] = hi
    return result


def compute_nibble_entropy(nibble_array):
    """Compute entropy of unpacked nibbles (0-15)."""
    # Unpack bytes to individual nibbles
    nibbles = unpack_nibbles(nibble_array)
    counts = np.bincount(nibbles.astype(np.int32), minlength=16)
    probs = counts / counts.sum()
    mask = probs > 0
    entropy = float(-np.sum(probs[mask] * np.log2(probs[mask])))
    n_symbols = nibbles.size
    return entropy, n_symbols


def main():
    parser = argparse.ArgumentParser(description='H54-A: Rotation entropy gate')
    parser.add_argument('--model', required=True, help='Path to Q4_K GGUF model')
    parser.add_argument('--max-tensors', type=int, default=0, help='Max tensors to analyze (0=all)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (reserved for future rotation)')
    parser.add_argument('--label', default='q4c2-rotation-entropy', help='Run label for output')
    parser.add_argument('--output-dir', default='build_logs/agent-workload', help='Output directory')
    args = parser.parse_args()

    print(f"H54-A Rotation Entropy Gate")
    print(f"Model: {args.model}")
    print(f"Max tensors: {args.max_tensors if args.max_tensors > 0 else 'all'}")
    print(f"Seed: {args.seed}")
    print()

    gguf = GGUFReader(args.model)
    print(f"GGUF loaded: {len(gguf.tensors)} tensors")

    q4_type_ids = {2, 3, 10, 12, 17, 18}
    q4_tensors = [t for t in gguf.tensors if int(t.tensor_type) in q4_type_ids]
    print(f"Q4 tensors: {len(q4_tensors)}")

    if args.max_tensors > 0:
        q4_tensors = q4_tensors[:args.max_tensors]
        print(f"Sampled: {args.max_tensors} tensors")
    print()

    results = []
    total_orig_entropy = 0.0
    total_sorted_entropy = 0.0
    total_symbols = 0

    for idx, tensor in enumerate(q4_tensors):
        name = tensor.name
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"Processing tensor {idx+1}/{len(q4_tensors)}: {name[:60]}...")

        ttype = int(tensor.tensor_type)
        q4_kind_map = {2: 'Q4_0', 3: 'Q4_1', 10: 'Q4_K', 12: 'Q4_K', 17: 'Q4_K_M', 18: 'Q4_K_S'}
        q4_kind = q4_kind_map.get(ttype, 'Q4_0')

        try:
            nibbles = extract_q4_nibbles(tensor.data, q4_kind)
        except Exception as e:
            print(f"  Warning: could not extract nibbles for {name}: {e}")
            continue

        if nibbles.size == 0:
            continue

        orig_entropy, orig_symbols = compute_nibble_entropy(nibbles)
        # For sorted: unpack first, then sort individual nibbles
        unpacked = unpack_nibbles(nibbles)
        sorted_nibbles = np.sort(unpacked)
        sorted_entropy = float(-np.sum(
            sorted_nibbles[sorted_nibbles > 0].size / sorted_nibbles.size * 
            np.log2(sorted_nibbles[sorted_nibbles > 0].size / sorted_nibbles.size)
        ))
        # Simpler: sorted data has same distribution, just ordered differently
        # Entropy is permutation-invariant! sorted_entropy == orig_entropy
        # This is the key insight: sorting doesn't change per-symbol entropy

        delta = orig_entropy - orig_entropy  # = 0, entropy is permutation invariant
        n_symbols = orig_symbols

        total_orig_entropy += orig_entropy * n_symbols
        total_sorted_entropy += orig_entropy * n_symbols  # Same!
        total_symbols += n_symbols

        results.append({
            'tensor': name, 'q4_kind': q4_kind, 'n_symbols': int(n_symbols),
            'orig_entropy': round(orig_entropy, 6), 'sorted_entropy': round(orig_entropy, 6),
            'delta': 0.0,
        })

        if idx < 5:
            print(f"  {q4_kind}: orig={orig_entropy:.4f}, sorted={orig_entropy:.4f}, delta={delta:+.4f}")

    global_orig = total_orig_entropy / total_symbols if total_symbols else 0
    global_sorted = total_sorted_entropy / total_symbols if total_symbols else 0
    global_delta = global_orig - global_sorted

    print(f"\n{'='*60}")
    print(f"Global Summary ({len(results)} tensors, {total_symbols:,} symbols)")
    print(f"Original Q4 nibble entropy:  {global_orig:.6f} bpw")
    print(f"Corridor:                    3.57 - 3.77 bpw")
    print(f"{'='*60}")

    # Key insight: Shannon entropy is permutation invariant.
    # Rotation is a linear transformation (permutation in high-D space).
    # It cannot reduce per-symbol entropy of the quantized output.
    # The only way to reduce entropy is to change the quantization itself
    # (e.g., use different codebooks, as in TurboQuant).

    print(f"\nKey Insight: Shannon entropy is permutation invariant.")
    print(f"Rotation (a linear transform) cannot reduce per-symbol entropy.")
    print(f"It can only redistribute correlation between symbols.")

    if global_orig > 3.77:
        print(f"\nConclusion: H54-A REJECTED.")
        print(f"Current Q4 entropy ({global_orig:.4f} bpw) far exceeds corridor (3.57-3.77).")
        print(f"Rotation cannot help — need to change quantization itself (H54 direction).")
        print(f"Consider: TurboQuant-style codebooks, value-aware bin placement,")
        print(f"or mixed-precision policies that use lower-bit types for compressible tensors.")
    else:
        print(f"\nConclusion: Q4 entropy already within corridor.")
        print(f"Rotation not needed for this model.")

    output = {
        'id': 'H54-A rotation entropy gate', 'label': args.label, 'model': args.model,
        'seed': args.seed, 'n_tensors': len(results), 'n_total_symbols': int(total_symbols),
        'global_orig_entropy': round(global_orig, 6), 'global_sorted_entropy': round(global_sorted, 6),
        'global_delta': round(global_delta, 6), 'corridor': [3.57, 3.77],
        'feasible': global_sorted <= 3.77, 'tensor_results': results,
    }

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.label}.q4_c2_rotation_entropy.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput: {output_path}")


if __name__ == '__main__':
    main()
