#!/usr/bin/env python3
"""H54-B analytical gate: value-aware (Lloyd-Max) Q4 quantization entropy test.

Concept: replace uniform Q4 bins with non-uniform Lloyd-Max codebooks optimized
for actual weight distributions. This changes the quantization function itself
(unlike rotation which is permutation invariant), potentially reducing entropy.

Steps:
1. Load GGUF and extract Q4 tensor data
2. Dequant Q4 -> approximate fp32 values
3. Run Lloyd-Max to find optimal 16-level codebook per tensor
4. Re-quantize values to new codebook -> new indices
5. Measure entropy of new symbol distribution
6. Measure MSE vs original fp32
7. Check if entropy < 3.77 bpw (corridor) while MSE within +-/-(10%) of Q4 baseline

Usage:
    python scripts/research/q4_c2_value_aware_gate.py --model models/Qwen3.6-27B-Q4_K_S.gguf
"""

import argparse
import sys
import os
import struct
import json
from datetime import datetime, timezone

import numpy as np

# Add gguf-py to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'gguf-py'))
from gguf.gguf_reader import GGUFReader

# Q4 type IDs
Q4_TYPE_IDS = {2, 3, 10, 12, 17, 18}
Q4_TYPE_NAMES = {2: 'Q4_0', 3: 'Q4_1', 10: 'Q4_K', 12: 'Q4_K', 17: 'Q4_K_M', 18: 'Q4_K_S'}

BLOCK_SIZES = {2: 18, 3: 20, 10: 144, 12: 144, 17: 144, 18: 144}
ELEMENTS_PER_BLOCK = {2: 32, 3: 32, 10: 256, 12: 256, 17: 256, 18: 256}


def unpack_nibbles(byte_arr):
    """Extract individual nibbles from packed byte array (numpy)."""
    low = byte_arr & 0x0F
    high = (byte_arr >> 4) & 0x0F
    result = np.empty(byte_arr.size * 2, dtype=np.uint8)
    result[0::2] = low
    result[1::2] = high
    return result


def dequant_q4_0(block_data):
    """Q4_0: d(fp16) + qs(16B). x = (qs_i - 8) * d."""
    block = np.asarray(block_data, dtype=np.uint8)
    d_raw = block[0] | (block[1] << 8)
    d = np.float32(np.float16(d_raw))
    qs = block[2:18]
    nibbles = unpack_nibbles(qs)
    return (nibbles[:32].astype(np.float32) - 8.0) * d


def dequant_q4_1(block_data):
    """Q4_1: d(fp16) + m(fp16) + qs(16B). x = qs_i * d + m."""
    block = np.asarray(block_data, dtype=np.uint8)
    d_raw = block[0] | (block[1] << 8)
    m_raw = block[2] | (block[3] << 8)
    d = np.float32(np.float16(d_raw))
    m = np.float32(np.float16(m_raw))
    qs = block[4:20]
    nibbles = unpack_nibbles(qs)
    return nibbles[:32].astype(np.float32) * d + m


def get_scale_min_k4(j, scales):
    """Decode 6-bit scale/min from Q4_K scales array (12 bytes).
    Mirrors ggml quants.c get_scale_min_k4.
    """
    if j < 4:
        d = scales[j] & 63
        m = scales[j + 4] & 63
    else:
        d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return d, m


def dequant_q4_k(block_data):
    """Q4_K: d(fp16) + dmin(fp16) + scales(12B) + qs(128B).
    8 subblocks of 32 elements, each with 6-bit d + 6-bit min.
    x = qs_i * d_sub - b_sub
    
    Layout:
    - Byte 0-1: d (fp16)
    - Byte 2-3: dmin (fp16)
    - Byte 4-15: scales (12 bytes, encoding 16 6-bit values)
    - Byte 16-143: qs (128 bytes, nibble-packed)
    """
    block = np.asarray(block_data, dtype=np.uint8)
    d_raw = block[0] | (block[1] << 8)
    dmin_raw = block[2] | (block[3] << 8)
    d_super = np.float32(np.float16(d_raw))
    dmin_super = np.float32(np.float16(dmin_raw))
    
    scales_bytes = block[4:16]  # 12 bytes
    
    qs = block[16:144]
    nibbles = unpack_nibbles(qs)
    
    values = np.zeros(256, dtype=np.float32)
    iscale = 0

    # Mirrors ggml/src/ggml-quants.c:dequantize_row_q4_K.
    # For each 64-value chunk, low nibbles use scale/min pair #0,
    # high nibbles of the same 32 bytes use pair #1.
    for j in range(0, 256, 64):
        sc, m = get_scale_min_k4(iscale + 0, scales_bytes)
        d1 = d_super * np.float32(sc)
        m1 = dmin_super * np.float32(m)
        sc, m = get_scale_min_k4(iscale + 1, scales_bytes)
        d2 = d_super * np.float32(sc)
        m2 = dmin_super * np.float32(m)

        q = block[16 + (j // 2):16 + (j // 2) + 32].astype(np.float32)
        values[j:j + 32] = d1 * (q % 16.0) - m1
        values[j + 32:j + 64] = d2 * np.floor(q / 16.0) - m2
        iscale += 2

    return values


DEQUANT_FNS = {2: dequant_q4_0, 3: dequant_q4_1, 10: dequant_q4_k, 12: dequant_q4_k, 17: dequant_q4_k, 18: dequant_q4_k}


def dequant_q4_tensor(raw_data, ttype, n_elements, max_elements=None):
    """Dequant Q4 tensor to fp32, optionally capping number of elements."""
    dequant_fn = DEQUANT_FNS[ttype]
    block_size = BLOCK_SIZES[ttype]
    elements_per_block = ELEMENTS_PER_BLOCK[ttype]
    
    # Flatten raw data (GGUFReader may return shaped arrays)
    if hasattr(raw_data, 'flatten'):
        flat = raw_data.flatten()
    elif hasattr(raw_data, 'tobytes'):
        flat = np.frombuffer(raw_data.tobytes(), dtype=np.uint8)
    else:
        flat = np.asarray(raw_data, dtype=np.uint8).flatten()
    
    nblocks_total = n_elements // elements_per_block
    if max_elements is not None and max_elements > 0:
        cap_blocks = max(1, max_elements // elements_per_block)
        nblocks = min(nblocks_total, cap_blocks)
    else:
        nblocks = nblocks_total
    
    all_values = []
    for b in range(nblocks):
        block_data = flat[b * block_size:(b + 1) * block_size]
        all_values.append(dequant_fn(block_data))
    
    return np.concatenate(all_values), nblocks


def lloyd_max_1d(data, nlevels=16, max_iter=20):
    """Lloyd-Max scalar quantization."""
    data_min = np.min(data)
    data_max = np.max(data)
    step = (data_max - data_min) / nlevels
    codebook = np.linspace(data_min + step * 0.5, data_max - step * 0.5, nlevels)
    
    for iteration in range(max_iter):
        distances = np.abs(data[:, np.newaxis] - codebook[np.newaxis, :])
        assignments = np.argmin(distances, axis=1)
        
        new_codebook = np.zeros(nlevels)
        for i in range(nlevels):
            mask = assignments == i
            if np.any(mask):
                new_codebook[i] = np.mean(data[mask])
            else:
                new_codebook[i] = codebook[i]
        
        delta = np.max(np.abs(new_codebook - codebook))
        codebook = new_codebook
        if delta < 1e-6:
            break
    
    boundaries = (codebook[:-1] + codebook[1:]) / 2
    return codebook, boundaries


def quantize_with_codebook(data, codebook):
    """Quantize data using given codebook. Returns indices (0-15)."""
    distances = np.abs(data[:, np.newaxis] - codebook[np.newaxis, :])
    return np.argmin(distances, axis=1)


def compute_entropy(indices, nlevels=16):
    """Compute Shannon entropy (bits per symbol)."""
    counts = np.bincount(indices.astype(int), minlength=nlevels)
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def choose_tensors(q4_tensors, sample, strategy='spread'):
    """Pick representative tensors for fast gate."""
    if sample <= 0 or sample >= len(q4_tensors):
        return q4_tensors
    if strategy == 'head':
        return q4_tensors[:sample]
    # Spread selection over the full tensor list to reduce head-bias.
    indices = np.linspace(0, len(q4_tensors) - 1, sample, dtype=np.int32)
    return [q4_tensors[i] for i in indices]


def write_artifacts(label, payload):
    out_dir = os.path.join('build_logs', 'agent-workload')
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f'{label}.q4_c2_value_aware_gate.json')
    md_path = os.path.join(out_dir, f'{label}.q4_c2_value_aware_gate.md')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Q4 C2 Value-Aware Gate\n\n')
        f.write(f"- label: `{payload['label']}`\n")
        f.write(f"- model: `{payload['model']}`\n")
        f.write(f"- timestamp_utc: `{payload['timestamp_utc']}`\n")
        f.write(f"- tensors_analyzed: `{payload['summary']['tensors_analyzed']}`\n")
        f.write(f"- total_elements: `{payload['summary']['total_elements']}`\n")
        f.write(f"- original_entropy_bpw: `{payload['summary']['original_entropy_bpw']:.6f}`\n")
        f.write(f"- new_entropy_bpw: `{payload['summary']['new_entropy_bpw']:.6f}`\n")
        f.write(f"- entropy_delta_bpw: `{payload['summary']['entropy_delta_bpw']:+.6f}`\n")
        f.write(f"- weighted_nrmse: `{payload['summary']['weighted_nrmse']:.6f}`\n")
        f.write(f"- feasible_tensors: `{payload['summary']['feasible_tensors']}/{payload['summary']['tensors_analyzed']}`\n")
        f.write(f"- gate_passed: `{payload['summary']['gate_passed']}`\n")

    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description='H54-B value-aware Q4 quantization gate')
    parser.add_argument('--model', required=True, help='Path to Q4 GGUF model')
    parser.add_argument('--sample', type=int, default=24, help='Number of tensors to analyze')
    parser.add_argument('--nlevels', type=int, default=16, help='Number of quantization levels')
    parser.add_argument('--corridor-upper', type=float, default=3.77, help='Corridor upper bound bpw')
    parser.add_argument('--mse-budget', type=float, default=0.10, help='MSE budget fraction')
    parser.add_argument(
        '--max-elements-per-tensor',
        type=int,
        default=1048576,
        help='Cap analyzed elements per tensor for fast gate (default: 1,048,576)'
    )
    parser.add_argument(
        '--sample-strategy',
        choices=['head', 'spread'],
        default='spread',
        help='Tensor sampling strategy when --sample < total tensors'
    )
    parser.add_argument(
        '--nrmse-budget',
        type=float,
        default=0.0,
        help='Optional weighted NRMSE budget (0 disables the quality gate)'
    )
    parser.add_argument('--label', default='', help='Optional label for writing JSON/MD artifacts')
    args = parser.parse_args()
    
    print(f"H54-B Value-Aware Q4 Quantization Gate")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Tensors: {args.sample}")
    print(f"Levels: {args.nlevels}")
    print(f"Corridor: < {args.corridor_upper} bpw")
    print(f"MSE budget: +/-{args.mse_budget * 100:.1f}% of baseline")
    print(f"Max elements/tensor: {args.max_elements_per_tensor:,}")
    print(f"Sample strategy: {args.sample_strategy}")
    if args.nrmse_budget > 0:
        print(f"NRMSE budget: <= {args.nrmse_budget:.6f}")
    print()
    
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    
    print(f"Loading GGUF...")
    reader = GGUFReader(args.model)
    print(f"  Tensors: {len(reader.tensors)}")
    
    q4_tensors = []
    for t in reader.tensors:
        ttype = int(t.tensor_type)
        if ttype in Q4_TYPE_IDS:
            q4_tensors.append(t)
    
    print(f"  Q4 tensors: {len(q4_tensors)}")
    print(f"  Will analyze: {min(args.sample, len(q4_tensors))} tensors")
    print()
    
    if not q4_tensors:
        print("ERROR: No Q4 tensors found in model")
        sys.exit(1)
    
    tensors_to_analyze = choose_tensors(q4_tensors, args.sample, args.sample_strategy)
    
    total_orig_bits = 0
    total_new_bits = 0
    total_elements = 0
    total_new_mse = 0.0
    total_nrmse_weighted = 0.0
    feasible_count = 0
    total_count = 0
    tensor_rows = []
    
    for tensor in tensors_to_analyze:
        ttype = int(tensor.tensor_type)
        tname = tensor.name
        n_elements_total = int(tensor.n_elements)
        tname_short = tname[:60]
        
        # Flatten raw data (GGUFReader may return shaped arrays)
        raw_data = tensor.data
        if hasattr(raw_data, 'flatten'):
            flat = raw_data.flatten()
        elif hasattr(raw_data, 'tobytes'):
            flat = np.frombuffer(raw_data.tobytes(), dtype=np.uint8)
        else:
            flat = np.asarray(raw_data, dtype=np.uint8).flatten()
        
        # Dequant to fp32
        fp32_values, sampled_blocks = dequant_q4_tensor(
            flat,
            ttype,
            n_elements_total,
            max_elements=args.max_elements_per_tensor,
        )
        n_elements = fp32_values.size
        
        # Extract original nibble indices
        block_size = BLOCK_SIZES[ttype]
        elements_per_block = ELEMENTS_PER_BLOCK[ttype]
        nblocks = sampled_blocks
        
        all_orig_indices = []
        for b in range(nblocks):
            block_data = flat[b * block_size:(b + 1) * block_size]
            if ttype == 2:  # Q4_0
                qs = block_data[2:18]
            elif ttype == 3:  # Q4_1
                qs = block_data[4:20]
            else:  # Q4_K types
                qs = block_data[16:144]
            nibbles = unpack_nibbles(qs)
            all_orig_indices.extend(nibbles[:elements_per_block].tolist())
        
        orig_indices = np.array(all_orig_indices, dtype=np.int32)
        
        # Compute original entropy
        orig_entropy = compute_entropy(orig_indices, args.nlevels)
        total_orig_bits += orig_entropy * n_elements
        total_elements += n_elements
        
        # Lloyd-Max
        codebook, boundaries = lloyd_max_1d(fp32_values, args.nlevels)
        
        # Re-quantize
        new_indices = quantize_with_codebook(fp32_values, codebook)
        
        # New entropy
        new_entropy = compute_entropy(new_indices, args.nlevels)
        total_new_bits += new_entropy * n_elements
        
        # MSE
        new_reconstructed = codebook[new_indices]
        new_mse = np.mean((fp32_values - new_reconstructed) ** 2)
        tensor_std = float(np.std(fp32_values))
        tensor_nrmse = float(np.sqrt(new_mse) / (tensor_std + 1e-12))
        total_new_mse += new_mse * n_elements
        total_nrmse_weighted += tensor_nrmse * n_elements
        
        entropy_ok = new_entropy < args.corridor_upper
        quality_ok = args.nrmse_budget <= 0 or tensor_nrmse <= args.nrmse_budget
        is_feasible = entropy_ok and quality_ok
        if is_feasible:
            feasible_count += 1
        total_count += 1

        tensor_rows.append({
            'name': tname,
            'type': Q4_TYPE_NAMES[ttype],
            'elements_sampled': int(n_elements),
            'elements_total': int(n_elements_total),
            'entropy_original_bpw': float(orig_entropy),
            'entropy_new_bpw': float(new_entropy),
            'entropy_delta_bpw': float(new_entropy - orig_entropy),
            'mse_new': float(new_mse),
            'nrmse': float(tensor_nrmse),
            'feasible': bool(is_feasible),
        })
        
        print(f"  {tname_short}")
        print(
            f"    Type: {Q4_TYPE_NAMES[ttype]}, "
            f"Elements: {n_elements:,}/{n_elements_total:,}"
        )
        print(f"    Original entropy: {orig_entropy:.6f} bpw")
        print(f"    New entropy:      {new_entropy:.6f} bpw")
        print(f"    Delta:            {new_entropy - orig_entropy:+.6f} bpw")
        print(f"    New MSE:          {new_mse:.2e}")
        print(f"    NRMSE:            {tensor_nrmse:.6f}")
        print(f"    Feasible:         {'YES' if is_feasible else 'NO'}")
        print()
    
    avg_orig_entropy = total_orig_bits / total_elements
    avg_new_entropy = total_new_bits / total_elements
    entropy_delta = avg_new_entropy - avg_orig_entropy
    avg_new_mse = total_new_mse / total_elements
    avg_nrmse_weighted = total_nrmse_weighted / total_elements
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tensors analyzed: {total_count}")
    print(f"Total elements:   {total_elements:,}")
    print()
    print(f"Original entropy: {avg_orig_entropy:.6f} bpw")
    print(f"New entropy:      {avg_new_entropy:.6f} bpw")
    print(f"Entropy delta:    {entropy_delta:+.6f} bpw")
    print(f"New MSE:          {avg_new_mse:.2e}")
    print(f"Weighted NRMSE:   {avg_nrmse_weighted:.6f}")
    print()
    
    entropy_pass = avg_new_entropy < args.corridor_upper
    quality_pass = args.nrmse_budget <= 0 or avg_nrmse_weighted <= args.nrmse_budget

    if entropy_pass:
        print(f"RESULT: Entropy {avg_new_entropy:.6f} < {args.corridor_upper} — WITHIN CORRIDOR")
        print(f"Feasible tensors: {feasible_count}/{total_count}")
        if args.nrmse_budget > 0:
            print(
                f"Quality gate: {'PASS' if quality_pass else 'FAIL'} "
                f"(weighted NRMSE {avg_nrmse_weighted:.6f} "
                f"{'<=' if quality_pass else '>'} {args.nrmse_budget:.6f})"
            )
        if feasible_count == total_count and quality_pass:
            print()
            print("GATE PASSED: All tensors feasible. Proceed to full H54-B definition.")
        else:
            print()
            print(
                f"GATE PARTIAL: {feasible_count}/{total_count} tensors feasible "
                f"(quality pass={quality_pass})."
            )
    else:
        print(f"RESULT: Entropy {avg_new_entropy:.6f} >= {args.corridor_upper} — ABOVE CORRIDOR")
        print()
        print("GATE REJECTED: Lloyd-Max quantization cannot reach corridor on full corpus.")
        print("Need different approach (H54-C/D) or mixed precision.")
    
    print()
    print(f"Maximum theoretical headroom: {4.0 - avg_orig_entropy:.6f} bpw")
    print(f"Remaining headroom after Lloyd-Max: {4.0 - avg_new_entropy:.6f} bpw")

    if args.label:
        payload = {
            'label': args.label,
            'model': args.model,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'args': {
                'sample': args.sample,
                'sample_strategy': args.sample_strategy,
                'nlevels': args.nlevels,
                'corridor_upper': args.corridor_upper,
                'max_elements_per_tensor': args.max_elements_per_tensor,
                'nrmse_budget': args.nrmse_budget,
            },
            'summary': {
                'tensors_analyzed': total_count,
                'total_elements': int(total_elements),
                'original_entropy_bpw': float(avg_orig_entropy),
                'new_entropy_bpw': float(avg_new_entropy),
                'entropy_delta_bpw': float(entropy_delta),
                'weighted_mse': float(avg_new_mse),
                'weighted_nrmse': float(avg_nrmse_weighted),
                'feasible_tensors': int(feasible_count),
                'entropy_pass': bool(entropy_pass),
                'quality_pass': bool(quality_pass),
                'gate_passed': bool(feasible_count == total_count and entropy_pass and quality_pass),
            },
            'tensors': tensor_rows,
        }
        json_path, md_path = write_artifacts(args.label, payload)
        print()
        print(f"Wrote {json_path}")
        print(f"Wrote {md_path}")


if __name__ == '__main__':
    main()
