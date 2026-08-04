# D066: P003 H54-B Value-Aware Quantization Scout

Date: 2026-05-28
Previous: D063 (symbol-level reordering rejected)
Status: **scout — defining direction**

## Context

Permutation-only transforms cannot reduce per-symbol Shannon entropy. A useful
continuation therefore has to change the quantization mapping itself.

The fundamental problem remains: current Q4 layout has `H1=3.864885 bpw`, leaving only `~0.135 bpw` headroom for any compression overhead. We need to change the quantization itself.

## H54-B: Value-Aware Quantization

**Core idea:** Standard Q4 uses uniform bins (16 equally-spaced levels between min/max per block). Most LLM weight distributions are NOT uniform — they have heavy tails, sharp peaks near zero, and sparse outliers. Non-uniform bin placement that matches actual data distributions could produce lower-entropy symbol distributions.

### Why This Could Work

Uniform Q4 quantization maps values to nearest of 16 equally-spaced bins. In a zero-centered Gaussian (typical for LLM weights), this creates a near-uniform symbol distribution because every bin covers roughly the same probability mass. Hence `H1 ≈ 3.865 bpw` — close to the max entropy `log2(16) = 4.0`.

Non-uniform bins can change this:
- **More bins near zero**: captures the high-density center more precisely, creating bias toward center symbols
- **Fewer bins in tails**: coarse quantization in low-density regions pushes outlier values to edge symbols
- **Result**: spikier symbol distribution with lower entropy

The key question: can we design bins such that the resulting symbol distribution has `H1 < 3.77 bpw` (corridor upper bound)?

### Lloyd-Max / k-Means Quantization

The standard approach for optimal scalar quantization is Lloyd-Max algorithm:
1. Start with initial bin boundaries
2. Assign each sample to nearest bin center
3. Recompute bin centers as mean of assigned samples
4. Reassign boundaries as midpoints between centers
5. Repeat until convergence

This minimizes MSE for a given number of levels. For our purposes, we need to minimize entropy while keeping MSE within acceptable bounds.

### Constraints

- **MSE budget:** Q4_K_S already has quantization error. New scheme must not increase it significantly. Current Q4_K_S perplexity drop is the baseline.
- **Decode complexity:** P003 hard limit `≤1.35`. Non-uniform dequant needs lookup table (16 entries) + index multiplication. This is cheap if table is small, but adds decode complexity.
- **Block-level vs global:** Per-block adaptive bins (like Q4_K already does with min/max scales) means each block has different bin boundaries. This is actually compatible with the Q4_K structure — we just replace uniform spacing with non-uniform.
- **Storage overhead:** If each block stores its own 16-level codebook, overhead is `16 × 2 bytes = 32 bytes/block` for fp16, or `16 × 1 byte = 16 bytes/block` for quantized levels. For Q4_K blocks (144 bytes), this is `11%` overhead. For Q4_0 blocks (18 bytes), this is `89%` overhead — too much.

### Approach Options

#### B1. Per-Block Lloyd-Max with fp16 Codebook
- Compute optimal 16-level codebook per block via Lloyd-Max
- Store codebook as 16 fp16 values (32 bytes) + indices as nibbles (16 bytes for 32 symbols)
- Total per superblock: `128 nibble bytes + 32 codebook bytes = 160 bytes` vs Q4_K `144 bytes`
- Overhead: `+11%` storage
- **Pros:** theoretically optimal for each block's distribution
- **Cons:** storage overhead, codebook quantization error, compute cost at quantize time

#### B2. Global Codebook with Per-Block Scale
- Compute one global 16-level codebook from full corpus statistics
- Each block stores only a scale factor (like Q4_K already does)
- Dequant: `value = codebook[index] × scale`
- Storage: no overhead vs standard Q4 (codebook stored once in GGUF header)
- **Pros:** no storage overhead, simple decode
- **Cons:** one codebook for all tensors may not match individual distributions well

#### B3. Per-Tensor Codebook with Scale
- One codebook per tensor (not per block)
- Each block stores scale + indices
- Codebook stored in GGUF tensor metadata
- **Pros:** matches per-tensor distribution, moderate overhead
- **Cons:** codebook storage adds up for many small tensors

#### B4. Entropy-Optimized Lloyd-Max
- Standard Lloyd-Max minimizes MSE
- Modified version minimizes entropy directly: move bin boundaries to create more spiky distributions
- Trade-off: higher MSE but lower entropy
- Need to find Pareto front: entropy vs MSE trade-off curve

## Analytical Gate Design

Before building any runtime code, need to prove the concept analytically:

1. **Extract Q4 corpus:** Use `q4_c2_symbol_atlas.py` to get all Q4 nibble data from Qwen3.6-27B-Q4_K_S
2. **Reverse-engineer pre-quant values:** Dequant Q4 → get approximate fp32 values
3. **Apply Lloyd-Max:** Compute optimal 16-level codebooks using actual weight distributions
4. **Re-quantize:** Map fp32 values to new codebook indices
5. **Measure entropy:** Compute H1 of new symbol distribution
6. **Measure MSE:** Compare new quantization error vs standard Q4
7. **Check corridor:** Can entropy drop below `3.77 bpw` while MSE stays within `±10%` of Q4 baseline?

### Expected Results

For a zero-centered Gaussian with σ:
- Uniform Q4 bins: entropy ≈ `3.865 bpw`
- Optimal Lloyd-Max bins: entropy ≈ `3.5-3.7 bpw` (theoretical)
- Real LLM weights are more complex (multi-modal, heavy-tailed)

The question is whether real-world distributions allow enough entropy reduction.

## Next Steps

1. Create `scripts/research/q4_c2_value_aware_gate.py`
2. Implement analytical gate:
   - Dequant Q4 → fp32
   - Lloyd-Max 16-level codebook per tensor
   - Re-quantize → new indices
   - Measure entropy + MSE
3. Run on Qwen3.6-27B-Q4_K_S corpus
4. If entropy < `3.77 bpw` and MSE within budget → proceed to full H54-B definition
5. If it fails, reconsider mixed precision or a different codebook design.

## Artifacts

- This document
- `scripts/research/q4_c2_symbol_atlas.py` (D055) for baseline extraction
- `ggml/src/ggml-common.h` for Q4_K layout definition

## Key insight

Shannon entropy is permutation invariant — rotation cannot help.
Value-aware quantization is different: it changes the quantization function itself,
which DOES change the probability distribution p(x) of output symbols.
This is the fundamental distinction that makes H54-B potentially viable.
