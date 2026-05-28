# D064: P003 H54 Quantization Redesign Scout

Date: 2026-05-28  
Previous: D063 (H53 rejected)  
Status: **scout — defining the direction**

## Context

All symbol-level C2 compression formulations (H45-H53) failed to reach the
P003 payload corridor `3.57-3.77 bpw` on current Q4 layout:

- **H45** (entropy-bounded nibble stream): `3.864885 bpw` >= `3.7701` upper bound
- **H46** (superblock palette remap): `3.999999 bpw` — escape overhead kills gain
- **H47** (tuple dictionary coding): `4.500000 bpw` — tuple coding expands, not compresses
- **H48** (layer-adaptive mixed policy): 0 feasible policies in optimizer
- **H49** (context-conditioned entropy): `Hcond delta=0.000000` — adjacent bigrams add nothing
- **H50** (bounded-rANS micropages): `3.882697 bpw` best, 0 feasible configs
- **H53** (nibble reordering): empirical `delta=0`, analytical 0 feasible configs

**Root cause:** Current Q4 quantization produces payloads with `H1=3.864885 bpw`.
Maximum theoretical headroom is `4.0 - 3.864885 = 0.135115 bpw`. Any compression
scheme with overhead (headers, tables, indices, escape codes) cannot fit within
this tiny margin.

## H54 Direction: Change Quantization Itself

Instead of compressing the output of Q4 quantization, change the quantization
process to produce lower-entropy outputs.

### Why This Is Different

All previous C2 hypotheses tried to compress *after* quantization. The payload
is already a stream of 4-bit symbols. No matter how clever the coding, the
entropy floor is set by the quantization step.

H54 targets the root: **make the quantization produce more structured outputs**.

### Candidate Approaches

#### A. TurboQuant-Style Rotations (Householder Q)
- Rotate weight vectors through random orthogonal transform
- Spherical rotation concentrates energy, creating more compressible distributions
- Already exists in this fork (`ggml/src/ggml-turboq.c`)
- TBQ3_0/TBQ4_0 use this for 3-bit/4-bit
- **Key question:** Can we apply rotation to Q4 layout? The current Q4 uses
  per-block scales + 256 individual 4-bit values. Rotation would need to
  operate on the 256-element blocks before quantization.
- **Risk:** TBQ types are CPU-only; GPU decode support would need to be built
- **Scope:** Major — new quant type, new GGML ops, new GUI integration

#### B. Value-Aware Quantization
- Standard Q4 maps values to nearest of 16 uniform bins
- Adaptive binning could concentrate bins where data density is high
- This would create more "spiky" symbol distributions (some symbols rare, some common)
- Lower entropy = better compression
- **Risk:** Non-uniform bins change dequant semantics; need new quant/dequant ops
- **Scope:** Medium-High — new quant type, but can reuse existing block structure

#### C. TurboQuant + Q4 Hybrid
- Apply rotation to reduce entropy, but keep Q4-compatible layout
- Could produce "Q4-like" blocks that happen to have lower entropy
- Would need new quantization path but could potentially reuse Q4 decode
- **Risk:** Rotation changes values; exact Q4 compatibility impossible
- **Scope:** Medium — new quant path, potentially shared decode

#### D. Structured Random Projection
- Instead of per-block rotation, use a global/superblock random projection
- Creates correlated structure across blocks
- Could enable cross-block dictionary/predictive coding
- **Risk:** Breaks random access model; decode becomes sequential
- **Scope:** Very High — fundamental architecture change

## Scout Assessment

| Approach | Feasibility | Scope | P003 Relevance | Risk |
|----------|-------------|-------|----------------|------|
| A. TBQ Rotation | Medium | Major | High (proven concept) | CPU-only, GPU gap |
| B. Value-Aware | Medium | Medium-High | High (direct entropy) | Semantic change |
| C. TBQ+Q4 Hybrid | Low-Medium | Medium | Medium | Compatibility issues |
| D. Random Projection | Low | Very High | Medium-High | Random access broken |

## Recommended Path

**Start with A (TurboQuant-Style Rotations)** as the first H54 exploration:

1. TBQ already exists in this fork — code base is available
2. The rotation → compression pipeline is proven (TBQ3_0/TBQ4_0)
3. The key research question: can rotation reduce Q4 entropy enough?
4. First step: analytical — measure entropy of rotated Q4 blocks without building full runtime

**First analytical gate for H54-A:**
- Take Qwen3.6-27B-Q4_K_S payload
- Apply Householder Q rotation to 256-element blocks (same as TBQ4_0)
- Re-quantize rotated values to Q4 bins
- Measure resulting H1 entropy
- If H1 drops below `3.77 bpw`, the route is viable

**If H54-A fails**, explore B (Value-Aware) as next sub-direction.

## Next Steps

1. Create `scripts/research/q4_c2_rotation_entropy_gate.py`
2. Analytical-only: apply TBQ rotation to Q4 blocks, measure entropy
3. If promising, define full H54-A hypothesis with scope/cost estimate
4. Enter fast gate

## Artifacts

- This document
- `scripts/research/q4_c2_symbol_atlas.py` (D055) for baseline entropy extraction
- `ggml/src/ggml-turboq.c` / `ggml/src/ggml-turboq.h` for rotation reference
- `ggml/src/ggml-turboq-tables.h` for codebook reference
