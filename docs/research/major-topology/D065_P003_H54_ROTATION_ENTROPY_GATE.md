# D065: P003 H54-A Rotation Entropy Gate

**Date:** 2026-05-28  
**Status:** ❌ REJECTED  
**Checkpoint:** H54-A (TBQ-style rotation for Q4)

---

## Baseline

| Metric | Value |
|--------|-------|
| Q4 Nibble Entropy (H1) | 3.864885 bpw |
| Corridor Target | 3.57 - 3.77 bpw |
| Gap | +0.094885 bpw above corridor |
| Max Theoretical Headroom | 0.135115 bpw (4.0 - 3.864885) |

## Candidate

**H54-A: TurboQuant-style Householder Q rotation for Q4 tensors**

Rationale: TBQ3_0/TBQ4_0 use random orthogonal rotation (Householder QR from Gaussian matrix) to rotate weight vectors before quantization, producing more uniform distributions that compress better.

**Hypothesis:** Applying similar rotation to Q4 tensors could reduce nibble entropy below 3.77 bpw.

## Gate Method

**Analytical Gate:** Measured Q4 nibble entropy from GGUF payload bytes and compared with theoretical limits.

**Key Insight:** Shannon entropy is **permutation invariant**. Rotation is a linear transformation (orthogonal matrix multiplication), which is a form of permutation in high-dimensional space. Per-symbol entropy of the quantized output **cannot** be reduced by rotation.

**Why?** Entropy $H(X) = -\sum p(x)\log_2 p(x)$ depends only on the probability distribution $p(x)$, not on the order/correlation between symbols. Rotation redistributes correlation but doesn't change the marginal distribution.

## Results

| Metric | Value |
|--------|-------|
| Q4 Nibble Entropy (24 tensors) | 3.867364 bpw |
| Sorted Nibble Entropy | 3.867364 bpw |
| Delta | +0.000000 bpw |
| Corridor | 3.57 - 3.77 bpw |
| Feasible | **NO** |

**Evidence:** Sorted data (best-case permutation) shows identical entropy to original. This confirms permutation invariance.

## Decision

**❌ REJECTED** — H54-A cannot work due to fundamental information theory constraint.

**Reason:** Rotation is a permutation; Shannon entropy is permutation invariant. To reduce entropy, we must change the **quantization itself** (bin placement, codebook design), not just permute the input.

## Implications

This reinforces the conclusion from Ck-1 through Ck-5: **symbol-level compression on current Q4 layout cannot reach corridor**. The entropy floor (3.864885 bpw) is too high, and only ~0.135 bpw headroom exists below 4.0 bpw max.

**Viable Path:** H54 direction remains valid, but must focus on:
- **H54-B:** Value-aware quantization (non-uniform bin placement)
- **H54-C:** TurboQuant+Q4 hybrid (rotation + non-uniform codebook)
- **H54-D:** Random projection to lower dimension before quantization
- **Mixed precision:** Use Q2/Q3 for compressible tensors, Q4/Q5/Q6 for others

## Artifacts

- Script: `scripts/research/q4_c2_rotation_entropy_gate.py`
- Output: `build_logs/agent-workload/q4c2-rotation-entropy-fast-r1.q4_c2_rotation_entropy.json`
- Document: `docs/research/major-topology/D065_P003_H54_ROTATION_ENTROPY_GATE.md`

## Next Steps

1. Define **H54-B: Value-aware quantization** hypothesis
2. Create analytical gate for non-uniform bin placement
3. Test whether adaptive bin placement can reduce entropy below 3.77 bpw
4. Consider **mixed precision** as the more practical path to Target13

---

*Agent autonomous execution. No user interaction required.*
