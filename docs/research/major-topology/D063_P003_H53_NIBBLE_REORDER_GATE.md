# D063 - P003 H53 Nibble Reorder Gate

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: theory-only analytical gate (negative)

## Purpose

Test H53: can reordering nibbles within superblocks reduce effective entropy
enough to reach P003 corridor (3.57-3.77 bpw)?

## Method

Two-part gate:

1. **Empirical fast gate** (`scripts/research/q4_c2_nibble_reorder_gate.py`):
   - Sampled 24 largest Q4 tensors, 131072 blocks each
   - Sorted nibbles within each superblock
   - Measured conditional entropy on sorted vs original streams
   - Result: `Delta = 0.000000 bpw` — sorting within blocks gives no bigram
     entropy reduction (block boundaries dominate)

2. **Analytical gate** (`scripts/research/q4_c2_reorder_analytical_gate.py`):
   - Modeled sorted conditional entropy for block sizes 32-32768
   - Tested 4 permutation encoding methods
   - Result: no configuration reaches corridor 3.57-3.77 bpw

## Results

### Empirical gate
- Original conditional entropy: `3.867704 bpw`
- Sorted conditional entropy: `3.867704 bpw`
- Delta: `0.000000 bpw`
- Feasible configs: `0`

### Analytical gate
- Source H1: `3.864885 bpw`
- Max headroom: `0.135115 bpw`
- 44 configs tested (11 block sizes × 4 methods)
- Feasible configs: `0`
- Best overall: `N=32768, run_length, net=0.0096 bpw` (far below corridor)

## Interpretation

1. **Empirical**: Within-block sorting gives no benefit because bigram entropy
   is dominated by block-boundary transitions, which remain random even when
   intra-block order is sorted.

2. **Analytical**: Even with optimistic models, sorted data either:
   - Compresses too well (ECS: ~0.05 bpw) — far below corridor min 3.57
   - Has too much overhead (index/delta: 1-15 bpw) — far above corridor max 3.77

3. **Fundamental issue**: P003 corridor (3.57-3.77 bpw) leaves only `0.135 bpw`
   headroom over current entropy (`3.865 bpw`). Any compression scheme with
   headers/metadata/permutation info cannot fit within this margin.

4. **Conclusion**: Symbol-level reordering cannot solve P003. The problem is not
   compression algorithm choice — it's that Q4 quantization itself produces
   ~3.865 bpw entropy. The only viable path is changing quantization to produce
   lower-entropy payloads.

## Gate decision

1. H53 fast gate: negative (empirical + analytical)
2. H53 full gate: not warranted
3. H53 status: rejected
4. Next direction: H54 — change quantization to produce lower-entropy payloads

## Related artifacts

- `scripts/research/q4_c2_nibble_reorder_gate.py`
- `scripts/research/q4_c2_reorder_analytical_gate.py`
- `build_logs/agent-workload/q4c2-nibbler-order-qwen36-27b-q4ks-fast-r1.q4_c2_nibble_reorder_gate.json`
- `build_logs/agent-workload/q4c2-reorder-analytical-r1.q4_c2_reorder_analytical_gate.json`
