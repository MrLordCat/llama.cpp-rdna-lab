# D050 - P003 target13 feasibility gate (Q4 MetaComp)

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: measured analytical gate

## Goal

Validate whether the practical target (`~13 GiB`) is reachable for
`Qwen3.6-27B-Q4_K_S` by metadata-only compaction while preserving 4-bit payload.

## Method

New script:

- `scripts/research/q4_metacomp_target_solver.py`

Run:

- model: `models/Qwen3.6-27B-Q4_K_S.gguf`
- target: `13.0 GiB`
- label: `q4metacomp-target13-qwen36-27b-q4ks-r1`

## Measured numbers

From `q4metacomp-target13-qwen36-27b-q4ks-r1.q4_metacomp_target.md`:

- current total: `15.004 GiB`
- target total: `13.000 GiB`
- required savings: `2.004 GiB`

Q4 decomposition:

- Q4 total: `12.355 GiB`
- Q4 payload floor (strict 4-bit): `10.983 GiB`
- Q4 metadata overhead currently present: `1.373 GiB`

Feasibility:

- maximum metadata-only savings: `1.373 GiB`
- required metadata save fraction for target: `1.4597` (impossible)
- gap after removing **all** Q4 metadata: `0.631 GiB`
- additional payload reduction still needed: `0.2299 bpw`

Scenario table from the run confirms that even `meta_save_frac=1.00` predicts
`13.631 GiB`, above target.

## Key insight

Metadata-only Q4-MetaComp cannot reach 13 GiB on this model.
A second mechanism is required on top of metadata compaction.

This gives a concrete quantitative target for the second mechanism:

- effective payload average must improve from `4.0000 bpw` to roughly
  `3.7701 bpw` (assuming metadata is already fully removed), or
- equivalent combined tradeoff between residual metadata and payload reduction.

## Next route (P003 continuation)

1. Keep Phase 1 converter work for metadata compaction (still valuable).
2. Start a bounded design track for **Q4 payload-side sub-4bpw average** while
   preserving Q4 code semantics:
   - block/superblock-local entropy coding of 4-bit symbols,
   - compact side-table for decode,
   - fallback path per tensor/block for worst-case entropy.
3. Require that any payload-side design is tested as:
   - offline size gate first,
   - then runtime correctness gate,
   - then lane fit/TPS/quality gate.

## Decision

Keep P003 open, but split into two mandatory components:

- C1: metadata compaction (already scoped),
- C2: payload average-bpw reduction without Q3 class transition.

Only C1+C2 together can plausibly hit the 13 GiB target.

## Artifacts

- `scripts/research/q4_metacomp_target_solver.py`
- `build_logs/agent-workload/q4metacomp-target13-qwen36-27b-q4ks-r1.q4_metacomp_target.json`
- `build_logs/agent-workload/q4metacomp-target13-qwen36-27b-q4ks-r1.q4_metacomp_target.md`
- `docs/research/major-topology/D049_P003_Q4_METACOMP_PHASE0_ESTIMATE.md`
