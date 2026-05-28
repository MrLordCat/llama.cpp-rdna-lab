# D051 - P003 Phase 1 tensor plan gate (target13)

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: planning gate (per-tensor)

## Scope

Turn D050 global feasibility numbers into a concrete per-tensor Phase 1 plan for
converter implementation.

## Method

New script:

- `scripts/research/q4_metacomp_phase1_plan.py`

Run:

- model: `models/Qwen3.6-27B-Q4_K_S.gguf`
- target: `13.0 GiB`
- label: `q4metacomp-phase1plan-target13-qwen36-27b-q4ks-r1`

## Results

Global numbers (consistent with D050):

- current total: `15.004 GiB`
- required savings: `2.004 GiB`
- full Q4 metadata budget: `1.373 GiB`
- unresolved after full metadata compact: `0.631 GiB`
- payload-side gap at full metadata compact: `0.2299 bpw`

Per-tensor planner behavior:

- Metadata-first allocation saturates all major Q4 tensors at `meta_save_frac=1.0`.
- Residual requirement remains positive for high-share tensors, proving that
  Phase 1 metadata converter cannot close target alone.

## Decision

Keep Phase 1 converter implementation as mandatory C1 pipeline, but treat C2 as
hard requirement from day one:

- C1 (converter): metadata compaction + roundtrip validation.
- C2 (payload-side): sub-4bpw average mechanism while preserving Q4 code
  semantics.

For code planning this means:

1. Build converter interfaces with two outputs:
   - compact metadata stream,
   - optional payload-side auxiliary stream hook.
2. Keep fallback-per-tensor contract to disable C2 where entropy/distribution is
   unfavorable.

## Artifacts

- `scripts/research/q4_metacomp_phase1_plan.py`
- `build_logs/agent-workload/q4metacomp-phase1plan-target13-qwen36-27b-q4ks-r1.q4_metacomp_phase1_plan.json`
- `build_logs/agent-workload/q4metacomp-phase1plan-target13-qwen36-27b-q4ks-r1.q4_metacomp_phase1_plan.md`
- `docs/research/major-topology/D050_P003_Q4_TARGET13_FEASIBILITY_GATE.md`
