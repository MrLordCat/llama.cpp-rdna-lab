# E089 Vulkan 12k Q3_K LOAD_VEC_A=4 Stride20 Recheck

## Metadata

- Experiment ID: E089
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: local master after E086 kept and E087/E088 rejected
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, fixed pp7488 prefill gate

## Hypothesis

- Statement: After E086 changes Q3_K A-load grouping, the original coopmat stride20 may become competitive with E082 stride18.
- Mechanism: `LOAD_VEC_A=4` changes write pattern into shared memory; the best LDS stride may differ from the previous `LOAD_VEC_A=2` regime.

## Benchmark Plan

- Baseline: E086 stride18 + corrected Q3_K loadvec4, fixed pp7488 r3 `961.82 tok/s`.
- Candidate: Q3_K coopmat `SHMEM_STRIDE = BK/2+4` (stride20) with E086 loadvec4 intact.
- Runs: r1 gate first; revert if below E086.

## Result

- Outcome: reject and revert to E086/E082 stride18.
- Delta: fixed pp7488 r1 was `911.74 tok/s`, `-5.21%` vs E086 stride18 r3 `961.82 tok/s`.
- Recommendation: keep Q3_K coopmat stride18. The E082 LDS-stride win remains necessary after corrected Q3_K loadvec4.

Artifact:

- `build_logs/agent-workload/e089-vulkan-q3-loadvec4-stride20-pp7488.md`