# ROCm Deep Plan (Execution + Planning)

This folder contains implementation dossiers for the current Qwen ROCm acceleration blockers.

Current constraints for planning-phase points:

- No runtime code modifications for points that are still in planning state.
- No performance claims from unmerged code.
- No test execution changes beyond point protocol unless explicitly approved.
- Each point must pass its theory gates before implementation starts.

## Active lane (reference)

- Model: models/Qwen3.6-27B-Q3_K_S.gguf
- Context: ctx=12288
- Batch/ubatch: 6144/192
- KV: q4_0/q4_0
- Workload: scripts/agent_workload_bench.py --tasks v2-review --real-context-mode repo-snapshot --no-reuse --no-disable-thinking

## Current points (one document per point)

1. P1 prefill shape route blocker
   - File: P1_prefill_shape_route.md
   - Status: implemented and validated on active lane (boundary cliff recovery confirmed)
2. P2 MMVQ decode and linker blocker
   - File: P2_mmvq_decode_and_link_blocker.md
   - Status: deep theoretical pre-implementation research updated (implementation not started)
3. P3 HIP build pressure and TU split
   - File: P3_hip_build_pressure_and_tu_split.md
4. P4 validation protocol and gates
   - File: P4_validation_protocol_and_gates.md
5. P5 long-run server stability and throughput
   - File: P5_long_run_server_stability_and_throughput.md
   - Status: log-derived hypotheses captured (planning complete, implementation not started)

## Definition of done for planning stage

- Scope, code map, findings, solution strategy, planned file changes, validation gates, and rollback criteria are written for each point.
- Risks and unknowns are explicit.
- Execution order is defined and dependencies are clear.

After this planning stage is approved, implementation and testing can start in controlled steps.
