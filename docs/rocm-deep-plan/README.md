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
   - Status: implemented and validated for blocker scope (Stage A+B+C+D, build-unblock + observability complete)
3. P3 HIP build pressure and TU split
   - File: P3_hip_build_pressure_and_tu_split.md
   - Status: closed for build-system objective; default/reduced runtime gates passed, mmvq-focused retained as non-production debug profile (not valid for active prompt-heavy lane)
4. P4 validation protocol and gates
   - File: P4_validation_protocol_and_gates.md
   - Status: deferred for now (use as protocol guardrails only)
5. P5 long-run server stability and throughput
   - File: P5_long_run_server_stability_and_throughput.md
   - Status: deferred for now (secondary 64k lane, not active speed target)
6. P6 prefill chunk contract alignment
   - File: P6_prefill_chunk_contract_alignment.md
   - Status: implemented (core hooks + build/smoke validation complete; full active-lane performance gate pending)
7. P7 RDNA4 GDN kernel hotpath rework
   - File: P7_rdna4_gdn_kernel_hotpath_rework.md
   - Status: pass2 complete; kernel-only uplift remains neutral, but root cause isolated to physical context n_ubatch inflation under shape-score. Experimental opt-in context cap restores ub512 to ub192 corridor; policy hardening pending.
8. P8 RDNA4 FATTN selector retargeting
   - File: P8_rdna4_fattn_selector_retargeting.md
   - Status: deep research complete, falsifiable theory upgraded; verdict CONDITIONAL GO globally but NO-GO as primary track on active lane

## Recommended execution order after theory stage

- P6 trace-first contract instrumentation, then adaptive chunk policy gate.
- P7 non-KDA GDN hotpath rework with env-guarded rollout.
- P8 selector retargeting only after P6/P7 first-pass results, and only for explicitly ambiguous shape bands.

## Definition of done for planning stage

- Scope, code map, findings, execution details, theoretical confirmation matrix, and GO/NO-GO verdict are written for each point.
- Risks and unknowns are explicit.
- Execution order is defined and dependencies are clear.

After this planning stage is approved, implementation and testing can start in controlled steps.
