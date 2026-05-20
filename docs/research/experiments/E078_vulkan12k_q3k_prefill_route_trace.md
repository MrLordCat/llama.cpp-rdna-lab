# E078 Vulkan 12k Q3_K Prefill Route Trace

## Metadata

- Experiment ID: E078
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: local master, post-E076 workspace
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, ctx=12288, b=4096, ub=1024, q4_0/q4_0 KV, FlashAttention on, thinking on, no reuse, no prime, `--spec-type none`, task `quick/triage_diff`

## Hypothesis

- Statement: The remaining 12k Vulkan prompt gap is dominated by the active Q3_K large-prefill matmul route, and a code/route change in that path may close part of the ROCm prefill gap.
- Mechanism: Perf trace should identify whether the hot path is MMQ/int-dot, coopmat dequant matmul, FlashAttention, or a route-selection miss.
- Why now: E076 rejected valid no-code 32k gates; the next H31 work needs source-level Q3_K evidence.

## Math / Theory

- Assumptions: Same-lane cold runs are comparable only with matching ctx/batch/ubatch/KV/spec/reuse/thinking/task settings.
- Expected speedup corridor: at least +10-20% local Q3_K matmul improvement is needed for a visible prompt-heavy wall win; FlashAttention-only changes cannot close the gap because its share is much smaller.
- Failure conditions: Changes that reduce shader resources but do not improve wall time, or routes that add more overhead than matmul savings, should be reverted.

## Implementation Plan

1. Minimal code surface to change: Vulkan Q3_K matmul route and shaders only.
2. Guard rails: use `--runs 1` gates; compare to same-lane Vulkan baseline and ROCm control; do not use perf logger runs as speed claims.
3. Rollback path: revert each negative shader/runtime probe and rebuild `build-vulkan`.

## Benchmark Plan

- Baseline command: `scripts/agent_workload_bench.py` 12k lane with `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`, no reuse, no prime, `--spec-type none`.
- Candidate command: same command plus the candidate shader/runtime change or opt-in env.
- Number of runs: 1 for gate; 3 only for promising or borderline wins.
- Artifacts path: `build_logs/agent-workload/e078-*`

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- error rate
- prefill/decode split
- route/pipeline stats where diagnostic-only

## Result

- Outcome: route identified; code probes rejected and reverted.
- Delta:
  - Fresh same-lane controls: Vulkan `6.4679` aggregate, `905.64 tok/s` prompt, `40.36 tok/s` decode; ROCm `7.1936` aggregate, `1129.76 tok/s` prompt, `28.63 tok/s` decode.
  - Perf logger diagnostic: Q3_K `MUL_MAT` accounted for about `6164.503 ms` / `71.84%`; FlashAttention about `644.656 ms` / `7.51%`.
  - Active pipeline: `matmul_q3_k_f32_f16acc_aligned_l`, `140 VGPR`, `45 SGPR`, `22528 B LDS`, `0 B scratch`.
  - Q3_K `LOAD_VEC_A=4` probe: `6.4560` aggregate, `903.78 tok/s` prompt, `40.34 tok/s` decode; pipeline VGPR fell to `118`, but wall/prompt did not improve.
  - Coopmat-created Q3_K `Q8_1`/MMQ route probe: `3.7979` aggregate, `491.94 tok/s` prompt, `40.33 tok/s` decode; rejected.
  - Other rejected gates from this route pass: Q3_K MMQ `BK_STEP=8` `6.3616` / `887.93 tok/s`; post-guard `wm32-wn32` `4.8218` / `643.2 tok/s`; post-guard `wm128-wn32` `6.3220` / `881.97 tok/s`; current f16 KV route `5.7473` / `902.67 tok/s` prompt with decode regression.
- Confidence: high for route identification and rejection of the tested probes; speed gates used same cold lane and all recent runs had errors 0.
- Recommendation: continue H31 on active `mul_mm.comp` / `mul_mm_funcs.glsl` Q3_K coopmat dequant path, not MMQ-only probes. Do not promote Q8_1/MMQ route under coopmat for this lane.

## Notes

- Surprises: The Q8_1 route is absent under the coopmat pipeline creation branch by default (`q8_mmp=0`) even when int-dot and contiguous `src1` are available. Creating it experimentally made prefill much slower, so absence is not the current bottleneck.
- Follow-up action: Search for lower-risk Q3_K coopmat dequant optimizations or tile/resource changes that keep correctness guard coverage and improve wall time, then re-run the same 12k cold gate.