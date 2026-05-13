# E011: H14 C01 Shape-Presence + Narrow Stream-k Gate

Date: 2026-05-13
Owner: Copilot
Experiment ID: E011
Stage: analytic gate + micro A/B + trace validation

## Goal

Increase the hit-rate of C01 kernel experiments by forcing a shape-presence check before any shape-scoped MMQ change.

This run tested one narrow mechanism-only probe:
- disable stream-k only for `RDNA4 + Q3_K + ne11 in {139,140}`.
- do not change route selection.

## Pre-Gate (Amdahl)

From C01 baseline trace:
- `MUL_MAT forward total = 969.854 ms`
- target q3_K direct cluster = `386.397 ms`
- share: `f = 0.3984`

Model:

`S = 1 / ((1 - f) + f / s_local)`

Required local cluster speedup:
- `S=1.01x` -> `s_local=1.0255x`
- `S=1.02x` -> `s_local=1.0518x`
- `S=1.03x` -> `s_local=1.0789x`
- `S=1.05x` -> `s_local=1.1357x`

Gate implication:
- sub-5% local gains are unlikely to create stable lane uplift.

## Code Probe

Temporary env-gated prototype in `ggml/src/ggml-cuda/mmq.cu`:
- `GGML_MMQ_RDNA4_Q3K_139_140_NO_STREAMK=1`

Status:
- reverted after validation (no keep).

## Commands

Build:

```bash
cmake --build build-rocm-vec --config Release -j 16
```

Lane A/B (`review_bug,patch_sim`, `r1`):

```bash
python scripts/agent_workload_bench.py --label e011-c01-two-tasks-r1-baseline --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks full --task-ids review_bug,patch_sim --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 192 --cache-type-k q4_0 --cache-type-v q4_0 --max-tokens 120 --real-context-mode repo-snapshot --no-reuse --background-server-policy fail --task-fail-timeout 0

GGML_MMQ_RDNA4_Q3K_139_140_NO_STREAMK=1 python scripts/agent_workload_bench.py --label e011-c01-two-tasks-r1-candidate-nostreamk139140 --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks full --task-ids review_bug,patch_sim --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 192 --cache-type-k q4_0 --cache-type-v q4_0 --max-tokens 120 --real-context-mode repo-snapshot --no-reuse --background-server-policy fail --task-fail-timeout 0
```

Trace validation:

```bash
GGML_TRACE_MMQ_TIMING=1 GGML_TRACE_MMQ_TIMING_SYNC=1 python scripts/agent_workload_bench.py --label e011-c01-two-tasks-trace-r1-baseline-mmqtiming --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks full --task-ids review_bug,patch_sim --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 192 --cache-type-k q4_0 --cache-type-v q4_0 --max-tokens 120 --real-context-mode repo-snapshot --no-reuse --background-server-policy fail --task-fail-timeout 0 --trace-preset kernel-full

GGML_MMQ_RDNA4_Q3K_139_140_NO_STREAMK=1 GGML_TRACE_MMQ_TIMING=1 GGML_TRACE_MMQ_TIMING_SYNC=1 python scripts/agent_workload_bench.py --label e011-c01-two-tasks-trace-r1-candidate-nostreamk139140-mmqtiming --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks full --task-ids review_bug,patch_sim --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 192 --cache-type-k q4_0 --cache-type-v q4_0 --max-tokens 120 --real-context-mode repo-snapshot --no-reuse --background-server-policy fail --task-fail-timeout 0 --trace-preset kernel-full
```

## Measured

TPS A/B:
- baseline: `9.34`
- candidate: `9.36`

Trace A/B:
- baseline: `6.64`
- candidate: `6.56`

Critical validation finding:
- MMQ timing lines in trace show `ncols_max=192` for this lane slice.
- The 139/140-scoped condition was effectively not exercised in this experiment.

## Decision

Decision: `revert`

Reason:
- shape-scoped toggle was not active on measured buckets;
- trace run regressed;
- no mechanism-level evidence of a valid improvement.

## Key Lesson

For C01 shape-specific prototypes:
1. verify target `ncols` histogram in the exact lane first,
2. only then run shape-scoped kernel A/B,
3. reject any run where the target condition did not activate.
