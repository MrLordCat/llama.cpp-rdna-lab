# E048 H21 Prefill ubatch2048 hipBLASLt Route

## Metadata

- Experiment ID: E048
- Date: 2026-05-17
- Owner: Codex
- Branch/Commit: local `master`
- Target lane: Qwen3.6-27B-Q3_K_S cold-first prefill lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on.

## Hypothesis

- Statement: enabling the hipBLASLt-backed rocBLAS route may improve the new large-prefill `cublas_backend` path.
- Mechanism: E045 moved the practical cold-first prefill baseline to `ubatch=2048`, where large `Q3_K` and `Q4_K` matmuls route through cuBLAS/hipBLAS after fp16 dequant staging. The older E044 `ROCBLAS_USE_HIPBLASLT=1` gate was done on `ubatch=192`, where the Q3 hot path was MMQ/MMVQ-heavy, so it did not strongly test large GEMM selection.
- Why now: H20 compute16 and H08 GDN chunk gates did not beat the current `ubatch=2048` baseline, leaving large GEMM routing as the next cheap code/runtime-adjacent check.

## Math / Theory

- Assumptions:
  - E045 `ubatch=2048` prompt trace total: `13969.454 ms`.
  - `MUL_MAT`: `9053.320 ms`, `64.81%` of prompt trace.
  - `Q3_K` share inside `MUL_MAT`: `84.32%`.
  - Most `ubatch=2048` prompt Q3_K/Q4_K routes are `cublas_backend` with `src1_ne=(...,2048,...)`.
- Expected speedup corridor:
  - A `3%` local improvement in `MUL_MAT` would project to roughly `1.2%` aggregate wall speedup.
  - A `5%` local improvement in `MUL_MAT` would project to roughly `2.0%` aggregate wall speedup.
- Failure conditions:
  - hipBLASLt selects a slower kernel or adds planning overhead.
  - Windows ROCm runtime overhead dominates small/medium GEMMs.
  - Benefit appears only in trace noise and not in cold r1/r3 wall metrics.

## Implementation Plan

1. Minimal code surface to change: none for the gate; use `ROCBLAS_USE_HIPBLASLT=1`.
2. Guard rails: if positive, keep as explicit profile/env recommendation first; only code/GUI-promote after r3 and route trace.
3. Rollback path: unset `ROCBLAS_USE_HIPBLASLT`; no code rollback needed for the gate.

## Benchmark Plan

- Baseline command:
  - `python scripts/agent_workload_bench.py --label prefill-e048-hipblaslt-base-r1 ... --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --real-context-mode repo-snapshot --no-reuse --no-v2-prime-pass --no-disable-thinking --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0"`
- Candidate command:
  - same command with `ROCBLAS_USE_HIPBLASLT=1`.
- Number of runs:
  - r1 gate, r3 confirmation only if r1 is positive.
- Artifacts path:
  - `build_logs/agent-workload/prefill-e048-hipblaslt-*.csv`
  - `build_logs/agent-workload/prefill-e048-hipblaslt-*.server.log`

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- error rate
- prefill/decode split
- route/timing trace only if candidate is near-positive

## Result

- Outcome: neutral/noise.
- Baseline: `prefill-e048-hipblaslt-base-r1 = 11.5443 TPS`, prompt eval `1179.855 tok/s`, prompt mean `6288.010 ms`, decode eval `29.485 tok/s`.
- Candidate: `prefill-e048-hipblaslt-on-r1 = 11.5557 TPS`, prompt eval `1180.290 tok/s`, prompt mean `6285.765 ms`, decode eval `29.515 tok/s`.
- Delta: aggregate `+0.10%`; prompt eval `+0.04%`.
- Confidence: too small to distinguish from run noise, and far below the `>= +1%` r1 gate for promotion.
- Recommendation: reject as a default/profile change; `ROCBLAS_USE_HIPBLASLT=1` does not materially improve this lane.

## Notes

- Surprises: hipBLASLt did not meaningfully change large-prefill wall time despite the `cublas_backend` route.
- Follow-up action: prefer route-local timing/instrumentation over more library-wide GEMM toggles.
