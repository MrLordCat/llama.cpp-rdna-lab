# E253 ROCm Src1 Reuse + Batch8192 Stack Gate

## Metadata

- Experiment ID: E253
- Date: 2026-05-25
- Owner: Copilot
- Branch/Commit: local `master` after `b4f2fddf5`
- Target lane: Qwen3.6-27B-Q3_K_S cold-first 12k repo-snapshot lane, `ctx=12288`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff`, `spec=none`, no reuse, no prime, thinking on.

## Hypothesis

- Statement: the confirmed E248 adjacent `src1` fp16 reuse may stack with a larger outer `batch=8192` prompt chunk shape.
- Mechanism: E248 saves a small amount of activation staging on adjacent Q3_K cuBLAS calls. E224 showed `batch=8192/ubatch=2048` can be slightly better than `6144/2048` in r1, likely because the prompt fits in one outer batch instead of two.
- Risk: prior batch sweeps were mostly noise/tie, and E248's ceiling is small; a combined stack is unlikely to close the 10 TPS target alone.

## Benchmark Plan

- Run `batch=8192, ubatch=2048` control with no experimental envs.
- Run the same shape with `GGML_EXPERIMENTAL_CUBLAS_SRC1_F16_REUSE=1` and `GGML_EXPERIMENTAL_CUBLAS_SRC1_F16_REUSE_MIN_NCOLS=1`.
- Promote to r3 only if r1 beats the E248 final3 best (`7.9608 TPS`) by a meaningful margin.

## Result

- Control `batch=8192/ubatch=2048`: `e253-rocm12k-b8192-control-r1` completed at `7.2756 TPS`, prompt `6636.51 ms`, decode `2113.16 ms`.
- Candidate with E248 reuse (`MIN_NCOLS=1`): `e253-rocm12k-b8192-src1reuse-r1` completed without benchmark errors but collapsed to `2.4701 TPS`, prompt `23715.59 ms`, decode `2158.68 ms`.
- Safety follow-up attempt: a `max_ncols=6144` guard was tested, but it did not fix the issue because the reuse path activates on internal `ubatch`-width matmuls, not the outer prompt batch. Guarded validation still measured `2.6003 TPS` at `batch=8192` and `2.5578 TPS` at `batch=6144`. The guard patch was reverted.

## Decision

- Reject as a speed stack. `batch=8192` did not beat the E248 final3 best, and broadening src1 reuse to this shape is strongly negative.
- Reclassify E248 as unstable after current rebuild revalidation. Do not recommend `GGML_EXPERIMENTAL_CUBLAS_SRC1_F16_REUSE=1` for launch/autotune profiles until a separate correctness/lifetime fix explains why the earlier E248 final3 positive result no longer reproduces.

## Artifacts

- `build_logs/agent-workload/e253-rocm12k-b8192-control-r1.diagnostics.md`
- `build_logs/agent-workload/e253-rocm12k-b8192-src1reuse-r1.diagnostics.md`
- `build_logs/agent-workload/e253-rocm12k-b8192-src1reuse-r1.server.log`
- `build_logs/agent-workload/e253b-rocm12k-b8192-src1guard-r1.diagnostics.md`
- `build_logs/agent-workload/e253b-rocm12k-b6144-src1guard-r1.diagnostics.md`
