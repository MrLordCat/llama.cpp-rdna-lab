# E032 - C01 F32 MMF Wide SSM Probe

## Metadata

- Experiment ID: E032
- Date: 2026-05-16
- Owner: Codex
- Target lane: C01 cold-first, `Qwen3.6-27B-Q3_K_S`, `ctx=12288`, `b=6144`, `ub=192`, KV `q4_0/q4_0`, `review_bug,patch_sim`, no-reuse, thinking on.

## Hypothesis

- Statement: route the Qwen SSM F32 alpha/beta `MUL_MAT` shape through tiled no-id `MMF` instead of `cublas_backend`.
- Mechanism: the E030 trace shows steady F32 cuBLAS SSM work around `1294 ms`; if `MMF` can handle `ncols_dst=192`, it might avoid some small-GEMM rocBLAS overhead.
- Why now: after E031 closed the small Q4_K center, this was the next visible non-Q3 subcenter with a measurable ceiling.

## Math / Theory

- Current steady F32 cuBLAS split from `c01-e030-resume-r1-resources.server.log`:
  - `src0=(5120,48,1,1)`, `src1=(5120,192,1,1)`, `dst=(48,192,1,1)`: `1294.385 ms`.
- A local 10% win on this shape would be roughly `129 ms` inside steady `MUL_MAT forward`, or under 1% total wall on this lane.
- Failure gate:
  - if route does not flip from `cublas_backend` to `mul_mat_f_direct`, runtime deltas are noise and the experiment is rejected.

## Implementation Plan

1. Prototype no-id `MMF` column tiling for `ncols_dst > 16`.
2. Add RDNA4 env gate `GGML_CUDA_RDNA4_F32_MMF_WIDE` scoped to F32 SSM `src0_ne[1] == 48`, `src1_ncols <= 192`.
3. Build, run trace A/B, inspect route activation.
4. Revert if activation fails or target/runtime regress.

## Benchmark Plan

- Baseline:
  `c01-e032-mmfwide-control-r1`
- Candidate:
  `GGML_CUDA_RDNA4_F32_MMF_WIDE=1 c01-e032-mmfwide-candidate-r1`
- Runs: trace `r1`, two C01 tasks.
- Artifacts:
  - `build_logs/agent-workload/c01-e032-mmfwide-control-r1.csv`
  - `build_logs/agent-workload/c01-e032-mmfwide-candidate-r1.csv`
  - `build_logs/agent-workload/c01-e032-mmfwide-control-r1.server.log`
  - `build_logs/agent-workload/c01-e032-mmfwide-candidate-r1.server.log`

## Result

- Trace wall TPS: `6.3561 -> 6.4398` (`+1.32%`), but not accepted as a speed claim because the target route did not activate.
- Route check:
  - SSM `src0=(5120,48)`, `src1=(5120,192)` remained `cublas_backend`.
  - `mul_mat_vec_f_direct` only covered the existing tiny `ncols=2` cold/setup rows.
- Root cause:
  - current RDNA4 `MMF` path is not a cheap F32 route. `mul_mat_f` under `AMD_WMMA_AVAILABLE` only supports `half2`/`nv_bfloat162` for `MMF_ROWS_PER_BLOCK`.
  - F32 `ggml_cuda_should_use_mmf` requires `amd_mfma_available(cc)`, not RDNA4 WMMA.
  - the SSM shape also has `src0_ne[1]=48`, not divisible by current `MMF_ROWS_PER_BLOCK=32`.
- Decision: reject / no-activation.
- Code state: prototype reverted; `llama-server` rebuilt after revert.

## Notes

- Do not revisit F32 SSM via cheap `MMF` routing unless RDNA4 gets a real F32-capable MMF/MFMA path or a separate rows-16 F32 kernel is deliberately designed.
- The trace deltas in Q3/Q4 buckets during this run are treated as measurement noise because the intended F32 route did not flip.
