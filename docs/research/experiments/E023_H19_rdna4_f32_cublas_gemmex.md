# E023 H19 RDNA4 F32 cuBLAS GemmEx Route

## Metadata

- Experiment ID: E023
- Date: 2026-05-16
- Owner: Codex
- Target lane: `Qwen3.6-27B-Q3_K_S`, `tasks=quick`, `task_ids=review_bug,patch_sim`, `ctx=12288`, `batch=6144`, `ubatch=192`, `q4_0/q4_0`, `spec=none`, no-reuse
- Hypothesis ID: H19

## Hypothesis

- Statement: On RDNA4/ROCm, routing F32 cuBLAS backend GEMMs through `cublasGemmEx(... CUDA_R_32F, CUBLAS_COMPUTE_32F ...)` may select a better rocBLAS kernel than `cublasSgemm` for Qwen SSM prompt shapes.
- Target shapes:
  - `blk.*.ssm_alpha.weight x attn_norm`: `src0=(5120,48)`, `src1=(5120,192)`, `dst=(48,192)`
  - `blk.*.ssm_beta.weight x attn_norm`: same shape
- Why now:
  - Current C01 trace shows `MUL_MAT f32 ne=(48,192,1,1)` at about `1249 ms`, using `route=cublas_backend`.
  - `MMF` custom kernel is not a legal cheap route for this shape because no-id MMF only supports `ncols_dst <= 16`.

## Math / Theory

- One SSM alpha/beta GEMM is roughly `2 * 48 * 192 * 5120 = 94.4M` FLOPs.
- Current trace average for this shape is about `0.17-0.18 ms`, so the kernel is not a huge wall share but is large enough for a cheap route probe.
- Best-case wall impact:
  - The shape contributes about `1.25 s` of `22.13 s` sync-only CUDA_NODE trace time.
  - A `10%` route win would be about `0.6%` total trace time.
- Gate:
  - only worth keeping if both aggregate TPS and target F32 `MUL_MAT` timing improve.

## Implementation Plan

1. Add a temporary env-gated RDNA4 branch in the F32 cuBLAS backend:
   - `GGML_RDNA4_F32_CUBLAS_GEMMEX=1`
   - use `cublasGemmEx` with `CUDA_R_32F`, `CUBLAS_COMPUTE_32F`, `CUBLAS_GEMM_DEFAULT_TENSOR_OP`
2. Build `llama-server`.
3. Run one C01 r1 screen.
4. If promising, run kernel trace; otherwise revert.

## Result

- Runtime screen:
  - baseline/reference: `c01-e015-rdna4-y64w4-r3 = 9.6080 TPS`
  - candidate: `c01-e023-rdna4-f32-gemmex-r1 = 9.42 TPS`
  - delta: about `-1.96%`
- Target trace:
  - baseline focus trace, `MUL_MAT f32 ne=(48,192)`: count `7296`, sum `1249.117 ms`, avg `0.1712 ms`
  - candidate trace, same target: count `3648`, sum `674.858 ms`, avg `0.1850 ms`
  - target avg delta: about `+8.1%` slower
- Broader trace signal:
  - other centers also worsened, including MMQ and GDN averages, so this was not an isolated target-only tradeoff.

## Decision

- `reject`
- Reason: both aggregate TPS and the intended F32 SSM target timing regressed.
- Code state: env-gated `GemmEx` branch reverted; `llama-server` rebuilt after rollback.
- Artifacts:
  - `build_logs/agent-workload/c01-e023-rdna4-f32-gemmex-r1.csv`
  - `build_logs/agent-workload/c01-e023-rdna4-f32-gemmex-trace-r1.server.log`
