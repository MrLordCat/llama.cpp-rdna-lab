# E252 ROCm cuBLAS Fast16 Compute Gate

## Metadata

- Experiment ID: E252
- Date: 2026-05-25
- Owner: Copilot
- Branch/Commit: local `master` after `b4f2fddf5`
- Target lane: Qwen3.6-27B-Q3_K_S cold-first 12k repo-snapshot lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff`, `spec=none`, no reuse, no prime, thinking on.

## Hypothesis

- Statement: `HIPBLAS_COMPUTE_32F_FAST_16F` may improve the RDNA4 f16 hipBLAS path for staged Q3_K large prefill while preserving direct f32 output.
- Mechanism: E046 rejected `CUBLAS_COMPUTE_16F` because it uses f16 output plus a conversion back to f32. A fast16 compute type with f32 C/D avoids that extra result conversion and could select a faster rocBLAS kernel for f16 inputs.
- Why now: E228/E248 show the current cold bottleneck remains GEMM-side Q3_K cuBLAS. Library selector/grouped routes are exhausted, but this exact compute contract has not been gated on the current build.

## Implementation Plan

1. Add an opt-in env gate: `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F_FAST_16F=1`.
2. Keep f32 output and the existing fp16 staging contract.
3. If the route fails to build, fails at runtime, or regresses cold A/B, revert or leave only as a documented experiment if it is useful for diagnostics.

## Benchmark Plan

- Build `llama-server` in `build-rocm-vec`.
- Run one cold control and one cold candidate on the active Q3 lane.
- Use `--runs 3` only if candidate is clearly positive in r1.

## Result

- Build: `cmake --build build-rocm-vec --target llama-server --config Release -j 8` reached `Linking CXX executable bin\\llama-server.exe` without compile errors.
- Control: `e252-rocm12k-fast16-control-r1` completed, aggregate `7.3314 TPS`, prompt `6564.49 ms`, decode `2115.64 ms`.
- Candidate: `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F_FAST_16F=1` started normally, reached first prompt chunk (`6144` tokens), then failed inside `hipblasGemmEx` with `CUBLAS_STATUS_NOT_SUPPORTED` for the `HIP_R_16F x HIP_R_16F -> HIP_R_32F` contract using the fast16 compute type.
- Code status: the experimental env gate was reverted after the API gate failed, so no crashing runtime knob is kept.

## Decision

- Reject. ROCm 7.1 / hipBLAS on this Windows RDNA4 stack does not expose the desired `32F_FAST_16F` f32-output GEMM contract for the active Q3_K path. Do not repeat this compute-contract route unless ROCm/hipBLAS support changes.

## Artifacts

- `build_logs/agent-workload/e252-rocm12k-fast16-control-r1.diagnostics.md`
- `build_logs/agent-workload/e252-rocm12k-fast16-candidate-r1.diagnostics.md`
- `build_logs/agent-workload/e252-rocm12k-fast16-candidate-r1.server.log`
