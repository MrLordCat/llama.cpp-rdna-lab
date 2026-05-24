# E210 ROCm Q3_K Padded Storage Split Guard

## Metadata

- Experiment ID: E210
- Date: 2026-05-24
- Owner: Codex
- Branch/Commit: master after `efdcf48fe`
- Target lane: H43 Q3_K padded storage default-readiness on ROCm `build-rocm-vec`

## Hypothesis

- Statement: padded Q3_K storage must fail safe on CUDA split buffers until the split allocation/set/get path is physically padded-aware.
- Mechanism: keep the current env-gated padded storage route for normal CUDA buffers, but make the runtime predicate return false for split-buffer tensors. This avoids interpreting raw split-buffer model shards as `block_q3_K_padded`.
- Why now: E209 closed backend async raw-copy hazards, but split-buffer raw layout is still a correctness edge before any default-on discussion.

## Math / Theory

- Assumptions:
  - no TPS claim; this is correctness/safety work;
  - single-GPU Qwen lane should retain the existing opt-in padded route;
  - split-buffer tensors should silently stay on raw Q3_K layout until a real split padded-storage implementation exists.
- Expected speedup corridor:
  - none directly.
- Failure conditions:
  - normal non-split padded smokes stop passing;
  - the guard disables the known single-GPU opt-in route;
  - broader Q3_K smoke exposes a new supported-op correctness error.

## Implementation Plan

1. Minimal code surface:
   - `ggml/src/ggml-cuda/ggml-cuda.cu` Q3_K padded-storage predicate only.
2. Guard rails:
   - default remains unchanged;
   - no split-buffer speed claim;
   - run Q3_K `MUL_MAT` smokes and a broader Q3_K backend-op smoke under the opt-in env.
3. Rollback path:
   - revert the predicate guard if it breaks the known E201-P2a smokes.

## Benchmark Plan

- Candidate command: `test-backend-ops` with `GGML_CUDA_Q3K_PADDED_STORAGE=1 GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1`.
- Number of runs: correctness only.
- Artifacts path: `build_logs/agent-workload/e210-rocm-q3k-padded-split-guard-*`.

## Metrics

- build status
- Q3_K `MUL_MAT` smoke status
- broad Q3_K backend-op smoke supported rows/errors
- no TPS claim

## Result

- Outcome: kept as default-off correctness/safety work.
- Delta:
  - `cmake --build build-rocm-vec --config Release -j 8` passed;
  - non-split opt-in route remained active: prompt-like Q3_K `MUL_MAT` traced as `mul_mat_q_direct`, `split=0`, `q3k_padded=1`;
  - padded prompt-like smoke passed: `MUL_MAT type_a=q3_K,type_b=f32,m=1,n=64,k=256`;
  - padded decode-like smoke passed: `MUL_MAT type_a=q3_K,type_b=f32,m=16,n=1,k=256`;
  - broad Q3_K backend-op smoke under padded storage reported `total=52`, `supported=13`, `unsupported=39`, `supported_errors=0`;
  - supported broad-smoke rows included `MUL_MAT_ID` at `n=1` and `n=32`, so the narrow MoE-style matrix-id smoke did not expose a padded-storage correctness error.
- Confidence: medium/high for the intended fail-safe guard. It proves the single-GPU opt-in route still activates and supported Q3_K backend-op smokes have no correctness errors, but it does not implement split-buffer padded storage.
- Recommendation: keep the split guard. Multi-GPU/split Q3_K tensors should remain raw-layout unless a later experiment adds physical padded split allocation/set/get.

## Notes

- This still does not make H43 default. It narrows one unsafe edge so the opt-in route is less fragile.
- Artifacts:
  - `build_logs/agent-workload/e210-rocm-q3k-padded-mulmat-prompt-smoke.csv`
  - `build_logs/agent-workload/e210-rocm-q3k-padded-mulmat-decode-smoke.csv`
  - `build_logs/agent-workload/e210-rocm-q3k-padded-broad-smoke.txt`
  - `build_logs/agent-workload/e210-rocm-q3k-padded-route-proof.txt`
