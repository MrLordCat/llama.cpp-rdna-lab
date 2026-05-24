# E201 ROCm Q3_K Padded Storage P1/P2a Prototype

## Metadata

- Experiment ID: E201
- Date: 2026-05-24
- Owner: Codex
- Branch/Commit: master after E200 (`ef0f12e49`)
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: ROCm can safely test Vulkan-like Q3_K `110 -> 112` device storage if the first code slice is env-gated and correctness-first.
- Mechanism: add backend-private padded Q3_K storage for non-split CUDA/HIP buffers, add padded-aware Q3_K dequant and MMVQ accessors, and temporarily avoid MMQ under the padded-storage knob until MMQ gets its own accessor pass.
- Why now: E199/E200 show quick padded-layout shortcuts are invalid, but a staged storage route is technically sliceable if `test-backend-ops` catches stride/copy mistakes before real server.

## Math / Theory

- Assumptions:
  - padding replacement cost is `1.818%` of selected Q3_K storage, not a duplicate model copy;
  - `MUL_MAT n=1` exercises the decode-like MMVQ path;
  - `MUL_MAT n=64` can exercise prompt/matrix paths, but P2a may route away from MMQ while MMQ is not padded-aware.
- Expected speedup corridor:
  - no speed claim in P1/P2a;
  - P1/P2a is only a correctness gate for the larger route.
- Failure conditions:
  - build fails;
  - padded-storage env corrupts Q3_K `MUL_MAT`;
  - model load uses unsupported partial Q3_K set/get or split buffer path;
  - disabling MMQ under the knob makes any speed benchmark invalid as a parity claim.

## Implementation Plan

1. Minimal code surface to change:
   - `ggml/src/ggml-common.h`: padded Q3_K block struct;
   - `ggml/src/ggml-cuda/common.cuh`: env helper;
   - `ggml/src/ggml-cuda/ggml-cuda.cu`: non-split Q3_K padded alloc/set/get/copy and MMQ guard;
   - `ggml/src/ggml-cuda/convert.cu`: padded-aware Q3_K dequant dispatch;
   - `ggml/src/ggml-cuda/vecdotq.cuh` and `mmvq.cu`: padded-aware MMVQ vecdot/kernel launch.
2. Guard rails:
   - env gate `GGML_CUDA_Q3K_PADDED_STORAGE=1`;
   - default behavior unchanged;
   - no speed benchmark until Q3_K `MUL_MAT` correctness passes;
   - real-server run is sanity only unless MMQ padded accessor is implemented.
3. Rollback path:
   - revert E201 code hunks if build/correctness fails.

## Benchmark Plan

- Baseline command: E200 current-tree Q3_K `MUL_MAT` smokes.
- Candidate command:

```powershell
$env:GGML_CUDA_Q3K_PADDED_STORAGE='1'
build-rocm-vec\bin\test-backend-ops.exe test -b ROCm0 -o MUL_MAT -p "type_a=q3_K,type_b=f32,m=16,n=1,k=256" --output csv
build-rocm-vec\bin\test-backend-ops.exe test -b ROCm0 -o MUL_MAT -p "type_a=q3_K,type_b=f32,m=1,n=64,k=256" --output csv
```

- Number of runs: correctness `r1`; no speed r3 in P1/P2a.
- Artifacts path: `build_logs/agent-workload/e201-rocm-q3k-padded-*`.

## Metrics

- build status
- Q3_K `MUL_MAT` correctness status
- route trace if available
- real-server text sanity only if correctness passes

## Result

- Outcome: pending.
- Delta: pending.
- Confidence: pending.
- Recommendation: pending.

## Notes

- This is deliberately not the final performance branch. It only proves or rejects the storage/copy/stride foundation.
