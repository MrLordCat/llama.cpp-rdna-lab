# E209 ROCm Q3_K Padded Storage Async Safety

## Metadata

- Experiment ID: E209
- Date: 2026-05-24
- Owner: Codex
- Branch/Commit: master after `18e531f24`
- Target lane: H43 Q3_K padded storage default-readiness on ROCm `build-rocm-vec`

## Hypothesis

- Statement: the E201-P2a opt-in speed route cannot become default-ready while backend async tensor APIs can raw-copy Q3_K padded storage.
- Mechanism: add the same host pack/unpack semantics to async set/get for padded Q3_K tensors, fail-closed for 2D partial async access, and reject async device-device copies involving padded Q3_K until a physical-layout aware copy is implemented.
- Why now: the user asked whether the latest speedup is default. It is committed but still opt-in; closing obvious correctness gaps is the right next step before any default-on discussion.

## Math / Theory

- Assumptions:
  - No TPS claim; this is correctness/safety work.
  - Existing measured speed remains tied to `GGML_CUDA_Q3K_PADDED_STORAGE=1 GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1`.
- Expected speedup corridor:
  - none directly.
- Failure conditions:
  - build fails;
  - Q3_K padded `MUL_MAT` correctness smoke regresses;
  - async path still allows raw 110-byte copies into 112-byte physical storage.

## Implementation Plan

1. Minimal code surface to change:
   - `ggml/src/ggml-cuda/ggml-cuda.cu` async backend tensor helpers.
2. Guard rails:
   - default behavior unchanged unless padded storage env is enabled;
   - unsupported partial 2D async access aborts like the sync buffer API;
   - async cpy returns `false` for padded tensors rather than copying raw bytes.
3. Rollback path:
   - revert the small async helper patch if build or smokes fail.

## Benchmark Plan

- Baseline command: existing E201 padded smokes.
- Candidate command: rebuild `build-rocm-vec`, then repeat Q3_K `MUL_MAT` smokes with and without padded env.
- Number of runs: correctness only.
- Artifacts path: `build_logs/agent-workload/e209-rocm-q3k-padded-async-safety-*`.

## Metrics

- build status
- `test-backend-ops` Q3_K `MUL_MAT` status
- no TPS claim

## Result

- Outcome: kept as default-off correctness/safety work.
- Delta:
  - `cmake --build build-rocm-vec --config Release -j 8` passed after the patch;
  - default raw Q3_K smoke passed: `MUL_MAT type_a=q3_K,type_b=f32,m=1,n=64,k=256`;
  - padded storage + MMQ smoke passed for the same prompt-like shape with `GGML_CUDA_Q3K_PADDED_STORAGE=1 GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1`;
  - padded storage + MMQ decode-like smoke passed: `MUL_MAT type_a=q3_K,type_b=f32,m=16,n=1,k=256`;
  - Q3_K `CPY` remains unsupported/fail-closed under padded storage in `test-backend-ops`, matching the new async device-device copy guard.
- Confidence: medium/high for the narrow safety gap. This validates full raw host set/get packing semantics through the backend API smoke surface, but it does not cover split buffers, partial views, or MoE-specific tensor movement.
- Recommendation: keep the patch. H43 is still not default-ready, but async raw-copy corruption is no longer an obvious blocker for the opt-in route.

## Notes

- This does not make H43 default. It only removes one correctness blocker.
- Verification commands:
  - `python -m py_compile gui\llama_gui.py gui\build_manager.py gui\dependency_checker.py gui\hardware_detector.py`
  - `git diff --check`
