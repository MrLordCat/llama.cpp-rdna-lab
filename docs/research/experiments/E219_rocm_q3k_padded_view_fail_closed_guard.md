# E219 ROCm Q3_K padded view fail-closed guard

## Metadata

- Experiment ID: E219
- Date: 2026-05-24
- Owner: Copilot
- Hypothesis ID: H43
- Target lane: ROCm Q3_K padded-storage default-readiness

## Hypothesis

- Statement: while backend-private Q3_K padded storage is opt-in and not default-ready, Q3_K tensor views into padded owner buffers must fail closed to avoid silent raw-vs-padded offset corruption.
- Mechanism: detect Q3_K views whose owner is physically allocated in padded layout and abort explicitly instead of silently falling back to raw-layout assumptions.
- Why now: E216 closed partial set/get for block-aligned slices, but view semantics remained an open correctness surface in H43.

## Math / Theory

- Assumptions:
  - raw Q3_K block size is 110 bytes, padded block size is 112 bytes;
  - Q3_K view metadata and pointer arithmetic are still raw-layout based;
  - if owner storage is physically padded, treating a Q3_K view as raw risks silent misaddressing.
- Expected speedup corridor:
  - no TPS claim; safety/default-readiness only.
- Failure conditions:
  - build failure;
  - canonical padded Q3_K `MUL_MAT` smokes regress.

## Implementation Plan

1. Minimal code surface:
   - `ggml/src/ggml-cuda/ggml-cuda.cu`
2. Guard rails:
   - keep padded storage default-off and env-gated;
   - abort only for Q3_K view into physically padded owner storage;
   - non-view Q3_K padded route behavior unchanged.
3. Rollback path:
   - revert fail-closed guard if canonical smokes fail.

## Benchmark Plan

- Build: `cmake --build build-rocm-vec --target llama-server test-backend-ops -j 8`
- Correctness under opt-in env:
  - prompt-like smoke: `MUL_MAT type_a=q3_K,type_b=f32,m=1,n=64,k=256`
  - decode-like smoke: `MUL_MAT type_a=q3_K,type_b=f32,m=16,n=1,k=256`
  - broad smoke: `MUL_MAT,MUL_MAT_ID` with `type_a=q3_K|type=q3_K|type_src=q3_K|type_dst=q3_K`

## Result

- Outcome: keep.
- Delta: no speed claim. This is a correctness hardening step.
- Confidence: medium-high for intended safety scope.
- Recommendation: keep the fail-closed guard while H43 remains default-off; only revisit after explicit view-safe padded offset policy exists.

## Measured Data

Build:

- `cmake --build build-rocm-vec --target llama-server test-backend-ops -j 8` -> `ninja: no work to do` after incremental rebuild.

Correctness:

| Check | Result |
| --- | --- |
| prompt-like Q3_K `MUL_MAT`, `m=1,n=64,k=256` | passed |
| decode-like Q3_K `MUL_MAT`, `m=16,n=1,k=256` | passed |
| broad Q3_K smoke (`MUL_MAT,MUL_MAT_ID`) | `13/13` passed on ROCm0 |

Artifacts:

- `build_logs/agent-workload/e219-rocm-q3k-padded-viewguard-prompt-smoke.txt`
- `build_logs/agent-workload/e219-rocm-q3k-padded-viewguard-decode-smoke.txt`
- `build_logs/agent-workload/e219-rocm-q3k-padded-viewguard-broad-smoke.txt`

## Notes

- This patch intentionally does not make padded storage default.
- The goal is fail-fast correctness: avoid silent corruption risk on unsupported Q3_K view semantics when owner storage is physically padded.
