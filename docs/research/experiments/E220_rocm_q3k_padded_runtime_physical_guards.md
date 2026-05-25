# E220 ROCm Q3_K padded runtime physical-layout guards

## Metadata

- Experiment ID: E220
- Date: 2026-05-24
- Owner: Copilot
- Hypothesis ID: H43
- Target lane: ROCm Q3_K padded-storage default-readiness

## Hypothesis

- Statement: H43 still has correctness risk if runtime paths key off "padded enabled" rather than "physically padded allocation".
- Mechanism:
  - add runtime predicate split in CUDA backend code: physical padded allocation must be proven before padded set/get/cpy and dequant-side route decisions;
  - add MMQ-side fail-closed guard for Q3_K views into physically padded owner storage.
- Why now: E219 added fail-closed guard in one predicate path, but runtime checks were still mixed between allocation-intent and physical-layout checks.

## Math / Theory

- Assumptions:
  - padded Q3_K is a storage contract, not only an env toggle;
  - runtime copy/dequant/MMQ decisions should depend on physical tensor allocation, not intent-only predicates;
  - unsupported view semantics must fail closed, not silently fall back to raw addressing on padded owner storage.
- Expected speedup corridor:
  - no speed claim; correctness/default-readiness hardening.
- Failure conditions:
  - build failure;
  - canonical padded Q3_K `MUL_MAT`/`MUL_MAT_ID` smoke regressions;
  - broad smoke support surface changes unexpectedly.

## Implementation Plan

1. Minimal code surface:
   - `ggml/src/ggml-cuda/ggml-cuda.cu`
   - `ggml/src/ggml-cuda/mmq.cu`
2. Guard rails:
   - keep default policy unchanged (`GGML_CUDA_Q3K_PADDED_STORAGE=1` + `GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1` opt-in);
   - runtime decisions require physical padded allocation proof;
   - unsupported padded-owner Q3_K views abort explicitly.
3. Rollback path:
   - revert guards if build/smokes regress.

## Benchmark Plan

- Build:
  - `cmake --build build-rocm-vec --target llama-server test-backend-ops -j 8`
- Correctness smokes with padded env:
  - prompt-like: `MUL_MAT type_a=q3_K,type_b=f32,m=1,n=64,k=256`
  - decode-like: `MUL_MAT type_a=q3_K,type_b=f32,m=16,n=1,k=256`
  - broad: `MUL_MAT,MUL_MAT_ID` with `type_a=q3_K|type=q3_K|type_src=q3_K|type_dst=q3_K`
- Optional sanity A/B on active lane:
  - control label `e220-rocm12k-control-r1`
  - padded label `e220-rocm12k-padded-r1`

## Result

- Outcome: keep.
- Delta:
  - correctness: canonical prompt/decode smokes pass; broad Q3_K smoke remains `13/13` passed on ROCm0;
  - quick lane A/B: `7.22` (control) vs `7.18` (padded), effectively tie/noise in r1 context.
- Confidence: medium-high for hardening scope.
- Recommendation: keep E220 guards and continue H43 on remaining readiness surfaces. Do not treat this as a speed branch.

## Measured Data

Build:

- `cmake --build build-rocm-vec --target llama-server test-backend-ops -j 8` succeeded (`ninja: no work to do` after incremental rebuild).

Correctness:

| Check | Result |
| --- | --- |
| prompt-like Q3_K `MUL_MAT` smoke | pass |
| decode-like Q3_K `MUL_MAT` smoke | pass |
| broad Q3_K (`MUL_MAT,MUL_MAT_ID`) | `13/13` pass |

Quick A/B (active 12k lane, r1):

| Label | Aggregate completion TPS |
| --- | ---: |
| `e220-rocm12k-control-r1` | `7.22` |
| `e220-rocm12k-padded-r1` | `7.18` |

Artifacts:

- `build_logs/agent-workload/e220-rocm-q3k-padded-physical-prompt-smoke.txt`
- `build_logs/agent-workload/e220-rocm-q3k-padded-physical-decode-smoke.txt`
- `build_logs/agent-workload/e220-rocm-q3k-padded-physical-broad-smoke.txt`
- `build_logs/agent-workload/e220b-rocm-q3k-padded-viewguard-broad-smoke.txt`
- `build_logs/agent-workload/e220-rocm12k-control-r1.diagnostics.md`
- `build_logs/agent-workload/e220-rocm12k-padded-r1.diagnostics.md`

## Notes

- E220 is a safety/readiness hardening step, not a throughput optimization.
- The key change is consistency: runtime behavior now requires physical padded-layout proof, and both backend/MMQ view surfaces fail closed for unsupported padded-owner views.
