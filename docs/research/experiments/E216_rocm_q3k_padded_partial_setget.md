# E216 ROCm Q3_K Padded Partial Set/Get

## Metadata

- Experiment ID: E216
- Date: 2026-05-24
- Owner: Codex
- Branch/Commit: master after `552d4d0f4`
- Target lane: H43 ROCm Q3_K padded-storage default-readiness

## Hypothesis

- Statement: H43 padded Q3_K storage should support block-aligned partial set/get paths instead of aborting on every 2D or offset copy.
- Mechanism: map raw `block_q3_K` byte offsets to physical `block_q3_K_padded` byte offsets, pack/unpack the requested block-aligned slices, and keep unsupported unaligned slices fail-fast.
- Why now: E209/E210/E213 made the opt-in route safer, but partial tensor movement remains an obvious default-readiness gap.

## Math / Theory

- Assumptions:
  - host/API offsets are in raw `sizeof(block_q3_K) == 110` byte units;
  - physical ROCm padded storage uses `sizeof(block_q3_K_padded) == 112` byte units;
  - block-aligned slices can be mapped by `padded_offset = raw_block_index * 112`.
- Expected speedup corridor:
  - no direct TPS change; this is safety/default-readiness work.
- Failure conditions:
  - build fails;
  - padded `MUL_MAT` smokes regress;
  - broad Q3_K smoke exposes a supported-op correctness error.

## Implementation Plan

1. Minimal code surface to change:
   - `ggml/src/ggml-cuda/ggml-cuda.cu` padded Q3_K set/get helpers.
2. Guard rails:
   - default behavior remains env-gated;
   - unaligned partial requests still assert rather than silently corrupt;
   - split buffers remain raw-layout guarded.
3. Rollback path:
   - revert helper changes if build or Q3_K smokes fail.

## Benchmark Plan

- Build: `cmake --build build-rocm-vec --target llama-server test-backend-ops -j`
- Correctness: Q3_K `MUL_MAT` prompt-like/decode-like smokes and broad Q3_K smoke under `GGML_CUDA_Q3K_PADDED_STORAGE=1 GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1`.
- Number of runs: correctness only.
- Artifacts path: `build_logs/agent-workload/e216-rocm-q3k-padded-partial-*`.

## Metrics

- build status
- supported Q3_K smoke errors
- no TPS claim

## Result

- Outcome: keep as default-off safety work. Padded Q3_K storage now supports block-aligned partial set/get and 2D set/get through host pack/unpack mapping instead of aborting on every partial path.
- Delta: no speed claim. Build passed, prompt-like and decode-like padded Q3_K `MUL_MAT` smokes passed, and broad Q3_K smoke preserved the previous support surface: `total=52`, `supported=13`, `unsupported=39`, `supported_errors=0`.
- Confidence: medium for the intended non-split, non-view storage slice. The implementation keeps unaligned slices fail-fast and leaves split buffers raw-layout guarded.
- Recommendation: keep this patch as another H43 default-readiness step, but do not make padded storage default yet. Q3_K views and real split-buffer padded storage remain open correctness surfaces.

## Measured Data

Build:

- `cmake --build build-rocm-vec --target llama-server test-backend-ops -j` passed.

Correctness:

| Check | Result |
| --- | --- |
| prompt-like Q3_K `MUL_MAT`, `m=1,n=64,k=256` | supported `1`, no error |
| decode-like Q3_K `MUL_MAT`, `m=16,n=1,k=256` | supported `1`, no error |
| broad Q3_K smoke | `total=52`, `supported=13`, `unsupported=39`, `supported_errors=0` |

Artifacts:

- `build_logs/agent-workload/e216-rocm-q3k-padded-partial-prompt-smoke.csv`
- `build_logs/agent-workload/e216-rocm-q3k-padded-partial-decode-smoke.csv`
- `build_logs/agent-workload/e216-rocm-q3k-padded-partial-broad-smoke.csv`

## Notes

- Surprises: the broad smoke must use the full Q3_K regex (`type_a`, `type`, `type_src`, `type_dst`) to match the prior 52-case coverage; `type_a=q3_K` alone only sees the supported `MUL_MAT` / `MUL_MAT_ID` rows.
- Follow-up action: the next H43 default-readiness slice should address view semantics or decide that padded storage remains opt-in until a view-safe policy exists. This patch intentionally does not change default behavior.
