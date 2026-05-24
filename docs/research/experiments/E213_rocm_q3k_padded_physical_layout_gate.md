# E213 ROCm Q3_K Padded Physical Layout Gate

## Metadata

- Experiment ID: E213
- Date: 2026-05-24
- Owner: Codex
- Branch/Commit: master after `7f11047cd`
- Hypothesis ID: H43 ROCm Q3_K padded-storage default-readiness
- Target lane: correctness/safety slice for ROCm `build-rocm-vec`, Q3_K padded storage

## Hypothesis

- Statement: Q3_K padded MMQ/MMVQ accessors should be gated by the tensor's actual backend storage layout, not only by `GGML_CUDA_Q3K_PADDED_STORAGE*` environment variables.
- Mechanism: compute whether the source tensor is physically allocated with the padded Q3_K size and pass that physical-layout boolean into MMQ/MMVQ dispatch. Split/raw shards must remain raw-layout even when padded env is enabled.
- Why now: E212 revalidated a real opt-in speed signal, but the route is not default-ready while local MMQ/MMVQ predicates can still treat raw split/compute buffers as padded based on env alone.

## Math / Theory

- Assumptions:
  - a physical padded Q3_K tensor has `sizeof(block_q3_K_padded) == 112` bytes per block plus the same row padding rule;
  - a raw/split Q3_K tensor still uses `sizeof(block_q3_K) == 110` byte blocks;
  - checking `ggml_backend_buffer_get_alloc_size(buffer, tensor)` against the expected padded size is a cheap runtime proof of the layout contract.
- Expected speedup corridor:
  - no direct TPS claim. This should be neutral for the existing single-GPU opt-in route and safer for split/default-readiness.
- Failure conditions:
  - build failure;
  - Q3_K padded `MUL_MAT` smokes fail;
  - padded route trace no longer reports `q3k_padded=1` for the known non-split opt-in path.

## Benchmark Plan

- Build: `cmake --build build-rocm-vec --target llama-server test-backend-ops -j`
- Correctness:
  - padded prompt-like Q3_K `MUL_MAT` smoke
  - padded decode-like Q3_K `MUL_MAT` smoke
  - broad Q3_K smoke if the narrow checks pass
- Route proof:
  - one small trace with `GGML_CUDA_Q3K_PADDED_STORAGE=1 GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1` should still show `q3k_padded=1` on the intended non-split route.

## Result

- Outcome: keep. MMQ/MMVQ padded accessors are now gated by actual padded allocation size instead of env-only predicates.
- Delta: no speed claim. This is a correctness/default-readiness guard. Post-patch opt-in regression r1 measured `11.9068 TPS`, prompt `1213.25 tok/s`, decode `30.55 tok/s`, errors `0`, with normal `Thinking Process` output; this is inside the E212 opt-in r1/r3 corridor and not promoted as a new gain.
- Confidence: medium-high for the single-GPU non-split slice. Build passed, narrow Q3_K smokes passed, broad Q3_K smoke matched previous support coverage, and MMQ route proof still reports `q3k_padded=1`.
- Recommendation: keep this safety patch. It reduces the chance that future default/policy work accidentally runs padded kernels on raw split/compute shards. It does not make H43 default-ready by itself.

## Measured Data

Build:

- `cmake --build build-rocm-vec --target llama-server test-backend-ops -j` passed.

Correctness:

| Check | Result |
| --- | --- |
| prompt-like Q3_K `MUL_MAT`, `m=1,n=64,k=256` | supported `1`, no error |
| decode-like Q3_K `MUL_MAT`, `m=16,n=1,k=256` | supported `1`, no error |
| broad `q3_K` smoke | `total=52`, `supported=13`, `unsupported=39`, `supported_errors=0` |
| MMQ route proof | `q3k_padded=1` for non-split padded route |

Regression run:

| Metric | Value |
| --- | ---: |
| aggregate completion TPS | `11.9068` |
| prompt eval TPS | `1213.25` |
| decode eval TPS | `30.55` |
| prompt eval ms | `6117.44` |
| decode eval ms | `3928.60` |
| errors | `0` |

Artifacts:

- `build_logs/agent-workload/e213-rocm-q3k-padded-physical-mmq-route-proof.txt`
- `build_logs/agent-workload/e213-rocm-q3k-padded-physical-broad-smoke.csv`
- `build_logs/agent-workload/e213-rocm-q3k-padded-physical-regression-r1.diagnostics.md`
- `build_logs/agent-workload/e213-rocm-q3k-padded-physical-regression-r1.jsonl`

## Notes

- This is a safety/default-readiness step, not a new route body. It should not be counted as a speed breakthrough unless followed by a measured default policy change.
- The key workflow correction is that env flags are not enough to prove layout. A tensor may be raw because it is split, a view, a compute buffer, or otherwise not allocated through the padded non-split path; kernels now receive the physical-layout fact from the caller.
