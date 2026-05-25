# E239 ROCm Q3_K Padded Dequant Half2 Gate

## Metadata

- Experiment ID: E239
- Date: 2026-05-25
- Owner: Codex
- Branch/Commit: master after `0f7c97007`
- Target lane: H42 ROCm large-Q3_K prefill route, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, full offload, cold/no reuse

## Hypothesis

- Statement: the Q3_K padded-storage dequant staging kernel may be wasting store/cast bandwidth by writing four scalar half values per thread.
- Mechanism: a half2-specialized `Q3_K padded -> fp16` kernel can write two half values per store pair while preserving the same arithmetic and layout.
- Why now: E228/E217 show Q3_K is still the cold-first bottleneck, but GEMM-side dominates. This is a cheap body-level staging gate with a clear reject rule before any wall A/B.

## Math / Theory

- Assumptions:
  - E228 robust Q3_K split: total `3783.195 ms`, `src0_convert_ms=508.206 ms`, `gemm_ms=2906.403 ms`.
  - E217 top bucket `(17408,5120,2048)` has `src0_ms=184.753 ms` and GEMM `1147.612 ms`.
  - This route cannot create a +20% wall win alone; it is only worth keeping if it is a clean, low-risk staging reduction that can stack with later body work.
- Expected speedup corridor:
  - Candidate must reduce robust `src0_ms` by at least `5%` on the dominant Q3_K buckets without moving `gemm_ms` or wall negatively.
  - If point timing is flat, do not run wall A/B.
- Failure conditions:
  - compiler already emits equivalent vector stores;
  - half2 store path increases register pressure or instruction count;
  - any rounding/layout difference fails backend correctness smoke.

## Implementation Plan

1. Minimal code surface to change:
   - add an env-gated half2 specialized padded Q3_K dequant kernel in `ggml/src/ggml-cuda/convert.cu`.
2. Guard rails:
   - default off via `GGML_EXPERIMENTAL_Q3K_PADDED_DEQUANT_HALF2=1`;
   - point-level cublas split timing before wall A/B;
   - backend correctness smoke if point timing moves.
3. Rollback path:
   - revert the env-gated kernel if point timing is flat/regressive.

## Benchmark Plan

- Baseline command:
  - current `build-rocm-vec/bin/llama-server.exe`, no env candidate, one trace run.
- Candidate command:
  - same run with `GGML_EXPERIMENTAL_Q3K_PADDED_DEQUANT_HALF2=1`.
- Shared trace env:
  - `GGML_TRACE_CUBLAS_SPLIT_TIMING=1`
  - `GGML_TRACE_CUBLAS_SPLIT_DETAIL=1`
  - `GGML_TRACE_CUBLAS_SPLIT_TIMING_PRE_SYNC=1`
  - `GGML_TRACE_CUBLAS_SPLIT_TIMING_MIN_NCOLS=1024`
  - `GGML_TRACE_CUBLAS_Q3K_ROUTE=1`
  - `GGML_TRACE_CUBLAS_Q3K_ROUTE_MIN_NCOLS=1024`
- Number of runs:
  - one run for point gate; wall only if robust `src0_ms` moves.
- Artifacts path:
  - `build_logs/agent-workload/e239-*`.

## Metrics

- robust Q3_K split total and `src0_ms`
- dominant bucket `src0_ms` for `(17408,5120,2048)`, `(5120,17408,2048)`, `(17408,5120,1345)`
- prompt/decode context only; no wall speed claim from trace-heavy runs

## Result

- Outcome: rejected and runtime probe reverted.
- Delta:
  - Correctness smoke with `GGML_EXPERIMENTAL_Q3K_PADDED_DEQUANT_HALF2=1` passed Q3_K `MUL_MAT` on ROCm0 (`11/11` supported rows, errors empty).
  - Build gate passed for `build-rocm-vec`.
  - Robust cublas split, all Q3_K rows:
    - control: total `3785.823 ms`, `src0_ms=517.104`, `src1_ms=364.091`, `gemm_ms=2904.598`.
    - candidate: total `3776.429 ms`, `src0_ms=495.805`, `src1_ms=375.452`, `gemm_ms=2905.105`.
    - `src0_ms` improved `-4.12%`, below the `5%` promotion gate, while `src1_ms` rose by `+11.361 ms`.
  - Dominant bucket `(17408,5120,2048)`:
    - control: total `1398.47 ms`, `src0_ms=183.82`, `src1_ms=84.50`, `gemm_ms=1130.15`.
    - candidate: total `1393.85 ms`, `src0_ms=175.31`, `src1_ms=88.71`, `gemm_ms=1129.81`.
  - Trace-context prompt eval tied: control `1082.69 tok/s`, candidate `1082.28 tok/s`.
- Confidence: medium-high. The local staging move is real but too small, and it does not move the trace-context wall enough to justify keeping code.
- Recommendation: do not keep the half2 padded-dequant kernel. A staging-only Q3_K dequant rewrite is below the current cold-first ceiling unless paired with a stronger GEMM/body change.

## Notes

- This is intentionally not a GEMM-side fix. It is a small staging gate with a hard stop if point timing does not move.
- Why it did not advance: the kernel shaved scalar half stores/casts, but current Q3_K is already GEMM-dominated. The small `src0_ms` win was mostly absorbed by tiny `src1_ms`/noise shifts and left prompt timing unchanged.
