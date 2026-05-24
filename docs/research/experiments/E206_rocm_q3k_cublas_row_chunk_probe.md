# E206 ROCm Q3_K cublas row-chunk topology probe

## Metadata

- Experiment ID: E206
- Date: 2026-05-24
- Owner: Codex
- Branch/Commit: master after E204/E205
- Target lane: H42 ROCm large-Q3_K cublas path on Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: a row-chunked Q3_K hipBLAS fallback may reduce hot-shape staging/residency pressure enough to expose a local point win, even though it is not the final direct kernel H42 wants.
- Mechanism: keep the proven hipBLAS GEMM route, but dequantize Q3_K weights into a smaller fp16 tile and call GEMM per row chunk instead of materializing the whole hot matrix at once.
- Why now: E204 rejected routing hot shapes to the current MMQ body. Before attempting a much larger direct kernel, this is the cheapest body/topology probe that changes the current cublas route itself.

## Math / Theory

- Assumptions:
  - E192 Q3_K split total was `5213.358 ms`, with `src0_convert_ms=1637.070`, `src1_ms=364.309`, `gemm_ms=3203.883`.
  - The dominant `17408x5120@2048` family is GEMM-heavy, so row chunking must not increase GEMM time enough to erase any staging benefit.
  - The `6144x5120@2048` bucket is staging-heavy, so it is the best local chance for this topology.
- Expected speedup corridor:
  - low confidence; require point-level `sum_ms` improvement before any wall run.
  - A `+2%` prefill-only speedup projects only about `1.0155x` wall at `prefill_share=0.78`, so tiny local wins are not enough.
- Failure conditions:
  - hot bucket `sum_ms` is unchanged or worse;
  - GEMM time rises after splitting rows;
  - route activation does not cover the intended hot shapes.

## Implementation Plan

1. Minimal code surface to change:
   - `ggml/src/ggml-cuda/ggml-cuda.cu`: temporary default-off row chunk env in the Q3_K fp16 cublas branch.
2. Guard rails:
   - env-gated only: `GGML_EXPERIMENTAL_Q3K_CUBLAS_ROW_CHUNK=1`;
   - default behavior unchanged;
   - first validation is point split timing, not wall TPS.
3. Rollback path:
   - if point timing is neutral/negative, revert the row-chunk code and keep only generally useful timing instrumentation.

## Benchmark Plan

- Baseline command:
  - split timing run on `build-rocm-vec/bin/llama-server.exe` with `GGML_TRACE_CUBLAS_SPLIT_TIMING=1`, `GGML_TRACE_CUBLAS_SPLIT_DETAIL=1`, `GGML_TRACE_CUBLAS_SPLIT_TIMING_PRE_SYNC=1`, `GGML_TRACE_CUBLAS_SPLIT_TIMING_MIN_NCOLS=1024`.
- Candidate command:
  - same trace with `GGML_EXPERIMENTAL_Q3K_CUBLAS_ROW_CHUNK=1`, `GGML_EXPERIMENTAL_Q3K_CUBLAS_ROW_CHUNK_ROWS=<rows>`, and exact shape filters.
- Number of runs:
  - point trace `r1`; wall `r1` only if hot point timing improves.
- Artifacts path:
  - `build_logs/agent-workload/e206-rocm-q3k-rowchunk-*`

## Metrics

- hot-shape `sum_ms`, `src0_convert_ms`, `gemm_ms`
- route activation count
- aggregate completion TPS only if point timing moves first

## Result

- Outcome: rejected; row-chunking worsened both point timing and wall timing.
- Delta:
  - Control trace aggregate: `11.36 TPS`.
  - `6144x5120@2048` control point: `count=288`, `src0=65.417 ms`, `src1=58.476 ms`, `gemm=329.563 ms`, `sum=453.422 ms`, `avg=1.574 ms`.
  - `6144x5120@2048` row-chunk candidate: aggregate `11.29 TPS`; point `count=288`, `src0=82.886 ms`, `src1=60.928 ms`, `gemm=404.314 ms`, `sum=548.155 ms`, `avg=1.903 ms`.
  - `6144` local delta: `+20.9%` slower by point sum.
  - `17408x5120@2048` control point: `count=756`, `src0=361.649 ms`, `src1=167.158 ms`, `gemm=2280.947 ms`, `sum=2809.796 ms`, `avg=3.717 ms`.
  - `17408x5120@2048` row-chunk candidate: aggregate `11.26 TPS`; point `count=756`, `src0=410.413 ms`, `src1=153.879 ms`, `gemm=2388.226 ms`, `sum=2952.568 ms`, `avg=3.906 ms`.
  - `17408` local delta: `+5.1%` slower by point sum.
- Confidence: high for rejection. Exact filters activated the intended shapes, event counts matched, and both the staging part and GEMM part moved the wrong way.
- Recommendation: do not tune row sizes. This topology adds per-chunk staging/GEMM overhead instead of relieving residency pressure. The temporary runtime row-chunk code was reverted; keep only `GGML_TRACE_CUBLAS_SPLIT_TIMING_PRE_SYNC=1` as default-off point-review instrumentation. Continue H42 only with a genuinely new fused/direct body, or move to H43 padded-storage correctness.

## Notes

- Analytical gate:
  - `python scripts/research/formula_sanity_checks.py` passed.
  - `python scripts/research/speedup_model.py --baseline-tps 7.6054 --prefill-share 0.78 --draft-len 1 --accept-rate 0 --spec-overhead 0 --flash-prefill-speedup 1.02 --decode-kernel-speedup 1.00` projected `1.0155x` wall for a `+2%` prefill-only gain.
  - `python scripts/research/required_acceptance.py --target-wall 1.02 --draft-len 1 --prefill-share 0.78 --prefill-speedup 1.02 --decode-kernel-speedup 1.00 --spec-overhead 0` showed the `1.02x` wall target is unreachable under only `1.02x` prefill assumptions.
- Follow-up action:
  - H42 cublas row splitting is closed.
  - Next useful prefill branch is either a true direct `Q3_K x F16` kernel/body or the H43 backend-private padded-storage correctness slice.

## Artifacts

- `build_logs/agent-workload/e206-rocm-q3k-rowchunk-control-r1.server.log`
- `build_logs/agent-workload/e206-rocm-q3k-rowchunk-control-r1.diagnostics.md`
- `build_logs/agent-workload/e206-rocm-q3k-rowchunk-6144-r1.server.log`
- `build_logs/agent-workload/e206-rocm-q3k-rowchunk-6144-r1.diagnostics.md`
- `build_logs/agent-workload/e206-rocm-q3k-rowchunk-17408-r1.server.log`
- `build_logs/agent-workload/e206-rocm-q3k-rowchunk-17408-r1.diagnostics.md`
