# E215 ROCm Q3_K cuBLAS Src1 Reuse Gate

## Metadata

- Experiment ID: E215
- Date: 2026-05-24
- Owner: Codex
- Branch/Commit: master after `f542fbc63`
- Target lane: H35/H42 ROCm Q3_K cuBLAS split, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: large-Q3_K prefill may still waste some time converting the same `src1` activation from f32 to f16 for adjacent cuBLAS matmuls, especially attention projection groups and FFN gate/up pairs.
- Mechanism: add a default-off trace keyed by `src1` tensor, device pointer, width, and prompt chunk size to measure repeated conversion locality and upper-bound a transient f16 activation reuse route.
- Why now: E214 rejects current-MMQ retile as a large-prefill replacement. E192 shows Q3_K cuBLAS split still has `src1_ms=364.309 ms`; this is smaller than src0 staging/GEMM but might expose a narrow, low-memory prefill-side route if repeats are immediate.

## Math / Theory

- Assumptions:
  - E192 Q3_K cuBLAS split: `src0_convert_ms=1637.070`, `src1_ms=364.309`, `gemm_ms=3203.883`.
  - A perfect `src1` conversion reuse can only remove repeated `src1_ms`, not GEMM or src0 staging.
  - A useful code prototype needs tight gaps; long-gap reuse would become a graph-lifetime activation cache and risk stale data or memory pressure.
- Expected speedup corridor:
  - If repeated `src1` conversions cluster with `gap<=2` and cover `>100 ms` in the trace, an env-gated transient cache prototype may be worth point-testing.
  - If repeats are sparse, long-gap, or below roughly `1%` wall ceiling, reject without code.
- Failure conditions:
  - build fails;
  - trace cannot identify repeated keys;
  - repeat locality is too weak to justify a cache prototype.

## Implementation Plan

1. Minimal code surface to change:
   - add default-off `GGML_TRACE_CUBLAS_SRC1_REUSE` instrumentation to `ggml/src/ggml-cuda/ggml-cuda.cu`.
2. Guard rails:
   - instrumentation only, no route behavior change;
   - keep existing split timing as the timing source;
   - no speed claim from trace runs.
3. Rollback path:
   - keep if useful and default-off; revert if noisy or build-hostile.

## Benchmark Plan

- Baseline command: current split trace with `GGML_TRACE_CUBLAS_SPLIT_TIMING=1`, detail/pre-sync enabled.
- Candidate command: same trace plus `GGML_TRACE_CUBLAS_SRC1_REUSE=1`.
- Number of runs: one diagnostic trace.
- Artifacts path: `build_logs/agent-workload/e215-rocm-q3k-cublas-src1-reuse-*`.

## Metrics

- repeated `src1` key count
- gap distribution for repeated keys
- total repeated `src1_ms` and `gap<=2` repeated `src1_ms`
- route context only for aggregate TPS

## Result

- Outcome: reject cache prototype for now; keep default-off instrumentation.
- Delta:
  - build passed: `cmake --build build-rocm-vec --target llama-server -j`;
  - diagnostic trace completed with `errors=0`, aggregate trace-context `7.0269 TPS`, prompt `1076.00 tok/s`, decode `30.25 tok/s`;
  - `GGML_TRACE_CUBLAS_SRC1_REUSE` emitted `1396` Q3_K cuBLAS rows and `412` unique src1 keys;
  - total traced Q3_K `src1_ms` was `348.555 ms`;
  - repeated same-key rows were common (`984` calls, `235.972 ms`), but most are long prompt-chunk repeats with `gap=347..349`, which cannot safely reuse old activation data;
  - tight/immediate reuse was only `572` calls and `112.408 ms`, all `gap=1`.
- Confidence: high that a safe immediate-only src1 f16 cache has a low ceiling on this lane; medium that longer-gap repeats are chunk replays with changed activation data and should not be reused.
- Recommendation: do not implement a src1 f16 activation cache now. The best safe ceiling is about `112 ms` on a `~9.1 s` trace wall (`~1.2%`) before cache lookup, allocation, graph-capture, and stale-data safeguards. Continue H42/H43 with larger structural routes; keep this trace as default-off tooling.

## Measured Data

Artifacts:

- `build_logs/agent-workload/e215-rocm-q3k-cublas-src1-reuse-r1.server.log`
- `build_logs/agent-workload/e215-rocm-q3k-cublas-src1-reuse-r1.diagnostics.md`

Reuse summary:

| Metric | Value |
| --- | ---: |
| trace rows | `1396` |
| unique src1 keys | `412` |
| total src1 ms | `348.555 ms` |
| repeated calls | `984` |
| repeated src1 ms | `235.972 ms` |
| tight `gap<=2` calls | `572` |
| tight `gap<=2` src1 ms | `112.408 ms` |

Gap distribution:

| Gap | Calls | Src1 ms |
| ---: | ---: | ---: |
| `1` | `572` | `112.408 ms` |
| `347` | `32` | `6.560 ms` |
| `348` | `222` | `46.600 ms` |
| `349` | `158` | `70.400 ms` |

Interpretation:

- `gap=1` is the only safe reuse class for a transient same-step cache.
- `gap=347..349` matches later prompt chunks: the tensor address/key repeats, but the activation contents are different, so a cache would need chunk-aware invalidation and would no longer be a cheap immediate reuse route.
- The immediate ceiling is below the threshold for a risky cache prototype, especially after E198 already showed that small activation-cache wins can vanish under graph/runtime overhead.

## Notes

- Surprises:
  - The trace found many repeated keys, but the useful part is much smaller than the raw repeat count suggests.
- Follow-up action:
  - no src1 cache code;
  - keep `GGML_TRACE_CUBLAS_SRC1_REUSE` as diagnostic instrumentation;
  - return to H43 default-readiness or a genuinely new H42 body.
