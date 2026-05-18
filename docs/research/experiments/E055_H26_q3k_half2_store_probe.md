# E055 H26 Q3_K Half2 Store Probe

## Metadata

- Experiment ID: E055
- Date: 2026-05-18
- Owner: Copilot
- Branch/Commit: local `master`
- Target lane: Qwen3.6-27B-Q3_K_S cold-first prefill lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on.

## Hypothesis

- Statement: a Q3_K fp16 conversion kernel that preserves the current 64-thread geometry but writes two `half2` pairs per thread can reduce Q3_K `src0_convert_ms` on the large cuBLAS path.
- Mechanism: E054 shows allocation is negligible and the current conversion kernel is the real target. E051 already rejected a 128-thread/two-value geometry, so this probe changes store/vectorization shape without changing the 64-thread occupancy profile.
- Why now: E053/E054 narrow the current post-C01 P1 target to `convert.cu`, specifically Q3_K fp16 conversion/store.

## Math / Theory

- Assumptions:
  - E054 Q3_K `src0_convert_ms`: `3370.32 ms`.
  - E054 target `6144x5120@ncols2048` conversion: `1430.88 ms`.
  - E053 full-wall estimate for Q3_K dequant-only work: about `10.1%` aggregate share after discounting decode time.
- Expected speedup corridor:
  - `10%` local conversion gain is likely too small for a keep decision.
  - `20%` local conversion gain projects roughly `+1.7%` aggregate.
  - `25%` local conversion gain projects roughly `+2.1%` aggregate and is the practical keep threshold.
- Failure conditions:
  - HIP compiler already coalesces scalar half stores.
  - `half2` packing increases arithmetic/register pressure.
  - Full-lane r1 is negative or split-detail does not show lower `src0_convert_ms`.

## Implementation Plan

1. Add a fp16-only Q3_K half2 conversion kernel in `ggml/src/ggml-cuda/convert.cu`.
2. Gate it with `GGML_CUDA_Q3K_DEQUANT_HALF2=1`.
3. Keep the existing 64-thread kernel as default.
4. Build ROCm `llama-server`.
5. Run r1 full-lane candidate. If it is not clearly negative, run split-detail trace to check local conversion.
6. Revert unless runtime and target trace support a keep/iterate decision.

## Benchmark Plan

- Candidate command:
  - `GGML_CUDA_Q3K_DEQUANT_HALF2=1 python scripts/agent_workload_bench.py --label prefill-e055-q3half2-r1 --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks quick --task-ids triage_diff,review_bug --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0" --real-context-mode repo-snapshot --no-reuse --background-server-policy fail --task-fail-timeout 0 --no-v2-prime-pass --no-disable-thinking --max-tokens 120`
- Optional split-detail trace:
  - same candidate env plus `GGML_TRACE_CUBLAS_SPLIT_TIMING=1 GGML_TRACE_CUBLAS_SPLIT_DETAIL=1 GGML_TRACE_CUBLAS_SPLIT_TIMING_MIN_NCOLS=1024`, label `prefill-e055-q3half2-detail-r1`.
- Number of runs:
  - `--runs 1`; r3 only if r1 and trace are promising.
- Artifacts path:
  - `build_logs/agent-workload/prefill-e055-q3half2-r1.*`
  - `build_logs/agent-workload/prefill-e055-q3half2-detail-r1.*`

## Metrics

- aggregate completion TPS
- prompt eval TPS
- Q3_K `src0_convert_ms`
- target `6144x5120@ncols2048` conversion time
- errors/correct server completion

## Result

- Outcome: rejected; prototype code reverted.
- Full-lane candidate: `prefill-e055-q3half2-r1 = 11.86 TPS` (r1, trace-off). This is only `+0.78%` vs E053 trace-off control `11.7681 TPS`, within the range where local evidence is required before keeping code.
- Split-detail trace: `prefill-e055-q3half2-detail-r1 = 11.23 TPS`, diagnostic-only.
- Local Q3_K split result:
  - E054 default Q3_K `src0_convert_ms = 3370.32 ms`; E055 half2 `3317.06 ms`, delta `-53.26 ms` (`-1.58%`).
  - E054 target `6144x5120@ncols2048` conversion `1430.88 ms`; E055 half2 `1425.06 ms`, delta `-5.82 ms` (`-0.41%`).
  - E055 Q3_K `src0_alloc_ms = 6.03 ms`, still negligible.
- Confidence: medium. The env-gated path compiled and ran, and split-detail shows the local effect is real but far below the E053 `>=25%` local keep gate.
- Recommendation: do not keep or repeat the simple half2 store variant. Future Q3_K conversion work needs a larger structural change than scalar-to-half2 store packing.

## Notes

- Surprises: aggregate r1 looked mildly positive, but split-detail showed only a `1.58%` Q3_K conversion reduction and a `0.41%` target-shape reduction. The apparent full-lane gain is not enough evidence to keep the code.
- Follow-up action: keep E054 split-detail instrumentation, keep `convert.cu` reverted to default, and choose a different Q3_K conversion/layout idea only if it plausibly clears the `>=25%` local gate.