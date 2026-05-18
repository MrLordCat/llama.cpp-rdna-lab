# E049 H22 Prefill cuBLAS Split Timing

## Metadata

- Experiment ID: E049
- Date: 2026-05-17
- Owner: Codex
- Branch/Commit: local `master`
- Target lane: Qwen3.6-27B-Q3_K_S cold-first prefill lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on.

## Hypothesis

- Statement: after E046 and E048 rejected broad GEMM route knobs, the next promising prefill work should be selected by measured `cublas_backend` stage shares.
- Mechanism: the current large-prefill path dequantizes Q3_K/Q4_K weights into fp16, converts activations to fp16, runs cuBLAS/hipBLAS GEMM, and sometimes converts output. If dequant dominates, a fused or cached-weight idea has a different ceiling than a GEMM selector; if GEMM dominates, tile/library route work is more plausible.
- Why now: E045 trace shows `MUL_MAT` is `64.81%` of prompt trace and mostly Q3_K, while E046 compute16 and E048 hipBLASLt were not useful.

## Math / Theory

- Assumptions:
  - Current confirmed baseline: `prefill-current-ub2048-base-r3 = 11.6534 TPS`, prompt eval `1197.5567 tok/s`.
  - E045 trace prompt total: `13969.454 ms`.
  - E045 `MUL_MAT`: `9053.320 ms` (`64.81%` of prompt trace).
  - Q3_K share inside `MUL_MAT`: `84.32%`, about `54.65%` of prompt trace.
- Selection gate:
  - If one split stage is `>=20%` of the large `cublas_backend` local time, a `10%` local optimization can plausibly reach about `1%` wall in the full lane.
  - If all non-GEMM stages are small, avoid dequant/cache code and focus only on GEMM route/kernel ideas.
- Failure conditions:
  - Stream synchronizations distort absolute timing.
  - Trace spam captures decode/init calls instead of prompt calls unless filtered by `src1_ncols`.
  - Diagnostic timing must not be used as a speed claim.

## Implementation Plan

1. Add env-gated instrumentation in `ggml_cuda_op_mul_mat_cublas`.
2. Gate by `GGML_TRACE_CUBLAS_SPLIT_TIMING=1`; default runtime must be unchanged.
3. Add `GGML_TRACE_CUBLAS_SPLIT_TIMING_MIN_NCOLS` so prompt runs can filter out decode calls.
4. Build ROCm server, run one trace lane, parse stage shares, then either choose a code candidate or close the avenue.

## Benchmark Plan

- Analytical gate:
  - `python scripts/research/formula_sanity_checks.py`
  - `python scripts/research/speedup_model.py --baseline-tps 11.6534 --prefill-share 0.6481 --flash-prefill-speedup 1.02 --draft-len 1 --accept-rate 0 --spec-overhead 0 --decode-kernel-speedup 1.0`
  - `python scripts/research/required_acceptance.py --target-wall 1.01 --draft-len 1 --prefill-share 0.6481 --prefill-speedup 1.02 --decode-kernel-speedup 1.0 --spec-overhead 0.0`
- Trace command:
  - `python scripts/agent_workload_bench.py --label prefill-e049-cublas-split-trace-r1 ... --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --real-context-mode repo-snapshot --no-reuse --no-v2-prime-pass --no-disable-thinking --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0"` with `GGML_TRACE_CUBLAS_SPLIT_TIMING=1` and `GGML_TRACE_CUBLAS_SPLIT_TIMING_MIN_NCOLS=1024`.
- Number of runs:
  - one diagnostic trace; no speed claim.
- Artifacts path:
  - `build_logs/agent-workload/prefill-e049-cublas-split-trace-r1.csv`
  - `build_logs/agent-workload/prefill-e049-cublas-split-trace-r1.server.log`

## Metrics

- src0 conversion/dequant ms
- src1 conversion ms
- GEMM ms
- output conversion ms
- shape/qtype/path distribution for `src1_ncols >= 1024`

## Result

- Outcome: diagnostic success.
- Delta: diagnostic-only
- Analytical gate:
  - `formula_sanity_checks.py`: OK.
  - `speedup_model.py`: a `2%` local prefill improvement at `64.81%` prefill share projects `1.0129x` wall, `11.8034 TPS`.
  - `required_acceptance.py`: non-spec placeholder check is feasible for `1.010x`; acceptance output is not meaningful for this diagnostic, but confirms the script path.
- Trace:
  - `prefill-e049-cublas-split-trace-r1 = 11.01 TPS`; this is sync-instrumented and not a speed claim.
  - Large traced calls: `4456` timing rows, total `13280.90 ms`.
  - All traced calls: `src0 26.93%`, `src1 6.08%`, `GEMM 66.99%`.
  - Q3_K traced calls: total `10589.99 ms`, `src0 32.29%`, `src1 6.74%`, `GEMM 60.97%`.
  - Q3_K without the first one-time `516 ms` GEMM outlier: `src0 33.90%`, `src1 7.08%`, `GEMM 59.02%`.
  - Dequant-heavy shape: Q3_K `row_diff=6144, ne10=5120, ncols=2048` total `1839.27 ms`, `src0 1438.91 ms` (`78.23%`), `src1 63.97 ms`, `GEMM 336.39 ms`.
- Default-off sanity:
  - `prefill-e049-posttrace-default-r1 = 11.92 TPS` with all trace envs unset; use only as a no-regression smoke, not a new baseline.
- Confidence: high for hotspot selection; low for absolute timing because the instrumentation synchronizes the stream after each stage.
- Recommendation: do not pursue generic GEMM-route toggles next; screen dequant-dominated shape routes first, but require local timing before code.

## Notes

- Surprises: the target `6144x5120 @ ncols=2048` Q3_K shape is dominated by repeated `src0` dequant, unlike the larger FFN/QKV shapes where GEMM dominates.
- Follow-up action: E050 checks whether MMQ can beat that shape locally before any shape-specific route override.
