# E054 H25 Q3_K cuBLAS Staging Detail

## Metadata

- Experiment ID: E054
- Date: 2026-05-18
- Owner: Copilot
- Branch/Commit: local `master`
- Target lane: Qwen3.6-27B-Q3_K_S cold-first prefill lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on.

## Hypothesis

- Statement: before coding a Q3_K dequant/layout candidate, split the measured `src0_ms` into temporary fp16 buffer allocation/staging time and actual Q3_K conversion/store time.
- Mechanism: E053 confirms Q3_K `src0` is still the largest actionable measured target, but the current cuBLAS split timing starts before `src0_as_f16.alloc(ne)` and stops after `to_fp16_cuda(...)`. If allocation or pool behavior is a meaningful part of `src0_ms`, the next branch should target staging/reuse or allocation layout rather than only `dequantize_block_q3_K`.
- Why now: E053 raises the dequant-only keep gate to about `>=25%` local improvement for `>=2%` aggregate TPS. A wrong target would burn a long ROCm build cycle for a low-ceiling kernel change.

## Math / Theory

- Assumptions:
  - E053 control: `11.7681 TPS`.
  - E053 Q3_K split: `src0 3386.36 / 10369.29 ms` (`32.66%` of Q3_K split time).
  - E053 full-wall estimate for Q3_K dequant-only work: about `10.1%` aggregate share after discounting decode time.
- Expected speedup corridor:
  - If `src0_convert_ms` is most of `src0_ms`, a dequant/layout candidate must plausibly improve that conversion by about `25%` locally.
  - If `src0_alloc_ms` is material, target temporary-buffer reuse, allocation granularity, or staging layout first.
- Failure conditions:
  - The detail trace is sync-instrumented and cannot be used as a TPS claim.
  - CPU-side allocation timing and GPU conversion timing are different clocks; use them only for stage selection.

## Implementation Plan

1. Extend the existing default-off cuBLAS split timing config with `GGML_TRACE_CUBLAS_SPLIT_DETAIL=1`.
2. Preserve existing `GGML_TRACE_CUBLAS_SPLIT_TIMING=1` output by default.
3. When detail is enabled, log `src0_alloc_ms`, `src0_convert_ms`, `src1_alloc_ms`, and `src1_convert_ms` for fp16 cuBLAS path rows.
4. Build `build-rocm-vec` server.
5. Run one diagnostic trace on the current lane and parse target Q3_K shapes.

## Benchmark Plan

- Detail trace command:
  - `GGML_TRACE_CUBLAS_SPLIT_TIMING=1 GGML_TRACE_CUBLAS_SPLIT_DETAIL=1 GGML_TRACE_CUBLAS_SPLIT_TIMING_MIN_NCOLS=1024 python scripts/agent_workload_bench.py --label prefill-e054-cublas-split-detail-r1 --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks quick --task-ids triage_diff,review_bug --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0" --real-context-mode repo-snapshot --no-reuse --background-server-policy fail --task-fail-timeout 0 --no-v2-prime-pass --no-disable-thinking --max-tokens 120`
- Number of runs:
  - `--runs 1`; diagnostic gate only.
- Artifacts path:
  - `build_logs/agent-workload/prefill-e054-cublas-split-detail-r1.*`

## Metrics

- Q3_K `src0_alloc_ms` vs `src0_convert_ms`
- target `6144x5120@ncols2048` alloc/convert split
- src1 allocation/conversion sanity
- next-candidate decision: allocation/staging vs dequant-layout vs reject

## Result

- Outcome: diagnostic success; Q3_K `src0` cost is conversion/store, not allocation.
- Trace:
  - `prefill-e054-cublas-split-detail-r1 = 11.1878 TPS`; sync trace only, not a speed claim.
  - Large split rows: `4456`, total `12967.72 ms`.
  - All traced calls: `src0 3528.57 ms` (`27.21%`), `GEMM 8634.89 ms` (`66.59%`).
  - Q3_K traced calls: rows `2792`, total `10291.61 ms`, `src0 3377.02 ms` (`32.81%`), `src1 711.46 ms`, `GEMM 6203.00 ms` (`60.27%`).
  - Q3_K `src0_alloc_ms`: `6.12 ms`, only `0.18%` of Q3_K `src0`.
  - Q3_K `src0_convert_ms`: `3370.32 ms`, `99.80%` of Q3_K `src0`.
  - Q3_K without the first one-time GEMM outlier: `src0 3370.54 ms`, `src0_alloc_ms 3.61 ms` (`0.11%` of src0), `src0_convert_ms 3366.35 ms` (`99.88%` of src0).
- Target shape:
  - Q3_K `row_diff=6144, ne00=5120, ne10=5120, ncols=2048`: rows `288`, total `1829.21 ms`.
  - `src0 1430.95 ms` (`78.23%` of target total).
  - `src0_alloc_ms 0.00 ms`; `src0_convert_ms 1430.88 ms` (`99.99%` of target `src0`).
  - Tail shapes for the same projection stay conversion-dominated: `ncols=1278` has `151.00 / 151.01 ms` convert/src0, and `ncols=1259` has `149.60 / 149.61 ms` convert/src0.
- Sanity:
  - Q4_K `src0` is also conversion-dominated (`151.45 / 151.55 ms` convert/src0), but its total share is much smaller than Q3_K.
  - `src1` allocation is negligible (`1.24 ms` alloc vs `709.59 ms` convert for Q3_K rows).
- Delta: no candidate delta; diagnostic-only.
- Confidence: high. The alloc contribution is below `1%` of `src0` for every major Q3_K shape except one first-use pool-growth outlier, and even that does not change the aggregate decision.
- Recommendation: do not pursue pool allocation/reuse as the next P1 branch. Proceed to a guarded Q3_K fp16 conversion/layout kernel candidate in `ggml/src/ggml-cuda/convert.cu`, preserving the 64-thread occupancy profile and targeting vectorized/coalesced half stores. Keep the `>=25%` local conversion-gain gate from E053.

## Notes

- Surprises: the dequant-heavy `z-*` shape has effectively zero allocation cost after first pool growth; the measured `78.23% src0` is real conversion/store time.
- Follow-up action: start E055 as a guarded Q3_K fp16 conversion-layout probe. Do not repeat the rejected 128-thread geometry from E051.