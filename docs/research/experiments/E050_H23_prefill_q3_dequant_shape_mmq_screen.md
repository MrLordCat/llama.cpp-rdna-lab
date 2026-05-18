# E050 H23 Prefill Q3 Dequant-Shape MMQ Screen

## Metadata

- Experiment ID: E050
- Date: 2026-05-17
- Owner: Codex
- Branch/Commit: local `master`
- Target lane: Qwen3.6-27B-Q3_K_S cold-first prefill lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on.

## Hypothesis

- Statement: the Q3_K `row_diff=6144, ne10=5120, ncols=2048` prefill shape may be a narrow exception where MMQ beats the current cuBLAS path.
- Mechanism: E049 split timing shows this shape is dequant-dominated on the cuBLAS route: total `1839.27 ms`, `src0 dequant 1438.91 ms` (`78.23%`), `src1 63.97 ms`, `GEMM 336.39 ms`. MMQ avoids fp16 weight dequant but uses a different quantized matmul kernel.
- Why now: broad `GGML_CUDA_FORCE_MMQ_RUNTIME=1` was already negative, but E049 shows a specific shape with a different cost structure than GEMM-dominant FFN/QKV shapes.

## Math / Theory

- Assumptions:
  - E049 traced total for all large cuBLAS calls: `13280.90 ms`.
  - Target shape share inside that trace: `1839.27 / 13280.90 = 13.85%`.
  - Current full-lane prompt/wall share proxy from E045/E049: about `64.81%`.
  - Effective wall share for this shape is about `8.98%`.
- Selection gate:
  - A `20%` local improvement in this shape projects roughly `+1.8%` wall, enough to justify a guarded code route.
  - If forced-MMQ timing for the shape is worse than cuBLAS split timing, reject without implementing a shape route.
- Failure conditions:
  - MMQ saves dequant but loses more in quantized matmul compute or src1 quantization.
  - Broad forced-MMQ trace overhead makes absolute timing noisy; compare only large same-shape local timings.
  - A narrow route may still regress cold wall due launch/allocator effects.

## Implementation Plan

1. First run existing `GGML_CUDA_FORCE_MMQ_RUNTIME=1` with `GGML_TRACE_MMQ_TIMING=1` and sync timing.
2. Parse the target shape `type=11`, `nrows_x=6144`, `ncols_max=2048`.
3. Only if local MMQ timing beats E049 target cuBLAS timing, implement an env-gated shape-only route.
4. If local timing fails, keep only the diagnostic trace and reject.

## Benchmark Plan

- Analytical gate:
  - `python scripts/research/formula_sanity_checks.py`
  - `python scripts/research/speedup_model.py --baseline-tps 11.6534 --prefill-share 0.0898 --flash-prefill-speedup 1.20 --draft-len 1 --accept-rate 0 --spec-overhead 0 --decode-kernel-speedup 1.0`
  - `python scripts/research/required_acceptance.py --target-wall 1.01 --draft-len 1 --prefill-share 0.0898 --prefill-speedup 1.20 --decode-kernel-speedup 1.0 --spec-overhead 0.0`
- Trace command:
  - `python scripts/agent_workload_bench.py --label prefill-e050-broad-mmq-timing-r1 ... --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --real-context-mode repo-snapshot --no-reuse --no-v2-prime-pass --no-disable-thinking --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0"` with `GGML_CUDA_FORCE_MMQ_RUNTIME=1`, `GGML_TRACE_MMQ_TIMING=1`, and `GGML_TRACE_MMQ_TIMING_SYNC=1`.
- Number of runs:
  - one diagnostic timing trace before any route code.
- Artifacts path:
  - `build_logs/agent-workload/prefill-e050-broad-mmq-timing-r1.csv`
  - `build_logs/agent-workload/prefill-e050-broad-mmq-timing-r1.server.log`

## Metrics

- target-shape MMQ total timing
- target-shape MMQ selected `mmq_x`
- target-shape timing vs E049 cuBLAS split timing
- aggregate TPS only as a sanity signal, not as a speed claim while broad forced-MMQ is enabled

## Result

- Outcome: negative.
- Trace:
  - `prefill-e050-broad-mmq-timing-r1 = 10.05 TPS`; broad forced-MMQ remains a sanity-negative route and is not a speed candidate.
  - Target MMQ shape `type=11, nrows_x=6144, ncols_max=2048`: `2529.35 ms`, `288` calls, avg `8.78 ms`, `mmq_x=128`, `mmq_y=64`.
  - E049 current cuBLAS split timing for the same shape: `1839.27 ms`, avg `6.39 ms`.
- Delta: target MMQ is `+690.08 ms` slower, about `+37.52%` local regression.
- Analytical gate:
  - `formula_sanity_checks.py`: OK.
  - `speedup_model.py`: a `20%` local improvement at estimated wall share `0.0898` projects `1.0152x`, `11.8305 TPS`.
  - `required_acceptance.py`: non-spec placeholder check is feasible for `1.010x`; acceptance output is not meaningful for this route screen.
- Confidence: high enough to reject without implementing a shape-specific route.
- Recommendation: do not add a Q3_K `6144x5120` large-prefill MMQ override; the quantized kernel loses more than the saved dequant.

## Notes

- Surprises: even the most dequant-heavy Q3_K shape is still slower on MMQ at `ncols=2048`.
- Follow-up action: shift away from MMQ route overrides for large prefill; next dequant-oriented work should consider reducing repeated dequant or improving the Q3_K->fp16 conversion itself.
