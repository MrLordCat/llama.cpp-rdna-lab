# E051 H24 Q3_K Dequant128 Probe

## Metadata

- Experiment ID: E051
- Date: 2026-05-17
- Owner: Codex
- Branch/Commit: local `master`
- Target lane: Qwen3.6-27B-Q3_K_S cold-first prefill lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on.

## Hypothesis

- Statement: a Q3_K fp16 dequant kernel with 128 threads/block and two output values per thread may reduce repeated Q3_K `src0` conversion time on RDNA4.
- Mechanism: the current `dequantize_block_q3_K` uses 64 threads/block and each thread writes four contiguous values. E049 shows Q3_K `src0` dequant is `3419.41 ms` of `13280.90 ms` traced large calls (`25.75%` of that trace), so a small local dequant gain has enough wall ceiling to test.
- Why now: route-level alternatives were rejected: compute16 (E046), hipBLASLt (E048), broad/shape MMQ (E050).

## Math / Theory

- Assumptions:
  - E049 traced large-call total: `13280.90 ms`.
  - Q3_K src0 dequant total: `3419.41 ms` (`25.75%` of traced large-call time).
  - Prompt/wall share proxy from E045/E049: `64.81%`.
  - Effective wall share for Q3_K dequant is about `16.69%`.
- Selection gate:
  - A `10%` local Q3_K dequant improvement projects about `+1.5%` wall.
  - A neutral or negative r1 full-lane result is enough to revert, because the code is experimental and shape-wide.
- Failure conditions:
  - More threads per block reduce occupancy or increase scheduler overhead.
  - The kernel becomes memory/store limited, so splitting work across more threads does not help.
  - Any correctness instability or response errors require immediate revert.

## Implementation Plan

1. Add `dequantize_block_q3_K_f16_128` for fp16 output only.
2. Gate it with `GGML_CUDA_Q3K_DEQUANT_128=1`; default keeps the existing 64-thread kernel.
3. Build ROCm server.
4. Run r1 candidate against the same cold-first prefill lane.
5. Revert code if r1 is not positive.

## Benchmark Plan

- Analytical gate:
  - `python scripts/research/formula_sanity_checks.py`
  - `python scripts/research/speedup_model.py --baseline-tps 11.6534 --prefill-share 0.1669 --flash-prefill-speedup 1.10 --draft-len 1 --accept-rate 0 --spec-overhead 0 --decode-kernel-speedup 1.0`
  - `python scripts/research/required_acceptance.py --target-wall 1.01 --draft-len 1 --prefill-share 0.1669 --prefill-speedup 1.10 --decode-kernel-speedup 1.0 --spec-overhead 0.0`
- Candidate command:
  - `python scripts/agent_workload_bench.py --label prefill-e051-q3dequant128-r1 ... --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --real-context-mode repo-snapshot --no-reuse --no-v2-prime-pass --no-disable-thinking --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0"` with `GGML_CUDA_Q3K_DEQUANT_128=1`.
- Number of runs:
  - r1 gate, r3 only if r1 is positive.
- Artifacts path:
  - `build_logs/agent-workload/prefill-e051-q3dequant128-r1.csv`
  - `build_logs/agent-workload/prefill-e051-q3dequant128-r1.server.log`

## Metrics

- aggregate completion TPS
- prompt eval TPS
- benchmark errors
- optional E049 split trace if r1 is promising

## Result

- Outcome: negative; code reverted.
- Baseline references:
  - Current r3 lane baseline: `prefill-current-ub2048-base-r3 = 11.6534 TPS`.
  - Same-session default-off smoke after E049 instrumentation: `prefill-e049-posttrace-default-r1 = 11.92 TPS`.
- Candidate: `prefill-e051-q3dequant128-r1 = 11.46 TPS`.
- Delta: `-1.66%` vs current r3 baseline; `-3.86%` vs same-session default-off smoke.
- Analytical gate:
  - `formula_sanity_checks.py`: OK.
  - `speedup_model.py`: a `10%` local dequant improvement at estimated wall share `0.1669` projects `1.0154x`, `11.8329 TPS`.
  - `required_acceptance.py`: non-spec placeholder check is feasible for `1.010x`; acceptance output is not meaningful for this kernel probe.
- Confidence: r1 is enough to reject because the candidate is clearly below both comparison references.
- Recommendation: keep the original 64-thread Q3_K dequant kernel; do not keep the 128-thread fp16 variant.

## Notes

- Surprises: adding more threads per block did not reduce the repeated Q3_K dequant cost; likely memory/store or occupancy pressure dominates.
- Follow-up action: future Q3_K dequant work needs a different memory/layout strategy, not just more threads per block.
