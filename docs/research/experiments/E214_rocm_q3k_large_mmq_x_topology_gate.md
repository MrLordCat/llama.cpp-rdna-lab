# E214 ROCm Q3_K Large MMQ-X Topology Gate

## Metadata

- Experiment ID: E214
- Date: 2026-05-24
- Owner: Codex
- Branch/Commit: master after `0d5b7f802`
- Target lane: H42 ROCm large-Q3_K prefill route, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: the current H42 forced-MMQ body loses badly on the dominant `17408x5120@2048` Q3_K bucket partly because the default `mmq_x=128` shape has high LDS/register pressure and only one active block per SM.
- Mechanism: use the existing `GGML_MMQ_RDNA4_Q3_FORCE_MMQ_X` knob to force a smaller `mmq_x` on the exact H42 hot-shape route and check whether lower per-block resources improve synchronized point timing.
- Why now: E204/E208 reject selector-only current-MMQ promotion and stream-k tuning, but they leave one cheap topology/resource gate open before writing a new kernel body.

## Math / Theory

- Assumptions:
  - E204 H42-v0 forced the hot shapes onto current MMQ and found `17408x5120@2048` much slower than cublas (`3028.918 ms` vs E192 `1425.411 ms`).
  - E208 exact `17408x5120@2048` forced-MMQ point improved only from `2150.169 ms` to `2071.570 ms` when disabling stream-k, still far from cublas.
  - Default forced-MMQ reports about `40 KiB` LDS and `max_blocks_per_sm=1` for this shape.
- Expected speedup corridor:
  - A useful topology signal should improve exact point timing by more than trace noise and move resources in the expected direction.
  - If point timing does not improve, no wall A/B is justified.
- Failure conditions:
  - smaller `mmq_x` lowers resources but total point time rises because extra tiles/workgroups dominate;
  - point time improves locally but remains far from the cublas split route and cannot plausibly move wall;
  - route activation drifts away from exact `17408x5120@2048`.

## Implementation Plan

1. Minimal code surface to change: none; use existing env gates.
2. Guard rails:
   - exact H42 matcher only: `row_diff=17408`, `ne00=5120`, `ncols=2048`;
   - `GGML_TRACE_MMQ_TIMING=1`, `GGML_TRACE_MMQ_TIMING_SYNC=1`, and `GGML_TRACE_MMQ_RESOURCES=1`;
   - point-level review before any wall A/B.
3. Rollback path: no code to revert; document rejected env values.

## Benchmark Plan

- Baseline command: `GGML_EXPERIMENTAL_Q3K_HOT_DIRECT_ROUTE=1` with exact match and no `GGML_MMQ_RDNA4_Q3_FORCE_MMQ_X`.
- Candidate command: same route with `GGML_MMQ_RDNA4_Q3_FORCE_MMQ_X=96` and `64`.
- Number of runs: one trace run per point gate; no r3 unless exact point timing produces a strong signal.
- Artifacts path: `build_logs/agent-workload/e214-rocm-q3k-mmqx-*.{server.log,diagnostics.md}`.

## Metrics

- exact `mul_mat_q_case` timing rows for `type=11`, `nrows_x=17408`, `ncols_max=2048`, `ncols_dst=2048`
- robust point sum/mean excluding startup outliers
- resource fields: `mmq_x_best`, `mmq_x_forced`, LDS, regs, max blocks per SM, occupancy
- aggregate completion TPS only as trace context

## Result

- Outcome: reject; no wall promotion.
- Delta:
  - exact timing rows stayed matched on both sides: `378` rows for `type=11`, `nrows_x=17408`, `ncols_max=2048`, `ncols_dst=2048`;
  - default `mmq_x=128`: point sum `2078.510 ms`, average `5.499 ms`, robust sum `<10 ms` `2062.360 ms`;
  - forced `mmq_x=64`: point sum `2494.060 ms`, average `6.598 ms`, robust sum `<10 ms` `2476.660 ms`;
  - point regression: `+19.99%` all rows, `+20.09%` robust rows;
  - trace-context aggregate regressed `6.7422 -> 6.6220 TPS`, prompt `1019.13 -> 995.39 tok/s`;
  - resources moved in the expected direction but did not help: `regs 183 -> 142`, shared `40448 -> 30976 B`, `max_blocks_per_sm 1 -> 2`, occupancy `6.25% -> 12.50%`.
- Confidence: high for rejecting the simple smaller-`mmq_x` topology. The point signal is large and goes the wrong way despite better resource telemetry.
- Recommendation: stop this H42 sub-branch. Do not sweep `GGML_MMQ_RDNA4_Q3_FORCE_MMQ_X` further on the current large-MMQ body unless a new kernel body changes the work balance; smaller `mmq_x` adds enough extra tile work to dominate the resource improvement.

## Measured Data

Artifacts:

- `build_logs/agent-workload/e214-rocm-q3k-mmqx128-point-r1.server.log`
- `build_logs/agent-workload/e214-rocm-q3k-mmqx128-point-r1.diagnostics.md`
- `build_logs/agent-workload/e214-rocm-q3k-mmqx64-point-r1.server.log`
- `build_logs/agent-workload/e214-rocm-q3k-mmqx64-point-r1.diagnostics.md`

Point table:

| Variant | Timing rows | Point sum | Avg point | Robust sum `<10 ms` | Trace TPS | Prompt tok/s | Resources |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default `mmq_x=128` | `378` | `2078.510 ms` | `5.499 ms` | `2062.360 ms` | `6.7422` | `1019.13` | `183 regs`, `40448 B`, `1 block/SM`, `6.25% occ` |
| forced `mmq_x=64` | `378` | `2494.060 ms` | `6.598 ms` | `2476.660 ms` | `6.6220` | `995.39` | `142 regs`, `30976 B`, `2 blocks/SM`, `12.50% occ` |

## Notes

- Surprises:
  - resource telemetry improved substantially, but point timing worsened substantially. This is a clean bottleneck-shift signal inside the same route body: lowering LDS/register pressure by shrinking `mmq_x` increases tile/workgroup overhead enough to lose.
- Follow-up action:
  - close current-MMQ topology tuning for the dominant H42 shape;
  - return to a genuinely new Q3_K x F16 body or graph/storage scheduling route that reduces real staging/work, rather than retuning today's Q8-staged large-MMQ tile shape.
