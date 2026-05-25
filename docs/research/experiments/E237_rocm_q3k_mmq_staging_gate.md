# E237 ROCm Q3_K MMQ Staging Gate

## Metadata

- Experiment ID: E237
- Date: 2026-05-25
- Owner: Codex
- Branch/Commit: master after `a4e402ecc`
- Target lane: H42 ROCm large-Q3_K prefill route, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, full offload, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: the existing RDNA4 MMQ double-buffer LDS staging path may improve the current direct `Q3_K x F16` MMQ body for the dominant `17408x5120@2048` bucket when enabled for dense non-MoE Q3_K.
- Mechanism: the staged path preloads the next Q3_K tile into LDS while the current tile is consumed, potentially reducing exposed X-side tile load latency in the current MMQ body.
- Why now: E204/E214 rejected selector-only current-MMQ promotion and MMQ-X retile, but the body still has one already-implemented RDNA4 topology variant that was gated to MoE only. This is a cheap body-level gate before writing a new kernel.

## Math / Theory

- Assumptions:
  - E214 exact `17408x5120@2048` forced-MMQ point is `2078.510 ms`, still far slower than the E228 cublas split `1398.159 ms`.
  - A useful staged-MMQ signal must be large, not a small polish: the current MMQ body needs roughly a one-third local improvement just to approach cublas on this bucket.
  - Shared-memory pressure can increase: default `mmq_x=128` used about `40448 B`; double-buffer X staging is expected to move near the 64 KiB limit.
- Expected speedup corridor:
  - Strong candidate: exact point sum drops by `>25%` and resources still fit at `mmq_x=128`.
  - Weak/no candidate: point timing ties/regresses, or selector falls back to a smaller `mmq_x` that repeats the E214 tile-count loss.
- Failure conditions:
  - extra LDS footprint reduces occupancy or forces a smaller tile;
  - added synchronization/loading work dominates any latency hiding;
  - point improves locally but remains too far from cublas to justify wall A/B.

## Implementation Plan

1. Minimal code surface to change:
   - extend the existing RDNA4 MMQ staging predicate in `ggml/src/ggml-cuda/mmq.cuh` with a default-off dense Q3_K env knob.
2. Guard rails:
   - env-gated only: `GGML_RDNA4_Q3_MMQ_STAGING=1`;
   - keep the existing MoE knob unchanged: `GGML_RDNA4_MOE_MMQ_STAGING=1`;
   - exact H42 matcher only for the first measurement: `row_diff=17408`, `ne00=5120`, `ncols=2048`;
   - point-level timing and resource review before any wall A/B.
3. Rollback path:
   - revert the predicate extension if the probe is not worth keeping as opt-in instrumentation.

## Benchmark Plan

- Baseline command:
  - `GGML_EXPERIMENTAL_Q3K_HOT_DIRECT_ROUTE=1` with exact `17408x5120@2048` filters and MMQ timing/resource trace.
- Candidate command:
  - same exact route with `GGML_RDNA4_Q3_MMQ_STAGING=1`.
- Number of runs:
  - one trace run per point gate; no wall A/B unless exact point timing moves strongly.
- Artifacts path:
  - `build_logs/agent-workload/e237-rocm-q3k-mmqstaging-*.{server.log,diagnostics.md}`.

## Metrics

- exact `mul_mat_q_case` timing rows for `type=11`, `nrows_x=17408`, `ncols_max=2048`, `ncols_dst=2048`
- point sum/mean and robust sum excluding startup outliers
- resource fields: `mmq_x_best`, LDS bytes, regs, max blocks per SM, occupancy, `rdna4_staging_req/eff`
- aggregate completion TPS only as trace context

## Result

- Outcome: reject; runtime probe reverted, no wall A/B.
- Delta:
  - exact timing rows stayed matched on both sides: `378` rows for `type=11`, `nrows_x=17408`, `ncols_max=2048`, `ncols_dst=2048`;
  - control current-MMQ point sum: `2053.749 ms`, mean `5.433 ms`, robust `<10 ms` sum `2037.182 ms`;
  - staged-MMQ point sum: `2496.968 ms`, mean `6.606 ms`, robust `<10 ms` sum `2468.594 ms`;
  - point regression: `+21.58%` all rows, `+21.18%` robust rows;
  - resources stayed at `mmq_x=128`, `regs=183`, `max_blocks_per_sm=1`, `occupancy=6.25%`, but shared memory rose `40448 B -> 61956 B` (`61.72% -> 94.54%` of the local SMEM limit);
  - trace-context prompt eval regressed `1056.65 -> 996.00 tok/s`.
- Confidence: high for rejecting dense Q3_K reuse of the current RDNA4 MMQ staging body. The route activated cleanly (`rdna4_staging_req=1`, `rdna4_staging_eff=1`) and the exact point signal moved the wrong way.
- Recommendation: do not extend the existing MoE/RDNA4 staging path to dense Q3_K prefill. The current MMQ body remains exhausted for H42; next work should either change the route body/topology more deeply or return to H43-style storage/correctness work.

## Notes

- Surprises:
  - The staged path fit without forcing a smaller tile, but near-limit LDS usage plus extra synchronization/loading made the dominant bucket slower rather than hiding latency.
- Follow-up action:
  - keep no dense Q3_K staging code;
  - classify this with E214 as another current-MMQ body dead end;
  - continue only with a genuinely different Q3_K body/layout route or a correctness-first storage route.

## Measured Data

Artifacts:

- `build_logs/agent-workload/e237-rocm-q3k-mmqstaging-control-r1.server.log`
- `build_logs/agent-workload/e237-rocm-q3k-mmqstaging-control-r1.diagnostics.md`
- `build_logs/agent-workload/e237-rocm-q3k-mmqstaging-candidate-r1.server.log`
- `build_logs/agent-workload/e237-rocm-q3k-mmqstaging-candidate-r1.diagnostics.md`

Point table:

| Variant | Timing rows | Point sum | Avg point | Robust sum `<10 ms` | Trace prompt tok/s | Resources |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| current MMQ | `378` | `2053.749 ms` | `5.433 ms` | `2037.182 ms` | `1056.65` | `183 regs`, `40448 B`, `1 block/SM`, `6.25% occ` |
| dense Q3_K staging | `378` | `2496.968 ms` | `6.606 ms` | `2468.594 ms` | `996.00` | `183 regs`, `61956 B`, `1 block/SM`, `6.25% occ` |
