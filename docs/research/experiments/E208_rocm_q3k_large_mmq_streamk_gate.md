# E208 ROCm Q3_K Large-MMQ Stream-K Gate

## Metadata

- Experiment ID: E208
- Date: 2026-05-24
- Owner: Codex
- Branch/Commit: master after `18e531f24`
- Target lane: H42 large-Q3_K prefill route on Qwen3.6-27B-Q3_K_S, ROCm `build-rocm-vec`, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: current large-prefill Q3_K MMQ may lose to hipBLAS partly because the RDNA4 stream-k decomposition underutilizes or over-synchronizes the large hot buckets.
- Mechanism: use the existing exact H42 hot-shape gate to route only the dominant `17408x5120@2048` family to MMQ, then compare default RDNA4 stream-k against tiled MMQ by raising `GGML_MMQ_RDNA4_STREAM_K_MIN_NE11` above `2048`.
- Why now: E204 rejected broad current-MMQ routing, E207 showed padded storage does not rescue the `10240x5120@2048` current-MMQ body, and the next useful question is whether the large-MMQ topology itself has any point-level headroom before writing a new kernel body.

## Math / Theory

- Assumptions:
  - E192 cublas `17408x5120@2048` point was `1425.411 ms` across `378` calls.
  - E204 current-MMQ on the same bucket was much slower (`3028.918 ms`), so only a large point-level topology win would matter.
  - E014/E027 already rejected stream-k/force-x work on the old C01 `ncols_max=192` lane; this E208 gate is a different exact large-prefill shape and must not become a broad sweep.
- Expected speedup corridor:
  - useful only if tiled MMQ materially lowers the exact bucket point time;
  - if the point remains slower than current-MMQ or still far above cublas, stop immediately.
- Failure conditions:
  - exact route activation leaks to other hot buckets;
  - point timing is neutral/slower;
  - tiled topology causes resource cliff or timeout.

## Implementation Plan

1. Minimal code surface to change: none; use existing env-gated H42 matcher and `GGML_MMQ_RDNA4_STREAM_K_MIN_NE11`.
2. Guard rails:
   - exact match only: `MATCH_ROW_DIFF=17408`, `MATCH_NE00=5120`, `MATCH_NCOLS=2048`;
   - no force-x sweep;
   - no wall run unless point timing improves enough to plausibly compete with cublas.
3. Rollback path: env-only probe; no code rollback needed.

## Benchmark Plan

- Baseline command: exact `17408x5120@2048` hotshape MMQ trace with default stream-k.
- Candidate command: same trace with `GGML_MMQ_RDNA4_STREAM_K_MIN_NE11=4096` to disable RDNA4 stream-k for `ncols=2048`.
- Number of runs: one point trace per side.
- Artifacts path: `build_logs/agent-workload/e208-rocm-q3k-large-mmq-streamk-*`.

## Metrics

- exact-bucket route count
- Q3_K MMQ point timing for `17408x5120@2048`
- resource telemetry (`regs`, shared memory, occupancy)
- aggregate completion TPS only as trace context

## Result

- Outcome: rejected as a promotion path; useful diagnostic only.
- Delta:
  - exact route activation matched on both sides: `378` `mul_mat_q_direct_hotshape` rows and no cublas leakage for `17408x5120@2048`;
  - default RDNA4 stream-k forced-MMQ: `2150.169 ms`, average `5.688 ms`, trace context `6.7987 TPS`, prompt `1032.85 tok/s`;
  - tiled/no-stream-k forced-MMQ (`GGML_MMQ_RDNA4_STREAM_K_MIN_NE11=4096`): `2071.570 ms`, average `5.480 ms`, trace context `6.9106 TPS`, prompt `1053.51 tok/s`;
  - local forced-MMQ point delta: `-78.599 ms` (`+3.65%` faster);
  - resources stayed identical (`mmq_x=128`, `regs=183`, `nbytes_shared=40448`, `occupancy=6.25%`);
  - candidate output sanity was normal `Thinking Process:` text with `errors=0`.
- Confidence: high that no-stream-k is locally better than default stream-k inside current large MMQ, and high that this still is not enough to replace cublas. E192/E206 cublas timing for the same family is about `3.7 ms/call`; the no-stream-k MMQ candidate is `5.48 ms/call`.
- Recommendation: do not promote wall A/B and do not tune stream-k further. Stream-k is a secondary tax, not the root cause. The H42 blocker is the current large-Q3_K MMQ body itself: `183` regs, `40 KiB` shared, one block/SM, and too much Q8-staged work relative to hipBLAS. Continue only with a new fused/direct body or graph scheduling/storage route that avoids this current-MMQ topology.

## Notes

- This is not a return to broad MMQ forcing. It is a one-shape topology gate to decide whether current-MMQ has any remaining large-prefill headroom.

## Measured Data

Artifacts:

- `build_logs/agent-workload/e208-rocm-q3k-17408-defaultstreamk-point-r1.server.log`
- `build_logs/agent-workload/e208-rocm-q3k-17408-defaultstreamk-point-r1.diagnostics.md`
- `build_logs/agent-workload/e208-rocm-q3k-17408-nostreamk-point-r1.server.log`
- `build_logs/agent-workload/e208-rocm-q3k-17408-nostreamk-point-r1.diagnostics.md`

Point table:

| Variant | Hotshape routes | Timing rows | Total point | Avg point | Trace TPS | Prompt tok/s | Resources |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default stream-k | `378` | `378` | `2150.169 ms` | `5.688 ms` | `6.7987` | `1032.85` | `183 regs`, `40448 B`, `6.25% occ` |
| no stream-k | `378` | `378` | `2071.570 ms` | `5.480 ms` | `6.9106` | `1053.51` | `183 regs`, `40448 B`, `6.25% occ` |

Interpretation:

- No-stream-k slightly improves the bad forced-MMQ route, but does not close the gap to the normal cublas route.
- This changes the next H42 design constraint: do not spend time on stream-k/selector tuning; the replacement route must reduce Q3_K/Q8 staging or accumulator/shared-memory pressure at the body level.
