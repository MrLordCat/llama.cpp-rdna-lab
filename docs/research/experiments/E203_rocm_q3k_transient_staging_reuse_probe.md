# E203 ROCm Q3_K transient staging reuse probe

## Metadata

- Experiment ID: E203
- Date: 2026-05-24
- Owner: Copilot
- Branch/Commit: master after E190 follow-up
- Target lane: ROCm H35/H39 route diagnostic on Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: repeated Q3_K `src0` staging is not useful as a persistent cache, but it may still have enough temporal locality inside one graph step to support transient same-step reuse.
- Mechanism: if repeated keys for the same `src0` tensor/range recur close enough in launch order, one temporary fp16 staging could be reused by nearby sibling matmuls and released immediately, avoiding large persistent residency.
- Why now: E103 proved repeated staging exists, E104 rejected persistent cache, and E190 follow-up ruled out helper-level MMVQ live-state tweaks as the current limiter.

## Math / Theory

- Assumptions:
  - E192 large-Q3_K split: `src0_convert_ms=1637.070`, `gemm_ms=3203.883` on the practical real-context wall lane.
  - full persistent cache is not viable because E103 estimated about `42.002 GiB` fp16 footprint for unlimited reuse.
- Expected speedup corridor:
  - no claim yet; this is an instrumentation gate.
  - if a large fraction of repeated keys recur within a short event gap, a transient reuse prototype can still have a credible `+1%..+4%` wall ceiling.
- Failure conditions:
  - repeated keys remain widely separated in event order, making transient reuse nearly as residency-heavy as persistent cache;
  - adding locality metrics disturbs normal routing or creates non-trivial sync behavior;
  - the nearest repeated keys are not the expensive ones.

## Implementation Plan

1. Minimal code surface to change:
   - extend the existing default-off `GGML_TRACE_CUBLAS_Q3K_ROUTE` trace in `ggml/src/ggml-cuda/ggml-cuda.cu` with event-order locality metrics.
2. Guard rails:
   - no runtime behavior change unless tracing env is enabled;
   - do not add any reuse cache or new compute route in this step.
3. Rollback path:
   - keep the added metrics default-off, or revert if the trace becomes noisy or misleading.

## Benchmark Plan

- Baseline command:
  - none for speed; use one diagnostic trace under the existing H35 lane.
- Candidate command:
  - `GGML_TRACE_CUBLAS_Q3K_ROUTE=1 GGML_TRACE_CUBLAS_Q3K_ROUTE_MIN_NCOLS=1024 GGML_TRACE_CUBLAS_SPLIT_TIMING=1 GGML_TRACE_CUBLAS_SPLIT_DETAIL=1 GGML_TRACE_CUBLAS_SPLIT_TIMING_MIN_NCOLS=1024 python scripts/agent_workload_bench.py ...`
- Number of runs:
  - 1 trace run.
- Artifacts path:
  - `build_logs/agent-workload/e203-rocm-q3k-transient-reuse-*`

## Metrics

- repeated-key event gap (`event_idx`, `gap_since_prev`, min/max/avg gap per key)
- repeated-key convert time concentration
- aggregate completion TPS only as diagnostic context

## Result

- Outcome: reject as the immediate next code path; temporal locality is too weak for a simple transient same-step reuse branch.
- Delta:
  - diagnostic run `e203-rocm-q3k-transient-reuse-r1`: `7.05 TPS` (trace-only context, not a speed claim).
  - repeated-key locality summary from `e203-rocm-q3k-transient-reuse-r1.server.log`:
    - `repeats=1047`
    - `mean_gap=349.00`
    - `min_gap=349`
    - `max_gap=349`
    - no repeated key with `gap <= 128`
- Confidence: high for the structural verdict. The new trace fields are deterministic here: repeated keys recur on a fixed long cadence rather than in a short sibling cluster.
- Recommendation: do not build a simple transient reuse cache as the next branch. Move to a hot-shape direct `Q3_K x F16` route (H42), because the repeated staging is real but not temporally local enough for a cheap same-step reuse mechanism.

## Notes

- Surprises:
  - the gap distribution collapsed to a single cadence value (`349`) rather than a mixed near/far spectrum;
  - this means repeated Q3_K staging is structurally real but ordered like a regular graph sweep, not a small sibling burst.
- Follow-up action:
  - escalate directly to H42 hot-shape direct route work;
  - keep the added route-locality fields in the default-off trace as future evidence if a more elaborate graph-scheduling idea appears.