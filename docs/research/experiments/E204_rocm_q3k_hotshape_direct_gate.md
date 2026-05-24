# E204 ROCm Q3_K hot-shape direct gate

## Metadata

- Experiment ID: E204
- Date: 2026-05-24
- Owner: Copilot
- Branch/Commit: master after E203
- Target lane: ROCm large-Q3_K cublas path on Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: the next credible ROCm route after E203 is a shape-specialized direct `Q3_K x F16` branch for the hottest large-Q3_K cublas shapes, not another helper-level reuse tweak.
- Mechanism: by matching the actual hot row/width forms in the current cublas route, a future direct kernel can remove full-row fp16 staging and improve compute locality on the same route body.
- Why now: E203 rejected transient same-step reuse as the immediate next path; E192 still says large-Q3_K cublas is the practical wall bottleneck.

## Math / Theory

- Assumptions:
  - E192 split: `src0_convert_ms=1637.070`, `gemm_ms=3203.883`.
  - likely first target rows are `17408`, `10240`, `6144`, `5120` with `ne00 in {5120,17408}` and large `src1_ncols`.
- Expected speedup corridor:
  - no runtime speed claim yet; this step only adds a fail-closed route-match gate.
- Failure conditions:
  - hot-shape matches are too sparse on the practical lane;
  - the gate shape does not align with the true expensive rows in current logs.

## Implementation Plan

1. Minimal code surface to change:
   - add default-off `GGML_EXPERIMENTAL_Q3K_HOT_DIRECT_ROUTE` shape-match hook in `ggml/src/ggml-cuda/ggml-cuda.cu`.
2. Guard rails:
   - fail-closed: logging only, no route change yet.
3. Rollback path:
   - keep gate default-off or revert if it adds no value.

## Benchmark Plan

- Baseline command:
  - one diagnostic trace on the standard ROCm lane.
- Candidate command:
  - enable `GGML_EXPERIMENTAL_Q3K_HOT_DIRECT_ROUTE=1` together with existing split/route tracing to confirm matches.
- Number of runs:
  - 1 trace run.
- Artifacts path:
  - `build_logs/agent-workload/e204-rocm-q3k-hotshape-gate-*`

## Metrics

- count of hot-shape route matches
- matched row/width forms
- diagnostic TPS only as context

## Result

- Outcome: keep as the next direct-route implementation anchor.
- Delta:
  - diagnostic run `e204-rocm-q3k-hotshape-gate-r1`: `7.17 TPS` (trace-only context, not a speed claim).
  - hot-shape gate matched `1140` route events on the practical lane.
  - matched forms were concentrated in the intended Qwen buckets:
    - `row=17408 ne00=5120 ncols=2048` (`378`)
    - `row=17408 ne00=5120 ncols=1345` (`126`)
    - `row=5120 ne00=17408 ncols=2048` (`189`)
    - `row=5120 ne00=17408 ncols=1345` (`63`)
    - `row=10240 ne00=5120 ncols=2048` (`144`)
    - `row=10240 ne00=5120 ncols=1345` (`48`)
    - `row=6144 ne00=5120 ncols=2048` (`144`)
    - `row=6144 ne00=5120 ncols=1345` (`48`)
- Confidence: high for route-surface selection. The gate hits the exact large-Q3_K cublas shapes repeatedly across the active lane.
- Recommendation: proceed to a real shape-scoped direct `Q3_K x F16` prototype on this matched surface. This branch now has better activation evidence than H41 did.

## Notes

- Surprises:
  - the match count is high enough that a future direct route would not be a niche fallback; it would cover a large, regular part of the practical lane.
- Follow-up action:
  - implement the first real shape-scoped direct-route prototype on the `17408/5120/10240/6144` matched family;
  - keep the gate as the activation check while the first prototype stays env-gated.

## H42-v0 Prototype Follow-up

Date: 2026-05-24

Implementation:

- added an env-gated route override in `ggml/src/ggml-cuda/ggml-cuda.cu`:
  - `GGML_EXPERIMENTAL_Q3K_HOT_DIRECT_ROUTE=1`
  - matched hot Q3_K shapes are forced from the cublas route onto the existing direct `mul_mat_q` path;
  - route log label: `mul_mat_q_direct_hotshape`.

Validation run:

- artifact: `build_logs/agent-workload/e204-rocm-q3k-hotshape-directproto-r1.diagnostics.md`
- server log: `build_logs/agent-workload/e204-rocm-q3k-hotshape-directproto-r1.server.log`

Observed route activation:

- `hotshape=1140`
- `cublas=1232`
- `q_direct=0` for the non-hot generic direct label, meaning the hot-shape override became the active direct route on the intended surface.

Representative matched routes:

- `blk.0.attn_qkv.weight`: `src0_ne=(5120,10240)`, `src1_ne=(5120,2048)`
- `blk.0.attn_gate.weight`: `src0_ne=(5120,6144)`, `src1_ne=(5120,2048)`
- `blk.0.ffn_gate/up.weight`: `src0_ne=(5120,17408)`, `src1_ne=(5120,2048)`
- `blk.0.ffn_down.weight`: `src0_ne=(17408,5120)`, `src1_ne=(17408,2048)`

Measured outcome:

| Variant | Aggregate TPS | Prompt Eval TPS | Decode Eval TPS | Prompt Eval Mean |
| --- | ---: | ---: | ---: | ---: |
| E204 gate-only trace | `7.17` | baseline not normalized, but same-lane trace context | same-lane trace context | same-lane trace context |
| H42-v0 direct override | `6.18` | `907.78` | `30.94` | `8249.83 ms` |

Interpretation:

- this is a real route-body prototype and it is negative for the existing `mul_mat_q` path on these hot shapes;
- the negative result does not kill H42 as a whole, but it rejects the naive transfer "force current direct MMQ route onto the hot cublas shapes";
- therefore the next H42 step, if pursued, must be a genuinely new shape-specialized route rather than a selector override to today's `mul_mat_q` implementation.

Root-cause follow-up:

- focused MMQ timing/resource trace on the H42-v0 route showed that all rerouted hot prefill buckets landed on the same current MMQ geometry:
  - `mmq_x_best=128`, `mmq_y=64`, `regs=190`, `nbytes_shared=40448`, `max_blocks_per_sm=1`, `occupancy_pct=6.25`.
- local same-shape comparison against E192 cublas split shows why broad H42-v0 regressed:

| Shape | E192 cublas sum | H42-v0 MMQ sum | MMQ / cublas |
| --- | ---: | ---: | ---: |
| `17408x5120@2048` | `1425.411 ms` | `3028.918 ms` | `2.125x` |
| `5120x17408@2048` | `901.308 ms` | `1225.932 ms` | `1.360x` |
| `6144x5120@2048` | `879.019 ms` | `1261.515 ms` | `1.435x` |
| `10240x5120@2048` | `751.862 ms` | `535.949 ms` | `0.713x` |
| tails (`@1345`) | all slower on MMQ |  |  |

- conclusion: broad H42-v0 was dominated by seven losing buckets; only `10240x5120@2048` was a credible selector-salvage candidate.

## H42-v1 Narrow Subshape Follow-up

Date: 2026-05-24

Implementation:

- extended the existing H42 config in `ggml/src/ggml-cuda/ggml-cuda.cu` with optional exact filters:
  - `GGML_EXPERIMENTAL_Q3K_HOT_DIRECT_ROUTE_MATCH_ROW_DIFF`
  - `GGML_EXPERIMENTAL_Q3K_HOT_DIRECT_ROUTE_MATCH_NE00`
  - `GGML_EXPERIMENTAL_Q3K_HOT_DIRECT_ROUTE_MATCH_NCOLS`
- used them to isolate only the locally winning bucket: `row_diff=10240`, `ne00=5120`, `ncols=2048`.

Important validation note:

- CMake Tools initially rebuilt a different active build tree; the decisive runs below were taken only after rebuilding the actual perf binary `build-rocm-vec/bin/llama-server.exe`.

Route validation:

- route hits in the decisive traced run: `10240x5120@2048 -> 144` and no other hot bucket.

Measured outcome:

| Variant | Aggregate TPS | Prompt Eval TPS | Prompt Eval Mean |
| --- | ---: | ---: | ---: |
| trace-only control | `7.6054` | `1184.90` | `6320.35 ms` |
| narrow H42-v1 trace | `7.6028` | `1183.04` | `6330.28 ms` |
| no-trace control | `7.6054` | `1184.90` | `6320.35 ms` |
| narrow H42-v1 no-trace | `7.5544` | `1172.71` | `6386.05 ms` |

Interpretation:

- the selector salvage succeeded structurally: the matcher can now isolate a single promising hot bucket;
- it still failed as a speed route: same-overhead trace A/B is effectively tied, and clean no-trace A/B is slightly negative;
- therefore the current direct MMQ body is exhausted as an H42 vehicle, even when narrowed to the only locally favorable prefill bucket.

Decision:

- keep the env-gated override only as a documented negative control;
- do not promote the current hot-shape direct-MMQ override;
- keep the new exact-filter envs only as research instrumentation;
- if H42 continues, move to a new kernel/topology design for the `17408x5120` dominant bucket, or a route that avoids `src0` fp16 staging without falling back to today's Q8-staged MMQ body.