# E259 Vulkan 12k Practical KV / Shape Gates

## Metadata

- Experiment ID: E259
- Date: 2026-05-26
- Owner: Copilot
- Branch/Commit: local dirty tree after E257; E258 source prototype reverted
- Target lane: Vulkan, Qwen3.6-27B-Q3_K_S, `ctx=12288`, cold/no-reuse/no-prime, thinking on, `quick:triage_diff`, `max_tokens=64`, `spec=none`

## Hypothesis

- Statement: after the E257 `b7168/ub1024` rebaseline and the rejected E258 source topology, a small practical route might still improve cold wall time by changing only safe launch shape or KV dtype.
- Mechanism: test whether one full prompt chunk (`batch=7680`) removes enough scheduling overhead, and whether f16 KV improves Vulkan prompt/decode behavior at 12k while still fitting in local VRAM.
- Why now: E258 showed a Q3_K layout route could improve prompt eval slightly but hurt decode. Before another source topology, close the cheap no-code candidates around the new E257 profile.

## Math / Theory

- Assumptions: current best same-lane E257 profile is `7.0319 TPS`, prompt `999.22 tok/s`, decode `40.93 tok/s` at `ctx=12288,b=7168,ub=1024,q4_0/q4_0`.
- Expected speedup corridor: no-code probes need a clear r3 win before changing GUI defaults; a sub-1% result is treated as noise/optional, not progress toward the +20% target.
- Failure conditions: wall TPS below E257 best, decode regression that cancels prefill gain, or any fit/errors.

## Benchmark Plan

- Baseline: E257 `e257-vulkan12k-shape-b7168-ub1024-r3`.
- Candidates:
  - `e259-vulkan12k-shape-b7680-ub1024-r1`: q4/q4 KV, `batch=7680`, `ubatch=1024`.
  - `e259-vulkan12k-kvf16-b7168-ub1024-r1`: f16/f16 KV, `batch=7168`, `ubatch=1024`.
  - `e259-vulkan12k-kvf16-b7168-ub1024-r3`: r3 confirmation for the promising f16 KV r1.
- Number of runs: r1 for gates, r3 only for the f16 KV confirmation.
- Artifacts path: `build_logs/agent-workload/e259-*`.

## Metrics

- aggregate completion TPS (wall)
- prompt eval TPS
- decode eval TPS
- error rate
- batch chunking summary

## Result

| Label | KV | Batch | UBatch | Runs | TPS | Prompt tok/s | Decode tok/s | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E257 best | q4_0/q4_0 | 7168 | 1024 | 3 | 7.0319 | 999.22 | 40.93 | baseline |
| `e259-vulkan12k-shape-b7680-ub1024-r1` | q4_0/q4_0 | 7680 | 1024 | 1 | 6.9795 | 994.34 | 40.07 | reject |
| `e259-vulkan12k-kvf16-b7168-ub1024-r1` | f16/f16 | 7168 | 1024 | 1 | 7.1417 | 1022.10 | 40.11 | promising r1 only |
| `e259-vulkan12k-kvf16-b7168-ub1024-r3` | f16/f16 | 7168 | 1024 | 3 | 7.0543 | 1008.37 | 40.00 | tie / not default |

- Outcome: no-code tie, not a promotion.
- Delta: f16 KV r3 is `+0.32%` wall and `+0.92%` prompt vs E257, but decode is `-2.27%` and the wall delta is too small for a default change.
- Confidence: medium; r3 removed most of the r1 uplift.
- Recommendation: keep the GUI Vulkan 12k dense 27B default on q4/q4 KV. f16/f16 KV can remain a manual/diagnostic option for 12k when VRAM is available, but it is not part of the +20% route.

## Notes

- `batch=7680` produced a single `7489/7489` prompt chunk but still lost wall TPS, so E257's `b7168/ub1024` remains the shape.
- f16 KV slightly improves prompt mean but costs decode enough that it is noise-level on the full task.
- Next Vulkan source work still needs a new Q3_K topology with a clearer local mechanism; nearby shape/KV tweaks are exhausted for this 12k lane.
