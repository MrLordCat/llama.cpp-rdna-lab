# E207 ROCm Q3_K Padded Hotshape MMQ Gate

## Metadata

- Experiment ID: E207
- Date: 2026-05-24
- Owner: Codex
- Branch/Commit: master after `18e531f24`
- Target lane: H42/H43 bridge on Qwen3.6-27B-Q3_K_S, ROCm `build-rocm-vec`, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: E201-P2a changed the Q3_K MMQ loader/storage contract enough that the one previously local-positive H42 bucket (`10240x5120@2048`) deserves a narrow point-level retest with padded physical storage.
- Mechanism: keep the existing exact hot-shape matcher from E204-R2, but add `GGML_CUDA_Q3K_PADDED_STORAGE=1` and `GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1` so the direct MMQ route reads 112-byte padded Q3_K blocks instead of raw 110-byte blocks.
- Why now: broad H42 current-MMQ forcing and selector-only salvage were rejected, but E201-P2a is a real route-body input change with measured MMQ point speedup on small-prompt shapes.

## Math / Theory

- Assumptions:
  - E204-R2 isolated `10240x5120@2048` with `144` route hits and no wall win on the raw current MMQ body.
  - E201-P2a improved short-lane Q3_K MMQ point timing by `+8.34%`, but the active large-prefill path still mostly uses cublas/H42.
  - A tiny prefill-only improvement is not enough: the analytic gate projects only `1.0155x` wall for `+2%` prefill speedup at `prefill_share=0.78`.
- Expected speedup corridor:
  - promotion requires a clear local point win on the exact bucket before any wall run;
  - without point movement, this branch is rejected immediately as another selector/body mismatch.
- Failure conditions:
  - point timing is neutral or slower;
  - route activation leaks to other hot buckets;
  - candidate output has errors or corrupt text in any later wall sanity.

## Implementation Plan

1. Minimal code surface to change: none for the first gate; use existing env-gated H42 matcher and H43 padded storage/MMQ knobs.
2. Guard rails:
   - exact match only: `MATCH_ROW_DIFF=10240`, `MATCH_NE00=5120`, `MATCH_NCOLS=2048`;
   - point timing first with `GGML_CUDA_DISABLE_GRAPHS=1`;
   - no wall claim unless the point gate moves.
3. Rollback path: env-only probe; no code rollback needed.

## Benchmark Plan

- Baseline command: active-lane trace with the exact hotshape route but without padded storage.
- Candidate command: same trace plus `GGML_CUDA_Q3K_PADDED_STORAGE=1` and `GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1`.
- Number of runs: one point trace for each side; wall `r1` only if the exact bucket improves.
- Artifacts path: `build_logs/agent-workload/e207-rocm-q3k-padded-hotshape-*`.

## Metrics

- exact-bucket route count
- Q3_K MMQ point timing for `10240x5120@2048`
- aggregate completion TPS only as trace context
- prompt/decode split only if wall promoted

## Result

- Outcome: rejected as a speed route; no wall run promoted.
- Delta:
  - route activation matched exactly on both sides: `144` `mul_mat_q_direct_hotshape` rows and `144` timing rows;
  - raw current-MMQ control: `total_sum_ms=495.784`, robust `<10 ms` sum `483.000 ms`, average `3.378 ms`;
  - padded-storage MMQ candidate: `total_sum_ms=496.678`, robust `<10 ms` sum `483.606 ms`, average `3.382 ms`;
  - robust point delta: `+0.13%` slower, effectively a tie/negative;
  - trace context aggregate: `7.4805 -> 7.4638 TPS`, prompt `1166.57 -> 1158.86 tok/s`, decode `30.44 -> 30.79 tok/s`;
  - candidate output sanity was normal `Thinking Process:` text with `errors=0`.
- Confidence: high for rejecting this exact narrow branch. The matcher did not leak to cublas for the bucket, `q3k_padded=1` confirmed the padded loader, resources stayed identical (`regs=183`, `occupancy=6.25%`), and point timing did not move.
- Recommendation: do not run wall confirmation and do not keep iterating selector-style H42 salvage. Padded storage helps the small MMQ lane from E201-P2a, but it does not rescue the large-prefill current-MMQ body. Continue only with a genuinely new H42 body/topology for dominant `17408x5120`/`5120x17408` families, or broaden H43 correctness coverage separately.

## Notes

- Analytical gate:
  - `python scripts/research/formula_sanity_checks.py` passed.
  - `python scripts/research/speedup_model.py --baseline-tps 7.6054 --prefill-share 0.78 --draft-len 1 --accept-rate 0 --spec-overhead 0 --flash-prefill-speedup 1.02 --decode-kernel-speedup 1.00` projected `1.0155x`.
  - `python scripts/research/required_acceptance.py --target-wall 1.02 --draft-len 1 --prefill-share 0.78 --prefill-speedup 1.02 --decode-kernel-speedup 1.00 --spec-overhead 0` reported the target unreachable under only `+2%` prefill assumptions.
- Follow-up action: run point trace A/B.

## Measured Data

Artifacts:

- `build_logs/agent-workload/e207-rocm-q3k-hotshape-10240-raw-point-r1.server.log`
- `build_logs/agent-workload/e207-rocm-q3k-hotshape-10240-raw-point-r1.diagnostics.md`
- `build_logs/agent-workload/e207-rocm-q3k-hotshape-10240-padded-point-r1.server.log`
- `build_logs/agent-workload/e207-rocm-q3k-hotshape-10240-padded-point-r1.diagnostics.md`

Point table:

| Variant | Hotshape routes | Timing rows | `q3k_padded` | Robust sum `<10 ms` | Robust avg | Full sum | Resources |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw current MMQ | `144` | `144` | `0` | `483.000 ms` | `3.378 ms` | `495.784 ms` | `183 regs`, `6.25% occ` |
| padded-storage MMQ | `144` | `144` | `1` | `483.606 ms` | `3.382 ms` | `496.678 ms` | `183 regs`, `6.25% occ` |

Interpretation:

- The physical padded loader is active but does not change large `10240x5120@2048` MMQ economics.
- This separates the E201-P2a win from H42 large-prefill: P2a is still a valid small/decode MMQ opt-in, while H42 still needs a new route body rather than a loader-only/selector-only retest.
