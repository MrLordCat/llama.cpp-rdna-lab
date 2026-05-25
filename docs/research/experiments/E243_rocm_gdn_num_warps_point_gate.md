# E243 ROCm GDN num-warps point gate

## Metadata

- Experiment ID: E243
- Date: 2026-05-25
- Owner: Codex
- Branch/Commit: master after `bcbb9d996`
- Target lane: ROCm cold-first L1, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, `spec=none`, no reuse, thinking on, `max_tokens=64`

## Hypothesis

- Statement: after E230 showed a large synchronized GDN chunk-size point win that did not convert to wall TPS, a smaller block-geometry change might reduce GDN kernel time without changing chunk boundaries.
- Mechanism: temporarily expose `num_warps` in `launch_gated_delta_net` and compare the default `4` warps against `2` warps with synchronized GDN timing.
- Why now: E228 still shows `GATED_DELTA_NET/f32` as the second traced hotspot, but E230 warned that only strong point evidence should be promoted to wall A/B.

## Math / Theory

- Assumptions:
  - E241 current cold r1 baseline is `7.6932 TPS`; +20% target is about `9.23 TPS`.
  - E228/E230 place GDN around `~1.0 s` in a trace-heavy run, below the Q3_K `MUL_MAT` share.
  - To matter for the cold +20% target, a GDN-only path would need a very large local win and wall conversion, not a low-single-digit point improvement.
- Expected speedup corridor:
  - Promote to no-trace wall A/B only if point timing moves clearly enough to plausibly beat noise and prior E230 bottleneck shift.
- Failure conditions:
  - point timing is flat or only low-single-digit percent;
  - any wall movement appears only in trace-heavy TPS, not in synchronized point totals.

## Implementation Plan

1. Minimal code surface to change:
   - temporary env-gated `GGML_GDN_NUM_WARPS` in `ggml/src/ggml-cuda/gated_delta_net.cu`;
   - add `num_warps` to temporary trace logs.
2. Guard rails:
   - default remains `4`;
   - accepted values limited to `1`, `2`, or `4`;
   - no wall A/B unless point timing is strong.
3. Rollback path:
   - revert the temporary code if point gate is insufficient.

## Benchmark Plan

- Baseline command:
  - `GGML_TRACE_GDN_PATH=1 GGML_TRACE_GDN_TIMING=1 GGML_TRACE_GDN_TIMING_SYNC_HIP=1 GGML_TRACE_GDN_TIMING_PRE_SYNC_HIP=1`
  - `python scripts/agent_workload_bench.py --server-bin build-rocm-vec/bin/llama-server.exe ... --label e243-rocm12k-gdn-warps4-point-r1 ... --server-extra "--spec-type none"`
- Candidate command:
  - same, plus `GGML_GDN_NUM_WARPS=2`
- Number of runs:
  - one point-level run per variant.
- Artifacts path:
  - `build_logs/agent-workload/e243-rocm12k-gdn-warps4-point-r1.*`
  - `build_logs/agent-workload/e243-rocm12k-gdn-warps2-point-r1.*`

## Metrics

- synchronized GDN `total_ms` sum
- robust GDN `total_ms < 1` sum
- GDN chunk bucket sums
- trace-context aggregate TPS only as context, not a speed claim

## Result

- Outcome: reject and revert.
- Delta:
  - all GDN timing rows: `1002.233 -> 987.097 ms` (`-1.51%`);
  - robust rows with `total_ms < 1`: `999.479 -> 984.521 ms` (`-1.50%`);
  - dominant `n_tokens_chunk=128`: `980.679 -> 965.886 ms` (`-1.51%`);
  - trace-context aggregate TPS was effectively tied: `7.4809 -> 7.4870`, not used as a speed claim.
- Confidence: high that this mechanism is too small; medium that GDN geometry generally is exhausted, because E230 already found a larger local GDN point win that still failed wall A/B.
- Recommendation: do not continue GDN warp-count / block-geometry tuning for the current +20% cold target. Keep current default, rebuild `build-rocm-vec` after rollback, and return to structural Q3_K body/layout/topology work.

## Measured Data

| Variant | Rows | Sum ms | Robust rows `<1 ms` | Robust sum ms | Trace TPS | Prompt ms | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default `num_warps=4` | `2976` | `1002.233` | `2975` | `999.479` | `7.4809` | `6457.74` | `31.00` |
| `GGML_GDN_NUM_WARPS=2` | `2976` | `987.097` | `2975` | `984.521` | `7.4870` | `6450.64` | `30.98` |

Chunk bucket timing:

| Chunk | Default count | Default sum | Warps2 count | Warps2 sum | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `96` | `4.123 ms` | `96` | `3.912 ms` | `-5.12%` |
| `2` | `48` | `7.532 ms` | `48` | `7.469 ms` | `-0.84%` |
| `65` | `48` | `9.899 ms` | `48` | `9.830 ms` | `-0.70%` |
| `128` | `2784` | `980.679 ms` | `2784` | `965.886 ms` | `-1.51%` |

## Notes

- Surprises:
  - unlike older E108, `num_warps=2` was not slower after the current driver/code state, but the effect was far too small.
- Follow-up action:
  - no code kept;
  - `build-rocm-vec` was rebuilt after rollback so future perf runs match source.
