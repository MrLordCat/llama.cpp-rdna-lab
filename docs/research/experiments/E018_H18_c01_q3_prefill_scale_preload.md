# E018 H18 C01 Q3_K Prefill Scale Preload

## Metadata

- Experiment ID: E018
- Date: 2026-05-14
- Owner: Codex
- Branch/Commit: local working tree after E017
- Target lane: `Qwen3.6-27B-Q3_K_S`, `tasks=quick`, `task_ids=review_bug,patch_sim`, `ctx=12288`, `batch=6144`, `ubatch=192`, `q4_0/q4_0`, `spec=none`, no-reuse

## Hypothesis

- Statement: RDNA4 Q3_K WMMA prefill can reduce LDS pressure by preloading per-row Q3 scales from `x_df` once per k-fragment and reusing them across the `j0` loop.
- Mechanism: In `vec_dot_q8_0_16_q8_1_mma`, `x_df[i*stride + k0/4]` depends on row and k-fragment, not on destination column tile `j0`. For active `ncols=192`, `mmq_x=96`, `tile_C::J=16`, the `j0` loop has 6 iterations, so the same scale values are read repeatedly from LDS.
- Why now: Fresh phase split shows prompt/prefill dominates wall time, and prompt-phase `MMQ type=11 ncols_max=192` is `7490.845 ms` (`44.06%` of prompt CUDA_NODE).

## Math / Theory

- Assumptions:
  - Active RDNA4 geometry: `mmq_x=96`, `mmq_y=64`, `nwarps=4`, `tile_C=16x16`, `tile_C::ne=8`, `ntx=1`.
  - `j0` iterations per active bucket: `96 / 16 = 6`.
  - `x_df` scale loads in the current AMD WMMA inner loop are repeated for each `j0`.
- Expected speedup corridor:
  - Per thread per k-fragment, x-scale LDS loads drop from approximately `6 * 8 = 48` to `8`.
  - The candidate adds about `8` float registers for cached scales.
  - Because the target bucket is `~44%` of prompt CUDA_NODE, even a `1-3%` target-bucket gain would be visible as roughly `0.4-1.3%` prompt-GPU improvement.
- Failure conditions:
  - Extra registers lower occupancy or scheduling enough to offset reduced LDS traffic.
  - Compiler already hoists the repeated LDS load, making the manual cache neutral or worse.
  - Change affects non-Q3 paths or non-RDNA4 paths.

## Implementation Plan

1. Minimal code surface to change:
   - Add an RDNA4/WMMA-only `vec_dot_q3_K_q8_1_mma` helper in `ggml/src/ggml-cuda/mmq.cuh`.
   - Point only `GGML_TYPE_Q3_K` MMQ trait at this helper.
2. Guard rails:
   - Non-RDNA4 and non-WMMA paths call the existing generic `vec_dot_q8_0_16_q8_1_mma`.
   - No geometry, selector, shared-memory, or Q4_K changes.
3. Rollback path:
   - Revert the trait change and helper.

## Benchmark Plan

- Baseline command:
  - Existing post-E015 reference: `c01-e015-rdna4-y64w4-r3` (`9.6080 TPS`)
  - Fresh trace/control: `focus-c01-current-hotspots-r1`
- Candidate command:
  - Same lane after rebuild, `runs=1` first.
- Number of runs:
  - `runs=1` for screen, `runs=3` only if target/runtime positive.
- Artifacts path:
  - `build_logs/agent-workload/c01-e018-*`

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- error rate
- prefill/decode split
- `CUDA_NODE op=MUL_MAT kind=forward`
- `MMQ type=11 ncols_max=192`
- prompt-phase target timing if kernel-full trace is collected

## Result

- Outcome: regression on target trace
- Delta:
  - Non-trace r1: `9.6317 TPS`, slightly above E015 r3 reference `9.6080 TPS`, but too small to trust alone.
  - Kernel-full trace: `focus-c01-current-hotspots-r1 -> c01-e018-q3-scale-preload-trace-r1`.
  - `CUDA_NODE`: `22127.070 -> 22498.676 ms` (`+371.606 ms`).
  - `CUDA_NODE op=MUL_MAT kind=forward`: `14412.924 -> 14562.136 ms` (`+149.212 ms`).
  - `MMQ`: `9846.201 -> 9905.244 ms` (`+59.043 ms`).
  - Target `MMQ type=11 ncols_max=192`: `9048.863 -> 9103.787 ms` (`+54.924 ms`).
- Confidence: enough to reject at r1 because the required hotspot metric regressed.
- Recommendation: reject and revert code; do not run `runs=3`.

## Notes

- Surprises:
  - The manual scale preload likely increased register/scheduling pressure more than it reduced LDS traffic, or the compiler already handled the repeated load well enough.
  - Non-trace TPS was slightly positive, but the target trace failed the causal requirement.
- Follow-up action:
  - Runtime code was reverted and `llama-server` rebuilt.
  - Next C01 candidates should avoid extra per-thread register arrays unless the theory gate can show a larger limiting-term win.
