# E019 H18 C01 Q3_K Load-Tiles Scale Fusion

## Metadata

- Experiment ID: E019
- Date: 2026-05-14
- Owner: Codex
- Branch/Commit: local working tree after E018 reject/revert
- Target lane: `Qwen3.6-27B-Q3_K_S`, `tasks=quick`, `task_ids=review_bug,patch_sim`, `ctx=12288`, `batch=6144`, `ubatch=192`, `q4_0/q4_0`, `spec=none`, no-reuse

## Hypothesis

- Statement: RDNA4 Q3_K prefill can reduce `load_tiles_q3_K` overhead by fusing scale unpack into the quant-bit tile load pass.
- Mechanism: The MMA path currently traverses Q3_K rows once to load quant bits (`x_qs`) and then again to unpack/store scales (`x_df`). In the first pass each row already has 16 participating lanes (`kqsx=0..15`); lanes `kqsx=0..3` can compute the four scale groups for the same row, allowing the second pass to be skipped for MMA/WMMA paths.
- Why now: E018 showed that adding inner-loop register arrays regresses target timing. This candidate targets `load_tiles_q3_K` instead and should avoid increased inner-loop register pressure.

## Math / Theory

- Assumptions:
  - Active RDNA4 geometry: `mmq_y=64`, `nwarps=4`, physical warp `32`.
  - Current quant pass maps `threads_per_row=16`, `nrows=2`, so every row already has lanes `0..15` available.
  - Scale pass maps `4` lanes per row and covers the same `64` rows separately.
- Expected speedup corridor:
  - Removes one extra row pass over `64` Q3_K rows per MMQ tile load for MMA path.
  - Does not change shared footprint, tile count, or write-back.
  - Expected effect is small: `0.2-1.0%` target bucket if `load_tiles_q3_K` overhead is visible.
- Failure conditions:
  - Fusing scale work into the quant pass increases instruction pressure enough to hurt scheduling.
  - Compiler already overlaps or optimizes the separate pass well.
  - Additional branch on `kqsx < 4` costs more than the removed pass.

## Implementation Plan

1. Minimal code surface to change:
   - In `load_tiles_q3_K`, compute/store `x_df` during the first pass for MMA/WMMA paths when `kqsx < 4`.
   - Compile the old second scale pass only for non-MMA/DP4A paths.
2. Guard rails:
   - No selector, geometry, shared-memory, or Q4_K changes.
   - Non-MMA path remains structurally unchanged.
3. Rollback path:
   - Revert the `load_tiles_q3_K` local change.

## Benchmark Plan

- Baseline command:
  - Existing post-E015 reference: `c01-e015-rdna4-y64w4-r3` (`9.6080 TPS`)
  - Fresh trace/control: `focus-c01-current-hotspots-r1`
- Candidate command:
  - Same lane after rebuild, `runs=1` first.
- Number of runs:
  - `runs=1` for screen, target trace only if non-trace is not clearly negative.
- Artifacts path:
  - `build_logs/agent-workload/c01-e019-*`

## Metrics

- aggregate completion TPS
- prompt eval TPS
- `CUDA_NODE op=MUL_MAT kind=forward`
- `MMQ type=11 ncols_max=192`

## Result

- Outcome: rejected
- Delta: `9.6080 TPS` E015 reference -> `8.2082 TPS` candidate r1 (`-14.57%`).
- Confidence: high enough to reject without trace; lane settings matched and errors were `0`.
- Recommendation: keep the original separate Q3_K scale pass for MMA/WMMA paths.

## Notes

- Candidate artifact:
  - `build_logs/agent-workload/c01-e019-q3-loadtiles-fuse-scales-r1.csv`
  - `build_logs/agent-workload/c01-e019-q3-loadtiles-fuse-scales-r1.diagnostics.md`
  - `build_logs/agent-workload/c01-e019-q3-loadtiles-fuse-scales-r1.server.log`
- Measured server timing:
  - aggregate completion TPS: `8.2082`
  - prompt eval TPS mean: `694.15`
  - decode eval TPS mean: `30.465`
  - prompt eval mean: `10641.07 ms`
  - decode eval mean: `3938.945 ms`
- Decision detail:
  - The non-trace screen was far below the current C01 reference and below the expected noise band.
  - No target trace was run because this failed the cheap screen before trace overhead.
  - Runtime code was reverted and `llama-server` was rebuilt.
- Surprise:
  - Removing the separate scale pass looked cheap analytically, but fusing extra work into the quant-load lanes likely worsened scheduling/register pressure enough to dominate the removed loop.
- Follow-up action:
  - Avoid fusing more unpack work into `load_tiles_q3_K` quant lanes unless a compiler/resource trace shows spare registers and issue slots.
  - Prefer next probes that reduce tile count/shared footprint or change scheduling without increasing per-lane instruction pressure in the quant load pass.
