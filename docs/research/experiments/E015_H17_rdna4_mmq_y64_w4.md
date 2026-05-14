# E015 H17 RDNA4 MMQ y64/w4 C01 keep

## Metadata

- Experiment ID: E015
- Date: 2026-05-14
- Owner: Codex
- Branch/Commit: `master`, base `0cd0c1f05`
- Target lane: `Qwen3.6-27B-Q3_K_S`, `ctx=12288`, `batch=6144`, `ubatch=192`, `q4_0/q4_0`, `review_bug,patch_sim`, `no-reuse`, thinking on

## Hypothesis

- Statement: RDNA4 MMQ may prefer a smaller y tile and fewer warps for the active C01 `Q3_K/ncols_max=192` bucket.
- Mechanism: `mmq_y=128/nwarps=8` uses high LDS per block (`shared_pct=88.09` for Q3_K). Pairing `mmq_y=64` with `nwarps=4` preserves the MMA write-back invariant while reducing shared pressure and per-block work.
- Why now: E014 showed simple selector knobs were not enough, but the compile failure for `mmq_y=64` exposed the real invariant: `nwarps * tile_C::I == mmq_y`.

## Math / Theory

- Assumptions:
  - RDNA4 physical warp size on this lane is `32`.
  - Existing block shape is `8 * 32 = 256` threads with `mmq_y=128`.
  - Candidate block shape is `4 * 32 = 128` threads with `mmq_y=64`.
- Expected speedup corridor: `+1%` to `+4%` if the active bucket is LDS/occupancy/resource sensitive.
- Failure conditions: lower occupancy/waves could reduce throughput if the path is math-limited rather than resource-placement limited.

## Implementation Plan

1. Minimal code surface to change:
   - `ggml/src/ggml-cuda/mmq.cuh`
   - RDNA4 host/device `mmq_y` returns `64`.
   - RDNA4 host/device MMQ `nwarps` returns `4`.
2. Guard rails:
   - RDNA4-only branch.
   - paired `runs=3` control after reverting candidate.
   - trace target check on `MUL_MAT forward` and `MMQ type=11 ncols_max=192`.
3. Rollback path:
   - restore RDNA4 to generic AMD `mmq_y=128`, `nwarps=8`.

## Benchmark Plan

- Baseline command: `scripts/agent_workload_bench.py --label c01-e015-control-postrevert-r3 ... --runs 3`
- Candidate command: `scripts/agent_workload_bench.py --label c01-e015-rdna4-y64w4-r3 ... --runs 3`
- Number of runs: `3`
- Artifacts path: `build_logs/agent-workload/c01-e015-*`

## Metrics

- aggregate completion TPS (wall)
- bootstrap delta CI
- trace target bucket timing
- error/timeout rate

## Result

- Outcome: win.
- Delta:
  - baseline: `9.3974 TPS`
  - candidate: `9.6080 TPS`
  - aggregate delta: `+0.2107 TPS` (`+2.24%`)
  - bootstrap 95% CI: `[+0.1855, +0.2368]` TPS
  - statistical verdict: positive
- Confidence: high for this lane.
- Recommendation: keep RDNA4 `mmq_y=64/nwarps=4` as default; monitor broader RDNA4 MMQ lanes.

## Notes

- Trace cross-check (`c01-poste013-r1-resources` -> `c01-e015-rdna4-y64w4-trace-r1`):
  - trace TPS: `6.61 -> 6.69`
  - `CUDA_NODE op=MUL_MAT kind=forward`: `15498.053 -> 14984.576 ms` (`-513.477 ms`)
  - `MMQ`: `10887.326 -> 10381.647 ms` (`-505.679 ms`)
  - `MMQ type=11 ncols_max=192`: `9949.928 -> 9551.391 ms` (`-398.537 ms`)
  - Q3 resource line changed from `mmq_y=128`, `shared_pct=88.09`, `occupancy_pct=12.50`, `waves_per_sm=8.00` to `mmq_y=64`, `shared_pct=54.49`, `occupancy_pct=6.25`, `waves_per_sm=4.00`.
- The lower reported occupancy did not hurt this lane; lower shared pressure and smaller y tile were net positive.
