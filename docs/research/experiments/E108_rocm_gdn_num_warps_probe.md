# E108 ROCm GDN Num-Warps Probe

## Metadata

- Experiment ID: E108
- Date: 2026-05-20
- Owner: Codex
- Branch/Commit: master @ 533bb5ed2 plus local default-off prototype
- Target lane: Qwen3.6-27B-Q3_K_S ROCm, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: the hot GATED_DELTA_NET contract may benefit from fewer warps per block on RDNA4 by reducing per-block pressure even if it increases the number of blocks.
- Mechanism: current GDN uses one warp per output column and `num_warps=4` columns per block. `num_warps=2` or `1` may improve occupancy/scheduling for `S_v=128`, `KDA=false`, `keep_intermediates=false`, `n_seqs=1`.
- Why now: E107 speculative/no-code routes did not improve cold-first q4 KV TPS. E053/E106 keep GDN as the second actionable prompt hotspot after Q3_K staging, and this probe is not a rejected chunk-size or fast-exp repeat.

## Math / Theory

- Assumptions: GDN full-wall share is roughly `8-9%` on the active lane.
- Expected speedup corridor: a `10%` local GDN win is about `+0.8-0.9%` aggregate; a `20%` local win is about `+1.6-1.8%`.
- Failure conditions: extra blocks/launch scheduling dominate; default `4` warps was already near optimum; wall TPS and prompt eval do not improve.

## Implementation Plan

1. Minimal code surface to change: add default-off `GGML_GDN_NUM_WARPS=1|2|4` override in `ggml/src/ggml-cuda/gated_delta_net.cu`.
2. Guard rails: default remains `4`; reject invalid values; do not alter chunk-size, fast-exp, or math.
3. Rollback path: remove the env override if no candidate beats same-session control.

## Benchmark Plan

- Baseline command: E106 no-spec cold-first control `e106-rocm-q3k-control-r1`, plus same-session post-build control if needed.
- Candidate command: same lane with `GGML_GDN_NUM_WARPS=2`, and optionally `1` if `2` is promising or diagnostic.
- Number of runs: one-run gate first; promote only clear positive to 3-run confirmation.
- Artifacts path:
  - `build_logs/agent-workload/e108-rocm-gdn-warps2-r1.*`
  - optional `build_logs/agent-workload/e108-rocm-gdn-warps1-r1.*`

## Metrics

- aggregate completion TPS (wall)
- prompt eval tok/s
- decode eval tok/s
- GDN path trace contract if needed

## Result

- Outcome: reject and revert runtime prototype.
- Delta: post-build control `e108-rocm-gdn-control-r1` measured `11.7604 TPS`; `GGML_GDN_NUM_WARPS=2` measured `11.7408 TPS`; `GGML_GDN_NUM_WARPS=1` measured `11.7258 TPS`.
- Confidence: medium; one-run gates showed no positive signal and both candidates lost against the same-session control.
- Recommendation: keep default `num_warps=4`; do not add a GDN warp-count runtime knob.

## Notes

- Why the hypothesis missed: reducing warps per block likely increased block scheduling overhead or reduced useful occupancy more than it reduced pressure. The existing `4`-warp geometry appears close enough for this `S_v=128`, single-sequence contract.
- Workflow change: future GDN geometry probes need a resource/occupancy argument before build work, not just a plausible "less pressure" story. Closed nearby routes now include chunk-size sweeps, fast-exp style probes, and `num_warps=1/2`.
- The temporary `GGML_GDN_NUM_WARPS` code was removed and the ROCm server was rebuilt after revert.
