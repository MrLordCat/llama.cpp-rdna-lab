# E031 C01 Q4_K Force-X Sub-32KiB Probe

## Metadata

- Experiment ID: E031
- Date: 2026-05-16
- Owner: Codex
- Branch/Commit: local working tree after E030
- Target lane: `Qwen3.6-27B-Q3_K_S`, `tasks=quick`, `task_ids=review_bug,patch_sim`, `ctx=12288`, `batch=6144`, `ubatch=192`, `q4_0/q4_0`, `spec=none`, no-reuse, thinking on

## Hypothesis

- Statement: The secondary C01 `Q4_K` MMQ bucket might improve if `mmq_x` is forced from `96` to `80`, dropping shared memory below the 32 KiB residency boundary.
- Mechanism: Fresh resource trace shows `Q4_K type=12,ncols_max=192` uses `mmq_x=96`, `mmq_y=64`, shared `33664` bytes, regs `200`, and `max_blocks_per_sm=1`. Forcing `mmq_x=80` should reduce the Q8 tile and ids footprint enough to fit below 32 KiB.
- Why now: The dominant `Q3_K` force-x and half-scale branches are closed, but `Q4_K` is still a measurable secondary C01 center (`~6%` of steady `MUL_MAT forward`).

## Math / Theory

- Baseline `Q4_K` resource:
  - `mmq_x=96`, `ncols_max=192`, `ntiles_x=2`
  - shared `33664` bytes
  - `max_blocks_per_sm=1`, waves `4.0`
- Candidate projection:
  - `mmq_x=80`, `ntiles_x=3`
  - shared should drop below 32 KiB and potentially allow `2` blocks/SM
- Expected speedup corridor:
  - Best case: Q4 bucket improves enough to move wall by `~0.2-0.4%`.
  - Failure condition: the `3` vs `2` x-tile penalty dominates the occupancy win.

## Implementation Plan

1. Add temporary env-gated RDNA4 Q4 force-x helper:
   - `GGML_MMQ_RDNA4_Q4_FORCE_MMQ_X`
2. Build `llama-server`.
3. Run same-build control r1 and candidate r1.
4. Revert runtime code if the cheap screen is not positive.

## Benchmark Plan

- Baseline:
  - `c01-e031-q4force-control-r1`
- Candidate:
  - `GGML_MMQ_RDNA4_Q4_FORCE_MMQ_X=80`
  - `c01-e031-q4force-x80-r1`
- Number of runs: `1` cheap screen
- Artifacts path: `build_logs/agent-workload/`

## Metrics

- cold run #1 aggregate completion TPS
- prompt/decode split
- decision stats against same-build control

## Result

- Outcome: regression; code reverted
- Baseline: `9.4522 TPS`
- Candidate: `9.4026 TPS`
- Delta: `-0.0495 TPS` (`-0.52%`)
- Decision stats: bootstrap 95% CI `[-0.0591,-0.0400]` TPS, verdict `negative`
- Prompt eval: `853.885 -> 846.515 tok/s`
- Decode eval: `30.12 -> 30.11 tok/s`

## Notes

- The failure mode matches the analytic risk: the extra x tile count is not paid back by the likely residency improvement.
- Do not continue Q4 force-x below 32 KiB on this lane unless a later trace changes `ncols_max` or tile geometry.
- Runtime code was reverted and `llama-server` was rebuilt.
