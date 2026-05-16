# E020 H18 C01 Q3_K Half-Scale Compact X96

## Metadata

- Experiment ID: E020
- Date: 2026-05-14
- Owner: Codex
- Branch/Commit: local working tree after E019 reject/revert
- Target lane: `Qwen3.6-27B-Q3_K_S`, `tasks=quick`, `task_ids=review_bug,patch_sim`, `ctx=12288`, `batch=6144`, `ubatch=192`, `q4_0/q4_0`, `spec=none`, no-reuse

## Hypothesis

- Statement: RDNA4 Q3_K prefill can improve by packing the per-row Q3 scales in shared memory as `half` and using a compact Q3-only MMA row stride.
- Mechanism: Current E015 Q3_K MMQ uses `35712` bytes shared for `mmq_x=96, mmq_y=64`, so resource telemetry reports `max_blocks_per_sm=1`. A compact Q3 scale layout projects `32640` bytes, below `32768`, while preserving `mmq_x=96` and the `2` x-tile count for `ncols=192`.
- Why now: E018 and E019 rejected extra register/fused-load work. This candidate attacks the limiting shared-memory threshold instead of adding more work to the existing hot lanes.

## Math / Theory

- Source artifact: `build_logs/agent-workload/c01-e020-q3-halfscale-compact-theory.md`
- Baseline resource:
  - shared: `35712` bytes (`54.49%` of `65536`)
  - registers: `160`
  - block threads: `128`
  - `max_blocks_per_sm=1`, waves `4.0`
- Current shared split:
  - Q3 x tile: `21504` bytes
  - Q8 y tile: `13824` bytes
  - misc: `384` bytes
- Candidate projection:
  - Q3 row stride: `84 -> 72` ints
  - Q3 x tile: `21504 -> 18432` bytes
  - total shared: `35712 -> 32640` bytes
  - x tile count for `ncols=192`: unchanged at `2`
- Expected speedup corridor:
  - Best case: more block residency hides latency in Q3 MMQ, improving target bucket by `1-4%`.
  - Worst case: half-to-float conversion and changed row padding/bank behavior dominate, causing regression.

## Implementation Plan

1. Add RDNA4/HIP-only compact Q3 stride for `GGML_TYPE_Q3_K`; keep IQ2 paths on the old stride.
2. In `load_tiles_q3_K`, for RDNA4 store `d * scale` as `half` instead of `float`.
3. Add a Q3-specific MMA vec-dot helper that reads half scales with `__half2float`.
4. Keep non-RDNA4 and non-Q3 behavior unchanged.

## Benchmark Plan

- Baseline:
  - current best reference: `c01-e015-rdna4-y64w4-r3` (`9.6080 TPS`)
  - resource/trace reference: `c01-e015-rdna4-y64w4-trace-r1`
- Candidate:
  - `c01-e020-q3-halfscale-compact-r1`
  - If r1 is not clearly negative, run `c01-e020-q3-halfscale-compact-trace-r1` with `--trace-preset kernel-full`.
- Decision rule:
  - Keep only if runtime is positive and target `MMQ type=11 ncols_max=192` / `MUL_MAT forward` improve.
  - Revert if cheap screen is negative or trace target regresses.

## Metrics

- aggregate completion TPS
- prompt eval TPS
- resource telemetry for `type=11,ncols_max=192`
- `CUDA_NODE op=MUL_MAT kind=forward`
- `MMQ type=11 ncols_max=192`

## Result

- Outcome: research-positive but runtime-inconclusive; runtime code reverted
- Runtime delta: `9.6080 -> 9.6017 TPS` (`-0.07%`) against E015 r3 reference
- Decision stats: bootstrap 95% CI `[-0.0380, +0.0239]` TPS, verdict `inconclusive`
- Target trace delta against E015 trace:
  - `nbytes_shared`: `35712 -> 32640`
  - `max_blocks_per_sm`: `1 -> 2`
  - occupancy: `6.25% -> 12.50%`
  - waves: `4.00 -> 8.00`
  - regs: `160 -> 158`
  - `MMQ type=11 ncols_max=192`: `9551.391 -> 9451.261 ms` (`-100.130 ms`, about `-1.05%`)
  - total `MMQ`: `10381.647 -> 10300.173 ms` (`-81.474 ms`)
- Recommendation: do not keep as default; preserve as a future combined layout/scheduling candidate.

## Notes

- Candidate artifacts:
  - `build_logs/agent-workload/c01-e020-q3-halfscale-compact-theory.md`
  - `build_logs/agent-workload/c01-e020-q3-halfscale-compact-r1b.csv`
  - `build_logs/agent-workload/c01-e020-q3-halfscale-compact-trace-r1b.server.log`
  - `build_logs/agent-workload/c01-e020-q3-halfscale-compact-r3.csv`
- Surprise:
  - The intended shared-memory residency effect was real, but runtime did not move. The target MMQ bucket improved while other CUDA nodes and pre-sync overhead worsened enough to cancel it.
  - The first trace (`trace-r1`) accidentally used the old host-side dynamic shared size (`35712`), so it did not test the occupancy theory. The valid trace is `trace-r1b`.
- Follow-up action:
  - Do not re-test the same half-scale compact layout alone.
  - Future work could combine compact Q3 shared layout with a scheduling/pre-sync fix or a layout that avoids the non-MMQ slowdowns.
  - Runtime code was reverted and `llama-server` was rebuilt.
