# E021 H13 C01 Dense Q3 MMQ Staging

## Metadata

- Experiment ID: E021
- Date: 2026-05-14
- Owner: Codex
- Target lane: `Qwen3.6-27B-Q3_K_S`, `tasks=quick`, `task_ids=review_bug,patch_sim`, `ctx=12288`, `batch=6144`, `ubatch=192`, `q4_0/q4_0`, `spec=none`, no-reuse
- Hypothesis ID: H13

## Hypothesis

- Statement: The existing RDNA4 MMQ staging path may help dense C01 Q3_K prefill if enabled narrowly for the active `type=11,ncols=192,mmq_x=96,mmq_y=64` bucket.
- Mechanism: Current dense C01 Q3_K uses a non-staged loop that reloads the Q3 tile for each K block. The staged path double-buffers the Q3 tile in LDS and can move the next Q3 tile load earlier in the loop.
- Why now: E020 showed that Q3 shared-memory layout is a real limiting axis, but its occupancy win did not translate to aggregate TPS. Dense staging is a different Q3 load-scheduling probe and reuses existing guarded code.

## Math / Theory

- Current E015 Q3 resource: `mmq_x=96`, `mmq_y=64`, shared `35712` bytes.
- Shared split from E020 theory:
  - Q3 x tile: `21504` bytes
  - Q8 y tile: `13824` bytes
  - misc/ids: `384` bytes
- Dense staged projection:
  - shared = `2 * 21504 + 13824 + 384 + 4 = 57220` bytes
  - fits within `65536` byte SMEM limit
  - stays above `32 KiB`, so expected occupancy remains one block/SM
  - tile count for `ncols=192` remains `2`
- Gate: worth one r1 because it changes Q3 load scheduling without changing tile count. It must improve target MMQ timing, not only wall noise.

## Implementation Plan

1. Add an env-gated dense-Q3 path to the existing RDNA4 staging gate:
   - `GGML_RDNA4_DENSE_Q3_MMQ_STAGING=1`
   - active only for `type == GGML_TYPE_Q3_K`, `args.ids_dst == nullptr`, RDNA4/HIP
2. Build ROCm server.
3. Run r1 C01 screen with the env flag.
4. If r1 is promising, run kernel trace; otherwise revert code.

## Benchmark Plan

- Reference: current best `c01-e015-rdna4-y64w4-r3 = 9.6080 TPS`
- Candidate:
  - `c01-e021-dense-q3-staging-r1`
  - optional `c01-e021-dense-q3-staging-trace-r1`

## Result

- Outcome: reject; runtime code reverted
- Runtime screen:
  - reference: `c01-e015-rdna4-y64w4-r3 = 9.6080 TPS`
  - candidate: `c01-e021-dense-q3-staging-r1 = 8.6216 TPS`
  - delta: `-10.27%`
- Activation trace:
  - artifact: `build_logs/agent-workload/c01-e021-dense-q3-staging-activation-r1.server.log`
  - confirmed `rdna4_staging_req=1` and `rdna4_staging_eff=1` for `type=11,ncols_max=192`
  - trace compare is one-task vs two-task, so totals are not directly comparable; per-call target timing is the useful signal
  - `MMQ type=11 ncols_max=192` average timing worsened by about `25.9%` (`0.447 ms -> 0.563 ms`)
  - top target shapes worsened similarly, including `MUL_MAT ne=(17408,192,1,1)` average `0.671 ms -> 0.842 ms`
- Decision:
  - reject dense Q3 staging in the current form
  - do not promote `GGML_RDNA4_DENSE_Q3_MMQ_STAGING`
  - keep existing MoE-only staging gate unchanged

## Notes

- The analytic gate was correct that dense Q3 staging fits SMEM at `mmq_x=96`, but fitting is not enough: the staged loop adds LDS pressure/barrier work and worsens the active Q3 bucket.
- Runtime code was reverted and `llama-server` was rebuilt after rollback.
