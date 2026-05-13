# E010 H13 RDNA4 MoE/MMQ LDS Staging

## Metadata

- Experiment ID: E010
- Date: 2026-05-13
- Owner: Copilot
- Branch/Commit: local `master` after `2b6f47c02`
- Target lane: Qwen3.6 MoE35B `b=1024`, `ub=1024`, ROCm RX 9070 XT (`gfx1201`)

## Hypothesis

- Statement: A guarded RDNA4-specific adaptation of Stormrage's MoE MMQ staging idea can improve MoE prefill without changing default dense/runtime behavior.
- Mechanism: add an opt-in MMQ variant or route knob that changes LDS staging / padding / occupancy for RDNA4 MoE prefill, then enable it only when `GGML_CUDA_CC_IS_RDNA4(cc)` and an experiment env var are set.
- Why now: Stormrage `RDNA2_MATMUL_OPT_V1` showed that MoE prefill can be sensitive to MMQ LDS layout. Local extra `b=1024,ub=1024` MoE results are already strong, but TurboKV still trails q4 on decode and may still have prefill headroom.

## Math / Theory

- Assumptions: the local MoE prefill route uses MMQ or a related quantized matmul path where LDS staging/occupancy can matter; RDNA4 behavior differs enough from RDNA2 to require a separate gate.
- Expected speedup corridor: +3% to +10% MoE prefill if the active route is LDS/occupancy limited; 0% if current route is WMMA/F16 or already optimal.
- Failure conditions: no active MMQ route for tested shapes, increased register pressure, worse dense negative control, or decode regression.

## Implementation Plan

1. Minimal code surface to change: inspect `ggml/src/ggml-cuda/mmq.cuh` and dispatch helpers; add the smallest opt-in RDNA4 gate that can A/B the candidate path.
2. Guard rails: env var off by default; RDNA4-only runtime gate; do not change default q4/TKV behavior.
3. Rollback path: remove the candidate branch/gate if A/B is neutral or negative.

## Benchmark Plan

- Baseline command: `llama-bench` MoE35B `q4_0/q4_0` and `turbo4_0/turbo2_0`, `p=512,2048,4096`, `n=128`, `b=1024`, `ub=1024`, `r=3`.
- Candidate command: same commands with the RDNA4 experiment env var enabled.
- Number of runs: use `r=3` for llama-bench rows; use focused `r=1` probes during route discovery.
- Artifacts path: `build_logs/agent-workload/e010-rdna4-moe-mmq-*`.

## Metrics

- prefill tok/s for pp512/pp2048/pp4096
- decode tok/s for tg128
- dense negative control
- build success and runtime stability

## Result

- Outcome: `iterate` (current staged LDS layout is not admissible on this lane)
- Delta (measured, `r=1`, `b=1024`, `ub=1024`, `q4_0/q4_0`):
	- `pp512`: `2868.82 -> 2887.18` (`+0.64%`)
	- `pp2048`: `3634.31 -> 3644.80` (`+0.29%`)
	- `pp4096`: `3614.84 -> 3611.46` (`-0.09%`)
	- `tg128`: `97.96 -> 102.14` (`+4.26%`, single-run noise-prone)
- Confidence: medium for route/fallback facts, low-medium for performance delta (`r=1`)
- Recommendation:
	- keep safety fallback (prevents runtime abort when `GGML_RDNA4_MOE_MMQ_STAGING=1` cannot fit SMEM);
	- treat current staging variant as NO-GO for active lane until shared-memory footprint is reduced enough to keep `rdna4_staging=1` effective.

## Route Verification

- `GGML_TRACE_CUDA_MUL_MAT_ID_ROUTE` in verbose run confirms MoE expert matmuls are dispatched as `route=mul_mat_q_direct`.
- `GGML_TRACE_MMQ_PATH` with `GGML_RDNA4_MOE_MMQ_STAGING=1` shows `rdna4_staging=0` in `mul_mat_q_case` (effective fallback path), i.e. staged LDS variant is currently rejected by MMQ SMEM constraints.

## Runtime Incident + Fix

- Initial candidate run with `GGML_RDNA4_MOE_MMQ_STAGING=1` aborted at `mmq_x_best=0` (`mmq.cuh` default branch).
- Applied fix: in `mul_mat_q_case`, if staged mode yields no admissible `mmq_x`, automatically retry selection with staging disabled and continue safely.

## Notes

- Stormrage's RDNA2 accelerator is not copied directly. This experiment tests whether the underlying idea has an RDNA4-specific version.
- Current implementation status: guarded candidate remains opt-in, but effective staging is auto-disabled on tested shapes (`ncols_max=128`) due SMEM fit limits.