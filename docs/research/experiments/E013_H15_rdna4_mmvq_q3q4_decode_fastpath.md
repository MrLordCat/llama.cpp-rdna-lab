# E013 H15 RDNA4 MMVQ Q3/Q4 Decode Fast Path

## Metadata

- Experiment ID: E013
- Date: 2026-05-13
- Owner: Codex
- Branch/Commit: local dirty `master`
- Target lane: C01/C02 Qwen3.6-27B dense decode lane, `ctx=12288`, `b=6144`, `ub=192`, KV `q4_0/q4_0`, no-reuse, thinking on

## Hypothesis

- Statement: An RDNA4-scoped MMVQ Q3_K/Q4_K decode launch variant can reduce decode matvec cost on the active Qwen lane without increasing VRAM.
- Mechanism: The current Qwen-hot `small_k` policy improved MMVQ route cost, but the C02 bucket still contributes measurable decode time. A narrow compile-time launch variant or policy knob may improve occupancy/work distribution for `ncols_dst=1`.
- Why now: MTP was memory-expensive and slower on this machine; MMVQ changes target the same user-visible decode speed without extra model memory.

## Math / Theory

- Assumptions: Candidate affects only Q3_K/Q4_K MMVQ decode calls and does not shift correctness or graph routes outside the target bucket.
- Expected speedup corridor: +2% to +8% on decode-heavy C02 hotspot; lower wall gain on prompt-heavy tasks.
- Failure conditions: target bucket not activated, extra template variant increases register pressure, or runtime gain is lost in non-MMVQ work.

## Implementation Plan

1. Minimal code surface to change: `ggml/src/ggml-cuda/mmvq.cu`.
2. Guard rails: preserve existing small_k overrides and keep the default change RDNA4/Q3_K/ncols_dst=1 scoped.
3. Rollback path: revert the local `GGML_TYPE_Q3_K: return 2` hunk if runtime/hotspot evidence is negative.

## Benchmark Plan

- Baseline command: fresh C01/C02 resource trace with current `build-rocm-vec/bin/llama-server.exe`.
- Candidate command: same lane after rebuilding `build-rocm-vec`, with candidate env knob enabled if needed.
- Number of runs: `runs=1` for iteration; `runs=3` only if positive/borderline.
- Artifacts path: `build_logs/agent-workload/e013-h15-*`.

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- error rate
- MMVQ `type=11/q3_K ncols_dst=1` timing
- `MUL_MAT forward` timing

## Result

- Outcome: keep.
- Runtime delta: paired current control `9.1629 TPS` -> candidate `9.3847 TPS`, `+2.42%`.
- Bootstrap: candidate-control mean delta `+0.2216 TPS`, 95% CI `[+0.2019, +0.2442]`, verdict positive.
- Trace delta: initial trace baseline `6.3251 TPS` -> candidate `6.5513 TPS`, `+3.58%`.
- Hotspot evidence: trace compare showed `CUDA_NODE op=MUL_MAT kind=forward -252.983 ms`, `MMQ -108.954 ms`, and `MMQ type=11 ncols_max=192 -96.409 ms`.
- Recommendation: keep the RDNA4 Q3_K `ncols_dst=1` launch policy at `nwarps=2`; do not broaden to Q4_K further without a separate Q4-heavy lane.

## Notes

- Initial env/template attempt failed to compile because HIP could not use the extra template knob cleanly in `__launch_bounds__`; reverted that shape.
- The kept patch is smaller: `calc_nwarps()` returns `2` for `GGML_TYPE_Q3_K` on RDNA4 `ncols_dst=1`.
- Q4_K was not changed because it already uses the RDNA4 `nwarps=8` whitelist and the measured gain came from the Q3-heavy lane.
- Follow-up probe `Q3_K nwarps=4` was rejected: `9.3847 -> 9.2136 TPS`, `-1.82%`, bootstrap CI `[-0.2072, -0.1335]` TPS.
- Follow-up action: run a dedicated Q4_K decode lane before touching Q4_K policy.
