# E086 Vulkan 12k Q3_K Correct LOAD_VEC_A=4 Probe

## Metadata

- Experiment ID: E086
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: local master after E082 stride18 kept and E085 tile scout rejected
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, fixed pp7488 prefill gate

## Hypothesis

- Statement: The Q3_K coopmat load path may benefit from a correct 4-value A load variant now that stride18 reduced shared-memory pressure.
- Mechanism: The previous simple `LOAD_VEC_A=4` generator probe did not make the Q3_K branch write two `f16vec2` slots per invocation. This probe adds a Q3_K pair-dequant helper and makes `LOAD_VEC_A=4` cover four values correctly.
- Why now: E082 shows Q3_K shared-memory/resource pressure is movable; reducing load-loop invocation count may stack with stride18.

## Math / Theory

- Assumptions: For Q3_K, one pair index dequants two values; `LOAD_VEC_A=4` should map each vector index to two consecutive pair indices and write two shared-memory slots.
- Expected speedup corridor: +1% to +4% prompt eval if fewer load-loop invocations offset doubled per-invocation dequant work.
- Failure conditions: Extra inlined helper code increases VGPR/ALU pressure, or load-loop overhead is not material.

## Implementation Plan

1. Add a Q3_K `dequant_q3_k_pair()` helper in `mul_mm_funcs.glsl`.
2. Add a `LOAD_VEC_A == 4` Q3_K branch that writes `buf_idx` and `buf_idx + 1`.
3. Add `q3_k` to `load_vec_quant = 4` generation in `vulkan-shaders-gen.cpp`.
4. Guard rails: fixed pp7488 against E082 stride18 r3 `922.62 tok/s`; revert if not promising.

## Benchmark Plan

- Baseline command: E082 stride18 pp7488 r3, `922.62 tok/s`.
- Candidate command: same pp7488 gate after correct Q3_K loadvec4 change.
- Number of runs: 1 gate; 3 only if positive or borderline.
- Artifacts path: `build_logs/agent-workload/e086-*`

## Metrics

- fixed pp7488 prompt throughput
- pipeline stats if promising

## Result

- Outcome: keep.
- Delta: fixed pp7488 improved from E082 stride18 `922.62 tok/s` to `961.82 ± 25.60 tok/s` (`+4.25%`). The first r1 gate was `949.65 tok/s` (`+2.93%`).
- Workload validation: 12k prompt-heavy `triage_diff` run reached `6.6277` aggregate TPS, prompt eval `934.8 tok/s`, decode `40.13 tok/s`, prompt tokens `7489`, errors `0`.
- Resource stats: active `matmul_q3_k_f32_f16acc_aligned_l` dropped from E082 `118 VGPR / 45 SGPR / 20480 B LDS` to `113 VGPR / 45 SGPR / 20480 B LDS`, no scratch.
- Confidence: medium-high for source-level improvement; r3 variance is high, but both fixed pp gate and full workload moved positive with matching prompt-token count.
- Recommendation: keep corrected Q3_K `LOAD_VEC_A=4` on top of stride18 and continue searching in the same active `mul_mm.comp` Q3_K path. Vulkan still trails ROCm on raw prefill: E086 fixed pp7488 is `-12.4%` vs ROCm `1097.66 tok/s`.

Artifacts:

- `build_logs/agent-workload/e086-vulkan-q3-correct-loadvec4-pp7488.md`
- `build_logs/agent-workload/e086-vulkan-q3-correct-loadvec4-pp7488-r3.md`
- `build_logs/agent-workload/e086-vulkan12k-q3-correct-loadvec4-r1.diagnostics.md`
- `build_logs/agent-workload/e086-vulkan-q3-correct-loadvec4-pipeline-stats.log`