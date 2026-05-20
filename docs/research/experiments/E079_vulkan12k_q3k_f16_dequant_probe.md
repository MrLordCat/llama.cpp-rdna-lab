# E079 Vulkan 12k Q3_K F16 Dequant Probe

## Metadata

- Experiment ID: E079
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: local master after E078 rollback
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, ctx=12288, b=4096, ub=1024, q4_0/q4_0 KV, FlashAttention on, thinking on, no reuse, no prime, `--spec-type none`, task `quick/triage_diff`

## Hypothesis

- Statement: The active Q3_K coopmat dequant path may spend unnecessary f32 ALU/register pressure computing values that are immediately stored as `f16vec2` in shared memory.
- Mechanism: Switching the Q3_K `mul_mm_funcs.glsl` dequant multiply/subtract to `FLOAT_TYPE` / `FLOAT_TYPEV2` lets the active f16 coopmat shader do the dequant stage in f16 before shared storage.
- Why now: E078 showed active `matmul_q3_k_f32_f16acc_aligned_l` is the hot route, while MMQ/Q8_1 and load-vector probes did not improve wall time.

## Math / Theory

- Assumptions: Since the shared-memory value is f16, f16 dequant arithmetic may be acceptable for this quantized path and may reduce register pressure.
- Expected speedup corridor: +1% to +4% prompt eval if f32 dequant ALU/register pressure is material.
- Failure conditions: f16 arithmetic changes logits too much, compiler already narrows the operations, or f16 conversions increase pressure.

## Implementation Plan

1. Minimal code surface to change: Q3_K branch in `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_funcs.glsl` only.
2. Guard rails: same 12k cold gate; pipeline stats if the speed result is promising or surprising.
3. Rollback path: restore Q3_K branch to f32 `dl`/`vec2` arithmetic and rebuild.

## Benchmark Plan

- Baseline command: E078 Vulkan 12k control, `6.4679` aggregate / `905.64 tok/s` prompt.
- Candidate command: same command after shader change.
- Number of runs: 1 gate; 3 only if positive or borderline.
- Artifacts path: `build_logs/agent-workload/e079-*`

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- error rate
- prefill/decode split

## Result

- Outcome: reject and revert shader probe; keep CMake dependency fix.
- Delta: fixed pp7488 gate regressed from `884.96 tok/s` baseline to `872.01 tok/s` candidate (`-1.46%`).
- Confidence: high for rejection of this micro-change. The first workload run reported `12.3347` wall TPS, but it had only `3066` prompt tokens versus the active baseline's `7489`, so it is not comparable.
- Recommendation: keep Q3_K dequant arithmetic in f32/vec2 before f16 shared-memory storage. For future shader-helper edits, rely on the CMake `*.glsl` dependency fix so `mul_mm_funcs.glsl` changes regenerate the SPIR-V artifacts automatically.

## Notes

- Surprises: `mul_mm.comp.cpp` originally depended on `mul_mm.comp` and generator sources, but not on included helper files such as `mul_mm_funcs.glsl`; touching `mul_mm.comp` was needed to force a fair test before the dependency fix.
- Follow-up action: continue H31 on Q3_K coopmat route, but skip f16 dequant arithmetic as a candidate.

## Key Measurements

| Config | Gate | Throughput | Notes |
| --- | --- | ---: | --- |
| Baseline, forced shader regen | `llama-bench pp7488` | `884.96 tok/s` | `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`, b4096/ub1024, q4_0/q4_0 |
| E079 f16 dequant, forced shader regen | `llama-bench pp7488` | `872.01 tok/s` | rejected |
| E079 first workload run | `quick/triage_diff` | `12.3347 TPS`, prompt `856.00 tok/s` | invalid A/B due `3066` prompt tokens |

Artifacts:

- `build_logs/agent-workload/e079-vulkan-baseline-forcedregen-pp7488.md`
- `build_logs/agent-workload/e079-vulkan-q3-f16dequant-forcedregen-pp7488.md`
- `build_logs/agent-workload/e079-vulkan12k-q3-f16-dequant-r1.diagnostics.md`