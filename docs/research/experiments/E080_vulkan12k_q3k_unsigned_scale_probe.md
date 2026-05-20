# E080 Vulkan 12k Q3_K Unsigned Scale Probe

## Metadata

- Experiment ID: E080
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: local master after E079 rollback and Vulkan shader dependency fix
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, fixed pp7488 prefill gate first, then 12k cold workload only if promising

## Hypothesis

- Statement: The Q3_K coopmat dequant scale path may pay extra instructions/register pressure by converting a 0..63 scale nibble to `int8_t` before subtracting 32.
- Mechanism: Keeping the packed scale as `uint` and computing `float(us) - 32.0f` preserves the same value range while avoiding signed 8-bit conversion in the active `mul_mm.comp` Q3_K shader.
- Why now: E079 rejected f16 arithmetic but confirmed helper-shader changes are now rebuilt correctly through CMake dependencies.

## Math / Theory

- Assumptions: Q3_K scale values are assembled from 6 bits (`0..63`), so `float(uint_us) - 32.0f` is equivalent to `float(int8_t_us - 32)`.
- Expected speedup corridor: +0.5% to +2% prompt eval if sign-extension/register pressure matters.
- Failure conditions: Compiler already canonicalizes the old form, or `uint` arithmetic increases pressure.

## Implementation Plan

1. Minimal code surface to change: Q3_K branch in `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_funcs.glsl` only.
2. Guard rails: fixed `llama-bench pp7488` against same-session baseline `884.96 tok/s`; revert if not above baseline.
3. Rollback path: restore `int8_t us` expression and rebuild.

## Benchmark Plan

- Baseline command: E079 forced-regeneration baseline pp7488, `884.96 tok/s`.
- Candidate command: same pp7488 gate after shader change.
- Number of runs: 1 gate; 3 only if positive or borderline.
- Artifacts path: `build_logs/agent-workload/e080-*`

## Metrics

- fixed pp7488 prompt throughput
- optional active 12k workload split if pp gate wins

## Result

- Outcome: reject and revert.
- Delta: fixed pp7488 gate regressed from `884.96 tok/s` baseline to `881.07 tok/s` candidate (`-0.44%`).
- Confidence: medium-high; single-run prompt-only gate was close but below baseline, and expected upside was small.
- Recommendation: keep the original `int8_t us` Q3_K scale expression.

## Key Measurements

| Config | Gate | Throughput | Decision |
| --- | --- | ---: | --- |
| Baseline, E079 forced shader regen | `llama-bench pp7488` | `884.96 tok/s` | baseline |
| E080 unsigned scale | `llama-bench pp7488` | `881.07 tok/s` | reject |

Artifacts:

- `build_logs/agent-workload/e080-vulkan-q3-unsigned-scale-pp7488.md`