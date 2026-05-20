# E081 Vulkan 12k Q3_K Coopmat Stride16 Probe

## Metadata

- Experiment ID: E081
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: local master after E080 rollback
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, fixed pp7488 prefill gate first

## Hypothesis

- Statement: The active Q3_K coopmat shader may be LDS-limited enough that removing the extra shared-memory padding improves occupancy or residency.
- Mechanism: For Q3_K, `BK=32` and the current coopmat `SHMEM_STRIDE` is `BK / 2 + 4 = 20`. A Q3-only stride of `BK / 2 = 16` reduces `buf_a + buf_b` shared memory by about 4 KiB per workgroup.
- Why now: E078 showed the active route is `mul_mm.comp` Q3_K coopmat, and E079/E080 rejected arithmetic micro-changes.

## Math / Theory

- Assumptions: The Q3_K branch writes exactly 16 `f16vec2` slots per K tile for `BK=32`, so stride 16 is bounds-safe.
- Expected speedup corridor: +1% to +3% prompt eval if LDS occupancy wins exceed any bank-conflict cost.
- Failure conditions: The +4 padding avoids important LDS bank conflicts or cooperative-matrix load alignment constraints.

## Implementation Plan

1. Minimal code surface to change: Q3_K-only `SHMEM_STRIDE` in `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp`.
2. Guard rails: fixed `llama-bench pp7488` against same-session baseline `884.96 tok/s`; revert if not above baseline.
3. Rollback path: restore common coopmat `SHMEM_STRIDE (BK / 2 + 4)` and rebuild.

## Benchmark Plan

- Baseline command: E079 forced-regeneration baseline pp7488, `884.96 tok/s`.
- Candidate command: same pp7488 gate after Q3_K stride change.
- Number of runs: 1 gate; 3 only if positive or borderline.
- Artifacts path: `build_logs/agent-workload/e081-*`

## Metrics

- fixed pp7488 prompt throughput
- optional pipeline stats if pp gate is positive

## Result

- Outcome: reject and revert.
- Delta: fixed pp7488 gate regressed from `884.96 tok/s` baseline to `802.36 tok/s` candidate (`-9.33%`).
- Confidence: high; the regression is large enough that no 3-run confirmation is useful.
- Recommendation: keep the current coopmat `SHMEM_STRIDE (BK / 2 + 4)` for Q3_K.

## Key Measurements

| Config | Gate | Throughput | Decision |
| --- | --- | ---: | --- |
| Baseline, E079 forced shader regen | `llama-bench pp7488` | `884.96 tok/s` | baseline |
| E081 Q3_K stride16 | `llama-bench pp7488` | `802.36 tok/s` | reject |

Artifacts:

- `build_logs/agent-workload/e081-vulkan-q3-stride16-pp7488.md`