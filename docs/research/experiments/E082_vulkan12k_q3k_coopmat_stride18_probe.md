# E082 Vulkan 12k Q3_K Coopmat Stride18 Probe

## Metadata

- Experiment ID: E082
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: local master after E081 rollback
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, fixed pp7488 prefill gate first

## Hypothesis

- Statement: The Q3_K coopmat shared-memory path may benefit from a smaller but nonzero padding than the current stride 20.
- Mechanism: A Q3-only `SHMEM_STRIDE = BK / 2 + 2` gives stride 18 for `BK=32`, reducing LDS versus stride 20 while avoiding the no-padding stride16 regression.
- Why now: E081 showed padding matters; the next useful question is whether the exact `+4` padding is optimal for RDNA4.

## Math / Theory

- Assumptions: Q3_K writes 16 `f16vec2` slots per K tile, so stride 18 remains bounds-safe.
- Expected speedup corridor: +0.5% to +3% prompt eval if lower LDS and different bank spacing help.
- Failure conditions: cooperative-matrix loads prefer stride alignment to multiples of 4, or stride20 is already the best bank pattern.

## Implementation Plan

1. Minimal code surface to change: Q3_K-only `SHMEM_STRIDE` in `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp`.
2. Guard rails: fixed `llama-bench pp7488`; revert on compile/runtime failure or any regression.
3. Rollback path: restore common coopmat `SHMEM_STRIDE (BK / 2 + 4)` and rebuild.

## Benchmark Plan

- Baseline command: E079 forced-regeneration baseline pp7488, `884.96 tok/s`.
- Candidate command: same pp7488 gate after Q3_K stride change.
- Number of runs: 1 gate; 3 only if positive or borderline.
- Artifacts path: `build_logs/agent-workload/e082-*`

## Metrics

- fixed pp7488 prompt throughput
- compile/runtime stability

## Result

- Outcome: keep candidate for continued H31 testing.
- Delta: fixed pp7488 improved from same-session baseline `908.23 +/- 14.66 tok/s` to `922.62 +/- 2.45 tok/s` (`+1.58%`). Candidate r1 was `897.95 tok/s`; candidate stats run was `924.62 tok/s`.
- Confidence: medium. The r3 candidate is above baseline, and resource stats improved, but the baseline r3 variance is high.
- Recommendation: keep the Q3_K-only stride18 change while continuing to search; it is not enough to catch ROCm by itself.

## Key Measurements

| Config | Gate | Throughput | Notes |
| --- | --- | ---: | --- |
| Baseline, same session | `llama-bench pp7488 r3` | `908.23 +/- 14.66 tok/s` | stride20 |
| E082 stride18 | `llama-bench pp7488 r3` | `922.62 +/- 2.45 tok/s` | kept |
| E082 stride18 stats run | `llama-bench pp7488 r1` | `924.62 tok/s` | pipeline stats enabled |
| ROCm control | `llama-bench pp7488 r1` | `1097.66 tok/s` | still target; Vulkan remains about `-15.9%` |

Pipeline stats for `matmul_q3_k_f32_f16acc_aligned_l` under stride18:

- `numUsedVgprs: 118`
- `numUsedSgprs: 45`
- `ldsSizePerLocalWorkGroup: 20480`
- `scratchMemUsageInBytes: 0`

Artifacts:

- `build_logs/agent-workload/e082-vulkan-q3-stride18-pp7488.md`
- `build_logs/agent-workload/e082-vulkan-q3-stride18-pp7488-r3.md`
- `build_logs/agent-workload/e082-vulkan-baseline-pp7488-r3.md`
- `build_logs/agent-workload/e082-vulkan-q3-stride18-pipestats-pp7488.log`
- `build_logs/agent-workload/e082-rocm-control-pp7488-r1.md`