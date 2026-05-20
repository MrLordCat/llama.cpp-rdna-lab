# E083 Vulkan 12k Q3_K Coopmat Stride19 Probe

## Metadata

- Experiment ID: E083
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: local master after E082 stride18 kept
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, fixed pp7488 prefill gate

## Hypothesis

- Statement: A neighboring Q3_K coopmat shared-memory stride may beat E082 stride18 by improving LDS bank spacing.
- Mechanism: Stride19 keeps nonzero padding, remains below the original stride20 LDS footprint, and changes row spacing relative to LDS banks.
- Why now: E082 found a small but measurable resource/perfill improvement with stride18.

## Math / Theory

- Assumptions: Q3_K writes 16 `f16vec2` slots per K tile, so stride19 is bounds-safe.
- Expected speedup corridor: +0.5% to +2% over stride18 if bank spacing dominates.
- Failure conditions: extra LDS over stride18 or odd stride harms cooperative-matrix loads.

## Implementation Plan

1. Minimal code surface to change: Q3_K-only `SHMEM_STRIDE` in `mul_mm.comp`, from `BK / 2 + 2` to `BK / 2 + 3`.
2. Guard rails: fixed pp7488 r1 against E082 stride18 r3 `922.62 tok/s`; revert to stride18 if not promising.
3. Rollback path: restore Q3_K stride18 and rebuild.

## Benchmark Plan

- Baseline command: E082 stride18 pp7488 r3, `922.62 tok/s`.
- Candidate command: same pp7488 gate after Q3_K stride19 change.
- Number of runs: 1 gate; 3 only if positive or borderline.
- Artifacts path: `build_logs/agent-workload/e083-*`

## Metrics

- fixed pp7488 prompt throughput

## Result

- Outcome: reject and revert to E082 stride18.
- Delta: fixed pp7488 gate regressed from E082 `922.62 tok/s` r3 to `633.65 tok/s` r1 (`-31.3%`).
- Confidence: high; the regression is too large to continue.
- Recommendation: avoid odd Q3_K coopmat strides in this path unless a future correctness/resource trace explains the failure.

## Key Measurements

| Config | Gate | Throughput | Decision |
| --- | --- | ---: | --- |
| E082 stride18 | `llama-bench pp7488 r3` | `922.62 tok/s` | kept baseline |
| E083 stride19 | `llama-bench pp7488 r1` | `633.65 tok/s` | reject |

Artifacts:

- `build_logs/agent-workload/e083-vulkan-q3-stride19-pp7488.md`