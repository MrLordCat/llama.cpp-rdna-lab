# E084 Vulkan 12k Q3_K Coopmat Stride22 Probe

## Metadata

- Experiment ID: E084
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: local master after E083 rollback to E082 stride18
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, fixed pp7488 prefill gate

## Hypothesis

- Statement: If E082's win came from LDS bank spacing rather than lower LDS footprint, a larger even non-multiple-of-4 stride may also help.
- Mechanism: Q3_K stride22 (`BK / 2 + 6`) keeps an even stride with different bank spacing than stride20, while increasing shared memory versus both stride18 and stride20.
- Why now: Stride18 improved, stride16 and stride19 regressed; stride22 separates bank-pattern effects from LDS-size effects.

## Math / Theory

- Assumptions: Q3_K writes 16 `f16vec2` slots per K tile, so stride22 is bounds-safe and under the 32 KiB shared-memory limit.
- Expected speedup corridor: +0.5% to +2% over stride18 only if bank spacing dominates.
- Failure conditions: Higher LDS footprint cancels any bank-pattern gain.

## Implementation Plan

1. Minimal code surface to change: Q3_K-only `SHMEM_STRIDE` in `mul_mm.comp`, from `BK / 2 + 2` to `BK / 2 + 6`.
2. Guard rails: fixed pp7488 r1 against E082 stride18 r3 `922.62 tok/s`; revert to stride18 if not promising.
3. Rollback path: restore Q3_K stride18 and rebuild.

## Benchmark Plan

- Baseline command: E082 stride18 pp7488 r3, `922.62 tok/s`.
- Candidate command: same pp7488 gate after Q3_K stride22 change.
- Number of runs: 1 gate; 3 only if positive or borderline.
- Artifacts path: `build_logs/agent-workload/e084-*`

## Metrics

- fixed pp7488 prompt throughput

## Result

- Outcome: reject and revert to E082 stride18.
- Delta: fixed pp7488 gate reached `894.36 tok/s`, below E082 stride18 r3 `922.62 tok/s` (`-3.06%` vs stride18).
- Confidence: high enough to close this neighbor; no 3-run confirmation needed.
- Recommendation: keep stride18 as the best tested Q3_K coopmat stride so far.

## Key Measurements

| Config | Gate | Throughput | Decision |
| --- | --- | ---: | --- |
| E082 stride18 | `llama-bench pp7488 r3` | `922.62 tok/s` | kept baseline |
| E084 stride22 | `llama-bench pp7488 r1` | `894.36 tok/s` | reject |

Artifacts:

- `build_logs/agent-workload/e084-vulkan-q3-stride22-pp7488.md`