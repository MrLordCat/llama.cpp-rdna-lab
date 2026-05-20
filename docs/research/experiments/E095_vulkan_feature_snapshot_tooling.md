# E095 Vulkan Feature Snapshot Tooling

## Metadata

- Experiment ID: E095
- Date: 2026-05-20
- Owner: Copilot
- Type: workflow/tooling + device capability gate
- Target lane: H31 Vulkan Q3_K prefill research on RX 9070 XT / AMD proprietary LLPC

## Hypothesis

- Statement: Before chasing coopmat2 or shader-compiler leads, capture compiler and device feature support in a reproducible no-build artifact.
- Mechanism: `glslc` feature-test support and `vulkaninfo` runtime extension support can diverge; both are needed before considering coopmat route work.
- Why now: E094 confirmed the 32k gap remains prefill-side, while local gates closed helper-only, invalid tile, same-as-base tile, and naive BK-depth ideas.

## Math / Theory

- Assumptions: `glslc` support alone is insufficient; runtime device extensions/features decide which backend paths can actually execute.
- Expected speedup corridor: none; this is a gate/tooling snapshot.
- Failure conditions: Do not treat compiler-only coopmat2 support as a runnable AMD path.

## Implementation Plan

1. Minimal code surface to change: add `scripts/research/vulkan_feature_snapshot.py`.
2. Guard rails: no build, no benchmark, no source route changes.
3. Rollback path: remove the tool if it becomes redundant with a better environment probe.

## Benchmark Plan

- Baseline command: `python scripts/research/vulkan_feature_snapshot.py | tee build_logs/agent-workload/e095-vulkan-feature-snapshot.md`
- Candidate command: not applicable.
- Number of runs: 1.
- Artifacts path: `build_logs/agent-workload/e095-vulkan-feature-snapshot.md`

## Metrics

Compiler feature tests:

| Feature test | Status |
| --- | --- |
| `coopmat` | OK |
| `coopmat2` | OK |
| `integer_dot` | OK |
| `bfloat16` | OK |

Runtime/device signals:

| Signal | Status |
| --- | --- |
| GPU | AMD Radeon RX 9070 XT |
| Driver | AMD proprietary driver `26.3.1 (LLPC)` |
| Vulkan API | `1.4.344` |
| `VK_KHR_cooperative_matrix` | yes |
| `VK_NV_cooperative_matrix2` | no |
| `subgroupSizeControl` | yes |
| `subgroupSize` | `64` |
| Vulkan 1.3 subgroup range | `32..64` |

## Result

- Outcome: keep tool and artifact.
- Delta: no TPS claim.
- Confidence: high for local environment capability snapshot; `glslc` and `vulkaninfo` agree that KHR coopmat is available, while runtime NV coopmat2 is not.
- Recommendation: do not pursue `mul_mm_cm2`/NV coopmat2 as an AMD RX 9070 XT acceleration route on this driver. Continue H31 on the active KHR coopmat `mul_mm.comp` route or on non-coopmat mechanisms with explicit gates.

## Notes

- Surprises: `glslc` can compile the coopmat2 feature test, but the AMD device does not expose `VK_NV_cooperative_matrix2`.
- Follow-up action: add this snapshot command to the future-agent readiness flow for Vulkan performance work.
