# E094 Vulkan/ROCm 32k Spec-None Controls

## Metadata

- Experiment ID: E094
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: local master after E093 tooling
- Target lane: Qwen3.6-27B-Q3_K_S, 32k repo-snapshot control, b=5120, ub=1024, q4_0/q4_0 KV, FlashAttention on, thinking on, no reuse, `--spec-type none`

## Hypothesis

- Statement: A fresh 32k spec-none control can verify whether the post-E093 Vulkan state changes the broader 32k gap against ROCm.
- Mechanism: Same task/model/ctx/batch/ubatch/KV controls isolate backend prefill and decode behavior without speculative decoding coverage effects.
- Why now: E093 corrected the tile-profile baseline and the workspace had completed fresh `vscode-vulkan32k-control-r1` and `vscode-rocm32k-control-r1` runs.

## Math / Theory

- Assumptions: Compare only same-lane cold runs with matching task, prompt size, max tokens, no reuse, thinking on, KV format, and spec mode.
- Expected speedup corridor: Closing Vulkan to ROCm at this lane would require a large prompt-side improvement; decode is already faster on Vulkan.
- Failure conditions: Do not use these controls to rank speculative profiles or E076 ngram-mod runs directly.

## Implementation Plan

1. Minimal code surface to change: none.
2. Guard rails: document as control-only; no promotion, no candidate claim.
3. Rollback path: not applicable.

## Benchmark Plan

- Baseline command: completed VS Code task `bench: vulkan q3 32k control` with `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`.
- Candidate command: completed matching ROCm task using `build-rocm-vec/bin/llama-server.exe`.
- Number of runs: 1 each.
- Artifacts path: `build_logs/agent-workload/vscode-vulkan32k-control-r1.*`, `build_logs/agent-workload/vscode-rocm32k-control-r1.*`

## Metrics

| Backend | Aggregate TPS | Prompt eval TPS | Decode eval TPS | Prompt tokens | Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vulkan force-large | `9.7504` | `888.75` | `33.02` | `7673` | `0` |
| ROCm | `10.9702` | `1149.81` | `28.46` | `7673` | `0` |

## Result

- Outcome: control checkpoint, no candidate.
- Delta: Vulkan trails ROCm by `-11.12%` aggregate and `-22.70%` prompt eval, while Vulkan decode is `+16.02%` faster.
- Confidence: medium for control shape; both are r1 cold controls and match lane settings, but no r3 confirmation was needed because this is not a candidate promotion.
- Recommendation: keep E076 conclusion. The 32k gap remains prompt/prefill-side; no-code/spec-none control does not change the active H31 direction.

## Notes

- Surprises: none; numbers are consistent with Vulkan having stronger decode but weaker Q3_K prompt prefill than ROCm.
- Follow-up action: continue only with no-build gates or source-level mechanisms that can plausibly move the active Q3_K prefill hotspot, not more tile/env sweeps.
