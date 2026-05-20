# E085 Vulkan 12k Q3_K Stride18 Tile Scout

## Metadata

- Experiment ID: E085
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: local master after E082 stride18 kept
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, fixed pp7488 prefill gate

## Hypothesis

- Statement: E082's lower LDS/VGPR footprint may make previously weak legal AMD large-matmul tile variants viable again.
- Mechanism: Re-testing `wn32`, `wn16`, and `wm128-wn32` on top of stride18 checks whether Q3_K coopmat tile shape and shared-memory stride are coupled.

## Result

- Outcome: reject all tile variants; keep E082 stride18 base.
- Delta: all tested variants were below E082 stride18 r3 `922.62 tok/s`.
- Confidence: medium; these are r1 scout gates, but none was close enough to justify confirmation.
- Recommendation: continue source-level shader work with base stride18, not extra `GGML_VK_AMD_LARGE_MATMUL_VARIANT` knobs.

## Key Measurements

| Config | Gate | Throughput | Decision |
| --- | --- | ---: | --- |
| E082 stride18 | `llama-bench pp7488 r3` | `922.62 tok/s` | kept baseline |
| stride18 + `wn32` | `llama-bench pp7488 r1` | `915.98 tok/s` | reject |
| stride18 + `wn16` | `llama-bench pp7488 r1` | `771.75 tok/s` | reject |
| stride18 + `wm128-wn32` | `llama-bench pp7488 r1` | `892.75 tok/s` | reject |

Artifacts:

- `build_logs/agent-workload/e085-vulkan-q3-stride18-wn32-pp7488.md`
- `build_logs/agent-workload/e085-vulkan-q3-stride18-wn16-pp7488.md`
- `build_logs/agent-workload/e085-vulkan-q3-stride18-wm128wn32-pp7488.md`