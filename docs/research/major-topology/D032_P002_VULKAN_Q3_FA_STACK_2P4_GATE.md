# D032 P002 Vulkan Q3+FA Stack 2.4 Gate

Date: 2026-05-27.

## Lane

- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Backend: Vulkan on RX 9070 XT.
- Baseline: D012 `2.0013 TPS` r3 on the 130k cold quick lane.
- Target: `2.4 TPS`, same lane.

## Gate Artifact

- Script: `scripts/research/vulkan_q3_fa_stack_2p4_gate.py`.
- Artifact: `build_logs/agent-workload/d032-vulkan-q3-fa-stack-2p4-gate.md`.

## Inputs

- D030 all-Q3 point: `5691.67 ms`.
- D010 full-trace FlashAttention point: `693.77 ms`.
- Required D030 point savings: `1174.57 ms`.

## Stack Math

| FA local speedup | FA point savings | Q3 savings still needed | Q3 local speedup still needed |
| ---: | ---: | ---: | ---: |
| `1.00x` | `0.00 ms` | `1174.57 ms` | `1.2600x` |
| `1.25x` | `138.75 ms` | `1035.82 ms` | `1.2225x` |
| `1.50x` | `231.26 ms` | `943.31 ms` | `1.1987x` |
| `2.00x` | `346.88 ms` | `827.68 ms` | `1.1702x` |

## Decision

Do not pivot to FA-only work for the Vulkan `2.4 TPS` target. Even a `2.0x` FA
shader-body win still leaves about `827.68 ms` of Q3 point savings, or `1.1702x`
local on all-Q3.

A target-closing stack is plausible only if a true Q3_K body/compressed-dot
candidate reaches roughly the `1.18-1.20x` local band and then stacks with a
substantial FA win. D032 therefore keeps Q3_K as the first implementation gate,
but allows a Q3+FA stack as the route to `2.4 TPS` once a real Q3 body point or
static proof exists.