# D004 P002 Vulkan FFN Route Ceiling

Date: 2026-05-26

Status: design/scout gate, no source speed claim.

## Lane

- Backend: Vulkan, `build-vulkan`.
- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Contract: `ctx=131072,batch=512,ubatch=256,q4_0/q4_0,FlashAttention,spec=none,--no-mmap`.
- Workload: `quick:triage_diff`, `real-context-chars=24576`, cold-first, no reuse, no v2 prime, thinking on.
- Baseline anchor: `p002-vulkan-ub256-current-r1`, `1.6654 TPS`, prompt `866.47 tok/s`, decode `43.42 tok/s`.
- Target: `2.0 TPS` (`1.2009x` total speedup).

## What Changed In The Gate

`scripts/research/vulkan_route_ceiling.py` was corrected so dense FFN matching is
not hardcoded to `n=1024`. The P002 trace uses `n=256` for the dominant prompt
route, so the old report undercounted the actual FFN share.

Corrected artifact:

- `build_logs/agent-workload/d004-vulkan130k-route-ceiling-corrected.md`.

## Corrected Route Shares

| Route | Parsed share | Required local speedup for 2 TPS | Note |
| --- | ---: | ---: | --- |
| Dense FFN gate/up Q3_K | `34.91%` | `1.920x` | two sibling `17408x5120` projections |
| Dense FFN down Q3_K | `24.61%` | `3.123x` | `5120x17408` projection after SwiGLU |
| Dense FFN gate/up + down Q3_K | `59.52%` | `1.391x` | main dense FFN route |
| All Q3_K MUL_MAT | `80.50%` | `1.262x` | whole large-prefill Q3_K route |
| All FLASH_ATTN_EXT | `7.60%` | unreachable alone | q4/q4 long-KV FA route |
| All Q3_K MUL_MAT + FA | `88.10%` | `1.234x` | combined prefill core |

The practical route target is therefore Q3_K/FFN. FA remains relevant for a
heavier long-KV lane, but it is not a solo route to `2 TPS` on this quick lane.

## Gate/Up Fusion Model

Artifact:

- `build_logs/agent-workload/d004-vulkan130k-ffn-gateup-route-model.md`.

The model checks the old 64k idea again at the active 130k shape
`17408x256x5120`: fuse sibling gate/up matmuls so one activation tile feeds two
Q3_K A streams, then write the SwiGLU output directly.

Base-tile estimate:

| Item | Value |
| --- | ---: |
| Required local speedup for gate/up alone | `1.920x` |
| Dual-A LDS | `29696 B` |
| Accumulators | `16 -> 32` fragments |
| Local ceiling with unchanged A proxy | `1.417x` |
| Projected wall speedup from that ceiling | `1.114x` |

This means a gate/up-only dual-A route is too weak as a standalone 2 TPS path.
It may still be a stack component, but only after a broader FFN/Q3_K design
also reduces the down projection or repeated Q3_K A-side work.

## Decision

Reject gate/up-only fusion as the next source prototype. The corrected ceiling
keeps Vulkan as the closest backend to the 2 TPS target, but the next design
must be one of:

- a whole dense FFN route that changes gate/up, SwiGLU, and down together;
- a Q3_K layout/repack route that lowers repeated A-dequant work across FFN and
  non-FFN Q3_K shapes;
- a combined Q3_K+FA stack only if a heavier trace shows FA share has grown.

Do not spend runtime prototype time on a simple dual-A/same-B shader unless a
new model shows how it also reduces the down route or A-side Q3_K repetition.

## Follow-Up

D005 found and kept the low-risk part of this route: the existing Vulkan
split-K/reduce path is now used by default for the Q3_K FFN-down shape
`m=5120,n>=128,k=17408`. That moves the active Vulkan baseline to `1.7898 TPS`.
The remaining 2 TPS work should use the post-D005 ceiling, not the pre-D005
`1.6654 TPS` gate.
