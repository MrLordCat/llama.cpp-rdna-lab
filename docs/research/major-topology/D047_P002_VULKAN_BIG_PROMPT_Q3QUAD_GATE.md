# D047 - P002 Vulkan practical big-prompt q3quad gate

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: measured gate (no source edits)

## Scope

Next Vulkan-only route gate on the same practical lane: disable Q3 quad dequant
while keeping the D043 lane shape and lowtile3.

- keep: `GGML_VK_QK_LOW_TILE_SPLIT_K=3`
- candidate: `GGML_VK_Q3K_QUAD_DEQUANT=0`

Lane contract (matched):

- model: `Qwen3.6-27B-Q3_K_S.gguf`
- ctx: `131072`
- batch/ubatch: `512/256`
- kv: `q4_0/q4_0`
- flash-attn: on
- task: `quick:triage_diff`
- real-context: `repo-snapshot`, `real-context-chars=152000`
- prompt scale: `task_prompt_tokens=56425`
- max_tokens: `16`
- server extra: `--spec-type none --no-mmap`
- reuse mode: cold/no-reuse/no-prime

## Measured Results

| Profile | Label | Aggregate TPS | Prompt tok/s | Decode tok/s | Prompt ms | Decode ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| D043 lowtile3 baseline | `d043-vulkan130k-big-c152k-lowtile3-noreuse-mt16-b512-ub256-r1` | `0.1787` | `636.09` | `21.53` | `88706.65` | `743.02` |
| D047 no-q3quad candidate | `d047-vulkan130k-big-c152k-lowtile3-noq3quad-noreuse-mt16-b512-ub256-r1` | `0.1748` | `622.40` | `21.36` | `90656.93` | `749.16` |

D047 vs D043:

- aggregate TPS: `-2.18%`
- prompt tok/s: `-2.15%`
- decode tok/s: `-0.79%`
- prompt ms: `+2.20%`
- decode ms: `+0.83%`

## Decision

1. Reject disabling Q3 quad dequant on this practical lane.
2. Keep q3quad dequant enabled as part of the active Vulkan practical profile.
3. Continue route-level Q3_K/FFN/body work; this gate confirms q3quad remains a
   net-positive component under big-prompt pressure.

## Artifacts

- `build_logs/agent-workload/d047-vulkan130k-big-c152k-lowtile3-noq3quad-noreuse-mt16-b512-ub256-r1.diagnostics.md`
- `build_logs/agent-workload/d047-vulkan130k-big-c152k-lowtile3-noq3quad-noreuse-mt16-b512-ub256-r1.server.log`
- `build_logs/agent-workload/d043-vulkan130k-big-c152k-lowtile3-noreuse-mt16-b512-ub256-r1.diagnostics.md`
