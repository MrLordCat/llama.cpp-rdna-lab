# D045 - P002 Vulkan practical big-prompt bn256 gate

Date: 2026-05-27  
Owner: Copilot/perf workspace  
Status: measured gate (no source edits)

## Scope

After D043, the next Vulkan-only route gate on the same practical lane was to
measure the contribution of AMD `bn256` by disabling its auto-default while
keeping the D043 lowtile setting:

- `GGML_VK_QK_LOW_TILE_SPLIT_K=3` (kept)
- `GGML_VK_DISABLE_AMD_BN256_DEFAULT=1` (candidate)

Lane contract (matched to D043):

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
| D045 no-bn256 candidate | `d045-vulkan130k-big-c152k-lowtile3-nobn256-noreuse-mt16-b512-ub256-r1` | `0.1669` | `593.85` | `21.66` | `95016.08` | `738.72` |

D045 vs D043:

- aggregate TPS: `-6.60%`
- prompt tok/s: `-6.64%`
- decode tok/s: `+0.60%`
- prompt ms: `+7.11%`
- decode ms: `-0.58%`

## Decision

1. Reject `GGML_VK_DISABLE_AMD_BN256_DEFAULT=1` for the practical 130k lane.
2. Keep `bn256` auto-default behavior enabled for this lane family.
3. Continue with route-level Q3_K/FFN/body hypotheses under the enforced
   `ubatch<=256` VRAM constraint.

## Artifacts

- `build_logs/agent-workload/d045-vulkan130k-big-c152k-lowtile3-nobn256-noreuse-mt16-b512-ub256-r1.diagnostics.md`
- `build_logs/agent-workload/d045-vulkan130k-big-c152k-lowtile3-nobn256-noreuse-mt16-b512-ub256-r1.server.log`
- `build_logs/agent-workload/d043-vulkan130k-big-c152k-lowtile3-noreuse-mt16-b512-ub256-r1.diagnostics.md`
