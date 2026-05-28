# D043 - P002 Vulkan practical big-prompt lowtile3 gate

Date: 2026-05-27  
Owner: Copilot/perf workspace  
Status: measured gate (no source edits)

## Scope

After D041/D042, the next cheap Vulkan-only candidate on the practical lane was
to force `Q3_K` low-tile split-K to `3`:

- `GGML_VK_QK_LOW_TILE_SPLIT_K=3`

This is meaningful because the current default logic for the candidate shapes
with `n>=512` typically selects lowtile `2` unless overridden.

Lane contract (unchanged):

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
| D041 no-reuse baseline | `d041-vulkan130k-big-c152k-noreuse-mt16-b512-ub256-r1` | `0.1766` | `628.88` | `21.46` | `89723.13` | `745.48` |
| D043 lowtile3 candidate | `d043-vulkan130k-big-c152k-lowtile3-noreuse-mt16-b512-ub256-r1` | `0.1787` | `636.09` | `21.53` | `88706.65` | `743.02` |

D043 vs D041:

- aggregate TPS: `+1.19%`
- prompt tok/s: `+1.15%`
- decode tok/s: `+0.33%`
- prompt ms: `-1.13%`
- decode ms: `-0.33%`

## Decision

1. Keep as a small positive gate, but not a target-closing route.
2. Improvement is real but too small for practical goals (`prompt 900`,
   `decode 30`) on the big-prompt lane.
3. Continue Vulkan-only work toward a larger Q3_K/FFN/body effect; lowtile tuning
   alone is now a secondary knob.

## Artifacts

- `build_logs/agent-workload/d043-vulkan130k-big-c152k-lowtile3-noreuse-mt16-b512-ub256-r1.diagnostics.md`
- `build_logs/agent-workload/d043-vulkan130k-big-c152k-lowtile3-noreuse-mt16-b512-ub256-r1.server.log`
- `build_logs/agent-workload/d041-vulkan130k-big-c152k-noreuse-mt16-b512-ub256-r1.diagnostics.md`