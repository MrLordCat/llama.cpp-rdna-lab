# D042 - P002 Vulkan practical big-prompt first4 host-KV gate

Date: 2026-05-27  
Owner: Copilot/perf workspace  
Status: measured gate (no source edits)

## Scope

After D041 rejected reuse/checkpoint as a primary limiter, this gate checks a
cheap Vulkan residency variant on the same practical lane:

- force direct host-KV on `first 4/16` layers
- keep cold/no-reuse/no-prime
- keep all other lane controls unchanged

Lane contract:

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

Variant env:

- `LLAMA_VK_KV_HOST_LAYERS=4`
- `LLAMA_VK_KV_HOST_POSITION=first`
- `LLAMA_VK_KV_HOST_DIRECT=1`

## Measured Results

| Profile | Label | Aggregate TPS | Prompt tok/s | Decode tok/s | Prompt ms | Decode ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| D041 no-reuse last3 baseline | `d041-vulkan130k-big-c152k-noreuse-mt16-b512-ub256-r1` | `0.1766` | `628.88` | `21.46` | `89723.13` | `745.48` |
| D042 first4 candidate | `d042-vulkan130k-big-c152k-first4-noreuse-mt16-b512-ub256-r1` | `0.1768` | `629.61` | `20.93` | `89619.26` | `764.54` |

D042 vs D041 deltas:

- aggregate TPS: `+0.11%`
- prompt tok/s: `+0.12%`
- decode tok/s: `-2.47%`
- prompt ms: `-0.12%`
- decode ms: `+2.56%`

## Log Evidence

Candidate log confirms the intended route:

- `Vulkan host-KV residency guard enabled for first 4/16 KV layers, mode=kv, direct`
- `Vulkan0 KV buffer size = 1728.00 MiB`
- `Vulkan_Host_Direct KV buffer size = 576.00 MiB`
- `graph splits = 2`

## Decision

1. Reject `first4` direct host-KV as a practical big-prompt speed route.
2. The candidate is wall-tied with a small decode regression and does not move
   toward targets (`prompt 900`, `decode 30`).
3. Keep the lane on the D041/D040 profile and continue with true Q3_K/FFN/body
   scaling hypotheses instead of host-KV placement sweeps.

## Artifacts

- `build_logs/agent-workload/d042-vulkan130k-big-c152k-first4-noreuse-mt16-b512-ub256-r1.diagnostics.md`
- `build_logs/agent-workload/d042-vulkan130k-big-c152k-first4-noreuse-mt16-b512-ub256-r1.server.log`
- `build_logs/agent-workload/d041-vulkan130k-big-c152k-noreuse-mt16-b512-ub256-r1.diagnostics.md`