# D040 - P002 130k big-prompt backend delta (Vulkan vs ROCm)

Date: 2026-05-27  
Owner: Copilot/perf workspace  
Status: measured gate (no source edits)

## Scope

Measure same practical big-prompt lane on both backends before choosing the next route.

Lane contract used in both runs:

- model: `Qwen3.6-27B-Q3_K_S.gguf`
- ctx: `131072`
- batch: `512`
- kv: `q4_0/q4_0`
- flash-attn: `on`
- task: `quick:triage_diff`
- real-context: `repo-snapshot`, `real-context-chars=152000`
- prompt scale: `task_prompt_tokens=56425`
- generation: `max_tokens=16`
- reuse: `on` (repeated/steady class), no prime pass

Backend-specific knobs:

- Vulkan: `ubatch=256`, `--spec-type none --no-mmap`
- ROCm: `ubatch=128`, `--spec-type none`

## Measured Results

| Backend | Label | Aggregate TPS | Prompt tok/s | Decode tok/s | Prompt ms | Decode ms | Prompt tokens | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vulkan | `p002-vulkan130k-big-c152k-b512-ub256-r1` | `0.1758` | `626.06` | `21.60` | `90127.14` | `740.84` | `56425` | `0` |
| ROCm | `p002-rocm130k-big-c152k-b512-ub128-r1` | `0.1023` | `363.81` | `13.82` | `155096.18` | `1157.78` | `56425` | `0` |

Derived deltas (Vulkan vs ROCm, same lane):

- wall/aggregate TPS: `+71.85%`
- prompt throughput: `+72.08%`
- decode throughput: `+56.30%`
- prompt time: `-41.89%`
- decode time: `-36.01%`

## Residency and route notes

- Vulkan run used the active practical lane setting `--no-mmap` and retained the known host-KV/direct residency behavior from this lane family.
- ROCm run completed cleanly but remained significantly slower on both prompt and decode components at the same prompt token scale.
- On this practical lane, wall time remains prompt-dominated despite the decode gap.

## Decision

1. Keep this pair as the practical backend comparison checkpoint for big-prompt `ctx=131072` repeated/steady runs.
2. Do not reopen ROCm micro-route exploration from this result alone; ROCm remains behind on this lane.
3. Choose the next speed hypothesis on the Vulkan side, focused on prompt-side Q3_K/FFN/body work that can move the practical wall metric, not decode-only micro changes.

## Artifacts

- `build_logs/agent-workload/p002-vulkan130k-big-c152k-b512-ub256-r1.diagnostics.md`
- `build_logs/agent-workload/p002-vulkan130k-big-c152k-b512-ub256-r1.server.log`
- `build_logs/agent-workload/p002-rocm130k-big-c152k-b512-ub128-r1.diagnostics.md`
- `build_logs/agent-workload/p002-rocm130k-big-c152k-b512-ub128-r1.server.log`
