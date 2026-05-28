# D041 - P002 Vulkan practical big-prompt no-reuse control

Date: 2026-05-27  
Owner: Copilot/perf workspace  
Status: measured gate (no source edits)

## Scope

User-directed focus is now Vulkan-only on the practical big-prompt lane, with
targets:

- prompt: `900 tok/s`
- decode: `30 tok/s`

This note isolates the reuse/checkpoint hypothesis on the same practical lane
as D040.

Lane contract:

- model: `Qwen3.6-27B-Q3_K_S.gguf`
- backend: Vulkan (`build-vulkan`)
- ctx: `131072`
- batch/ubatch: `512/256`
- kv: `q4_0/q4_0`
- flash-attn: on
- task: `quick:triage_diff`
- real-context: `repo-snapshot`, `real-context-chars=152000`
- prompt scale: `task_prompt_tokens=56425`
- server extra: `--spec-type none --no-mmap`

Controls:

- D040 comparator (repeated/steady): reuse on (`p002-vulkan130k-big-c152k-b512-ub256-r1`)
- D041 candidate: `--no-reuse --no-v2-prime-pass` (harness injects
  `--cache-ram 0 --ctx-checkpoints 0`)

## Measured Results

### Same max-tokens comparison (strict control)

| Mode | Label | max_tokens | Aggregate TPS | Prompt tok/s | Decode tok/s | Prompt ms | Decode ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reuse on (D040) | `p002-vulkan130k-big-c152k-b512-ub256-r1` | `16` | `0.1758` | `626.06` | `21.60` | `90127.14` | `740.84` |
| No-reuse (D041) | `d041-vulkan130k-big-c152k-noreuse-mt16-b512-ub256-r1` | `16` | `0.1766` | `628.88` | `21.46` | `89723.13` | `745.48` |

D041 no-reuse vs D040 reuse (same max=16):

- aggregate TPS: `+0.46%`
- prompt tok/s: `+0.45%`
- decode tok/s: `-0.65%`
- prompt ms: `-0.45%`
- decode ms: `+0.63%`

### Decode sanity at longer generation

| Mode | Label | max_tokens | Aggregate TPS | Prompt tok/s | Decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| No-reuse | `d041-vulkan130k-big-c152k-noreuse-mt64-b512-ub256-r1` | `64` | `0.6911` | `631.73` | `20.09` |

This confirms decode remains around `~20 tok/s` on this practical lane, still
well below the `30 tok/s` target.

## Log Evidence

- No-reuse run reports prompt cache disabled:
  `prompt cache is disabled - use --cache-ram N to enable it`
- D041 mt16 has no `slot create_check` checkpoint events.
- D040 reuse run creates repeated context checkpoints (at 8192-token cadence).

Despite checkpoint removal, measured prompt/decode stayed nearly flat.

## Decision

1. Reject reuse/checkpoint overhead as the primary limiter on the practical
   big-prompt Vulkan lane.
2. Keep D040 as backend checkpoint and D041 as no-reuse control evidence.
3. Next Vulkan-only route should target true kernel/body scaling under long
   prompt (Q3_K/FFN/route-body), not session cache semantics.

## Artifacts

- `build_logs/agent-workload/d041-vulkan130k-big-c152k-noreuse-mt16-b512-ub256-r1.diagnostics.md`
- `build_logs/agent-workload/d041-vulkan130k-big-c152k-noreuse-mt16-b512-ub256-r1.server.log`
- `build_logs/agent-workload/d041-vulkan130k-big-c152k-noreuse-mt64-b512-ub256-r1.diagnostics.md`
- `build_logs/agent-workload/d041-vulkan130k-big-c152k-noreuse-mt64-b512-ub256-r1.server.log`
- `build_logs/agent-workload/p002-vulkan130k-big-c152k-b512-ub256-r1.diagnostics.md`
- `build_logs/agent-workload/p002-vulkan130k-big-c152k-b512-ub256-r1.server.log`