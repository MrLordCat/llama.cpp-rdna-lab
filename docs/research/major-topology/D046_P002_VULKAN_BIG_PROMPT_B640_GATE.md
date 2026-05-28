# D046 - P002 Vulkan practical big-prompt b640 gate

Date: 2026-05-27  
Owner: Copilot/perf workspace  
Status: measured gate (no source edits)

## Scope

After D045, the next bounded candidate under the enforced `ubatch<=256` rule
was to test whether a larger outer batch improves prompt-side throughput on the
same practical lane:

- candidate shape change: `batch=640` (from `512`)
- keep: `ubatch=256`, `GGML_VK_QK_LOW_TILE_SPLIT_K=3`, default bn256 behavior

Lane contract (otherwise unchanged):

- model: `Qwen3.6-27B-Q3_K_S.gguf`
- ctx: `131072`
- batch/ubatch: `640/256` candidate vs `512/256` baseline
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
| D046 b640 candidate | `d046-vulkan130k-big-c152k-lowtile3-noreuse-mt16-b640-ub256-r1` | `0.1640` | `583.47` | `21.92` | `96706.21` | `730.01` |

D046 vs D043:

- aggregate TPS: `-8.23%`
- prompt tok/s: `-8.27%`
- decode tok/s: `+1.81%`
- prompt ms: `+9.02%`
- decode ms: `-1.75%`

## Decision

1. Reject `batch=640` for this practical 130k lane.
2. Keep `batch=512` as the active shape on Vulkan big-prompt lane under current
   constraints.
3. Continue with route-level Q3_K/FFN/body candidates; shape-only batch increase
   did not move toward practical targets (`prompt 900`, `decode 30`).

## Artifacts

- `build_logs/agent-workload/d046-vulkan130k-big-c152k-lowtile3-noreuse-mt16-b640-ub256-r1.diagnostics.md`
- `build_logs/agent-workload/d046-vulkan130k-big-c152k-lowtile3-noreuse-mt16-b640-ub256-r1.server.log`
- `build_logs/agent-workload/d043-vulkan130k-big-c152k-lowtile3-noreuse-mt16-b512-ub256-r1.diagnostics.md`
