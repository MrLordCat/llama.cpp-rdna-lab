# D044 - P002 Vulkan practical big-prompt ub512 abort

Date: 2026-05-27  
Owner: Copilot/perf workspace  
Status: aborted (out-of-lane)

## Scope

D044 attempted a practical-lane run with `ubatch=512`:

- label: `d044-vulkan130k-big-c152k-lowtile3-noreuse-mt16-b512-ub512-r1`
- intent: quick pressure test after D043 (`ubatch=256`)

Lane contract context (unchanged otherwise):

- model: `Qwen3.6-27B-Q3_K_S.gguf`
- ctx: `131072`
- batch: `512`
- kv: `q4_0/q4_0`
- flash-attn: on
- task: `quick:triage_diff`
- real-context: `repo-snapshot`, `real-context-chars=152000`
- max_tokens: `16`
- server extra: `--spec-type none --no-mmap`
- reuse mode: cold/no-reuse/no-prime

## Observed Runtime Behavior

The run did not complete and was stopped after clear severe slowdown in prefill.
Progress snapshots in server log remained far behind the D041/D043 corridor, for
example:

- `n_tokens=12288` (`progress=0.217776`)
- `n_tokens=17920` (`progress=0.317590`)
- `n_tokens=20480` (`progress=0.362960`)
- `n_tokens=24064` (`progress=0.426478`)

No diagnostics file was produced for D044.

## Decision

1. Close D044 as aborted and rejected for the active practical lane.
2. Enforce session rule: do not run `ubatch > 256` on this Vulkan 130k
   big-prompt lane due to VRAM pressure.
3. Continue only with `ubatch<=256` candidates and route-level Q3_K/FFN/body
   hypotheses.

## Artifacts

- `build_logs/agent-workload/d044-vulkan130k-big-c152k-lowtile3-noreuse-mt16-b512-ub512-r1.server.log`
