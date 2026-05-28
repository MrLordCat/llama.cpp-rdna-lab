# D074 - P003 H54-B guarded runtime application (CPU residency proof)

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: runtime-applied proof (env-gated, fail-closed)

## Scope

Implement the first runtime-applied stage after D073 and verify that sidecar rows
change real model residency (not just logs), while keeping default behavior unchanged.

## What Was Implemented

Code path:

- `src/llama-model-loader.cpp`
- `src/llama-model-loader.h`

New runtime behavior:

- Existing gate remains: `LLAMA_Q4_METACOMP_ENABLE=1` + sidecar path in
  `LLAMA_Q4_METACOMP_SIDECAR`.
- New explicit apply gate: `LLAMA_Q4_METACOMP_FORCE_CPU_SELECTED=1`.
- Under apply gate, validated selected Q4 tensors are forced to CPU-compatible
  buffer selection during tensor placement.
- Any parse/validation/selection issue remains fail-closed and falls back to
  legacy placement.

Default path:

- Unchanged when `LLAMA_Q4_METACOMP_ENABLE` is not set.

## Applied A/B Evidence (same practical lane shape)

Control (D048-style fit-auto baseline):

- `q4fitauto-vulkan130k-big-c152k-b512-ub256-r2.server.log`
- model buffers:
  - `Vulkan0 model buffer size = 11665.45 MiB`
  - `Vulkan_Host model buffer size = 3708.06 MiB`

Applied candidate (D074, env-gated):

- sidecar: `build_logs/agent-workload/q4metacomp-forcecpu-smoke-r1.sidecar.json`
- run log: `q4metacomp-forcecpu-vulkan130k-big-c152k-b512-ub256-r1.server.log`
- runtime confirmation:
  - `q4 metacomp sidecar loaded: selected=128 ... validated_q4=128`
  - `q4 metacomp applied summary: forced_cpu=128 validated_selected=128`
- model buffers:
  - `Vulkan0 model buffer size = 8619.29 MiB`
  - `Vulkan_Host model buffer size = 6754.22 MiB`

Observed residency shift:

- GPU model buffer delta: `-3046.16 MiB`
- Host model buffer delta: `+3046.16 MiB`

This proves D074 is an applied runtime path (real residency change), not only
analytical projection.

## Decision

Keep D074 as a guarded applied step.

- Positive: runtime sidecar rows now produce measurable placement change.
- Limitation: this is residency redistribution (GPU -> host), not payload
  compression. It does not satisfy the long-term target of reducing total model
  footprint by compact Q4 representation.

## Next Step

Proceed to true compressed-storage route (new format/runtime decode path) where
resident bytes for selected tensors shrink without shifting the same bytes to
host RAM.

## Artifacts

- `src/llama-model-loader.cpp`
- `src/llama-model-loader.h`
- `docs/research/major-topology/D074_P003_H54B_GUARDED_RUNTIME_APPLICATION_CPU_RESIDENCY.md`
- `build_logs/agent-workload/q4metacomp-forcecpu-smoke-r1.sidecar.json`
- `build_logs/agent-workload/q4metacomp-forcecpu-vulkan130k-big-c152k-b512-ub256-r1.server.log`
- `build_logs/agent-workload/q4fitauto-vulkan130k-big-c152k-b512-ub256-r2.server.log`
