# Major Topology Program Board

This directory is for post-E264 architecture work. It is intentionally upstream
of normal E### benchmarking: use it to rank designs before editing backend code.

## Active Program

| Field | Value |
| --- | --- |
| Program | P002 130k dense Qwen3.6 Vulkan/ROCm residency route |
| Model | `models/Qwen3.6-27B-Q3_K_S.gguf` |
| Backend | Vulkan and ROCm on RX 9070 XT / AMD proprietary driver + HIP SDK 7.1 |
| Lane | `ctx=131072,b=512,q4_0/q4_0,FlashAttention,spec=none,real-context-chars=24576,max_tokens=16`; Vulkan `ub=256`, ROCm `ub=128` until rechecked |
| Reuse/prime | off / off |
| Thinking | on |
| Current best | P002 quick target reached by D012 Vulkan opt-in stack: `b512/ub256` `2.0013 TPS` r3 with `bn256 + lowtile3 + q3quad + GLU fast path`; D035 hardens those route pieces plus a narrow host-KV guard as source defaults and recovers the fresh slow pocket to `1.8736 TPS` r1, but remains below D012; new active Vulkan target is `2.4 TPS`; D005 default split-K anchor remains `1.7898 TPS` r3; ROCm `b512/ub128` `1.5200 TPS` r3 and paused after D013-D027 |
| Current trace | Vulkan: [P002_VULKAN_130K_EVIDENCE.md](P002_VULKAN_130K_EVIDENCE.md), D003 cliff gate: [D003_P002_VULKAN_UBATCH_CLIFF_GATE.md](D003_P002_VULKAN_UBATCH_CLIFF_GATE.md), D004 FFN ceiling: [D004_P002_VULKAN_FFN_ROUTE_CEILING.md](D004_P002_VULKAN_FFN_ROUTE_CEILING.md), D005 split-K: [D005_P002_VULKAN_FFN_DOWN_SPLITK.md](D005_P002_VULKAN_FFN_DOWN_SPLITK.md), D006 output/residency: [D006_P002_VULKAN_RESIDENCY_OUTPUT_PLACEMENT.md](D006_P002_VULKAN_RESIDENCY_OUTPUT_PLACEMENT.md), D007 FFN block gate: [D007_P002_VULKAN_FFN_BLOCK_ROUTE_GATE.md](D007_P002_VULKAN_FFN_BLOCK_ROUTE_GATE.md), D012 2 TPS stack: [D012_P002_VULKAN_Q3QUAD_GLU_2TPS_STACK.md](D012_P002_VULKAN_Q3QUAD_GLU_2TPS_STACK.md), D028 2.4 target gate: [D028_P002_VULKAN_2P4_TARGET_GATE.md](D028_P002_VULKAN_2P4_TARGET_GATE.md), D029 whole-FFN gate: [D029_P002_VULKAN_WHOLE_FFN_2P4_GATE.md](D029_P002_VULKAN_WHOLE_FFN_2P4_GATE.md), D030 all-Q3 gate: [D030_P002_VULKAN_ALLQ3_2P4_GATE.md](D030_P002_VULKAN_ALLQ3_2P4_GATE.md), D031 Q3S layout-body gate: [D031_P002_VULKAN_Q3S_LAYOUT_BODY_2P4_GATE.md](D031_P002_VULKAN_Q3S_LAYOUT_BODY_2P4_GATE.md), D032 Q3+FA stack gate: [D032_P002_VULKAN_Q3_FA_STACK_2P4_GATE.md](D032_P002_VULKAN_Q3_FA_STACK_2P4_GATE.md), D033 q3-octa gate: [D033_P002_VULKAN_Q3_OCTA_PREBUILD_GATE.md](D033_P002_VULKAN_Q3_OCTA_PREBUILD_GATE.md), D034 residency recheck: [D034_P002_VULKAN_130K_RESIDENCY_RECHECK.md](D034_P002_VULKAN_130K_RESIDENCY_RECHECK.md), D035 default guard hardening: [D035_P002_VULKAN_D012_DEFAULT_GUARD_HARDENING.md](D035_P002_VULKAN_D012_DEFAULT_GUARD_HARDENING.md); ROCm D002 paused: [D002_P002_ROCM_LOW_LEVEL_Q3K_BODY.md](D002_P002_ROCM_LOW_LEVEL_Q3K_BODY.md) |
| Primary risk | 130k KV/context/working set spills beyond 16 GB VRAM into system RAM |
| Pause checkpoint | Paused by user on 2026-05-27 before a public `llama-bench` comparison track. On resume, rerun the D012 same-lane control first; do not compare new candidates against the D034 `~0.37 TPS` slow pocket. |

Previous program P001 (`ctx=12288` Vulkan Q3_K dense prompt route) is historical.
Its kept best is E257 r3 `7.0319 TPS`, prompt `999.22 tok/s`, decode `40.93 tok/s`; E258/E259/E260/E264 closed nearby transfers.

## Current Rejection Fence

Do not reopen these for the 130k program without a new mechanism and a design note:

- Q3_K transpose-A as a single layout for both prompt and decode.
- f16 KV as the archived 12k dense default.
- `batch=7680`, `batch=8192`, graphics queue, `--no-mmap` transfer gates from 12k. For 130k, queue/mmap must be re-tested because residency pressure changed.
- `GGML_VK_DISABLE_F16=1` or broad f32acc/f16-disable pivots.
- Per-layer FFN activation casts to F16.
- Helper-only Q3_K arithmetic rewrites, pair-scale reuse, packed32 helper-only
  rewrites, nearby stride tweaks, and large current-tile variants already
  rejected by H31 history.
- Output-layer placement as a launch/profile fix. D006 showed it can recover
  prompt eval from a 130k residency cliff, but decode falls to about `22 tok/s`,
  so it is diagnostic evidence rather than a route to `2 TPS`.
- Broad backend-host KV placement as a launch/profile fix. D034 showed partial
  `Vulkan_Host` KV can recover the current slow pocket up to `1.9826 TPS`, but
  it remains below D012 and pays decode back (`~37 tok/s` vs D012 `42.72`). D035
  keeps only a narrow Qwen35-like auto guard for default stability; do not treat
  host-KV sweeps as a speed route.

## Candidate Queue

| ID | Candidate | Status | Next required artifact |
| --- | --- | --- | --- |
| T1 | Dual-layout Q3_K storage: raw decode layout plus prefill layout for matrix routes | design-needed | VRAM/storage model and fail-closed tensor movement plan |
| T2 | Shader-native Q3_K prompt layout: compact signed/scale-expanded block for coopmat matmul | D031 rejects compact Q3S/signed-nibble plus scale-expanded layout-body as target-closing route | Reopen only with a compute body that reduces matrix work itself, not just unpack metadata |
| T3 | Fused dense FFN block: gate/up/swiglu/down tiled route | D029 rejects activation-only and naive streaming whole-FFN routes; D007 still proves the graph surface exists | reopen only if the design reduces Q3_K matmul work itself or becomes part of a broader all-Q3 dataflow; launch/hidden-materialization-only fusion is below the `2.4 TPS` bar |
| T3a | Vulkan all-Q3/body-layout target for `2.4 TPS` | D030 rejects current q3quad extension, scale-only metadata/helper reuse, signed-nibble-only storage, Q8_1/int-dot, expanded layouts, and neighboring tile tweaks; D031 rejects compact Q3S layout-body; D032 shows FA cannot carry alone; D033 rejects q3-octa/`LOAD_VEC_A=8` repeat | First implementation gate remains a true Q3_K body/compressed-dot route; a Q3+FA stack is only useful after Q3 reaches about `1.18-1.20x` local point/static proof |
| T4 | FA shader-body work for long-KV and 130k RAM-spill lane | lower-priority for P002 quick | rerun evidence if a heavier long-KV fill shifts FA above Q3_K |
| T5 | Vulkan/ROCm differential harness | first Vulkan pack complete | run `bench: vulkan q3 130k route trace`, `bench: vulkan q3 130k q3 stats`, then `research: vulkan evidence pack` |
| T6 | ROCm low-level Q3_K MMQ/FFN body | paused by user after D013-D027 rejected as speed routes | reopen only with a new compressed-GEMM/FFN dataflow proof; active target moved back to Vulkan `2.4 TPS` |
| T7 | Vulkan `ub>=320` residency cliff | D003 closed as non-speed recovery | do not promote `GGML_VK_ENABLE_MEMORY_PRIORITY=1`; revisit only with a shader/body/lifetime change that beats `ub256` |
| T8 | Vulkan output-layer residency relief | D006 closed as diagnostic, not speed route | next work must be a source/topology design with a residency model; do not continue ubatch/output-placement sweeps |
| T8a | Vulkan backend-host KV residency relief | D035 keeps a narrow Qwen35-like host-KV auto guard for default stability (`1.8736 TPS` recovery), while broad D034 host-KV sweeps remain diagnostic and below D012 | reopen only with a lifetime/migration design that preserves decode and beats D012, not as a launch/KV placement sweep |
| T9 | Vulkan q3quad/bn256/lowtile promotion hardening | D035 promotes guarded source defaults for AMD `bn256`, Q3_K quad dequant, and Q3 low-tile split-K; D012 `2.0013 TPS` r3 remains the speed comparator | run confirm3 before final default-promotion wording; keep exact D012 r3 as baseline until a matching confirmation beats or ties it |

## Required Evidence Pack

Before any source prototype, attach or link:

- paired Vulkan and ROCm 130k lane controls on the current binaries;
- route trace for Q3_K and FA;
- server diagnostics covering startup, mmap/no-mmap, allocator/residency, and RAM pressure;
- `vulkan_perf_shape_summary.py` output for the active trace;
- relevant SPIR-V opcode/resource summaries;
- Amdahl/ceiling estimate for the touched route;
- correctness plan for prompt and decode routes;
- rollback path.

Vulkan helper: `scripts/research/vulkan_evidence_pack.py` converts a benchmark
label into `build_logs/agent-workload/<label>.vulkan-evidence.md`. It summarizes
diagnostics, residency/startup signals, Vulkan route/perf traces, pipeline
resource lines, SPIR-V opcode fingerprints, and optional Amdahl ceiling sketches
via `--baseline-tps`. If route/perf traces are absent, it marks the missing
gates instead of pretending the pack is complete. Use
`--extra-log <q3-stats.server.log>` to merge a separate `--trace-preset
vulkan-q3-stats` resource run into the same pack; the matching VS Code task is
`bench: vulkan q3 130k q3 stats`.

## Branch Discipline

- Use one branch/worktree per candidate topology.
- Keep negative prototypes out of `master` unless they are default-off diagnostic
  tools with a documented reason to keep them.
- If a candidate fails before lane A/B, write a design/scout rejection instead
  of creating a benchmark-shaped E### note.
