# Major Topology Workflow

Status update (2026-07-20): the primary project model is now
`Qwen3.6-27B-Q4_K_M.gguf` with the D089 dual-ROCm 49K baseline. The Q3_K_S
program described below remains valid as a secondary model-specific topology
program. Do not carry its target math or TPS rows into Q4 experiments.

Closure update (2026-08-13): the P002 D002/D028 programs are closed and the
P003 Q3_K_S program is parked. Use this workflow only after an explicit reopen;
D096 is the active primary research program.

This document starts the post-E264 research mode. The earlier E### loop was good
for quick gates, but the historical Vulkan/ROCm Q3_K lane exhausted nearby
flags, batch shapes, f16 pivots, helper rewrites, and simple layout flips. The
model-scoped target described here is dense `Qwen3.6-27B-Q3_K_S` at `ctx=131072`
(~130k), where RAM-spill/residency is expected on 16 GB VRAM. The next useful
work should be treated as an architecture program, not as another micro-probe.

## Why Reset

Current dense Qwen3.6-27B 130k objective:

- Historical lane: `ctx=131072,b=512,q4_0/q4_0,spec=none`, cold/no-reuse/no-prime, thinking on, `real-context-chars=24576`, `max_tokens=16`; Vulkan current best uses `ub=256` with `--no-mmap`, ROCm uses `ub=128`.
- Required first evidence is the P002 130k quick baseline, historically recorded
  by E265 and recentered after D005/ROCm checks: Vulkan `1.7898 TPS` r3, ROCm `1.5200 TPS` r3. Do
  not promote any topology without beating the same-backend P002 quick lane or
  explicitly labeling a heavier residency stress result.
- Expected system behavior: the full working set will not be cleanly VRAM-resident on RX 9070 XT 16 GB, so mmap/no-mmap, startup time, host RAM pressure, PCIe movement, and allocator/residency diagnostics are part of the target.

Archived dense Qwen3.6-27B 12k Vulkan evidence:

- Current kept profile: E257, `ctx=12288,b=7168,ub=1024,q4_0/q4_0,spec=none`,
  cold/no-reuse/no-prime, thinking on.
- Current kept result: `7.0319 TPS` r3, prompt `999.22 tok/s`, decode
  `40.93 tok/s`.
- E257 trace: `MUL_MAT q3_K = 82.71%`, `FLASH_ATTN_EXT = 9.60%`.
- Rejected since E257: transpose-A Q3_K storage, f16 KV default, `batch=7680`,
  graphics queue, `--no-mmap`, `batch=8192`, broad f16-disable, and graph-level
  FFN F16 `src1` casts.

The remaining target is too large for local helper edits. To get a meaningful
step, the candidate must reduce algorithmic work, memory traffic, or global
intermediate traffic on a high-share route.

## Repository Order

Use this split before opening a new large prototype:

1. Commit the measured tail as a history package.
   - Include E249-E264 experiment notes, `RESULTS_LOG.md`, `HYPOTHESES.md`,
     `BENCHMARKS.md`, GUI preset/autotune updates, and diagnostic scout code that
     is referenced by an experiment note.
   - Do not keep negative runtime/shader prototypes in source.
2. Commit this workflow reset separately.
   - It should be documentation only.
   - It becomes the entry point for the next research branch.
3. Open one focused worktree or branch per major topology.
   - One branch owns one topology family.
   - Bench artifacts and experiment notes must use that topology ID.
4. Keep build outputs and model/log artifacts out of git.
   - Keep only small diagnostic scripts and markdown notes under version control.

## Experiment Classes

Use more than one class of note. Not every idea should become a benchmark.

| Class | Prefix | Purpose | Required output |
| --- | --- | --- | --- |
| Program | `P` | Multi-experiment research theme | goal, lane, ceiling model, open questions |
| Design | `D` | Architecture proposal before code | mechanism, resource model, risk list, rejection analogs |
| Scout | `S` | Standalone tool or trace to answer one design question | script/log/table, no wall TPS claim unless it runs the lane |
| Measured gate | `P`/`D`/`S` artifact, optional `E` only for narrow ledger entries | Real A/B or correctness gate | measured result attached to the owning topology note; result-log row only when promoted |

For post-E264 topology work, P/D/S notes are the default home. E### is historical
or a deliberate narrow measured ledger entry; do not use it for design scouts,
tooling packs, or large route proposals before the owning topology note exists.

## Gate Ladder

Do not skip gates for a large topology.

1. Lane lock
   - State exact model, backend, ctx, batch, ubatch, KV, spec, reuse, thinking,
     task, max tokens, and real-context mode.
   - Name the current best and the comparison target.
2. Route evidence pack
   - Fresh route trace and perf summary for the active binary.
   - Shader/resource fingerprint for touched Vulkan pipelines or HIP kernels.
   - Shape table for hot buckets and estimated wall share.
3. Ceiling model
   - Estimate wall gain from local route speedup.
   - Reject if the maximum plausible wall gain is below the cost/risk threshold.
4. Resource model
   - RDNA4-specific: VGPR, SGPR, LDS, scratch, occupancy, barriers, workgroup
     count, queue/sync points, and VRAM residency.
   - Vulkan-specific: subgroup 64, KHR coopmat shape, LLPC resource stats,
     generated SPIR-V route, and fallback proof.
   - ROCm-specific: gfx1201 compile pressure, graph capture safety, rocBLAS/HIP
     contract, and split-buffer/storage semantics.
5. Correctness scout
   - Run a small deterministic correctness check before a server benchmark.
   - For storage/layout changes, cover upload, matmul, matvec/decode, views/copy,
     and fallback paths.
6. Point timing
   - Measure the exact hot shape or a faithful standalone proxy.
   - A point win must be large enough to survive integration overhead.
7. Lane A/B
   - Run paired r1 control/candidate first.
   - Use r3 only when the candidate is borderline or promising.
8. Decision
   - Keep only reproducible wins or documented opt-in diagnostics.
   - Revert negative prototypes before closing the experiment.

## RDNA4 Rules

- Larger tiles are not automatically better. E143/E146 show that lower workgroup
  count can lose to VGPR/LDS/scratch and occupancy.
- Smaller BK is not automatically better. E144 shows lower LDS can lose to extra
  K-loop and barrier cadence.
- F16 route pivots are not automatically better. E259/E260/E264 show that KV,
  broad f16-disable, and per-layer activation casts can all lose wall time.
- A prefill-only layout cannot break decode. E258 improved prompt slightly but
  hurt decode enough to lose wall time.
- Queue/mmap lessons are lane-specific. E260 rejected transferring 64k Vulkan
  graphics-queue/no-mmap behavior to the 12k dense lane; the 130k lane must
  re-test those knobs because RAM-spill/residency pressure is now central.
- Point wins do not guarantee wall wins. If wall regresses, classify as
  bottleneck shift and stop iterating that local mechanism.

## Candidate Topology Families

These are the current high-level families that still deserve design work.

### T1: Dual-Layout Q3_K Storage

Keep decode-safe raw Q3_K storage while adding a separate prefill-optimized
layout for matrix routes only. This is the corrected version of the E258 lesson:
transpose/layout work may help prompt, but decode must keep its fast route.

First design questions:

- Can VRAM afford a second Q3_K layout only for high-share FFN tensors?
- Can Vulkan select prefill-layout matmul while matvec/decode uses raw layout?
- Can upload/copy/view semantics be made fail-closed without touching all tensor
  movement paths at once?

### T2: Shader-Native Q3_K Prompt Layout

Design a backend-private Q3_K representation that reduces hmask/scale/helper
work in the coopmat prompt shader without expanding to full fp16 or int8. This
may be a signed-nibble or scale-group-expanded layout, but it needs an
instruction/resource proof before code.

First design questions:

- Which operations disappear from the active `matmul_q3_k_f32_f16acc_aligned_l`
  shader?
- How many bytes are added per block and per full model?
- Does the layout preserve matvec/decode or require a dual-layout contract?

### T3: Fused Dense FFN Block

Replace separate `gate`, `up`, `swiglu`, and `down` materialization with a tiled
FFN route that streams hidden tiles through the activation and accumulates the
down projection. This is complex, but it is one of the few ideas that can attack
global intermediate traffic and multiple Q3_K matmuls together.

First design questions:

- What is the memory traffic saved for `n=1024/2048` prompt chunks?
- Can gate/up share the same input tile without doubling accumulator pressure?
- Can down accumulation be tiled without a huge partial-output/reduce overhead?
- Is a Vulkan prototype easier than a ROCm prototype on this machine?

### T4: FA Shader-Body Work

FlashAttention is a secondary share on the archived 12k lane but a major share
on long-context lanes. For 130k, re-measure Q3_K vs FA shares before choosing
between Q3_K topology and FA shader-body work.

First design questions:

- Which KV ranges dominate current FA time?
- Can a shader-body change improve active KHR coopmat FA without falling back to
  scalar or split/reduce routes?

### T5: Differential Harness

Build better tools before more code if the design cannot be ranked. The harness
should compare Vulkan vs ROCm route buckets, resources, and shape timings for
the same lane, then emit a short target table.

First Vulkan piece added: `scripts/research/vulkan_evidence_pack.py`. It builds
a label-scoped markdown pack from benchmark diagnostics/server logs, optional
Vulkan perf rows, route trace lines, pipeline stats, and SPIR-V opcode summaries.
Use the VS Code tasks `bench: vulkan q3 130k route trace`, `bench: vulkan q3
130k q3 stats`, and `research: vulkan evidence pack` to capture a fresh
P002-lane pack before a large shader or route rewrite. The pack also includes a
diagnostic-only Amdahl ceiling sketch when `--baseline-tps` is provided; use it
to rank candidates, not as a speed claim.

First design questions:

- Which exact shapes differ most between Vulkan and ROCm after matching lane
  contracts?
- Which differences are kernel-body, layout, graph scheduling, or memory
  residency effects?

## Immediate Next Work

1. Finish the 130k workspace prep and keep old 12k/64k lanes explicitly historical.
2. Run the Vulkan and ROCm 130k baseline tasks sequentially, preserving diagnostics/server logs.
3. Create or update the first 130k program note under `docs/research/major-topology/`.
4. Collect a fresh route/residency evidence pack for the selected topology before editing code.
5. Only then open a prototype branch/worktree.

## Stop Conditions For The New Mode

Stop and write a design rejection when:

- the ceiling model cannot reach at least `+10%` wall on the active lane;
- the design requires broad tensor storage changes but has no fail-closed plan;
- the resource model predicts near-limit LDS/VGPR without a compensating
  algorithmic reduction;
- the proposal repeats an E257-E264 rejected route under a new name without explaining why 130k RAM-spill/residency changes the mechanism;
- a prototype needs more than one subsystem changed before it can prove a local
  mechanism.
