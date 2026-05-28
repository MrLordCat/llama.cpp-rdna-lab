# D048 - P003 Q4 MetaComp kickoff (Q4 size reduction without Q3)

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: kickoff + baseline capture

## Goal

Start a dedicated research track to reduce Q4 model VRAM usage **without**
changing nominal quant class to Q3.

Target problem statement:

- keep Q4 payload semantics (4-bit weight codes),
- reduce effective runtime VRAM footprint,
- avoid major throughput collapse and quality regression.

## Baseline Snapshot (captured now)

Model under test: `models/Qwen3.6-27B-Q4_K_S.gguf` on RX 9070 XT 16 GB.

### Practical 130k lane fit/TPS

1. Full offload fit check (same practical lane contract as big-prompt work):
   - label: `q4fit-vulkan130k-big-c152k-b512-ub256-r1`
   - config: `ctx=131072,b=512,ub=256,ngl=999,q4_0/q4_0,--spec-type none --no-mmap`
   - result: server exits before ready (no fit). Log indicates projected device use
     above free VRAM and inability to meet free-memory target.

2. Fit-auto TPS baseline:
   - label: `q4fitauto-vulkan130k-big-c152k-b512-ub256-r1`
   - config: same lane, `ngl=-1`
   - result:
     - aggregate TPS: `0.1178`
     - prompt tok/s: `427.86`
     - decode tok/s: `4.15`
     - prompt ms: `131877.2`
     - decode ms: `3857.3`

### Quality baseline (BFCL-lite smoke subset)

- label: `q4metacomp-bfcl-default8-r1`
- harness: `scripts/research/bfcl_lite_pilot.py`
- endpoint: local OpenAI-compatible server at `127.0.0.1:8088`
- result: `8/8` (`100%`) on the default 8-case subset
  (`simple_python,multiple,parallel,irrelevance`, 2 cases each).

## Research Thesis

There is room below current Q4 runtime footprint because practical Q4 memory
is not only 4-bit codes; metadata/layout/allocator overhead also contributes.

Key direction: **Q4-MetaComp**

- Keep 4-bit weight codes.
- Compress metadata hierarchy (scales/mins/aux headers) using superblock-local
  structure and compact deltas.
- Preserve a deterministic GPU-friendly decode contract for fused kernels.
- Add fallback path for difficult blocks to preserve quality.

## Candidate Algorithm Family (Q4-MetaComp)

### A. Superblock shared-scale with delta-coded block scales

- Build superblocks (e.g., 4-8 existing Q4 blocks).
- Store one base scale (or base pair) per superblock.
- Store per-block scale deltas in compact signed representation.
- Optional escape marker for outlier blocks to store full local scale.

Expected effect: reduce metadata bytes/weight while preserving 4-bit codebook.

### B. Layer-adaptive metadata mode

- Allow per-tensor mode selection:
  - mode 0: legacy Q4 layout (safe fallback),
  - mode 1: MetaComp compact layout,
  - mode 2: compact + sparse escapes.
- Selection criterion from offline sensitivity scan.

Expected effect: avoid forcing fragile tensors into aggressive compact mode.

### C. Runtime decode integration

- Add fused decode+matmul kernel path for MetaComp tensors.
- Keep memory access fixed-stride and coalesced where possible.
- Never decompress entire tensor to fp16/fp32 in global memory.

Expected effect: contain throughput penalty.

## Measurement Gates

A candidate is accepted only if all gates pass on the same lane/contract.

1. Fit gate:
   - must run on target lane where baseline full offload currently fails, or
     must increase free VRAM headroom by a meaningful margin.

2. TPS gate (big-prompt practical lane):
   - compare against current Q4 fit-auto baseline and Q3 practical reference;
   - no claim without matching tasks/task_ids/no-reuse/context settings.

3. Quality gate:
   - BFCL-lite default subset must not regress materially from baseline 8/8;
   - if expanded subset is used, report category-level deltas.

4. Safety gate:
   - deterministic decode equivalence tests for representative blocks;
   - numerical sanity checks for outlier/escape paths.

## Implementation Plan

### Phase 0: format spec + estimator (no kernel changes)

- Specify Q4-MetaComp binary layout and version tag.
- Build offline memory estimator tool for projected bytes/weight and VRAM.
- Output theoretical savings per tensor and whole model.

Deliverable: layout spec + estimator report.

### Phase 1: offline converter prototype

- Implement converter from existing Q4 tensor layout to MetaComp layout.
- Add reversible debug path and checksum validation.
- Emit per-tensor conversion diagnostics (delta range, escape ratio).

Deliverable: conversion tool + roundtrip checker.

### Phase 2: correctness-first runtime path (slow path allowed)

- Add runtime loader support and CPU/GPU reference decode verification.
- Run strict tensor/block equivalence tests.
- Keep feature behind env gate (default off).

Deliverable: correctness green on selected tensors.

### Phase 3: fused Vulkan path

- Implement/port fused decode+matmul for MetaComp blocks in Vulkan route.
- Compare point kernels vs legacy Q4 on hot shapes.
- Iterate on register/LDS pressure and occupancy.

Deliverable: point benchmark parity target on hot path.

### Phase 4: lane A/B and promotion decision

- Run practical 130k lane A/B with same workload contract.
- Run BFCL-lite quality A/B.
- Decide keep/reject based on fit + TPS + quality gates.

Deliverable: promotion or rejection note in results log.

## Immediate Next Steps

1. Implement Phase 0 estimator script and run it on `Qwen3.6-27B-Q4_K_S`.
2. Define exact success threshold for VRAM reduction and allowed TPS delta.
3. Create env-gated placeholder flags for future runtime integration.

## Artifacts

- `build_logs/agent-workload/q4fit-vulkan130k-big-c152k-b512-ub256-r1.server.log`
- `build_logs/agent-workload/q4fitauto-vulkan130k-big-c152k-b512-ub256-r1.diagnostics.md`
- `build_logs/agent-workload/q4metacomp-bfcl-default8-r1.bfcl_lite.summary.md`
