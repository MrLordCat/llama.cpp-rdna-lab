# DFlash Integration Preparation (Port Plan for llama.cpp-with-GUI)

Date: 2026-05-22
Scope: preparation and documentation only (no runtime DFlash code merged yet)

## Objective

Prepare a safe, staged plan to integrate BeeLlama-style DFlash into this fork
without breaking the active ROCm/RDNA4 performance lane.

This document answers:

- where to vendor/port from;
- what to implement first;
- how to split risk by phase;
- what to benchmark/validate before promotion.

## Current Local State (Gap)

Local speculative modes include `draft`, `mtp`, `ngram-*`, but no DFlash mode.

Confirmed local anchors:

- `common/speculative.cpp` enum and dispatch contain no DFlash state;
- `common/arg.cpp` `--spec-type` does not include `dflash`;
- no local `dflash_*` runtime/backend plumbing paths are present.

## Source of Truth for Porting

Primary external reference repo:

- `Anbeeld/beellama.cpp` (branch: `main`)

High-value discovery pointers in Bee:

- `src/models/dflash_draft.cpp`
- `common/speculative.cpp` and `common/speculative.h`
- `tools/server/server-context.cpp`
- `tools/server/server-adaptive-dm.h`
- `src/llama-context.h` / `src/llama-context.cpp`
- `src/llama-cparams.h`
- `src/llama-graph.h` / `src/llama-graph.cpp`
- `src/llama-kv-cache*.{h,cpp}`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-cuda/cross-ring-interleave.cu`
- `ggml/src/ggml-cuda/argmax.cu`
- `tests/test-dflash-plumbing.cpp`
- `tests/test-dflash-ring.cpp`

## Vendoring / Porting Policy

Do not import Bee repo wholesale. Use surgical porting with explicit provenance.

1. Port behavior into existing local files when possible.
2. Add new files only for clearly isolated DFlash units.
3. Keep each imported chunk traceable to upstream origin path and commit.
4. Avoid importing upstream docs/workflows/agent instruction files into protected local areas.
5. Keep optional debug/profile toggles env-gated and default-off.

## Proposed Where-To-Vendor Map

Detailed source-to-target mapping is tracked in:

- `docs/research/dflash/VENDOR_MANIFEST.md`

Detailed execution playbooks are tracked in:

- `docs/research/dflash/PHASE_PLAYBOOK.md`
- `docs/research/dflash/BRANCH_AND_COMMIT_PLAN.md`
- `docs/research/dflash/COMPATIBILITY_MATRIX.md`
- `docs/research/dflash/IMPLEMENTATION_RUNBOOK.md`
- `docs/research/dflash/FUTURE_WORKFLOW.md`
- `docs/research/dflash/KICKOFF_PACKET.md`

### A. Core speculative runtime (mandatory)

- Local targets:
  - `common/speculative.h`
  - `common/speculative.cpp`
  - `common/common.h`
  - `common/arg.cpp`

- Port content:
  - `spec-type dflash` mode;
  - DFlash state object and contract checks;
  - cross-context window parameters;
  - draft/accept cycle hooks;
  - optional profiling flags.

### B. Model/runtime graph glue (mandatory)

- Local targets:
  - `src/models/` (new file likely needed: `dflash_draft.cpp`)
  - `src/llama-context.h`
  - `src/llama-context.cpp`
  - `src/llama-cparams.h`
  - `src/llama-graph.h`
  - `src/llama-graph.cpp`
  - `src/llama-arch.cpp`
  - `src/llama-model.cpp`
  - `include/llama.h`

- Port content:
  - drafter graph input/output contract;
  - target hidden capture bookkeeping;
  - prefill/verify span plumbing in cparams;
  - API toggles for reduced verifier and capture control;
  - architecture/tensor-name mapping required by DFlash draft GGUF metadata.

### C. Server orchestration (mandatory)

- Local targets:
  - `tools/server/server-context.cpp`
  - optional new helper if needed: `tools/server/server-adaptive-dm.h`

- Port content:
  - DFlash verify/accept loop;
  - rollback and prefill suffix flush logic;
  - adaptive draft-depth controller hooks (phase-gated);
  - profile counters and guarded diagnostics.

### D. CUDA/HIP backend plumbing (phase-2 mandatory)

- Local targets:
  - `ggml/src/ggml-cuda/ggml-cuda.cu`
  - `ggml/src/ggml-cuda/cross-ring-interleave.cu`
  - `ggml/src/ggml-cuda/argmax.cu`
  - any required CMake target wiring in `ggml/src/ggml-cuda/`

- Port content:
  - cross-ring GPU alloc/write/interleave APIs;
  - backend proc-address exported hooks for DFlash runtime;
  - optional reduced-logit top-k verifier path;
  - stream sync helpers required by DFlash rollout.

Note for this fork: these CUDA-family files are also the HIP path on ROCm.
All changes here must be validated on local RDNA4 lane.

### E. Optional utility integration (phase-3+)

- `common/download.h` / `common/download.cpp`: optional DFlash draft auto-discovery.
- Converter metadata support only after core runtime path is stable.

## Phased Implementation Plan

## Phase 0: Contract + skeleton (no speed claim)

- Add `spec-type dflash` parsing and enum plumbing.
- Add DFlash state skeleton that fails closed if required pieces are missing.
- Add docs for launch args and expected model constraints.

Exit criteria:

- build passes;
- `llama-server --spec-type dflash` is recognized;
- clean error path if no DFlash drafter provided.

## Phase 1: Functional CPU-safe path

- Implement DFlash draft cycle with CPU-safe hidden/cross handling first.
- Keep GPU ring/cuda hook path disabled or fallback-only.

Exit criteria:

- correctness on short smoke runs;
- no output corruption vs baseline at same seed/temp;
- no regressions for non-DFlash modes.

## Phase 2: GPU ring and verifier optimization

- Add backend hook path for cross-ring and reduced verifier logits.
- Enable ROCm/HIP path through `ggml-cuda` sources.

Exit criteria:

- stable DFlash on local ROCm build;
- no crashes with graph reuse on/off;
- controlled A/B gain on repeated/decode-heavy lanes.

## Phase 3: Adaptive depth + server hardening

- Add adaptive draft-depth policy (`profit`-style first, fringe optional).
- Add profile categories and acceptance histograms.

Exit criteria:

- adaptive mode does not regress cold-first default profile;
- repeated-session lane improves with reproducible logs;
- diagnostics remain default-off.

## Phase 4: GUI surface

- Add GUI toggles and safe presets for DFlash mode and key args.
- Keep explicit warning labels for model compatibility and lane intent.

Exit criteria:

- GUI launches correct server args;
- preset UX does not alter current non-DFlash defaults.

## Validation Matrix for This Fork

Mandatory checks per phase:

1. build and smoke:
   - server starts with `spec-type none` and with `spec-type dflash`.
2. correctness:
   - no malformed output/corruption in deterministic runs.
3. compatibility:
   - existing ngram/mtp paths unchanged when DFlash is off.
4. performance lanes split:
   - cold-first prompt-heavy lane (no reuse);
   - repeated/steady decode-heavy lane (where DFlash is expected to win).
5. documentation:
   - add experiment note and `RESULTS_LOG` row for each keep/reject decision.

## Risk Register

1. Scope explosion: DFlash touches runtime, server, model graph, backend hooks.
2. Backend mismatch: Bee results are mostly CUDA; local target is ROCm/RDNA4.
3. Graph reuse hazards: capture offsets and reduced-logit toggles can desync reuse keys.
4. False speed claims: DFlash can boost repeated decode while neutral/negative on cold prompt-heavy.
5. Maintenance load: large diff surface against upstream llama.cpp moving target.

## Immediate Next Actions (Preparation Track)

1. Freeze upstream source anchors in a vendor manifest with commit hash and file map.
2. Add `H40` hypothesis track for DFlash integration readiness.
3. Create Phase-0 implementation checklist branch plan (no backend hooks yet).
4. Start with minimal CLI+state skeleton once the active ROCm tuning agent finishes.
